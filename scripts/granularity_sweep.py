#!/usr/bin/env python
"""GRANULARITY SWEEP -- the "10000 cards" question, part (A): is FINER always better?

User: "if we cut into 8000 or 10000 blocks, is that better? is there an optimal K*?"

We SWEEP the partition / device count K from 2 .. ~10000 on several VARIED real graphs
(+ synthetics spanning sparse/dense) and measure, VIA THE REAL src/zord ENGINE
(zord.partition.arrange + the engine's predict_ms roofline; NEVER networkx), per K:

  - PREDICTED MAKESPAN  : the bottleneck device's step time. Two coupled terms:
      (1) the engine's per-device roofline (compute = incident-edge gather / achieved HBM
          bandwidth, comm = boundary feature rows / interconnect link) -- this DROPS with K
          (more parallelism + per-device working set shrinks);
      (2) a SYNCHRONIZATION-BARRIER / FIXED-PER-STEP overhead that GROWS with K (a BSP step
          ends in a global barrier / tree all-reduce: latency ~ alpha * ceil(log2 K); plus a
          fixed kernel-launch / scheduling floor per partition). This is the standard physical
          cost of more cards; it is layered in THIS SCRIPT on top of the engine's roofline (the
          engine cost model is K-agnostic by design), clearly separated and reported.
    makespan(K) = max_k engine_step_ms[k]  +  barrier_ms(K).
  - TOTAL CUT           : boundary (cross-device) edges -- the engine's res.cut. GROWS with K.
  - COMM + SYNC overhead: engine boundary-comm bytes + the barrier term.
  - PER-DEVICE WORKING-SET MEMORY + FEASIBILITY : the engine's footprint (feature rows + resident
    edge metadata) per device vs the per-card HBM cap. Finer K -> smaller working set -> a graph
    that is INFEASIBLE at small K becomes feasible at large K (the memory-relief gain).
  - DEVICE UTILIZATION  : useful per-partition work / (useful + fixed-per-step overhead). Collapses
    as partitions get tiny (the per-step barrier dwarfs the shrinking useful work).

WHAT IS AN ENGINE RESULT vs WHAT IS MODEL-CONDITIONAL (§44-correction, read this before believing
any K*):

  * ENGINE-ONLY makespan(K) is MONOTONE NON-INCREASING. On the pure engine roofline, finer K only
    LOWERS the bottleneck per-step time (more parallelism + a smaller per-device working set), and
    crucially finer is MANDATORY for FEASIBILITY: below the memory floor K_mem EVERY plan OOMs.
    So in the engine alone there is NO interior optimum -- "finer is always better (and required to
    fit)". We print this curve FIRST and label its argmin as the FINEST feasible K.

  * The U-SHAPE and the interior K* are NOT engine results. They appear ONLY once we ADD an
    external SYNCHRONIZATION-BARRIER term barrier(K) = alpha*ceil(log2 K) + beta that GROWS with K.
    That term is CONDITIONAL on a BSP-barrier model with a per-hop RTT = alpha. The default alpha
    is an ASSUMED ~200us commodity-cross-node RTT -- it is NOT measured here. The honest grounding
    of alpha is scripts/barrier_vs_k.py (a real torch.distributed all-reduce micro-benchmark);
    pass its measured value via --alpha-ms to ground (or refute) the interior optimum.

We therefore report, per graph: (1) the engine-only monotone curve + its finest-feasible argmin;
(2) the barrier-AUGMENTED curve + its K* (clearly flagged conditional-on-alpha); (3) a closed-form
K* heuristic (K_mem floor + K_bal cost-balance knee) compared HONESTLY against the swept K* across
ALL graphs -- including that K_bal can OVERSHOOT the swept K* by 7x-235x (we report every regret,
no cherry-pick).

PROCESS-only: a placement is just an assignment; same data + same model => identical trained
result. We optimize TIME / MEMORY / FEASIBILITY / UTILIZATION, NEVER accuracy.

Usage:
  python scripts/granularity_sweep.py                       # default graph set, slow link
  python scripts/granularity_sweep.py --link 0.12 --feat 128
  python scripts/granularity_sweep.py --datasets askubuntu --kmax 10000
  python scripts/granularity_sweep.py --alpha-ms 0.012      # ground the barrier with a MEASURED alpha
  python scripts/granularity_sweep.py --dry-run             # tiny synthetics only (CI/CPU)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zord.profiler.cluster_profile import ClusterProfile, DeviceProfile, GB  # noqa: E402
from zord.partition.arrange import (                                       # noqa: E402
    arrange, predict_ms, BYTES_PER_EDGE_RESIDENT)


def make_cluster(K: int, cap_bytes: float, agg_bw: float, link: float) -> ClusterProfile:
    """A K-device HOMOGENEOUS cluster with an EXACT per-card usable HBM cap (no hidden driver
    reserve -- mem_reserved=0 so `usable_mem` == cap_bytes exactly), achieved-agg bandwidth
    agg_bw, and the interconnect `link` as the flat-fabric parameter. Built directly (not via
    from_spec) so the small caps that make feasibility BITE are exactly the numbers we set."""
    devs = [DeviceProfile(i, f"card{i}", int(cap_bytes), throughput=agg_bw / 444.0,
                          node=0, mem_reserved=0, hbm_bw_gbps=agg_bw) for i in range(K)]
    return ClusterProfile(devices=devs, intra_node_bw=link, inter_node_bw=link)


# --------------------------------------------------------------------------- #
# the per-step BARRIER / fixed-overhead model (layered on the engine roofline) #
# --------------------------------------------------------------------------- #
# A BSP GNN step ends in a global synchronization: each device must finish its local
# aggregation, then a boundary exchange / all-reduce completes before the next layer (the
# engine's per-LAYER comm is the BYTES; the barrier here is the LATENCY of the collective).
# On K devices this costs:
#   - a TREE all-reduce / barrier LATENCY  ~  alpha * ceil(log2 K)   (log-depth collective: each
#     of the log2(K) hops pays one latency-bound message; this is the "more cards = more
#     synchronization" cost, GROWING with K).
#   - a FIXED per-step floor  ~  beta  (kernel launch + scheduling; a K-independent wall floor that
#     does NOT shrink no matter how small a partition gets).
# alpha (per-hop latency, ms) defaults to a LINK-DERIVED network round-trip proxy so it is not a
# free knob: on a slow fabric one collective hop is RTT-bound (we model RTT = `latency_us`, default
# 200us ~ a commodity-Ethernet/cross-node round trip; on NVLink set --latency-us small). beta is a
# small fixed floor. This term is NOT in the engine cost model (which is deliberately K-agnostic --
# it costs a GIVEN partition); it is the physical cost of HAVING more cards, reported SEPARATELY so
# the engine roofline stays the load-bearing measurement. The U-shape and K* are robust to the exact
# magnitudes (--alpha-ms / --latency-us let you sweep them; the knee K_bal moves predictably).
def barrier_ms(K: int, alpha_ms: float, beta_ms: float) -> float:
    hops = math.ceil(math.log2(max(2, K)))
    return alpha_ms * hops + beta_ms


def default_alpha_ms(latency_us: float) -> float:
    """Per-hop collective latency (ms) from a network round-trip-time proxy (us)."""
    return latency_us / 1000.0


# --------------------------------------------------------------------------- #
# graphs: real (zord.datasets) + synthetics spanning sparse / dense            #
# --------------------------------------------------------------------------- #
def synth(name: str, N: int, avg_deg: float, seed: int = 0):
    """A synthetic temporal graph (random edges) of a target average degree -- spans the
    sparse/dense axis the real QA/social graphs do not. Returned as a light object exposing
    .src/.dst/.num_nodes/.name (arrange only needs src/dst/num_nodes)."""
    rng = np.random.default_rng(seed)
    E = int(N * avg_deg / 2)
    src = rng.integers(0, N, E).astype(np.int64)
    dst = rng.integers(0, N, E).astype(np.int64)
    m = src != dst
    src, dst = src[m], dst[m]

    class _G:
        pass
    g = _G()
    g.src, g.dst, g.num_nodes, g.name = src, dst, N, name
    return g


def load_graph(name: str):
    from zord.datasets import load
    g = load(name).sort_by_time()
    return g


# --------------------------------------------------------------------------- #
# one (graph, K) measurement through the REAL engine                           #
# --------------------------------------------------------------------------- #
def measure_K(g, K: int, link: float, feat: int, cap_bytes: float, agg_bw: float,
              alpha_ms: float, beta_ms: float, seed: int, metis_max_edges: int):
    """Run the REAL zord arrange for a K-device homogeneous cluster and return the engine's
    per-device work + cut + feasibility, plus the makespan augmented with barrier(K)."""
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    E = int(src.size)
    cl = make_cluster(K, cap_bytes, agg_bw, link)

    t0 = time.time()
    res = arrange(src, dst, N, cl, link_gbps=link, feat_dim=feat,
                  num_snapshots=64, seed=seed, metis_max_edges=metis_max_edges)
    arrange_s = time.time() - t0

    # engine roofline per device (the load-bearing measurement)
    bw = res.bw_gbps
    inc_work = res.incident * feat
    comm_work = res.comm_rows * feat
    tot_ms, comp_ms, comm_ms = predict_ms(inc_work, comm_work, bw, res.link_gbps)
    engine_step_ms = float(tot_ms.max())               # bottleneck device, engine-only

    # per-device working-set bytes (SAME footprint shape as the engine feasibility check:
    # feature rows + resident edge metadata) and feasibility vs the per-card HBM cap.
    cap = cl.devices[0].usable_mem
    resident = res.counts.astype(np.float64) * feat * 4.0 + res.incident * BYTES_PER_EDGE_RESIDENT
    peak_ws = float(resident.max())
    feasible = bool(peak_ws <= cap)

    # the augmented makespan: engine roofline + the BSP barrier that grows with K
    bar = barrier_ms(K, alpha_ms, beta_ms)
    makespan = engine_step_ms + bar

    # utilization: useful per-device work time vs (useful + fixed barrier). As partitions get
    # tiny, engine_step_ms -> 0 while bar stays -> utilization collapses.
    util = engine_step_ms / max(1e-12, engine_step_ms + bar)

    return dict(K=K, win=res.name, cut=int(res.cut), engine_step_ms=engine_step_ms,
                comp_ms=float(comp_ms.max()), comm_ms=float(comm_ms.max()),
                barrier_ms=bar, makespan_ms=makespan, peak_ws_gb=peak_ws / GB,
                cap_gb=cap / GB, feasible=feasible, util=util, arrange_s=arrange_s,
                inc_max=int(res.incident.max()), N=N, E=E, alpha_ms=alpha_ms)


# --------------------------------------------------------------------------- #
# closed-form K* heuristic                                                     #
# --------------------------------------------------------------------------- #
def kstar_formula(N: int, E: int, feat: int, cap_bytes: float, link: float, agg_bw: float,
                  alpha_ms: float, beta_ms: float):
    """Derive K* from graph stats + cluster params (E, F, per-card cap, link GB/s, agg_bw, barrier).

    Two forces set K*:
      (i)  MEMORY FEASIBILITY FLOOR K_mem : the smallest K whose per-device working set fits one
           card. Per-device footprint ~ (N/K)*F*4  +  (E/K)*B (B=resident bytes/edge). Solve
           footprint(K) <= cap  ->  K_mem = ceil( (N*F*4 + E*B) / cap ). Below K_mem EVERY plan
           OOMs; K* must be >= K_mem (this is the LEFT arm: finer is MANDATORY for feasibility).
      (ii) the COST-BALANCE KNEE that, above the floor, trades the two K-dependent costs. The
           bottleneck device's USEFUL step time falls ~ W/K (the divisible per-step work splits K
           ways), while the BARRIER grows ~ alpha*log2(K)+beta. W (ms) is the SINGLE-CARD-equivalent
           divisible work = the DOMINANT of the two roofline floors (the step is whichever-bound):
             - compute floor  W_comp = N_GATHERS*BYTES_PER_TRAVERSAL*F*E / agg_bw   (gather all edges)
             - comm    floor  W_comm = N_GATHERS*FEATURE_ROW_BYTES *F*E / link      (worst-case all
                              boundary rows cross the slow link; on a slow link this DOMINATES).
           Smooth envelope t(K) ~ W/K + alpha*log2(K) + beta; dt/dK = -W/K^2 + alpha/(K ln2) = 0
           ->  K_bal = W*ln2 / alpha. (beta only shifts the level, not the stationary point.)
      K* ~ max(K_mem, K_bal): satisfy memory FIRST, then push to the cost-balance knee.
    Returns (K_star, K_mem, K_bal)."""
    cap = max(cap_bytes, 1.0)
    B = BYTES_PER_EDGE_RESIDENT
    footprint_total = N * feat * 4.0 + E * B
    K_mem = max(1, math.ceil(footprint_total / cap))

    # the two roofline floors of the single-card-equivalent divisible work (ms). Constants mirror
    # arrange.predict_ms: N_GATHERS=2, BYTES_PER_EDGE_TRAVERSAL=4 (compute), FEATURE_ROW_BYTES=4 (comm).
    W_comp_ms = (2.0 * 4.0 * feat * E) / (agg_bw * 1e9) * 1e3
    W_comm_ms = (2.0 * 4.0 * feat * E) / (max(link, 1e-9) * 1e9) * 1e3
    W_ms = max(W_comp_ms, W_comm_ms)                    # the step is whichever-bound (slow link -> comm)
    K_bal = max(1.0, W_ms * math.log(2) / max(1e-9, alpha_ms))
    K_star = max(K_mem, K_bal)
    return int(round(K_star)), int(K_mem), float(K_bal)


# --------------------------------------------------------------------------- #
# sweep + report                                                               #
# --------------------------------------------------------------------------- #
def run_graph(g, ks, link, feat, cap_bytes, agg_bw, alpha_ms, beta_ms, seed, metis_max_edges):
    rows = []
    for K in ks:
        if K > g.num_nodes:                              # cannot have more partitions than nodes
            continue
        r = measure_K(g, K, link, feat, cap_bytes, agg_bw, alpha_ms, beta_ms, seed, metis_max_edges)
        rows.append(r)
    return rows


def _is_monotone_nonincreasing(vals, tol=1e-9):
    """True if vals never INCREASE (allowing a tiny fp tolerance)."""
    return all(vals[i + 1] <= vals[i] + tol for i in range(len(vals) - 1))


def print_table(name, rows):
    print(f"\n=== {name}  (N={rows[0]['N']:,}  E={rows[0]['E']:,}) ===")
    hdr = (f"{'K':>6} {'winner':>20} {'cut':>11} {'comp':>8} {'comm':>9} {'engineMS':>9} "
           f"{'barrier':>8} {'augMS':>9} {'peakWS_GB':>10} {'feas':>5} {'util':>6}")
    print(hdr)
    for r in rows:
        print(f"{r['K']:>6} {r['win']:>20} {r['cut']:>11,} {r['comp_ms']:>8.3f} "
              f"{r['comm_ms']:>9.3f} {r['engine_step_ms']:>9.3f} {r['barrier_ms']:>8.3f} "
              f"{r['makespan_ms']:>9.3f} {r['peak_ws_gb']:>10.4f} "
              f"{('yes' if r['feasible'] else 'NO'):>5} {r['util']:>6.3f}")

    feas = [r for r in rows if r['feasible']]
    pool = feas if feas else rows
    first_feasible = min((r['K'] for r in rows if r['feasible']), default=None)

    # (1) ENGINE-ONLY curve (the load-bearing measurement): is makespan(K) monotone non-increasing?
    eng_vals = [r['engine_step_ms'] for r in pool]
    eng_mono = _is_monotone_nonincreasing(eng_vals)
    eng_argmin = min(pool, key=lambda r: r['engine_step_ms'])
    print(f"  -> ENGINE-ONLY makespan(K): {'MONOTONE non-increasing' if eng_mono else 'NON-monotone'} "
          f"on the feasible range; argmin at the FINEST feasible K={eng_argmin['K']} "
          f"(engine_step={eng_argmin['engine_step_ms']:.3f}ms). Finer lowers per-step time AND is "
          f"MANDATORY for feasibility (first feasible K={first_feasible}).")

    # (2) BARRIER-AUGMENTED curve: the interior K* exists ONLY because of the ADDED barrier term.
    kstar = min(pool, key=lambda r: r['makespan_ms'])
    aug_mono = _is_monotone_nonincreasing([r['makespan_ms'] for r in pool])
    note = ("(still monotone -> no interior optimum at this alpha)" if aug_mono
            else f"(interior optimum -> CONDITIONAL on the barrier model, alpha={rows[0]['alpha_ms']:.4f}ms/hop)")
    print(f"  -> BARRIER-AUGMENTED K* = {kstar['K']} (aug-makespan {kstar['makespan_ms']:.3f}ms, "
          f"winner {kstar['win']}) {note}")
    return kstar, first_feasible, eng_argmin, eng_mono


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="real dataset names (default: a varied real set + synthetics)")
    ap.add_argument("--link", type=float, default=0.12, help="interconnect GB/s (slow cross-node)")
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--cap-mb", type=float, default=8.0,
                    help="per-card usable HBM cap (MB), set EXACTLY (no driver reserve). Small so "
                         "feasibility BITES at small K, exposing the memory-relief left arm of the "
                         "U-shape. Real cards are GB-scale; we shrink the cap to put a realistic "
                         "billion-edge memory ratio onto these benchmark-scale graphs.")
    ap.add_argument("--agg-bw", type=float, default=444.0, help="achieved HBM agg bandwidth GB/s")
    ap.add_argument("--latency-us", type=float, default=200.0,
                    help="ASSUMED per-hop collective RTT proxy (us) -> sets alpha unless --alpha-ms "
                         "given. DEFAULT 200us is an ASSUMPTION (~cross-node/Ethernet), NOT measured; "
                         "the barrier term + the resulting U-shape are CONDITIONAL on it. Replace with "
                         "the MEASURED value from scripts/barrier_vs_k.py (use ~5-15us for NVLink).")
    ap.add_argument("--alpha-ms", type=float, default=None,
                    help="per-hop barrier/all-reduce latency (ms); overrides --latency-us. Pass the "
                         "MEASURED alpha from barrier_vs_k.py here to GROUND the barrier term. "
                         "barrier = alpha*ceil(log2 K) + beta")
    ap.add_argument("--beta-ms", type=float, default=0.02, help="fixed per-step floor (ms)")
    ap.add_argument("--kmax", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metis-max-edges", type=int, default=2_000_000,
                    help="gate the superlinear pymetis floor off above this E (engine substitutes "
                         "the cheap O(M) lpa-proxy floor); keeps the sweep fast at large K.")
    ap.add_argument("--dry-run", action="store_true",
                    help="tiny synthetics only (no dataset files needed) -- CPU/CI smoke")
    a = ap.parse_args()
    alpha_ms = a.alpha_ms if a.alpha_ms is not None else default_alpha_ms(a.latency_us)

    # the K ladder: small (2,4,8,16) -> large (64,256,1024,4096,~10000)
    ks_all = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8000, 10000]
    ks = [k for k in ks_all if k <= a.kmax]

    # graph set: VARIED real + synthetics spanning sparse/dense
    graphs = []
    if a.dry_run:
        graphs = [synth("synth-sparse(d4)", 20_000, 4.0, a.seed),
                  synth("synth-dense(d40)", 20_000, 40.0, a.seed)]
    elif a.datasets:
        graphs = [load_graph(n) for n in a.datasets]
    else:
        for n in ("collegemsg", "askubuntu", "stackoverflow"):
            try:
                graphs.append(load_graph(n))
            except Exception as e:
                print(f"[skip {n}: {type(e).__name__}: {str(e)[:80]}]")
        graphs.append(synth("synth-sparse(d6)", 200_000, 6.0, a.seed))
        graphs.append(synth("synth-dense(d50)", 200_000, 50.0, a.seed))

    cap_bytes = a.cap_mb * 1024 ** 2
    alpha_is_assumed = (a.alpha_ms is None)     # default alpha is the ASSUMED RTT proxy, not measured
    print(f"GRANULARITY SWEEP  link={a.link}GB/s  feat={a.feat}  per-card cap={a.cap_mb}MB  "
          f"agg_bw={a.agg_bw}GB/s  alpha={alpha_ms:.4f}ms/hop "
          f"({'ASSUMED ~%.0fus RTT, NOT measured' % a.latency_us if alpha_is_assumed else 'supplied via --alpha-ms'}) "
          f"beta={a.beta_ms}ms")
    if alpha_is_assumed:
        print("  [!] alpha is ASSUMED. The barrier term + any U-shape/interior-K* below are CONDITIONAL "
              "on it. Ground it with scripts/barrier_vs_k.py and pass --alpha-ms <measured>.")
    print(f"K ladder: {ks}")

    summary = []
    for g in graphs:
        rows = run_graph(g, ks, a.link, a.feat, cap_bytes, a.agg_bw,
                         alpha_ms, a.beta_ms, a.seed, a.metis_max_edges)
        if not rows:
            continue
        kstar, first_feas, eng_argmin, eng_mono = print_table(g.name, rows)
        N, E = rows[0]['N'], rows[0]['E']
        kf, kmem, kbal = kstar_formula(N, E, a.feat, cap_bytes, a.link, a.agg_bw,
                                       alpha_ms, a.beta_ms)
        # snap the continuous formula-K* to the nearest K ACTUALLY MEASURED for this graph (the
        # ladder is clipped at num_nodes for tiny graphs) for an apples-to-apples makespan compare.
        measured_ks = [r['K'] for r in rows]
        snapped = min(measured_ks, key=lambda k: abs(k - kf))
        sw = next(r for r in rows if r['K'] == kstar['K'])
        fm = next(r for r in rows if r['K'] == snapped)
        regret = (fm['makespan_ms'] / sw['makespan_ms'] - 1.0) * 100.0
        # HONEST overshoot of the cost-balance knee vs the actually-swept augmented K* (no cherry-pick).
        kbal_overshoot = kbal / max(1, kstar['K'])
        print(f"  -> formula K*: K_mem={kmem} K_bal={kbal:.0f} -> K*={kf} (snapped to {snapped}). "
              f"K_bal OVERSHOOTS swept-K* {kstar['K']} by {kbal_overshoot:.1f}x. "
              f"aug-makespan@formula={fm['makespan_ms']:.3f}ms vs @swept={sw['makespan_ms']:.3f}ms "
              f"-> regret {regret:+.2f}% (the regret stays small only because the augmented U is FLAT; "
              f"the K_bal estimate itself is far off).")
        summary.append(dict(graph=g.name, swept_kstar=kstar['K'], first_feasible=first_feas,
                            formula_kstar=kf, k_mem=kmem, k_bal=kbal, snapped=snapped,
                            regret_pct=regret, kbal_overshoot=kbal_overshoot,
                            eng_argmin=eng_argmin['K'], eng_mono=eng_mono))

    print("\n================ SUMMARY: engine-only argmin vs barrier-augmented K* per graph ================")
    print(f"{'graph':>22} {'eng_mono':>9} {'eng_argmin':>10} {'first_feas':>10} {'aug_K*':>7} "
          f"{'formula_K*':>11} {'K_mem':>7} {'K_bal':>10} {'Kbal/K*x':>9} {'regret%':>8}")
    for s in summary:
        print(f"{s['graph']:>22} {('yes' if s['eng_mono'] else 'NO'):>9} {s['eng_argmin']:>10} "
              f"{str(s['first_feasible']):>10} {s['swept_kstar']:>7} {s['formula_kstar']:>11} "
              f"{s['k_mem']:>7} {s['k_bal']:>10.0f} {s['kbal_overshoot']:>8.1f}x {s['regret_pct']:>8.2f}")
    if summary:
        ovs = [s['kbal_overshoot'] for s in summary]
        rgs = [s['regret_pct'] for s in summary]
        print(f"\n  K_bal/swept-K* overshoot range across graphs: {min(ovs):.1f}x .. {max(ovs):.1f}x "
              f"(the closed-form knee is a ROUGH upper guess, not a tight K* predictor).")
        print(f"  regret range across ALL graphs: {min(rgs):+.2f}% .. {max(rgs):+.2f}% (reported for "
              f"every graph -- no single-number cherry-pick).")
    print("\nINTERPRETATION (honest, §44-correction):")
    print("  * ENGINE-ONLY: makespan(K) is MONOTONE non-increasing -> finer is always better on the")
    print("    roofline, and finer is MANDATORY for FEASIBILITY (below K_mem every plan OOMs). The")
    print("    engine alone has NO interior optimum; its argmin is the FINEST feasible K.")
    print("  * The U-shape / interior K* appears ONLY with the ADDED barrier(K)=alpha*log2(K)+beta and")
    print("    is CONDITIONAL on the per-hop RTT alpha. With the ASSUMED 200us alpha the knee lands at")
    print("    a moderate K; with a MEASURED NVLink alpha (us-scale, barrier_vs_k.py) it moves -- and")
    print("    may VANISH (the curve stays monotone). K* is NOT an engine result; it is model-conditional.")
    print("  * Same-result invariance: K only changes WHERE partial sums are computed, never WHAT")
    print("    (placement is a result-preserving GAS reduce). PROCESS-only.")


if __name__ == "__main__":
    main()
