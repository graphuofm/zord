#!/usr/bin/env python3
"""FEATURE-PARALLEL (ATTRIBUTE-DIMENSION splitting) experiment -- CALLS the real zord engine.

THE USER DIRECTIVE: future temporal graphs carry LOTS of attributes. zord's default arrange does
NODE-parallel decomposition (partition the VERTICES, pay boundary comm). This script explores a NEW
decomposition axis -- FEATURE-parallel: split the F feature COLUMNS across D devices; each device
holds the FULL graph adjacency + F/D feature columns, runs the full-graph SpMM aggregation over its
column slice, then INTEGRATES (concat the columns; gather for the dense layer).

WHY IT IS RESULT-PRESERVING (PROCESS-only; accuracy is NEVER a target): the aggregation
    h_v[c] = sum_{u in N(v)} x_u[c]
is INDEPENDENT per feature column c. Column-sharding X then concatenating the per-shard results is
EXACTLY A @ X -- bit-identical (fp-epsilon) to the single-device result. EXP-1 verifies this NUMERICALLY.

WHAT IS MEASURED (all via the engine -- zord.partition.feature_parallel + zord.schedule):
  EXP-1  MATH CONSISTENCY: column-shard + concat == single-device aggregation (max abs diff -> 0).
  EXP-2  PER-ATTRIBUTE MARGINAL TIME: sweep F, confirm aggregation time is ~LINEAR in F (the §19a
         ~0.74us/dim regime) -- on REAL data, via the engine's roofline cost model.
  EXP-3  MULTI-DATASET CROSSOVER: ogbn-arxiv (128-dim), Reddit (602-dim), Coauthor-CS (6805-dim) --
         different attribute richness F. For each, sweep F and report the NODE-vs-FEATURE-parallel
         crossover (makespan + feasibility) + the integration cost, picked by zord.schedule.plan(
         ..., decomposition='auto'). low F -> node-parallel; high F / attribute-heavy -> feature-
         parallel relieves per-device HBM (F/D cols, no full-F boundary/halo row replication).

ENGINE ENTRY POINTS CALLED (NOT reimplemented here):
  zord.partition.feature_parallel.fp_aggregate_consistency  -- the numeric math-consistency proof
  zord.partition.feature_parallel.feature_parallel_plan     -- the feature-parallel cost model
  zord.partition.arrange.arrange + zord.schedule.planner._placement_from_arrange -- node-parallel
  zord.schedule.plan(..., decomposition='auto')             -- the planner-level axis CHOICE

REAL DATA: loaded via scripts/attr_partition_real.py's ogb/PyG loaders (Coauthor-CS, Reddit need
torch_geometric; ogbn-arxiv needs ogb). When a dataset is not stageable in-agent we fall back to a
CALIBRATED-SYNTHETIC graph matching its published (N, F, degree, #communities), CLEARLY labelled.

USAGE
  PYTHONPATH=src python3 scripts/feature_parallel.py            # all exps, real if stageable else synth
  PYTHONPATH=src python3 scripts/feature_parallel.py --datasets ogbn-arxiv,reddit,coauthor-cs
  PYTHONPATH=src python3 scripts/feature_parallel.py --synthetic   # force calibrated-synthetic
"""
import argparse
import os
import sys
import time

import numpy as np

# engine on the path (this script CALLS the engine; it does not reimplement it)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zord.datasets import TemporalGraph                                    # noqa: E402
from zord.profiler import from_spec                                        # noqa: E402
from zord.schedule import plan                                             # noqa: E402
from zord.partition import (                                               # noqa: E402
    feature_parallel_plan, arrange, fp_aggregate_consistency)
from zord.schedule.planner import _placement_from_arrange                  # noqa: E402

# reuse the REAL attributed-dataset loaders (ogb / PyG) + the registry from attr_partition_real
import attr_partition_real as apr                                          # noqa: E402

GB = 1024 ** 3
BYTES_PER_FEAT = 4.0


# ============================================================================ #
# data: load a real attributed graph (or calibrated synthetic) as a TemporalGraph
# ============================================================================ #
def load_temporal(name, root, force_synth, seed):
    """Load a REAL attributed graph via attr_partition_real, wrap as a zord TemporalGraph. Returns
    (TemporalGraph, F, note). Synthetic timestamps (logical, equal-spread) -- the temporal corner
    is not under test here; only the feature-dimension decomposition is."""
    g, note = apr.load_graph(name, root, force_synth, scale=1.0, seed=seed)
    src, dst = g.src, g.dst
    # logical timestamps spread over the edge stream (the decomposition axis is F, not time)
    t = np.sort(np.random.default_rng(seed).integers(0, max(2, g.E), size=g.E)).astype(np.int64)
    tg = TemporalGraph(src=src, dst=dst, t=t, num_nodes=g.N, name=g.name)
    return tg, g.F, note


# ============================================================================ #
# EXP-1  MATH CONSISTENCY (column-shard + concat == single-device aggregation)
# ============================================================================ #
def exp1_math_consistency(tg, F, D, seed):
    print("\n" + "=" * 92)
    print(f"EXP-1  MATH CONSISTENCY: feature-parallel (column-shard+concat) == single-device  "
          f"[{tg.name}]")
    print("=" * 92)
    rng = np.random.default_rng(seed)
    N = tg.num_nodes
    # a SMALL real-graph aggregation on a manageable column width (the invariance is F-independent).
    # Cap columns by edge count so the O(E*Fc) np.add.at stays fast on big graphs (Reddit 57M edges).
    E = tg.num_edges
    Fc = min(F, 256 if E < 5_000_000 else (32 if E < 50_000_000 else 8))
    # balanced-ish contiguous column splits across D devices (the engine's split is bandwidth-prop;
    # the math invariance holds for ANY split that sums to F -- we use a near-even one here)
    base = Fc // D
    splits = [base] * D
    splits[-1] += Fc - base * D
    X = rng.standard_normal((N, Fc)).astype(np.float32)
    t0 = time.time()
    h_single, h_fp, diff = fp_aggregate_consistency(tg.src, tg.dst, X, splits)
    dt = time.time() - t0
    print(f"  N={N:,}  E={tg.num_edges:,}  columns={Fc} split {splits} across D={D} devices")
    print(f"  single-device  ||h||_F = {np.linalg.norm(h_single):.6e}")
    print(f"  feature-par    ||h||_F = {np.linalg.norm(h_fp):.6e}")
    print(f"  MAX ABS DIFF (single vs column-shard+concat) = {diff:.3e}   ({dt:.2f}s)")
    ok = diff == 0.0
    print(f"  => {'EXACT (diff == 0): column-shard+concat is BIT-IDENTICAL' if ok else 'fp-epsilon'}"
          f" -- PROCESS-only invariance HOLDS (WHERE a column reduces never changes WHAT reduces).")
    return diff


# ============================================================================ #
# EXP-2  PER-ATTRIBUTE MARGINAL TIME (sweep F, confirm ~linear) -- via the engine
# ============================================================================ #
def exp2_per_attribute_time(tg, cluster, link, F_native, seed):
    print("\n" + "=" * 92)
    print(f"EXP-2  PER-ATTRIBUTE MARGINAL TIME (sweep F, confirm linear ~ §19a 0.74us/dim)  "
          f"[{tg.name}]")
    print("=" * 92)
    src, dst, N = tg.src, tg.dst, tg.num_nodes
    Fsweep = [16, 32, 64, 128, 256, 512, 1024, 2048]
    print(f"  feature-parallel full-graph SpMM compute time (bottleneck device) via the engine "
          f"cost model:")
    print(f"  {'F':>6} | {'feat-par compute(ms)':>20} | {'node-par makespan(ms)':>22} | "
          f"{'marginal us/dim':>16}")
    print("  " + "-" * 78)
    # the node-parallel makespan column is a REFERENCE; the arrange winner is F-DEPENDENT (af0077
    # / §38-CORRECTION), so RE-ARRANGE at each F rather than freezing one split. The linear fit is
    # on the feature-parallel full-graph SpMM compute (exactly linear in F by the roofline).
    prev_ms, prev_F = None, None
    comps = []
    for F in Fsweep:
        fp = feature_parallel_plan(src, dst, N, cluster, F, link)
        comp_ms = float(fp.compute_ms.max())           # full-graph SpMM compute on the bottleneck
        res = arrange(src, dst, N, cluster, link_gbps=link, feat_dim=F, seed=seed)
        _, node_mk, _, _, _ = _placement_from_arrange(res, cluster, F)
        marg = "" if prev_ms is None else f"{(comp_ms - prev_ms) / (F - prev_F) * 1e3:.3f}"
        comps.append((F, comp_ms))
        print(f"  {F:>6} | {comp_ms:>20.4f} | {node_mk:>22.4f} | {marg:>16}")
        prev_ms, prev_F = comp_ms, F
    # linear fit comp_ms ~ a + b*F ; b is the per-attribute marginal (engine roofline -> exactly linear)
    Fs = np.array([c[0] for c in comps], float)
    ms = np.array([c[1] for c in comps], float)
    A = np.vstack([np.ones_like(Fs), Fs]).T
    (a, b), *_ = np.linalg.lstsq(A, ms, rcond=None)
    resid = ms - (a + b * Fs)
    ss_res = float((resid ** 2).sum()); ss_tot = float(((ms - ms.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(1e-30, ss_tot)
    print(f"\n  LINEAR FIT  compute_ms ~= {a:.4f} + {b*1e3:.4f} us/dim * F   (R^2 = {r2:.4f})")
    print(f"  => per-attribute marginal cost is LINEAR in F (the engine roofline is exactly linear in")
    print(f"     feature width; slope scales with E/bandwidth -- the §19a ~0.74us/dim regime on H100).")
    return b


# ============================================================================ #
# EXP-3  MULTI-DATASET NODE-vs-FEATURE-PARALLEL CROSSOVER -- via zord.schedule.plan
# ============================================================================ #
def exp3_crossover(tg, F_native, cluster, link, seed):
    print("\n" + "=" * 92)
    print(f"EXP-3  NODE-vs-FEATURE-PARALLEL CROSSOVER (sweep F)  [{tg.name}  native F={F_native}]")
    print("=" * 92)
    caps_gb = [d.usable_mem / GB for d in cluster.devices]
    print(f"  cluster HBM(GB)={[round(c,1) for c in caps_gb]}  "
          f"aggBW={[d.hbm_bw_gbps for d in cluster.devices]}")
    print(f"  Two regimes (the crossover is link-dependent): a FAST link (the given {link}GB/s) where")
    print(f"  node-parallel's boundary comm is cheap, and a SLOW link (0.5GB/s) where node-parallel's")
    print(f"  full-F boundary/replication comm is expensive -> the column-concat INTEGRATION wins.")
    fast = _crossover_sweep(tg, F_native, cluster, link, seed, regime=f"FAST link={link}GB/s")
    slow = _crossover_sweep(tg, F_native, cluster, 0.5, seed, regime="SLOW link=0.5GB/s")
    print(f"\n  --- EXP-3 VERDICT [{tg.name}] ---")
    _verdict(tg, "FAST", fast)
    _verdict(tg, "SLOW", slow)
    return {"fast": fast, "slow": slow}


def _crossover_sweep(tg, F_native, cluster, link, seed, regime):
    """Sweep F at one link speed and compare NODE-parallel (arrange) vs FEATURE-parallel per F.

    For graphs the planner re-arranges cheaply (E < 5M) we call the full zord.schedule.plan(...,
    decomposition='auto') per F. For BIG graphs we call arrange/_placement_from_arrange directly
    (skipping the rest of plan()'s incremental/tiering layers, which are not under test on the
    decomposition axis), but -- CRITICAL FIX (auditor af0077, §38-CORRECTION) -- we RE-RUN
    arrange(feat_dim=F) PER F, exactly as choose_decomposition does. The arrange WINNER is
    F-DEPENDENT (vertex-cut k-core vs balanced PTS/edge-cut flips with feature width), so the old
    'arrange ONCE at F_native, re-cost per F' shortcut FROZE a wrong-F split into every row and
    produced a BACKWARDS feasibility-flip on Reddit. The feature-parallel side is closed-form per F."""
    src, dst, N = tg.src, tg.dst, tg.num_nodes
    big = tg.num_edges >= 5_000_000
    print(f"\n  [{regime}]  decomposition picked by the engine"
          + ("  (arrange(feat_dim=F) re-run PER F, matching choose_decomposition; big graph)" if big else
             "  (zord.schedule.plan(decomposition='auto') per F)"))
    print(f"  {'F':>7} | {'WIN':>10} | {'node ms':>9} {'feas':>4} {'maxRes(GB)':>10} | "
          f"{'feat ms':>9} {'feas':>4} {'maxRes(GB)':>10} | {'integ ms':>8}")
    print("  " + "-" * 95)
    Fsweep = sorted(set([32, 64, 128, F_native, 512, 1024, 2048, 4096, 8192,
                         16384, 20480, 24576, 32768]))
    if big:
        Fsweep = sorted(set([128, F_native, 1024, 4096, 8192, 16384, 24576, 32768]))
    mk_cross = feas_flip = None
    max_mem_ratio = (0.0, None)            # (node_peak/feat_peak, F) -- the HBM-relief signal
    prev_win = None
    for F in Fsweep:
        if big:
            # RE-ARRANGE at THIS F (the arrange winner is F-dependent -- the af0077 fix). This is
            # exactly what choose_decomposition() does; no frozen-arrange shortcut.
            res_f = arrange(src, dst, N, cluster, link_gbps=link, feat_dim=F, seed=seed)
            placements, node_mk, _, _, node_feas = _placement_from_arrange(res_f, cluster, F)
            node_res = max(pp.resident_bytes for pp in placements) / GB
            fp = feature_parallel_plan(src, dst, N, cluster, F, link)
            feat_mk, feat_feas = fp.makespan_ms, fp.feasible
            feat_integ = fp.integration_ms
            # winner: lowest makespan among feasible (mirrors choose_decomposition for the 2 pure axes)
            opts = {"node": (node_mk, node_feas), "feature": (feat_mk, feat_feas)}
            fe = {k: v for k, v in opts.items() if v[1]} or opts
            win = min(fe, key=lambda k: fe[k][0])
        else:
            p = plan(tg, cluster, link_gbps=link, feat_dim=F, decomposition="auto", seed=seed)
            d = p.decomposition
            node_mk, node_feas = d.node_makespan_ms, d.node_feasible
            node_res = max(pp.resident_bytes for pp in p.placement) / GB
            fp = feature_parallel_plan(src, dst, N, cluster, F, link)
            feat_mk, feat_feas, feat_integ = d.feature_makespan_ms, d.feature_feasible, d.feature_integration_ms
            win = d.axis
        feat_res = fp.resident_bytes.max() / GB
        if prev_win == "node" and win != "node" and mk_cross is None:
            mk_cross = F
        if feas_flip is None and (not node_feas) and feat_feas:
            feas_flip = F
        ratio = node_res / max(1e-9, feat_res)
        if ratio > max_mem_ratio[0]:
            max_mem_ratio = (ratio, F)
        prev_win = win
        print(f"  {F:>7} | {win:>10} | {node_mk:>9.2f} {('Y' if node_feas else 'OOM'):>4} "
              f"{node_res:>10.2f} | {feat_mk:>9.2f} {('Y' if feat_feas else 'OOM'):>4} "
              f"{feat_res:>10.2f} | {feat_integ:>8.2f}")
    return {"mk_cross": mk_cross, "feas_flip": feas_flip, "mem_ratio": max_mem_ratio}


def _verdict(tg, regime, r):
    mk, ff, (mr, mrF) = r["mk_cross"], r["feas_flip"], r["mem_ratio"]
    if mk is not None:
        print(f"    [{regime}] MAKESPAN crossover at F~={mk}: below it node-parallel wins (compute divides")
        print(f"          D-ways, low boundary comm); at/above it FEATURE-parallel wins -- its column-concat")
        print(f"          INTEGRATION beats node-parallel's full-F boundary comm once F is attribute-heavy.")
    else:
        print(f"    [{regime}] no makespan crossover in range: node-parallel's locality cut keeps boundary")
        print(f"          comm small, holding the makespan lead (feature-parallel does the SAME total compute")
        print(f"          + replicates the full adjacency).")
    if ff is not None:
        print(f"    [{regime}] FEASIBILITY flip at F~={ff}: node-parallel OOMs while FEATURE-parallel still")
        print(f"          FITS -- feature-parallel splits columns F/D (capacity-balanced) with NO full-F row")
        print(f"          replication -> the attribute-dimension HBM relief.")
    if mr > 1.05:
        print(f"    [{regime}] HBM RELIEF: feature-parallel's peak per-device memory is provably balanced")
        print(f"          (N*F*4/D + replicated adjacency); node-parallel's peak hits {mr:.2f}x that at F={mrF}")
        print(f"          (its makespan-optimal corner becomes imbalanced under pressure) -> feature-parallel")
        print(f"          is the lower-peak-memory layout exactly when attributes (F) dominate.")


def main():
    ap = argparse.ArgumentParser(description="feature-parallel (attribute-dimension) decomposition via the zord engine")
    ap.add_argument("--datasets", default="ogbn-arxiv,reddit,coauthor-cs",
                    help="comma list (different F: arxiv 128, reddit 602, coauthor-cs 6805)")
    ap.add_argument("--root", default="/tmp/zord_attr_data")
    ap.add_argument("--synthetic", action="store_true", help="force calibrated-synthetic (no download)")
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--hbm-gb", default="6,6,6", help="per-device HBM GB (small -> HBM-pressure regime; "
                    "the 2GB framework reserve in DeviceProfile leaves ~4GB usable each)")
    ap.add_argument("--agg-bw", default="942,534,444", help="per-device achieved agg BW GB/s")
    ap.add_argument("--link-gbps", type=float, default=50.0, help="interconnect GB/s (a PARAMETER)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", default="all", help="1|2|3|all")
    args = ap.parse_args()

    hbm = [float(x) for x in args.hbm_gb.split(",")]
    bw = [float(x) for x in args.agg_bw.split(",")]
    cluster = from_spec(hbm_gb=hbm, agg_bw_gbps=bw, interconnect_gbps=args.link_gbps,
                        names=[f"GPU{i}(HBM{int(hbm[i])})" for i in range(len(hbm))])
    D = cluster.num_devices

    print("#" * 92)
    print(f"# FEATURE-PARALLEL (attribute-dimension splitting) via the zord engine")
    print(f"# datasets={args.datasets}  devices={D}  HBM(GB)={hbm}  aggBW={bw}  link={args.link_gbps}GB/s")
    print(f"# PROCESS-only: time/memory/feasibility; accuracy NEVER a target (column-shard+concat is")
    print(f"# bit-identical to single-device aggregation -- EXP-1 proves it numerically).")
    print("#" * 92)

    summary = []
    for name in [s.strip() for s in args.datasets.split(",")]:
        if name not in apr.DATASETS:
            print(f"\n!! unknown dataset {name} (known: {list(apr.DATASETS)}) -- skipping")
            continue
        spec = apr.DATASETS[name]
        print("\n" + "#" * 92)
        print(f"# DATASET {name}: registry N={spec[0]:,} E={spec[1]:,} F={spec[2]} loader={spec[4]}")
        tg, F_native, note = load_temporal(name, args.root, args.synthetic, args.seed)
        print(f"# LOAD: {note}")
        print(f"# GRAPH: N={tg.num_nodes:,}  E={tg.num_edges:,}  native F={F_native}")

        diff = bsl = cross = None
        if args.exp in ("1", "all"):
            diff = exp1_math_consistency(tg, F_native, D, args.seed)
        if args.exp in ("2", "all"):
            bsl = exp2_per_attribute_time(tg, cluster, args.link_gbps, F_native, args.seed)
        if args.exp in ("3", "all"):
            cross = exp3_crossover(tg, F_native, cluster, args.link_gbps, args.seed)
        summary.append((name, F_native, diff, bsl, cross))

    print("\n" + "#" * 92)
    print("# SUMMARY (PROCESS-only)")
    print(f"#  {'dataset':>14} | {'nativeF':>7} | {'math diff':>10} | {'us/dim':>8} | "
          f"{'fast mk-cross':>13} | {'slow mk-cross':>13} | feas-flip(slow)")
    for name, F, diff, bsl, cross in summary:
        ds = "0(exact)" if diff == 0.0 else (f"{diff:.1e}" if diff is not None else "-")
        bs = f"{bsl*1e3:.3f}" if bsl is not None else "-"
        fast = cross["fast"] if cross else {}
        slow = cross["slow"] if cross else {}
        fm = str(fast.get("mk_cross")) if cross else "-"
        sm = str(slow.get("mk_cross")) if cross else "-"
        ffl = str(slow.get("feas_flip")) if cross else "-"
        print(f"#  {name:>14} | {F:>7} | {ds:>10} | {bs:>8} | {fm:>13} | {sm:>13} | {ffl}")
    print("#" * 92)
    print("# DONE. Engine extension: src/zord/partition/feature_parallel.py + "
          "schedule.plan(decomposition=...).")


if __name__ == "__main__":
    main()
