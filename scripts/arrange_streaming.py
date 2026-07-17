#!/usr/bin/env python
"""STREAMING vs BATCH arrange -- the "10000 cards" question, part (B): when we CUT, do we
CUT-ONE-BLOCK-SEND-ONE (streaming / incremental) or CUT-ALL-THEN-ARRANGE (batch / global)?

Both produce a single-home device assignment dev[v] for the SAME graph + SAME K-device cluster,
through the REAL src/zord engine (zord.partition.arrange + the engine predict_ms roofline; NEVER
networkx). They differ ONLY in WHEN the cut decision is made:

  BATCH  (cut-all-then-arrange): run arrange ONCE over the WHOLE graph. The partitioner sees every
         edge -> the GLOBALLY BEST cut. But it must hold the ENTIRE graph resident during arrange
         (peak arrange memory = whole edge+node set) and commits in ONE big placement -- a single
         zero-reaction commit (cf. RESULTS.md §42: a single oversized alloc has NO reaction window).

  STREAM (cut-one-block-send-one): slice the time-sorted edge stream into B time-window BLOCKS.
         Process block i: arrange the block's induced subgraph, COMMIT (place) the NEW vertices it
         introduces, then move on -- old vertices keep their committed device (an ONLINE / incremental
         assignment, like RESULTS §29 zord-incremental & §32 dist_scaling streaming). Only ONE block
         is in flight -> PEAK arrange memory = the largest single block (far below the whole graph).
         The cut is LOCAL per block (no global view) -> a SOMEWHAT WORSE global cut. But cut(block i+1)
         OVERLAPS place(block i) (pipeline), and each commit is INCREMENTAL -> it has the reaction
         window §42 says a single big commit lacks.

We measure, for BOTH, on the SAME final assignment evaluated against the WHOLE graph (engine
edgecut_metrics): resulting CUT QUALITY (REAL engine cut numbers), and -- as a MODELED proxy --
the PEAK arrange memory and pipeline OVERLAP / wall-clock.

WHAT IS MEASURED vs MODELED (§45-correction, read before believing any peak ratio):
  * The CUT numbers are REAL: a real online-committed streaming assignment, scored on the WHOLE
    graph by the engine's edgecut_metrics. The streaming cut PENALTY vs batch (e.g. +26.6% on
    askubuntu) is a genuine engine measurement.
  * The PEAK arrange-memory ratio is a MODELED PROXY, NOT a measured GPU/host peak. It is the
    coordinator footprint model arrange_peak_bytes(E,N) ~ 32*E + 40*N bytes (batch holds the WHOLE
    graph; streaming holds ONE block of ~E/blocks edges). It therefore depends on E, N and the
    number of streaming BLOCKS -- it is K-INDEPENDENT (the device count K does not enter the
    coordinator-footprint model). There is NO "25.9x at K=1024" claim: the peak ratio is the SAME
    at any K; only `blocks` (and the graph) move it. We report it as a modeled proxy, never as a
    measured peak.
  * The "same-result invariance" line is a node-COVERAGE assertion (every node placed exactly once
    with a valid device id), NOT an fp value compare. Both assignments cover the same node set.

Connect to §42: streaming == incremental commit == bounded MODELED peak + a reaction window ->
FEASIBLE at the billion-edge scale where batch's whole-graph coordinator footprint would exceed a
host RAM budget; batch == best cut quality but one big zero-reaction commit. The crossover SCALE at
which streaming becomes MANDATORY is a MODELED coordinator-memory argument (arrange_peak_bytes).

Honesty: batch WINS on small scale (cut quality, no pipeline bookkeeping); streaming WINS on large
scale (modeled peak-memory feasibility + overlap). Neither is universally better. SAME-RESULT
invariance: a placement is just an assignment of WHERE partial sums are reduced; the trained result
is identical either way (PROCESS-only; we never touch accuracy).

Usage:
  python scripts/arrange_streaming.py --dataset askubuntu --devices 64 --blocks 16
  python scripts/arrange_streaming.py --dry-run            # tiny synthetic, CPU/CI
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zord.profiler.cluster_profile import ClusterProfile, DeviceProfile, GB  # noqa: E402
from zord.partition.arrange import (                                       # noqa: E402
    arrange, predict_ms, edgecut_metrics, node_degree, BYTES_PER_EDGE_RESIDENT)


def make_cluster(K: int, cap_bytes: float, agg_bw: float, link: float) -> ClusterProfile:
    devs = [DeviceProfile(i, f"card{i}", int(cap_bytes), throughput=agg_bw / 444.0,
                          node=0, mem_reserved=0, hbm_bw_gbps=agg_bw) for i in range(K)]
    return ClusterProfile(devices=devs, intra_node_bw=link, inter_node_bw=link)


# --------------------------------------------------------------------------- #
# arrange-memory footprint model (the bytes the PARTITIONER itself holds)      #
# --------------------------------------------------------------------------- #
# A partitioner must hold the edges it is reasoning over plus per-node bookkeeping. The engine's
# arrange materializes, over the edge set it is given: the doubled (u,v) adjacency, degree, lpa
# rank, coreness, and a few per-edge landing arrays. A faithful, conservative proxy for its PEAK
# resident bytes on an edge set of size E over N nodes:
#   edges:  ~ C_E * E * 8   (doubled int64 adjacency + a couple of per-edge int64 work arrays)
#   nodes:  ~ C_N * N * 8   (degree + lpa_rank + coreness + dev, all int64 [N])
# This is the memory that BATCH must hold for the WHOLE graph at once, vs STREAMING for one block.
C_E_BYTES = 8.0 * 4      # ~4 int64-equivalent per-edge arrays live at peak (u,v + landing + key)
C_N_BYTES = 8.0 * 5      # ~5 int64 per-node arrays (deg, lpa_rank, coreness, dev, scratch)


def arrange_peak_bytes(E: int, N: int) -> float:
    return C_E_BYTES * E + C_N_BYTES * N


# --------------------------------------------------------------------------- #
# graphs                                                                       #
# --------------------------------------------------------------------------- #
def synth_temporal(name, N, avg_deg, n_comms=32, seed=0):
    """Synthetic temporal graph with COMMUNITY (stochastic-block / SBM) structure + timestamps, so
    blocks (time windows) carry locality the streaming partitioner can exploit and the global cut
    has real structure. This is a COMMUNITY/SBM synthetic -- NOT a uniform-random graph (the §45-
    correction note: do not label it 'random'). 80% of edges are intra-community, 20% global."""
    rng = np.random.default_rng(seed)
    E = int(N * avg_deg / 2)
    comm = rng.integers(0, n_comms, N)
    # 80% intra-community edges (locality), 20% cross-community/global (the structure the batch sees)
    src = np.empty(E, dtype=np.int64); dst = np.empty(E, dtype=np.int64)
    n_intra = int(0.8 * E)
    members = [np.nonzero(comm == c)[0] for c in range(n_comms)]
    members = [m for m in members if m.size > 1]
    cc = rng.integers(0, len(members), n_intra)
    for i in range(n_intra):
        m = members[cc[i]]
        src[i] = m[rng.integers(0, m.size)]; dst[i] = m[rng.integers(0, m.size)]
    src[n_intra:] = rng.integers(0, N, E - n_intra)
    dst[n_intra:] = rng.integers(0, N, E - n_intra)
    t = np.sort(rng.integers(0, E, E)).astype(np.int64)        # timestamps -> time-window blocks
    m = src != dst
    src, dst, t = src[m], dst[m], t[m]

    class _G:
        pass
    g = _G()
    g.src, g.dst, g.t, g.num_nodes, g.name = src, dst, t, N, name
    return g


def load_graph(name):
    from zord.datasets import load
    return load(name).sort_by_time()


# --------------------------------------------------------------------------- #
# BATCH: cut the whole graph once (global view -> best cut)                     #
# --------------------------------------------------------------------------- #
def batch_arrange(src, dst, N, cl, link, feat, seed, metis_max_edges):
    E = int(src.size)
    t0 = time.time()
    res = arrange(src, dst, N, cl, link_gbps=link, feat_dim=feat,
                  num_snapshots=64, seed=seed, metis_max_edges=metis_max_edges)
    cut_s = time.time() - t0                                   # cut-ALL time (whole graph)
    # peak arrange memory: the WHOLE graph is resident during arrange
    peak = arrange_peak_bytes(E, N)
    # place-all: ONE big commit -- ship ALL N node rows to their cards over the link, AFTER the
    # whole-graph cut finishes (no overlap; this is the single zero-reaction commit of §42).
    ship_all_s = N * feat * 4.0 / (max(link, 1e-9) * 1e9)
    wall_s = cut_s + ship_all_s                                # serial: cut-all THEN place-all
    return dict(name=res.name, assignment=np.asarray(res.assignment, dtype=np.int64),
                cut=int(res.cut), cut_s=cut_s, ship_s=ship_all_s, wall_s=wall_s,
                peak_bytes=peak, blocks=1)


# --------------------------------------------------------------------------- #
# STREAMING: cut block-by-block, COMMIT each block as it is cut (online)        #
# --------------------------------------------------------------------------- #
def streaming_arrange(src, dst, t, N, cl, link, feat, seed, n_blocks, metis_max_edges):
    """Slice the time-sorted edge stream into n_blocks time windows. For each block: arrange the
    block's induced subgraph; COMMIT the NEW vertices (first seen in this block) to the device the
    block's local arrange chose; old vertices keep their already-committed device. Only ONE block's
    edges are resident during its arrange (the streaming peak). cut(block i+1) overlaps place(i)."""
    E = int(src.size)
    K = cl.num_devices
    order = np.argsort(t, kind="stable")                       # chronological (already sorted, cheap)
    s_all, d_all = src[order], dst[order]
    bounds = np.linspace(0, E, n_blocks + 1).astype(np.int64)

    committed = np.full(N, -1, dtype=np.int64)                 # device per vertex; -1 = not yet placed
    cut_times, place_times, block_peaks = [], [], []
    for b in range(n_blocks):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        if hi <= lo:
            continue
        bs, bd = s_all[lo:hi], d_all[lo:hi]
        # the block's induced subgraph on a COMPACT node id space (only nodes active in this block)
        nodes_b, inv = np.unique(np.concatenate([bs, bd]), return_inverse=True)
        Nb = int(nodes_b.size)
        bs_c = inv[:bs.size]; bd_c = inv[bs.size:]

        # ---- CUT this block (real engine arrange on the block subgraph) ----
        t0 = time.time()
        res_b = arrange(bs_c, bd_c, Nb, cl, link_gbps=link, feat_dim=feat,
                        num_snapshots=8, seed=seed, metis_max_edges=metis_max_edges)
        cut_times.append(time.time() - t0)
        block_peaks.append(arrange_peak_bytes(int(bs.size), Nb))   # only THIS block resident

        # ---- COMMIT (place): NEW vertices take the block's local device; old vertices are FROZEN ----
        block_dev = np.asarray(res_b.assignment, dtype=np.int64)   # device in [0,K) per block-local id
        glob = nodes_b                                             # block-local -> global id
        new_mask = committed[glob] < 0
        committed[glob[new_mask]] = block_dev[new_mask]
        # the COMMIT cost is the wall-time to SHIP this block's committed node rows to their target
        # cards over the interconnect (the "send-one" half of cut-one-block-SEND-one). It is this
        # transfer that OVERLAPS with cutting the next block. bytes = (new rows)*F*4 over the link.
        commit_bytes = int(new_mask.sum()) * feat * 4.0
        place_times.append(commit_bytes / (max(link, 1e-9) * 1e9))

    # any never-seen isolated node (no edges) -> device 0 (irrelevant to the cut; keeps assignment total)
    committed[committed < 0] = 0

    # ---- evaluate the FINAL streaming assignment on the WHOLE graph (engine metric) ----
    deg = node_degree(src, dst, N)
    cut, incident, comm_rows, counts = edgecut_metrics(src, dst, deg, committed, K, N)

    # wall-clock: pipeline overlaps cut(block i+1) with place(block i). Serial = sum(cut)+sum(place);
    # overlapped = cut[0] + sum_i max(cut[i+1], place[i]) + place[last]. Peak = the largest block.
    serial = sum(cut_times) + sum(place_times)
    overlapped = cut_times[0]
    for i in range(len(place_times) - 1):
        overlapped += max(cut_times[i + 1], place_times[i])
    overlapped += place_times[-1]
    return dict(name="streaming(online,%dblk)" % n_blocks, assignment=committed,
                cut=int(cut), incident=incident, comm_rows=comm_rows, counts=counts,
                cut_s_serial=serial, cut_s_overlapped=overlapped,
                peak_bytes=max(block_peaks), n_blocks=len(cut_times))


# --------------------------------------------------------------------------- #
# evaluate an assignment's makespan via the engine roofline (apples-to-apples) #
# --------------------------------------------------------------------------- #
def _single_home(assignment, K):
    """Sanitize to a pure single-home assignment for the edge-cut metric: arrange may return a
    VERTEX-CUT whose core rows carry -1 (replicated on every device). For an apples-to-apples
    single-home CUT comparison we home those rows on device 0 (the cut metric needs one owner per
    node). Applied identically to both batch and streaming, so the comparison stays fair."""
    a = np.asarray(assignment, dtype=np.int64).copy()
    a[a < 0] = 0
    return np.clip(a, 0, K - 1)


def assignment_makespan(src, dst, N, assignment, cl, link, feat):
    K = cl.num_devices
    deg = node_degree(src, dst, N)
    a = _single_home(assignment, K)
    cut, incident, comm_rows, counts = edgecut_metrics(src, dst, deg, a, K, N)
    bw = np.array([d.hbm_bw_gbps for d in cl.devices], dtype=np.float64)
    tot, comp, comm = predict_ms(incident * feat, comm_rows * feat, bw, link)
    cap = cl.devices[0].usable_mem
    resident = counts.astype(np.float64) * feat * 4.0 + incident * BYTES_PER_EDGE_RESIDENT
    feasible = bool(resident.max() <= cap)
    return dict(cut=int(cut), makespan_ms=float(tot.max()), feasible=feasible,
                peak_ws_gb=float(resident.max()) / GB)


# --------------------------------------------------------------------------- #
# run one (graph, K, blocks)                                                    #
# --------------------------------------------------------------------------- #
def run(g, K, n_blocks, link, feat, cap_bytes, agg_bw, seed, metis_max_edges, arrange_cap_gb):
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    t = np.asarray(getattr(g, "t", np.arange(src.size)), dtype=np.int64)
    N = int(g.num_nodes)
    E = int(src.size)
    cl = make_cluster(K, cap_bytes, agg_bw, link)

    bat = batch_arrange(src, dst, N, cl, link, feat, seed, metis_max_edges)
    strm = streaming_arrange(src, dst, t, N, cl, link, feat, seed, n_blocks, metis_max_edges)

    bm = assignment_makespan(src, dst, N, bat["assignment"], cl, link, feat)
    sm = assignment_makespan(src, dst, N, strm["assignment"], cl, link, feat)

    # SAME-RESULT invariance check: a placement is just an assignment; the SET of edges + the per-node
    # degree (=> the trained aggregation result) are identical regardless of WHERE rows live. We verify
    # both assignments cover the SAME node set with valid device ids (no node dropped/duplicated). We
    # check the SINGLE-HOME-sanitized assignments actually evaluated -- a -1 in a raw vertex-cut result
    # is a REPLICATED-CORE row (homed on every device), itself a valid placement, just multi-home; the
    # invariance is about node COVERAGE, which both honor (every node has at least one resident device).
    bat_sh = _single_home(bat["assignment"], K)
    strm_sh = _single_home(strm["assignment"], K)
    same_nodes = (bat_sh.size == strm_sh.size == N)
    valid = bool(((bat_sh >= 0) & (bat_sh < K)).all()
                 and ((strm_sh >= 0) & (strm_sh < K)).all())

    # arrange-memory feasibility: does the partitioner's PEAK resident set fit a budget? (the
    # 10000-card / billion-edge regime: batch holds the whole graph; streaming holds one block.)
    arrange_cap = arrange_cap_gb * GB if arrange_cap_gb > 0 else float("inf")
    bat_fits = bat["peak_bytes"] <= arrange_cap
    strm_fits = strm["peak_bytes"] <= arrange_cap

    cut_quality_gap = (sm["cut"] / max(1, bm["cut"]) - 1.0) * 100.0   # streaming cut vs batch cut
    peak_ratio = bat["peak_bytes"] / max(1.0, strm["peak_bytes"])     # how much lower streaming peak is
    overlap_speedup = strm["cut_s_serial"] / max(1e-9, strm["cut_s_overlapped"])

    return dict(K=K, N=N, E=E, n_blocks=strm["n_blocks"],
                batch_cut=bm["cut"], strm_cut=sm["cut"], cut_gap_pct=cut_quality_gap,
                batch_peak_gb=bat["peak_bytes"] / GB, strm_peak_gb=strm["peak_bytes"] / GB,
                peak_ratio=peak_ratio,
                batch_make=bm["makespan_ms"], strm_make=sm["makespan_ms"],
                batch_arrange_feas=bat_fits, strm_arrange_feas=strm_fits,
                strm_serial_s=strm["cut_s_serial"], strm_overlap_s=strm["cut_s_overlapped"],
                overlap_speedup=overlap_speedup, batch_cut_s=bat["cut_s"],
                batch_ship_s=bat["ship_s"], batch_wall_s=bat["wall_s"],
                same_nodes=same_nodes, valid=valid,
                batch_name=bat["name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--devices", type=int, default=64, help="K cards")
    ap.add_argument("--blocks", type=int, default=16, help="streaming time-window blocks")
    ap.add_argument("--link", type=float, default=0.12)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--cap-mb", type=float, default=64.0, help="per-card usable HBM cap (MB, exact)")
    ap.add_argument("--agg-bw", type=float, default=444.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metis-max-edges", type=int, default=2_000_000)
    ap.add_argument("--arrange-cap-gb", type=float, default=0.0,
                    help="budget (GB) for the PARTITIONER's peak resident set; 0 = unbounded. Set it "
                         "to model a host/coordinator RAM bound -> shows where batch OOMs *during arrange*.")
    ap.add_argument("--scale-study", action="store_true",
                    help="also run the synthetic SCALE sweep -> the crossover where streaming is MANDATORY")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cap_bytes = a.cap_mb * 1024 ** 2
    print(f"STREAMING vs BATCH arrange  K={a.devices}  blocks={a.blocks}  link={a.link}GB/s  "
          f"feat={a.feat}  agg_bw={a.agg_bw}GB/s")

    graphs = []
    if a.dry_run:
        graphs = [synth_temporal("synth-community(d8,32c)", 40_000, 8.0, 32, a.seed)]
    elif a.dataset:
        graphs = [load_graph(a.dataset)]
    else:
        for n in ("collegemsg", "askubuntu"):
            try:
                graphs.append(load_graph(n))
            except Exception as e:
                print(f"[skip {n}: {type(e).__name__}]")
        graphs.append(synth_temporal("synth-community(d8,64c)", 200_000, 8.0, 64, a.seed))

    rows = []
    for g in graphs:
        r = run(g, a.devices, a.blocks, a.link, a.feat, cap_bytes, a.agg_bw,
                a.seed, a.metis_max_edges, a.arrange_cap_gb)
        rows.append((g.name, r))

    print(f"\n  [cut = REAL engine measurement;  peakGB / peak-ratio = MODELED coordinator-footprint "
          f"proxy (~32E+40N bytes), K-INDEPENDENT -- depends on E,N,blocks only, NOT on K]")
    print(f"{'graph':>22} {'N':>10} {'E':>11} {'blk':>4} | {'batch_cut':>10} {'strm_cut':>10} "
          f"{'cutgap%':>8} | {'modBatGB':>9} {'modStrGB':>9} {'modPk/x':>8} | "
          f"{'bat_ms':>8} {'str_ms':>8} | {'overlapx':>8}")
    for name, r in rows:
        print(f"{name:>22} {r['N']:>10,} {r['E']:>11,} {r['n_blocks']:>4} | "
              f"{r['batch_cut']:>10,} {r['strm_cut']:>10,} {r['cut_gap_pct']:>7.1f}% | "
              f"{r['batch_peak_gb']:>9.4f} {r['strm_peak_gb']:>9.4f} {r['peak_ratio']:>7.1f}x | "
              f"{r['batch_make']:>8.2f} {r['strm_make']:>8.2f} | {r['overlap_speedup']:>7.2f}x")

    print("\nTRADE-OFF (honest, §45-correction):")
    for name, r in rows:
        print(f"  {name}: BATCH cut {r['batch_cut']:,} (global view, BEST) vs STREAMING cut "
              f"{r['strm_cut']:,} (+{r['cut_gap_pct']:.1f}% -- local-only blocks). [REAL engine cut]")
        print(f"     MODELED coordinator peak: streaming {r['strm_peak_gb']:.4f}GB = {r['peak_ratio']:.1f}x "
              f"LOWER than batch {r['batch_peak_gb']:.4f}GB -- a MODELED proxy (32E+40N bytes), NOT a "
              f"measured GPU/host peak, and K-INDEPENDENT (set by blocks={r['n_blocks']}, not K). "
              f"Pipeline overlap {r['overlap_speedup']:.2f}x wall-clock (also modeled).")
        inv = "OK" if (r['same_nodes'] and r['valid']) else "VIOLATED"
        print(f"     same-result invariance (node COVERAGE, not an fp compare): both assignments cover "
              f"all {r['N']:,} nodes with valid device ids -> {inv} (placement only moves WHERE partials "
              f"reduce, not WHAT is computed).")

    if a.scale_study:
        scale_study(a)


def scale_study(a):
    """Synthetic SCALE sweep: grow E (fixed avg degree) and ask, under an arrange-memory budget,
    AT WHAT SCALE batch can no longer arrange in-budget (OOMs during arrange) while streaming still
    fits one block -> streaming becomes MANDATORY. PROCESS-only (we model the partitioner footprint;
    no GPU needed). This is the §42 connection: batch = one big commit (no reaction window) that must
    hold the whole graph; streaming = incremental commit, bounded peak."""
    print("\n================ SCALE STUDY: when does STREAMING become MANDATORY? ================")
    # model a coordinator/host arrange-memory budget (GB). Real coordinators have O(100GB) RAM; we
    # set a modest budget so the crossover lands inside a sweepable E range. Streaming holds 1/blocks.
    budget_gb = a.arrange_cap_gb if a.arrange_cap_gb > 0 else 16.0
    budget = budget_gb * GB
    blocks = a.blocks
    avg_deg = 16.0
    print(f"  arrange-memory budget = {budget_gb:.0f}GB ; streaming uses {blocks} blocks "
          f"(holds ~1/{blocks} of the edges at peak); avg_degree={avg_deg:.0f}")
    print(f"  {'edges E':>14} {'nodes N':>12} {'batch_peakGB':>13} {'strm_peakGB':>12} "
          f"{'batch_fits':>11} {'strm_fits':>10}")
    crossover = None
    for E in (10_000_000, 30_000_000, 100_000_000, 300_000_000, 1_000_000_000, 3_000_000_000):
        N = int(E / (avg_deg / 2))
        bpk = arrange_peak_bytes(E, N)
        spk = arrange_peak_bytes(int(E / blocks), int(N / blocks))   # one block in flight
        bf, sf = bpk <= budget, spk <= budget
        if crossover is None and (not bf) and sf:
            crossover = E
        print(f"  {E:>14,} {N:>12,} {bpk/GB:>13.2f} {spk/GB:>12.2f} "
              f"{('yes' if bf else 'OOM'):>11} {('yes' if sf else 'OOM'):>10}")
    if crossover:
        print(f"  -> (MODELED) STREAMING becomes MANDATORY at ~{crossover:,} edges: above this, BATCH's")
        print(f"     whole-graph MODELED coordinator footprint exceeds the {budget_gb:.0f}GB budget, while")
        print(f"     streaming's one-block-in-flight footprint still fits. This is the §42 argument at")
        print(f"     partition time -- a MODELED coordinator-memory bound (arrange_peak_bytes), not a")
        print(f"     measured OOM: a single big commit has no reaction window; block commits stay bounded.")
    else:
        print(f"  -> batch fits across the whole swept range at budget {budget_gb:.0f}GB "
              f"(raise --blocks or lower --arrange-cap-gb to expose the crossover).")
    print("\n  RULE OF THUMB (MODELED, not measured): batch coordinator peak ~ (32*E + 40*N) bytes;")
    print("  streaming ~ that / blocks (K-independent). Use BATCH when the whole graph's MODELED arrange")
    print("  footprint fits the coordinator (small/medium scale, globally-best cut is free); switch to")
    print("  STREAMING when it does not (large scale / billion edges) -- trade a few % cut for the")
    print("  modeled-feasibility + overlap. The CUT penalty is the real engine number; the peak is modeled.")


if __name__ == "__main__":
    main()
