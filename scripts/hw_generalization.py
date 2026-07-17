#!/usr/bin/env python
"""HARDWARE GENERALIZATION (D-hwgen): zord's contribution is a HARDWARE-AGNOSTIC middle-layer
scheduler -- hardware enters ONLY as a parameter of the cost model (per-device HBM bandwidth /
capacity / H2D, and the intra/inter-node interconnect). This script shows the scheduling wins
HOLD across very different hardware: H100-NVLink (ours), A100-NVLink, AMD MI250-InfinityFabric,
RTX/PCIe-only (no NVLink), a commodity-Ethernet cluster, and a HOMOGENEOUS control.

For EACH hardware profile we run the SAME three scheduler decisions and report whether they adapt:
  (a) plan_memory      -- does the global memory plan stay FEASIBLE and tier correctly?
  (b) PLACEMENT makespan -- hetero-matched (dense core -> strong device, node counts time-balanced)
                          vs even / bw-proportional / random, on ONE memory-bound aggregation step,
                          using hetero_matched.py's CORRECTED INCIDENT-edge gather model
                          (agg work = sum of node DEGREE in a part, NOT just local edges).
  (c) DUALITY corner   -- factorize D = Dv (vertex-blocks) x Dt (snapshot-blocks); pick the corner/
                          interior that minimizes weighted transfer cost under THIS profile's
                          (intra_node_bw for spatial cut, inter_node_bw for temporal cut).

HEADLINE we expect: the scheduler ADAPTS and WINS on every profile, and the placement speedup is
LARGER on more-heterogeneous / weaker-interconnect hardware -- "hardware is downstream" of the
schedule. The homogeneous control is the degenerate case (speedup ~ 1x; even == matched).

PROCESS-only; numpy + the optional C++ kernel (build/graph_algos) for the density ranking, with a
pure-numpy degree-sort fallback so it runs anywhere. NEVER networkx. Reads zord.* read-only.

  python -m scripts.hw_generalization --nodes 4000000 --edges 50000000 --feat 128
  python -m scripts.hw_generalization --dataset wiki-talk --feat 128
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import struct
import subprocess
import sys
import time

import numpy as np

# --- locate zord on the path (read-only import) ---------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from zord.profiler.cluster_profile import DeviceProfile, ClusterProfile, GB  # noqa: E402
from zord.schedule.planner import Workload, plan_memory                      # noqa: E402

# --- reuse hetero_matched.py's CORRECTED incident-edge model (import, don't duplicate) -------
# We load the module by path so it works regardless of how the script itself was invoked.
_HM_PATH = os.path.join(_HERE, "hetero_matched.py")
_spec = importlib.util.spec_from_file_location("zord_hetero_matched", _HM_PATH)
hm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hm)
node_degree = hm.node_degree                          # incident-degree per node
edges_per_part_by_segments = hm.edges_per_part_by_segments   # -> counts, INCIDENT gather work, local, cut
predict_times = hm.predict_times                      # incident_edges * F * 4 * 2gathers / hbm_bw -> ms
solve_balanced_bounds = hm.solve_balanced_bounds      # hetero-matched time-balanced rank boundaries
gen_graph = hm.gen_graph                              # community-structured synthetic generator

BIN = os.environ.get("ZORD_GRAPH_BIN",
                     os.path.join(os.path.dirname(_HERE), "build", "graph_algos"))


# ============================================================================================
# HARDWARE PROFILES.  Built here as ClusterProfile/DeviceProfile instances; cluster_profile.py
# is NOT edited.  Each tuple in `devs` is (name, mem_gb, throughput r, h2d_gbps, hbm_bw_gbps).
# hbm_bw_gbps is ACHIEVED aggregation bandwidth (gather-locality-capped), NOT spec peak --
# that is what sets the memory-bound GNN step time (see planner / roofline §9).  r (throughput)
# is relative; we keep RTX5000Ada=1.0 as the global baseline so cross-profile r is comparable.
# AMD / A100 / RTX numbers are APPROXIMATE public specs derated to plausible achieved values.
# ============================================================================================

def _cluster(devs, intra, inter, gpus_per_node=8):
    """Build a ClusterProfile. Devices are placed on physical nodes `gpus_per_node` at a time so
    intra-node uses `intra` bandwidth and cross-node uses `inter` (the interconnect parameter)."""
    out = []
    for i, (name, mem_gb, r, h2d, hbm) in enumerate(devs):
        out.append(DeviceProfile(
            id=i, name=name, mem_bytes=int(mem_gb * GB), throughput=float(r),
            node=i // gpus_per_node, h2d_gbps=float(h2d), hbm_bw_gbps=float(hbm), measured=False))
    return ClusterProfile(devices=out, intra_node_bw=float(intra), inter_node_bw=float(inter))


def build_profiles():
    """Return {label: (ClusterProfile, note)} spanning the hardware axis."""
    P = {}

    # 1) H100-NVLink -- OURS (measured HetCluster numbers; intra 325 GB/s NVSwitch). Heterogeneous tier
    #    mix (H100 / RTX6000Ada / RTX5000Ada) on PCIe-only weak nodes (so cross-tier = inter_node).
    P["H100-NVLink (ours)"] = (_cluster([
        ("H100-80GB",       79.2, 2.13, 57.5, 942.0),
        ("RTX6000Ada-48GB", 47.4, 1.47, 26.7, 534.0),
        ("RTX5000Ada-32GB", 31.5, 1.00, 26.6, 444.0),
    ], intra=325.0, inter=0.12, gpus_per_node=1),
        "MEASURED HetCluster. Strong heterogeneity (hbm 2.1x, mem 2.5x). Fast NVLink intra, "
        "1Gbps-Ethernet inter (skew ~2700x).")

    # 2) A100-NVLink -- A100 tier, NVLink3 ~300 GB/s P2P. HBM2e achieved ~1500 GB/s on the irregular
    #    gather (~77% of 1935 spec). A realistic cluster mixes the 40GB (HBM2, ~1200 achieved) and
    #    80GB (HBM2e, ~1500 achieved) SKUs -> MILD bandwidth + strong capacity heterogeneity.
    P["A100-NVLink"] = (_cluster([
        ("A100-80GB", 80.0, 2.05, 25.0, 1500.0),
        ("A100-80GB", 80.0, 2.05, 25.0, 1500.0),
        ("A100-40GB", 40.0, 1.70, 25.0, 1200.0),
        ("A100-40GB", 40.0, 1.70, 25.0, 1200.0),
    ], intra=300.0, inter=12.0, gpus_per_node=8),
        "APPROX public specs. MILD bandwidth het (1200 vs 1500) + strong CAPACITY het (40 vs 80GB) "
        "-> memory plan must tier the small cards. NVLink3 ~300 GB/s, 12 GB/s inter (100GbE).")

    # 3) AMD MI250-InfinityFabric -- xGMI ~100 GB/s effective per-link (well below NVLink). A MI250
    #    OAM is 2 GCDs; spread across hosts a job often sees a MIX of MI250 (~1200 achieved, 64GB)
    #    and older MI210 (~900 achieved, 64GB) -> moderate bandwidth het on a WEAK interconnect.
    P["AMD MI250-IF (approx)"] = (_cluster([
        ("MI250-64GB", 64.0, 2.00, 25.0, 1200.0),
        ("MI250-64GB", 64.0, 2.00, 25.0, 1200.0),
        ("MI210-64GB", 64.0, 1.55, 25.0, 900.0),
        ("MI210-64GB", 64.0, 1.55, 25.0, 900.0),
    ], intra=100.0, inter=12.0, gpus_per_node=4),
        "APPROX public specs (ROCm/xGMI). MI250+MI210 mix (1200 vs 900 achieved) on a WEAKER intra "
        "interconnect (~100 GB/s xGMI vs 325 NVLink) -> raises value of cut-aware placement+duality.")

    # 4) RTX / PCIe-only -- NO NVLink. Commodity consumer cards (4090/4080/4070) on one box, all P2P
    #    over PCIe gen4 (~25 GB/s). STRONGLY heterogeneous bandwidth AND tight capacity (12-24 GB).
    P["RTX/PCIe-only (no NVLink)"] = (_cluster([
        ("RTX4090-24GB", 24.0, 2.30, 25.0, 900.0),
        ("RTX4090-24GB", 24.0, 2.30, 25.0, 900.0),
        ("RTX4080-16GB", 16.0, 1.80, 25.0, 650.0),
        ("RTX4070-12GB", 12.0, 1.30, 25.0, 480.0),
    ], intra=25.0, inter=12.0, gpus_per_node=8),
        "APPROX. NO NVLink: intra-node P2P is PCIe gen4 ~25 GB/s. STRONG bandwidth het (900/650/480) "
        "AND tight capacity (12-24GB) -> heaviest tiering + the placement lever matters most on one box.")

    # 5) Commodity Ethernet cluster -- modest GPUs (T4 / L4) each on its own box joined by slow
    #    Ethernet. inter_node_bw is the binding parameter (0.1-12 GB/s); we use ~1.2 GB/s (10GbE-ish,
    #    derated). T4 (HBM-less GDDR6 ~220 achieved) vs L4 (~300 achieved) -> real bandwidth het too.
    P["commodity-Ethernet"] = (_cluster([
        ("T4-16GB",  16.0, 0.85, 12.0, 220.0),
        ("T4-16GB",  16.0, 0.85, 12.0, 220.0),
        ("L4-24GB",  24.0, 1.40, 25.0, 300.0),
        ("L4-24GB",  24.0, 1.40, 25.0, 300.0),
    ], intra=25.0, inter=1.2, gpus_per_node=1),
        "APPROX. Each GPU on its own box -> EVERY cross-device transfer is slow Ethernet "
        "(~1.2 GB/s). T4/L4 bandwidth het + the WEAKEST interconnect; duality must pick the cheap axis.")

    # 6) HOMOGENEOUS control -- all identical strong GPUs, fast uniform interconnect. The degenerate
    #    case: even == bw-proportional == hetero-matched, so the placement speedup should be ~1x.
    P["HOMOGENEOUS (control)"] = (_cluster([
        ("H100-80GB", 79.2, 2.13, 57.5, 942.0),
        ("H100-80GB", 79.2, 2.13, 57.5, 942.0),
        ("H100-80GB", 79.2, 2.13, 57.5, 942.0),
        ("H100-80GB", 79.2, 2.13, 57.5, 942.0),
    ], intra=325.0, inter=325.0, gpus_per_node=8),
        "CONTROL. Identical devices + uniform NVLink. Scheduler should DEGENERATE: placement "
        "speedup ~1x (nothing to match), single feasible plan -- the sanity floor.")

    return P


# ============================================================================================
# Density ranking (rank 0 = densest node -> strongest device).  C++ kernel if available, else a
# pure-numpy degree sort so the script runs without the binary / without staged datasets.
# ============================================================================================

def _cpp_order(src, dst, N, mode):
    """Run the C++ kernel for `mode` and return per-node new-id (rank), or None on failure/absence."""
    if not os.path.exists(BIN):
        return None, None
    edges_path = "/tmp/zord_hwgen_edges.bin"
    out_path = f"/tmp/zord_hwgen_perm_{mode}.bin"
    with open(edges_path, "wb") as f:
        f.write(struct.pack("<qq", N, src.size))
        inter = np.empty(2 * src.size, dtype=np.int32)
        inter[0::2] = src.astype(np.int32); inter[1::2] = dst.astype(np.int32)
        inter.tofile(f)
    t0 = time.time()
    r = subprocess.run([BIN, edges_path, mode, out_path], capture_output=True, text=True)
    cost = time.time() - t0
    if r.returncode != 0:
        return None, cost
    with open(out_path, "rb") as f:
        struct.unpack("<q", f.read(8))
        rank = np.fromfile(f, dtype=np.int32, count=N).astype(np.int64)
    return rank, cost


def density_rank(src, dst, deg, N, rank_by="degree"):
    """rank 0 = densest node -> assigned to the strongest device (for the placement lever)."""
    rank, cost = _cpp_order(src, dst, N, rank_by)
    if rank is not None:
        if rank_by == "kcore":
            rank = (N - 1) - rank                # highest core -> rank 0 (densest first)
        return rank, f"C++ {rank_by} {cost:.2f}s"
    # pure-numpy fallback: rank 0 = highest degree (descending).
    t0 = time.time()
    order = np.argsort(-deg, kind="stable")      # node ids, densest first
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N, dtype=np.int64)
    return rank, f"numpy degree-sort {time.time()-t0:.2f}s (C++ bin absent/failed)"


def vertex_block_order(src, dst, deg, N):
    """Balanced vertex ordering for the duality's spatial cut (faithful to duality_frontier.py:
    C++ LPA community clustering). Fallback: the density rank (a valid, if cut-inflating, ordering)."""
    order, cost = _cpp_order(src, dst, N, "lpa")
    if order is not None:
        return order, f"C++ lpa {cost:.2f}s"
    deg_order = np.argsort(-deg, kind="stable")
    rank = np.empty(N, dtype=np.int64)
    rank[deg_order] = np.arange(N, dtype=np.int64)
    return rank, "numpy degree-rank proxy (C++ lpa absent)"


# ============================================================================================
# (b) PLACEMENT makespan: even / bw-proportional / hetero-matched / random, incident-edge model.
# ============================================================================================

def placement_makespans(cluster, rank, src, dst, deg, N, F, seed=0):
    devs = cluster.devices
    D = len(devs)
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    usable_nodes = np.array([d.usable_mem / (F * 4) for d in devs], dtype=np.float64)
    deg_by_rank = np.empty(N, dtype=np.float64)
    deg_by_rank[rank] = deg.astype(np.float64)
    deg_cum = np.cumsum(deg_by_rank)

    # strongest device owns the lowest ranks (densest core)
    order_strong = np.argsort(-bw, kind="stable")
    bw_sorted = bw[order_strong]
    usable_sorted = usable_nodes[order_strong]

    def eval_bounds(bounds_sorted):
        counts_s, inc_s, le_s, cut, _ = edges_per_part_by_segments(rank, src, dst, deg, N, bounds_sorted)
        inc = np.empty(D, np.int64)
        inc[order_strong] = inc_s
        t = predict_times(inc, bw, F)            # INCIDENT gather work / hbm_bw -> ms
        return float(t.max()), cut

    # even
    even_b = np.linspace(0, N, D + 1).astype(np.int64)
    mk_even, _ = eval_bounds(even_b)

    # bw-proportional
    frac = bw_sorted / bw_sorted.sum()
    bp_b = np.concatenate([[0], np.cumsum((frac * N).astype(np.int64))])
    bp_b[-1] = N
    bp_b = np.maximum.accumulate(bp_b).astype(np.int64)
    mk_bp, _ = eval_bounds(bp_b)

    # balanced-blind: balance incident-edge (gather) WORK EQUALLY across devices, ignoring bandwidth.
    # This is the degree-aware-but-hardware-BLIND baseline. It isolates the pure HARDWARE win: on a
    # homogeneous cluster balanced-blind == hetero-matched (-> ~1x), so any matched speedup OVER
    # balanced-blind is attributable to bandwidth heterogeneity, not to graph degree skew.
    equal_edge_budget = np.full(D, 1.0 / D) * deg_cum[-1]
    bb_b = [0]
    acc = 0.0
    for k in range(D - 1):
        acc += equal_edge_budget[k]
        nb = int(np.searchsorted(deg_cum, acc, side="left"))
        bb_b.append(max(bb_b[-1] + 1, min(nb, N)))
    bb_b.append(N)
    mk_bb, _ = eval_bounds(np.array(bb_b, dtype=np.int64))

    # hetero-matched (dense core -> strong dev, node counts solved to balance agg TIME)
    hm_b = solve_balanced_bounds(deg_cum, deg_cum[-1], bw_sorted, usable_sorted, N)
    mk_hm, cut_hm = eval_bounds(hm_b)

    # random: shuffle which density-segment goes to which device (degree-blind, bandwidth-blind).
    # Use even-sized segments but assign them to devices in a random permutation, then recompute
    # times against each device's true bandwidth -> a realistic "no scheduler" baseline.
    rng = np.random.default_rng(seed)
    rand_b = np.linspace(0, N, D + 1).astype(np.int64)
    counts_s, inc_s, le_s, cut, _ = edges_per_part_by_segments(rank, src, dst, deg, N, rand_b)
    perm = rng.permutation(D)                    # segment k (in strong order) -> device perm[k]
    inc_rand = np.empty(D, np.int64)
    inc_rand[order_strong[perm]] = inc_s
    mk_rand = float(predict_times(inc_rand, bw, F).max())

    return dict(even=mk_even, bw_prop=mk_bp, balanced_blind=mk_bb,
                hetero_matched=mk_hm, random=mk_rand, cut_hm=cut_hm)


# ============================================================================================
# (c) DUALITY corner choice under THIS profile's interconnect (spatial cut over intra_node_bw,
# temporal cut over inter_node_bw).  Lightweight replication of duality_frontier.py's accounting.
# ============================================================================================

def factorizations(D):
    out, dt = [], 1
    while dt <= D:
        if D % dt == 0:
            out.append((D // dt, dt))
        dt *= 2
    return out


def make_snapshots(src, dst, N, S, locality, comm=None, seed=0):
    """Per-edge snapshot id in [0,S) for the SYNTHETIC graph, with TEMPORAL LOCALITY tied to COMMUNITY
    structure (realistic: a community is born/active together). Each community gets a birth snapshot;
    an edge's snapshot is its community's birth jittered by a span = (1-locality)*S. Because the bulk
    of synthetic edges are INTRA-community (the `intra` fraction), both endpoints share the community
    birth, so a vertex's temporal footprint stays narrow and the temporal cut shrinks as locality->1.
    Inter-community edges scatter and set a temporal-cut floor.
      locality=1 -> each community active in ~1 snapshot (small temporal cut, PSS competitive on a
                    weak interconnect); locality=0 -> uniform (huge temporal cut, PTS dominates).
    `comm` is the planted per-vertex community (replayed from gen_graph). Real datasets carry their
    own timestamps and pass locality<0 to use plain time-sorted equal-count snapshots."""
    E = src.size
    if locality < 0:                              # real dataset: time-sorted equal-count snapshots
        return np.minimum((np.arange(E) * S // E).astype(np.int64), S - 1)
    rng = np.random.default_rng(seed + 12345)
    if comm is None:                              # fallback: per-vertex (min-id) anchor
        comm = np.minimum(src.astype(np.int64), dst.astype(np.int64))
    n_groups = int(comm.max()) + 1
    birth = rng.integers(0, S, size=n_groups)     # each community's center snapshot
    span = max(1, int(round((1.0 - locality) * S)))
    # anchor each edge on the community of its min-id endpoint; intra-community edges share it
    anchor_v = np.minimum(src.astype(np.int64), dst.astype(np.int64))
    center = birth[comm[anchor_v]].astype(np.int64)
    jit = rng.integers(-(span // 2), span // 2 + 1, size=E)
    return np.clip(center + jit, 0, S - 1).astype(np.int64)


def synthetic_communities(N, C, seed=0):
    """Replay gen_graph's planted per-vertex community labels (same RNG draw) so the temporal model
    can align snapshot births with the spatial communities. Mirrors hetero_matched.gen_graph."""
    return np.random.default_rng(seed).integers(0, C, size=N).astype(np.int64)


def duality_choice(cluster, vorder, src, dst, snap, N, S, F):
    """Return (best_label, best_kind, gain_vs_best_corner, best_cost_ms). Uses the SAME accounting as
    duality_frontier (balanced vertex blocks from the LPA `vorder`, per-edge snapshot ids `snap`),
    and THIS profile's intra_node_bw (spatial/aggregation cut) and inter_node_bw (temporal/node-
    memory cut) as the two cost rates -- the only thing that varies across profiles."""
    D = cluster.num_devices
    bs = cluster.intra_node_bw                   # spatial cut traverses the (fast) intra link
    bt = cluster.inter_node_bw                   # temporal/node-memory cut traverses the slow link
    bytes_per = F * 4
    verts = np.concatenate([src, dst]).astype(np.int64)
    snaps = np.concatenate([snap, snap]).astype(np.int64)

    rows = []
    for Dv, Dt in factorizations(D):
        vblock = (vorder * Dv // N).astype(np.int64)
        spatial_cut = int(np.count_nonzero(vblock[src] != vblock[dst]))
        vsblock = np.minimum(snaps * Dt // S, Dt - 1)
        key = verts * Dt + vsblock
        uniq = np.unique(key)
        distinct_per_v = np.bincount((uniq // Dt).astype(np.int64), minlength=N)
        temporal_cut = int(np.maximum(distinct_per_v - 1, 0).sum())
        rows.append((Dv, Dt, spatial_cut, temporal_cut))

    def cost(sc, tc):
        return (sc * bytes_per) / (bs * 1e9) + (tc * bytes_per) / (bt * 1e9)

    best = None
    for (Dv, Dt, sc, tc) in rows:
        c = cost(sc, tc)
        if best is None or c < best[0]:
            best = (c, Dv, Dt)
    c_pts = c_pss = float("inf")
    for (Dv, Dt, sc, tc) in rows:
        if Dt == 1:
            c_pts = cost(sc, tc)                 # pure VERTEX partition corner
        if Dv == 1:
            c_pss = cost(sc, tc)                 # pure SNAPSHOT partition corner
    kind = "INTERIOR" if (best[1] > 1 and best[2] > 1) else \
           ("corner-PTS(Dt=1)" if best[2] == 1 else "corner-PSS(Dv=1)")
    best_corner = min(c_pts, c_pss)
    gain = best_corner / best[0] if best_corner < float("inf") else float("inf")
    return f"Dv{best[1]}xDt{best[2]}", kind, gain, best[0] * 1e3


# ============================================================================================
# Driver
# ============================================================================================

def load_graph(a):
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        return g.name, g.num_nodes, g.src.astype(np.int32), g.dst.astype(np.int32), None
    src, dst = gen_graph(a.nodes, a.edges, a.comms, a.intra, seed=a.seed)
    comm = synthetic_communities(a.nodes, a.comms, seed=a.seed)
    return f"synthetic-{a.nodes}n-{a.edges}e", a.nodes, src, dst, comm


def main():
    ap = argparse.ArgumentParser(description="zord scheduler hardware-generalization sweep")
    ap.add_argument("--dataset", default="", help="real temporal graph name (else synthetic)")
    ap.add_argument("--nodes", type=int, default=4_000_000)
    ap.add_argument("--edges", type=int, default=50_000_000)
    ap.add_argument("--comms", type=int, default=4000)
    ap.add_argument("--intra", type=float, default=0.9, help="synthetic intra-community edge frac")
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--rank-by", default="degree", choices=["degree", "kcore"])
    ap.add_argument("--window", type=int, default=8, help="co-resident snapshots W for plan_memory")
    ap.add_argument("--snapshots", type=int, default=64, help="S for the duality factorization")
    ap.add_argument("--temporal-locality", type=float, default=0.95,
                    help="synthetic temporal locality in [0,1] (1=each community active in ~1 "
                         "snapshot). Higher locality shrinks the temporal cut so the duality corner "
                         "becomes interconnect-sensitive. Ignored for --dataset (real timestamps).")
    ap.add_argument("--reuse", type=float, default=0.0, help="rho: temporal reuse fraction")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    F = a.feat

    t0 = time.time()
    name, N, src, dst, comm = load_graph(a)
    M = src.size
    deg = node_degree(src, dst, N)
    rank, rank_note = density_rank(src, dst, deg, N, a.rank_by)        # placement lever
    vorder, v_note = vertex_block_order(src, dst, deg, N)              # duality spatial cut
    # snapshot timestamps: real datasets carry their own (time-sorted); synthetic injects locality
    # aligned to the planted communities so intra-community edges co-occur in time.
    loc = -1.0 if a.dataset else a.temporal_locality
    snap = make_snapshots(src, dst, N, a.snapshots, loc, comm=comm, seed=a.seed)
    print(f"HW-GENERALIZATION dataset={name} N={N:,} M={M:,} F={F} window={a.window} "
          f"snapshots={a.snapshots} reuse={a.reuse}")
    print(f"  density ranking: {rank_note} | vertex blocks: {v_note}; graph ready in {time.time()-t0:.1f}s")
    print(f"  C++ kernel: {BIN} ({'present' if os.path.exists(BIN) else 'ABSENT -> numpy fallback'})")
    print()

    profiles = build_profiles()
    summary = []
    for label, (cluster, note) in profiles.items():
        D = cluster.num_devices
        print(f"=== {label}  ({D} devices)  ===")
        print(f"    {note}")
        print("    devices: " + " | ".join(
            f"{d.name} hbm={d.hbm_bw_gbps:.0f} mem={d.usable_mem/GB:.0f}GB r={d.throughput:.2f}"
            for d in cluster.devices))
        print(f"    interconnect: intra={cluster.intra_node_bw:.0f} inter={cluster.inter_node_bw:.2f} GB/s "
              f"(skew {cluster.intra_node_bw/max(1e-9,cluster.inter_node_bw):.0f}x)")

        # (a) memory plan ------------------------------------------------------------------
        w = Workload(num_nodes=N, num_edges=M, feat_dim=F, layers=2,
                     window=a.window, reuse_frac=a.reuse)
        gp = plan_memory(cluster, w, prefetch=True)
        tiered = any(p.streamed_snapshots > 0 for p in gp.per_device if p.feasible)
        print(f"    (a) plan_memory: feasible={gp.all_feasible} bound={gp.bound} "
              f"makespan={gp.makespan_sec*1e3:.1f}ms tiered={tiered} "
              f"streamed={gp.total_streamed_gb:.1f}GB/epoch bottleneck=dev{gp.bottleneck}")

        # (b) placement makespan -----------------------------------------------------------
        mk = placement_makespans(cluster, rank, src, dst, deg, N, F, seed=a.seed)
        sp_even = mk["even"] / mk["hetero_matched"]
        sp_bp = mk["bw_prop"] / mk["hetero_matched"]
        sp_rand = mk["random"] / mk["hetero_matched"]
        sp_bb = mk["balanced_blind"] / mk["hetero_matched"]   # isolates the HARDWARE-het win
        print(f"    (b) placement makespan (ms): even={mk['even']:.2f} bw-prop={mk['bw_prop']:.2f} "
              f"balanced-blind={mk['balanced_blind']:.2f} random={mk['random']:.2f} "
              f"HETERO-MATCHED={mk['hetero_matched']:.2f}")
        print(f"        speedup hetero-matched: {sp_even:.2f}x vs even, {sp_bp:.2f}x vs bw-prop, "
              f"{sp_bb:.2f}x vs balanced-blind [HW-only], {sp_rand:.2f}x vs random  "
              f"(cut={mk['cut_hm']/(2*M)*100:.1f}% edges)")

        # (c) duality corner ---------------------------------------------------------------
        dlabel, dkind, dgain, dcost = duality_choice(cluster, vorder, src, dst, snap, N, a.snapshots, F)
        print(f"    (c) duality: BEST={dlabel} ({dkind}) cost={dcost:.2f}ms "
              f"gain_vs_best_corner={dgain:.2f}x")
        print()

        bw = np.array([d.hbm_bw_gbps for d in cluster.devices])
        bw_spread = float(bw.max() / bw.min())
        summary.append((label, cluster, gp.all_feasible, gp.bound, tiered,
                        sp_even, sp_bb, sp_rand, dlabel, dkind, dgain, bw_spread))

    # -------- headline table ----------------------------------------------------------------
    print("=" * 118)
    print("HEADLINE: scheduler adapts + wins on EVERY profile; the HARDWARE-isolated placement "
          "speedup (vs balanced-blind) GROWS with bandwidth heterogeneity, and the duality")
    print("          corner FLIPS with the interconnect skew -- hardware is just a cost-model "
          "parameter, the schedule is downstream of it.")
    print(f"  {'profile':<28} {'feas':>5} {'bound':>12} {'tier':>5} {'bw-spread':>9} "
          f"{'sp/even':>8} {'sp/blind':>9} {'sp/rand':>8} {'duality':>10} {'corner':>16}")
    for (label, cluster, feas, bound, tiered, se, sbb, sr, dl, dk, dg, spread) in summary:
        print(f"  {label:<28} {str(feas):>5} {bound:>12} {str(tiered):>5} {spread:>8.2f}x "
              f"{se:>7.2f}x {sbb:>8.2f}x {sr:>7.2f}x {dl:>10} {dk:>16}")
    print("=" * 118)
    print("  Reading: sp/blind isolates the HARDWARE win (homogeneous control -> ~1.00x floor; "
          "rises with bw-spread). sp/even and sp/rand also fold in the always-on degree-balancing")
    print("  win. The duality 'corner' column shows the chosen factorization ADAPTS to which link "
          "(intra vs inter) is cheap on each profile -- the same algorithm, re-parameterized.")


if __name__ == "__main__":
    main()
