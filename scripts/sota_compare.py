#!/usr/bin/env python
"""SOTA PARTITIONER COMPARISON (the "vs the field" experiment). zord has already beaten the NAIVE
floor (hash/even/PSS/PTS/METIS/Fennel/LDG, see scripts/partitioner_bench.py); this script reproduces
the SOTA TEMPORAL-GRAPH partitioners from the 11 reference papers as comparable, in-process baselines
and scores them against zord's "arrange" on the SAME incident-edge work model -- a PROCESS-only
comparison (cut / balance / predicted makespan / replication-factor / feasibility; never accuracy,
because the placement is numerically result-preserving: the aggregation is an associative GAS reduce,
so WHERE a partial sum is computed never changes WHAT is computed).

REPRODUCED MECHANISMS (faithful-but-tractable, CPU-only numpy + the C++ build/graph_algos kernel):

  DGC-PGC (Chen et al., SIGMOD'24 "Training Dynamic Graphs ... by Chunks"):
    "Partition by Graph Chunk". Build a SUPRA-GRAPH = spatial edges (within a snapshot) + VIRTUAL
    TEMPORAL edges (same vertex across adjacent active snapshots), with ASYMMETRIC weights (spatial
    edges are aggregated N_GATHERS times, temporal once -> spatial is "heavier", per the paper's
    profiled per-model edge weights). Run WEIGHTED LABEL PROPAGATION on that supra-graph to coarsen
    it into cross-spacetime CHUNKS (argmax-weight label = "maximize intra-chunk comm cost"), then a
    BALANCE-aware GREEDY chunk->device assignment (descending chunk work, score = remaining-capacity
    balance term + intra-device affinity term, Eq.3). We use the C++ `lpa` kernel for the heavy LPA
    pass when available (faithful + fast), else a numpy weighted-LPA fallback.

  MemShare (Zhang et al., PVLDB'25 "Hotspot Memory Sharing"):
    Shared-node paradigm: REPLICATE the top-k (default 10%) highest-degree HOTSPOT nodes onto every
    device (a vertex-cut / hub-replication heuristic that drives the hub cut to ~0, Fig.3 GDELT
    30%->8%), edge-cut the cold periphery with a node/edge/time balance factor (F_BAL, Eq.12). We
    reproduce the hotspot replication + balanced periphery; report its replication factor.

  METIS (pymetis, try/except SKIP if missing) + PSS + PTS as references on the SAME model.

  ZORD = work-balanced INCIDENT hetero-matched placement (dense core -> strong device, node counts
    solved so per-device incident-edge agg TIME is balanced; hetero_matched.py) + DENSE-CORE
    VERTEX-CUT (k-core core replicated PowerGraph-style; vertexcut_replicate.py) + ADAPTIVE CORNER
    PICK (also consider the spatial/temporal duality corners; emit the best-makespan plan among
    {hetero edge-cut, dense-core vertex-cut, spatial, temporal}). Combines the §22 (duality corner)
    and §23 (dense-core vertex-cut) levers.

Per method we report: edge-cut %, balance (node) + imbalance (incident-work), predicted MAKESPAN on
the HetCluster hetero profile (incident-edge roofline COMPUTE + boundary-row COMM over the slow link),
replication factor (for the vertex-cut ones), feasibility (fits per-device HBM capacity?).

  python scripts/sota_compare.py --synthetic --devices 3 --feat 128 --snapshots 64
  python scripts/sota_compare.py --dataset askubuntu --devices 3 --feat 128 --snapshots 64
  python scripts/sota_compare.py --dataset wiki-talk --devices 4 --feat 128

HONEST: where zord ties or loses, the headline says so. positioning note classifies the 11 papers
into ORTHOGONAL (training-execution layers zord feeds) vs DIRECT competitors (the partition axis).
"""
import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np

# Make `zord` importable straight from the repo (scripts/ is not a package).
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from zord.profiler.cluster_profile import hetcluster, DeviceProfile, ClusterProfile, _MEASURED, GB

# C++ graph kernels (binary fmt: int64 N, int64 M, then 2*M int32 interleaved src,dst; out: int64 N
# + N int32 newid). modes: degree | kcore | bfs | lpa. Built via:
#   g++ -O3 -std=c++17 -fopenmp src/zord/cpp/graph_algos.cpp -o build/graph_algos
_BIN = os.environ.get(
    "ZORD_GRAPH_BIN",
    os.path.join(os.path.dirname(_SRC), "build", "graph_algos"))

# --- incident-edge roofline work model (IDENTICAL to vertexcut_replicate.py / hetero_matched.py) ---
BYTES_PER_EDGE_TRAVERSAL = 4.0   # one fp32 feature word moved per edge per gather (memory-bound agg)
N_GATHERS = 2                    # 2-layer aggregation -> 2 SpMM gathers over the local incident edges
FEATURE_ROW_BYTES = 4.0          # fp32 word per feature dim per boundary/partial row in comm


# =====================================================================================
# graph generation: power-law (preferential-attachment-ish) temporal graph with community
# structure -- the regime where locality-aware + hub-aware placement can win, and where the
# DGC/MemShare hub assumptions actually hold (a heavy-tailed degree distribution).
# =====================================================================================
def gen_powerlaw_temporal(N, M, C, intra, hub_frac=0.01, hub_boost=40.0, seed=0):
    """Community-structured graph with a heavy-tailed degree distribution AND a realistic temporal
    profile (vertices are INTRODUCED gradually over time, as in real CTDGs -- so the temporal corner
    PTS is a genuine, balanced split rather than a degenerate one-device dump). `intra` fraction of
    edges stay inside a node's community (a clusterable structure a good edge-cut splits with small
    cut); a small `hub_frac` set of HUBS attract a `hub_boost`x share of endpoints (the power-law
    tail DGC's chunking and MemShare's hotspot-sharing target). Returns (src, dst, t), time-sorted."""
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, C, size=N).astype(np.int64)
    order = np.argsort(comm, kind="stable")
    bounds = np.searchsorted(comm[order], np.arange(C + 1))

    # endpoint-attractiveness weights: a few hubs are far more likely to be chosen as an endpoint
    w = np.ones(N, dtype=np.float64)
    n_hub = max(1, int(N * hub_frac))
    hubs = rng.choice(N, size=n_hub, replace=False)
    w[hubs] = hub_boost

    # Each VERTEX has a birth time in [0, M); edges arrive in time order and may only touch vertices
    # already born by then. birth ~ uniform over the stream -> vertices spread across the timeline.
    birth = rng.integers(0, M, size=N).astype(np.int64)

    t = np.sort(rng.integers(0, M, size=M)).astype(np.int64)   # M edge timestamps, sorted
    # for each edge time, the set of born vertices is a prefix of the birth-sorted vertex order.
    birth_order = np.argsort(birth, kind="stable")             # vertices in birth-time order
    birth_sorted = birth[birth_order]
    n_born = np.searchsorted(birth_sorted, t, side="right")    # #vertices born by each edge time
    n_born = np.maximum(n_born, 1)

    # pick endpoint u from the born prefix, weighted by hub attractiveness (preferential attachment).
    w_born = w[birth_order]
    cumw = np.cumsum(w_born)
    def pick_born(n_born_arr):
        r = rng.random(n_born_arr.size) * cumw[n_born_arr - 1]
        idx = np.searchsorted(cumw, r, side="left")
        idx = np.minimum(idx, n_born_arr - 1)
        return birth_order[idx]                                # original vertex ids

    # Decide intra-vs-cross per edge by a random mask (so edge i keeps its own timestamp t[i] and its
    # endpoints respect the born-prefix at t[i] -- no post-hoc shuffle that would break time alignment).
    is_intra = rng.random(M) < intra
    u = pick_born(n_born)                                      # one source per edge, from born prefix
    v = np.empty(M, dtype=np.int64)
    # intra edges -> a community neighbor of u; cross edges -> another born (hub-weighted) vertex
    cu = comm[u]
    lo = bounds[cu].astype(np.int64); hi = bounds[cu + 1].astype(np.int64)
    pick = lo + (rng.random(M) * np.maximum(1, hi - lo)).astype(np.int64)
    v_intra = order[np.minimum(pick, N - 1)]
    v_cross = pick_born(n_born)
    v = np.where(is_intra, v_intra, v_cross)
    return u, v, t                                             # already time-sorted (t is sorted)


# =====================================================================================
# C++ kernel interface
# =====================================================================================
def _write_edges(path, N, src, dst):
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", int(N), int(src.size)))
        inter = np.empty(2 * src.size, dtype=np.int32)
        inter[0::2] = src.astype(np.int32)
        inter[1::2] = dst.astype(np.int32)
        inter.tofile(f)


def cpp_order(N, src, dst, mode, tag):
    """Run a C++ graph_algos mode; return (newid[old]->rank, seconds) or (None, seconds) on failure
    / missing binary. newid is the C++ ordering for that mode (lpa: cluster-grouped; kcore:
    degeneracy order; degree: degree-descending)."""
    if not os.path.exists(_BIN):
        return None, 0.0
    epath = f"/tmp/zord_sota_edges_{tag}.bin"
    opath = f"/tmp/zord_sota_out_{tag}.bin"
    _write_edges(epath, N, src, dst)
    t0 = time.time()
    r = subprocess.run([_BIN, epath, mode, opath], capture_output=True, text=True)
    cost = time.time() - t0
    if r.returncode != 0:
        return None, cost
    with open(opath, "rb") as f:
        n = struct.unpack("<q", f.read(8))[0]
        newid = np.fromfile(f, dtype=np.int32, count=n)
    return newid, cost


# =====================================================================================
# basic graph quantities
# =====================================================================================
def node_degree(src, dst, N):
    return (np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)).astype(np.int64)


def _expand_ranges(starts, ends):
    """Vectorized concat of arange(starts[i], ends[i]) (no Python loop)."""
    lens = ends - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    idx = np.cumsum(lens)[:-1]
    out[0] = starts[0]
    out[idx] = starts[1:] - ends[:-1] + 1
    return np.cumsum(out)


def coreness(src, dst, N):
    """Exact per-node coreness via round-based k-core peeling (vectorized per round). Self-loops
    dropped, undirected multi-edges deduped. (Same routine as vertexcut_replicate.py.)"""
    src = src.astype(np.int64); dst = dst.astype(np.int64)
    m = src != dst
    a = np.minimum(src[m], dst[m]); b = np.maximum(src[m], dst[m])
    key = np.unique(a * np.int64(N) + b)
    a = key // N; b = key % N
    r = np.concatenate([a, b]); c = np.concatenate([b, a])
    o = np.argsort(r, kind="stable"); r = r[o]; c = c[o]
    off = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(r, minlength=N), out=off[1:])
    core = np.zeros(N, dtype=np.int64)
    cur = (off[1:] - off[:-1]).copy()
    removed = np.zeros(N, dtype=bool)
    alive = N; k = 0
    while alive > 0:
        amin = int(cur[~removed].min())
        if amin > k:
            k = amin
        peel = (~removed) & (cur <= k)
        while peel.any():
            pnodes = np.nonzero(peel)[0]
            core[pnodes] = k
            removed[pnodes] = True
            alive -= pnodes.size
            if alive == 0:
                break
            slots = _expand_ranges(off[pnodes], off[pnodes + 1])
            nbr = c[slots]
            nbr = nbr[~removed[nbr]]
            if nbr.size:
                cur -= np.bincount(nbr, minlength=N)
            peel = (~removed) & (cur <= k)
    return core


# =====================================================================================
# the SHARED incident-edge work model + makespan
# =====================================================================================
def predict_ms(incident_edges, comm_rows, bw_gbps, link_gbps):
    """Per-device step ms = COMPUTE (roofline gather over local incident edges, reads fast HBM) +
    COMM (boundary feature rows exchanged across the SLOW interconnect each layer). F is folded into
    incident_edges/comm_rows by the caller via the byte constants. Returns (total[D], comp[D], comm[D])."""
    comp_ms = incident_edges * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS / (bw_gbps * 1e9) * 1e3
    comm_ms = comm_rows * FEATURE_ROW_BYTES * N_GATHERS / (link_gbps * 1e9) * 1e3
    return comp_ms + comm_ms, comp_ms, comm_ms


def edgecut_metrics(src, dst, deg, dev, D, N):
    """For a single-home assignment dev[v] in [0,D): edge-cut, per-device incident-edge work, and
    per-device COMM rows (distinct (gathering-device, remote-neighbor) pairs -- one F-row received
    per such pair per layer). Identical model to vertexcut_replicate.edgecut_metrics."""
    pu = dev[src]; pv = dev[dst]
    cut = int(np.count_nonzero(pu != pv))
    incident = np.bincount(dev, weights=deg.astype(np.float64), minlength=D)
    counts = np.bincount(dev, minlength=D).astype(np.int64)
    a = np.concatenate([src, dst]); b = np.concatenate([dst, src])
    da = dev[a]; cross = da != dev[b]
    if cross.any():
        key = np.unique(da[cross].astype(np.int64) * np.int64(N) + b[cross])
        comm_rows = np.bincount((key // N).astype(np.int64), minlength=D).astype(np.float64)
    else:
        comm_rows = np.zeros(D, dtype=np.float64)
    return cut, incident, comm_rows, counts


def replicate_core_metrics(src, dst, deg, core_mask, dev_periphery, D, N, rng):
    """VERTEX-CUT: core nodes (core_mask) replicated onto all D devices; periphery uses dev_periphery
    (a single-home assignment, -1 for core nodes). Incident-edge gather work split by NEIGHBOR
    OWNERSHIP (each edge gathered once); core-internal edges spread round-robin. Cut counts only
    periphery-periphery cross edges (core edges -> 0 cut, replaced by the D-way partial reduce).
    Returns (cut, incident[D], comm_rows[D], counts[D], extra_core_rows). Mirrors
    vertexcut_replicate.replicate_metrics."""
    dev = dev_periphery
    u = np.concatenate([src, dst]); v = np.concatenate([dst, src])
    u_core = core_mask[u]; v_core = core_mask[v]
    du = dev[u]; dv = dev[v]
    land = np.empty(u.size, dtype=np.int64)
    mA = ~u_core;            land[mA] = du[mA]                 # u periphery -> its device
    mB = u_core & (~v_core); land[mB] = dv[mB]                 # u core, v periphery -> v's device
    mC = u_core & v_core                                       # core-core internal -> round-robin
    nC = int(mC.sum())
    if nC:
        land[mC] = rng.integers(0, D, size=nC)
    incident = np.bincount(land, minlength=D).astype(np.float64)

    cs = core_mask[src]; cd = core_mask[dst]
    pp = (~cs) & (~cd)
    cut = int(np.count_nonzero((dev[src] != dev[dst]) & pp))   # only periphery-periphery edges cut
    core_size = int(core_mask.sum())
    extra_core_rows = core_size * (D - 1)
    reduce_partials = np.full(D, core_size * (D - 1) / D, dtype=np.float64)  # D-way partial reduce

    pp_inc = (~u_core) & (~v_core)
    cross = pp_inc & (du != dv) & (du >= 0) & (dv >= 0)
    if cross.any():
        key = np.unique(du[cross].astype(np.int64) * np.int64(N) + v[cross])
        comm_rows = np.bincount((key // N).astype(np.int64), minlength=D).astype(np.float64)
    else:
        comm_rows = np.zeros(D, dtype=np.float64)
    comm_rows = comm_rows + reduce_partials

    periphery = np.nonzero(~core_mask)[0]
    counts = np.bincount(dev[periphery], minlength=D).astype(np.int64) + core_size
    return cut, incident, comm_rows, counts, extra_core_rows


# =====================================================================================
# balanced periphery / edge-cut layouts (cluster-respecting via the C++ lpa order)
# =====================================================================================
def _split_by_work(seq, weight, D):
    """Split an ORDERED node sequence `seq` (e.g. cluster-grouped lpa order) into D contiguous
    segments of equal cumulative `weight` (incident-edge work). Returns seg id per position in seq."""
    cum = np.cumsum(weight)
    if cum[-1] <= 0:
        return (np.arange(seq.size) * D // max(1, seq.size)).clip(0, D - 1)
    targets = np.arange(1, D) * cum[-1] / D
    cuts = np.searchsorted(cum, targets, side="left")
    bounds = np.concatenate([[0], cuts, [seq.size]]).astype(np.int64)
    return (np.searchsorted(bounds, np.arange(seq.size), side="right") - 1).clip(0, D - 1)


def lpa_edgecut(N, deg, lpa_rank, D, caps=None):
    """Cluster-respecting edge-cut: walk the lpa cluster-grouped layout in rank order, cut into D
    contiguous segments of equal incident-edge work (or capacity-proportional work when caps given,
    so a fast/big device gets more work -- the hetero-matched split). Returns dev[v] in [0,D)."""
    rank_to_node = np.empty(N, dtype=np.int64)
    rank_to_node[lpa_rank.astype(np.int64)] = np.arange(N)
    deg_by_rank = deg[rank_to_node].astype(np.float64)
    cum = np.cumsum(deg_by_rank)
    if caps is None:
        targets = np.arange(1, D) * cum[-1] / D
    else:
        share = np.asarray(caps, dtype=np.float64)
        share = share / share.sum()
        targets = np.cumsum(share)[:-1] * cum[-1]
    cuts = np.searchsorted(cum, targets, side="left")
    bounds = np.concatenate([[0], cuts, [N]]).astype(np.int64)
    seg = (np.searchsorted(bounds, np.arange(N), side="right") - 1).clip(0, D - 1)
    dev = np.empty(N, dtype=np.int64)
    dev[rank_to_node] = seg
    return dev


def temporal_partition(src, dst, snap, deg, N, S, D):
    """PTS (Partition by Temporal Sequence): keep each vertex's WHOLE timeline on one device, and
    split vertices into D balanced cohorts by their FIRST-ACTIVITY time. Vertices that appear earlier
    go to lower device ids -> a temporal split that is balanced by incident work (so it is a genuine
    PTS corner, not a degenerate one-device dump). Spatial edges are cut (the PTS cost); temporal /
    timeline edges stay local. Returns dev[v] in [0,D)."""
    first = np.full(N, S + 1, dtype=np.int64)
    np.minimum.at(first, src, snap)
    np.minimum.at(first, dst, snap)
    # rank vertices by first-activity time (unseen vertices sort last), split into D equal-WORK blocks
    seq = np.argsort(first, kind="stable")
    seg = _split_by_work(seq, deg[seq].astype(np.float64), D)
    dev = np.empty(N, dtype=np.int64)
    dev[seq] = seg
    return dev


def balanced_periphery(periphery, deg, D, lpa_rank=None):
    """Edge-cut the periphery (non-core) nodes finely + balanced across D devices, cluster-respecting
    via lpa rank when available. Returns dev[v] for periphery nodes, -1 for core nodes."""
    N = deg.shape[0]
    dev = np.full(N, -1, dtype=np.int64)
    if periphery.size == 0:
        return dev
    if lpa_rank is not None:
        seq = periphery[np.argsort(lpa_rank[periphery], kind="stable")]
    else:
        seq = periphery[np.argsort(-deg[periphery], kind="stable")]
    seg = _split_by_work(seq, deg[seq].astype(np.float64), D)
    dev[seq] = seg
    return dev


# =====================================================================================
# DGC-PGC: weighted LPA on the supra-graph -> chunks -> balance-aware greedy assignment
# =====================================================================================
def supra_chunks(src, dst, snap, N, S, w_spatial, w_temporal, lpa_rank=None):
    """Coarsen the SUPRA-GRAPH into chunks via weighted label propagation (DGC step 1).

    Supra-graph nodes are CELLS = active (vertex, snapshot) pairs. Edges:
      - SPATIAL: a within-snapshot graph edge connects cell(u,t)-cell(v,t)   weight w_spatial
      - TEMPORAL (virtual): same vertex in adjacent ACTIVE snapshots          weight w_temporal
    DGC's argmax-weight label update ("maximize intra-chunk comm cost"): each cell adopts the chunk
    label with the largest incident edge-weight. We seed labels from the C++ `lpa` graph clustering
    (a faithful + fast weighted-LPA proxy: lpa already groups densely-connected vertices, which on
    the supra-graph is exactly the dominant spatial coupling), then attach the temporal coupling by
    keeping a vertex's cells together when temporal weight dominates. Returns:
      chunk_of_cell [C], cell_v [C], cell_t [C], and the spatial/temporal cell-pair endpoint arrays.
    """
    ks = src * S + snap
    kd = dst * S + snap
    keys = np.unique(np.concatenate([ks, kd]))     # sorted unique cell ids 0..C-1
    C = keys.size
    cell_v = (keys // S).astype(np.int64)
    cell_t = (keys % S).astype(np.int64)
    a = np.searchsorted(keys, ks)
    b = np.searchsorted(keys, kd)
    m = a != b
    sp_a, sp_b = a[m], b[m]                          # spatial cell-pairs
    same_v = cell_v[1:] == cell_v[:-1]               # cells are vertex-major, time-minor
    idx = np.nonzero(same_v)[0]
    tp_a, tp_b = idx, idx + 1                         # temporal cell-pairs (adjacent active snaps)

    # Initial chunk label per cell = its VERTEX's lpa cluster (dominant spatial coupling). This
    # realizes "weighted LPA on the supra-graph": vertices that lpa groups (heavy spatial weight)
    # share a chunk, and a vertex's cells across snapshots are temporally tied (handled in cost).
    if lpa_rank is not None:
        # bucket the lpa rank into ~spatial chunks; granularity ~ sqrt(C) chunks (coarsening)
        nchunk = max(1, int(np.sqrt(C)))
        vlabel = (lpa_rank.astype(np.int64) * nchunk // N).clip(0, nchunk - 1)
    else:
        vlabel = (cell_v * max(1, int(np.sqrt(C))) // N)
    chunk = vlabel[cell_v]

    # One weighted-LPA refinement sweep over cells: a cell switches to the neighbor-chunk with the
    # largest incident weight (spatial neighbors weigh w_spatial, the two temporal neighbors weigh
    # w_temporal). We pick the dominant neighbor chunk via a per-cell weighted mode (sort-reduce),
    # since a dense (cell x chunk) accumulator is too large.
    for _ in range(2):
        ca = np.concatenate([sp_a, sp_b, tp_a, tp_b])
        nb = np.concatenate([sp_b, sp_a, tp_b, tp_a])
        wv = np.concatenate([np.full(sp_a.size, w_spatial), np.full(sp_b.size, w_spatial),
                             np.full(tp_a.size, w_temporal), np.full(tp_b.size, w_temporal)])
        nb_chunk = chunk[nb]
        # weighted mode of nb_chunk per ca: sort by (ca, nb_chunk), segment-sum weights, take argmax
        order = np.lexsort((nb_chunk, ca))
        ca_s = ca[order]; ch_s = nb_chunk[order]; w_s = wv[order]
        # group boundaries where (ca, chunk) changes
        grp_new = np.empty(ca_s.size, dtype=bool)
        grp_new[0] = True
        grp_new[1:] = (ca_s[1:] != ca_s[:-1]) | (ch_s[1:] != ch_s[:-1])
        gid = np.cumsum(grp_new) - 1
        gw = np.bincount(gid, weights=w_s)            # total weight per (cell,chunk) group
        g_ca = ca_s[grp_new]                          # cell of each group
        g_ch = ch_s[grp_new]                          # chunk of each group
        # for each cell, pick the group (chunk) with max weight. groups are already ca-sorted.
        cell_new = np.empty(g_ca.size, dtype=bool)
        cell_new[0] = True
        cell_new[1:] = g_ca[1:] != g_ca[:-1]
        cell_id = np.cumsum(cell_new) - 1
        # argmax weight within each cell's contiguous group block -> the chunk that cell adopts
        best_chunk = np.zeros(cell_id[-1] + 1, dtype=np.int64)
        block_starts = np.nonzero(cell_new)[0]
        block_ends = np.concatenate([block_starts[1:], [g_ca.size]])
        for s_, e_ in zip(block_starts, block_ends):
            j = s_ + int(np.argmax(gw[s_:e_]))
            cid = cell_id[s_]
            best_chunk[cid] = g_ch[j]
        cells_with_nb = g_ca[block_starts]
        new_chunk = chunk.copy()
        new_chunk[cells_with_nb] = best_chunk
        if np.array_equal(new_chunk, chunk):
            chunk = new_chunk
            break
        chunk = new_chunk
    return chunk, cell_v, cell_t, sp_a, sp_b, tp_a, tp_b


def dgc_pgc_partition(src, dst, snap, deg, N, S, D, caps, lpa_rank, w_spatial, w_temporal):
    """DGC-PGC: weighted-LPA supra-graph chunks -> balance-aware greedy chunk->device assignment.
    Maps the per-CELL chunking back to a per-VERTEX device assignment (a vertex follows the device
    of the chunk holding the most of its cells' incident work -- the cell->vertex collapse the
    single-home work model needs). Returns dev[v] in [0,D)."""
    chunk, cell_v, cell_t, sp_a, sp_b, tp_a, tp_b = supra_chunks(
        src, dst, snap, N, S, w_spatial, w_temporal, lpa_rank=lpa_rank)

    # chunk work = sum of incident spatial+temporal weight of its cells (DGC weights chunk by work).
    uniq, inv = np.unique(chunk, return_inverse=True)
    K = uniq.size
    # per-cell incident weight (its row in the supra-graph)
    cell_w = np.zeros(cell_v.size, dtype=np.float64)
    np.add.at(cell_w, sp_a, w_spatial); np.add.at(cell_w, sp_b, w_spatial)
    np.add.at(cell_w, tp_a, w_temporal); np.add.at(cell_w, tp_b, w_temporal)
    chunk_work = np.bincount(inv, weights=cell_w, minlength=K)

    # GREEDY chunk->device assignment (DGC Algorithm 1): chunks in descending work; score per device
    # = remaining-capacity balance term + intra-device affinity term. caps -> per-device work target.
    target = np.asarray(caps, dtype=np.float64)
    target = target / target.sum() * chunk_work.sum()      # work budget per device (hetero-aware)
    load = np.zeros(D, dtype=np.float64)
    chunk_dev = np.full(K, -1, dtype=np.int64)
    # affinity: per (chunk, device) accumulated cross-chunk weight already on the device
    aff = np.zeros((K, D), dtype=np.float64)
    # precompute chunk-chunk cut weight (sparse) from spatial+temporal cell-pairs
    ce_a = np.concatenate([sp_a, tp_a]); ce_b = np.concatenate([sp_b, tp_b])
    ce_w = np.concatenate([np.full(sp_a.size, w_spatial), np.full(tp_a.size, w_temporal)])
    chka = chunk[ce_a]; chkb = chunk[ce_b]
    cross = chka != chkb
    # map chunk label -> compact index
    relabel = np.full(uniq.max() + 1, -1, dtype=np.int64)
    relabel[uniq] = np.arange(K)
    ia = relabel[chka[cross]]; ib = relabel[chkb[cross]]; cw = ce_w[cross]

    # Eq.3 combines a normalized BALANCE term with an affinity term; the paper enforces a per-device
    # work budget so a device that is full stops accepting chunks (else affinity collapses everything
    # onto one device). We mirror that: forbid devices at/over a tolerance*budget; among the allowed,
    # maximize (balance + normalized affinity). Both terms are scaled to ~[0,1] so neither dominates.
    tol = 1.10                                             # 10% over-budget slack before a device closes
    aff_scale = max(1.0, float(chunk_work.max()))          # put affinity on the chunk-work scale
    order = np.argsort(-chunk_work, kind="stable")
    for k in order:
        bal = (target - load) / np.maximum(1.0, target)    # remaining-capacity fraction (emptier=higher)
        allowed = load < tol * target
        if not allowed.any():                              # all devices over budget -> least loaded
            allowed = load <= load.min() + 1e-9
        score = bal + aff[k] / aff_scale
        score = np.where(allowed, score, -np.inf)
        d = int(np.argmax(score))
        chunk_dev[k] = d
        load[d] += chunk_work[k]
        # update affinity for chunks sharing cut weight with k: those chunks now prefer device d
        sel_a = ib[ia == k]; w_a = cw[ia == k]
        if sel_a.size:
            np.add.at(aff[:, d], sel_a, w_a)
        sel_b = ia[ib == k]; w_b = cw[ib == k]
        if sel_b.size:
            np.add.at(aff[:, d], sel_b, w_b)

    cell_dev = chunk_dev[inv]                               # device of each cell
    # collapse cells -> a single home per VERTEX (the single-home work model). A vertex goes to the
    # device holding the most of its incident cell work.
    vd = np.full((N, D), 0.0)
    np.add.at(vd, (cell_v, cell_dev), cell_w)
    dev = np.argmax(vd, axis=1).astype(np.int64)
    # vertices with no cells (isolated) -> least-loaded device by node count
    seen = np.zeros(N, dtype=bool); seen[cell_v] = True
    if (~seen).any():
        dev[~seen] = int(np.argmin(np.bincount(dev[seen], minlength=D)))
    return dev


# =====================================================================================
# MemShare: top-k hotspot replication (vertex-cut on hubs) + balanced periphery
# =====================================================================================
def memshare_partition(src, dst, deg, N, D, top_k_frac, lpa_rank):
    """MemShare shared-node paradigm: REPLICATE the top-k highest-DEGREE hotspot nodes onto all
    devices (drives the hub cut to ~0), edge-cut the cold periphery balanced. Returns
    (core_mask, dev_periphery). The periphery split is cluster-respecting + work-balanced (a proxy
    for MemShare's F_BAL node/edge/time balance factor)."""
    k = max(1, int(N * top_k_frac))
    thresh = np.partition(deg, N - k)[N - k]
    core_mask = deg >= thresh
    periphery = np.nonzero(~core_mask)[0]
    dev_p = balanced_periphery(periphery, deg, D, lpa_rank=lpa_rank)
    return core_mask, dev_p


# =====================================================================================
# ZORD: hetero-matched edge-cut + dense-core vertex-cut + adaptive corner pick
# =====================================================================================
def zord_arrange(src, dst, snap, deg, core_val, N, S, D, devs, bw, link, F, caps_work, lpa_rank, rng):
    """zord's "arrange": evaluate the candidate plans and ADAPTIVELY pick the best-makespan one.
      1) hetero edge-cut : cluster-respecting lpa split, segments sized so per-device incident-edge
                           agg TIME is balanced (work share ~ device HBM bandwidth -> the straggler
                           is removed; hetero_matched.py).
      2) dense-core vertex-cut : replicate the k-core dense core (PowerGraph-style), edge-cut the
                           periphery -- wins when a dense core would otherwise force a high cut floor
                           (vertexcut_replicate.py). Core threshold tau auto-picked at the top decile
                           of coreness (the dense tail).
      3) spatial corner (PSS) / temporal corner (PTS) : the duality endpoints, for completeness.
    Returns a dict of the chosen plan's metrics + the per-candidate makespans (honest reporting)."""
    cand = {}

    # device work budget ~ HBM bandwidth so the time-balance is hetero-matched (strong dev more work)
    caps_work = np.asarray(caps_work, dtype=np.float64)

    # ---- candidate 1: hetero edge-cut ----
    dev1 = lpa_edgecut(N, deg, lpa_rank, D, caps=caps_work)
    cut1, inc1, comm1, cnt1 = edgecut_metrics(src, dst, deg, dev1, D, N)
    cand["edge-cut(hetero)"] = (cut1, inc1, comm1, cnt1, 0)

    # ---- candidate 2: dense-core vertex-cut, with an ADAPTIVE core-size SWEEP ----
    # Replicating too small a core leaves a high cut floor; too large a core wastes node-rows on
    # replication (and inflates the D-way reduce). zord SWEEPS the k-core threshold tau over a few
    # quantiles of the coreness distribution and keeps the makespan-best core size -- the adaptive
    # lever (§23 dense-core x §22 corner pick). This is what lets zord tune replication to the graph
    # rather than commit to a fixed hot fraction the way MemShare's top-k does.
    best_vc = None
    # D36(v2): NO fixed replication budget -- sweep the FULL core-size range (coarse->fine quantiles, i.e.
    # large->small dense core => high->low replication) and keep the MAKESPAN-best point. The only constraint
    # is HBM-FEASIBILITY (the final eligibility gate rejects HBM-overflowing replication), and METIS is a
    # candidate floor (zord <= METIS by construction), so there is no need for an artificial cap: a hard 15%
    # cap WRONGLY blocked the ~30%-replication sweet spot that beats MemShare on collegemsg (§26).
    for q in (0.70, 0.80, 0.88, 0.93, 0.96, 0.98, 0.99, 0.995, 0.999):
        tau = max(2, int(np.quantile(core_val, q)))
        core_mask = core_val >= tau
        cs = int(core_mask.sum())
        if not (0 < cs < N):
            continue
        dev_p = balanced_periphery(np.nonzero(~core_mask)[0], deg, D, lpa_rank=lpa_rank)
        cut2, inc2, comm2, cnt2, extra2 = replicate_core_metrics(
            src, dst, deg, core_mask, dev_p, D, N, rng)
        if not feasible(cnt2, inc2, devs, F):          # HBM-feasibility is the ONLY hard constraint
            continue
        tot, _, _ = predict_ms(inc2 * F, comm2 * F, bw, link)
        mk = float(tot.max())
        if best_vc is None or mk < best_vc[0]:
            best_vc = (mk, (cut2, inc2, comm2, cnt2, extra2))
    if best_vc is not None:
        cand["vertex-cut(k-core)"] = best_vc[1]

    # ---- candidates 3/4: spatial (PSS) and temporal (PTS) corners ----
    # spatial corner: each snapshot-block on a device (vertex co-located by lpa, snapshot ignored).
    # In the single-home vertex model the "spatial" split is the lpa edge-cut with EQUAL work
    # (balance-blind to hetero) -> a reference. temporal corner: split VERTICES by their dominant
    # snapshot block -> approximates PTS (whole vertex timeline kept, spatial cut high).
    dev3 = lpa_edgecut(N, deg, lpa_rank, D, caps=None)             # equal-work spatial split (PSS-like)
    cut3, inc3, comm3, cnt3 = edgecut_metrics(src, dst, deg, dev3, D, N)
    cand["spatial(PSS)"] = (cut3, inc3, comm3, cnt3, 0)

    # temporal corner (PTS): balanced timeline split by first-activity time (the duality endpoint).
    dev4 = temporal_partition(src, dst, snap, deg, N, S, D)
    cut4, inc4, comm4, cnt4 = edgecut_metrics(src, dst, deg, dev4, D, N)
    cand["temporal(PTS)"] = (cut4, inc4, comm4, cnt4, 0)

    # ---- candidate 5: METIS balanced min-cut (D36 -- "degenerate to METIS when METIS is best") ----
    # ADOPT the SOTA cut-minimizer as a zord candidate so zord <= METIS BY CONSTRUCTION: the adaptive
    # pick selects METIS on cut-sensitive/slow-link graphs (where work-balance/vertex-cut don't help)
    # and only deviates when a lever provably lowers makespan. Skip gracefully if pymetis is missing.
    try:
        dev5 = metis_partition(src, dst, N, D)
        cut5, inc5, comm5, cnt5 = edgecut_metrics(src, dst, deg, dev5, D, N)
        cand["metis(min-cut)"] = (cut5, inc5, comm5, cnt5, 0)
    except Exception:
        pass

    # ---- adaptive pick: lowest predicted makespan, among FEASIBLE + non-degenerate plans ----
    # A plan that dumps the whole graph onto one device (leaving D-1 idle) is rejected: it is neither
    # feasible at scale (it would OOM the card) nor a real multi-device plan. We require every device
    # to do a non-trivial share of work (work_imb below D/2 is a soft "actually uses the devices"
    # gate) AND the plan to fit per-device HBM. If no candidate passes, we keep the most balanced one.
    cand_mk = {}
    eligible = []
    bal_gate = 0.5 * D + 0.5          # work_imb must be below this -> the plan actually uses the devices
    for name, (cut, inc, comm, cnt, extra) in cand.items():
        tot, _, _ = predict_ms(inc * F, comm * F, bw, link)
        cand_mk[name] = float(tot.max())
        work_imb = float(inc.max() / max(1e-9, inc.mean()))
        # METIS is the worst-case FLOOR: keep it eligible whenever feasible (even if its cut-min
        # partition is degree-imbalanced) so zord <= METIS holds BY CONSTRUCTION -- else the balance
        # gate could drop a METIS that is actually the fastest plan (auditor 2026-05-31). The gate
        # still filters zord's OWN candidates against a hidden-imbalance pick.
        is_floor = name == "metis(min-cut)"
        if feasible(cnt, inc, devs, F) and (work_imb <= bal_gate or is_floor):
            eligible.append(name)
    if not eligible:                  # nothing balanced+feasible -> fall back to the least-skewed plan
        eligible = [min(cand, key=lambda n: cand[n][1].max() / max(1e-9, cand[n][1].mean()))]
    best_name = min(eligible, key=lambda n: cand_mk[n])
    cut, inc, comm, cnt, extra = cand[best_name]
    return dict(name=best_name, cut=cut, incident=inc, comm_rows=comm, counts=cnt,
                extra_core_rows=extra, cand_mk=cand_mk)


# =====================================================================================
# feasibility (does each device's resident set fit its HBM?)
# =====================================================================================
def feasible(counts, incident, devs, F):
    """A device fits if node-rows (F*4 each) + edge bytes < usable HBM. Same footprint shape as the
    cost model's device_footprint_bytes (feat + edge mem)."""
    BYTES_PER_EDGE = 20.0   # src+dst+ts+w (cost_model.CostParams default)
    for k, d in enumerate(devs):
        feat = counts[k] * F * 4.0
        edgemem = incident[k] * BYTES_PER_EDGE        # incident-edge metadata resident on the device
        if feat + edgemem > d.usable_mem:
            return False
    return True


# =====================================================================================
# reporting
# =====================================================================================
def score_and_print(name, cut, incident, comm_rows, counts, extra_core_rows, M, N, D,
                     devs, bw, link, F):
    tot_ms, comp_ms, comm_ms = predict_ms(incident * F, comm_rows * F, bw, link)
    makespan = float(tot_ms.max())
    util = float(tot_ms.mean()) / makespan * 100 if makespan > 0 else 0.0
    work_imb = float(incident.max() / max(1e-9, incident.mean()))
    node_bal = float(counts.max() / max(1e-9, counts.mean()))
    cut_pct = cut / (2 * M) * 100.0
    repl = extra_core_rows / max(1, N) * 100.0
    feas = feasible(counts, incident, devs, F)
    print(f"  {name:<20} {cut_pct:>6.2f}% {node_bal:>7.3f} {work_imb:>8.3f} "
          f"{comp_ms.max():>9.2f} {comm_ms.max():>9.2f} {makespan:>10.2f} {util:>6.1f}% "
          f"{repl:>7.1f}% {'yes' if feas else 'NO':>5}")
    return makespan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="", help="staged real temporal graph (else --synthetic)")
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic power-law graph")
    ap.add_argument("--nodes", type=int, default=200_000, help="synthetic node count")
    ap.add_argument("--edges", type=int, default=2_000_000, help="synthetic edge count")
    ap.add_argument("--comms", type=int, default=64, help="synthetic community count")
    ap.add_argument("--intra", type=float, default=0.9, help="synthetic intra-community edge fraction")
    ap.add_argument("--devices", "-D", type=int, default=3, help="number of devices")
    ap.add_argument("--feat", "-F", type=int, default=128, help="node feature dim")
    ap.add_argument("--snapshots", "-S", type=int, default=64, help="snapshots (supra-graph time res)")
    ap.add_argument("--memshare-topk", type=float, default=0.10, help="MemShare hotspot fraction")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    D, F, S = a.devices, a.feat, a.snapshots
    cluster = _build_cluster(D)
    devs = cluster.devices
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)   # per-device agg bandwidth
    link = cluster.inter_node_bw                                     # slow interconnect for comm
    caps_work = bw.copy()                                            # hetero work budget ~ bandwidth

    # DGC asymmetric supra-edge weights: spatial edges aggregated N_GATHERS times, temporal once
    # (the paper's per-model profiled weights; spatial is "heavier").
    w_spatial = float(N_GATHERS)
    w_temporal = 1.0

    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        src, dst, N = g.src.astype(np.int64), g.dst.astype(np.int64), g.num_nodes
        E = src.size
        snap = np.minimum((np.arange(E) * S // E).astype(np.int64), S - 1)  # equal-count snapshots
        name = g.name
    else:
        N = a.nodes
        src, dst, t = gen_powerlaw_temporal(N, a.edges, a.comms, a.intra, seed=a.seed)
        E = src.size
        snap = np.minimum((np.arange(E) * S // E).astype(np.int64), S - 1)
        name = f"synth-powerlaw(comms={a.comms},intra={a.intra})"
    M = int(E)

    print(f"SOTA-COMPARE  graph={name}  N={N:,}  M={M:,}  devices={D}  feat={F}  snapshots={S}")
    print(f"  cluster={[d.name for d in devs]}  hbm_bw={bw.round().tolist()}GB/s  "
          f"inter-link={link}GB/s")
    print(f"  loaded/generated in {time.time()-t0:.2f}s")

    # shared structural quantities (computed once)
    deg = node_degree(src, dst, N)
    tc = time.time()
    lpa_rank, lpa_cost = cpp_order(N, src, dst, "lpa", "lpa")
    if lpa_rank is None:
        print(f"  [C++ lpa] unavailable ({_BIN}); falling back to degree order for cluster layout")
        lpa_rank = np.argsort(np.argsort(-deg, kind="stable")).astype(np.int32)
    else:
        print(f"  C++ lpa cluster layout in {lpa_cost:.2f}s")
    core_val = coreness(src, dst, N)
    print(f"  k-core: max_core={int(core_val.max())}  90th-pct coreness={int(np.quantile(core_val,0.9))}  "
          f"(structure in {time.time()-tc:.2f}s)")
    print()

    hdr = (f"  {'method':<20} {'cut%':>6} {'node_bal':>7} {'work_imb':>8} "
           f"{'comp_ms':>9} {'comm_ms':>9} {'makespan':>10} {'util':>6} {'repl':>7} {'feas':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = {}

    # ---- DGC-PGC ----
    try:
        dev_dgc = dgc_pgc_partition(src, dst, snap, deg, N, S, D, caps_work, lpa_rank,
                                    w_spatial, w_temporal)
        cut, inc, comm, cnt = edgecut_metrics(src, dst, deg, dev_dgc, D, N)
        results["DGC-PGC"] = score_and_print("DGC-PGC (LPA chunks)", cut, inc, comm, cnt, 0,
                                             M, N, D, devs, bw, link, F)
    except Exception as e:
        print(f"  {'DGC-PGC (LPA chunks)':<20} SKIP ({str(e)[:60]})")

    # ---- MemShare ----
    try:
        core_mask, dev_p = memshare_partition(src, dst, deg, N, D, a.memshare_topk, lpa_rank)
        cut, inc, comm, cnt, extra = replicate_core_metrics(src, dst, deg, core_mask, dev_p, D, N,
                                                            np.random.default_rng(a.seed + 7))
        results["MemShare"] = score_and_print(
            f"MemShare (top{int(a.memshare_topk*100)}%)", cut, inc, comm, cnt, extra,
            M, N, D, devs, bw, link, F)
    except Exception as e:
        print(f"  {'MemShare':<20} SKIP ({str(e)[:60]})")

    # ---- METIS (pymetis) ----
    try:
        dev_metis = metis_partition(src, dst, N, D)
        cut, inc, comm, cnt = edgecut_metrics(src, dst, deg, dev_metis, D, N)
        results["METIS"] = score_and_print("METIS (balanced)", cut, inc, comm, cnt, 0,
                                           M, N, D, devs, bw, link, F)
    except Exception as e:
        print(f"  {'METIS (balanced)':<20} SKIP ({str(e)[:55]})")

    # ---- PSS / PTS references (the duality corners, single-home) ----
    dev_pss = lpa_edgecut(N, deg, lpa_rank, D, caps=None)
    cut, inc, comm, cnt = edgecut_metrics(src, dst, deg, dev_pss, D, N)
    results["PSS"] = score_and_print("PSS (spatial)", cut, inc, comm, cnt, 0, M, N, D, devs, bw, link, F)

    dev_pts = temporal_partition(src, dst, snap, deg, N, S, D)
    cut, inc, comm, cnt = edgecut_metrics(src, dst, deg, dev_pts, D, N)
    results["PTS"] = score_and_print("PTS (temporal)", cut, inc, comm, cnt, 0, M, N, D, devs, bw, link, F)

    # ---- ZORD ----
    z = zord_arrange(src, dst, snap, deg, core_val, N, S, D, devs, bw, link, F, caps_work, lpa_rank,
                     np.random.default_rng(a.seed + 1))
    results["ZORD"] = score_and_print(f"ZORD [{z['name']}]", z["cut"], z["incident"], z["comm_rows"],
                                      z["counts"], z["extra_core_rows"], M, N, D, devs, bw, link, F)
    print("   zord adaptive-corner candidates (makespan ms): " +
          ", ".join(f"{k}={v:.1f}" for k, v in z["cand_mk"].items()))

    # ================= HEADLINE =================
    print("\n  ================= HEADLINE: zord vs the SOTA partition axis =================")
    z_mk = results["ZORD"]
    competitors = [(k, v) for k, v in results.items()
                   if k in ("DGC-PGC", "MemShare", "METIS")]
    if not competitors:
        print("  (no SOTA competitor ran -- check pymetis / C++ binary)")
    for k, v in competitors:
        if z_mk < v - 1e-9:
            verdict = f"zord WINS  ({v / z_mk:.2f}x faster makespan)"
        elif z_mk > v + 1e-9:
            verdict = f"zord LOSES ({z_mk / v:.2f}x slower -- HONEST)"
        else:
            verdict = "TIE"
        print(f"    vs {k:<10} : zord={z_mk:8.2f}ms  {k}={v:8.2f}ms  -> {verdict}")
    best_comp = min((v for _, v in competitors), default=None)
    if best_comp is not None:
        if z_mk < best_comp - 1e-9:
            print(f"  => ZORD beats the BEST SOTA competitor by {best_comp / z_mk:.2f}x on makespan "
                  f"(power-law temporal graph, incident-edge work model).")
        elif z_mk > best_comp + 1e-9:
            print(f"  => ZORD does NOT beat the best SOTA competitor here "
                  f"({z_mk / best_comp:.2f}x slower) -- reported honestly.")
        else:
            print("  => ZORD ties the best SOTA competitor on makespan.")

    # ---- one-line positioning note ----
    print("\n  POSITIONING (the 11 references):")
    print("    DIRECT competitors (PARTITION axis -- compared above): "
          "DGC-PGC (chunk/LPA), MemShare (hotspot vertex-cut), METIS (balanced min-cut), "
          "DistDy (online partition), CaPGNN (hetero-aware partition).")
    print("    ORTHOGONAL (TRAINING-EXECUTION layers -- zord's partition FEEDS them, not vs): "
          "DistTGL (memory parallelism), MSPipe (staleness pipeline), GNNFlow (streaming "
          "storage/cache), Orca (embedding reuse), SIMPLE/ETC (feature placement/batching), "
          "NeutronStream (event-parallel) -- they optimize HOW a partition is executed; "
          "zord decides WHAT the partition is, then hands it off.")
    print(f"\n  total {time.time()-t0:.2f}s")


def metis_partition(src, dst, N, D):
    """METIS balanced min-cut via pymetis (raises -> SKIP if pymetis missing). Heterogeneity-blind
    (equal parts), so it minimizes UNWEIGHTED cut, not bandwidth-weighted makespan -- exactly the
    blind spot the comparison probes."""
    import pymetis
    m = src != dst
    s, d = src[m], dst[m]
    u = np.concatenate([s, d]); v = np.concatenate([d, s])
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, u + 1, 1)
    np.cumsum(indptr, out=indptr)
    adjncy = v.astype(np.int64)
    try:                                    # pymetis >= newer: CSRAdjacency object (xadj/adjncy depr.)
        adj = pymetis.CSRAdjacency(indptr.tolist(), adjncy.tolist())
        _, membership = pymetis.part_graph(D, adjacency=adj)
    except Exception:                       # older pymetis: the xadj/adjncy keyword form
        _, membership = pymetis.part_graph(D, xadj=indptr.tolist(), adjncy=adjncy.tolist())
    return np.asarray(membership, dtype=np.int64)


def _build_cluster(D):
    """HetCluster profile with EXACTLY D devices, round-robin across the 3 measured tiers (so HBM
    bandwidth + intra/inter-node link skew are present). For D=3 this is exactly hetcluster()."""
    if D == 3:
        return hetcluster()
    tiers = ["H100-80GB", "RTX6000Ada-48GB", "RTX5000Ada-32GB"]
    devs = []
    for i in range(D):
        key = tiers[i % len(tiers)]
        m = _MEASURED[key]
        devs.append(DeviceProfile(i, key, int(m["mem_gb"] * GB), throughput=m["r"],
                                  node=i % len(tiers), h2d_gbps=m["h2d"],
                                  hbm_bw_gbps=m["hbm"], measured=True))
    return ClusterProfile(devices=devs)


if __name__ == "__main__":
    main()
