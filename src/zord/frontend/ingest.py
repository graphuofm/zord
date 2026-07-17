"""FRONT-END (K1): ingest temporal/attributed graphs into the zord pipeline.

This is the single entry the middle-end (the scheduler / arrange / planner) calls.
It takes a raw temporal graph -- either a DTDG sequence of snapshots or a CTDG event
stream -- and produces ONE clean structure, ``TemporalGraphInput``, wrapping:

  - the in-memory ``TemporalGraph`` payload (SoA edges; the MIDDLE consumes it directly),
  - ``snap``  : the per-edge supra-graph time axis (snapshot id per edge). The supra-graph
                over cells (v, t) has SPATIAL edges (within a snapshot) and TEMPORAL edges
                (same vertex, adjacent snapshots); ``snap`` is the time coordinate of every
                spatial edge and is exactly what ``arrange``'s PTS corner / the space-time
                duality consume. ``build_snap`` reproduces, bit-for-bit, the array that
                ``arrange``/``planner.plan`` build internally so the MIDDLE can pass it
                through unchanged (no caller re-derives it).
  - ``feat_bytes`` : the optional per-node feature-size vector F_v (passed through untouched),
  - ``stats`` : a CHEAP ``GraphStats`` probe (the GRAPH half of the L2 calibration) that
                sets the w_S/w_T regime: clusterability (LPA-modularity proxy), persistence
                (THEORY 9.4's temporal autocorrelation rho(v)), degree tails, feature stats.

Everything here is pure numpy + dataclasses: NO torch, import-safe on a CPU box.
Graph structure (clusterability) goes through ``zord.partition.cpp_kernel`` (C++ when the
binary is built, vectorized-numpy fallback otherwise) -- NEVER networkx, per project rule,
so it scales to 100M-edge graphs. On graphs above ``sample_edges`` the *structural* probes
(clusterability / feature homophily) are estimated on a uniform edge sample, while the
COUNTS (N, E, degrees, persistence) stay EXACT.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..partition import cpp_kernel

if TYPE_CHECKING:  # import-only types (no runtime cost / no torch)
    from ..datasets.temporal_graph import TemporalGraph
    from ..storage.backend import StorageBackend


GB: int = 1024 ** 3


# --------------------------------------------------------------------------- #
# C++ HOT-PATH kernels (optional): supra_build + graph_stats.                  #
#                                                                              #
# The structural passes that touch EVERY edge/cell at 100M-1B scale move to    #
# C++ (the numpy concat+unique allocates ~5x M int64 and OOMs/slows at billion #
# edge). Both kernels are standalone main()s doing little-endian binary file   #
# I/O, resolved via env var then <repo>/build/<name>, EXACTLY mirroring        #
# partition.cpp_kernel.graph_bin_path. A pure-numpy fallback (the existing     #
# build_supra_cells / sort-over-2E loops) keeps the planner ALWAYS running     #
# (correctly, slower) when the binary is absent.                               #
# --------------------------------------------------------------------------- #
def _repo_root() -> str:
    """Repo root = four levels up from this file (src/zord/frontend/ingest.py)."""
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))


def supra_build_bin_path() -> str:
    """Resolve the supra_build binary: $ZORD_SUPRA_BUILD_BIN, else <repo>/build/supra_build."""
    env = os.environ.get("ZORD_SUPRA_BUILD_BIN")
    if env:
        return env
    return os.path.join(_repo_root(), "build", "supra_build")


def have_supra_build() -> bool:
    return os.path.exists(supra_build_bin_path())


def graph_stats_bin_path() -> str:
    """Resolve the graph_stats binary: $ZORD_GRAPH_STATS_BIN, else <repo>/build/graph_stats."""
    env = os.environ.get("ZORD_GRAPH_STATS_BIN")
    if env:
        return env
    return os.path.join(_repo_root(), "build", "graph_stats")


def have_graph_stats() -> bool:
    return os.path.exists(graph_stats_bin_path())


def _write_triples(path: str, N: int, S: int, src: np.ndarray, dst: np.ndarray,
                   snap: np.ndarray) -> None:
    """Write the shared int64 N,S,M + int32 triples[3*M] (src,dst,snap) input prefix.

    IDENTICAL layout to supra_solve.py::write_input's prefix, so supra_build and
    graph_stats (and supra_solver) all read the same writer's bytes.
    """
    M = int(src.size)
    trip = np.empty(3 * M, dtype=np.int32)
    trip[0::3] = src.astype(np.int32)
    trip[1::3] = dst.astype(np.int32)
    trip[2::3] = snap.astype(np.int32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<qqq", int(N), int(S), int(M)))
        trip.tofile(fp)


# --------------------------------------------------------------------------- #
# numpy fallback for the supra-cell materialization (copied from              #
# scripts/supra_solve.py::build_supra_cells -- the reference the C++ mirrors). #
# --------------------------------------------------------------------------- #
def _build_supra_cells_numpy(src: np.ndarray, dst: np.ndarray, snap: np.ndarray,
                             N: int, S: int):
    """Return (cell_v, cell_t, sp_a, sp_b, tp_a, tp_b), all int64/int32.

    Cells are unique (vertex, snapshot) pairs carrying an incident edge, ordered by
    (vertex, snapshot) -- IDENTICAL canonical order to supra_build.cpp / supra_solver.cpp.
    Spatial pairs are kept in EDGE order dropping a==b (no dedup of parallel edges);
    temporal pairs connect consecutive same-vertex cells.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    snap = np.asarray(snap, dtype=np.int64)
    S = int(S)
    if src.size == 0:
        z = np.zeros(0, dtype=np.int64)
        zi = np.zeros(0, dtype=np.int32)
        return z.copy(), z.copy(), zi.copy(), zi.copy(), zi.copy(), zi.copy()
    ks = src * S + snap
    kd = dst * S + snap
    keys = np.unique(np.concatenate([ks, kd]))       # sorted unique == cell ids 0..C-1
    cell_v = (keys // S).astype(np.int64)
    cell_t = (keys % S).astype(np.int64)
    a = np.searchsorted(keys, ks)
    b = np.searchsorted(keys, kd)
    m = a != b                                        # drop self/same-cell pairs
    sp_a = a[m].astype(np.int32)
    sp_b = b[m].astype(np.int32)
    same_v = cell_v[1:] == cell_v[:-1]                # consecutive same-vertex cells
    idx = np.nonzero(same_v)[0]
    tp_a = idx.astype(np.int32)
    tp_b = (idx + 1).astype(np.int32)
    return cell_v, cell_t, sp_a, sp_b, tp_a, tp_b


def build_supra(graph: "TemporalGraph", snap: np.ndarray, num_snapshots: int):
    """Materialise the active-cell table + spatial/temporal cell-pair lists.

    Prefers the C++ ``build/supra_build`` (the O(M log M) sort/unique + searchsorted that
    OOMs/slows in numpy at billion-edge scale); falls back to the pure-numpy
    ``build_supra_cells`` logic (copied from scripts/supra_solve.py) when the binary is
    absent or its run fails -- so the MIDDLE always gets cells (correctly, slower).

    Returns:
      cell_v, cell_t : int64 [C]   per-cell coordinates (vertex-major, snapshot-minor)
      sp_a, sp_b     : int32 [.]   spatial cell-pairs (within-snapshot edge endpoints, a!=b)
      tp_a, tp_b     : int32 [.]   temporal cell-pairs (same vertex, adjacent active snapshots)
    """
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)
    snap = np.asarray(snap, dtype=np.int64)
    N = int(graph.num_nodes)
    S = int(num_snapshots)

    binp = supra_build_bin_path()
    if os.path.exists(binp) and src.size > 0:
        try:
            with tempfile.TemporaryDirectory(prefix="zord_kernel_") as tmp:
                ipath = os.path.join(tmp, "in.bin")
                opath = os.path.join(tmp, "out.bin")
                _write_triples(ipath, N, S, src, dst, snap)
                r = subprocess.run([binp, ipath, opath], capture_output=True, text=True)
                if r.returncode == 0:
                    with open(opath, "rb") as fp:
                        C = struct.unpack("<q", fp.read(8))[0]
                        cell_v = np.fromfile(fp, dtype=np.int32, count=C).astype(np.int64)
                        cell_t = np.fromfile(fp, dtype=np.int32, count=C).astype(np.int64)
                        (n_sp,) = struct.unpack("<q", fp.read(8))
                        sp = np.fromfile(fp, dtype=np.int32, count=2 * n_sp)
                        (n_tp,) = struct.unpack("<q", fp.read(8))
                        tp = np.fromfile(fp, dtype=np.int32, count=2 * n_tp)
                    sp_a = sp[0::2].copy(); sp_b = sp[1::2].copy()
                    tp_a = tp[0::2].copy(); tp_b = tp[1::2].copy()
                    return cell_v, cell_t, sp_a, sp_b, tp_a, tp_b
        except (OSError, struct.error, ValueError):
            pass  # fall through to numpy
    return _build_supra_cells_numpy(src, dst, snap, N, S)


# --------------------------------------------------------------------------- #
# GraphStats -- the cheap graph probe (the GRAPH half of the L2 calibration).  #
# --------------------------------------------------------------------------- #
@dataclass
class GraphStats:
    """Cheap structural probe of a temporal graph.

    The scheduler/prober reads these to calibrate the space/time cut weights
    (w_S vs w_T) and to pre-filter the decomposition axis. Counts are exact;
    the two structural ratios (clusterability, feature_homophily) may be sampled
    on very large graphs (see ``graph_stats``).
    """
    num_nodes: int
    num_edges: int
    avg_degree: float
    density: float
    max_degree: int
    deg_p99: int
    num_snapshots: int
    mean_snapshot_nodes: float
    max_snapshot_nodes: int
    clusterability: float            # LPA-modularity proxy in [0, 1] (community structure)
    persistence: float               # mean rho(v) = (|T_v|-1)/(S-1) over active vertices
    feature_homophily: Optional[float] = None
    feat_dim_mean: float = 0.0
    feat_dim_max: float = 0.0
    ingest_sec: float = 0.0

    def summary(self) -> str:
        fh = "n/a" if self.feature_homophily is None else f"{self.feature_homophily:.3f}"
        return (
            f"GraphStats(N={self.num_nodes:,} E={self.num_edges:,} "
            f"avg_deg={self.avg_degree:.2f} density={self.density:.2e} "
            f"max_deg={self.max_degree:,} p99_deg={self.deg_p99:,}\n"
            f"  snapshots={self.num_snapshots} mean_snap_nodes={self.mean_snapshot_nodes:.1f} "
            f"max_snap_nodes={self.max_snapshot_nodes:,}\n"
            f"  clusterability={self.clusterability:.3f} persistence={self.persistence:.3f} "
            f"feat_homophily={fh} feat_dim(mean/max)={self.feat_dim_mean:.1f}/{self.feat_dim_max:.1f}\n"
            f"  ingest={self.ingest_sec*1e3:.1f} ms)"
        )


# --------------------------------------------------------------------------- #
# TemporalGraphInput -- the clean object FRONT hands the optimizer.           #
# --------------------------------------------------------------------------- #
@dataclass
class TemporalGraphInput:
    """The one structure the FRONT-END hands the MIDDLE-END.

    Wraps everything the optimizer needs so no downstream caller re-derives
    ``snap`` (the supra-graph time axis) or ``feat_bytes`` (the F_v vector).
    """
    graph: "TemporalGraph"
    snap: np.ndarray                 # int64 [E], per-edge snapshot id (the time axis)
    feat_bytes: Optional[np.ndarray]  # float64 [N] or None  (per-node feature sizes F_v)
    stats: GraphStats
    mode: str = "dtdg"               # "dtdg" (snapshot) | "ctdg" (event stream)

    @property
    def num_nodes(self) -> int:
        return int(self.graph.num_nodes)

    @property
    def num_edges(self) -> int:
        return int(self.graph.num_edges)

    def summary(self) -> str:
        return (
            f"TemporalGraphInput(name={self.graph.name!r} mode={self.mode} "
            f"N={self.num_nodes:,} E={self.num_edges:,} S={self.stats.num_snapshots} "
            f"feat_bytes={'yes' if self.feat_bytes is not None else 'no'})\n"
            f"  {self.stats.summary()}"
        )


# --------------------------------------------------------------------------- #
# build_snap -- the supra-graph time axis (must match arrange/planner exactly) #
# --------------------------------------------------------------------------- #
def build_snap(graph: "TemporalGraph", num_snapshots: int = 64) -> np.ndarray:
    """Per-edge int64 snapshot id over the TIME-SORTED edge stream.

    This is the supra-graph time coordinate: the cell of edge e is (its endpoints,
    snap[e]). Equal-COUNT buckets (not equal-time-width) so every snapshot carries
    a comparable amount of work -- and, critically, this is *bit-for-bit* the array
    ``arrange`` builds internally when ``snap is None`` (arrange.py:429):

        snap = np.minimum((np.arange(E) * S // max(1, E)).astype(int64), S - 1)

    so MIDDLE/scheduler can pass this through to ``arrange`` unchanged. The graph is
    sorted by time in place first (the stream order the snapshot id indexes into).
    """
    graph.sort_by_time()
    E = int(graph.num_edges)
    S = int(num_snapshots)
    if E == 0:
        return np.zeros(0, dtype=np.int64)
    snap = np.minimum((np.arange(E) * S // max(1, E)).astype(np.int64), S - 1)
    return snap


# --------------------------------------------------------------------------- #
# graph_stats -- the cheap probe                                              #
# --------------------------------------------------------------------------- #
def _clusterability_from_lpa(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> float:
    """LPA-modularity proxy in [0, 1]: fraction of edges that fall WITHIN the
    label-propagation communities, corrected for the chance/expected within-edge
    fraction (a simplified Newman modularity, mapped to [0, 1]).

    ``cpp_kernel.lpa_rank`` returns rank[old]->position, i.e. a community-GROUPED
    node ORDER, not raw labels. We recover community blocks by binning the ordering
    into sqrt(N)-ish contiguous bands (nodes of the same community are contiguous in
    the LPA order, so a contiguous band approximates a community). The modularity-like
    score is (within-edge fraction) - (expected within-edge fraction), clamped to
    [0, 1]. This is cheap (O(E)) and torch-/networkx-free.
    """
    N = int(num_nodes)
    E = int(src.size)
    if N <= 1 or E == 0:
        return 0.0
    rank = cpp_kernel.lpa_rank(N, src, dst)        # rank[old] -> position in LPA order
    rank = np.asarray(rank, dtype=np.int64)
    # Bin the LPA order into B contiguous bands ~ communities. B grows with N but is
    # kept modest so each band is a plausible community (the score is a proxy, not exact).
    B = int(max(2, min(N, round(np.sqrt(N)))))
    band = np.minimum((rank * B) // N, B - 1).astype(np.int64)   # band[old] in [0, B)
    same = band[src] == band[dst]
    within = float(np.count_nonzero(same)) / float(E)
    # expected within-fraction under the configuration null ~ sum_b (vol_b / 2m)^2.
    deg = cpp_kernel.node_degree(src, dst, N).astype(np.float64)
    two_m = float(deg.sum())
    if two_m <= 0:
        return 0.0
    vol_b = np.bincount(band, weights=deg, minlength=B)
    expected = float(np.square(vol_b / two_m).sum())
    mod = within - expected                                       # Newman-style modularity
    if mod <= 0.0:
        return 0.0
    # Normalize by the max attainable (1 - expected) so a perfectly modular graph -> 1.
    denom = max(1e-9, 1.0 - expected)
    return float(min(1.0, mod / denom))


def _cpp_graph_stats(src: np.ndarray, dst: np.ndarray, snap: np.ndarray,
                     N: int, S: int):
    """Run ``build/graph_stats`` for the EXACT counts half of GraphStats.

    Returns ``(deg[N], per_snapshot_nodes[S], Tv[N])`` (all int64) or ``None`` so the
    caller falls back to the numpy bincount/unique loops. The C++ kernel computes the
    undirected degree, the per-snapshot distinct active-node count, and |T_v| (distinct
    snapshots per vertex) in O(E log E) over the full stream -- the passes that hang in
    numpy at 100M-1B edges. Same int64 N,S,M + int32 triples[3*M] input prefix as
    supra_build / supra_solver.
    """
    binp = graph_stats_bin_path()
    if not os.path.exists(binp) or src.size == 0:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="zord_kernel_") as tmp:
            ipath = os.path.join(tmp, "in.bin")
            opath = os.path.join(tmp, "out.bin")
            _write_triples(ipath, N, S, src, dst, snap)
            r = subprocess.run([binp, ipath, opath], capture_output=True, text=True)
            if r.returncode != 0:
                return None
            with open(opath, "rb") as fp:
                (n1,) = struct.unpack("<q", fp.read(8))
                deg = np.fromfile(fp, dtype=np.int32, count=n1).astype(np.int64)
                (s1,) = struct.unpack("<q", fp.read(8))
                per_snap = np.fromfile(fp, dtype=np.int32, count=s1).astype(np.int64)
                (n2,) = struct.unpack("<q", fp.read(8))
                tv = np.fromfile(fp, dtype=np.int32, count=n2).astype(np.int64)
    except (OSError, struct.error, ValueError):
        return None
    if deg.size != N or per_snap.size != S or tv.size != N:
        return None
    return deg, per_snap, tv


def _persistence(src: np.ndarray, dst: np.ndarray, snap: np.ndarray,
                 num_nodes: int, num_snapshots: int) -> float:
    """Mean over ACTIVE vertices of rho(v) = (|T_v| - 1) / (S - 1), where |T_v| is the
    number of distinct snapshots in which v appears (THEORY 9.4's temporal autocorrelation
    that L_temporal is weighted by). Computed exactly: per-edge snapshot id -> per-vertex
    distinct-snapshot count via a sort over (vertex, snap) pairs (no networkx).
    """
    S = int(num_snapshots)
    N = int(num_nodes)
    if S <= 1 or src.size == 0:
        return 0.0
    # Endpoints touch a (vertex, snapshot) cell; count DISTINCT snapshots per vertex.
    v = np.concatenate([src, dst]).astype(np.int64)
    s = np.concatenate([snap, snap]).astype(np.int64)
    key = v * np.int64(S) + s                       # unique cell key
    key = np.unique(key)                            # distinct (v, s) cells
    vv = key // S                                   # vertex of each distinct cell
    # |T_v| = number of distinct cells per vertex; active vertices = those that appear.
    tv = np.bincount(vv, minlength=N).astype(np.float64)
    active = tv > 0
    if not active.any():
        return 0.0
    rho = (tv[active] - 1.0) / float(S - 1)
    return float(rho.mean())


def _feature_homophily(src: np.ndarray, dst: np.ndarray,
                       feat_bytes: np.ndarray, num_nodes: int) -> Optional[float]:
    """Cheap homophily proxy on the per-node feature-SIZE signal F_v: correlation of
    endpoint feature sizes across edges, mapped to [0, 1]. (We only have F_v here, not
    full feature vectors, so this measures whether feature-heavy nodes connect to other
    feature-heavy nodes -- the signal that matters for attribute-aware placement.)"""
    fb = np.asarray(feat_bytes, dtype=np.float64)
    if fb.size != num_nodes or src.size == 0:
        return None
    a = fb[src]
    b = fb[dst]
    sa, sb = a.std(), b.std()
    if sa <= 1e-12 or sb <= 1e-12:
        return 1.0  # constant feature sizes -> perfectly homophilous on this signal
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r):
        return 1.0
    return float((r + 1.0) * 0.5)   # map [-1, 1] -> [0, 1]


def graph_stats(graph: "TemporalGraph", num_snapshots: int = 64,
                feat_bytes: Optional[np.ndarray] = None,
                sample_edges: int = 2_000_000) -> GraphStats:
    """Compute the cheap ``GraphStats`` probe.

    Exact (always full-graph): N, E, avg/max/p99 degree, density, per-snapshot node
    counts, and persistence (THEORY 9.4 rho(v)). Sampled on graphs above ``sample_edges``:
    clusterability (LPA-modularity proxy) and feature_homophily, estimated on a uniform
    edge sample to cap cost on >100M-edge graphs. All via cpp_kernel + numpy bincount;
    NEVER networkx.
    """
    t0 = time.perf_counter()
    graph.sort_by_time()
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)
    N = int(graph.num_nodes)
    E = int(src.size)
    S = int(num_snapshots)
    snap = build_snap(graph, num_snapshots=S)

    # --- EXACT counts (full graph): degree dist, per-snapshot active nodes, |T_v|. ---
    # Routes through build/graph_stats (the O(E log E) unique passes that hang in numpy at
    # 100M-1B edges) when present; the numpy bincount/unique loops are the fallback. Both
    # paths derive the IDENTICAL scalars below from the same three arrays.
    cpp = _cpp_graph_stats(src, dst, snap, N, S) if E > 0 and S > 0 else None
    if cpp is not None:
        deg, per_snap_nodes, tv = cpp
        per_snap_nodes = per_snap_nodes.astype(np.float64)
    elif E > 0:
        deg = cpp_kernel.node_degree(src, dst, N).astype(np.int64)
        # per-snapshot distinct active-node count: unique (snap, vertex) cells per snap
        v = np.concatenate([src, dst]).astype(np.int64)
        sp = np.concatenate([snap, snap]).astype(np.int64)
        cell = np.unique(sp * np.int64(N) + v)
        per_snap_nodes = np.bincount(cell // N, minlength=S).astype(np.float64)
        # |T_v| = distinct snapshots per vertex (distinct (vertex, snap) cells per vertex)
        keyv = np.unique(v * np.int64(S) + sp)
        tv = np.bincount(keyv // S, minlength=N).astype(np.int64)
    else:
        deg = np.zeros(N, dtype=np.int64)
        per_snap_nodes = np.zeros(S, dtype=np.float64)
        tv = np.zeros(N, dtype=np.int64)

    # --- degree-derived scalars ---
    if E > 0:
        max_degree = int(deg.max())
        deg_p99 = int(np.percentile(deg, 99))
        avg_degree = float(E) / float(max(1, N))
    else:
        max_degree = 0
        deg_p99 = 0
        avg_degree = 0.0
    # density of a directed simple graph ~ E / (N*(N-1))
    density = float(E) / float(max(1, N * (N - 1))) if N > 1 else 0.0

    # --- snapshot occupancy + persistence (THEORY 9.4 rho(v) = (|T_v|-1)/(S-1)) ---
    if E > 0 and S > 0:
        nonempty = per_snap_nodes[per_snap_nodes > 0]
        mean_snapshot_nodes = float(nonempty.mean()) if nonempty.size else 0.0
        max_snapshot_nodes = int(per_snap_nodes.max()) if per_snap_nodes.size else 0
        if S > 1:
            active = tv > 0
            persistence = (float(((tv[active].astype(np.float64) - 1.0) / float(S - 1)).mean())
                           if active.any() else 0.0)
        else:
            persistence = 0.0
    else:
        mean_snapshot_nodes = 0.0
        max_snapshot_nodes = 0
        persistence = 0.0

    # --- (possibly sampled) structural ratios ---
    if E > sample_edges and sample_edges > 0:
        rng = np.random.default_rng(0)
        sel = rng.choice(E, size=sample_edges, replace=False)
        s_src, s_dst = src[sel], dst[sel]
    else:
        s_src, s_dst = src, dst
    clusterability = _clusterability_from_lpa(N, s_src, s_dst)

    # --- feature stats (from F_v if given, else edge-feature dim from the graph) ---
    feature_homophily: Optional[float] = None
    feat_dim_mean = 0.0
    feat_dim_max = 0.0
    if feat_bytes is not None:
        fb = np.asarray(feat_bytes, dtype=np.float64)
        if fb.size:
            feat_dim_mean = float(fb.mean())
            feat_dim_max = float(fb.max())
        feature_homophily = _feature_homophily(s_src, s_dst, feat_bytes, N)
    elif getattr(graph, "efeat", None) is not None:
        fe = graph.efeat
        feat_dim_mean = float(fe.shape[1])
        feat_dim_max = float(fe.shape[1])

    ingest_sec = time.perf_counter() - t0
    return GraphStats(
        num_nodes=N, num_edges=E, avg_degree=avg_degree, density=density,
        max_degree=max_degree, deg_p99=deg_p99, num_snapshots=S,
        mean_snapshot_nodes=mean_snapshot_nodes, max_snapshot_nodes=max_snapshot_nodes,
        clusterability=clusterability, persistence=persistence,
        feature_homophily=feature_homophily, feat_dim_mean=feat_dim_mean,
        feat_dim_max=feat_dim_max, ingest_sec=ingest_sec,
    )


# --------------------------------------------------------------------------- #
# ingest -- the FRONT-END entry                                               #
# --------------------------------------------------------------------------- #
def ingest(graph: "TemporalGraph", *, num_snapshots: int = 64,
           feat_bytes: Optional[np.ndarray] = None, mode: str = "dtdg",
           compute_stats: bool = True) -> TemporalGraphInput:
    """The FRONT-END entry: sort by time, build the supra-graph time axis ``snap``,
    compute the ``GraphStats`` probe (timed -> ``stats.ingest_sec``), and wrap into a
    ``TemporalGraphInput``.

    mode='dtdg' : discrete-time snapshot model (equal-count snapshot buckets).
    mode='ctdg' : continuous-time event stream -- keep event order; ``snap`` is still
                  built (the equal-count bucketing is a uniform binning of the event
                  stream, used only as the supra-graph time axis) but we record mode so
                  downstream code does not assume per-snapshot DTDG semantics.

    ``feat_bytes`` (the F_v vector) is passed through UNTOUCHED. Full precision; no torch.
    """
    if mode not in ("dtdg", "ctdg"):
        raise ValueError(f"mode must be 'dtdg' or 'ctdg', got {mode!r}")
    t0 = time.perf_counter()
    graph.sort_by_time()
    snap = build_snap(graph, num_snapshots=num_snapshots)
    fb = None if feat_bytes is None else np.asarray(feat_bytes, dtype=np.float64)

    if compute_stats:
        stats = graph_stats(graph, num_snapshots=num_snapshots, feat_bytes=fb)
        # fold the full sort+snap+probe wall time into ingest_sec (feeds JobEstimate.front_sec)
        stats.ingest_sec = time.perf_counter() - t0
    else:
        E = int(graph.num_edges)
        N = int(graph.num_nodes)
        avg = float(E) / float(max(1, N))
        stats = GraphStats(
            num_nodes=N, num_edges=E, avg_degree=avg,
            density=float(E) / float(max(1, N * (N - 1))) if N > 1 else 0.0,
            max_degree=0, deg_p99=0, num_snapshots=int(num_snapshots),
            mean_snapshot_nodes=0.0, max_snapshot_nodes=0,
            clusterability=0.0, persistence=0.0,
            feat_dim_mean=(float(fb.mean()) if fb is not None and fb.size else 0.0),
            feat_dim_max=(float(fb.max()) if fb is not None and fb.size else 0.0),
            ingest_sec=time.perf_counter() - t0,
        )
    return TemporalGraphInput(graph=graph, snap=snap, feat_bytes=fb, stats=stats, mode=mode)


def ingest_dataset(name: str, *, prefer: str = "hetcluster", num_snapshots: int = 64,
                   feat_bytes: Optional[np.ndarray] = None) -> TemporalGraphInput:
    """Convenience: ``datasets.load(name)`` then ``ingest()``.

    DTDG is assumed (the staged datasets are snapshot/interaction graphs); pass
    ``feat_bytes`` to fold the per-node feature-size vector through.
    """
    from ..datasets.loaders import load   # lazy: avoids import-time dataset deps
    graph = load(name, prefer=prefer)
    return ingest(graph, num_snapshots=num_snapshots, feat_bytes=feat_bytes, mode="dtdg")


def ingest_stream(backend: "StorageBackend", t0: int, t1: int, *,
                  num_snapshots: int = 64) -> TemporalGraphInput:
    """CTDG window: pull the events in [t0, t1) from a ``StorageBackend`` via
    ``backend.range()`` and ingest them as a continuous-time event stream (mode='ctdg').

    Used by the online subsystem (dynamic_online): each step ingests the next time
    window off the backend without re-loading the whole graph.
    """
    sub = backend.range(int(t0), int(t1))
    return ingest(sub, num_snapshots=num_snapshots, mode="ctdg")
