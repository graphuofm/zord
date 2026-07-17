"""FEATURE-PARALLEL decomposition (the ATTRIBUTE-DIMENSION axis) -- a NEW decomposition axis
for zord's arrange, orthogonal to the existing NODE-parallel (vertex-partition + boundary comm)
plan in arrange.py.

MOTIVATION (user): future temporal graphs carry LOTS of attributes. When the feature dimension F
is large, per-device FEATURE MEMORY (N*F*4 bytes) dominates the HBM footprint -- not the graph.
Node-parallel splits the VERTICES (each device homes N/D rows x ALL F columns, pays boundary comm
for the cut edges). Feature-parallel splits the COLUMNS: each device holds the FULL graph adjacency
+ F/D feature COLUMNS, runs the full-graph SpMM aggregation on its column-slice, then INTEGRATES
(concat the column slices; gather for the dense layer).

WHY IT IS RESULT-PRESERVING (the PROCESS-only invariant -- never accuracy):
  the aggregation h_v = sum_{u in N(v)} x_u is computed INDEPENDENTLY per feature column:
      h_v[c] = sum_{u in N(v)} x_u[c]
  so column-sharding the feature matrix X[:, c0:c1] and running the SAME full-graph SpMM A @ X[:, c0:c1]
  on each device, then CONCATENATING the column slices, yields EXACTLY A @ X. No vertex is split, no
  partial sum crosses a device DURING aggregation (the whole adjacency is local), so the only comm is
  the INTEGRATION (column concat) before the dense layer mixes columns. WHERE a column is reduced never
  changes WHAT is reduced -> bit-identical (fp-epsilon) to the single-device result. See
  fp_aggregate_consistency() for the numeric proof.

COST MODEL (shares arrange.predict_ms's roofline byte constants so the comparison is apples-to-apples):
  NODE-parallel  per device d : compute over its INCIDENT edges x ALL F cols  + boundary comm over the
                                slow link (the arrange.py model -- handled there).
  FEATURE-parallel per device : compute over the FULL graph's edges (every device traverses the whole
                                adjacency) x (F/D) cols  + an INTEGRATION comm (gather the F/D-wide
                                aggregated rows of the boundary/output for the dense layer). Per-device
                                feature MEMORY = full adjacency + (F/D)*N feature bytes -> the HBM relief.

CROSSOVER (the user's question -- depends on F vs graph size + link):
  feature-parallel COMPUTE = D * (full-graph work) * (F/D) = full-graph work * F  -- i.e. the TOTAL
  aggregation FLOPs/bytes are the SAME as one device doing it all (no work saved on compute; it is
  REPLICATED-adjacency, column-split). Its win is MEMORY: each device stores only F/D columns, so an
  attribute-heavy graph that OOMs node-parallel (N/D rows still x ALL F cols can be huge if F huge AND
  the cut replicates) can FIT feature-parallel. Node-parallel DIVIDES the compute (each device only its
  incident edges) but pays boundary comm + homes full-F rows. So:
     low  F  -> node-parallel wins (compute divides D-ways, feature memory is small anyway);
     high F  -> feature-parallel relieves HBM (F/D cols/device) and the integration comm is cheaper than
                node-parallel's full-F boundary comm once F is large enough that feature memory binds.
  This module costs BOTH and the planner picks the lower feasible makespan (HYBRID = a 2D grid is also
  costed: Dn node-groups x Df feature-groups).

PROCESS-only: time / memory / feasibility; accuracy is never a target. NEVER networkx.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .arrange import (
    BYTES_PER_EDGE_TRAVERSAL, N_GATHERS, FEATURE_ROW_BYTES, BYTES_PER_EDGE_RESIDENT,
    node_degree,
)
from ..profiler.cluster_profile import ClusterProfile


# ============================================================================ #
# MATH CONSISTENCY -- the numeric proof that column-shard + concat == single-dev #
# ============================================================================ #
def fp_aggregate_consistency(src, dst, X, splits):
    """Compute the 1-layer SpMM aggregation h = A @ X (sum-of-neighbours, the PER-COLUMN
    independent reduce) two ways and return (h_single, h_featparallel, max_abs_diff):

      single-device     : h = A @ X over ALL F columns at once.
      feature-parallel   : split X's COLUMNS into `splits` contiguous slices, aggregate each slice
                           on its own (the full adjacency is replicated, so each device runs the SAME
                           full-graph SpMM on its column slice), then CONCATENATE the column slices.

    Because h_v[c] = sum_{u in N(v)} X[u, c] is independent across columns c, the concatenation of the
    per-slice results is EXACTLY the single-device result -> max_abs_diff is 0 / fp-epsilon (the
    PROCESS-only invariance: WHERE a column is reduced never changes WHAT is reduced). This is a
    standalone numeric check that the decomposition the planner costs is result-preserving.

    src,dst : int64 [E] edge endpoints (treated UNDIRECTED: both directions aggregated).
    X       : float32/64 [N, F] node feature matrix.
    splits  : list of column counts that sum to F (the per-device column shard sizes).
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    X = np.asarray(X)
    N, F = X.shape
    assert int(sum(splits)) == F, "column splits must sum to F"

    def _agg(Xcols):
        # h[v] = sum over neighbours u of Xcols[u]  (undirected: aggregate along both directions)
        h = np.zeros((N, Xcols.shape[1]), dtype=np.float64)
        np.add.at(h, dst, Xcols[src].astype(np.float64))
        np.add.at(h, src, Xcols[dst].astype(np.float64))
        return h

    h_single = _agg(X)

    # feature-parallel: aggregate each contiguous COLUMN slice independently, then concat
    parts = []
    c0 = 0
    for w in splits:
        c1 = c0 + int(w)
        parts.append(_agg(X[:, c0:c1]))      # device d's full-graph SpMM over its F/D columns
        c0 = c1
    h_fp = np.concatenate(parts, axis=1)     # INTEGRATION: concat the column slices

    max_abs_diff = float(np.abs(h_single - h_fp).max())
    return h_single, h_fp, max_abs_diff


# ============================================================================ #
# FEATURE-PARALLEL cost model (shares arrange.predict_ms's roofline constants)   #
# ============================================================================ #
@dataclass
class FeatureParallelPlan:
    """A feature-parallel (column-shard) layout costed on a heterogeneous cluster.

    Each device holds the FULL graph adjacency + a contiguous slice of F_d feature COLUMNS,
    runs the full-graph SpMM aggregation over those F_d columns, then INTEGRATES (concat the
    column slices + gather the aggregated rows for the dense layer)."""
    name: str
    cols_per_device: np.ndarray       # F_d feature columns on each device (sums to F)
    compute_ms: np.ndarray            # per-device full-graph SpMM over F_d cols (roofline)
    integ_ms: np.ndarray              # per-device integration comm (gather F_d-wide rows over link)
    epoch_ms: np.ndarray              # compute + integration per device
    feat_bytes: np.ndarray            # per-device feature memory = F_d * N * 4
    resident_bytes: np.ndarray        # feat_bytes + FULL adjacency metadata (replicated)
    feasible_mask: np.ndarray         # per-device fits its HBM cap
    makespan_ms: float                # max epoch_ms
    bottleneck: int
    feasible: bool
    integration_ms: float             # total integration comm on the bottleneck (reporting)
    bound: str                        # "compute" | "integration-comm" | "infeasible"


def _columns_by_weight(F: int, weight: np.ndarray) -> np.ndarray:
    """Split F columns across D devices PROPORTIONAL to `weight` (deterministic remainder to the
    heaviest weights). Returns int F_d[D] summing to F. With weight=bandwidth -> TIME-balanced (a
    strong device gets more columns -> balanced full-graph SpMM time). With weight=HBM-capacity ->
    MEMORY-balanced (a big-HBM device holds more columns -> the feature-memory relief: each device's
    F_d*N*4 is sized to its cap). Reduces to an even split when weights are equal."""
    weight = np.asarray(weight, dtype=np.float64)
    D = weight.size
    if F <= 0:
        return np.zeros(D, dtype=np.int64)
    s = weight.sum()
    share = weight / s if s > 0 else np.full(D, 1.0 / D)
    cols = np.maximum(0, np.floor(share * F)).astype(np.int64)
    rem = F - int(cols.sum())
    if rem > 0:
        order = np.argsort(-weight, kind="stable")
        for i in range(rem):
            cols[order[i % D]] += 1
    return cols


def feature_parallel_plan(src, dst, num_nodes: int, cluster: ClusterProfile,
                          F: int, link_gbps: float,
                          boundary_frac: float = 1.0, split: str = "auto",
                          name: str = "feature-parallel") -> FeatureParallelPlan:
    """Cost a FEATURE-PARALLEL (column-shard) layout on `cluster`.

    `split` selects how the F columns are divided across devices:
      "bandwidth" -> proportional to achieved aggregation bandwidth (TIME-balanced; a strong device
                     gets more columns so the full-graph SpMM finishes together). Lowest makespan, but
                     the high-bandwidth device then holds the most feature memory (F_d*N*4).
      "capacity"  -> proportional to HBM capacity (MEMORY-balanced; a big-HBM device holds more
                     columns). Relieves the per-device feature memory -> the FEASIBILITY win.
      "auto" (default) -> use the bandwidth split if it is FEASIBLE; otherwise fall back to the
                     capacity split (so feature-parallel keeps the best makespan when it fits, and
                     trades a little time for feasibility when feature memory binds). This is the
                     attribute-dimension HBM-relief lever.

    Each device d holds the FULL graph (all E edges resident) + F_d feature columns (sum F_d = F).
    The roofline uses the SAME byte constants as arrange.predict_ms so feature- vs node-parallel
    makespans are directly comparable:

      compute_ms[d]  = (2E incident traversals) * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS * F_d
                       / (hbm_bw[d] * 1e9) * 1e3
                       -- the full-graph SpMM over device d's F_d columns. Summed over devices this is
                       the SAME total work as a single device doing all F columns (no compute saved;
                       feature-parallel's win is MEMORY + comm-pattern, not compute division).
      integ_ms[d]    = (boundary_frac * N rows) * F_d * FEATURE_ROW_BYTES * N_GATHERS
                       / (link * 1e9) * 1e3
                       -- the INTEGRATION: each device ships its F_d-wide aggregated output rows so the
                       dense layer (which mixes columns) sees the full-width row. `boundary_frac` (in
                       [0,1]) is the fraction of the N rows that must be gathered across the link (1.0 =
                       all rows integrated; smaller when the dense layer is applied where columns already
                       live). This is the column-concat cost, NOT node-parallel's per-edge boundary comm.

    Per-device MEMORY = F_d * N * 4 (feature columns) + FULL adjacency metadata (E edges resident on
    EVERY device, since the graph is replicated). Feasible iff that fits the device's usable HBM.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    N = int(num_nodes)
    E = int(src.size)
    devs = cluster.devices
    D = cluster.num_devices
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    caps = np.array([d.usable_mem for d in devs], dtype=np.float64)
    link = max(float(link_gbps), 1e-9)
    adj_bytes = E * BYTES_PER_EDGE_RESIDENT          # FULL adjacency replicated on each device

    def _cost(cols):
        # COMPUTE: full-graph SpMM over F_d columns. The aggregation traverses every incident edge
        # (2E undirected traversals), each moving one fp32 word per column per gather -> the roofline
        # divides folded bytes by the device's achieved bandwidth. Width-agnostic, like predict_ms.
        full_traversals = 2.0 * E                    # the FULL graph is resident on every device
        compute_bytes = full_traversals * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS * cols.astype(np.float64)
        compute_ms = compute_bytes / (bw * 1e9) * 1e3
        # INTEGRATION comm: gather the F_d-wide aggregated rows over the link for the dense layer.
        integ_rows = max(0.0, float(boundary_frac)) * N
        integ_bytes = integ_rows * cols.astype(np.float64) * FEATURE_ROW_BYTES * N_GATHERS
        integ_ms = integ_bytes / (link * 1e9) * 1e3
        feat_bytes = cols.astype(np.float64) * N * 4.0
        resident = feat_bytes + adj_bytes
        feasible_mask = resident <= caps
        return compute_ms, integ_ms, feat_bytes, resident, feasible_mask

    cols_bw = _columns_by_weight(F, bw)              # TIME-balanced split
    cols_cap = _columns_by_weight(F, caps)           # MEMORY-balanced split (HBM relief)
    if split == "bandwidth":
        cols = cols_bw
    elif split == "capacity":
        cols = cols_cap
    else:                                            # "auto": bandwidth if feasible, else capacity
        *_, fm_bw = _cost(cols_bw)
        cols = cols_bw if bool(fm_bw.all()) else cols_cap

    compute_ms, integ_ms, feat_bytes, resident, feasible_mask = _cost(cols)
    epoch_ms = compute_ms + integ_ms
    feasible = bool(feasible_mask.all())
    bott = int(np.argmax(epoch_ms)) if D else 0
    makespan = float(epoch_ms.max()) if D else 0.0
    bound = ("infeasible" if not feasible
             else ("integration-comm" if integ_ms[bott] > compute_ms[bott] else "compute"))
    return FeatureParallelPlan(
        name=name, cols_per_device=cols, compute_ms=compute_ms, integ_ms=integ_ms,
        epoch_ms=epoch_ms, feat_bytes=feat_bytes,
        resident_bytes=resident, feasible_mask=feasible_mask,
        makespan_ms=makespan, bottleneck=bott, feasible=feasible,
        integration_ms=float(integ_ms[bott]) if D else 0.0, bound=bound)


# ============================================================================ #
# HYBRID: a 2D grid -- Dn node-groups x Df feature-groups                        #
# ============================================================================ #
@dataclass
class HybridPlan:
    """A hybrid 2D layout: the D devices form a Dn x Df grid; vertices are node-partitioned across
    Dn groups (each group's devices share the same vertex set) and feature COLUMNS are split across
    Df groups. Each device homes (N/Dn rows) x (F/Df cols) + its node-group's incident edges. It pays
    BOTH a node-parallel boundary comm (within a feature-group, over the cut) AND a feature integration
    (column concat across feature-groups). Costed as the worse of the two so the planner can prefer it
    when neither pure axis fits / wins (e.g. F large AND graph large)."""
    name: str
    Dn: int
    Df: int
    compute_ms: np.ndarray
    comm_ms: np.ndarray
    epoch_ms: np.ndarray
    feat_bytes: np.ndarray
    resident_bytes: np.ndarray
    feasible_mask: np.ndarray
    makespan_ms: float
    bottleneck: int
    feasible: bool
    bound: str


def _factor_pairs(D: int):
    """All (Dn, Df) with Dn*Df == D and Dn,Df >= 1 (the candidate grids for the hybrid sweep)."""
    pairs = []
    for dn in range(1, D + 1):
        if D % dn == 0:
            pairs.append((dn, D // dn))
    return pairs


def hybrid_plans(src, dst, num_nodes: int, cluster: ClusterProfile, F: int,
                 link_gbps: float, boundary_frac: float = 1.0):
    """Cost the HYBRID 2D grids (Dn x Df = D). Pure node-parallel is the Df==1 corner and pure
    feature-parallel is the Dn==1 corner of this same grid -- so the hybrid family CONTAINS both
    pure axes; we return the non-degenerate (Dn>1 AND Df>1) grids here and let the planner compare
    them against the pure-axis plans it costs separately. Returns a list[HybridPlan].

    Model (decomposable, sharing the roofline byte constants):
      each device homes N/Dn rows and F/Df columns. Compute = its node-group's incident-edge work
      (~2E/Dn traversals) over F/Df columns. Comm = node-parallel boundary comm (full-F-equivalent,
      but only F/Df wide here) + feature-integration (column concat) -- charged over the link. Memory =
      (N/Dn rows) x (F/Df cols) x 4 + the node-group's resident adjacency (E/Dn edges)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    N = int(num_nodes)
    E = int(src.size)
    devs = cluster.devices
    D = cluster.num_devices
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    caps = np.array([d.usable_mem for d in devs], dtype=np.float64)
    link = max(float(link_gbps), 1e-9)
    deg = node_degree(src, dst, N)
    mean_deg = float(deg.mean()) if N else 0.0

    out = []
    for dn, df in _factor_pairs(D):
        if dn == 1 or df == 1:
            continue                                     # pure corners costed elsewhere
        cols = max(1, F // df)
        rows = max(1, N // dn)
        edges_per_group = 2.0 * E / dn                   # node-group's incident traversals
        # compute over F/Df columns, balanced by bandwidth (use mean bw as the per-device proxy)
        compute_bytes = edges_per_group * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS * cols
        compute_ms = compute_bytes / (bw * 1e9) * 1e3
        # comm: node-parallel boundary (a fraction of rows crossing the cut, F/Df wide) + integration
        boundary_rows = boundary_frac * rows
        comm_bytes = boundary_rows * cols * FEATURE_ROW_BYTES * N_GATHERS
        comm_ms = np.full(D, comm_bytes / (link * 1e9) * 1e3, dtype=np.float64)
        epoch_ms = compute_ms + comm_ms
        feat_bytes = np.full(D, rows * cols * 4.0, dtype=np.float64)
        adj_bytes = (E / dn) * BYTES_PER_EDGE_RESIDENT
        resident = feat_bytes + adj_bytes
        feasible_mask = resident <= caps
        feasible = bool(feasible_mask.all())
        bott = int(np.argmax(epoch_ms))
        bound = ("infeasible" if not feasible
                 else ("comm" if comm_ms[bott] > compute_ms[bott] else "compute"))
        out.append(HybridPlan(
            name=f"hybrid({dn}x{df})", Dn=dn, Df=df, compute_ms=compute_ms, comm_ms=comm_ms,
            epoch_ms=epoch_ms, feat_bytes=feat_bytes, resident_bytes=resident,
            feasible_mask=feasible_mask, makespan_ms=float(epoch_ms.max()), bottleneck=bott,
            feasible=feasible, bound=bound))
    return out
