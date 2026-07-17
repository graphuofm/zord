"""ZORD ARRANGE -- the adaptive-corner partition-and-place that is worst-case-optimal
by construction (the validated win from scripts/sota_compare.py:zord_arrange, ported into
the engine). PROCESS-only: same graph + same model => same result; we optimize TIME /
MEMORY / FEASIBILITY, never accuracy (the placement is a result-preserving GAS reduce --
WHERE a partial sum is computed never changes WHAT is computed).

Given a temporal graph + a heterogeneous cluster (per-device HBM capacity, ACHIEVED
aggregation bandwidth, and an INTERCONNECT BANDWIDTH that is a PARAMETER -- zord must win
on the algorithm at ANY comm speed, so NVLink is never hardcoded), arrange() evaluates a
family of candidate plans and emits the one with the lowest PREDICTED MAKESPAN among the
feasible, non-degenerate ones:

  1. edge-cut (hetero-matched) : cluster-respecting LPA order split so per-device incident-
     edge aggregation TIME is balanced (work share ~ device HBM bandwidth -> strong device
     gets the dense core, the straggler is removed). (hetero_matched.py)
  2. dense-core vertex-cut     : replicate the k-core dense core PowerGraph-style, edge-cut
     the periphery -- with a NO-BUDGET sweep over coreness quantiles, gated ONLY by HBM
     feasibility, keeping the makespan-best core size. (vertexcut, §23/§26)
  3. spatial (PSS) corner      : equal-work LPA edge-cut (the balance-blind reference corner).
  4. temporal (PTS) corner     : balanced timeline split by first-activity time (duality end).
  5. METIS (min-cut) floor     : adopt the SOTA cut-minimizer as a candidate so zord <= METIS
     BY CONSTRUCTION; the adaptive pick selects it on cut-sensitive / slow-link graphs and
     only deviates when a lever provably lowers makespan. Skipped gracefully if pymetis absent.

The adaptive pick guarantees the worst-case-optimal property: zord <= min(candidates).
The floor is the exact METIS min-cut WHEN E <= METIS_MAX_EDGES (the literal "zord <= METIS");
above that size-gate METIS is skipped (it is superlinear -- D44) and the floor becomes the
cheap balanced lpa-proxy, so the guarantee weakens to "zord <= cheap-balanced-proxy" there.
All candidates share ONE decomposable incident-edge roofline cost (predict_ms), so the
comparison is apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import cpp_kernel
from ..profiler.cluster_profile import ClusterProfile

# incident-edge roofline work model (IDENTICAL to the validated scripts) --------------
BYTES_PER_EDGE_TRAVERSAL = 4.0   # one fp32 feature word moved per edge per gather (mem-bound agg)
N_GATHERS = 2                    # 2-layer aggregation -> 2 SpMM gathers over local incident edges
FEATURE_ROW_BYTES = 4.0          # fp32 word per feature dim per boundary/partial row in comm
BYTES_PER_EDGE_RESIDENT = 20.0   # src+dst+ts+w edge metadata resident on a device (cost_model default)

# ---- SCALE GATES (preserve small/medium decisions; only engage on BIG graphs) --------
# pymetis is a MULTILEVEL min-cut: superlinear in edges, the prime cause of the 49-min hang
# on 100M edges. Above this many edges we SKIP the pymetis call and substitute a CHEAP O(M)
# balanced-min-cut proxy (the lpa-order contiguous split -- locality-aware, so a reasonable
# cut floor). Below it METIS is still computed -> the literal exact "zord <= METIS" floor is
# UNCHANGED. Override per-call via arrange(..., metis_max_edges=...).
METIS_MAX_EDGES = 20_000_000
# The vertex-cut coreness sweep recomputes O(M) metrics PER quantile. Below this edge count
# the FULL 9-quantile sweep runs (decisions bit-identical to the validated scripts); above it
# we thin the sweep to a coarse subset so the planner stays well under a minute at 100M edges.
VERTEXCUT_FULL_SWEEP_MAX_EDGES = 5_000_000
# The full sweep (preserved verbatim for small/medium graphs).
_CORENESS_QUANTILES_FULL = (0.70, 0.80, 0.88, 0.93, 0.96, 0.98, 0.99, 0.995, 0.999)
# Thinned sweep at scale: a coarse subset spanning the same range (still feasibility-gated).
# Keeps the makespan-best core AMONG THE COARSE SUBSET -- may miss the full-sweep optimum by
# <~1% makespan above the gate (re-audit measured worst-case 0.61%): an honest speed/quality
# trade. 5 evals instead of 9 -> ~45% fewer O(M) passes.
_CORENESS_QUANTILES_COARSE = (0.80, 0.93, 0.98, 0.99, 0.999)


# ============================================================================ #
# the shared incident-edge work model + makespan (link bw is a PARAMETER)       #
# ============================================================================ #
def predict_ms(incident_edges: np.ndarray, comm_rows: np.ndarray,
               bw_gbps: np.ndarray, link_gbps: float):
    """Per-device step ms = COMPUTE (roofline gather over local incident edges, reads fast
    per-device HBM at bw_gbps[k]) + COMM (boundary feature rows exchanged across the SLOW
    interconnect `link_gbps` each layer). F is folded into incident_edges/comm_rows by the
    caller via the byte constants. Returns (total[D], comp[D], comm[D]). `link_gbps` is the
    interconnect-bandwidth PARAMETER -- nothing about NVLink is hardcoded.

    F_v GENERALIZATION (purely additive): the caller folds feature WIDTH into the two work
    vectors. For a SCALAR F it passes `incident_edges = incident*F` (a constant width per row);
    for a per-node feat_bytes VECTOR F_v it passes the FEATURE-WEIGHTED incident/comm work
    (sum over the edges/rows of the ACTUAL neighbor/row width F_v). The roofline formula is
    width-agnostic -- it just divides folded work by bandwidth -- so predict_ms is UNCHANGED
    and bit-identical when F_v is uniform (inc*F == feature-weighted-inc with F_v==F)."""
    bw_gbps = np.asarray(bw_gbps, dtype=np.float64)
    comp_ms = incident_edges * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS / (bw_gbps * 1e9) * 1e3
    link = max(link_gbps, 1e-9)
    comm_ms = comm_rows * FEATURE_ROW_BYTES * N_GATHERS / (link * 1e9) * 1e3
    return comp_ms + comm_ms, comp_ms, comm_ms


def feasible(counts: np.ndarray, incident: np.ndarray, devs, F: int,
             feat_bytes: Optional[np.ndarray] = None) -> bool:
    """A device fits if node-rows + resident edge metadata < its usable HBM. Same footprint
    shape as cost_model.device_footprint_bytes (feat + edge mem).

    feat_bytes : OPTIONAL per-device feature-memory vector [D] = sum_{v on k} F_v * 4 (the
                 ACTUAL feature bytes when per-node sizes are heterogeneous). When None ->
                 charge counts[k]*F*4 EXACTLY as today (uniform scalar F; bit-identical). When
                 given -> charge the real per-node feature bytes, so a device that homes the
                 high-F (multi-modal) rows is sized by its TRUE feature memory, not count*meanF.
                 The §33 placement-feasibility win lives here."""
    for k, d in enumerate(devs):
        feat = (counts[k] * F * 4.0) if feat_bytes is None else float(feat_bytes[k])
        edgemem = incident[k] * BYTES_PER_EDGE_RESIDENT
        if feat + edgemem > d.usable_mem:
            return False
    return True


# ============================================================================ #
# basic structural metrics                                                      #
# ============================================================================ #
def node_degree(src, dst, N):
    return cpp_kernel.node_degree(src, dst, N)


def edgecut_metrics(src, dst, deg, dev, D, N):
    """Single-home assignment dev[v] in [0,D): edge-cut, per-device incident-edge work,
    per-device COMM rows (distinct (gathering-device, remote-neighbor) pairs), node counts."""
    pu = dev[src]; pv = dev[dst]
    cut = int(np.count_nonzero(pu != pv))
    incident = np.bincount(dev, weights=deg.astype(np.float64), minlength=D)
    counts = np.bincount(dev, minlength=D).astype(np.int64)
    a = np.concatenate([src, dst]); b = np.concatenate([dst, src])
    da = dev[a]; cross = da != dev[b]
    if cross.any():
        keyc = np.unique(da[cross].astype(np.int64) * np.int64(N) + b[cross])
        comm_rows = np.bincount((keyc // N).astype(np.int64), minlength=D).astype(np.float64)
    else:
        comm_rows = np.zeros(D, dtype=np.float64)
    return cut, incident, comm_rows, counts


def replicate_core_metrics(src, dst, deg, core_mask, dev_periphery, D, N, rng, uv=None,
                           return_land=False):
    """VERTEX-CUT: core nodes replicated onto all D devices; periphery uses dev_periphery
    (single-home, -1 for core). Incident gather split by NEIGHBOR OWNERSHIP; core-internal
    edges round-robin. Cut counts only periphery-periphery cross edges. Returns
    (cut, incident[D], comm_rows[D], counts[D], extra_core_rows).

    `uv` is an OPTIONAL precomputed (u, v) = (concat[src,dst], concat[dst,src]) pair; the
    coreness sweep passes it so the doubled-edge arrays are built ONCE instead of per quantile
    (pure hoist -- the metrics are bit-identical with or without it).
    `return_land` (additive): also return the per-(doubled-)edge landing device `land` so the
    F_v feature-weighted fold reuses the EXACT same core-core random landing (the rng was already
    consumed here) instead of re-rolling it -> the uniform-F_v fold is bit-identical to inc*F."""
    dev = dev_periphery
    if uv is None:
        u = np.concatenate([src, dst]); v = np.concatenate([dst, src])
    else:
        u, v = uv
    u_core = core_mask[u]; v_core = core_mask[v]
    du = dev[u]; dv = dev[v]
    land = np.empty(u.size, dtype=np.int64)
    mA = ~u_core;            land[mA] = du[mA]
    mB = u_core & (~v_core); land[mB] = dv[mB]
    mC = u_core & v_core
    nC = int(mC.sum())
    if nC:
        land[mC] = rng.integers(0, D, size=nC)
    incident = np.bincount(land, minlength=D).astype(np.float64)

    cs = core_mask[src]; cd = core_mask[dst]
    pp = (~cs) & (~cd)
    cut = int(np.count_nonzero((dev[src] != dev[dst]) & pp))
    core_size = int(core_mask.sum())
    extra_core_rows = core_size * (D - 1)
    reduce_partials = np.full(D, core_size * (D - 1) / D, dtype=np.float64)

    pp_inc = (~u_core) & (~v_core)
    cross = pp_inc & (du != dv) & (du >= 0) & (dv >= 0)
    if cross.any():
        keyc = np.unique(du[cross].astype(np.int64) * np.int64(N) + v[cross])
        comm_rows = np.bincount((keyc // N).astype(np.int64), minlength=D).astype(np.float64)
    else:
        comm_rows = np.zeros(D, dtype=np.float64)
    comm_rows = comm_rows + reduce_partials

    periphery = np.nonzero(~core_mask)[0]
    counts = np.bincount(dev[periphery], minlength=D).astype(np.int64) + core_size
    if return_land:
        return cut, incident, comm_rows, counts, extra_core_rows, land
    return cut, incident, comm_rows, counts, extra_core_rows


# ============================================================================ #
# PER-NODE feature-byte (F_v) work folding -- the §33 attribute generalization   #
# ============================================================================ #
# The scalar path folds a CONSTANT width F into the work: comp ~ incident[k]*F (each gathered
# row is F wide), comm ~ comm_rows[k]*F (each exchanged boundary row is F wide), feature memory
# ~ counts[k]*F*4. With a per-node feat_bytes VECTOR F_v those widths are HETEROGENEOUS: the
# gather over an edge a<-b moves the NEIGHBOR row of width F_v[b]; a boundary comm row for
# remote neighbor b is F_v[b] wide; a device's feature memory is sum_{v on k} F_v[v]*4. The
# folds below compute EXACTLY those feature-weighted per-device vectors. The FOLDS THEMSELVES are
# BIT-IDENTICAL to the scalar folds when F_v is uniform (sum of F over deg(k) edges==incident[k]*F).
# HONEST CAVEAT (auditor a12ff3): bit-identity to PRIOR RESULTS holds for the SCALAR path only
# (feat_bytes=None -> aware=False -> these folds are not even reached). A uniform F_v VECTOR is NOT
# guaranteed to reproduce the scalar PLAN, because arrange() in aware mode ADDS an extra
# edge-cut(feat-aware) candidate (capacity-split, not bandwidth-split) that can win the makespan and
# flip the winner. So: §30/§32-v2/§29 (all call with feat_bytes=None) are UNAFFECTED; F_v mode is a
# strictly new code path that may pick a different (capacity-matched) plan, by design.
def _feat_bytes_per_dev(dev, Fv, D, core_mask=None):
    """Per-device feature MEMORY bytes = sum_{v homed on k} F_v[v]*4, PLUS the replicated core
    (vertex-cut): the core rows live on EVERY device, so each device pays sum_{c in core} F_v[c]*4.
    Reduces to counts[k]*F*4 when F_v is uniform (incl. the core term -> core_size*F*4 on each)."""
    Fv = np.asarray(Fv, dtype=np.float64)
    if core_mask is None:
        per = np.bincount(dev, weights=Fv * 4.0, minlength=D).astype(np.float64)
    else:
        periphery = ~core_mask
        per = np.bincount(dev[periphery], weights=(Fv * 4.0)[periphery],
                          minlength=D).astype(np.float64)
        per = per + float((Fv[core_mask] * 4.0).sum())     # replicated core on each device
    return per


def _edgecut_feat_work(src, dst, dev, Fv, D, N):
    """Feature-WEIGHTED incident + comm work for a single-home edge-cut assignment. The gather
    of edge a<-b lands on dev[a] and moves the NEIGHBOR row of width F_v[b] (so incident work is
    sum of F_v[neighbor] over edges landing on k). Comm charges F_v[remote-neighbor] for each
    DISTINCT (gathering-device, remote-neighbor) pair. Reduces to (incident*F, comm_rows*F) when
    F_v==F. Returns (inc_fw[D], comm_fw[D])."""
    Fv = np.asarray(Fv, dtype=np.float64)
    a = np.concatenate([src, dst]); b = np.concatenate([dst, src])
    da = dev[a]
    inc_fw = np.bincount(da, weights=Fv[b], minlength=D).astype(np.float64)
    cross = da != dev[b]
    if cross.any():
        keyc = np.unique(da[cross].astype(np.int64) * np.int64(N) + b[cross])
        kd = (keyc // N).astype(np.int64)
        kb = (keyc % N).astype(np.int64)
        comm_fw = np.bincount(kd, weights=Fv[kb], minlength=D).astype(np.float64)
    else:
        comm_fw = np.zeros(D, dtype=np.float64)
    return inc_fw, comm_fw


def _replicate_feat_work(dev_periphery, core_mask, Fv, D, N, uv, land):
    """Feature-WEIGHTED incident + comm work for the VERTEX-CUT layout. It REUSES the exact
    per-edge landing `land` produced by replicate_core_metrics (so the core-core random landing
    is the SAME -- no rng re-roll), weighting each landed gather by the NEIGHBOR row width F_v[v]
    and each comm row by F_v[remote-neighbor]. The reduce-partials term (core rows reduced across
    devices) is feature-weighted too: core_feat_total*(D-1)/D per device, where
    core_feat_total = sum_{c in core} F_v[c]. Reduces EXACTLY to (incident*F, comm_rows*F) when
    F_v==F (verified). Returns (inc_fw[D], comm_fw[D])."""
    Fv = np.asarray(Fv, dtype=np.float64)
    dev = dev_periphery
    u, v = uv
    inc_fw = np.bincount(land, weights=Fv[v], minlength=D).astype(np.float64)

    u_core = core_mask[u]; v_core = core_mask[v]
    du = dev[u]; dv = dev[v]
    core_feat_total = float(Fv[core_mask].sum())
    reduce_partials = np.full(D, core_feat_total * (D - 1) / D, dtype=np.float64)
    pp_inc = (~u_core) & (~v_core)
    cross = pp_inc & (du != dv) & (du >= 0) & (dv >= 0)
    if cross.any():
        keyc = np.unique(du[cross].astype(np.int64) * np.int64(N) + v[cross])
        kd = (keyc // N).astype(np.int64)
        kv = (keyc % N).astype(np.int64)
        comm_fw = np.bincount(kd, weights=Fv[kv], minlength=D).astype(np.float64)
    else:
        comm_fw = np.zeros(D, dtype=np.float64)
    comm_fw = comm_fw + reduce_partials
    return inc_fw, comm_fw


# ============================================================================ #
# layout primitives (cluster-respecting via lpa rank)                           #
# ============================================================================ #
def _split_by_work(seq, weight, D):
    cum = np.cumsum(weight)
    if cum[-1] <= 0:
        return (np.arange(seq.size) * D // max(1, seq.size)).clip(0, D - 1)
    targets = np.arange(1, D) * cum[-1] / D
    cuts = np.searchsorted(cum, targets, side="left")
    bounds = np.concatenate([[0], cuts, [seq.size]]).astype(np.int64)
    return (np.searchsorted(bounds, np.arange(seq.size), side="right") - 1).clip(0, D - 1)


def lpa_edgecut(N, deg, lpa_rank, D, caps=None):
    """Cluster-respecting edge-cut: walk the lpa cluster-grouped layout in rank order, cut
    into D contiguous segments of equal incident-edge work (or capacity-proportional work
    when caps given -> the hetero-matched split). Returns dev[v] in [0,D)."""
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
    """PTS corner: keep each vertex's whole timeline on one device; split vertices into D
    balanced cohorts by FIRST-ACTIVITY time. Spatial edges cut (PTS cost), timeline local."""
    first = np.full(N, S + 1, dtype=np.int64)
    np.minimum.at(first, src, snap)
    np.minimum.at(first, dst, snap)
    seq = np.argsort(first, kind="stable")
    seg = _split_by_work(seq, deg[seq].astype(np.float64), D)
    dev = np.empty(N, dtype=np.int64)
    dev[seq] = seg
    return dev


def balanced_periphery(periphery, deg, D, lpa_rank=None):
    """Edge-cut the periphery (non-core) nodes finely + balanced across D devices,
    cluster-respecting via lpa rank when available. -1 for core nodes."""
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


def metis_partition(src, dst, N, D):
    """METIS balanced min-cut via pymetis (raises -> caller SKIPs if pymetis missing)."""
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
    try:
        adj = pymetis.CSRAdjacency(indptr.tolist(), adjncy.tolist())
        _, membership = pymetis.part_graph(D, adjacency=adj)
    except Exception:
        _, membership = pymetis.part_graph(D, xadj=indptr.tolist(), adjncy=adjncy.tolist())
    return np.asarray(membership, dtype=np.int64)


# ============================================================================ #
# the ARRANGE result + the adaptive-corner pick                                 #
# ============================================================================ #
@dataclass
class ArrangeResult:
    name: str                         # winning candidate name
    assignment: np.ndarray            # int32 [N] -> device (periphery home for vertex-cut)
    core_mask: Optional[np.ndarray]   # bool [N] replicated core (vertex-cut), else None
    cut: int
    incident: np.ndarray              # incident-edge work per device
    comm_rows: np.ndarray             # boundary feature rows per device
    counts: np.ndarray                # node counts per device (incl. replicated core)
    extra_core_rows: int              # replicated rows (vertex-cut)
    makespan_ms: float                # predicted makespan of the winning candidate
    candidate_makespans: dict         # {name: makespan_ms} -- honest per-candidate reporting
    bw_gbps: np.ndarray = field(default=None)
    link_gbps: float = 0.0
    # F_v-aware extras (None in the scalar path -> the planner falls back to incident*F /
    # counts*F*4 exactly as before, so the scalar plan is byte-identical). When the winner was
    # costed under a per-node feat_bytes vector these carry the WINNER's feature-weighted folded
    # work and ACTUAL per-device feature bytes so the planner reports them without recomputing.
    inc_folded: Optional[np.ndarray] = field(default=None)   # feature-weighted incident work [D]
    comm_folded: Optional[np.ndarray] = field(default=None)  # feature-weighted comm work [D]
    feat_bytes_dev: Optional[np.ndarray] = field(default=None)  # actual feature bytes per dev [D]

    @property
    def replication_pct(self) -> float:
        return self.extra_core_rows / max(1, self.counts.sum()) * 100.0


def arrange(src, dst, num_nodes, cluster: ClusterProfile,
            link_gbps: Optional[float] = None, feat_dim: int = 128,
            num_snapshots: int = 64, snap: Optional[np.ndarray] = None,
            seed: int = 0, metis_max_edges: int = METIS_MAX_EDGES,
            feat_bytes: Optional[np.ndarray] = None) -> ArrangeResult:
    """Run the adaptive-corner arrange and return the winning plan.

    link_gbps : interconnect bandwidth PARAMETER (GB/s). If None, taken from the cluster's
                inter_node_bw (the slow link). zord must win on the algorithm at ANY value.
    feat_dim  : node feature dimension F (folded into the byte costs).
    snap      : per-edge snapshot id in [0, num_snapshots) (for the PTS corner); if None,
                derived as equal-count snapshots over the (time-sorted) edge stream.
    metis_max_edges : SIZE GATE for the (superlinear) pymetis multilevel min-cut. With
                num_edges <= this, METIS is computed -> the literal exact "zord <= METIS"
                floor (UNCHANGED small/medium behavior). Above it, the pymetis call is SKIPPED
                and a CHEAP O(M) balanced-min-cut proxy (the lpa-order contiguous split) takes
                its place as the floor; "zord <= cheap-balanced-proxy" replaces the literal
                METIS floor, while zord <= min(the OTHER candidates) still holds by construction.
    feat_bytes : OPTIONAL per-node feature-size VECTOR F_v (numpy [N], dims or bytes per node;
                interpreted as feature DIMS, the same unit as feat_dim). When None -> the SCALAR
                feat_dim path runs UNCHANGED (bit-identical, the change is purely additive). When
                given -> feasibility and makespan use the ACTUAL per-node feature bytes
                (sum_{v on k} F_v, NOT counts*meanF), so zord routes feature-heavy nodes to
                high-HBM / high-bandwidth devices (the §33 attribute placement+feasibility win)
                and an extra F_v-capacity-matched candidate is evaluated. F_v need not match
                feat_dim. NOTE (auditor a12ff3): only the SCALAR path (feat_bytes=None) is
                bit-identical to prior results; a uniform F_v VECTOR can still flip the winner via
                the added edge-cut(feat-aware) candidate -> F_v mode is a deliberately new plan path.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    N = int(num_nodes)
    E = int(src.size)
    D = cluster.num_devices
    F = feat_dim
    S = int(num_snapshots)
    devs = cluster.devices
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    cap = np.array([d.usable_mem for d in devs], dtype=np.float64)
    link = float(link_gbps) if link_gbps is not None else float(cluster.inter_node_bw)
    caps_work = bw.copy()                       # hetero work budget ~ achieved HBM bandwidth
    rng = np.random.default_rng(seed + 1)
    Fv = None if feat_bytes is None else np.asarray(feat_bytes, dtype=np.float64)
    aware = Fv is not None

    if snap is None:
        snap = np.minimum((np.arange(E) * S // max(1, E)).astype(np.int64), S - 1)

    deg = node_degree(src, dst, N)
    lpa_rank = cpp_kernel.lpa_rank(N, src, dst)
    core_val = cpp_kernel.coreness(src, dst, N)

    cand = {}          # name -> (cut, incident, comm_rows, counts, extra, assignment, core_mask)

    # ---- candidate 1: hetero edge-cut (work share ~ HBM bandwidth) ----
    dev1 = lpa_edgecut(N, deg, lpa_rank, D, caps=caps_work)
    cut1, inc1, comm1, cnt1 = edgecut_metrics(src, dst, deg, dev1, D, N)
    cand["edge-cut(hetero)"] = (cut1, inc1, comm1, cnt1, 0, dev1, None)

    # ---- candidate 1b (F_v-aware ONLY): edge-cut whose contiguous segments are sized so the
    #      per-device FEATURE BYTES are CAPACITY-PROPORTIONAL -> heavy-F mass flows to the big-HBM
    #      device (the §33 placement lever). The layout is the SAME lpa locality order; only the
    #      balance WEIGHT (F_v, not deg) and the segment targets (per-device HBM cap) differ. Added
    #      only when F_v is given, so the scalar candidate set is untouched (purely additive). ----
    if aware:
        dev1b = lpa_edgecut(N, Fv, lpa_rank, D, caps=cap)
        cut1b, inc1b, comm1b, cnt1b = edgecut_metrics(src, dst, deg, dev1b, D, N)
        cand["edge-cut(feat-aware)"] = (cut1b, inc1b, comm1b, cnt1b, 0, dev1b, None)

    # doubled-edge arrays built ONCE (reused across all quantiles -- pure hoist, the per-quantile
    # metrics are bit-identical) and by the F_v folds. Hoisted above the closures that close over it.
    uv = (np.concatenate([src, dst]), np.concatenate([dst, src]))
    cand_land = {}     # name -> per-(doubled-)edge landing for the vertex-cut (F_v fold reuses it)

    # F_v folds (closures). When NOT aware these are EXACTLY the scalar folds inc*F / comm*F and
    # the scalar feasible(...,F) -> the whole F_v path is dead code, so the plan is bit-identical.
    # When aware they substitute the FEATURE-WEIGHTED work and ACTUAL per-device feature bytes.
    # For the vertex-cut the fold REUSES the candidate's stored `land` (same core-core landing) so
    # the uniform-F_v fold is bit-identical to inc*F (no rng re-roll).
    def _folded(name, assign, inc, comm, cmask):
        if not aware:
            return inc * F, comm * F
        if cmask is None:
            return _edgecut_feat_work(src, dst, assign, Fv, D, N)
        return _replicate_feat_work(assign, cmask, Fv, D, N, uv, cand_land[name])

    def _feasible(assign, inc, cnt, cmask):
        if not aware:
            return feasible(cnt, inc, devs, F)
        fb = _feat_bytes_per_dev(assign, Fv, D, core_mask=cmask)
        return feasible(cnt, inc, devs, F, feat_bytes=fb)

    # ---- candidate 2: dense-core vertex-cut, NO-budget coreness sweep (feasibility-only) ----
    # SCALE: thin the sweep above the gate. Below the gate the FULL 9-quantile sweep runs on the
    # same `uv` -> decisions UNCHANGED vs the old loop.
    quantiles = (_CORENESS_QUANTILES_FULL if E <= VERTEXCUT_FULL_SWEEP_MAX_EDGES
                 else _CORENESS_QUANTILES_COARSE)
    best_vc = None
    for q in quantiles:
        tau = max(2, int(np.quantile(core_val, q)))
        core_mask = core_val >= tau
        cs = int(core_mask.sum())
        if not (0 < cs < N):
            continue
        dev_p = balanced_periphery(np.nonzero(~core_mask)[0], deg, D, lpa_rank=lpa_rank)
        cut2, inc2, comm2, cnt2, extra2, land2 = replicate_core_metrics(
            src, dst, deg, core_mask, dev_p, D, N, rng, uv=uv, return_land=True)
        if not _feasible(dev_p, inc2, cnt2, core_mask):  # HBM-feasibility is the ONLY hard constraint
            continue
        # vertex-cut fold needs the EXACT landing for this quantile -> reuse land2 (no rng re-roll).
        cand_land["vertex-cut(k-core)"] = land2
        inc2_f, comm2_f = _folded("vertex-cut(k-core)", dev_p, inc2, comm2, core_mask)
        tot, _, _ = predict_ms(inc2_f, comm2_f, bw, link)
        mk = float(tot.max())
        if best_vc is None or mk < best_vc[0]:
            best_vc = (mk, (cut2, inc2, comm2, cnt2, extra2, dev_p, core_mask), land2)
    if best_vc is not None:
        cand["vertex-cut(k-core)"] = best_vc[1]
        cand_land["vertex-cut(k-core)"] = best_vc[2]   # the WINNING quantile's landing

    # ---- candidates 3/4: spatial (PSS) and temporal (PTS) corners ----
    dev3 = lpa_edgecut(N, deg, lpa_rank, D, caps=None)
    cut3, inc3, comm3, cnt3 = edgecut_metrics(src, dst, deg, dev3, D, N)
    cand["spatial(PSS)"] = (cut3, inc3, comm3, cnt3, 0, dev3, None)

    dev4 = temporal_partition(src, dst, snap, deg, N, S, D)
    cut4, inc4, comm4, cnt4 = edgecut_metrics(src, dst, deg, dev4, D, N)
    cand["temporal(PTS)"] = (cut4, inc4, comm4, cnt4, 0, dev4, None)

    # ---- candidate 5: min-cut FLOOR (zord <= floor by construction) ----
    # SIZE GATE: pymetis is a superlinear multilevel min-cut (the 49-min hang at 100M edges).
    # At/below metis_max_edges we compute the EXACT METIS floor (small/medium UNCHANGED). Above
    # it we SKIP pymetis and substitute a CHEAP O(M) balanced-min-cut proxy -- the lpa-order
    # contiguous EQUAL-work split (locality-aware, already computed as the PSS corner `dev3`).
    # Both register as the floor name `floor_name` and stay BALANCE-GATE-EXEMPT below (D41), so
    # the worst-case-optimal property holds: zord <= this floor, hence <= the other candidates.
    if E <= metis_max_edges:
        try:
            dev5 = metis_partition(src, dst, N, D)
            cut5, inc5, comm5, cnt5 = edgecut_metrics(src, dst, deg, dev5, D, N)
            cand["metis(min-cut)"] = (cut5, inc5, comm5, cnt5, 0, dev5, None)
        except Exception:
            pass
    else:
        # cheap balanced-min-cut proxy == the equal-work lpa contiguous split (PSS corner reused,
        # zero extra O(M) work). Named distinctly so reporting is honest about the substitution.
        cand["cheap-cut(lpa-proxy)"] = (cut3, inc3, comm3, cnt3, 0, dev3, None)

    # ---- adaptive pick: lowest predicted makespan among feasible + non-degenerate plans ----
    cand_mk = {}
    eligible = []
    bal_gate = 0.5 * D + 0.5
    for name, (cut, inc, comm, cnt, extra, assign, cmask) in cand.items():
        inc_f, comm_f = _folded(name, assign, inc, comm, cmask)
        tot, _, _ = predict_ms(inc_f, comm_f, bw, link)
        cand_mk[name] = float(tot.max())
        work_imb = float(inc.max() / max(1e-9, inc.mean()))
        # The min-cut FLOOR (exact METIS below the gate, or the cheap lpa proxy above it) is the
        # worst-case floor: keep it eligible whenever feasible -- even if its cut-minimizing
        # partition is degree-imbalanced (high incident work_imb) -- so that zord <= floor holds
        # BY CONSTRUCTION (else the gate could drop a floor that is actually the fastest plan,
        # breaking the guarantee; auditor 2026-05-31, D41). The balance gate still filters zord's
        # OWN candidates against a hidden-imbalance pick in the comm-bound regime.
        is_floor = name in ("metis(min-cut)", "cheap-cut(lpa-proxy)")
        if _feasible(assign, inc, cnt, cmask) and (work_imb <= bal_gate or is_floor):
            eligible.append(name)
    if not eligible:
        eligible = [min(cand, key=lambda n: cand[n][1].max() / max(1e-9, cand[n][1].mean()))]
    best_name = min(eligible, key=lambda n: cand_mk[n])
    cut, inc, comm, cnt, extra, assign, cmask = cand[best_name]
    if aware:
        inc_fold, comm_fold = _folded(best_name, assign, inc, comm, cmask)
        fb_dev = _feat_bytes_per_dev(assign, Fv, D, core_mask=cmask)
    else:
        inc_fold = comm_fold = fb_dev = None
    return ArrangeResult(
        name=best_name, assignment=assign.astype(np.int32), core_mask=cmask,
        cut=cut, incident=inc, comm_rows=comm, counts=cnt, extra_core_rows=extra,
        makespan_ms=cand_mk[best_name], candidate_makespans=cand_mk,
        bw_gbps=bw, link_gbps=link,
        inc_folded=inc_fold, comm_folded=comm_fold, feat_bytes_dev=fb_dev)


# ============================================================================ #
# C++ FAST PATH -- the whole adaptive-corner arrange in build/arrange (the      #
# performance core; numpy arrange() above is the reference + fallback).         #
# ============================================================================ #
import os as _os
import struct as _struct
import subprocess as _subprocess
import tempfile as _tempfile
from pathlib import Path as _Path


@dataclass
class AxisChoice:
    """The cost-EXACT decomposition-axis pick (BACKLOG D2, the honest replacement for the 62%
    partition_axes heuristic scorecard). We do NOT guess the axis from cheap stats; we COST each
    axis with the SAME roofline (structure/time exactly via arrange; feature/hybrid via the
    attr_cost derived decomposition model -- same byte constants, so apples-to-apples) and return
    the minimum-makespan axis. Exact under the cost model."""
    axis: str                         # "structure" | "time" | "feature" | "hybrid"
    makespan_ms: float
    costs: dict                       # {axis -> makespan_ms}
    arrange: Optional["ArrangeResult"] = None   # the structural plan (when structure/time wins)
    note: str = ""


def choose_axis(src, dst, num_nodes, cluster: ClusterProfile,
                link_gbps: Optional[float] = None, feat_dim: int = 128,
                num_snapshots: int = 64, snap: Optional[np.ndarray] = None,
                feat_bytes: Optional[np.ndarray] = None, layers: int = 2) -> AxisChoice:
    """Pick the decomposition axis by COSTING every axis (not guessing):
      structure/time -> the exact min over arrange's structural corners (edge-cut, dense-core
                        vertex-cut, PSS spatial, PTS temporal, the multilevel/metis min-cut floor);
                        arrange already separates the spatial(PSS) vs temporal(PTS) corner, so the
                        structure-vs-time decision is made exactly here.
      feature/hybrid -> the attr_cost derived feature-parallel + integration cost (and a sqrt(D)
                        hybrid), on the SAME byte roofline.
    Returns the argmin axis + per-axis makespans. This is the cost-exact D2 selector that
    SUPERSEDES the partition_axes 62% scorecard (which guessed from cheap stats without costing)."""
    from .attr_cost import feature_parallel_cost_ms, integration_cost_ms, node_parallel_cost_ms
    import math
    r = arrange_cpp(src, dst, num_nodes, cluster, link_gbps, feat_dim, num_snapshots, snap, feat_bytes=feat_bytes)
    N = int(num_nodes); E = int(np.asarray(src).size); D = cluster.num_devices
    bw = float(np.mean([d.hbm_bw_gbps for d in cluster.devices]))
    link = float(link_gbps) if link_gbps is not None else float(cluster.inter_node_bw)
    deg_avg = 2.0 * E / max(1, N)
    # arrange's winner is a structural corner; PTS == the time axis, everything else == structure.
    struct_axis = "time" if r.name.startswith("temporal") else "structure"
    feat_ms = feature_parallel_cost_ms(N, E, feat_dim, D, bw, link, layers) + integration_cost_ms(N, feat_dim, link, layers)
    Dh = max(1, int(round(math.sqrt(D))))                       # hybrid: ~sqrt(D) node x sqrt(D) feature
    hyb_node = node_parallel_cost_ms(E, feat_dim, deg_avg, Dh, bw, link, layers)
    hyb_feat = feature_parallel_cost_ms(N, E, feat_dim // max(1, Dh), Dh, bw, link, layers)
    hybrid_ms = max(hyb_node, hyb_feat) + integration_cost_ms(N, feat_dim // max(1, Dh), link, layers)
    costs = {struct_axis: float(r.makespan_ms), "feature": float(feat_ms), "hybrid": float(hybrid_ms)}
    # FEASIBILITY per axis (the §43/§33 driver: when the structural/node-parallel layout OOMs but
    # feature-parallel FITS, the attribute axis must win even at a higher makespan). Fit = the
    # per-device footprint (feature bytes + resident edge metadata) <= the SMALLEST device's HBM.
    caps = np.array([d.usable_mem for d in cluster.devices], dtype=np.float64)
    mincap = float(caps.min())
    struct_feat = (r.counts.astype(np.float64) * feat_dim * 4.0) if r.feat_bytes_dev is None else np.asarray(r.feat_bytes_dev)
    struct_fit = bool(np.all(struct_feat + np.asarray(r.incident) * BYTES_PER_EDGE_RESIDENT <= caps))
    feat_fit = ((feat_dim / D) * N * 4.0 + E * BYTES_PER_EDGE_RESIDENT) <= mincap   # full adj replicated + F/D cols
    hyb_fit = (((feat_dim / Dh) * (N / Dh) * 4.0) + E * BYTES_PER_EDGE_RESIDENT) <= mincap
    feas = {struct_axis: struct_fit, "feature": feat_fit, "hybrid": hyb_fit}
    feasible_axes = [a for a in costs if feas.get(a, True)]
    axis = min(feasible_axes, key=costs.get) if feasible_axes else min(costs, key=costs.get)
    return AxisChoice(axis=axis, makespan_ms=costs[axis], costs=costs,
                      arrange=(r if axis in ("structure", "time") else None),
                      note=(f"cost-exact pick over {sorted(costs)} (feasible={ {a:feas[a] for a in feas} }); "
                            f"arrange winner={r.name}"))


def _arrange_bin() -> str:
    """Resolve the arrange binary: $ZORD_ARRANGE_BIN, else <repo>/build/arrange."""
    env = _os.environ.get("ZORD_ARRANGE_BIN")
    if env:
        return env
    return str(_Path(__file__).resolve().parents[3] / "build" / "arrange")


def have_arrange_cpp() -> bool:
    return _os.path.exists(_arrange_bin())


def arrange_cpp(src, dst, num_nodes, cluster: ClusterProfile,
                link_gbps: Optional[float] = None, feat_dim: int = 128,
                num_snapshots: int = 64, snap: Optional[np.ndarray] = None,
                seed: int = 0, metis_max_edges: int = METIS_MAX_EDGES,
                feat_bytes: Optional[np.ndarray] = None) -> ArrangeResult:
    """Fast C++ adaptive-corner arrange (build/arrange): the candidate construction, cut/comm
    metrics, the dense-core coreness sweep, F_v folds, feasibility and the adaptive pick all run
    natively. Falls back to the numpy arrange() if the binary is absent. For small graphs
    (E <= metis_max_edges) the exact METIS floor candidate is added Python-side and compared, so
    the literal "zord <= METIS" guarantee is preserved; above the gate the C++ cheap-cut floor
    stands (matching arrange()'s size-gate). PROCESS-only; same plan semantics as arrange()."""
    binp = _arrange_bin()
    if not _os.path.exists(binp):
        return arrange(src, dst, num_nodes, cluster, link_gbps, feat_dim,
                       num_snapshots, snap, seed, metis_max_edges, feat_bytes)
    src = np.ascontiguousarray(src, dtype=np.int64)
    dst = np.ascontiguousarray(dst, dtype=np.int64)
    N, E, D, S = int(num_nodes), int(src.size), cluster.num_devices, int(num_snapshots)
    devs = cluster.devices
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype="<f8")
    cap = np.array([d.usable_mem for d in devs], dtype="<f8")
    link = float(link_gbps) if link_gbps is not None else float(cluster.inter_node_bw)
    if snap is None:
        snap = np.minimum((np.arange(E) * S // max(1, E)).astype(np.int64), S - 1)
    snap = np.ascontiguousarray(snap, dtype=np.int64)
    deg = node_degree(src, dst, N)
    rank = np.asarray(cpp_kernel.lpa_rank(N, src, dst), dtype=np.int64)
    cv = np.asarray(cpp_kernel.coreness(src, dst, N), dtype=np.int64)
    has_fv = 1 if feat_bytes is not None else 0

    buf = bytearray()
    buf += _struct.pack("<4q", N, E, D, S)
    buf += _struct.pack("<2d", link, float(feat_dim))
    buf += _struct.pack("<2i", has_fv, int(seed))
    buf += src.astype("<i4").tobytes() + dst.astype("<i4").tobytes() + snap.astype("<i4").tobytes()
    buf += deg.astype("<i8").tobytes() + rank.astype("<i8").tobytes() + cv.astype("<i8").tobytes()
    buf += bw.tobytes() + cap.tobytes()
    if has_fv:
        buf += np.ascontiguousarray(feat_bytes, dtype="<f8").tobytes()

    with _tempfile.TemporaryDirectory() as td:
        ip, op = _os.path.join(td, "in.bin"), _os.path.join(td, "out.bin")
        with open(ip, "wb") as fh:
            fh.write(buf)
        r = _subprocess.run([binp, ip, op], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"arrange binary failed: {r.stderr.decode('utf-8','replace')[-500:]}")
        with open(op, "rb") as fh:
            out = fh.read()
        stderr = r.stderr.decode("utf-8", "replace")

    o = 0
    (Nout,) = _struct.unpack_from("<q", out, o); o += 8
    dev = np.frombuffer(out, dtype="<i4", count=Nout, offset=o).astype(np.int32); o += 4 * Nout
    (hc,) = _struct.unpack_from("<i", out, o); o += 4
    core_mask = None
    if hc:
        core_mask = np.frombuffer(out, dtype=np.int8, count=Nout, offset=o).astype(bool); o += Nout
    (Dout,) = _struct.unpack_from("<q", out, o); o += 8
    def _vec():
        nonlocal o
        v = np.frombuffer(out, dtype="<f8", count=Dout, offset=o).copy(); o += 8 * Dout
        return v
    incident = _vec(); comm_raw = _vec(); counts = _vec().astype(np.int64)
    inc_fold = _vec(); comm_fold = _vec(); featb = _vec()
    (mk,) = _struct.unpack_from("<d", out, o); o += 8
    (ct,) = _struct.unpack_from("<d", out, o); o += 8
    (ecr,) = _struct.unpack_from("<q", out, o); o += 8

    winner = "?"; cand_mk = {}
    for line in stderr.splitlines():
        if line.startswith("WINNER "):
            winner = line.split()[1]
        elif line.startswith("STAT "):
            p = line.split()
            for tok in p:
                if tok.startswith("makespan="):
                    try:
                        cand_mk[p[1]] = float(tok.split("=")[1])
                    except ValueError:
                        pass

    # Adopt the cheapest feasible FLOOR. Candidates beyond the C++ arrange winner:
    #   * multilevel(zord-mincut): zord's OWN C++ multilevel min-cut -- ALL scales, NO pymetis,
    #     NO size-gate (verified cut == pymetis 1.00x). This is the real "zord <= its own
    #     metis-quality min-cut" floor that holds at 1B edges (fixes the old cheap-proxy fake).
    #   * metis(min-cut): pymetis, SMALL graphs only, as an external sanity floor.
    # Both costed with the SAME predict_ms; only the SCALAR path (feat_bytes is None) adds them.
    best = dict(name=winner, dev=dev, core=core_mask, cut=int(ct), inc=incident, comm=comm_raw,
                cnt=counts, ecr=int(ecr), mk=float(mk))
    if feat_bytes is None:
        floors = [("multilevel(zord-mincut)", lambda: cpp_kernel.multilevel_partition(src, dst, N, D))]
        if E <= metis_max_edges:
            floors.append(("metis(min-cut)", lambda: metis_partition(src, dst, N, D)))
        for fname, memfn in floors:
            try:
                mem = np.asarray(memfn(), dtype=np.int64)
                cutf, incf, commf, cntf = edgecut_metrics(src, dst, deg, mem, D, N)
                totf, _, _ = predict_ms(incf * feat_dim, commf * feat_dim, bw.astype(np.float64), link)
                mkf = float(totf.max()); cand_mk[fname] = mkf
                if feasible(cntf, incf, devs, feat_dim) and mkf < best["mk"]:
                    best = dict(name=fname, dev=mem.astype(np.int32), core=None, cut=int(cutf),
                                inc=incf, comm=commf, cnt=cntf, ecr=0, mk=mkf)
            except Exception:
                pass

    return ArrangeResult(
        name=best["name"], assignment=best["dev"], core_mask=best["core"], cut=best["cut"],
        incident=best["inc"], comm_rows=best["comm"], counts=best["cnt"], extra_core_rows=best["ecr"],
        makespan_ms=best["mk"], candidate_makespans=cand_mk,
        bw_gbps=bw.astype(np.float64), link_gbps=link,
        inc_folded=(inc_fold if (has_fv and best["name"] == winner) else None),
        comm_folded=(comm_fold if (has_fv and best["name"] == winner) else None),
        feat_bytes_dev=(featb if (has_fv and best["name"] == winner) else None))
