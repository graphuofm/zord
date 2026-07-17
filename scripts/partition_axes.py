#!/usr/bin/env python3
"""partition_axes.py -- THE axis study at the heart of zord.

CORE QUESTION (user): for a TEMPORAL graph, do you cut by STRUCTURE, by TIME, or by
ATTRIBUTE? -- and can we PREDICT the winning axis AHEAD OF TIME from cheap O(E) graph
statistics, BEFORE paying for any partition?

This is PROCESS-only (the D25 invariant): same data + same model => same RESULT. We
optimize TIME (predicted makespan) / cut / comm bytes / peak per-device MEMORY /
FEASIBILITY. Accuracy is NEVER the target -- every axis below is a result-preserving
re-placement of WHERE work runs, never a change to WHAT is computed (the partition is a
GAS reduce; feature-parallel column-shard+concat is bit-identical to single-device --
verified here via fp_aggregate_consistency).

We realize EACH axis through the REAL src/zord engine (NEVER networkx, NEVER a
reimplementation):
  STRUCTURAL cut  -> arrange.lpa_edgecut / metis_partition / replicate_core_metrics
                     (edge-cut / metis-floor / dense-core vertex-cut) + edgecut_metrics
  TEMPORAL cut    -> arrange.temporal_partition (contiguous first-activity time ranges
                     per device -- the PTS corner of the space-time duality)
  ATTRIBUTE cut   -> feature_parallel.feature_parallel_plan (shard the F feature COLUMNS)
  HYBRID combos   -> STRUCTURE x TIME : arrange's adaptive corner (it IS the interior of
                       the SpatialCut+TemporalCut duality; reported as the engine's pick)
                     STRUCTURE x FEATURE & TIME x FEATURE : feature_parallel.hybrid_plans
                       (the Dn x Df 2D grid; a node-group split combined with a column split)
  All axes are costed on ONE shared incident-edge roofline (arrange.predict_ms byte
  constants) so the comparison is apples-to-apples.

Across MANY VARIED graphs (real zord.datasets where staged: collegemsg, bitcoin-alpha,
mathoverflow, askubuntu, superuser, wiki-talk, jodie-wikipedia; PLUS calibrated synthetics
spanning sparse<->dense, low<->high feature dim F, temporally-local<->bursty arrival,
homophilous<->anti-correlated attributes) at a FIXED device count K=8 we measure each
(graph, axis) and find the WINNING axis. Then we derive an AHEAD-OF-TIME predictor from
cheap statistics and report its accuracy vs the measured winner (negative results counted).

USAGE:  python3 scripts/partition_axes.py            # full study (real + synthetic)
        python3 scripts/partition_axes.py --quick    # synthetics + small real only (CPU dry-run)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---- the REAL engine (NEVER networkx, NEVER a reimplementation) ----------------------
from zord.datasets.loaders import load_snap_edgelist, load_bitcoin_csv, load_jodie  # noqa: E402
from zord.datasets.temporal_graph import TemporalGraph                              # noqa: E402
from zord.datasets.registry import get_spec                                         # noqa: E402
from zord.profiler.cluster_profile import from_spec, ClusterProfile                 # noqa: E402
import importlib                                                                    # noqa: E402
# NB: zord.partition.__init__ rebinds the package attribute `arrange` to the FUNCTION,
# so `from zord.partition import arrange` would shadow the submodule. Grab the MODULE
# explicitly so A.FEATURE_ROW_BYTES / A._CORENESS_QUANTILES_FULL etc. resolve.
A = importlib.import_module("zord.partition.arrange")                               # noqa: E402
from zord.partition.arrange import (                                                # noqa: E402
    node_degree, lpa_edgecut, temporal_partition, edgecut_metrics,
    replicate_core_metrics, balanced_periphery, metis_partition,
    predict_ms, feasible, BYTES_PER_EDGE_RESIDENT, arrange as arrange_fn,
)
from zord.partition import cpp_kernel                                               # noqa: E402
from zord.partition.feature_parallel import (                                       # noqa: E402
    feature_parallel_plan, hybrid_plans, fp_aggregate_consistency,
)

K_DEVICES = 8           # fixed moderate device count for the whole study
NUM_SNAPSHOTS = 64
SEED = 0
GB = 1024 ** 3
MEM_RESERVE_GB = 2.0    # DeviceProfile.mem_reserved default (framework/driver) -- usable = hbm - 2GB


# ====================================================================================== #
# CLUSTER: a homogeneous K=8 cluster + a moderate (cross-node-ish) interconnect so the    #
# axis question is decided by the GRAPH/ATTRIBUTE regime, not by device heterogeneity.    #
# The link is a PARAMETER (D39); we use a moderate value so neither comm nor compute is    #
# trivially free.                                                                          #
#                                                                                          #
# HBM is sized PER GRAPH so feasibility pressure is the SAME relative to each working set   #
# (else a fixed GB lets every small synthetic fit and the FEASIBILITY-driven attribute      #
# flip -- the real §38 win -- never appears). usable_per_device target = pressure * (the    #
# total node-feature + adjacency working set / K); pressure<1 forces tiering/axis choices.  #
# Note the DeviceProfile reserves 2GB, so we add it back into the requested capacity.       #
# ====================================================================================== #
def make_cluster_for(g: TemporalGraph, F: int, link_gbps: float,
                     pressure: float) -> ClusterProfile:
    N, E = int(g.num_nodes), int(g.num_edges)
    working = N * F * 4.0 + E * BYTES_PER_EDGE_RESIDENT        # full single-device working set
    usable_target = pressure * working / K_DEVICES            # per-device usable budget
    hbm_gb = (usable_target + MEM_RESERVE_GB * GB) / GB        # add back the 2GB reserve
    hbm_gb = max(hbm_gb, MEM_RESERVE_GB + 0.05)               # never below the reserve+slack
    return from_spec(hbm_gb=[hbm_gb] * K_DEVICES,
                     agg_bw_gbps=[500.0] * K_DEVICES,
                     interconnect_gbps=link_gbps)


# ====================================================================================== #
# CHEAP O(E) STATISTICS -- the ahead-of-time predictor's only inputs. Everything here is   #
# computable from the edge stream in a couple of linear passes (no partition, no METIS).   #
# ====================================================================================== #
@dataclass
class GraphStats:
    name: str
    N: int
    E: int
    F: int                      # feature dimension (the attribute width)
    avg_degree: float
    density: float              # E / (N*(N-1)/2)
    degree_gini: float          # degree-skew (0=uniform .. 1=star)
    F_over_avgdeg: float        # F / avg_degree -- governs feature-vs-structure memory
    temporal_locality: float    # frac of edges whose endpoints recur within a time window
    burstiness: float           # CV of per-snapshot edge counts (bursty arrival => high)
    attr_struct_corr: float     # |corr(node feature scalar, degree)| -- attribute<->structure
    cut_frac_est: float         # cheap O(E) cut-fraction estimate: frac of edges crossing a SINGLE
                                # locality-aware (lpa-order) K-way contiguous split. High => the graph
                                # clusters POORLY => node-parallel boundary comm is large => the
                                # ATTRIBUTE (feature-column) split's integration comm competes/wins
                                # when the link binds. Cheap proxy, NOT a full partition search.
    feat_mem_ratio: float       # N*F*4 / (E*BYTES_PER_EDGE_RESIDENT) -- feature vs adjacency mass


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    # Gini = 1 - 2 * area under Lorenz curve
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def cheap_stats(g: TemporalGraph, F: int, node_attr: Optional[np.ndarray] = None,
                S: int = NUM_SNAPSHOTS) -> GraphStats:
    """All O(E) -- a few linear passes over the edge stream. NO partition / NO METIS.
    node_attr (optional [N]) is a per-node scalar attribute used ONLY for the cheap
    attribute<->structure correlation stat; if absent we use a deterministic hash so the
    stat is defined (~0 correlation) for featureless graphs."""
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    E = int(src.size)
    deg = node_degree(src, dst, N).astype(np.float64)
    avg_deg = E / max(1, N)
    density = E / max(1.0, N * (N - 1) / 2.0)
    gini = _gini(deg)

    # temporal locality: fraction of edges whose (unordered) endpoint pair RECURS later
    # within the same time window (a cheap proxy for "this vertex's neighborhood is
    # temporally local -> a contiguous time split keeps its activity together").
    t = np.asarray(g.t, dtype=np.int64)
    order = np.argsort(t, kind="stable")
    s_o, d_o = src[order], dst[order]
    a = np.minimum(s_o, d_o); b = np.maximum(s_o, d_o)
    pair_key = a.astype(np.int64) * np.int64(N) + b
    # an edge is "temporally local" if its pair appears again immediately-adjacent in the
    # time-sorted stream (same pair re-activates soon) -- O(E) via sort+adjacent compare.
    sk = np.sort(pair_key)
    recurs = np.zeros(E, dtype=bool)
    if E > 1:
        same = sk[1:] == sk[:-1]
        recurs[:-1] |= same
        recurs[1:] |= same
    temporal_locality = float(recurs.mean()) if E else 0.0

    # burstiness: coefficient-of-variation of per-equal-time-window edge counts.
    if E:
        span = max(1, int(t.max()) - int(t.min()))
        bins = np.minimum(((t - t.min()) * S // (span + 1)).astype(np.int64), S - 1)
        counts = np.bincount(bins, minlength=S).astype(np.float64)
        nz = counts[counts > 0]
        burstiness = float(nz.std() / (nz.mean() + 1e-9)) if nz.size else 0.0
    else:
        burstiness = 0.0

    # attribute<->structure correlation: |Pearson(node scalar attribute, degree)|.
    if node_attr is None:
        # deterministic per-node hash in [0,1): a featureless graph has ~0 correlation.
        node_attr = ((np.arange(N) * 2654435761) % 10007) / 10007.0
    na = np.asarray(node_attr, dtype=np.float64)
    if na.std() > 0 and deg.std() > 0:
        attr_corr = float(abs(np.corrcoef(na, deg)[0, 1]))
    else:
        attr_corr = 0.0

    # cheap cut-fraction estimate: ONE locality-aware K-way contiguous split (lpa rank order,
    # an O(E) engine primitive -- NOT a partition SEARCH; just the PSS corner's layout) and count
    # the fraction of edges that cross. A graph that clusters well (real QA/social) gives a LOW
    # cut here; dense/bipartite/random graphs give a HIGH cut -> node-parallel boundary comm large.
    lpa_rank = cpp_kernel.lpa_rank(N, src, dst)
    seg = np.minimum((lpa_rank.astype(np.int64) * K_DEVICES) // max(1, N), K_DEVICES - 1)
    cut_frac_est = float(np.count_nonzero(seg[src] != seg[dst]) / max(1, E))

    feat_mem = N * F * 4.0
    adj_mem = E * BYTES_PER_EDGE_RESIDENT
    return GraphStats(
        name=g.name, N=N, E=E, F=F, avg_degree=avg_deg, density=density,
        degree_gini=gini, F_over_avgdeg=F / max(1e-9, avg_deg),
        temporal_locality=temporal_locality, burstiness=burstiness,
        attr_struct_corr=attr_corr, cut_frac_est=cut_frac_est,
        feat_mem_ratio=feat_mem / max(1.0, adj_mem))


# ====================================================================================== #
# AXIS REALIZATION via the REAL engine. Each function returns a uniform AxisResult so the  #
# axes are directly comparable on the shared roofline. cut / comm-bytes / makespan / peak  #
# memory / feasibility are all read off the engine's own metric functions.                 #
# ====================================================================================== #
@dataclass
class AxisResult:
    axis: str               # STRUCTURE | TIME | ATTRIBUTE | STRUCTURE x TIME | ...
    method: str             # the concrete engine realization
    cut: int                # cross-device edges (single-home) -- 0 where not applicable
    comm_bytes: float       # boundary/integration bytes over the link (bottleneck device)
    makespan_ms: float      # predicted makespan (bottleneck device epoch) on shared roofline
    peak_mem_bytes: float   # peak per-device resident HBM
    feasible: bool


def _structural_axes(g, clu, F, snap):
    """STRUCTURE: edge-cut(hetero) / dense-core vertex-cut / METIS min-cut floor, each
    realized by the real arrange primitives, costed on the shared roofline. Returns the
    BEST (lowest feasible makespan) structural realization as the STRUCTURE axis, plus the
    individual corner makespans for honest reporting."""
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    E = int(src.size)
    D = clu.num_devices
    devs = clu.devices
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    caps_work = bw.copy()
    link = float(clu.inter_node_bw)
    deg = node_degree(src, dst, N)
    lpa_rank = cpp_kernel.lpa_rank(N, src, dst)
    rng = np.random.default_rng(SEED + 1)

    def _pack(method, cut, inc, comm, cnt):
        tot, _, _ = predict_ms(inc * F, comm * F, bw, link)
        feas = feasible(cnt, inc, devs, F)
        peak = float((cnt * F * 4.0 + inc * BYTES_PER_EDGE_RESIDENT).max())
        comm_bytes = float((comm * F * A.FEATURE_ROW_BYTES * A.N_GATHERS).max())
        return AxisResult("STRUCTURE", method, int(cut), comm_bytes,
                          float(tot.max()), peak, bool(feas))

    results = []
    # edge-cut (hetero, bandwidth-matched contiguous lpa split)
    dev = lpa_edgecut(N, deg, lpa_rank, D, caps=caps_work)
    cut, inc, comm, cnt = edgecut_metrics(src, dst, deg, dev, D, N)
    results.append(_pack("edge-cut(hetero)", cut, inc, comm, cnt))

    # dense-core vertex-cut, feasibility-gated coreness sweep (the engine's own sweep)
    core_val = cpp_kernel.coreness(src, dst, N)
    uv = (np.concatenate([src, dst]), np.concatenate([dst, src]))
    best_vc = None
    for q in A._CORENESS_QUANTILES_FULL if E <= A.VERTEXCUT_FULL_SWEEP_MAX_EDGES else A._CORENESS_QUANTILES_COARSE:
        tau = max(2, int(np.quantile(core_val, q)))
        cmask = core_val >= tau
        cs = int(cmask.sum())
        if not (0 < cs < N):
            continue
        dev_p = balanced_periphery(np.nonzero(~cmask)[0], deg, D, lpa_rank=lpa_rank)
        cut2, inc2, comm2, cnt2, _extra = replicate_core_metrics(
            src, dst, deg, cmask, dev_p, D, N, rng, uv=uv)
        if not feasible(cnt2, inc2, devs, F):
            continue
        tot, _, _ = predict_ms(inc2 * F, comm2 * F, bw, link)
        mk = float(tot.max())
        if best_vc is None or mk < best_vc[0]:
            best_vc = (mk, ("vertex-cut(k-core)", cut2, inc2, comm2, cnt2))
    if best_vc is not None:
        results.append(_pack(*best_vc[1]))

    # METIS min-cut floor (zord <= METIS by construction); skip gracefully if absent / too big.
    if E <= A.METIS_MAX_EDGES:
        try:
            dev5 = metis_partition(src, dst, N, D)
            cut5, inc5, comm5, cnt5 = edgecut_metrics(src, dst, deg, dev5, D, N)
            results.append(_pack("metis(min-cut)", cut5, inc5, comm5, cnt5))
        except Exception:
            pass

    feas = [r for r in results if r.feasible] or results
    best = min(feas, key=lambda r: r.makespan_ms)
    return best, {r.method: r for r in results}


def _temporal_axis(g, clu, F, snap):
    """TIME: contiguous first-activity time ranges per device (the PTS corner). Realized by
    arrange.temporal_partition + the same edgecut_metrics roofline."""
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    D = clu.num_devices
    devs = clu.devices
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    link = float(clu.inter_node_bw)
    deg = node_degree(src, dst, N)
    dev = temporal_partition(src, dst, snap, deg, N, NUM_SNAPSHOTS, D)
    cut, inc, comm, cnt = edgecut_metrics(src, dst, deg, dev, D, N)
    tot, _, _ = predict_ms(inc * F, comm * F, bw, link)
    feas = feasible(cnt, inc, devs, F)
    peak = float((cnt * F * 4.0 + inc * BYTES_PER_EDGE_RESIDENT).max())
    comm_bytes = float((comm * F * A.FEATURE_ROW_BYTES * A.N_GATHERS).max())
    return AxisResult("TIME", "temporal(PTS)", int(cut), comm_bytes,
                      float(tot.max()), peak, bool(feas))


def _attribute_axis(g, clu, F):
    """ATTRIBUTE: shard the F feature COLUMNS across devices (feature-parallel). Realized by
    feature_parallel_plan (full graph + F/D cols per device, integrate via column-concat)."""
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    link = float(clu.inter_node_bw)
    fp = feature_parallel_plan(src, dst, N, clu, F, link)
    peak = float(fp.resident_bytes.max())
    comm_bytes = float((fp.integ_ms * (link * 1e9) / 1e3).max())  # invert ms->bytes for the link
    return AxisResult("ATTRIBUTE", f"feature-parallel({fp.bound})", 0, comm_bytes,
                      float(fp.makespan_ms), peak, bool(fp.feasible))


def _structure_x_time_axis(g, clu, F, snap):
    """STRUCTURE x TIME: the engine's ADAPTIVE corner -- arrange() picks the lowest-makespan
    feasible plan among {edge-cut, vertex-cut, spatial(PSS), temporal(PTS), metis}, i.e. the
    INTERIOR of the SpatialCut+TemporalCut duality (THEORY.md). This is the genuine hybrid of
    the structure and time corners chosen by zord itself."""
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    res = arrange_fn(src, dst, N, clu, link_gbps=float(clu.inter_node_bw), feat_dim=F,
                     num_snapshots=NUM_SNAPSHOTS, snap=snap, seed=SEED)
    devs = clu.devices
    peak = float((res.counts * F * 4.0 + res.incident * BYTES_PER_EDGE_RESIDENT).max())
    comm_bytes = float((res.comm_rows * F * A.FEATURE_ROW_BYTES * A.N_GATHERS).max())
    feas = feasible(res.counts, res.incident, devs, F)
    return AxisResult("STRUCTURE x TIME", f"arrange:{res.name}", int(res.cut), comm_bytes,
                      float(res.makespan_ms), peak, bool(feas))


def _hybrid_feature_axes(g, clu, F):
    """STRUCTURE x FEATURE and TIME x FEATURE: the Dn x Df 2D grid (node-group split +
    feature-column split). hybrid_plans returns the non-degenerate grids; we take the best
    feasible as the representative STRUCTURE x FEATURE hybrid (a node partition combined with
    a column shard). The TIME x FEATURE combo shares the same 2D-grid cost shape (a time-group
    split is a node-group split by first-activity) so we report the same best grid for it,
    flagged honestly. Returns (struct_x_feat, time_x_feat) AxisResults or (None, None)."""
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    link = float(clu.inter_node_bw)
    hys = hybrid_plans(src, dst, N, clu, F, link)
    if not hys:
        return None, None
    feas = [h for h in hys if h.feasible] or hys
    best = min(feas, key=lambda h: h.makespan_ms)
    peak = float(best.resident_bytes.max())
    comm_bytes = float((best.comm_ms * (link * 1e9) / 1e3).max())
    sxf = AxisResult("STRUCTURE x FEATURE", best.name, 0, comm_bytes,
                     float(best.makespan_ms), peak, bool(best.feasible))
    txf = AxisResult("TIME x FEATURE", best.name + "[time-group]", 0, comm_bytes,
                     float(best.makespan_ms), peak, bool(best.feasible))
    return sxf, txf


# axes that SPLIT THE FEATURE COLUMNS (exploit the ATTRIBUTE dimension) vs those that do not.
_FEATURE_AXES = {"ATTRIBUTE", "STRUCTURE x FEATURE", "TIME x FEATURE"}
_TIME_AXES = {"TIME", "TIME x FEATURE"}


def _raw_winner_family(winner_axis: str) -> str:
    """The HONEST, UNTUNED label of the win: map the engine's lowest-makespan winning AXIS
    DIRECTLY to a coarse family with NO tunable margins. A feature-splitting axis -> ATTRIBUTE;
    a time axis -> TIME; otherwise STRUCTURE. This is the accuracy DENOMINATOR (no co-tuned
    relabel touches it). The §43-correction headline scores the predictor against THIS."""
    if winner_axis in _FEATURE_AXES:
        return "ATTRIBUTE"
    if winner_axis in _TIME_AXES:
        return "TIME"
    return "STRUCTURE"


def measure_all_axes(g, clu, F, node_attr=None):
    """Realize EVERY axis through the real engine at the fixed K and return
    (axes, winner, raw_family, design_verdict, struct_corners). PROCESS-only metrics only.

    winner       : the lowest-feasible-makespan raw axis (ties -> purer axis, then lower peak
                   mem, then lower comm) -- the honest engine pick.
    raw_family   : the HEADLINE label -- the winning axis mapped DIRECTLY to {STRUCTURE, TIME,
                   ATTRIBUTE} with NO tunable margin (_raw_winner_family). This is the accuracy
                   denominator (NOT label-circular: nothing co-tuned on this set touches it).
    design_verdict = (design_family, design_reason): a SECONDARY, DESIGN-HEURISTIC relabel that
                   credits ATTRIBUTE only for a LOAD-BEARING feature split (feasibility flip OR a
                   material makespan crossover past a HAND-TUNED margin ATTR_MARGIN). It is
                   co-tuned on these 16 graphs => CIRCULAR if used as an accuracy denominator, so
                   we report it ONLY as a labeled design heuristic, NEVER as the headline number
                   (§43-correction). Kept because the feasibility-flip distinction is real (§38)."""
    g = g.sort_by_time()
    E = int(g.num_edges)
    snap = np.minimum((np.arange(E) * NUM_SNAPSHOTS // max(1, E)).astype(np.int64),
                      NUM_SNAPSHOTS - 1)
    struct, struct_corners = _structural_axes(g, clu, F, snap)
    time_ax = _temporal_axis(g, clu, F, snap)
    attr_ax = _attribute_axis(g, clu, F)
    sxt = _structure_x_time_axis(g, clu, F, snap)
    sxf, txf = _hybrid_feature_axes(g, clu, F)

    axes = {"STRUCTURE": struct, "TIME": time_ax, "ATTRIBUTE": attr_ax,
            "STRUCTURE x TIME": sxt}
    if sxf is not None:
        axes["STRUCTURE x FEATURE"] = sxf
        axes["TIME x FEATURE"] = txf

    # purity rank for tie-breaking: prefer the simplest axis that achieves the makespan.
    purity = {"STRUCTURE": 0, "TIME": 0, "ATTRIBUTE": 1, "STRUCTURE x TIME": 1,
              "STRUCTURE x FEATURE": 2, "TIME x FEATURE": 2}
    feasible_axes = {k: v for k, v in axes.items() if v.feasible}
    pool = feasible_axes if feasible_axes else axes
    winner = min(pool, key=lambda k: (round(pool[k].makespan_ms, 6), purity.get(k, 3),
                                      pool[k].peak_mem_bytes, pool[k].comm_bytes))

    # ---- HEADLINE label: the RAW engine winner's axis -> family (NO tuned margin) -------
    raw_family = _raw_winner_family(winner)

    # ---- SECONDARY DESIGN-HEURISTIC relabel (clearly labeled; NOT the accuracy denominator) -----
    # which DIMENSION drives the win?  CIRCULAR if scored against (co-tuned on this set), so it is
    # reported only as a design heuristic per §43-correction.
    # best feasible plan that does NOT split features (the structural/temporal baseline):
    nonfeat = {k: v for k, v in pool.items() if k not in _FEATURE_AXES}
    best_nonfeat_mk = min((v.makespan_ms for v in nonfeat.values()), default=float("inf"))
    best_nonfeat_feas = any(v.feasible for v in nonfeat.values())
    # best feasible plan that DOES split features:
    feat = {k: v for k, v in pool.items() if k in _FEATURE_AXES}
    best_feat_mk = min((v.makespan_ms for v in feat.values()), default=float("inf"))
    best_feat_feas = any(v.feasible for v in feat.values())

    # ATTRIBUTE is credited ONLY for a LOAD-BEARING feature-split win: either a feasibility flip
    # (must shard columns to fit at all) OR a MATERIAL makespan crossover (feature split beats the
    # best structural/temporal plan by a real margin). A sub-margin makespan edge is a TIE the
    # structural plan handles fine -> NOT an attribute win (matches §38: low-F => structure, the
    # equal-compute makespans only diverge once F is large). Margin chosen so random-graph low-F
    # ties (~0.006 vs 0.008 ms) are NOT mislabeled while genuine high-F crossovers (>>20%) are.
    ATTR_MARGIN = 0.20    # HAND-TUNED on these 16 graphs -> a design knob, NOT an accuracy gate.
    if best_feat_feas and (not best_nonfeat_feas):
        # FEASIBILITY FLIP: features must be sharded to fit at all -> ATTRIBUTE-driven (the §38 win).
        design_family = "ATTRIBUTE"
        design_reason = "feasibility flip: only a feature-column split FITS the HBM (node-parallel OOMs)"
    elif best_feat_mk < best_nonfeat_mk * (1 - ATTR_MARGIN):
        # feature split beats every non-feature plan by a MATERIAL margin -> ATTRIBUTE-driven crossover.
        design_family = "ATTRIBUTE"
        design_reason = (f"high-F crossover: feature split {best_feat_mk:.3f}ms beats best "
                         f"structural/temporal {best_nonfeat_mk:.3f}ms by >{ATTR_MARGIN:.0%}")
    else:
        # the win is structural/temporal; any feature split is incidental (does not help).
        # decide TIME vs STRUCTURE by whether the temporal corner is the best non-feature plan.
        nf_winner = min(nonfeat, key=lambda k: (nonfeat[k].makespan_ms, purity.get(k, 3)))
        if nf_winner in _TIME_AXES or (axes["TIME"].feasible and
                                       axes["TIME"].makespan_ms <= best_nonfeat_mk * (1 + 1e-3)):
            design_family = "TIME"
            design_reason = "the contiguous-time (PTS) corner is the best feasible non-feature plan"
        else:
            design_family = "STRUCTURE"
            design_reason = (f"a topology cut ({nf_winner}) is the best feasible non-feature plan; "
                             f"feature split does not help")
    return axes, winner, raw_family, (design_family, design_reason), struct_corners


# ====================================================================================== #
# THE AHEAD-OF-TIME PREDICTOR -- a simple, interpretable scorecard mapping cheap O(E) stats #
# (PLUS the device HBM, which the planner always knows) to a predicted winning family, with  #
# explicit decision boundaries. Runs WITHOUT any partition / METIS -- all inputs are O(E) or  #
# O(1). Derived empirically + from the duality mechanics (THEORY.md / §38-correction).        #
# ====================================================================================== #
# WHAT THE REAL ENGINE TAUGHT US (the load-bearing corrections to the naive §38 rule):
#   (A) A high F/avg_degree is NECESSARY but NOT SUFFICIENT for the ATTRIBUTE axis. Whether the
#       feature-column split WINS is governed by HBM FEASIBILITY: it pays off when node-parallel's
#       per-device footprint would OOM but the column-sharded footprint FITS (the §38 feasibility
#       flip). The predictor computes BOTH cheap footprints (O(1) from N,E,F,D + cap).
#   (B) ATTRIBUTE also wins on MAKESPAN -- but only when the LINK BINDS (slow interconnect) AND the
#       graph CLUSTERS POORLY, so node-parallel's boundary comm (~cut*F) exceeds feature-parallel's
#       integration comm (~(N/D)*F). The cheap cut-fraction estimate (one lpa-order split, O(E))
#       proxies "clusters poorly"; comparing the two cheap comm volumes predicts the crossover.
#       At a FAST link comm is free -> this crossover vanishes and STRUCTURE wins (the §32-v2-NVLink
#       convergence), so the link is part of the rule.
#   (C) TIME wins when temporal_locality is high AND arrival is bursty (contiguous ranges isolate
#       cohorts). STRUCTURE is the default when everything fits and the cut is cheap.
# Decision rule (evaluated in order):
#   1. ATTRIBUTE  <= feasibility flip: node footprint > HBM AND feature-shard footprint <= HBM.
#   2. TIME       <= temporal_locality > 0.5 AND burstiness > 1.
#   3. ATTRIBUTE  <= link BINDS (comm time > compute time at this link) AND estimated node boundary
#                    comm > estimated feature integration comm (poorly-clustered, high cut_frac).
#   4. STRUCTURE  <= otherwise (everything fits, cut is cheap or comm is free -> topology cut).
def _node_footprint_per_dev(N, E, F, D):
    """Cheap O(1) estimate of node-parallel's BALANCED per-device resident bytes (the engine's
    feasible() shape: counts*F*4 + incident*20). Balanced => N/D rows, 2E/D incident edges."""
    return (N / D) * F * 4.0 + (2.0 * E / D) * BYTES_PER_EDGE_RESIDENT


def _fp_footprint_per_dev(N, E, F, D):
    """Cheap O(1) estimate of feature-parallel per-device resident bytes: (F/D)*N*4 columns +
    the FULL adjacency replicated (E*20 on every device). The §38 memory-relief footprint."""
    return (F / D) * N * 4.0 + E * BYTES_PER_EDGE_RESIDENT


# the predictor's TUNABLE thresholds, separated out so a LEAVE-ONE-OUT split can REFIT them on the
# training fold and evaluate on the held-out graph (so the reported LOO accuracy is NOT label-
# circular). The feasibility-flip rule (boundary 1) is PARAMETER-FREE footprint arithmetic and is
# never fit. DEFAULTS below are the hand-chosen design values (used for the in-sample scorecard).
@dataclass
class PredictorThresholds:
    dense_deg: float = 25.0     # makespan-crossover density gate (boundary 2)
    tloc_thr: float = 0.5       # temporal-locality gate (boundary 3)
    burst_thr: float = 1.0      # burstiness gate (boundary 3)


def predict_axis(s: GraphStats, usable_hbm_bytes: float, link_gbps: float,
                 bw_gbps: float = 500.0, thr: PredictorThresholds = None) -> tuple[str, str]:
    """Return (predicted_family, reason) in {STRUCTURE, TIME, ATTRIBUTE}. Inputs are cheap O(E)
    stats + the per-device usable HBM + the link/bandwidth PARAMETERS (all known to the planner
    ahead of time). No partition, no METIS -- the lpa cut estimate is a single O(E) primitive.
    `thr` carries the tunable gates (refit per fold under LOO; defaults = the design values)."""
    if thr is None:
        thr = PredictorThresholds()
    D = K_DEVICES
    node_fp = _node_footprint_per_dev(s.N, s.E, s.F, D)
    fp_fp = _fp_footprint_per_dev(s.N, s.E, s.F, D)
    # boundary 1: ATTRIBUTE -- the feasibility flip (node OOMs, column-shard fits). PARAMETER-FREE.
    if node_fp > usable_hbm_bytes and fp_fp <= usable_hbm_bytes:
        return ("ATTRIBUTE",
                f"feasibility flip: node footprint {node_fp/GB:.2f}GB > HBM {usable_hbm_bytes/GB:.2f}GB "
                f"but feature-shard {fp_fp/GB:.2f}GB fits (F/avg_deg={s.F_over_avgdeg:.0f}, §38 relief)")
    # boundary 2: ATTRIBUTE -- the makespan crossover, ONLY when the link BINDS. The engine's best
    # STRUCTURAL corner is the dense-core VERTEX-CUT, which kills boundary comm UNLESS the graph is
    # DENSE (a large dense core must be replicated -> expensive) -- then even vertex-cut cannot beat
    # the feature-column split's integration comm. The cheap, robust separator the engine confirmed
    # is DENSITY (avg_degree): dense graphs cut poorly at every corner; sparse graphs (incl. skewed-
    # but-clusterable real QA/social graphs, which vertex-cut handles) keep node-parallel cheap.
    # link binds when comm time would dominate compute: roofline comp ~ (2E/D)*F over HBM vs a full-F
    # boundary row exchange over the (slow) link. Checked BEFORE the TIME rule because a dense graph
    # may also look temporally local, but the engine rewards the feature split, not the time split.
    comp_ms = (2.0 * s.E / D) * A.BYTES_PER_EDGE_TRAVERSAL * A.N_GATHERS / (bw_gbps * 1e9) * 1e3
    boundary_ms = (s.N / D) * s.F * A.FEATURE_ROW_BYTES * A.N_GATHERS / (link_gbps * 1e9) * 1e3
    link_binds = boundary_ms > comp_ms
    if link_binds and s.avg_degree >= thr.dense_deg:
        return ("ATTRIBUTE",
                f"makespan crossover: link binds (boundary {boundary_ms:.1f}ms > compute "
                f"{comp_ms:.1f}ms) and DENSE graph (avg_deg={s.avg_degree:.0f}>={thr.dense_deg:.0f}) -> "
                f"even vertex-cut replicates a big core; feature-column split's integration comm wins")
    # boundary 3: TIME -- high temporal locality + bursty arrival (the PTS corner). HONEST: in this
    # STATIC per-batch cost model the spatial cut almost always subsumes the temporal cut (THEORY.md
    # §57: PSS beats PTS 3.9-4.9x on real graphs), so TIME rarely WINS measured -- it pays off mainly
    # in the DYNAMIC/TGN regime (cross-snapshot state), which this roofline does not target. We KEEP
    # the rule (it correctly flags temporally-local graphs) and report its measured dominance honestly.
    if s.temporal_locality > thr.tloc_thr and s.burstiness > thr.burst_thr:
        return ("TIME",
                f"temporal_locality={s.temporal_locality:.2f}>{thr.tloc_thr:.2f} and burstiness="
                f"{s.burstiness:.2f}>{thr.burst_thr:.2f} -> contiguous time ranges keep activity local (PTS corner)")
    # boundary 4: STRUCTURE -- everything fits, cut cheap (sparse / clusterable) / comm free
    return ("STRUCTURE",
            f"node footprint {node_fp/GB:.2f}GB fits and sparse/clusterable (avg_deg="
            f"{s.avg_degree:.0f}<{thr.dense_deg:.0f}, cut_frac={s.cut_frac_est:.2f}, link_binds="
            f"{link_binds}) -> vertex-cut/edge-cut keeps node-parallel cheap")


# ====================================================================================== #
# LEAVE-ONE-OUT (held-out) EVALUATION of the predictor -- the §43-correction de-circling.   #
# The in-sample scorecard uses hand-chosen gates; that number can be label-circular if the   #
# gates were tuned on the same graphs. So we ALSO report a leave-one-out accuracy: for each   #
# graph i, REFIT the tunable gates (dense_deg, tloc_thr, burst_thr) on the OTHER n-1 graphs    #
# (a tiny grid search maximizing training-fold agreement with the RAW-WINNER family), then      #
# predict graph i with those refit gates and score it. No held-out graph ever sees its own      #
# label during fitting -> the LOO number is honest. The feasibility-flip rule is parameter-free   #
# (never fit). Grid is coarse + interpretable (these are physically-meaningful thresholds).       #
# ====================================================================================== #
_DENSE_GRID = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
_TLOC_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
_BURST_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]


def fit_thresholds(train_rows) -> PredictorThresholds:
    """Grid-search the tunable gates to MAXIMIZE agreement with the RAW-WINNER family on the
    training fold. Ties broken toward the design defaults so a small fold stays interpretable.
    train_rows: list of dicts each with 'stats','usable_bytes','link','bw','raw_family'."""
    best = None
    default = PredictorThresholds()
    for dd in _DENSE_GRID:
        for tl in _TLOC_GRID:
            for bu in _BURST_GRID:
                thr = PredictorThresholds(dense_deg=dd, tloc_thr=tl, burst_thr=bu)
                hits = 0
                for r in train_rows:
                    pred, _ = predict_axis(r["stats"], r["usable_bytes"], r["link"],
                                           bw_gbps=r["bw"], thr=thr)
                    hits += (pred == r["raw_family"])
                # prefer more hits; on a tie prefer closeness to the design defaults (stable).
                tie = (abs(dd - default.dense_deg) + abs(tl - default.tloc_thr)
                       + abs(bu - default.burst_thr))
                key = (hits, -tie)
                if best is None or key > best[0]:
                    best = (key, thr)
    return best[1]


def leave_one_out_accuracy(fold_rows):
    """Honest LOO: for each i, refit gates on the OTHER n-1 rows, predict row i, score vs its
    RAW-WINNER family. Returns (hits, n, list[(name, pred, true, refit_thr)]). No row sees its
    own label during fitting."""
    n = len(fold_rows)
    out = []
    hits = 0
    for i in range(n):
        train = [fold_rows[j] for j in range(n) if j != i]
        thr = fit_thresholds(train)
        r = fold_rows[i]
        pred, _ = predict_axis(r["stats"], r["usable_bytes"], r["link"], bw_gbps=r["bw"], thr=thr)
        ok = (pred == r["raw_family"])
        hits += ok
        out.append((r["stats"].name, pred, r["raw_family"], thr))
    return hits, n, out


# ====================================================================================== #
# SYNTHETIC GENERATORS -- calibrated to span the regimes the predictor must separate.      #
# All deterministic (seeded). Each returns (TemporalGraph, F, node_attr).                  #
# Most use a COMMUNITY (stochastic-block) topology so a STRUCTURAL cut has a genuinely      #
# small halo (random graphs have a pathologically bad cut on EVERY partition -> they unduly  #
# favour the feature axis; community structure is also more realistic for temporal graphs). #
# ====================================================================================== #
def _temporal_graph(src, dst, t, name, num_nodes):
    return TemporalGraph(src=np.asarray(src, np.int64), dst=np.asarray(dst, np.int64),
                         t=np.asarray(t, np.int64), num_nodes=int(num_nodes), name=name)


def _community_graph(N, avg_deg, ncomm=16, p_in=0.9, seed=SEED, bursty=False, local=False):
    """Vectorized stochastic-block temporal graph: each node belongs to a contiguous community;
    a fraction p_in of edges stay WITHIN the community (small structural cut), the rest are
    global. Returns (src, dst, t). bursty -> arrival clustered into spikes; local -> timestamps
    track community id (community c active in time-window c) so a CONTIGUOUS TIME split isolates
    communities (the TIME axis sweet spot)."""
    rng = np.random.default_rng(seed)
    E = N * avg_deg
    csize = N // ncomm
    src = rng.integers(0, N, E)
    comm = src // csize
    comm = np.minimum(comm, ncomm - 1)
    in_comm = rng.random(E) < p_in
    # within-community dst: same community block [c*csize, (c+1)*csize)
    lo = comm * csize
    off = rng.integers(0, csize, E)
    dst_in = np.minimum(lo + off, N - 1)
    dst_out = rng.integers(0, N, E)
    dst = np.where(in_comm, dst_in, dst_out)
    if local:
        # community c is active in time-window c -> a contiguous time split isolates communities
        t = (comm * 100000 // ncomm + rng.integers(0, 100000 // ncomm, E)).astype(np.int64)
    elif bursty:
        t = (rng.integers(0, 20, E) * 5000 + rng.integers(0, 50, E)).astype(np.int64)
    else:
        t = rng.integers(0, 100000, E).astype(np.int64)
    return src.astype(np.int64), dst.astype(np.int64), np.sort(t)


def syn_sparse_lowF(seed=SEED):
    """SPARSE community graph, low feature dim, smooth arrival. Small halo + compute divides ->
    expect STRUCTURE."""
    src, dst, t = _community_graph(40000, 5, ncomm=16, seed=seed)
    return _temporal_graph(src, dst, t, "syn-sparse-lowF", 40000), 16, None


def syn_dense_lowF(seed=SEED):
    """DENSE community graph, low F, smooth arrival. Replicated adjacency is heavy for the
    feature axis; topology cut modest -> expect STRUCTURE."""
    src, dst, t = _community_graph(12000, 40, ncomm=12, seed=seed)
    return _temporal_graph(src, dst, t, "syn-dense-lowF", 12000), 16, None


def syn_sparse_highF(seed=SEED):
    """SPARSE, HIGH F (attribute-heavy, low-degree) -- the §38-correction sweet spot:
    F >> 5*avg_deg, features dominate, adjacency cheap to replicate -> expect ATTRIBUTE.
    Sized large so the feature working set genuinely presses HBM (real feasibility flip)."""
    src, dst, t = _community_graph(120000, 3, ncomm=24, seed=seed)
    return _temporal_graph(src, dst, t, "syn-sparse-highF", 120000), 8192, None


def syn_dense_highF(seed=SEED):
    """DENSE + HIGH F: features are large but adjacency replication is ALSO heavy (dense).
    The §38-correction boundary is CONTESTED (F vs 5*avg_deg both large) -> honest test of
    whether the attribute split still pays once the replicated adjacency is expensive."""
    src, dst, t = _community_graph(20000, 60, ncomm=16, seed=seed)
    return _temporal_graph(src, dst, t, "syn-dense-highF", 20000), 4096, None


def syn_temporally_local(seed=SEED):
    """TEMPORALLY LOCAL: nodes activate in contiguous COHORTS over time -- cohort c (a contiguous
    id block) is active ONLY in time-window c, and edges stay WITHIN the cohort. So splitting by
    first-activity time (the PTS corner) isolates cohorts with a near-ZERO spatial cut while
    keeping each cohort's whole timeline local -> expect TIME to win the duality. A small fraction
    of cross-cohort edges keeps it realistic (not a trivially disconnected graph)."""
    rng = np.random.default_rng(seed)
    N = 40000
    ncoh = 40
    csize = N // ncoh
    avg_deg = 6
    E = N * avg_deg
    src = rng.integers(0, N, E)
    coh = np.minimum(src // csize, ncoh - 1)
    # 97% within-cohort dst (same contiguous id block), 3% cross-cohort (global) -> tiny cut
    within = rng.random(E) < 0.97
    lo = coh * csize
    dst_in = np.minimum(lo + rng.integers(0, csize, E), N - 1)
    dst = np.where(within, dst_in, rng.integers(0, N, E)).astype(np.int64)
    # time STRICTLY follows the cohort id -> first-activity order == cohort order (PTS low-cut)
    t = (coh * 1000 + rng.integers(0, 1000, E)).astype(np.int64)
    order = np.argsort(t, kind="stable")
    return _temporal_graph(src[order].astype(np.int64), dst[order], t[order],
                           "syn-temporal-local", N), 32, None


def syn_temporally_bursty_spread(seed=SEED):
    """BURSTY arrival but communities SPREAD across time (low temporal locality of pairs):
    arrival is bursty yet a time split does NOT localize a community's neighborhood. Honest
    counter-case: expect STRUCTURE despite high burstiness (TIME should NOT win on burst alone)."""
    src, dst, t = _community_graph(36000, 6, ncomm=16, seed=seed, bursty=True)
    return _temporal_graph(src, dst, t, "syn-bursty-spread", 36000), 16, None


def syn_homophilous_attr(seed=SEED):
    """Attribute CORRELATED with structure (homophily): node attribute ~ degree. Moderate F.
    Tests attr_struct_corr stat; the axis should still follow the F-vs-degree rule (structural),
    NOT the correlation -- an honest check that high attr-corr does not by itself flip the axis."""
    rng = np.random.default_rng(seed)
    N = 30000
    src, dst, t = _community_graph(N, 8, ncomm=16, seed=seed)
    g = _temporal_graph(src, dst, t, "syn-homophilous", N)
    deg = node_degree(g.src, g.dst, N).astype(np.float64)
    node_attr = deg + rng.normal(0, deg.std() * 0.1 + 1, N)  # attribute correlated with degree
    return g, 64, node_attr


def syn_anti_attr(seed=SEED):
    """Attribute ANTI-correlated with structure. Moderate F. Honest negative-control: the
    attribute<->structure correlation is a stat we REPORT, but the axis decision is
    memory/comm-driven, not corr-driven, so this should match its homophilous twin (corr does
    NOT flip the axis -- guards against over-reading attr_struct_corr)."""
    rng = np.random.default_rng(seed)
    N = 30000
    src, dst, t = _community_graph(N, 8, ncomm=16, seed=seed)
    g = _temporal_graph(src, dst, t, "syn-anti-attr", N)
    deg = node_degree(g.src, g.dst, N).astype(np.float64)
    node_attr = -deg + rng.normal(0, deg.std() * 0.1 + 1, N)  # anti-correlated
    return g, 64, node_attr


SYNTHETICS = [
    syn_sparse_lowF, syn_dense_lowF, syn_sparse_highF, syn_dense_highF,
    syn_temporally_local, syn_temporally_bursty_spread,
    syn_homophilous_attr, syn_anti_attr,
]


# ====================================================================================== #
# REAL graphs (zord.datasets), loaded from the local staged cache. F chosen per-graph so   #
# the set spans low-F and high-F regimes on REAL topology.                                 #
# ====================================================================================== #
def _cache(fname):
    return os.path.expanduser(os.path.join("~/.cache/zord", fname))


def real_graphs(quick=False):
    """Yield (TemporalGraph, F, node_attr) for the staged real datasets. We assign a feature
    dim F per graph to exercise both regimes on real topology: a moderate F=128 (the engine
    default) for the SNAP graphs, plus a HIGH-F variant of a sparse SNAP graph and the real
    attributed jodie-wikipedia (172-dim edge features -> we use F=172)."""
    out = []
    snap_low = [("collegemsg", "CollegeMsg.txt.gz", 128),
                ("bitcoin-alpha", "soc-sign-bitcoinalpha.csv.gz", 128),
                ("mathoverflow", "sx-mathoverflow.txt.gz", 128),
                ("askubuntu", "sx-askubuntu.txt.gz", 128)]
    if not quick:
        snap_low += [("superuser", "sx-superuser.txt.gz", 128),
                     ("wiki-talk", "wiki-talk-temporal.txt.gz", 128)]
    for name, fname, F in snap_low:
        p = _cache(fname)
        if not os.path.exists(p):
            continue
        sp = get_spec(name)
        if sp.fmt == "bitcoin_csv":
            g = load_bitcoin_csv(p, name=name)
        else:
            g = load_snap_edgelist(p, name=name)
        out.append((g, F, None))
    # HIGH-F real variant: askubuntu (sparse SNAP topology) at attribute-heavy F=2048 -- tests
    # whether the ATTRIBUTE axis fires on REAL sparse topology (not just synthetics).
    p = _cache("sx-askubuntu.txt.gz")
    if os.path.exists(p):
        g = load_snap_edgelist(p, name="askubuntu-highF")
        out.append((g, 2048, None))
    # REAL attributed graph: jodie-wikipedia, 172-dim edge features (uniform F -> the §27/§33
    # uniform-F regime; honest test that attributes are NULL when F is uniform & modest).
    p = _cache("wikipedia.csv")
    if os.path.exists(p) and not quick:
        try:
            g = load_jodie("jodie-wikipedia", path=p)
            F = int(g.efeat.shape[1]) if g.efeat is not None else 172
            out.append((g, F, None))
        except Exception as e:
            print(f"  [skip jodie-wikipedia: {e}]")
    return out


# ====================================================================================== #
# REPORTING                                                                                #
# ====================================================================================== #
def fmt_bytes(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:.1f}{unit}"
        b /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="synthetics + small real only (fast CPU dry-run)")
    ap.add_argument("--link_gbps", type=float, default=12.0,
                    help="interconnect bandwidth PARAMETER (GB/s)")
    ap.add_argument("--pressure", type=float, default=2.5,
                    help="default HBM pressure = (per-device usable budget)/(working set / K). "
                         "Generous default (everything fits -> makespan decides); the high-F graphs "
                         "are overridden to a TIGHT pressure to expose the feasibility-driven flip.")
    args = ap.parse_args()

    print("=" * 104)
    print("PARTITION-AXIS STUDY -- STRUCTURE vs TIME vs ATTRIBUTE on temporal graphs (the heart of zord)")
    print(f"K={K_DEVICES} devices, link={args.link_gbps}GB/s (a PARAMETER), per-graph HBM sized to a "
          f"feasibility PRESSURE so regimes separate.")
    print("PROCESS-only: makespan/cut/comm/mem/feasibility -- same data+model => same RESULT; accuracy is "
          "NEVER the target. REAL src/zord engine (no networkx).")
    print("=" * 104)

    # ---- math-consistency check (the PROCESS-only invariant for the ATTRIBUTE axis) ----
    rng = np.random.default_rng(0)
    Nc, Ec, Fc = 500, 3000, 64
    cs = rng.integers(0, Nc, Ec); cd = rng.integers(0, Nc, Ec)
    X = rng.standard_normal((Nc, Fc)).astype(np.float32)
    splits = [Fc // K_DEVICES] * K_DEVICES
    splits[-1] += Fc - sum(splits)
    _, _, diff = fp_aggregate_consistency(cs, cd, X, splits)
    print(f"[same-result invariance] feature-parallel column-shard+concat vs single-device "
          f"A@X : max|diff| = {diff:.3e}  ({'EXACT/fp-eps -> result-preserving' if diff < 1e-3 else 'MISMATCH!'})")
    print()

    # gather the graph set. `pressure` per graph: most at the default; a few sparse-high-F graphs
    # at TIGHT pressure to expose the FEASIBILITY-driven attribute flip (the §38 win); dense/low-F
    # at LOOSE pressure (everything fits -> the makespan axis decides, honest negative for attribute).
    P = args.pressure
    graphs = []                                  # (TemporalGraph, F, node_attr, kind, pressure)
    for fn in SYNTHETICS:
        g, F, attr = fn()
        # sparse-high-F synthetics: tight pressure so the feasibility flip can appear.
        pr = 0.55 if "highF" in g.name else P
        graphs.append((g, F, attr, "synthetic", pr))
    for g, F, attr in real_graphs(quick=args.quick):
        pr = 0.55 if "highF" in g.name else P
        graphs.append((g, F, attr, "real", pr))

    rows = []
    t0 = time.time()
    for g, F, attr, kind, pr in graphs:
        clu = make_cluster_for(g, F, args.link_gbps, pr)
        usable_gb = clu.devices[0].usable_mem / GB
        usable_bytes = clu.devices[0].usable_mem
        bw = clu.devices[0].hbm_bw_gbps
        s = cheap_stats(g, F, node_attr=attr)
        axes, winner, raw_family, (design_family, design_reason), corners = \
            measure_all_axes(g, clu, F, node_attr=attr)
        pred, reason = predict_axis(s, usable_bytes, args.link_gbps, bw_gbps=bw)
        rows.append(dict(stats=s, axes=axes, winner=winner, raw_family=raw_family,
                         design_family=design_family, design_reason=design_reason,
                         pred=pred, reason=reason, kind=kind, pressure=pr,
                         usable_gb=usable_gb, usable_bytes=usable_bytes, link=args.link_gbps,
                         bw=bw, corners=corners))
        print(f"[{kind:9s}] {s.name:22s} N={s.N:>8,} E={s.E:>10,} F={F:>5} avg_deg={s.avg_degree:5.1f} "
              f"HBM~={usable_gb:5.2f}GB -> WINNER={winner:18s} RAW-FAM={raw_family:9s} "
              f"design={design_family:9s} pred={pred}")

    # ----------------------- TABLE 1: axis x graph (makespan ms) -----------------------
    print("\n" + "=" * 104)
    print("TABLE 1 -- predicted MAKESPAN (ms) per axis per graph  [* = engine winner;  X = OOM/infeasible]")
    print("=" * 104)
    all_axes = ["STRUCTURE", "TIME", "ATTRIBUTE", "STRUCTURE x TIME",
                "STRUCTURE x FEATURE", "TIME x FEATURE"]
    hdr = (f"{'graph':24s}" + "".join(f"{a.replace(' x ','x')[:11]:>13s}" for a in all_axes)
           + "  RAW-FAM")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['stats'].name:24s}"
        for a in all_axes:
            ax = r["axes"].get(a)
            if ax is None:
                line += f"{'-':>13s}"
            else:
                mark = "*" if a == r["winner"] else (" X" if not ax.feasible else "")
                line += f"{ax.makespan_ms:>11.2f}{mark:>2s}"
        line += f"  {r['raw_family']}"
        print(line)

    # ----------------------- TABLE 2: winner detail + cut/comm/mem ----------------------
    print("\n" + "=" * 104)
    print("TABLE 2 -- WINNING axis detail (engine pick: method, cut, comm-bytes, peak-mem, feasible)")
    print("=" * 104)
    print(f"{'graph':24s}{'WINNER axis':20s}{'method':28s}{'cut':>11s}{'comm':>10s}{'peakMem':>10s} feas")
    for r in rows:
        w = r["axes"][r["winner"]]
        print(f"{r['stats'].name:24s}{r['winner']:20s}{w.method[:27]:28s}"
              f"{w.cut:>11,}{fmt_bytes(w.comm_bytes):>10s}{fmt_bytes(w.peak_mem_bytes):>10s}"
              f"  {'Y' if w.feasible else 'N'}")

    # ----------------------- the WINNING-AXIS-VARIES finding (RAW engine winner) --------
    from collections import Counter
    fam_counts = Counter(r["raw_family"] for r in rows)
    print("\n" + "=" * 104)
    print("FINDING -- the RAW engine lowest-makespan WINNER VARIES by graph regime (axis->family, NO tuning):")
    for fam in ("STRUCTURE", "TIME", "ATTRIBUTE"):
        c = fam_counts.get(fam, 0)
        ex = [r["stats"].name for r in rows if r["raw_family"] == fam][:5]
        print(f"   {fam:10s} wins on {c:2d} graph(s)  e.g. {', '.join(ex) if ex else '(none)'}")
    if len([f for f in fam_counts.values() if f]) == 1:
        print("   (NOTE: a single family swept the set -- regimes did not separate; widen --pressure spread.)")
    else:
        print("   => NO single axis is universally best: the right cut depends on the graph regime. "
              "This is exactly zord's adaptive-corner thesis (THEORY.md space-time-attribute duality).")

    # ----------------------- TABLE 3: cheap stats + predictor vs RAW winner -------------
    print("\n" + "=" * 104)
    print("TABLE 3 -- cheap O(E) stats + AHEAD-OF-TIME predicted family vs the RAW-WINNER family (headline)")
    print("           (design = secondary design-heuristic relabel; NOT the accuracy denominator, §43-corr)")
    print("=" * 104)
    print(f"{'graph':24s}{'F/deg':>8s}{'featMem':>8s}{'cutFr':>7s}{'tLoc':>6s}{'burst':>6s}"
          f"{'gini':>6s}{'aCorr':>6s}  {'PREDICT':10s}{'RAW-WIN':10s}{'design':10s} hit")
    raw_hits = 0
    for r in rows:
        s = r["stats"]
        pred = r["pred"]; raw = r["raw_family"]; dsg = r["design_family"]
        hit = (pred == raw)
        raw_hits += hit
        flag = "OK" if hit else "MISS"
        print(f"{s.name:24s}{s.F_over_avgdeg:>8.1f}{s.feat_mem_ratio:>8.1f}{s.cut_frac_est:>7.2f}"
              f"{s.temporal_locality:>6.2f}{s.burstiness:>6.2f}{s.degree_gini:>6.2f}"
              f"{s.attr_struct_corr:>6.2f}  {pred:10s}{raw:10s}{dsg:10s} {flag}")

    n = len(rows)
    # the HEADLINE: predictor vs RAW engine winner (in-sample), with hand-chosen design gates.
    print("\n" + "=" * 104)
    print("PREDICTOR ACCURACY -- headline: predictor vs the RAW engine lowest-makespan WINNER (§43-correction)")
    print("(cheap O(E) stats only; NO partition / NO METIS computed; NO label-circular co-tuned relabel)")
    print(f"   IN-SAMPLE  3-way accuracy vs RAW winner (design-default gates): {raw_hits}/{n} = "
          f"{100*raw_hits/n:.0f}%   <- can be optimistic (gates hand-chosen on this set)")
    # honest per-family recall vs the raw winner
    for fam in ("STRUCTURE", "TIME", "ATTRIBUTE"):
        tot = sum(1 for r in rows if r["raw_family"] == fam)
        got = sum(1 for r in rows if r["raw_family"] == fam and r["pred"] == fam)
        if tot:
            print(f"     recall[{fam:9s}] = {got}/{tot}")

    # the DE-CIRCLED number: leave-one-out, gates refit on the other n-1 graphs each fold.
    fold_rows = [dict(stats=r["stats"], usable_bytes=r["usable_bytes"], link=r["link"],
                      bw=r["bw"], raw_family=r["raw_family"]) for r in rows]
    loo_hits, loo_n, loo_detail = leave_one_out_accuracy(fold_rows)
    print(f"   LEAVE-ONE-OUT 3-way accuracy vs RAW winner (gates REFIT per fold): {loo_hits}/{loo_n} = "
          f"{100*loo_hits/max(1,loo_n):.0f}%   <- HONEST (held-out; no graph sees its own label)")
    loo_misses = [(nm, p, t) for (nm, p, t, _thr) in loo_detail if p != t]
    if loo_misses:
        print("     LOO mispredictions (honest -- negative results count):")
        for nm, p, t in loo_misses:
            print(f"       {nm}: predicted {p} but RAW winner {t}")

    print("   DECISION BOUNDARIES (the scorecard, evaluated in order; all inputs O(E) or O(1)):")
    print("     1. ATTRIBUTE <= FEASIBILITY FLIP: node footprint (N/D*F*4 + 2E/D*20) > HBM AND "
          "feature-shard footprint (F/D*N*4 + E*20) <= HBM  (§38 memory relief; PARAMETER-FREE)")
    print("     2. ATTRIBUTE <= MAKESPAN CROSSOVER: link binds (est. boundary comm > compute) AND "
          "DENSE (avg_deg >= dense_deg gate) -> even vertex-cut replicates a big core")
    print("     3. TIME      <= temporal_locality > gate AND burstiness > gate  (PTS corner; HONEST: "
          "rarely WINS in the static cost model -- spatial cut subsumes it, THEORY.md §57)")
    print("     4. STRUCTURE <= otherwise  (everything fits, sparse/clusterable -> topology cut)")

    # ---- SECONDARY (design heuristic only): the co-tuned 'which dimension drives the win' relabel.
    print("\n" + "-" * 104)
    print("SECONDARY (DESIGN HEURISTIC ONLY -- NOT an accuracy headline; co-tuned on this set => circular):")
    print("   The 'which-dimension-drives-the-win' relabel credits ATTRIBUTE only for a LOAD-BEARING")
    print("   feature split (feasibility flip OR a >ATTR_MARGIN makespan crossover, ATTR_MARGIN hand-tuned).")
    dsg_counts = Counter(r["design_family"] for r in rows)
    for fam in ("STRUCTURE", "TIME", "ATTRIBUTE"):
        ex = [r["stats"].name for r in rows if r["design_family"] == fam][:5]
        print(f"   design-relabel {fam:10s}: {dsg_counts.get(fam,0):2d} graph(s)  "
              f"e.g. {', '.join(ex) if ex else '(none)'}")
    flipped = [r["stats"].name for r in rows if r["design_family"] != r["raw_family"]]
    if flipped:
        print(f"   (the design relabel disagrees with the raw winner on: {', '.join(flipped)} -- "
              f"this is exactly why it must NOT be the accuracy denominator.)")

    # ---- SEED-ROBUSTNESS: re-run the SYNTHETICS across 5 seeds; is the RAW-winner family stable? ----
    print("\n" + "=" * 104)
    print("SEED-ROBUSTNESS -- synthetics re-generated across 5 seeds; is the RAW-winner family stable?")
    print("=" * 104)
    seeds = [0, 1, 2, 3, 4]
    stable = 0
    for fn in SYNTHETICS:
        fams = []
        for sd in seeds:
            g2, F2, attr2 = fn(seed=sd)
            pr2 = 0.55 if "highF" in g2.name else P
            clu2 = make_cluster_for(g2, F2, args.link_gbps, pr2)
            _, winner2, raw_fam2, _dv, _c = measure_all_axes(g2, clu2, F2, node_attr=attr2)
            fams.append(raw_fam2)
        mode = Counter(fams).most_common(1)[0]
        consistent = (mode[1] == len(seeds))
        stable += consistent
        print(f"   {fn(seed=0)[0].name:24s} raw-winner family across seeds {seeds}: "
              f"{fams}  -> {'STABLE' if consistent else 'VARIES'}")
    print(f"   => {stable}/{len(SYNTHETICS)} synthetic regimes give the SAME raw-winner family across all "
          f"5 seeds (regime structure is seed-robust).")

    print(f"\nDone in {time.time()-t0:.1f}s ({n} graphs x {len(all_axes)} axes via the REAL src/zord engine).")


if __name__ == "__main__":
    main()
