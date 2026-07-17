"""ATTRIBUTE COST MODEL -- the DERIVED (real-math, pure-python/numpy, testable) per-node
feature-byte placement objective for the decomposition-axis choice (BACKLOG M2, feeds H2/P2).

WHAT THIS CLOSES (M2):  feature_parallel.py already COSTS the node-/feature-/hybrid axes on the
shared roofline and choose_decomposition() picks the lowest feasible makespan by EVALUATING those
plans on the actual cluster. What was MISSING is the *closed-form, derived* statement of WHEN each
axis wins -- the crossover dimension F* and the relief inequality -- as explicit functions of
(F, avg_deg, D, link, cap, hbm_bw), WITH the constants, not a rule of thumb. That is exactly what
this module supplies: cheap analytical pre-filter formulas (and a `rule` string) that the scheduler
and the paper quote, and that decide_axis() returns as an AttrDecision. It does NOT replace
choose_decomposition (which does the exact per-device costing via feature_parallel_plan/hybrid_plans);
it is the analytical crossover + rule that explains and fast-paths that choice.

THE TWO AXES (mirroring feature_parallel.py; same byte constants as arrange.predict_ms so the
node-/feature-parallel costs are apples-to-apples):

  NODE-parallel    : split the VERTICES. Each device homes ~N/D rows x ALL F columns and gathers
                     its ~E/D INCIDENT edges; it pays a BOUNDARY comm for the cut rows (the rows
                     whose neighbours live on another device). Compute DIVIDES D-ways; feature
                     memory per device is (N/D)*F*4 but the boundary comm is full-F-wide.
  FEATURE-parallel : split the COLUMNS. Each device holds the FULL graph adjacency (replicated) +
                     F/D feature COLUMNS, runs the full-graph SpMM over its F/D columns, then
                     INTEGRATES (column-concat / Megatron-style all-gather of the partial W-mix).
                     Compute is NOT divided (every device traverses all E edges -- same TOTAL work
                     as one device over all F columns); the WIN is MEMORY: each device stores only
                     F/D columns -> (F/D)*N*4 feature bytes -> the HBM relief for attribute-heavy F.

============================================================================================
THE DERIVED FEATURE-PARALLEL RELIEF INEQUALITY (the BACKLOG M2 statement, with the constants)
============================================================================================
Feature-parallel relieves HBM exactly when the per-device FEATURE bytes a node-parallel cut would
keep resident exceed the per-device ADJACENCY bytes the feature-parallel layout replicates.

  * Node-parallel keeps N*F*4 feature bytes split across D devices, BUT a (worst-case random) cut
    replicates / makes-remote a (D-1)/D fraction of the rows for the boundary; the feature mass that
    binds is the full N*F*4.  Feature-parallel instead REPLICATES the full adjacency on every device:
    each device pays E*BYTES_PER_EDGE_RESIDENT extra, but only holds (D-1)/D LESS of nothing on
    features beyond its F/D columns. The break-even where the feature footprint that node-parallel
    must move/replicate exceeds the adjacency that feature-parallel replicates is:

        N * F * 4   >   E * BYTES_PER_EDGE_RESIDENT * (D - 1) / D            (BYTES_PER_EDGE_RESIDENT = 20)

    Substitute E = avg_deg * N  and divide both sides by N*4:

        F   >   (BYTES_PER_EDGE_RESIDENT / 4) * avg_deg * (D - 1) / D
        F   >   5 * avg_deg * (D - 1) / D                                    (since 20 / 4 = 5)

  So the relief constant is c = BYTES_PER_EDGE_RESIDENT / FEATURE_ROW_BYTES = 20/4 = 5, scaled by the
  vertex-cut surface (D-1)/D. This is a DERIVED inequality with explicit constants, NOT a heuristic:
  attribute-heavy graphs (F above ~5*avg_deg) are where splitting the COLUMNS pays for replicating
  the adjacency. See `feature_relief_inequality()` (returns the threshold + the rule string).

============================================================================================
THE DERIVED MAKESPAN CROSSOVER F* (when feature-parallel BEATS node-parallel on TIME)
============================================================================================
Equate the two per-device bottleneck makespans (compute + comm), full precision, L layers:

  node_cost(F)    = L * [ 2*(E/D)*F*be / (B_hbm*1e9)            # compute over E/D incident edges
                          + boundary_rows*F*fr / (B_link*1e9) ] # boundary comm, boundary_rows~N*(D-1)/D
  feature_cost(F) = L * [ 2*E*(F/D)*be / (B_hbm*1e9)            # full graph, F/D columns
                          + N*F*fr / (B_link*1e9) ]             # integration all-gather of all rows

  (be = BYTES_PER_EDGE_TRAVERSAL = 4, fr = FEATURE_ROW_BYTES = 4.)

The COMPUTE terms are IDENTICAL (2*(E/D)*F == 2*E*(F/D)) -- feature-parallel saves NO compute. So the
makespan difference is purely the COMM term, and it is LINEAR in F:

  node_comm(F)    = L * boundary_rows * F * fr / (B_link*1e9)
  feature_comm(F) = L * N            * F * fr / (B_link*1e9)

With boundary_rows = N*(D-1)/D < N, node_comm < feature_comm for EVERY F>0 -- i.e. with this simple
worst-case integration model feature-parallel's all-gather (all N rows) is always heavier than the
node cut's boundary ((D-1)/D of N rows). The crossover where feature-parallel WINS therefore comes
from FEASIBILITY (memory), not from the linear comm term: feature-parallel is chosen once node-parallel
OOMs (its (N/D)*F*4 + replicated-core bytes exceed a device cap) while feature-parallel (F/D cols +
replicated adjacency) still fits. crossover_dim() reports the F at which the COMM terms would equalise
under a given integration fraction (boundary_frac on the FP side), exposing the lever; decide_axis()
then picks the min FEASIBLE cost so the memory crossover dominates. This matches feature_parallel.py's
"low F -> node wins (compute divides, feature memory small); high F -> feature-parallel relieves HBM".

PROCESS-only: these are TIME / MEMORY / FEASIBILITY formulas. Accuracy is never a target; the chosen
axis is result-preserving (feature_parallel.fp_aggregate_consistency / feature_recombine.verify_*).
Pure numpy + dataclasses, torch-free, import-safe on a CPU box. NEVER networkx.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

# Import the SHARED roofline byte constants from arrange so node-/feature-parallel analytical
# costs use EXACTLY the same words as the engine (apples-to-apples with predict_ms /
# feature_parallel_plan). BYTES_PER_EDGE_RESIDENT is the 20-byte resident edge record that drives
# the relief inequality's constant c = 20/4 = 5.
from .arrange import (
    BYTES_PER_EDGE_TRAVERSAL as _ARR_BYTES_PER_EDGE_TRAVERSAL,
    N_GATHERS as _ARR_N_GATHERS,
    FEATURE_ROW_BYTES as _ARR_FEATURE_ROW_BYTES,
    BYTES_PER_EDGE_RESIDENT as _ARR_BYTES_PER_EDGE_RESIDENT,
)
from ..profiler.cluster_profile import ClusterProfile

# Re-export the constants under the names the DESIGN CONTRACT specifies for this module, pinned to
# the arrange values so there is a single source of truth (a divergence would silently break the
# apples-to-apples comparison). FEATURE_ROW_BYTES/BYTES_PER_EDGE_TRAVERSAL = 4.0 (fp32 word);
# N_GATHERS = 2 (two SpMM gathers for the 2-layer aggregation).
FEATURE_ROW_BYTES: float = float(_ARR_FEATURE_ROW_BYTES)          # 4.0  (fp32 word, matches arrange)
BYTES_PER_EDGE_TRAVERSAL: float = float(_ARR_BYTES_PER_EDGE_TRAVERSAL)  # 4.0
N_GATHERS: int = int(_ARR_N_GATHERS)                              # 2
BYTES_PER_EDGE_RESIDENT: float = float(_ARR_BYTES_PER_EDGE_RESIDENT)    # 20.0 (resident edge record)

# The DERIVED relief constant: c = resident-edge-bytes / feature-row-bytes = 20 / 4 = 5. F* ~ c*avg_deg.
RELIEF_CONST: float = BYTES_PER_EDGE_RESIDENT / FEATURE_ROW_BYTES  # 5.0

if TYPE_CHECKING:  # pragma: no cover - import-safety: GraphStats lives in a sibling-owned module
    from ..frontend.ingest import GraphStats


@dataclass
class AttrDecision:
    """The DERIVED M2 attribute-axis decision: the three closed-form per-device bottleneck costs,
    the crossover dimension F* where feature-parallel would beat node-parallel on comm, the
    integration (recombine) cost, and the human-readable F >~ c*avg_deg relief rule with the
    actual constant c. `axis` is the analytical pick (min FEASIBLE closed-form cost); it FEEDS
    choose_decomposition / the scheduler -- it does not replace the exact per-device costing."""
    axis: str                       # "node" | "feature" | "hybrid"
    node_cost_ms: float             # node-parallel per-device bottleneck makespan (ms)
    feature_cost_ms: float          # feature-parallel per-device bottleneck makespan (ms)
    hybrid_cost_ms: float           # best balanced 2D-grid bottleneck makespan (ms)
    crossover_F: float              # DERIVED F* where node_cost == feature_cost (comm crossover)
    integration_ms: float           # the recombine / column-concat (+full-layer all-gather) cost
    rule: str                       # the F >~ c*avg_deg inequality WITH the actual constant c
    feasible_node: bool = True      # node-parallel fits every device's HBM cap
    feasible_feature: bool = True   # feature-parallel fits every device's HBM cap

    def summary(self) -> str:
        return (
            f"[attr-cost] axis={self.axis}  node={self.node_cost_ms:.2f}ms"
            f"{'' if self.feasible_node else '/OOM'}  feature={self.feature_cost_ms:.2f}ms"
            f"{'' if self.feasible_feature else '/OOM'}  hybrid={self.hybrid_cost_ms:.2f}ms  "
            f"integration={self.integration_ms:.2f}ms\n"
            f"            crossover F*={self.crossover_F:.1f}   {self.rule}"
        )


# ============================================================================ #
# CLOSED-FORM PER-DEVICE COSTS (the DERIVED M2 makespan formulas)               #
# ============================================================================ #
def node_parallel_cost_ms(num_edges: int, F: float, deg_avg: float, D: int,
                          hbm_bw_gbps: float, link_gbps: float, layers: int = 2) -> float:
    """DERIVED node-parallel per-device bottleneck makespan (ms), full precision.

    Each of the D devices homes ~N/D rows x ALL F columns and gathers its ~E/D INCIDENT edges,
    then pays a BOUNDARY comm for the cut rows. Modelled with the SHARED roofline constants:

      compute_ms = layers * 2*(E/D)*F * BYTES_PER_EDGE_TRAVERSAL / (B_hbm*1e9) * 1e3
                   -- the incident-edge SpMM over all F columns; compute DIVIDES D-ways (E/D).
      comm_ms    = layers * boundary_rows * F * FEATURE_ROW_BYTES / (B_link*1e9) * 1e3
                   -- boundary rows crossing the slow link. boundary_rows = N*(D-1)/D is the
                      worst-case random-cut surface (a (D-1)/D fraction of rows have a remote
                      neighbour), with N = E/deg_avg.

    Returns compute_ms + comm_ms (the per-device bottleneck; D=1 has zero comm)."""
    E = float(num_edges)
    F = float(F)
    D = max(1, int(D))
    layers = max(1, int(layers))
    B_hbm = max(float(hbm_bw_gbps), 1e-9) * 1e9
    B_link = max(float(link_gbps), 1e-9) * 1e9
    deg_avg = max(float(deg_avg), 1e-9)
    N = E / deg_avg

    edges_per_dev = E / D
    compute_bytes = layers * 2.0 * edges_per_dev * F * BYTES_PER_EDGE_TRAVERSAL
    compute_ms = compute_bytes / B_hbm * 1e3

    boundary_rows = N * (D - 1) / D                      # worst-case cut surface (0 when D==1)
    comm_bytes = layers * boundary_rows * F * FEATURE_ROW_BYTES
    comm_ms = comm_bytes / B_link * 1e3
    return float(compute_ms + comm_ms)


def feature_parallel_cost_ms(num_nodes: int, num_edges: int, F: float, D: int,
                             hbm_bw_gbps: float, link_gbps: float, layers: int = 2) -> float:
    """DERIVED feature-parallel per-device bottleneck makespan (ms), full precision.

    Each device holds the FULL graph (all E edges) + F/D feature COLUMNS, runs the full-graph SpMM
    over those F/D columns, then INTEGRATES (column-concat / Megatron all-gather). Shared constants:

      compute_ms     = layers * 2*E*(F/D) * BYTES_PER_EDGE_TRAVERSAL / (B_hbm*1e9) * 1e3
                       -- the FULL-graph SpMM over F/D columns. Note 2*E*(F/D) == 2*(E/D)*F, so the
                          per-device compute EQUALS node-parallel's: feature-parallel saves NO
                          compute (replicated adjacency, column-split); its win is MEMORY + comm.
      integration_ms = layers * N * F * FEATURE_ROW_BYTES / (B_link*1e9) * 1e3
                       -- the recombine: all N aggregated rows are column-concatenated (F-wide) over
                          the link before the dense layer mixes columns (see integration_cost_ms).

    Returns compute_ms + integration_ms (the per-device bottleneck)."""
    E = float(num_edges)
    N = float(num_nodes)
    F = float(F)
    D = max(1, int(D))
    layers = max(1, int(layers))
    B_hbm = max(float(hbm_bw_gbps), 1e-9) * 1e9
    B_link = max(float(link_gbps), 1e-9) * 1e9

    cols_per_dev = F / D
    compute_bytes = layers * 2.0 * E * cols_per_dev * BYTES_PER_EDGE_TRAVERSAL
    compute_ms = compute_bytes / B_hbm * 1e3

    integ_ms = integration_cost_ms(int(N), F, link_gbps, layers=layers, full_layer=False)
    return float(compute_ms + integ_ms)


def integration_cost_ms(num_nodes: int, F: float, link_gbps: float,
                        layers: int = 2, full_layer: bool = True) -> float:
    """The RECOMBINE / integration comm (ms) for the feature-parallel (column-shard) axis.

    AGGREGATION recombine (always charged): each device produces its F/D-wide aggregated rows; the
    column slices are CONCATENATED back to the full [N, F] row before the dense layer mixes columns.
    That gathers all N rows, F-wide, over the link once per layer:

      aggregation_ms = layers * N * F * FEATURE_ROW_BYTES / (B_link*1e9) * 1e3

    FULL-LAYER recombine (full_layer=True): the Megatron-style row/column-parallel dense layer also
    ALL-GATHERS the partial (X_shard @ W_shard) contributions so the single nonlinearity is applied
    once on the full-width sum -- closing the aggregation-ONLY H2 gap on the COST side. Each device's
    partial output is F-wide already, so the all-gather moves another layers*N*F*FEATURE_ROW_BYTES
    (the partial-sum reduce-scatter / all-gather of the dense output), doubling the recombine term:

      full_layer_ms  = 2 * aggregation_ms

    full_layer=False returns just the aggregation-concat term (the proven SpMM-only recombine).
    This is the formula side of runtime.feature_recombine.plan_recombine."""
    N = float(num_nodes)
    F = float(F)
    layers = max(1, int(layers))
    B_link = max(float(link_gbps), 1e-9) * 1e9
    aggregation_ms = layers * N * F * FEATURE_ROW_BYTES / B_link * 1e3
    if full_layer:
        # + the Megatron W-mix all-gather of the partial dense outputs (full-layer same-result).
        return float(2.0 * aggregation_ms)
    return float(aggregation_ms)


def crossover_dim(deg_avg: float, D: int, hbm_bw_gbps: float, link_gbps: float,
                  layers: int = 2, boundary_frac: float = 1.0) -> float:
    """SOLVE node_cost(F) == feature_cost(F) for F -> the DERIVED crossover dimension F*.

    Because the COMPUTE terms of node- and feature-parallel are identical (2*(E/D)*F == 2*E*(F/D)),
    the equation reduces to the COMM terms, which are both LINEAR in F:

      node_comm(F)    = layers * [N*(D-1)/D]      * F * fr / (B_link*1e9)
      feature_comm(F) = layers * [boundary_frac*N]* F * fr / (B_link*1e9)   (the integration gather)

    The F factor CANCELS -- with the default worst-case integration (boundary_frac=1, all N rows
    gathered) the two comm slopes are N*(D-1)/D vs N, so node_comm < feature_comm for ALL F and there
    is NO finite comm crossover (feature-parallel's win is then purely FEASIBILITY/memory, not time).
    We therefore report F* as the dimension at which the two COMM costs would equalise under the given
    integration fraction:

      * if boundary_frac < (D-1)/D : feature integration is CHEAPER than the node cut, so feature-
        parallel's comm beats node's for every F; F* = 0 (feature-parallel wins on comm from F=0+).
      * if boundary_frac > (D-1)/D : node's cut is cheaper for every F; F* = +inf (no comm crossover;
        feature-parallel can still win on MEMORY past the relief threshold 5*avg_deg*(D-1)/D).
      * equal slopes : F* is indeterminate -> reported as +inf.

    Returns F* (>=0; np.inf when feature-parallel never wins on comm). The relief inequality
    (feature_relief_inequality) gives the MEMORY threshold ~ RELIEF_CONST*deg_avg*(D-1)/D, which is
    the operative crossover this module reports in the AttrDecision.rule."""
    D = max(1, int(D))
    node_slope = (D - 1) / D                              # node boundary fraction of N
    feat_slope = max(0.0, float(boundary_frac))           # feature integration fraction of N
    if D == 1:
        return float("inf")                               # single device: node has no comm at all
    if feat_slope < node_slope - 1e-12:
        return 0.0                                        # feature comm cheaper for every F
    if feat_slope > node_slope + 1e-12:
        return float("inf")                               # node comm cheaper for every F
    return float("inf")                                   # equal slopes -> indeterminate


def feature_relief_inequality(deg_avg: float, D: int) -> tuple:
    """The DERIVED feature-parallel MEMORY-relief inequality, returned as (F_threshold, rule_string).

    Break-even (full derivation in the module docstring):

        N*F*4 > E*BYTES_PER_EDGE_RESIDENT*(D-1)/D
      <=> F   > (BYTES_PER_EDGE_RESIDENT/FEATURE_ROW_BYTES) * deg_avg * (D-1)/D
      <=> F   > 5 * deg_avg * (D-1)/D                            (BYTES_PER_EDGE_RESIDENT=20, /4 = 5)

    Returns (F_threshold, rule) where F_threshold = RELIEF_CONST*deg_avg*(D-1)/D and `rule` states the
    inequality WITH the constant c (=RELIEF_CONST=5) so the paper/CLI can quote it verbatim."""
    D = max(1, int(D))
    deg_avg = float(deg_avg)
    surface = (D - 1) / D
    F_threshold = RELIEF_CONST * deg_avg * surface
    rule = (
        f"feature-parallel relieves HBM when  N*F*4 > E*{BYTES_PER_EDGE_RESIDENT:g}*(D-1)/D  <=>  "
        f"F > {RELIEF_CONST:g}*avg_deg*(D-1)/D = {RELIEF_CONST:g}*{deg_avg:.2f}*{surface:.3f} "
        f"= {F_threshold:.1f}  (c={RELIEF_CONST:g}={BYTES_PER_EDGE_RESIDENT:g}/{FEATURE_ROW_BYTES:g})"
    )
    return float(F_threshold), rule


# ============================================================================ #
# HYBRID closed-form (best balanced 2D grid Dn x Df = D)                         #
# ============================================================================ #
def _hybrid_cost_ms(num_nodes: int, num_edges: int, F: float, deg_avg: float, D: int,
                    hbm_bw_gbps: float, link_gbps: float, layers: int = 2) -> float:
    """Closed-form best balanced 2D-grid bottleneck makespan over Dn*Df = D factor pairs.

    Each device homes N/Dn rows x F/Df columns, gathers ~2E/Dn incident traversals over F/Df cols,
    and pays a node-boundary comm (F/Df wide, (Dn-1)/Dn surface) PLUS a feature-integration concat
    (F/Df cols, all N/Dn rows). Mirrors feature_parallel.hybrid_plans' decomposable model; the pure
    corners (Df==1 node, Dn==1 feature) are costed by the dedicated functions and excluded here.
    Returns the MINIMUM bottleneck makespan over the non-degenerate grids (np.inf if none, e.g. D
    prime or D==1)."""
    E = float(num_edges)
    N = float(num_nodes)
    F = float(F)
    D = max(1, int(D))
    layers = max(1, int(layers))
    B_hbm = max(float(hbm_bw_gbps), 1e-9) * 1e9
    B_link = max(float(link_gbps), 1e-9) * 1e9

    best = float("inf")
    for dn in range(1, D + 1):
        if D % dn:
            continue
        df = D // dn
        if dn == 1 or df == 1:
            continue                                     # pure corners costed elsewhere
        cols = F / df
        edges_per_group = 2.0 * E / dn
        compute_ms = layers * edges_per_group * cols * BYTES_PER_EDGE_TRAVERSAL / B_hbm * 1e3
        boundary_rows = (N / dn) * (dn - 1) / dn          # node cut surface within the feature group
        integ_rows = N / dn                               # feature integration of this group's rows
        comm_rows = boundary_rows + integ_rows
        comm_ms = layers * comm_rows * cols * FEATURE_ROW_BYTES / B_link * 1e3
        best = min(best, compute_ms + comm_ms)
    return float(best)


# ============================================================================ #
# THE DECISION: evaluate the three closed forms at the graph's (E,N,deg,D,B)    #
# ============================================================================ #
def _feasible_node(num_nodes: int, num_edges: int, F: float, cluster: ClusterProfile) -> bool:
    """Worst-case node-parallel feasibility: the largest per-device feature footprint (N/D rows x F
    cols x 4) + its share of resident adjacency must fit the SMALLEST usable HBM. Conservative
    (assumes an even split onto the smallest device); the exact per-device check lives in arrange."""
    D = cluster.num_devices
    if D == 0:
        return False
    N = float(num_nodes)
    E = float(num_edges)
    feat_bytes = (N / D) * float(F) * 4.0
    adj_bytes = (E / D) * BYTES_PER_EDGE_RESIDENT
    need = feat_bytes + adj_bytes
    cap_min = min(d.usable_mem for d in cluster.devices)
    return need <= cap_min


def _feasible_feature(num_nodes: int, num_edges: int, F: float, cluster: ClusterProfile) -> bool:
    """Feature-parallel feasibility: each device holds F/D feature columns ((F/D)*N*4) PLUS the FULL
    replicated adjacency (E*BYTES_PER_EDGE_RESIDENT). Must fit the smallest usable HBM."""
    D = cluster.num_devices
    if D == 0:
        return False
    N = float(num_nodes)
    E = float(num_edges)
    feat_bytes = (float(F) / D) * N * 4.0
    adj_bytes = E * BYTES_PER_EDGE_RESIDENT                # replicated on EVERY device
    need = feat_bytes + adj_bytes
    cap_min = min(d.usable_mem for d in cluster.devices)
    return need <= cap_min


def decide_axis(stats: "GraphStats", cluster: ClusterProfile, F: float, link_gbps: float,
                *, layers: int = 2) -> AttrDecision:
    """Evaluate the three DERIVED closed-form costs at the graph's (E, N, deg_avg, D, B_hbm, link)
    and pick the lowest FEASIBLE one -> the analytical attribute-axis decision (M2).

    `stats` supplies num_nodes / num_edges / avg_degree (a frontend.ingest.GraphStats, duck-typed so
    this module is import-safe without that sibling module). `cluster` supplies D and the BOTTLENECK
    HBM bandwidth (the slowest device sets the per-device makespan we compare). The pick:
      1. compute node_cost, feature_cost, hybrid_cost at the graph size;
      2. test memory FEASIBILITY of node- and feature-parallel (the relief inequality made concrete);
      3. choose the MIN cost among FEASIBLE axes; if node-parallel OOMs but feature-parallel fits ->
         feature; if neither pure axis fits but a hybrid grid does and is cheapest -> hybrid;
      4. report the DERIVED crossover F* and the F >~ 5*avg_deg*(D-1)/D relief rule.

    Returns an AttrDecision. This FEEDS choose_decomposition / the scheduler (it is the cheap
    analytical pre-filter + the paper-quotable rule); the exact per-device costing stays in
    feature_parallel_plan / hybrid_plans."""
    N = int(getattr(stats, "num_nodes"))
    E = int(getattr(stats, "num_edges"))
    deg_avg = float(getattr(stats, "avg_degree", (E / max(1, N))))
    D = max(1, cluster.num_devices)
    F = float(F)
    layers = max(1, int(layers))
    # BOTTLENECK device bandwidth -- the slowest device sets the per-device makespan being compared
    # (consistent with the engine's max-over-devices bottleneck). Using the min is the conservative,
    # comparison-stable choice for the analytical pre-filter.
    B_hbm = min(d.hbm_bw_gbps for d in cluster.devices) if D else 1.0
    link = max(float(link_gbps), 1e-9)

    node_ms = node_parallel_cost_ms(E, F, deg_avg, D, B_hbm, link, layers=layers)
    feat_ms = feature_parallel_cost_ms(N, E, F, D, B_hbm, link, layers=layers)
    hyb_ms = _hybrid_cost_ms(N, E, F, deg_avg, D, B_hbm, link, layers=layers)
    integ_ms = integration_cost_ms(N, F, link, layers=layers, full_layer=True)
    F_star = crossover_dim(deg_avg, D, B_hbm, link, layers=layers)

    feas_node = _feasible_node(N, E, F, cluster)
    feas_feat = _feasible_feature(N, E, F, cluster)

    # candidate costs keyed by axis, with feasibility; hybrid feasibility is approximated as feasible
    # iff a non-degenerate grid exists (finite hyb_ms) -- the exact per-device fit is in hybrid_plans.
    cand = {
        "node": (node_ms, feas_node),
        "feature": (feat_ms, feas_feat),
    }
    if np.isfinite(hyb_ms):
        cand["hybrid"] = (hyb_ms, True)

    feasible_axes = {k: v for k, v in cand.items() if v[1]}
    pool = feasible_axes if feasible_axes else cand     # if nothing fits, pick least-bad cost
    axis = min(pool, key=lambda k: pool[k][0])

    F_threshold, relief_rule = feature_relief_inequality(deg_avg, D)
    relief_holds = F > F_threshold
    rule = (
        relief_rule
        + f"  ->  at F={F:g}: relief {'HOLDS' if relief_holds else 'does NOT hold'}"
        + (f"; node-parallel OOMs while feature-parallel fits -> feature axis"
           if (not feas_node and feas_feat) else
           f"; node-parallel fits -> compute divides D-ways, node axis preferred unless OOM")
    )

    return AttrDecision(
        axis=axis,
        node_cost_ms=float(node_ms),
        feature_cost_ms=float(feat_ms),
        hybrid_cost_ms=float(hyb_ms),
        crossover_F=float(F_star),
        integration_ms=float(integ_ms),
        rule=rule,
        feasible_node=bool(feas_node),
        feasible_feature=bool(feas_feat),
    )
