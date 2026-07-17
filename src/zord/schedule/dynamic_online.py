"""ADVANCED DYNAMICS -- the genuine ONLINE subsystem (BACKLOG P1).

`dynamic.py` (sibling) already gives the batch-level temporal win: when a NEW snapshot
arrives, REUSE the prior assignment and only re-place the CHANGED CONE under a migration
budget instead of re-partitioning from scratch. This module lifts that from a per-batch
re-run into a *designed online subsystem* for an EVOLVING EVENT STREAM, with three pieces
the BACKLOG names and that a blind per-snapshot re-run does NOT have:

  1. BOUNDED STALENESS. We do NOT re-arrange on every micro-batch. A `StalenessPolicy`
     lets up to `max_staleness_snapshots` worth of new events accumulate and be processed
     on the PRIOR layout (the cheap, no-migration path). The layout is only refreshed when
     (a) staleness is exceeded, or (b) drift is detected -- so steady-state ingestion pays
     ZERO migration, and the system spends its `rebalance_budget` only when it actually helps.

  2. DRIFT-TRIGGERED re-arrangement. `detect_drift` is a CHEAP structural score over the
     two GraphStats probes (clusterability / persistence / degree shift). When the stream's
     structure has moved enough (drift > threshold) we trigger a FULL re-schedule under the
     rebalance budget. Otherwise we keep the prior assignment and only RE-COST it. This is the
     P1 "drift detection -> changed-cone re-arrangement", not an unconditional re-partition.

  3. The EVENT-DEPENDENCY-GRAPH as the cut object (NeutronStream-style). `build_event_dependency`
     turns the NEW events of a window into a per-event dependency DAG: event e depends on the
     most-recent earlier event that touches either of its endpoints (its temporal "ear"). The
     transitive closure of that DAG over the new events is the CHANGED CONE -- exactly the set
     of vertices/edges whose state the incremental re-arrangement must re-touch, so the work is
     O(|cone(delta)|), not O(|G|). This is the cut object the online re-arrangement operates on.

PROCESS-ONLY (per the zord pivot): we never change WHAT is computed. We decide WHEN to
re-arrange and WHICH vertices migrate -- same stream + same model => same result. Everything
here is pure numpy + dataclasses and IMPORT-SAFE with no torch and even before its sibling
front-end (`frontend.ingest`) / `schedule.scheduler` modules are present (those are imported
LAZILY / under TYPE_CHECKING so this module loads and runs on a bare CPU box).
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

# These siblings already exist in the engine -- safe to import at module load.
from .dynamic import plan_incremental, partition_incremental, IncrementalPlan

if TYPE_CHECKING:  # pragma: no cover -- type-only; never imported at runtime
    from ..frontend.ingest import TemporalGraphInput, GraphStats
    from ..profiler.cluster_profile import ClusterProfile
    from .scheduler import SchedulePlan


# --------------------------------------------------------------------------- #
#  Policy / state / step records
# --------------------------------------------------------------------------- #
@dataclass
class StalenessPolicy:
    """When to refresh the layout vs. keep ingesting on the prior one.

    max_staleness_snapshots : how many snapshots of new events may be processed on the PRIOR
                              assignment before a refresh is FORCED (bounded staleness). 1 ->
                              refresh every snapshot; larger -> tolerate more drift for cheaper
                              steady state.
    drift_threshold         : structural drift score in [0,1] above which a re-arrangement is
                              triggered regardless of staleness (the change-point trigger).
    rebalance_budget        : migration budget (fraction of OLD vertices that may move) used by
                              the changed-cone re-arrangement when a refresh fires. This is the
                              same lever as dynamic.plan_incremental's migration_budget.
    """
    max_staleness_snapshots: int = 1
    drift_threshold: float = 0.15
    rebalance_budget: float = 0.05


@dataclass
class OnlineState:
    """The carried-over state between online steps (the layout + bookkeeping).

    assignment        : int32 [N] current device-per-vertex layout the runtime is executing on.
    meta              : free-form dict (e.g. last SchedulePlan handle, edge cursor) kept opaque.
    last_stats        : the GraphStats probe at the time of the last refresh (drift reference).
    processed_edges   : how many edges of the stream have been ingested so far (the cursor).
    staleness         : how many snapshots have been processed on the CURRENT layout without a
                        refresh (reset to 0 on every re-arrangement).
    total_migrated_bytes : cumulative node-memory bytes shipped over the link across all refreshes
                        (the running dynamic cost -- a static/from-scratch baseline pays far more).
    """
    assignment: np.ndarray
    meta: dict = field(default_factory=dict)
    last_stats: "Optional[GraphStats]" = None
    processed_edges: int = 0
    staleness: int = 0
    total_migrated_bytes: int = 0


@dataclass
class OnlineStep:
    """The decision + cost record for one online ingestion step.

    schedule_plan : the SchedulePlan in force after this step (full re-plan on a refresh; the
                    PRIOR plan, possibly re-costed, when we stayed on the existing layout). May
                    be None when the scheduler module is unavailable (torch-free CPU-sim path).
    rearranged    : True if this step triggered a re-arrangement (drift or staleness), else False.
    drift         : the structural drift score computed this step (0.0 on a cold start).
    moved_vertices: vertices that changed device this step (0 when not rearranged).
    reason        : human-readable trigger ("cold-start" / "drift>thr" / "staleness>=max" /
                    "reuse (bounded-staleness)").
    """
    schedule_plan: "Optional[SchedulePlan]"
    rearranged: bool
    drift: float
    moved_vertices: int
    reason: str

    def summary(self) -> str:
        head = (f"[online step] {self.reason}  drift={self.drift:.3f}  "
                f"rearranged={self.rearranged}  moved={self.moved_vertices:,} vertices")
        if self.schedule_plan is not None and hasattr(self.schedule_plan, "summary"):
            return head + "\n" + self.schedule_plan.summary()
        return head


# --------------------------------------------------------------------------- #
#  C++ HOT-PATH kernel resolution (build/changed_cone) -- optional, with a
#  pure-numpy/Python fallback so this module ALWAYS runs on a binary-absent box.
#  Mirrors partition.cpp_kernel.graph_bin_path exactly (env var first, else
#  <repo>/build/<name>; repo root = 4 dirs up from this file).
# --------------------------------------------------------------------------- #
def changed_cone_bin_path() -> str:
    """Resolve the changed_cone binary: $ZORD_CHANGED_CONE_BIN, else <repo>/build/changed_cone.
    Repo root is four levels up from this file (src/zord/schedule/dynamic_online.py)."""
    env = os.environ.get("ZORD_CHANGED_CONE_BIN")
    if env:
        return env
    here = os.path.abspath(__file__)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    return os.path.join(repo, "build", "changed_cone")


def have_changed_cone() -> bool:
    return os.path.exists(changed_cone_bin_path())


def _changed_cone_cpp(src: np.ndarray, dst: np.ndarray, lo: int, E: int):
    """Run build/changed_cone over the time-sorted (src,dst) view; return (ear,depth,cone) or None.

    IN(LE): int64 E; int64 new_edge_lo; int32 src[E]; int32 dst[E].
    OUT(LE): int64 n_new; int64 ear[n_new]; int64 depth[n_new]; int64 k; int64 cone[k].
    Returns None (caller falls back to the Python loop) if the binary is missing or the run
    fails -- never raises, so the online subsystem keeps running on a bare CPU box."""
    binp = changed_cone_bin_path()
    if not os.path.exists(binp):
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="zord_kernel_") as tmp:
            ipath = os.path.join(tmp, "in.bin")
            opath = os.path.join(tmp, "out.bin")
            with open(ipath, "wb") as fh:
                fh.write(struct.pack("<qq", int(E), int(lo)))
                src.astype("<i4", copy=False).tofile(fh)
                dst.astype("<i4", copy=False).tofile(fh)
            r = subprocess.run([binp, ipath, opath], capture_output=True, text=True)
            if r.returncode != 0:
                return None
            with open(opath, "rb") as fh:
                n_new = struct.unpack("<q", fh.read(8))[0]
                ear = np.fromfile(fh, dtype="<i8", count=n_new).astype(np.int64)
                depth = np.fromfile(fh, dtype="<i8", count=n_new).astype(np.int64)
                k = struct.unpack("<q", fh.read(8))[0]
                cone = np.fromfile(fh, dtype="<i8", count=k).astype(np.int64)
            if ear.shape[0] != n_new or depth.shape[0] != n_new or cone.shape[0] != k:
                return None
    except Exception:
        return None
    return ear, depth, cone


# --------------------------------------------------------------------------- #
#  Event-dependency graph (NeutronStream-style cut object)
# --------------------------------------------------------------------------- #
@dataclass
class EventDependencyGraph:
    """The per-event temporal dependency DAG over the NEW events of a window.

    For each new event e (an edge (src,dst,t) at offset new_edge_lo+i), its `ear[i]` is the
    LOCAL index (within the new-event block) of the most-recent EARLIER new event that shares an
    endpoint with e -- the event e directly depends on (its state must be applied first). -1 means
    e has no earlier same-vertex new event (a dependency root). `depth[i]` is the longest such
    dependency chain ending at e (its topological depth in the DAG). `cone` lists the UNIQUE
    vertices touched by the new events -- the CHANGED CONE the incremental re-arrangement re-places.

    This is exactly NeutronStream's event-dependency view: events on the same vertex are ordered,
    and the DAG's reachable set bounds the incremental work to O(|cone(delta)|).
    """
    ear: np.ndarray     # int64 [n_new]  local index of the immediate temporal predecessor (-1 = root)
    cone: np.ndarray    # int64 [k]      unique vertices touched by the new events (the changed cone)
    depth: np.ndarray   # int64 [n_new]  topological depth (longest dependency chain ending at e)

    @property
    def num_events(self) -> int:
        return int(self.ear.shape[0])

    @property
    def cone_size(self) -> int:
        return int(self.cone.shape[0])

    @property
    def max_depth(self) -> int:
        return int(self.depth.max()) if self.depth.size else 0

    def summary(self) -> str:
        return (f"[event-dep graph] new_events={self.num_events:,}  "
                f"changed-cone={self.cone_size:,} vertices  max-depth={self.max_depth}")


def build_event_dependency(src: np.ndarray, dst: np.ndarray, t: np.ndarray,
                           new_edge_lo: int) -> EventDependencyGraph:
    """Build the event-dependency DAG of the NEW events (edges at offset >= new_edge_lo).

    The stream is assumed time-sorted (loaders/ingest sort by time). For each new event we find
    the most-recent EARLIER new event that touches the same src or dst vertex -- that is the edge
    the event depends on, the temporal "ear". The transitive set of vertices touched is the
    CHANGED CONE the incremental re-arrangement operates on. Cost is O(n_new) (one forward pass
    over the new-event block; the per-vertex last-seen index is a hash-map walk), independent of
    the size of the carried-over graph -- the O(|cone(delta)|) guarantee.

    Returns an EventDependencyGraph. If there are no new events, all arrays are empty.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    E = int(src.shape[0])
    lo = max(0, int(new_edge_lo))
    lo = min(lo, E)
    n_new = max(0, E - lo)
    if n_new == 0:
        empty = np.empty(0, dtype=np.int64)
        return EventDependencyGraph(ear=empty, cone=empty.copy(), depth=empty.copy())

    # C++ HOT PATH: build/changed_cone does the O(n_new) forward pass at large-window scale.
    # Endpoints are int32 (the supra/graph kernels' canonical edge dtype); fall back to the
    # pure-Python loop below on any failure / binary-absent box. The result is byte-identical:
    # the C++ ear/depth/cone reproduce the same definitions (local ear index, depth=ear+1,
    # sorted-unique cone) so same stream => same dependency graph (process-only invariant).
    if have_changed_cone() and src.max(initial=-1) < 2**31 and dst.max(initial=-1) < 2**31:
        cpp = _changed_cone_cpp(src.astype(np.int32, copy=False),
                                dst.astype(np.int32, copy=False), lo, E)
        if cpp is not None:
            ear_c, depth_c, cone_c = cpp
            return EventDependencyGraph(ear=ear_c, cone=cone_c, depth=depth_c)

    ns = src[lo:]
    nd = dst[lo:]

    ear = np.full(n_new, -1, dtype=np.int64)
    depth = np.zeros(n_new, dtype=np.int64)
    # last_seen[v] = local index of the most recent new event touching vertex v (so far).
    last_seen: dict[int, int] = {}
    for i in range(n_new):
        u = int(ns[i]); v = int(nd[i])
        pu = last_seen.get(u, -1)
        pv = last_seen.get(v, -1)
        # immediate predecessor = the more-recent (larger local index) of the two endpoints'
        # last events; this is the single edge whose state e most-directly depends on.
        p = pu if pu > pv else pv
        ear[i] = p
        if p >= 0:
            depth[i] = depth[p] + 1
        last_seen[u] = i
        last_seen[v] = i

    cone = np.unique(np.concatenate([ns, nd]))
    return EventDependencyGraph(ear=ear, cone=cone, depth=depth)


# --------------------------------------------------------------------------- #
#  Drift detection (cheap structural change-point score)
# --------------------------------------------------------------------------- #
def _rel_shift(a: float, b: float) -> float:
    """Symmetric relative change |b-a| / max(|a|,|b|,eps) clipped to [0,1]."""
    a = float(a); b = float(b)
    denom = max(abs(a), abs(b), 1e-9)
    return float(min(1.0, abs(b - a) / denom))


def detect_drift(prev: "GraphStats", cur: "GraphStats") -> float:
    """Cheap structural drift score in [0,1] between two GraphStats probes.

    Combines the relative shift of the features that move the SPATIAL/TEMPORAL cut regime:
      - clusterability (community structure -> the spatial cut's headroom),
      - persistence    (temporal autocorrelation rho(v) -> THEORY 9.4's L_temporal weight),
      - avg_degree     (densification),
      - max-snapshot node count (a sudden hot-spot / burst).
    These are exactly the stats that set w_S/w_T, so a large shift means the previously chosen
    corner/layout may no longer be near-optimal -- the change-point that warrants a refresh.

    Robust to a missing/None `prev` (cold start) -> returns 1.0 (force a plan). Field access is
    duck-typed via getattr so this works even if GraphStats grows fields, and is torch-free.
    """
    if prev is None or cur is None:
        return 1.0
    g = lambda obj, name: float(getattr(obj, name, 0.0) or 0.0)
    parts = [
        (_rel_shift(g(prev, "clusterability"), g(cur, "clusterability")), 0.35),
        (_rel_shift(g(prev, "persistence"),    g(cur, "persistence")),    0.30),
        (_rel_shift(g(prev, "avg_degree"),     g(cur, "avg_degree")),     0.20),
        (_rel_shift(g(prev, "max_snapshot_nodes"),
                    g(cur, "max_snapshot_nodes")),                        0.15),
    ]
    score = sum(v * w for v, w in parts)
    return float(min(1.0, max(0.0, score)))


# --------------------------------------------------------------------------- #
#  Lazy access to the sibling scheduler (import-safe if not yet present)
# --------------------------------------------------------------------------- #
def _load_scheduler():
    """Import schedule.scheduler.schedule lazily so this module is usable even before the
    scheduler module is built (returns None then -> the CPU-sim incremental path is used)."""
    try:
        from .scheduler import schedule as _schedule  # type: ignore
        return _schedule
    except Exception:
        return None


def _stats_of(tgi: "TemporalGraphInput") -> "Optional[GraphStats]":
    """Best-effort GraphStats off a TemporalGraphInput (duck-typed; tolerant of None)."""
    return getattr(tgi, "stats", None)


def _graph_arrays(tgi: "TemporalGraphInput"):
    """Pull (src, dst, t, num_nodes) out of a TemporalGraphInput's .graph (duck-typed)."""
    g = getattr(tgi, "graph", tgi)
    if hasattr(g, "sort_by_time"):
        g = g.sort_by_time()
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    t = np.asarray(getattr(g, "t", np.arange(src.shape[0])), dtype=np.int64)
    N = int(getattr(g, "num_nodes", int(max(src.max(initial=-1), dst.max(initial=-1)) + 1)))
    return src, dst, t, N


# --------------------------------------------------------------------------- #
#  The online step (the designed P1 subsystem)
# --------------------------------------------------------------------------- #
def online_step(state: Optional[OnlineState], tgi: "TemporalGraphInput",
                cluster: "ClusterProfile", policy: StalenessPolicy = StalenessPolicy(),
                *, link_gbps: Optional[float] = None, feat_dim: int = 128,
                prior_plan: "Optional[SchedulePlan]" = None) -> tuple:
    """One step of online ingestion over the evolving stream. Returns (OnlineStep, OnlineState).

    `tgi`     : the CURRENT window/view of the stream (frontend.ingest produces it; carries the
                graph + per-edge snap + feat_bytes + GraphStats). For an evolving stream this is
                the graph AS OF this step (carried edges + the new ones since the last step).
    `state`   : the carried-over OnlineState (None -> cold start: full schedule(), place all).
    `cluster` : the hardware profile (passed straight through to schedule()).
    `policy`  : the StalenessPolicy governing refresh vs reuse + the rebalance budget.
    `prior_plan` : the SchedulePlan from the previous step (so a refresh can reuse its layout for
                the changed-cone migration; also threaded to schedule() as its `prior`).

    DECISION PROCEDURE (the P1 design, NOT a blind per-snapshot re-run):
      cold start          -> full schedule(); place everything; staleness=0.
      drift > threshold   -> REFRESH: full schedule() under the rebalance budget; the
                             changed-cone migration is costed against the prior layout.
      staleness >= max    -> REFRESH (bounded-staleness ceiling reached).
      otherwise           -> REUSE the prior layout: process the new events on it WITHOUT
                             migrating (staleness += 1), and only RE-COST (the prior plan stands).

    When the sibling scheduler module is unavailable (e.g. a bare CPU box mid-build), the refresh
    path degrades GRACEFULLY to a torch-free changed-cone re-arrangement via dynamic.plan_incremental
    (same algorithm the scheduler uses for the migration), and schedule_plan is left None. Either
    way the layout, drift, and migration cost are computed -- so the subsystem is testable now.
    """
    src, dst, t, N = _graph_arrays(tgi)
    cur_stats = _stats_of(tgi)
    schedule_fn = _load_scheduler()

    # ---- cold start: no prior layout -> a full schedule of the whole current view ---------
    if state is None or getattr(state, "assignment", None) is None:
        plan_obj, assignment = _full_schedule(schedule_fn, tgi, cluster, link_gbps,
                                               feat_dim, policy, prior_plan=prior_plan)
        if assignment is None:
            assignment = _cold_assignment(src, dst, N, cluster, policy)
        new_state = OnlineState(assignment=assignment.astype(np.int32), meta={"plan": plan_obj},
                                last_stats=cur_stats,
                                processed_edges=int(src.shape[0]), staleness=0,
                                total_migrated_bytes=0)
        return (OnlineStep(schedule_plan=plan_obj, rearranged=True, drift=0.0,
                           moved_vertices=int(np.count_nonzero(assignment >= 0)),
                           reason="cold-start"), new_state)

    # ---- decide: drift / staleness force a refresh; else reuse the prior layout -----------
    drift = detect_drift(state.last_stats, cur_stats) if cur_stats is not None else 0.0
    new_edge_lo = min(int(state.processed_edges), int(src.shape[0]))

    over_staleness = (state.staleness + 1) > max(1, policy.max_staleness_snapshots)
    drift_trigger = drift > policy.drift_threshold

    if not (over_staleness or drift_trigger):
        # REUSE: bounded-staleness -- process the new events on the existing layout, no migration.
        # We still build the event-dependency cone (cheap) so the runtime knows the changed set,
        # and we only RE-COST (the prior plan continues to govern).
        _ = build_event_dependency(src, dst, t, new_edge_lo)  # cut object for the runtime
        new_state = OnlineState(assignment=state.assignment, meta=state.meta,
                                last_stats=state.last_stats,
                                processed_edges=int(src.shape[0]),
                                staleness=state.staleness + 1,
                                total_migrated_bytes=state.total_migrated_bytes)
        return (OnlineStep(schedule_plan=prior_plan, rearranged=False, drift=drift,
                           moved_vertices=0, reason="reuse (bounded-staleness)"), new_state)

    # ---- REFRESH: a full re-schedule under the rebalance budget ----------------------------
    reason = "drift>thr" if drift_trigger else "staleness>=max"
    plan_obj, assignment, moved, migrated_bytes = _refresh(
        schedule_fn, tgi, cluster, link_gbps, feat_dim, policy, prior_plan,
        src, dst, N, new_edge_lo, state.assignment)

    new_state = OnlineState(assignment=assignment.astype(np.int32), meta={"plan": plan_obj},
                            last_stats=cur_stats,
                            processed_edges=int(src.shape[0]), staleness=0,
                            total_migrated_bytes=state.total_migrated_bytes + int(migrated_bytes))
    return (OnlineStep(schedule_plan=plan_obj, rearranged=True, drift=drift,
                       moved_vertices=int(moved), reason=reason), new_state)


# --------------------------------------------------------------------------- #
#  Helpers: full schedule, refresh, and the torch-free fallback layout
# --------------------------------------------------------------------------- #
def _full_schedule(schedule_fn, tgi, cluster, link_gbps, feat_dim, policy, *, prior_plan=None):
    """Call the sibling scheduler for a full (re)plan if available. Returns (plan_obj, assignment).

    assignment is extracted from the returned SchedulePlan.partition_plan; (None, None) when the
    scheduler is absent so the caller falls back to the torch-free path."""
    if schedule_fn is None:
        return None, None
    try:
        plan_obj = schedule_fn(tgi, cluster, link_gbps=link_gbps, feat_dim=feat_dim,
                               reuse_frac=0.0, prior=prior_plan)
        assignment = _assignment_of(plan_obj)
        return plan_obj, assignment
    except Exception:
        return None, None


def _assignment_of(plan_obj) -> "Optional[np.ndarray]":
    """Duck-type the int32 assignment out of a SchedulePlan / Plan (tolerant of shape changes)."""
    if plan_obj is None:
        return None
    pp = getattr(plan_obj, "partition_plan", plan_obj)
    asg = getattr(pp, "assignment", None)
    if asg is None and hasattr(pp, "arrange"):
        asg = getattr(getattr(pp, "arrange"), "assignment", None)
    return None if asg is None else np.asarray(asg, dtype=np.int32)


def _refresh(schedule_fn, tgi, cluster, link_gbps, feat_dim, policy, prior_plan,
             src, dst, N, new_edge_lo, prior_assignment):
    """Run a refresh: prefer the scheduler (gets a full SchedulePlan + costed migration); fall
    back to a torch-free changed-cone re-arrangement via dynamic.plan_incremental. Returns
    (plan_obj, assignment, moved_vertices, migrated_bytes)."""
    D = int(getattr(cluster, "num_devices", 1))
    link = float(link_gbps) if link_gbps is not None else float(getattr(cluster, "inter_node_bw", 1.0))

    # Always compute the changed-cone incremental plan (it is the migration cost and the
    # graceful fallback assignment). This is O(|cone(delta)|) per dynamic.plan_incremental.
    inc: IncrementalPlan = plan_incremental(
        src, dst, N, D, prior_assignment, new_edge_lo,
        migration_budget=policy.rebalance_budget, mem_dim=feat_dim, link_gbps=link)

    plan_obj, sched_asg = _full_schedule(schedule_fn, tgi, cluster, link_gbps, feat_dim,
                                         policy, prior_plan=prior_plan)
    if plan_obj is not None and sched_asg is not None:
        # The scheduler produced a full plan; report its layout, but cost the migration against
        # the prior layout (label-matched) -- the scheduler's own incremental plan if it carries
        # one, otherwise our changed-cone estimate.
        moved, migrated = _migration_cost(plan_obj, inc, sched_asg, prior_assignment, D, feat_dim)
        return plan_obj, sched_asg, moved, migrated

    # torch-free fallback: the changed-cone re-arrangement IS the layout.
    return None, inc.assignment, inc.moved_vertices, inc.migrated_bytes


def _migration_cost(plan_obj, inc, sched_asg, prior_assignment, D, feat_dim):
    """Migration cost of a scheduler refresh. Prefer the plan's own IncrementalPlan when present;
    else fall back to the changed-cone estimate `inc`."""
    pp = getattr(plan_obj, "partition_plan", plan_obj)
    pinc = getattr(pp, "incremental", None)
    if pinc is not None and getattr(pinc, "moved_vertices", None) is not None:
        return int(pinc.moved_vertices), int(getattr(pinc, "migrated_bytes", 0))
    return int(inc.moved_vertices), int(inc.migrated_bytes)


def _cold_assignment(src, dst, N, cluster, policy) -> np.ndarray:
    """Torch-free cold-start layout when the scheduler is unavailable: place ALL vertices via the
    same changed-cone LPA dynamic.partition_incremental uses with no prior (prior=None, all new).
    Result-preserving (process-only): this only decides WHERE vertices live."""
    D = int(getattr(cluster, "num_devices", 1))
    return partition_incremental(src, dst, N, D, None, 0, policy.rebalance_budget).astype(np.int32)


__all__ = [
    "StalenessPolicy", "run_stream", "StreamTrace",
    "OnlineState",
    "OnlineStep",
    "EventDependencyGraph",
    "build_event_dependency",
    "detect_drift",
    "online_step",
    "changed_cone_bin_path",
    "have_changed_cone",
]


# --------------------------------------------------------------------------- #
#  run_stream: the REAL streaming DRIVER (#7 depth). online_step is one step;   #
#  this drives the WHOLE evolving event stream window-by-window, threading the   #
#  OnlineState, so zord actually INGESTS a stream (growing or sliding window),    #
#  re-arranging ONLY on drift/staleness (bounded) and reusing otherwise, with     #
#  cumulative migration accounted. PROCESS-only: same stream+model => same plan    #
#  sequence. torch-free (uses online_step's CPU-sim fallback when the scheduler/    #
#  C++ binaries are absent), so it runs and is testable on any box.                 #
# --------------------------------------------------------------------------- #
import types as _types


@dataclass
class StreamTrace:
    steps: list = field(default_factory=list)            # OnlineStep per window
    drifts: list = field(default_factory=list)           # drift score per window
    final_state: "Optional[OnlineState]" = None
    @property
    def windows(self) -> int: return len(self.steps)
    @property
    def refreshes(self) -> int: return sum(1 for s in self.steps if s.rearranged)
    @property
    def reuses(self) -> int: return sum(1 for s in self.steps if not s.rearranged)
    @property
    def total_migrated_bytes(self) -> int:
        return int(self.final_state.total_migrated_bytes) if self.final_state else 0
    @property
    def max_staleness(self) -> int:
        st, mx = 0, 0
        for s in self.steps:
            st = 0 if s.rearranged else st + 1
            mx = max(mx, st)
        return mx
    def summary(self) -> str:
        return (f"[stream] {self.windows} windows: {self.refreshes} refresh / {self.reuses} reuse "
                f"(refresh_rate {self.refreshes/max(1,self.windows):.0%}); max_staleness={self.max_staleness}; "
                f"total_migrated={self.total_migrated_bytes/1e6:.1f} MB")


def _sample_clustering(src, dst, N, deg, sample: int = 256) -> float:
    """Cheap average local clustering coefficient over a node sample (drift cares about its SHIFT,
    not the absolute). Build CSR, sample high-degree-ish nodes, count neighbour-neighbour edges."""
    E = src.shape[0]
    if E == 0:
        return 0.0
    xadj = np.zeros(N + 1, dtype=np.int64); np.add.at(xadj, src + 1, 1); np.add.at(xadj, dst + 1, 1)
    np.cumsum(xadj, out=xadj)
    adj = np.empty(xadj[N], dtype=np.int64); pos = xadj[:-1].copy()
    for i in range(E):
        s, d = int(src[i]), int(dst[i])
        adj[pos[s]] = d; pos[s] += 1; adj[pos[d]] = s; pos[d] += 1
    cand = np.nonzero(deg >= 2)[0]
    if cand.size == 0:
        return 0.0
    rng = np.random.default_rng(0)
    sel = cand if cand.size <= sample else rng.choice(cand, sample, replace=False)
    tot = 0.0; cnt = 0
    for v in sel:
        nb = adj[xadj[v]:xadj[v + 1]]
        k = nb.size
        if k < 2:
            continue
        nbset = set(int(x) for x in nb)
        links = 0
        for u in nb:
            links += len(nbset.intersection(int(x) for x in adj[xadj[u]:xadj[u + 1]]))
        tot += links / (k * (k - 1)); cnt += 1
    return float(tot / cnt) if cnt else 0.0


def _window_stats(g):
    """Compute the GraphStats fields detect_drift uses (avg_degree, max_snapshot_nodes, persistence,
    clusterability), INLINE -- robust, no prober/binary dependency, so the drift trigger is live on
    any box. Drift cares about the SHIFT of these between windows."""
    src = np.asarray(g.src, dtype=np.int64); dst = np.asarray(g.dst, dtype=np.int64)
    t = getattr(g, "t", None); N = int(g.num_nodes); E = int(src.shape[0])
    if E == 0:
        return None
    deg = np.bincount(np.concatenate([src, dst]), minlength=N)
    active = deg > 0; nact = int(active.sum())
    avg_degree = float(2 * E / max(1, nact))
    if t is None or np.asarray(t).size == 0:
        max_snap = nact; persistence = 1.0
    else:
        t = np.asarray(t, dtype=np.int64)
        S = min(64, max(1, int(np.unique(t).size)))
        span = max(1, int(t.max() - t.min() + 1))
        tn = np.minimum((t - t.min()) * S // span, S - 1)
        ns = np.concatenate([src, dst]); ts = np.concatenate([tn, tn])
        key = np.unique(ns * S + ts)
        max_snap = int(np.bincount(key % S, minlength=S).max())
        per_node_snaps = np.bincount(key // S, minlength=N)
        persistence = float(per_node_snaps[active].mean() / S) if nact else 0.0
    clusterability = _sample_clustering(src, dst, N, deg)
    return _types.SimpleNamespace(avg_degree=avg_degree, max_snapshot_nodes=int(max_snap),
                                  persistence=persistence, clusterability=clusterability,
                                  num_nodes=N, num_edges=E)


def run_stream(src, dst, t, num_nodes, cluster, *, feat_bytes=None, window_edges: int = 0,
               num_windows: int = 8, policy: StalenessPolicy = StalenessPolicy(),
               link_gbps: Optional[float] = None, feat_dim: int = 128, growing: bool = True) -> StreamTrace:
    """Drive online ingestion over a time-sorted event stream.

    growing=True  -> the graph GROWS: window k sees events [0 : hi_k] (the evolving-graph regime).
    growing=False -> SLIDING window of `window_edges`: window k sees [hi_k - W : hi_k].
    Re-arranges only when online_step's drift/staleness policy fires; reuses otherwise. Returns a
    StreamTrace (refresh vs reuse counts, max staleness, cumulative migration)."""
    src = np.asarray(src, dtype=np.int64); dst = np.asarray(dst, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64); N = int(num_nodes); E = int(src.shape[0])
    order = np.argsort(t, kind="stable")                  # ensure time-sorted
    src, dst, t = src[order], dst[order], t[order]
    fb = None if feat_bytes is None else np.asarray(feat_bytes)
    W = int(window_edges) if window_edges > 0 else max(1, E // max(1, num_windows))
    bounds = list(range(W, E + 1, W))
    if not bounds or bounds[-1] != E:
        bounds.append(E)

    tr = StreamTrace(); state = None; prior_plan = None
    for hi in bounds:
        lo = 0 if growing else max(0, hi - W)
        g = _types.SimpleNamespace(src=src[lo:hi], dst=dst[lo:hi], t=t[lo:hi], num_nodes=N)
        view = _types.SimpleNamespace(graph=g, stats=_window_stats(g),
                                      feat_bytes=(None if fb is None else fb))
        step, state = online_step(state, view, cluster, policy, link_gbps=link_gbps,
                                   feat_dim=feat_dim, prior_plan=prior_plan)
        prior_plan = step.schedule_plan if step.schedule_plan is not None else prior_plan
        tr.steps.append(step); tr.drifts.append(step.drift)
    tr.final_state = state
    return tr
