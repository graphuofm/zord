"""THE SCHEDULER (L3) -- the one coherent conductor that ties FRONT -> MIDDLE -> BACK.

This is the Python POLICY layer (the "conductor"), NOT a hot path. It does NOT touch every
edge/cell: the heavy structural passes (supra-cell build, the weighted supra-graph partition,
the Belady/MRD cache simulation, the changed-cone closure) live in C++ kernels and are reached
through their thin Python wrappers (partition.allocate / runtime.bufferpool / schedule.dynamic_online).
The scheduler's own work is O(D) over devices and O(S) over snapshots -- closed-form cost
arithmetic and dataclass assembly -- so it runs in microseconds on a torch-free CPU box.

ONE entry, ONE object:
    schedule(tgi, cluster, ...) -> SchedulePlan

The decision procedure is a single documented algorithm that strings the proven sub-planners
into one plan. Hardware is a PARAMETER (the ``cluster`` ClusterProfile): nothing is HBM- or
NVLink-specific (N2). HOMOGENEOUS and HETEROGENEOUS clusters are both handled (N1): the
work/byte split is bandwidth-proportional inside arrange/plan_memory, which degenerates to an
even split when every device is identical, and the makespan is always the per-device max (so a
straggler is exposed either way). PROCESS-ONLY: the scheduler decides WHERE state lives, WHEN it
is staged, and WHICH axis decomposes the work -- never WHAT is computed (same data + model =>
same result; accuracy is a correctness check, never the objective).

================================================================================================
DECISION PROCEDURE (pseudocode; each step is the named sub-planner, see the matching module):

    schedule(tgi, cluster, *, link_gbps, feat_dim, num_snapshots, window, reuse_frac,
             cpu_agg_gbps, decomposition, cap_cells, prior, seed):

      # ---- STEP 1. PROBE + CALIBRATE  (profiler.prober) ----------------------------------
      probe  = probe_hardware(cluster)                 # spec-sheet or measured hardware
      calib  = calibrate(probe, tgi.stats,             # -> w_S, w_T (THEORY §2 link weights),
                         feat_dim=feat_dim,             #    sec_per_edge from measured bandwidth.
                         window=window)                 #    These w_S/w_T are what the C++ supra
                                                         #    solver consumes verbatim.

      # ---- STEP 2. ALLOCATE = the MIDDLE-END corner  (partition.allocate -> C++ kernels) --
      alloc  = allocate(tgi, calib,                    # cut -> axis -> byte-balance, in ONE plan.
                        decomposition=decomposition,    #   NODE axis : supra_solver (C++) cell_device
                        cap_cells=cap_cells,            #               + arrange (C++), min cost wins;
                        prior=prior.allocation,         #   FEATURE/HYBRID : column-shard plan.
                        seed=seed)                       #   alloc.assignment is vertex->device (int32).

      # ---- STEP 3. PLAN MEMORY = tiering / feasibility  (schedule.planner) ----------------
      w      = Workload(N, E, feat_dim, layers, window, reuse_frac,
                        feat_bytes=tgi.feat_bytes,       #   F_v-aware tiering when given
                        assignment=alloc.assignment)
      mem    = plan_memory(cluster, w)                  #   per-device HBM<->CPU tiering (no-OOM)

      # ---- STEP 4. BUFFER POOL = Belady/MRD over the known snapshot future  (runtime.bufferpool)
      bp     = [pool_from_plan(mem, w, d,               #   per device: the staged-bytes reduction.
                  num_snapshots=num_snapshots,           #   belady for a fixed DTDG schedule,
                  window=window,                          #   mrd for an evolving CTDG stream.
                  policy=('mrd' if tgi.mode=='ctdg' else 'belady'))
                for d in range(D)]

      # ---- STEP 5. CO-EXECUTION = CPU||GPU overlap on the tiered devices  (runtime.coexec) -
      ce     = plan_coexec(mem, cluster, w, cpu_agg_gbps=cpu_agg_gbps)

      # ---- STEP 6. RECOMBINE = feature-axis integration cost  (runtime.feature_recombine) --
      rc     = plan_recombine(alloc.cols_per_device, F, N, layers, link)  if axis in {feature,hybrid}
               else None

      # ---- STEP 7. MAKESPAN + BOUND ------------------------------------------------------
      back_ms = max(coexec_makespan_ms(ce), mem.makespan_sec*1e3) + (rc.recombine_ms if rc else 0)
      bound   = mem.bound (or coexec bound)

      # ---- STEP 8. INCREMENTAL = migration vs the prior batch  (schedule.dynamic) ---------
      incr   = plan_incremental(...) if prior is not None else None

      return SchedulePlan{calibration=calib, allocation=alloc, memory=mem, bufferpool=bp,
                          coexec=ce, recombine=rc, makespan_ms=back_ms, bound=bound,
                          incremental=incr, note=...}

COMPLEXITY:
    The scheduler body itself is O(D + S):  D = #devices (the per-device plan loops, the coexec
    split, the buffer-pool unit sizing), S = #snapshots (the buffer-pool access sequence length is
    O(S*window*epochs) but the simulation is delegated to the C++ kernel / numpy fallback, not the
    conductor). The genuinely super-linear / large-constant work (supra-cell build O(E log E),
    supra partition O(rounds*(E_S+E_T)), cache sim O(L), changed-cone O(n_new)) all lives behind
    the C++ kernels the sub-planners call. So adding a device or a snapshot costs the conductor a
    handful of float operations -- it is a policy layer, not a pass over the graph.

ESTIMATE (M3):  estimate_total_time() = front(import) + middle(arrange) + back(per-epoch)*epochs,
    where front = tgi.stats.ingest_sec (the FRONT-END ingest+probe wall time), middle = the
    allocate/arrange cost (we expose the measured allocate wall time, defaulting to the analytic
    makespan when not measured), and back = the per-epoch makespan_ms. This is the single number
    the CLI / driver budgets a job against; FULL PRECISION, hardware as a parameter.

IMPORT-SAFETY:  numpy + dataclasses only at module load. torch is NEVER imported here (the GNN
    execution lives in runtime.memtier.TieredExecutor, driven by a separate runtime driver). The
    sibling ``partition.allocate`` module is imported LAZILY inside schedule() so this module loads
    and is type-checkable even before allocate.py is built (matching the codebase's lazy-sibling
    pattern, e.g. dynamic_online._load_scheduler).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

# --- BACK-END cost arithmetic (all pure numpy / dataclasses; torch-free) -----------------
from .planner import Workload, GlobalPlan, plan_memory
from .dynamic import plan_incremental, IncrementalPlan
from ..profiler.prober import probe_hardware, calibrate, CostCalibration
from ..runtime.bufferpool import pool_from_plan, BufferPoolPlan
from ..runtime.coexec import plan_coexec, coexec_makespan_ms, CoExecPlan
from ..runtime.feature_recombine import plan_recombine, RecombineSpec

if TYPE_CHECKING:  # type-only: keep the module import-safe before allocate.py exists
    from ..frontend.ingest import TemporalGraphInput
    from ..profiler.cluster_profile import ClusterProfile
    from ..partition.allocate import AllocationPlan

GB: int = 1024 ** 3


# --------------------------------------------------------------------------- #
#  Lazy sibling import (allocate.py is a NEW module built by another owner)    #
# --------------------------------------------------------------------------- #
def _load_allocate():
    """Import partition.allocate.allocate lazily so this module loads even before allocate.py
    is present (mirrors dynamic_online._load_scheduler). Raises a clear error only when the
    scheduler is actually CALLED without allocate available -- import of scheduler stays safe."""
    from ..partition.allocate import allocate as _allocate  # type: ignore
    return _allocate


# --------------------------------------------------------------------------- #
#  The per-job time estimate (M3): front + middle + back*epochs               #
# --------------------------------------------------------------------------- #
@dataclass
class JobEstimate:
    """The M3 end-to-end wall-clock estimate, decomposed into the three pipeline stages.

    front_sec  : FRONT-END ingest + probe time (one-time graph import). Taken from
                 GraphStats.ingest_sec (the wall time ingest() folded in).
    middle_sec : MIDDLE-END allocate/arrange time (one-time partition + placement). The conductor
                 reports the MEASURED allocate wall time when the caller threads one in; otherwise
                 it falls back to the analytic per-epoch makespan as a conservative proxy (the
                 partition cost is bounded by one aggregation pass over the graph).
    back_per_epoch_sec : BACK-END per-epoch makespan (the tiering + co-exec + recombine step).
    num_epochs : training epochs the back-end pays.
    total_sec  : front + middle + back_per_epoch * num_epochs (the budgeted job time).
    """
    front_sec: float
    middle_sec: float
    back_per_epoch_sec: float
    num_epochs: int

    @property
    def total_sec(self) -> float:
        return float(self.front_sec + self.middle_sec
                     + self.back_per_epoch_sec * max(0, int(self.num_epochs)))

    def summary(self) -> str:
        return (f"[job estimate] total~={self.total_sec*1e3:.1f}ms = "
                f"front(import) {self.front_sec*1e3:.1f}ms + "
                f"middle(arrange) {self.middle_sec*1e3:.1f}ms + "
                f"back(per-epoch) {self.back_per_epoch_sec*1e3:.1f}ms x {self.num_epochs} epochs")


# --------------------------------------------------------------------------- #
#  The single object the CLI prints and dynamic_online carries between steps   #
# --------------------------------------------------------------------------- #
@dataclass
class SchedulePlan:
    """The ONE coherent schedule the conductor assembles (FRONT -> MIDDLE -> BACK).

    This is what the CLI prints, the runtime driver executes (via runtime.memtier.TieredExecutor),
    and dynamic_online carries between online steps. ``.partition_plan`` exposes the allocation so
    dynamic_online._assignment_of finds ``.partition_plan.assignment`` (the duck-typed contract).

    calibration : the self-calibrated CostCalibration (hardware + w_S/w_T) STEP 1 produced.
    allocation  : the AllocationPlan (assignment / cell_device / axis / cuts / byte balance) STEP 2.
    memory      : the per-device HBM<->CPU tiering GlobalPlan (no-OOM feasibility) STEP 3.
    bufferpool  : per-device BufferPoolPlan (Belady/MRD staged-bytes reduction) STEP 4.
    coexec      : per-device CoExecPlan (CPU||GPU overlap on the tiered devices) STEP 5.
    recombine   : the feature-axis integration cost (None for the node axis) STEP 6.
    makespan_ms : the BACK-END per-epoch makespan (the slowest device's overlapped step + recombine).
    bound       : what limits the makespan ("compute" | "pcie-staging" | "interconnect-comm" |
                  "cpu-coexec" | "infeasible").
    incremental : the migration plan vs the prior batch (None on a cold start) STEP 8.
    epochs      : training epochs (drives estimate_total_time's back contribution).
    note        : free-form provenance.
    """
    calibration: "CostCalibration"
    allocation: "AllocationPlan"
    memory: "GlobalPlan"
    bufferpool: list  # list[BufferPoolPlan]
    coexec: list      # list[CoExecPlan]
    recombine: "Optional[RecombineSpec]" = None
    makespan_ms: float = 0.0
    bound: str = "compute"
    incremental: "Optional[IncrementalPlan]" = None
    epochs: int = 1
    note: str = ""

    # ---- the duck-typed contract dynamic_online relies on -------------------------------
    @property
    def partition_plan(self):
        """The MIDDLE allocation, exposed under the name dynamic_online._assignment_of expects
        (it reads .partition_plan.assignment, falling back to .partition_plan.arrange.assignment)."""
        return self.allocation

    @property
    def assignment(self) -> Optional[np.ndarray]:
        """Convenience: the int32 vertex->device assignment (the layout the runtime executes)."""
        return getattr(self.allocation, "assignment", None)

    @property
    def feasible(self) -> bool:
        """Whether every device fits (the no-OOM guarantee holds on the back-end tiering)."""
        return bool(getattr(self.memory, "all_feasible", False))

    # ---- M3: front(import) + middle(arrange) + back(per-epoch)*epochs --------------------
    def estimate_total_time(self, num_epochs: Optional[int] = None, *,
                            front_sec: Optional[float] = None,
                            middle_sec: Optional[float] = None) -> JobEstimate:
        """The end-to-end job-time estimate (M3).

        num_epochs : epochs to budget the back-end against (defaults to self.epochs).
        front_sec  : override the FRONT ingest time (default: calibration.graph_stats.ingest_sec).
        middle_sec : override the MIDDLE allocate time (default: the AllocationPlan's measured
                     allocate wall time if it carries one, else the analytic per-epoch makespan as
                     a conservative proxy -- the partition is bounded by ~one aggregation pass).

        Returns a JobEstimate whose total_sec = front + middle + back_per_epoch * num_epochs.
        """
        epochs = int(num_epochs) if num_epochs is not None else int(self.epochs)
        # FRONT: the ingest+probe wall time the front-end folded into GraphStats.ingest_sec.
        if front_sec is None:
            gs = getattr(self.calibration, "graph_stats", None)
            front_sec = float(getattr(gs, "ingest_sec", 0.0) or 0.0)
        # MIDDLE: prefer a measured allocate wall time the AllocationPlan may carry; else use the
        # analytic per-epoch makespan as the conservative proxy for the one-time partition cost.
        if middle_sec is None:
            measured = getattr(self.allocation, "alloc_sec", None)
            middle_sec = float(measured) if measured is not None else float(self.makespan_ms) / 1e3
        back_per_epoch_sec = float(self.makespan_ms) / 1e3
        return JobEstimate(front_sec=float(front_sec), middle_sec=float(middle_sec),
                           back_per_epoch_sec=back_per_epoch_sec, num_epochs=epochs)

    # ---- the composed report ------------------------------------------------------------
    def summary(self) -> str:
        """Compose the child summaries into one report (the CLI prints this)."""
        lines = [
            f"================ zord SchedulePlan ================",
            f"makespan(per-epoch)~={self.makespan_ms:.2f}ms  bound={self.bound}  "
            f"feasible={self.feasible}" + (f"  [{self.note}]" if self.note else ""),
        ]
        # STEP 1
        if self.calibration is not None and hasattr(self.calibration, "summary"):
            lines.append(self.calibration.summary())
        # STEP 2
        if self.allocation is not None and hasattr(self.allocation, "summary"):
            lines.append(self.allocation.summary())
        elif self.allocation is not None:
            axis = getattr(self.allocation, "axis", "node")
            ws = getattr(self.allocation, "weighted_cost", float("nan"))
            sc = getattr(self.allocation, "spatial_cut", -1)
            tc = getattr(self.allocation, "temporal_cut", -1)
            lines.append(f"[allocation] axis={axis} spatial_cut={sc} temporal_cut={tc} "
                         f"weighted_cost={ws:.4g}")
        # STEP 3
        if self.memory is not None and hasattr(self.memory, "summary"):
            lines.append(self.memory.summary())
        # STEP 4
        if self.bufferpool:
            lines.append("buffer pool:")
            for bp in self.bufferpool:
                if hasattr(bp, "summary"):
                    lines.append("  " + bp.summary().replace("\n", "\n  "))
        # STEP 5
        if self.coexec:
            lines.append("co-execution:")
            for cp in self.coexec:
                if hasattr(cp, "summary"):
                    lines.append("  " + cp.summary())
            lines.append(f"  coexec makespan = {coexec_makespan_ms(self.coexec):.2f}ms")
        # STEP 6
        if self.recombine is not None and hasattr(self.recombine, "summary"):
            lines.append("recombine: " + self.recombine.summary())
        # STEP 8
        if self.incremental is not None:
            ic = self.incremental
            lines.append(
                f"incremental: moved={ic.moved_vertices:,} vertices "
                f"({ic.migrated_bytes/1e6:.2f} MB, {ic.migration_sec*1e3:.2f}ms over the link) "
                f"new={ic.new_vertices:,}  [{ic.note}]")
        lines.append(f"==================================================")
        return "\n".join(lines)


# =========================================================================================== #
#  THE ONE ENTRY -- the coherent FRONT -> MIDDLE -> BACK scheduler.                            #
# =========================================================================================== #
def schedule(tgi: "TemporalGraphInput", cluster: "ClusterProfile", *,
             link_gbps: Optional[float] = None, feat_dim: int = 128,
             num_snapshots: int = 64, window: int = 1, layers: int = 2,
             reuse_frac: float = 0.0, cpu_agg_gbps: float = 20.0,
             decomposition: str = "auto", cap_cells: int = 0,
             prior: "Optional[SchedulePlan]" = None, num_epochs: int = 1,
             seed: int = 0, measure: bool = False) -> SchedulePlan:
    """Produce the ONE coherent SchedulePlan for one temporal batch (the L3 conductor).

    This is the single entry the CLI and dynamic_online call. It runs the eight-step decision
    procedure documented in the module docstring, stringing the proven sub-planners together:
    probe/calibrate (prober) -> allocate (the MIDDLE corner, calls the C++ supra solver/build +
    arrange) -> plan_memory + bufferpool + coexec + recombine (the BACK-END) -> assemble.

    tgi          : the FRONT-END TemporalGraphInput (graph + per-edge snap + feat_bytes + stats +
                   mode). The ONE object FRONT hands MIDDLE; the scheduler reads tgi.stats for the
                   calibration, tgi.snap/feat_bytes through allocate, and tgi.mode to pick the
                   buffer-pool policy (mrd for an evolving ctdg stream, belady for a fixed dtdg one).
    cluster      : the hardware PROFILE -- the PARAMETER. Per-device HBM capacity + achieved
                   aggregation bandwidth + the interconnect bandwidth. HOMOGENEOUS (all devices
                   identical -> the bandwidth-proportional splits become even) and HETEROGENEOUS
                   (unequal devices -> work flows to the fast/big devices) are both handled; nothing
                   is HBM-/NVLink-specific.
    link_gbps    : interconnect bandwidth override (GB/s). Default: the probe's bottleneck link.
    feat_dim     : feature width F (folded into every byte cost).
    num_snapshots: S, the supra-graph time axis resolution + the buffer-pool schedule length.
    window       : co-resident snapshots per batch (the temporal batch -> tiering pressure).
    layers       : GNN depth (activation copies + recombine layer count + recompute cost).
    reuse_frac   : rho, the cross-snapshot reuse fraction (scales per-epoch staging/recompute).
    cpu_agg_gbps : the CPU co-executor's aggregation bandwidth (GB/s) for the coexec balance point.
    decomposition: "auto" (cost node vs feature vs hybrid, pick the cheapest feasible -- the
                   default), "node" (force node-parallel), or "feature"/"hybrid".
    cap_cells    : per-device supra-cell capacity cap passed to the C++ solver (0 = no cap).
    prior        : the previous step's SchedulePlan; its .allocation seeds the incremental
                   migration and its layout the changed-cone re-arrangement.
    num_epochs   : epochs the back-end pays (drives estimate_total_time).
    measure      : if True AND torch/CUDA present, the probe microbenchmarks achieved bandwidth;
                   default False (the torch-free spec-sheet path) so the conductor runs on a CPU box.

    Returns a SchedulePlan. PROCESS-ONLY: decides layout/staging/axis, never the computed result.
    """
    # ===== STEP 1. PROBE + CALIBRATE (profiler.prober) ===================================
    probe = probe_hardware(cluster, measure=measure)
    link = float(link_gbps) if link_gbps is not None else float(probe.link_gbps)
    stats = getattr(tgi, "stats", None)
    calib = calibrate(probe, stats, feat_dim=feat_dim, window=window, layers=layers)

    D = cluster.num_devices
    N = int(tgi.num_nodes)
    E = int(tgi.num_edges)

    # ===== STEP 2. ALLOCATE -- the MIDDLE-END corner (partition.allocate -> C++ kernels) ==
    # cut -> axis -> byte-balance composed into ONE AllocationPlan. allocate() reaches the C++
    # supra_solver/supra_build + arrange; it is the heavy MIDDLE pass, NOT the conductor's job.
    _allocate = _load_allocate()
    prior_alloc = getattr(prior, "allocation", None) if prior is not None else None
    alloc = _allocate(tgi, calib, decomposition=decomposition, cap_cells=cap_cells,
                      seed=seed, prior=prior_alloc)
    axis = getattr(alloc, "axis", "node")
    assignment = getattr(alloc, "assignment", None)

    # ===== STEP 3. PLAN MEMORY -- per-device HBM<->CPU tiering / feasibility (planner) ====
    # F_v-aware tiering when feat_bytes is present (spill the largest-F rows first); the
    # assignment ties each device to its TRUE feature bytes. Hardware split is bandwidth-
    # proportional inside plan_memory, so homogeneous => even, heterogeneous => weighted (N1).
    feat_bytes = getattr(tgi, "feat_bytes", None)
    # EDGE FEATURES into the memory account (attributes-first core): a resident edge carries its
    # F_e feature bytes alongside the 12-byte src/dst/w record (jodie 172-dim = 688 B/edge dwarfs
    # the structure record). fe_dim=0 -> bytes_per_edge=12, bit-identical to before.
    fe_dim = int(getattr(tgi, "edge_feat_dim", 0) or 0)
    if fe_dim == 0:
        ef = getattr(getattr(tgi, "graph", None), "efeat", None)
        if ef is not None and getattr(ef, "ndim", 0) == 2:
            fe_dim = int(ef.shape[1])
    workload = Workload(
        num_nodes=N, num_edges=E, feat_dim=feat_dim, layers=layers, window=window,
        reuse_frac=reuse_frac, feat_bytes=feat_bytes,
        bytes_per_edge=12 + 4 * fe_dim,
        assignment=(np.asarray(assignment) if (feat_bytes is not None and assignment is not None)
                    else None))
    mem = plan_memory(cluster, workload)

    # ===== STEP 4. BUFFER POOL -- Belady/MRD over the known snapshot future (bufferpool) ==
    # belady for a fixed DTDG schedule (the future IS the deterministic window order); mrd for an
    # evolving CTDG stream (the far future is unknown). One plan per device.
    bp_policy = "mrd" if getattr(tgi, "mode", "dtdg") == "ctdg" else "belady"
    bufferpool: list = []
    for d in range(D):
        try:
            bufferpool.append(pool_from_plan(
                mem, workload, device=d, num_snapshots=num_snapshots, window=window,
                num_epochs=num_epochs, policy=bp_policy))
        except Exception:
            # a device with no feasible plan / degenerate pool -> skip it; the makespan below
            # already reflects the infeasibility via mem.bound. The conductor must not crash.
            pass

    # ===== STEP 5. CO-EXECUTION -- CPU||GPU overlap on the tiered devices (coexec) ========
    coexec = plan_coexec(mem, cluster, workload, cpu_agg_gbps=cpu_agg_gbps)

    # ===== STEP 6. RECOMBINE -- the FEATURE-axis integration cost (feature_recombine) =====
    # Attached only when the MIDDLE chose the feature / hybrid axis (the column-shard W-mix
    # all-reduce is the integration the back-end pays per layer). None for the node axis.
    recombine: Optional[RecombineSpec] = None
    if axis in ("feature", "hybrid"):
        cols = _cols_per_device(alloc, feat_dim, D)
        if cols is not None:
            recombine = plan_recombine(cols, feat_dim, N, layers, link, full_layer=True)

    # ===== STEP 7. MAKESPAN + BOUND ======================================================
    # The back-end per-epoch makespan is the slowest device's OVERLAPPED step (coexec already folds
    # the resident compute + the exposed staging) -- but never below the memory plan's own makespan
    # (which captures the pure-tiering bound when there is no cold pool to co-execute). The feature
    # axis adds its recombine all-reduce on top (it is exposed cross-device comm, not overlapped).
    coexec_ms = coexec_makespan_ms(coexec)
    mem_ms = (mem.makespan_sec * 1e3) if np.isfinite(mem.makespan_sec) else float("inf")
    back_ms = max(coexec_ms, mem_ms)
    if recombine is not None and np.isfinite(back_ms):
        back_ms += float(recombine.recombine_ms)
    # the bound: memory plan's bound unless co-exec is the binding constraint on a tiered device.
    if not mem.all_feasible:
        bound = "infeasible"
    elif coexec_ms > mem_ms:
        # the slowest device is co-exec bound -> report whichever branch dominates there.
        bound = _coexec_bound(coexec)
    else:
        bound = mem.bound

    # ===== STEP 8. INCREMENTAL -- migration vs the prior batch (schedule.dynamic) =========
    incremental: Optional[IncrementalPlan] = None
    prior_assign = getattr(prior_alloc, "assignment", None) if prior_alloc is not None else None
    if prior is not None and prior_assign is not None and assignment is not None:
        src, dst = _graph_src_dst(tgi)
        incremental = plan_incremental(
            src, dst, N, D, np.asarray(prior_assign, dtype=np.int32),
            new_edge_lo=0, migration_budget=0.05, mem_dim=feat_dim, link_gbps=link)

    note = (f"axis={axis}; mode={getattr(tgi, 'mode', 'dtdg')}; bp={bp_policy}; "
            f"D={D} link={link:.3g}GB/s")
    return SchedulePlan(
        calibration=calib, allocation=alloc, memory=mem, bufferpool=bufferpool,
        coexec=coexec, recombine=recombine, makespan_ms=float(back_ms), bound=bound,
        incremental=incremental, epochs=int(num_epochs), note=note)


# --------------------------------------------------------------------------- #
#  Small POLICY helpers (closed-form / O(D); torch-free)                       #
# --------------------------------------------------------------------------- #
def _cols_per_device(alloc, feat_dim: int, D: int) -> Optional[np.ndarray]:
    """Per-device feature-COLUMN counts for the recombine cost on the feature/hybrid axis.

    Prefer the column split the AllocationPlan carries (the feature-parallel plan it composed);
    fall back to an even F/D split (the canonical column-parallel layout) so the recombine cost is
    always defined when the axis is feature/hybrid. Returns int64[D] summing to feat_dim, or None
    when D <= 0."""
    if D <= 0:
        return None
    # an AllocationPlan may expose the column split via its DecompositionChoice or a direct field.
    cols = getattr(alloc, "cols_per_device", None)
    if cols is None:
        dec = getattr(alloc, "decomposition", None)
        cols = getattr(dec, "feature_cols_per_device", None) if dec is not None else None
    if cols is not None:
        cols = np.asarray(cols, dtype=np.int64)
        if int(cols.sum()) == int(feat_dim) and cols.size == D:
            return cols
    # even F/D split: base width to each device + the remainder spread over the first devices.
    base = int(feat_dim) // D
    rem = int(feat_dim) - base * D
    cols = np.full(D, base, dtype=np.int64)
    if rem > 0:
        cols[:rem] += 1
    return cols


def _coexec_bound(coexec: list) -> str:
    """The bound of the slowest co-exec device (the one setting the makespan)."""
    finite = [cp for cp in coexec if np.isfinite(getattr(cp, "overlapped_ms", float("inf")))]
    if not finite:
        return "infeasible"
    worst = max(finite, key=lambda cp: cp.overlapped_ms)
    b = getattr(worst, "bound", "compute")
    # map the coexec branch names onto the scheduler's bound vocabulary.
    if b == "cpu-coexec":
        return "cpu-coexec"
    if b == "gpu":
        return "compute"
    return b


def _graph_src_dst(tgi):
    """Pull (src, dst) int64 arrays out of a TemporalGraphInput's .graph (duck-typed)."""
    g = getattr(tgi, "graph", tgi)
    if hasattr(g, "sort_by_time"):
        g = g.sort_by_time()
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    return src, dst


__all__ = ["SchedulePlan", "JobEstimate", "schedule"]
