"""Global memory scheduler / planner (zord core, post-D25).

Given a heterogeneous cluster (per-GPU HBM capacity, ACHIEVED aggregation bandwidth, and
CPU<->HBM PCIe bandwidth) and a temporal-GNN workload (N nodes, E edges, F features, L layers,
a window of W co-resident snapshots), decide -- FULL PRECISION -- the global memory layout:

  1. WORK BALANCE   : split nodes across GPUs proportional to ACHIEVED HBM bandwidth (not equal,
                      not by FLOPs) -- because the step is bandwidth-bound (roofline, RESULTS §9),
                      so time ~ assigned_bytes / hbm_bw. This balances makespan on unequal GPUs.
  2. TIERING        : per device, how many of the W snapshots' state stay RESIDENT in HBM vs are
                      STAGED from CPU RAM over PCIe (when W*per-snapshot-state exceeds capacity).
                      Guarantees no-OOM (feasibility) -- the thing baselines fail at.
  3. REUSE          : a fraction rho of nodes whose neighborhood is unchanged between consecutive
                      snapshots is reused, not recomputed/re-staged -> scales per-snapshot work
                      (after the first) by (1-rho). (rho measured offline; the temporal lever.)
  4. TIME / OVERLAP : predict epoch time = max(compute, exposed-staging). With prefetch (double
                      buffering) the PCIe copy of snapshot s overlaps compute of s-1, so staging
                      is HIDDEN whenever compute >= staging; otherwise the device is PCIe-bound.

The plan is a PREDICTION validated by the runtime experiments (oom_to_tiered for staging/overlap,
roofline for hbm_bw, end_to_end for compute). It degenerates to "all resident, one placement"
(~METIS) when everything fits; becomes a tiering+reuse scheduler under memory pressure.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..profiler.cluster_profile import ClusterProfile

GB = 1024 ** 3


@dataclass
class Workload:
    num_nodes: int
    num_edges: int
    feat_dim: int = 128
    layers: int = 2                 # GraphSAGE depth -> activation copies kept for backward
    window: int = 1                 # snapshots co-resident in a batch (the temporal batch)
    bytes_per_feat: int = 4         # FULL PRECISION (fp32). Not a compression knob.
    bytes_per_edge: int = 12        # src(4)+dst(4)+w(4) for the resident CSR/COO
    reuse_frac: float = 0.0         # rho: fraction of nodes unchanged vs previous snapshot
    # OPTIONAL per-node feature-size VECTOR F_v (numpy [N], feature DIMS per node). None -> the
    # uniform scalar feat_dim path runs UNCHANGED (bit-identical). When given, the tiering layer
    # sizes each device by its ACTUAL feature bytes (sum_{v on k} F_v) and spills the LARGEST-F
    # cold rows first -- the §33 attribute tiering win. Pair with `assignment` so plan_memory
    # knows WHICH nodes live on each device (else it falls back to the uniform mean over F_v).
    feat_bytes: Optional[np.ndarray] = None
    assignment: Optional[np.ndarray] = None   # int [N] device per node (for F_v-aware tiering)

    @property
    def avg_degree(self) -> float:
        return self.num_edges / max(1, self.num_nodes)


@dataclass
class MemoryPlan:
    device: int
    name: str
    work_nodes: int                 # nodes assigned to this GPU (balance share)
    work_edges: int
    resident_snapshots: int         # snapshots whose state lives in HBM
    streamed_snapshots: int         # snapshots staged from CPU RAM per epoch
    peak_hbm_bytes: int             # predicted peak HBM use (<= capacity if feasible)
    capacity_bytes: int
    compute_sec: float              # per-epoch compute (bandwidth-bound aggregation)
    staging_sec: float              # per-epoch PCIe staging (CPU->HBM) for streamed snapshots
    exposed_staging_sec: float      # staging NOT hidden behind compute (the PCIe-bound part)
    epoch_sec: float                # max(compute, staging) under prefetch overlap
    feasible: bool
    note: str = ""


@dataclass
class GlobalPlan:
    per_device: list[MemoryPlan]
    makespan_sec: float
    bottleneck: int                 # device index whose epoch_sec dominates
    bound: str                      # "compute" | "pcie-staging" | "infeasible"
    total_streamed_gb: float
    all_feasible: bool

    def summary(self) -> str:
        lines = [f"GlobalPlan makespan={self.makespan_sec*1e3:.1f}ms bound={self.bound} "
                 f"bottleneck=dev{self.bottleneck} feasible={self.all_feasible} "
                 f"streamed={self.total_streamed_gb:.1f}GB/epoch"]
        for p in self.per_device:
            lines.append(
                f"  dev{p.device} {p.name:<16} nodes={p.work_nodes:>10} "
                f"resident={p.resident_snapshots}/{p.resident_snapshots+p.streamed_snapshots} "
                f"peak={p.peak_hbm_bytes/GB:5.1f}/{p.capacity_bytes/GB:4.1f}GB "
                f"compute={p.compute_sec*1e3:6.1f}ms stage={p.staging_sec*1e3:6.1f}ms"
                f"(exposed {p.exposed_staging_sec*1e3:5.1f}ms) epoch={p.epoch_sec*1e3:6.1f}ms"
                f"{'  ['+p.note+']' if p.note else ''}")
        return "\n".join(lines)


def _snapshot_state_bytes(n: int, edges: int, w: Workload,
                          feat_dims_total: Optional[float] = None) -> int:
    """HBM bytes for ONE snapshot's resident state on a device holding n nodes / `edges` edges:
    node features + L activation copies (kept for backprop) + the resident adjacency.

    feat_dims_total : OPTIONAL sum of per-node feature DIMS on this device (sum_{v on k} F_v).
                      When None -> n*feat_dim (the uniform scalar path, bit-identical). When
                      given -> the ACTUAL heterogeneous feature dims, so features + activations
                      scale with the true per-device feature bytes (the §33 footprint)."""
    fdims = (n * w.feat_dim) if feat_dims_total is None else feat_dims_total
    feat = fdims * w.bytes_per_feat
    activ = w.layers * fdims * w.bytes_per_feat
    adj = edges * w.bytes_per_edge
    return int(feat + activ + adj)


# --- The §40-engine-v2 under-prediction fix. The OLD peak model, peak = (resident+reserve)*per_snap,
# counted ZERO of the runtime memory the executor actually holds, so it UNDER-predicted (78903: 52.9GB
# predicted vs 67.6GB measured = 27.8% UNSAFE). Two structural terms + one allocator margin recover a
# CONSERVATIVE UPPER bound: ----------------------------------------------------------------------------
#  (1) DOUBLE-BUFFER second feature buffer: while streaming, the executor keeps TWO device prefetch
#      buffers in flight (gbuf[0]/gbuf[1], oom_engine_gpu.py:210) so the H2D copy of the NEXT chunk
#      overlaps compute of the CURRENT one. reserve_buffers reserves ONE snapshot slot in the count;
#      the SECOND in-flight device feature copy (1 x feat) is omitted -> add it.
#  (2) FORWARD-PASS TRANSIENTS: each aggregation materializes several LIVE [n,F] intermediates ON TOP
#      of the persistent feature+activation banks (A@X, relu, .@W, mm(A,.) in aggregate()), which the
#      old model counted as zero. Sized at FWD_TRANSIENT_COPIES one-snapshot feature tensors.
FWD_TRANSIENT_COPIES = 3
#  (3) ALLOCATOR OVERHEAD: torch.cuda.max_memory_allocated() (the MEASURED peak) includes CUDA caching-
#      allocator fragmentation/rounding ON TOP of the live tensor bytes. Empirically the measured peak
#      ran ~1.23x the live-tensor estimate on 78903 (55.0GB live -> 67.6GB measured). We inflate the
#      whole peak by ALLOC_OVERHEAD (proportional, not a fixed count, since fragmentation scales with
#      the working set) so predicted_peak stays a conservative UPPER bound on the MEASURED peak.
ALLOC_OVERHEAD = 0.25
#  (D1 fix) ALL-RESIDENT (non-tiered) F_v PEAK: the §47 overhead above was reserved on the SCALAR/uniform
#  tiering path but NOT on the F_v ALL-RESIDENT path, where the working set fits WITHOUT spilling and the
#  plan exits through the row-tiering branch with a degenerate (0-byte) spill. That path reported the bare
#  resident-bank bytes and so UNDER-predicted ~10% (§46-wiki: predicted 63.5 vs measured 70.3GB). The F_v
#  column-sweep executor (oom_attr_gpu.py) holds the persistent feature+activation bank AND, per forward
#  step, a LIVE working set of one feature block + its activation copies, on top of which the CUDA caching
#  allocator adds fragmentation/rounding. We CANNOT reuse the uniform transient (FWD_TRANSIENT_COPIES x the
#  WHOLE F_v feature mass) -- that is ~3x the entire bank and would wildly over-predict / falsely flip
#  feasible (the §46 note). Instead we apply the SAME multiplicative allocator margin (ALLOC_OVERHEAD) to a
#  BOUNDED live forward working set = the resident feature bytes + one live activation copy (== 2x the
#  feature bytes, since each layer's activation is feature-sized). This recovers a CONSERVATIVE upper bound
#  (predicted 74.1 >= measured 70.3 on §46-wiki) without inflating the whole multi-snapshot bank.
ALLRESIDENT_LIVE_ACTIV_COPIES = 1


def _runtime_overhead_bytes(feat_one_snapshot_bytes: float, streaming: bool) -> int:
    """Structural HBM the EXECUTOR holds beyond the counted resident snapshot state (terms (1)+(2)
    above): the second double-buffer device feature copy (when streaming) + the forward-pass live
    intermediates. The multiplicative allocator margin (3) is applied to the FULL peak separately.

    feat_one_snapshot_bytes : node-feature bytes for ONE snapshot on this device (n*F*4, or the F_v
                              per-device feature bytes). Activations/adjacency are NOT re-added here.
    streaming               : whether any snapshot/row streams (adds the double-buffer term)."""
    double_buffer = feat_one_snapshot_bytes if streaming else 0.0   # the 2nd in-flight device buffer
    transients = FWD_TRANSIENT_COPIES * feat_one_snapshot_bytes     # live forward-pass intermediates
    return int(double_buffer + transients)


def plan_memory(cluster: ClusterProfile, w: Workload,
                prefetch: bool = True, reserve_buffers: int = 1) -> GlobalPlan:
    """Produce the global memory plan. Work is split proportional to achieved HBM bandwidth
    (bandwidth-bound step); each device then tiers the window between HBM and CPU staging to
    guarantee fit; epoch time is max(compute, staging) under prefetch overlap."""
    devs = cluster.devices
    nd = len(devs)
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=float)

    # ---- per-device node set + feature DIMS. Two paths, additively chosen: -----------------
    #  (A) F_v-AWARE (w.feat_bytes + w.assignment given): use the ACTUAL arrange assignment and
    #      per-node feature dims so each device's feature memory is the TRUE sum_{v on k} F_v.
    #  (B) default (scalar): bandwidth-proportional node split x uniform feat_dim -- UNCHANGED,
    #      bit-identical to before (feat_dims_total stays None -> n_k*feat_dim everywhere).
    Fv = None if w.feat_bytes is None else np.asarray(w.feat_bytes, dtype=np.float64)
    assign = None if w.assignment is None else np.asarray(w.assignment, dtype=np.int64)
    aware_tier = Fv is not None and assign is not None
    if aware_tier:
        # vertex-cut leaves assign==-1 for REPLICATED-CORE nodes (they live on EVERY device).
        # bincount cannot take -1 -> count only single-homed periphery, then ADD the replicated
        # core's count + feature bytes to ALL devices (mirrors arrange._feat_bytes_per_dev, which
        # uses core_mask). Guard fixes the crash AND the latent core undercount (auditor a12ff3).
        homed = assign >= 0
        node_share = np.bincount(assign[homed], minlength=nd).astype(np.int64)
        fdims_dev = np.bincount(assign[homed], weights=Fv[homed], minlength=nd).astype(np.float64)
        core = ~homed
        if core.any():
            node_share += int(core.sum())
            fdims_dev += float(Fv[core].sum())
    else:
        share = bw / bw.sum()                              # bandwidth-proportional work split
        node_share = np.maximum(1, np.round(share * w.num_nodes)).astype(np.int64)
        node_share[-1] += w.num_nodes - int(node_share.sum())
        node_share = np.maximum(1, node_share)
        fdims_dev = None

    plans: list[MemoryPlan] = []
    for k, d in enumerate(devs):
        n_k = int(node_share[k])
        edges_k = int(round(w.num_edges * n_k / max(1, w.num_nodes)))
        fdims_k = None if fdims_dev is None else float(fdims_dev[k])
        per_snap = _snapshot_state_bytes(n_k, edges_k, w, feat_dims_total=fdims_k)
        cap = d.usable_mem
        # RUNTIME OVERHEAD the executor holds beyond the counted snapshot state (the §40-engine-v2
        # under-prediction fix): the 2nd double-buffer device feature copy + forward-pass transients.
        # We RESERVE it out of capacity BEFORE deciding how many snapshots fit, so the conservative
        # peak (which re-adds this overhead below) provably stays <= cap -> the no-OOM guarantee holds.
        feat_one_snap = (n_k * w.feat_dim if fdims_k is None else fdims_k) * w.bytes_per_feat
        # whether tiering (streaming) will happen drives the double-buffer term: tiering iff the window
        # cannot be held even after reserving the always-present transient headroom + allocator margin.
        full_resident_peak = (1.0 + ALLOC_OVERHEAD) * (
            w.window * per_snap + _runtime_overhead_bytes(feat_one_snap, False))
        will_stream = full_resident_peak > cap
        overhead = _runtime_overhead_bytes(feat_one_snap, streaming=will_stream)
        # capacity LEFT for snapshot state after reserving the runtime overhead AND inflating by the
        # allocator margin, so the conservative peak (recomputed below) provably stays <= cap (no-OOM).
        budget_for_snaps = cap / (1.0 + ALLOC_OVERHEAD) - overhead
        fit = int(budget_for_snaps // max(1, per_snap))
        feasible = fit >= 1
        if not feasible:
            # A single snapshot does not fit. SCALAR path: declare infeasible (intra-snapshot node
            # tiling needed). F_v-AWARE path: TIER the feature bank at ROW granularity -- spill the
            # LARGEST-F rows to CPU until the resident feature bytes + edge/activation state fit
            # (the §33 tiering win: evicting the biggest rows frees the most HBM per spill). The
            # blind alternative (evict-by-count) would spill the small rows first and still OOM.
            if aware_tier:
                on = np.nonzero(assign == k)[0]
                row_bytes = Fv[on] * w.bytes_per_feat * (1 + w.layers)   # feat + L activation copies
                adj = edges_k * w.bytes_per_edge
                budget = cap - adj
                resident_total = float(row_bytes.sum())
                if budget <= 0:
                    plans.append(MemoryPlan(k, d.name, n_k, edges_k, 0, n_k, per_snap, cap,
                                            0.0, 0.0, 0.0, float("inf"), False,
                                            note="edge metadata alone exceeds HBM"))
                    continue
                # spill largest-F rows first: streamed = smallest bytes to free so resident <= budget
                order = np.argsort(-row_bytes, kind="stable")       # largest-F first
                csum = np.cumsum(row_bytes[order])
                need_free = resident_total - budget
                n_spill = int(np.searchsorted(csum, need_free, side="left") + 1)
                n_spill = min(n_spill, on.size)
                streamed_bytes = float(row_bytes[order[:n_spill]].sum())
                resident_bytes = resident_total - streamed_bytes + adj
                # NOTE: when the F_v bank ACTUALLY spills (need_free > 0), no _runtime_overhead_bytes term:
                # the executor (oom_attr_gpu.py) sweeps the resident feature mass in BOUNDED column blocks
                # (only BLK cols + L activations are live at once), whereas resident_bytes already charges
                # the ENTIRE resident feature mass as co-resident -- so this spilling peak is ALREADY a
                # conservative over-estimate (RESULTS §46: predicted 78.0 >= measured 71.5). Adding the
                # uniform per-snapshot transient (3x the full feature mass) would wildly over-predict and
                # falsely flip feasible. This spilling path is left BIT-IDENTICAL.
                # D1 fix -- ALL-RESIDENT (non-tiered) sub-case (need_free <= 0: the whole bank fits without
                # any real spill; the §46-wiki path that reported the bare 63.5GB and under-predicted the
                # 70.3GB measured peak). Here resident_bytes is the persistent bank with ZERO headroom for
                # the executor's live forward working set + CUDA allocator fragmentation, so we reserve the
                # §47 conservative overhead: apply the ALLOC_OVERHEAD margin to the BOUNDED live working set
                # (resident features + ALLRESIDENT_LIVE_ACTIV_COPIES activation copies, each feature-sized).
                # Bounded => it does NOT re-inflate the whole bank (which would exceed the cap) yet makes
                # predicted >= measured. The real-spilling branch above is untouched.
                if need_free <= 0.0:
                    feat_only = float((Fv[on] * w.bytes_per_feat).sum())      # resident feature bytes
                    live_working_set = feat_only * (1 + ALLRESIDENT_LIVE_ACTIV_COPIES)
                    resident_bytes = resident_bytes + ALLOC_OVERHEAD * live_working_set
                fits = resident_bytes <= cap + 1e-6
                # compute roofline (full feature gather still runs; spilled rows stream in to compute)
                agg_bytes = w.layers * (2 * edges_k * (fdims_k / max(1, n_k)) * w.bytes_per_feat)
                compute = agg_bytes / (d.hbm_bw_gbps * 1e9)
                staging = streamed_bytes / (d.h2d_gbps * 1e9)
                if prefetch:
                    exposed = max(0.0, staging - compute); epoch = compute + exposed
                else:
                    exposed = staging; epoch = compute + staging
                if need_free <= 0.0:
                    note = (f"F_v all-resident (no spill): bank {resident_bytes/GB:.1f}GB incl. §47 "
                            f"conservative runtime overhead -> fits" if fits
                            else "F_v all-resident bank + runtime overhead exceeds HBM")
                else:
                    note = (f"F_v-aware row tiering: spilled {n_spill} largest-F rows "
                            f"({streamed_bytes/GB:.1f}GB) -> fits" if fits
                            else "F_v-aware tiering still over (aggregate HBM insufficient)")
                plans.append(MemoryPlan(
                    k, d.name, n_k, edges_k, n_k - n_spill, n_spill, int(resident_bytes), cap,
                    compute, staging, exposed, epoch, bool(fits), note=note))
                continue
            plans.append(MemoryPlan(k, d.name, n_k, edges_k, 0, w.window, per_snap, cap,
                                    0.0, 0.0, 0.0, float("inf"), False,
                                    note="single snapshot exceeds HBM -> needs intra-snapshot node tiling"))
            continue
        resident = min(w.window, max(1, fit - reserve_buffers)) if w.window > fit else w.window
        resident = min(resident, w.window)
        streamed = w.window - resident
        peak = (min(resident + (reserve_buffers if streamed > 0 else 0), w.window)) * per_snap
        # ADD the structural runtime overhead (2nd double-buffer device feature copy when streaming +
        # forward-pass transients) THEN inflate by the allocator margin, so predicted_peak is a
        # CONSERVATIVE UPPER bound on the MEASURED peak (the §40-engine-v2 under-prediction fix).
        peak = int((1.0 + ALLOC_OVERHEAD) * (peak + overhead))

        # ---- compute (bandwidth-bound aggregation), full precision ----
        # per snapshot: L layers, each gathers nnz=2*edges_k rows of width F -> bytes ~ 2*edges_k*F*elem.
        # F_v-aware: use the device's MEAN feature dim (fdims_k/n_k) so heavy-F devices cost more.
        mean_F_k = w.feat_dim if fdims_k is None else (fdims_k / max(1, n_k))
        agg_bytes_per_snap = w.layers * (2 * edges_k * mean_F_k * w.bytes_per_feat)
        # reuse: only the first snapshot is full; the rest recompute the changed (1-rho) fraction
        eff_snaps = 1.0 + (w.window - 1) * (1.0 - w.reuse_frac)
        compute = eff_snaps * agg_bytes_per_snap / (d.hbm_bw_gbps * 1e9)

        # ---- staging (CPU->HBM over PCIe) for the streamed snapshots ----
        stage_bytes_per_snap = n_k * mean_F_k * w.bytes_per_feat   # features streamed in
        eff_streamed = streamed * (1.0 - w.reuse_frac) if streamed else 0
        staging = eff_streamed * stage_bytes_per_snap / (d.h2d_gbps * 1e9)

        if prefetch:
            exposed = max(0.0, staging - compute)          # compute hides staging if compute >= staging
            epoch = compute + exposed
        else:
            exposed = staging
            epoch = compute + staging                       # fully serialized

        note = "all resident (~single placement)" if streamed == 0 else \
               (f"tiered: {streamed} snapshot(s) staged from CPU"
                + (", PCIe-bound" if exposed > 0 else ", staging hidden by prefetch"))
        plans.append(MemoryPlan(k, d.name, n_k, edges_k, resident, streamed, int(peak), cap,
                                compute, staging, exposed, epoch, True, note))

    feas = [p for p in plans if p.feasible]
    all_feasible = len(feas) == nd
    if not feas:
        return GlobalPlan(plans, float("inf"), 0, "infeasible", 0.0, False)
    bott = max(range(nd), key=lambda k: plans[k].epoch_sec if plans[k].feasible else -1)
    makespan = plans[bott].epoch_sec
    bound = "pcie-staging" if plans[bott].exposed_staging_sec > 0 else "compute"
    if aware_tier:
        # streamed bytes are exact via staging_sec * h2d (row-tiering streams real per-row bytes,
        # not whole-snapshot multiples) -- robust for both the snapshot and the row-spill paths.
        total_streamed = sum(p.staging_sec * devs[p.device].h2d_gbps * 1e9 for p in feas) / GB
    else:
        total_streamed = sum(p.streamed_snapshots * p.work_nodes * w.feat_dim * w.bytes_per_feat
                             for p in feas) / GB
    return GlobalPlan(plans, makespan, bott, bound if all_feasible else "infeasible",
                      total_streamed, all_feasible)


# ============================================================================== #
# THE END-TO-END PLAN (the engine entry that embodies the validated algorithms).  #
#                                                                                 #
# Given a TemporalGraph + a cluster spec (device count, per-device HBM capacity +  #
# achieved-agg-bandwidth + interconnect bandwidth as a PARAMETER), zord returns a  #
# single Plan with: the partition ASSIGNMENT (adaptive-corner arrange), per-device  #
# PLACEMENT (node/edge/replication + HBM tiering), the VERTEX-CUT / replication      #
# decisions, the INCREMENTAL-MIGRATION plan vs the prior batch, and the predicted    #
# MAKESPAN + feasibility. Worst-case-optimal by construction: arrange <= min(        #
# candidates incl METIS). PROCESS-only: time / memory / feasibility; never accuracy. #
# ============================================================================== #
from ..partition.arrange import arrange, predict_ms, ArrangeResult   # noqa: E402
from ..partition.feature_parallel import (                            # noqa: E402
    feature_parallel_plan, FeatureParallelPlan, hybrid_plans, HybridPlan)
from .dynamic import plan_incremental, IncrementalPlan                # noqa: E402


@dataclass
class DevicePlacement:
    device: int
    name: str
    home_nodes: int                 # vertices homed on this device (single-home periphery)
    replicated_core: int            # replicated dense-core rows resident here (vertex-cut)
    incident_edges: int             # incident-edge gather work assigned here
    capacity_bytes: int
    resident_bytes: int             # predicted resident HBM (feat rows + edge metadata)
    compute_ms: float               # roofline aggregation time (achieved HBM bandwidth)
    comm_ms: float                  # boundary feature rows over the interconnect parameter
    epoch_ms: float                 # compute + comm for this device
    feasible: bool


@dataclass
class Plan:
    """The end-to-end ZORD plan for one temporal batch."""
    strategy: str                   # winning arrange candidate (edge-cut / vertex-cut / ...)
    assignment: np.ndarray          # int32 [N] device per vertex (periphery home for vertex-cut)
    core_mask: Optional[np.ndarray] # bool [N] replicated dense core (vertex-cut), else None
    placement: list                 # list[DevicePlacement]
    makespan_ms: float              # predicted makespan (bottleneck device epoch)
    bottleneck: int                 # device index that dominates
    bound: str                      # "compute" | "interconnect-comm" | "infeasible"
    feasible: bool                  # fits every device's HBM
    cut_edges: int                  # cross-device edges (single-home metric)
    replication_pct: float          # replicated-core rows as % of total resident rows
    candidate_makespans: dict       # {candidate: makespan_ms} -- honest reporting
    link_gbps: float                # the interconnect bandwidth used (parameter)
    incremental: Optional[IncrementalPlan] = None   # migration plan vs the prior batch
    memory: Optional[GlobalPlan] = None             # per-device HBM/PCIe tiering (plan_memory)
    # DECOMPOSITION-AXIS choice (None in the DEFAULT node-parallel path -> the plan is byte-identical
    # to before). Set only when plan(..., decomposition != "node"): carries the node-vs-feature-vs-
    # hybrid comparison + the winning axis (the §-attribute-dimension feature-parallel option).
    decomposition: "Optional[DecompositionChoice]" = None

    def summary(self) -> str:
        lines = [
            f"[zord plan] strategy={self.strategy}  makespan~={self.makespan_ms:.2f}ms  "
            f"bound={self.bound}  bottleneck=dev{self.bottleneck}  "
            f"feasible={self.feasible}  link={self.link_gbps:g}GB/s",
            f"  cut={self.cut_edges:,} edges   replication={self.replication_pct:.1f}%",
            "  candidate makespans (ms): " +
            ", ".join(f"{k}={v:.1f}" for k, v in self.candidate_makespans.items()),
        ]
        for p in self.placement:
            flag = "" if p.feasible else "  <-- OOM"
            lines.append(
                f"  dev{p.device} {p.name:<16} home={p.home_nodes:>10,} "
                f"+core={p.replicated_core:>8,} inc_edges={p.incident_edges:>11,} "
                f"resident={p.resident_bytes/GB:5.1f}/{p.capacity_bytes/GB:4.1f}GB "
                f"compute={p.compute_ms:6.2f}ms comm={p.comm_ms:6.2f}ms epoch={p.epoch_ms:6.2f}ms{flag}")
        if self.incremental is not None:
            ic = self.incremental
            lines.append(
                f"  incremental: moved={ic.moved_vertices:,} vertices "
                f"({ic.migrated_bytes/1e6:.2f} MB node-memory, {ic.migration_sec*1e3:.2f}ms "
                f"over the link)  new={ic.new_vertices:,}  [{ic.note}]")
        if self.memory is not None and (not self.memory.all_feasible
                                         or self.memory.total_streamed_gb > 0):
            lines.append(f"  memory-tiering: streamed={self.memory.total_streamed_gb:.1f}GB/epoch "
                         f"bound={self.memory.bound} feasible={self.memory.all_feasible}")
        if self.decomposition is not None:
            d = self.decomposition
            lines.append(f"  decomposition-axis: WINNER={d.axis} (node={d.node_makespan_ms:.1f}ms"
                         f"{'' if d.node_feasible else '/OOM'}, feature={d.feature_makespan_ms:.1f}ms"
                         f"{'' if d.feature_feasible else '/OOM'}, integration={d.feature_integration_ms:.2f}ms)")
        return "\n".join(lines)


def _placement_from_arrange(res: ArrangeResult, cluster: ClusterProfile, feat_dim: int):
    """Turn the arrange result's per-device incident/comm/counts into DevicePlacements with
    the SAME incident-edge roofline (achieved HBM bandwidth for compute, the interconnect
    PARAMETER for comm) and the SAME footprint shape used for feasibility.

    F_v generalization: when the arrange result carries per-node feat_bytes folds (res.inc_folded
    / res.feat_bytes_dev set), the roofline uses the FEATURE-WEIGHTED work and the footprint uses
    the ACTUAL per-device feature bytes (the §33 win). When they are None this is byte-identical
    to the scalar path (incident*F roofline, counts*F*4 feature memory)."""
    from ..partition.arrange import BYTES_PER_EDGE_RESIDENT, feasible
    F = feat_dim
    bw = res.bw_gbps
    inc_work = res.inc_folded if res.inc_folded is not None else res.incident * F
    comm_work = res.comm_folded if res.comm_folded is not None else res.comm_rows * F
    tot_ms, comp_ms, comm_ms = predict_ms(inc_work, comm_work, bw, res.link_gbps)
    D = cluster.num_devices
    core_size = int(res.core_mask.sum()) if res.core_mask is not None else 0
    placements = []
    for k, d in enumerate(cluster.devices):
        home = int(res.counts[k] - core_size)          # counts incl. replicated core
        feat_b = (res.feat_bytes_dev[k] if res.feat_bytes_dev is not None
                  else res.counts[k] * F * 4.0)
        resident = feat_b + res.incident[k] * BYTES_PER_EDGE_RESIDENT
        fits = resident <= d.usable_mem
        placements.append(DevicePlacement(
            device=k, name=d.name, home_nodes=max(0, home), replicated_core=core_size,
            incident_edges=int(res.incident[k]), capacity_bytes=int(d.usable_mem),
            resident_bytes=int(resident), compute_ms=float(comp_ms[k]),
            comm_ms=float(comm_ms[k]), epoch_ms=float(tot_ms[k]), feasible=bool(fits)))
    all_feasible = all(p.feasible for p in placements)
    bott = int(np.argmax(tot_ms))
    bound = ("infeasible" if not all_feasible
             else ("interconnect-comm" if comm_ms[bott] > comp_ms[bott] else "compute"))
    return placements, float(tot_ms.max()), bott, bound, all_feasible


# ============================================================================== #
# DECOMPOSITION-AXIS CHOICE: node-parallel vs feature-parallel vs hybrid.          #
#                                                                                  #
# zord's default arrange splits the VERTICES (node-parallel: each device homes     #
# N/D rows x ALL F columns + pays boundary comm). For ATTRIBUTE-HEAVY graphs (high  #
# F) feature MEMORY (N*F*4) dominates -> FEATURE-parallel (each device holds the    #
# FULL graph + F/D columns, integrates via column-concat) relieves per-device HBM.  #
# This is a SECOND decomposition axis. choose_decomposition() costs BOTH pure axes  #
# + the hybrid 2D grids on the SHARED roofline and returns the lowest FEASIBLE      #
# makespan. PROCESS-only: same data+model => same result (column-shard+concat is    #
# bit-identical to single-device; see feature_parallel.fp_aggregate_consistency).   #
# ============================================================================== #
@dataclass
class DecompositionChoice:
    """The winning decomposition axis + the per-axis costed alternatives (honest reporting)."""
    axis: str                       # "node" | "feature" | "hybrid(DnxDf)"
    makespan_ms: float              # winning makespan among the FEASIBLE axes
    feasible: bool
    node_makespan_ms: float         # pure node-parallel (arrange) makespan
    node_feasible: bool
    feature_makespan_ms: float      # pure feature-parallel makespan
    feature_feasible: bool
    feature_integration_ms: float   # the integration (column-concat) comm on the FP bottleneck
    feature_cols_per_device: Optional[np.ndarray]  # F_d columns per device (feature-parallel)
    feature_feat_gb_per_device: Optional[np.ndarray]  # per-device feature GB (feature-parallel)
    candidate_makespans: dict       # {axis_name: (makespan_ms, feasible)}
    crossover_note: str             # human-readable why this axis won

    def summary(self) -> str:
        lines = [f"[decomposition] winner={self.axis}  makespan~={self.makespan_ms:.2f}ms  "
                 f"feasible={self.feasible}",
                 f"  node-parallel : makespan={self.node_makespan_ms:.2f}ms  feasible={self.node_feasible}",
                 f"  feature-parallel: makespan={self.feature_makespan_ms:.2f}ms  "
                 f"feasible={self.feature_feasible}  integration={self.feature_integration_ms:.2f}ms"]
        if self.feature_cols_per_device is not None:
            gbs = (self.feature_feat_gb_per_device
                   if self.feature_feat_gb_per_device is not None else [])
            lines.append("    cols/device=" + str(list(self.feature_cols_per_device)) +
                         "  feat GB/device=" + str([round(float(x), 2) for x in gbs]))
        lines.append("  candidate makespans (ms): " +
                     ", ".join(f"{k}={v[0]:.1f}{'' if v[1] else '(OOM)'}"
                               for k, v in self.candidate_makespans.items()))
        lines.append("  " + self.crossover_note)
        return "\n".join(lines)


def choose_decomposition(graph, cluster: ClusterProfile, *, feat_dim: int = 128,
                         link_gbps: Optional[float] = None, seed: int = 0,
                         num_snapshots: int = 64,
                         boundary_frac: float = 1.0,
                         feat_bytes: Optional[np.ndarray] = None) -> DecompositionChoice:
    """Cost the NODE-parallel (arrange), FEATURE-parallel (column-shard), and HYBRID 2D-grid plans
    on the SAME roofline and pick the lowest FEASIBLE makespan. This is the planner-level axis choice
    the user asked for -- it does NOT mutate the default plan() path (which stays node-parallel +
    bit-identical); it is invoked only when plan(..., decomposition="auto"|"feature"|"hybrid").

    Returns a DecompositionChoice with the winner + every costed alternative (honest reporting + the
    feature-vs-node crossover). feat_bytes (per-node F_v) is forwarded to arrange so the node-parallel
    side can be attribute-aware too; the feature-parallel side uses the scalar F width (it splits the
    F COLUMNS, which is the uniform width, not the per-node row bytes)."""
    if hasattr(graph, "sort_by_time"):
        graph = graph.sort_by_time()
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)
    N = int(graph.num_nodes)
    link = float(link_gbps) if link_gbps is not None else float(cluster.inter_node_bw)
    F = int(feat_dim)
    Fv = None if feat_bytes is None else np.asarray(feat_bytes, dtype=np.float64)

    # --- NODE-parallel (the existing arrange path) ---
    res = arrange(src, dst, N, cluster, link_gbps=link, feat_dim=F,
                  num_snapshots=num_snapshots, seed=seed, feat_bytes=Fv)
    _, node_makespan, _, node_bound, node_feas = _placement_from_arrange(res, cluster, F)

    # --- FEATURE-parallel (column-shard) ---
    fp = feature_parallel_plan(src, dst, N, cluster, F, link, boundary_frac=boundary_frac)

    # --- HYBRID 2D grids (non-degenerate Dn>1 AND Df>1; the pure corners are above) ---
    hybs = hybrid_plans(src, dst, N, cluster, F, link, boundary_frac=boundary_frac)

    cand = {"node": (node_makespan, node_feas),
            "feature": (fp.makespan_ms, fp.feasible)}
    for h in hybs:
        cand[h.name] = (h.makespan_ms, h.feasible)

    # pick lowest makespan among FEASIBLE; if none feasible, lowest makespan overall (least-bad)
    feasible_axes = {k: v for k, v in cand.items() if v[1]}
    pool = feasible_axes if feasible_axes else cand
    best = min(pool, key=lambda k: pool[k][0])
    GBb = 1024 ** 3

    # crossover note: contrast node vs feature on memory + makespan
    node_feat_gb = (res.feat_bytes_dev if res.feat_bytes_dev is not None
                    else res.counts.astype(np.float64) * F * 4.0) / GBb
    fp_feat_gb = fp.feat_bytes / GBb
    if best == "feature":
        note = (f"feature-parallel WINS: F={F} is attribute-heavy -> per-device feature memory "
                f"max {fp_feat_gb.max():.2f}GB (F/D cols) << node-parallel max {node_feat_gb.max():.2f}GB "
                f"(full-F rows); integration {fp.integration_ms:.2f}ms beats node boundary comm.")
    elif best == "node":
        note = (f"node-parallel WINS: F={F} small enough that compute divides D-ways and feature "
                f"memory ({node_feat_gb.max():.2f}GB max) fits; feature-parallel replicates the full "
                f"graph + does the SAME total compute, so its makespan {fp.makespan_ms:.1f}ms is higher.")
    else:
        note = (f"hybrid {best} WINS: neither pure axis is best -- splitting BOTH vertices and "
                f"feature columns balances compute vs feature memory vs comm at F={F}.")

    return DecompositionChoice(
        axis=best, makespan_ms=pool[best][0], feasible=bool(pool[best][1]),
        node_makespan_ms=node_makespan, node_feasible=node_feas,
        feature_makespan_ms=fp.makespan_ms, feature_feasible=fp.feasible,
        feature_integration_ms=fp.integration_ms,
        feature_cols_per_device=fp.cols_per_device,
        feature_feat_gb_per_device=fp_feat_gb,
        candidate_makespans=cand, crossover_note=note)


def plan(graph, cluster: ClusterProfile, *,
         link_gbps: Optional[float] = None, feat_dim: int = 128,
         num_snapshots: int = 64, prior: "Optional[Plan]" = None,
         new_edge_lo: Optional[int] = None, migration_budget: float = 0.05,
         mem_dim: int = 100, window: int = 1, reuse_frac: float = 0.0,
         seed: int = 0, feat_bytes: Optional[np.ndarray] = None,
         decomposition: str = "node", boundary_frac: float = 1.0) -> Plan:
    """Produce the end-to-end ZORD plan for one temporal batch.

    graph      : a zord.datasets.TemporalGraph (or any object exposing .src/.dst/.num_nodes,
                 and optionally .t for the temporal corner / .sort_by_time()).
    cluster    : a ClusterProfile (use profiler.from_spec(...) to pass an arbitrary spec:
                 per-device HBM capacity + achieved agg-bandwidth + INTERCONNECT BW parameter).
    link_gbps  : interconnect bandwidth (GB/s). Defaults to the cluster's link. The cost model
                 takes this as a PARAMETER -- zord wins on the algorithm at ANY comm speed.
    prior      : the previous batch's Plan (its .assignment seeds the incremental adaptation).
    new_edge_lo: first NEW edge offset this batch (for the changed cone); default 0 (all new).
    migration_budget : max fraction of OLD vertices that may move (the temporal lever).
    mem_dim    : TGN node-memory width m (bytes migrated per moving vertex = m*4).
    window/reuse_frac : passed to plan_memory for the per-device HBM<->CPU tiering layer.
    feat_bytes : OPTIONAL per-node feature-size VECTOR F_v (numpy [N], feature DIMS per node).
                 When None -> the SCALAR feat_dim path runs UNCHANGED (bit-identical -- the whole
                 attribute machinery is dead code). When given -> arrange does ATTRIBUTE-AWARE
                 (feature-byte) placement + feasibility (heavy-F nodes -> high-HBM/high-bandwidth
                 devices) and the tiering layer spills the LARGEST-F rows first (the §33 win).
    decomposition : which DECOMPOSITION AXIS to use. DEFAULT "node" -> the existing node-parallel
                 arrange path, UNCHANGED + byte-identical (the feature-parallel machinery is dead
                 code). "auto" -> additionally cost FEATURE-parallel (split the F feature COLUMNS
                 across devices: each device holds the full graph + F/D cols, integrates via column-
                 concat) and the HYBRID 2D grids, then pick the lowest FEASIBLE makespan. The chosen
                 axis is recorded in Plan.decomposition; the assignment/placement returned is still
                 the node-parallel arrange (the runtime executor selects the axis from .decomposition).
                 "feature"/"hybrid" -> force that axis in the choice (still reports all alternatives).
    boundary_frac : fraction of rows integrated across the link in the feature-parallel/hybrid cost
                 (1.0 = all output rows column-concatenated for the dense layer; lower when the dense
                 layer runs where columns already live). Only used when decomposition != "node".

    Returns a Plan: {partition assignment, per-device placement, vertex-cut/replication
    decisions, incremental-migration plan vs the prior batch, predicted makespan + feasibility,
    and -- when decomposition != "node" -- the node-vs-feature-vs-hybrid axis CHOICE}.
    """
    if hasattr(graph, "sort_by_time"):
        graph = graph.sort_by_time()
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)
    N = int(graph.num_nodes)
    link = float(link_gbps) if link_gbps is not None else float(cluster.inter_node_bw)
    Fv = None if feat_bytes is None else np.asarray(feat_bytes, dtype=np.float64)

    # per-edge snapshot id for the temporal (PTS) corner: equal-count over the time-sorted stream
    E = int(src.size)
    snap = np.minimum((np.arange(E) * num_snapshots // max(1, E)).astype(np.int64),
                      num_snapshots - 1)

    # --- 1. ARRANGE: adaptive-corner partition + placement + vertex-cut decision ---
    res = arrange(src, dst, N, cluster, link_gbps=link, feat_dim=feat_dim,
                  num_snapshots=num_snapshots, snap=snap, seed=seed, feat_bytes=Fv)
    placements, makespan_ms, bott, bound, feas = _placement_from_arrange(res, cluster, feat_dim)

    # --- 2. INCREMENTAL-MIGRATION plan vs the prior batch (the dynamic win) ---
    incr = None
    prior_assign = prior.assignment if prior is not None else None
    lo = new_edge_lo if new_edge_lo is not None else (0 if prior is None else 0)
    if prior is not None:
        incr = plan_incremental(src, dst, N, cluster.num_devices, prior_assign,
                                new_edge_lo=lo, migration_budget=migration_budget,
                                mem_dim=mem_dim, link_gbps=link)

    # --- 3. per-device HBM<->CPU tiering layer (no-OOM under window pressure) ---
    # F_v-aware: pass the per-node feature dims AND the arrange assignment so the tiering layer
    # sizes each device by its TRUE feature bytes and spills the largest-F rows first (§33). When
    # feat_bytes is None these stay None -> the uniform snapshot-tiering path is UNCHANGED.
    w = Workload(num_nodes=N, num_edges=E, feat_dim=feat_dim, window=window,
                 reuse_frac=reuse_frac, feat_bytes=Fv,
                 assignment=(res.assignment if Fv is not None else None))
    mem = plan_memory(cluster, w)

    # --- 4. DECOMPOSITION-AXIS choice (ADDITIVE; OFF by default). When decomposition != "node",
    # cost FEATURE-parallel (column-shard) + the HYBRID grids and record the winning axis. The
    # node-parallel arrange above is UNTOUCHED -> the default plan is byte-identical to before. ---
    decomp = None
    if decomposition != "node":
        decomp = choose_decomposition(
            graph, cluster, feat_dim=feat_dim, link_gbps=link, seed=seed,
            num_snapshots=num_snapshots, boundary_frac=boundary_frac, feat_bytes=Fv)
        if decomposition in ("feature", "hybrid"):
            # FORCE the requested axis as the winner (still reports every alternative). For "hybrid"
            # pick the best feasible hybrid grid; fall back to feature-parallel if no hybrid exists.
            forced = None
            if decomposition == "feature":
                forced = "feature"
            else:
                hys = {k: v for k, v in decomp.candidate_makespans.items() if k.startswith("hybrid")}
                feas_h = {k: v for k, v in hys.items() if v[1]} or hys
                forced = min(feas_h, key=lambda k: feas_h[k][0]) if feas_h else "feature"
            mk, fe = decomp.candidate_makespans[forced]
            decomp = replace(decomp, axis=forced, makespan_ms=mk, feasible=bool(fe),
                             crossover_note=f"axis FORCED to {forced} via decomposition='{decomposition}'. "
                                            + decomp.crossover_note)

    return Plan(
        strategy=res.name, assignment=res.assignment, core_mask=res.core_mask,
        placement=placements, makespan_ms=makespan_ms, bottleneck=bott, bound=bound,
        feasible=feas, cut_edges=res.cut, replication_pct=res.replication_pct,
        candidate_makespans=res.candidate_makespans, link_gbps=link,
        incremental=incr, memory=mem, decomposition=decomp)


# ============================================================================== #
# D1 REGRESSION TEST: the §47 conservative runtime-overhead reservation must hold  #
# on the ALL-RESIDENT (non-tiered) F_v path too, so predicted_peak >= measured on  #
# EVERY path (the no-OOM C1 guarantee). §46-wiki measured 70.3GB while the planner  #
# predicted only 63.5GB (~10% UNDER) on the all-resident path. This rebuilds that   #
# exact config (wiki-talk topology size + a uniform modeled F_v whose all-resident  #
# working set is 63.5GB on the 78GB-usable H100) and asserts predicted >= measured. #
# ============================================================================== #
def test_all_resident_fv_peak_geq_measured():
    """§46-wiki: the F_v ALL-RESIDENT (no-spill) peak must be a CONSERVATIVE upper bound on the
    MEASURED 70.3GB (was 63.5GB, ~10% UNDER). It must also stay feasible (<= cap), and NOT regress
    the real-spilling F_v path (78905: predicted 78.0 >= measured 71.5) nor the scalar path."""
    from ..profiler.cluster_profile import from_spec
    N, E, L, W = 1_140_149, 7_833_140, 2, 1               # wiki-talk topology size, single snapshot
    MEASURED_GB = 70.3                                    # §46-wiki real-GPU max_memory_allocated
    cluster = from_spec(hbm_gb=[80.0], agg_bw_gbps=[942.0], interconnect_gbps=325.0,
                        h2d_gbps=57.5, names=["H100-80GB"])
    cap = cluster.devices[0].usable_mem
    # uniform modeled F_v sized so the all-resident working set == 63.5GB (feat+L*feat+adjacency):
    #   W*sum(F_v)*(1+L)*4 + E*bytes_per_edge = 63.5GB
    sum_fv = (63.5 * GB - E * 12) / (W * (1 + L) * 4.0)
    Fv = np.full(N, sum_fv / N, dtype=np.float64)
    w = Workload(num_nodes=N, num_edges=E, feat_dim=int(round(sum_fv / N)), window=W, layers=L,
                 reuse_frac=0.0, feat_bytes=Fv, assignment=np.zeros(N, dtype=np.int64))
    mem = plan_memory(cluster, w)
    p = mem.per_device[0]
    assert mem.total_streamed_gb < 0.01                          # genuinely all-resident (no real spill)
    assert p.feasible and p.peak_hbm_bytes <= cap                # provably fits the GPU
    # THE D1 FIX: predicted peak is now a CONSERVATIVE upper bound on the measured 70.3GB. This FAILS
    # under the pre-fix code, which reported the bare 63.5GB resident bank (~10% UNDER -> UNSAFE).
    assert p.peak_hbm_bytes / GB >= MEASURED_GB, (
        f"all-resident F_v peak {p.peak_hbm_bytes/GB:.2f}GB UNDER-predicts measured {MEASURED_GB}GB")


def test_all_resident_fix_does_not_regress_fv_spill_or_scalar():
    """The D1 fix is gated on the all-resident sub-case (need_free <= 0). The REAL-spilling F_v path
    and the SCALAR path must be byte-for-byte unchanged."""
    from ..profiler.cluster_profile import from_spec
    cluster = from_spec(hbm_gb=[80.0], agg_bw_gbps=[942.0], interconnect_gbps=325.0,
                        h2d_gbps=57.5, names=["H100-80GB"])
    cap = cluster.devices[0].usable_mem
    # (a) F_v real spill (78905-like): heavy multi-modal hubs force a genuine spill; peak pinned ~cap,
    #     predicted >= the measured 71.5GB, and the spill DOES stream real bytes.
    Ns, Es = 2_601_977, 63_497_050
    Fv = np.full(Ns, 64.0, dtype=np.float64)
    rng = np.random.default_rng(0)
    Fv[rng.choice(Ns, int(0.10 * Ns), replace=False)] = 49152.0
    ws = Workload(num_nodes=Ns, num_edges=Es, feat_dim=int(round(Fv.mean())), window=1, layers=2,
                  reuse_frac=0.0, feat_bytes=Fv, assignment=np.zeros(Ns, dtype=np.int64))
    sp = plan_memory(cluster, ws).per_device[0]
    assert sp.streamed_snapshots > 0 and sp.staging_sec > 0.0    # a REAL spill, not the all-resident case
    assert sp.peak_hbm_bytes <= cap + 1 and sp.peak_hbm_bytes / GB >= 71.5
    # (b) SCALAR path (feat_bytes None): the all-resident fix never touches it (aware_tier is False).
    wsc = Workload(num_nodes=2_000_000, num_edges=16_000_000, feat_dim=128, window=2)
    sc = plan_memory(cluster, wsc).per_device[0]
    assert sc.feasible and sc.streamed_snapshots == 0           # fits all-resident, scalar path unchanged


if __name__ == "__main__":
    test_all_resident_fv_peak_geq_measured()
    test_all_resident_fix_does_not_regress_fv_spill_or_scalar()
    print("planner D1 self-tests passed: all-resident F_v peak is conservative (>= measured 70.3GB), "
          "F_v-spill and scalar paths unchanged.")
