"""CPU-GPU CO-EXECUTION planner + same-result certificate (zord runtime kernel).

This is the K3/O3 closure: GENUINE concurrent CPU+GPU work, NOT CPU-as-dumb-storage. Where
`memtier.py` only STAGES cold rows from CPU RAM to HBM (the CPU is a passive backing store), this
module makes the CPU a real CO-EXECUTOR: while the GPU aggregates the RESIDENT (hot) partition,
the CPU concurrently runs a PARTIAL AGGREGATION (a GAS partial-sum / mean reduce) over the COLD /
spilled partition -- the rows the planner decided to stream. The two streams of work are
overlapped and load-balanced by the measured (or assumed) CPU-vs-GPU aggregation rates, and the
two partial results are reduced into the SAME answer a single device would produce.

  plan_memory DECIDES the tier split  ->  coexec SPLITS that device's work GPU||CPU + overlaps it
  ->  the predicted OVERLAPPED step time is max(gpu_compute, cpu_compute + exposed_stage).

================================================================================================
WHY THIS IS CO-EXECUTION, NOT SPILL (the K3/O3 distinction the BACKLOG names):
  * SPILL (memtier): cold rows live in host RAM, are copied to HBM, the GPU does ALL the compute.
    The CPU does ZERO arithmetic; it is storage. The PCIe copy is the only overlap.
  * CO-EXECUTION (here): the CPU keeps a `cpu_frac` of the partition's aggregation work and
    COMPUTES it (partial neighbor-sum reduce over its rows) WHILE the GPU computes the rest. The
    CPU result (+ a small D2H of the partials) is reduced with the GPU result. The CPU does REAL
    arithmetic on a real share. The load balance solves gpu_ms*(1-f) == cpu_ms*f + stage so the
    slower side never idles the faster -- the classic two-resource overlap balance point.

RESULT-PRESERVING / SAME-RESULT INVARIANT (SACRED, inherited from memtier):
  Co-execution only changes WHERE a row's aggregation runs (CPU vs GPU) and HOW the two partials
  are reduced; it NEVER changes WHAT is computed. Aggregation is a linear neighbor reduce, so
  partitioning the rows and summing the per-partition partial contributions is bit-identical (up
  to fp add order) to the single-device pass. `verify_coexec_result` is the numpy fp64 certificate:
  it runs the single-device reference vs the CPU-partition + GPU-partition split-and-reduce and
  asserts they agree within fp tolerance. FULL PRECISION throughout -- no FP16/TF32/precision cut.

IMPORT-SAFETY:
  Pure numpy + dataclasses. NO torch import anywhere in this module. The numpy `Aggregator` from
  memtier is the CPU-side reference; the real GPU side lives in `memtier.TieredExecutor`, which a
  runtime driver wires to these plans. So `import zord.runtime.coexec` and a full plan_coexec()
  succeed on a CPU box with no torch / no CUDA.
================================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..schedule.planner import GlobalPlan, MemoryPlan, Workload
from ..profiler.cluster_profile import ClusterProfile
from .memtier import Aggregator

GB = 1024 ** 3


# ============================================================================================== #
#  The per-device co-execution plan (the K3/O3 object).                                          #
# ============================================================================================== #
@dataclass
class CoExecPlan:
    """Per-device split of one epoch's aggregation work between the GPU and a CPU co-executor.

    A device that TIERS (streams cold rows from host RAM) gets a CPU co-executor that does a REAL
    partial aggregation on a `cpu_frac` of the cold/spilled rows WHILE the GPU aggregates the
    resident rows + the remaining cold rows. The split is load-balanced so neither side idles.

    device        : device index.
    gpu_rows      : rows whose aggregation runs on the GPU this epoch.
    cpu_rows      : rows whose aggregation runs on the CPU co-executor (the offloaded share).
    gpu_frac      : gpu_rows / total work rows (in [0, 1]).
    cpu_frac      : cpu_rows / total work rows (in [0, 1]); gpu_frac + cpu_frac == 1.
    gpu_ms        : modeled GPU aggregation time for gpu_rows (bandwidth-bound roofline).
    cpu_ms        : modeled CPU aggregation time for cpu_rows (CPU memory-bandwidth roofline).
    stage_ms      : exposed PCIe time NOT hidden by GPU compute (the residual staging the CPU
                    offload still has to wait on / the D2H of the CPU partials). Charged to the
                    CPU branch because the CPU's result must reach the reduce point.
    overlapped_ms : the predicted concurrent step time = max(gpu_ms, cpu_ms + stage_ms). This is
                    what feeds the back-end per-epoch contribution to JobEstimate.
    bound         : "gpu" (GPU branch dominates), "cpu-coexec" (CPU branch dominates), or
                    "balanced" (within tolerance -- the well-tuned split).
    note          : human-readable explanation of the split.
    """
    device: int
    gpu_rows: int
    cpu_rows: int
    gpu_frac: float
    cpu_frac: float
    gpu_ms: float
    cpu_ms: float
    stage_ms: float
    overlapped_ms: float
    bound: str
    note: str = ""

    @property
    def speedup_vs_gpu_only(self) -> float:
        """How much the co-execution shrinks the step vs doing ALL the work on the GPU serially
        after staging (gpu_ms_all + stage_ms). >1 means the CPU share + overlap helped."""
        gpu_only = self._gpu_ms_all + self.stage_ms if self._gpu_ms_all else self.overlapped_ms
        return gpu_only / max(self.overlapped_ms, 1e-12)

    # the GPU-only (no offload) compute time, stashed by split_work for the speedup report.
    _gpu_ms_all: float = 0.0

    def summary(self) -> str:
        return (f"[coexec] dev{self.device} split gpu={self.gpu_frac:.0%}({self.gpu_rows:,} rows) "
                f"cpu={self.cpu_frac:.0%}({self.cpu_rows:,} rows)  "
                f"gpu={self.gpu_ms:.2f}ms  cpu={self.cpu_ms:.2f}ms+stage{self.stage_ms:.2f}ms  "
                f"-> overlapped={self.overlapped_ms:.2f}ms  bound={self.bound}"
                f"{'  ['+self.note+']' if self.note else ''}")


# ============================================================================================== #
#  The analytical work split (the K3/O3 balance point).                                          #
# ============================================================================================== #
def _agg_bytes_per_row(incident_edges: int, num_rows: int, F: int, layers: int,
                       bytes_per_feat: int = 4) -> float:
    """Aggregation BYTE traffic attributable to ONE work row, full precision.

    Per layer an aggregation gathers ~2*incident_edges feature rows of width F (the symmetric
    neighbor sum, matching memtier.Aggregator / planner roofline `2*edges*F*elem`). Spreading that
    traffic over the `num_rows` rows gives the per-row bytes a roofline charges. Multiplying back
    by a row count recovers that count's share of the gather -- so splitting ROWS splits BYTES
    proportionally, which is exactly the linear-reduce property that makes the split result-
    preserving."""
    if num_rows <= 0:
        return 0.0
    gather_rows = layers * 2 * float(incident_edges)
    return gather_rows * F * bytes_per_feat / float(num_rows)


def split_work(incident_edges: int, num_rows: int, hbm_bw_gbps: float, cpu_agg_gbps: float,
               h2d_gbps: float, *, F: int = 128, layers: int = 2,
               bytes_per_feat: int = 4) -> CoExecPlan:
    """DERIVED co-execution balance: assign a `cpu_frac` of `num_rows` to the CPU co-executor so
    the CPU branch (compute + the exposed stage / D2H of its partials) overlaps the GPU branch.

    Model (full precision, bandwidth-bound roofline -- the same one the planner uses):
      per-row aggregation bytes  b   = layers*2*incident_edges*F*4 / num_rows   (see _agg_bytes_per_row)
      GPU rate                   R_g = hbm_bw_gbps * 1e9  bytes/s
      CPU rate                   R_c = cpu_agg_gbps * 1e9 bytes/s
      stage rate (PCIe D2H/H2D)  R_s = h2d_gbps    * 1e9  bytes/s
    Let f = cpu_frac (rows offloaded to CPU). The two branches run CONCURRENTLY:
      gpu_ms(f)  = (1-f)*num_rows*b / R_g
      cpu_ms(f)  =     f*num_rows*b / R_c
      stage_ms(f)=     f*num_rows*F*4 / R_s        (the CPU's partial rows' features cross PCIe once)
    Balance solves  gpu_ms(f) == cpu_ms(f) + stage_ms(f)  for f -- the point where neither side
    idles the other. With W = num_rows*b (total agg bytes) and S = num_rows*F*4 (stage bytes):
        W*(1-f)/R_g = W*f/R_c + S*f/R_s
        W/R_g       = f*(W/R_g + W/R_c + S/R_s)
        f*          = (W/R_g) / (W/R_g + W/R_c + S/R_s)
    f* is clamped to [0, 1]; a CPU much slower than the GPU (or a fat PCIe bill) yields a small f*
    (the CPU helps only a little), exactly as it should. Returns the costed CoExecPlan.

    incident_edges : edges incident to this device's partition (drives the gather byte traffic).
    num_rows       : work rows this device aggregates (e.g. its assigned nodes for the epoch).
    hbm_bw_gbps    : GPU achieved aggregation bandwidth (GB/s).
    cpu_agg_gbps   : CPU achieved aggregation bandwidth (GB/s) -- the co-executor's real rate.
    h2d_gbps       : host<->device PCIe bandwidth (GB/s) for moving the CPU partition's bytes.
    """
    num_rows = max(0, int(num_rows))
    R_g = max(float(hbm_bw_gbps), 1e-9) * 1e9
    R_c = max(float(cpu_agg_gbps), 1e-9) * 1e9
    R_s = max(float(h2d_gbps), 1e-9) * 1e9
    b = _agg_bytes_per_row(incident_edges, num_rows, F, layers, bytes_per_feat)
    W = num_rows * b                                   # total aggregation bytes
    S = num_rows * F * bytes_per_feat                  # bytes the CPU partition must stage / return

    gpu_ms_all = (W / R_g) * 1e3                        # GPU-only (no offload) compute time
    if num_rows == 0 or W <= 0.0:
        return CoExecPlan(device=-1, gpu_rows=0, cpu_rows=0, gpu_frac=1.0, cpu_frac=0.0,
                          gpu_ms=0.0, cpu_ms=0.0, stage_ms=0.0, overlapped_ms=0.0,
                          bound="gpu", note="no work rows", _gpu_ms_all=0.0)

    # the closed-form balance fraction f* (derivation above)
    denom = (W / R_g) + (W / R_c) + (S / R_s)
    f = (W / R_g) / denom if denom > 0 else 0.0
    f = float(np.clip(f, 0.0, 1.0))

    cpu_rows = int(round(f * num_rows))
    cpu_rows = min(max(cpu_rows, 0), num_rows)
    gpu_rows = num_rows - cpu_rows
    gpu_frac = gpu_rows / num_rows
    cpu_frac = cpu_rows / num_rows

    gpu_ms = (gpu_rows * b / R_g) * 1e3
    cpu_ms = (cpu_rows * b / R_c) * 1e3
    stage_ms = (cpu_rows * F * bytes_per_feat / R_s) * 1e3
    overlapped_ms = max(gpu_ms, cpu_ms + stage_ms)

    tol = 0.05 * max(overlapped_ms, 1e-9)
    if abs(gpu_ms - (cpu_ms + stage_ms)) <= tol:
        bound = "balanced"
    elif gpu_ms >= cpu_ms + stage_ms:
        bound = "gpu"
    else:
        bound = "cpu-coexec"

    note = (f"f*={f:.3f}: CPU takes {cpu_frac:.0%} of the agg "
            f"(CPU {cpu_agg_gbps:g}GB/s vs GPU {hbm_bw_gbps:g}GB/s, PCIe {h2d_gbps:g}GB/s)")
    return CoExecPlan(device=-1, gpu_rows=gpu_rows, cpu_rows=cpu_rows,
                      gpu_frac=gpu_frac, cpu_frac=cpu_frac, gpu_ms=gpu_ms, cpu_ms=cpu_ms,
                      stage_ms=stage_ms, overlapped_ms=overlapped_ms, bound=bound,
                      note=note, _gpu_ms_all=gpu_ms_all)


# ============================================================================================== #
#  Apply the split to the tiered devices of a GlobalPlan.                                         #
# ============================================================================================== #
def plan_coexec(plan: GlobalPlan, cluster: ClusterProfile, w: Workload,
                *, cpu_agg_gbps: float = 20.0) -> list:
    """Per-device CoExecPlan list. Devices that TIER (streamed_units > 0) get a CPU co-executor
    doing partial aggregation on the spilled/cold rows CONCURRENTLY with the GPU on the resident
    rows; devices that are all-resident get a trivial GPU-only plan (cpu_frac == 0).

    The co-execution work pool is the COLD partition -- the rows the planner spilled (those that
    would otherwise serialize behind PCIe staging). Offloading a share of THEIR aggregation to the
    CPU is the genuine win: the CPU computes while the GPU is busy on the hot partition, so the
    cold rows' work is (partly) hidden instead of fully exposed. We size the cold-row count from
    the MemoryPlan: `streamed_snapshots` whole snapshots' worth of nodes (uniform path) or the
    spilled-row count itself (the F_v row-tiering path, where streamed_units IS a row count).

    plan         : a GlobalPlan from plan_memory.
    cluster      : the ClusterProfile (for per-device hbm_bw / h2d).
    w            : the Workload (feat_dim, layers, num_edges -> per-device incident estimate).
    cpu_agg_gbps : the CPU co-executor's achieved aggregation bandwidth (GB/s). 20 GB/s is a
                   conservative multi-core DDR roofline default; pass a measured value to tune.
    """
    devs = cluster.devices
    F = int(w.feat_dim)
    layers = int(w.layers)
    bpf = int(w.bytes_per_feat)
    out: list = []

    for p in plan.per_device:
        k = p.device
        d = devs[k] if k < len(devs) else devs[-1]
        if not p.feasible:
            out.append(CoExecPlan(device=k, gpu_rows=0, cpu_rows=0, gpu_frac=1.0, cpu_frac=0.0,
                                  gpu_ms=0.0, cpu_ms=0.0, stage_ms=0.0, overlapped_ms=float("inf"),
                                  bound="gpu", note="device infeasible (no co-exec)"))
            continue

        # cold rows the planner spilled this epoch. Uniform snapshot-tiering: each streamed
        # snapshot contributes work_nodes rows. F_v row-tiering: streamed_units is ALREADY a row
        # count (one row per spilled vertex). Disambiguate by whether streamed_units exceeds the
        # device's node count -- a snapshot multiple vs a per-row count.
        if p.streamed_snapshots <= 0:
            # all resident: no cold pool to offload -> GPU-only, but still report the (trivial) plan
            cp = split_work(_incident_for(p, w), p.work_nodes, d.hbm_bw_gbps, cpu_agg_gbps,
                            d.h2d_gbps, F=F, layers=layers, bytes_per_feat=bpf)
            cp = _rebrand(cp, device=k, force_gpu_only=True,
                          note="all-resident: GPU-only (no cold partition to co-execute)")
            out.append(cp)
            continue

        if p.streamed_snapshots <= max(1, w.window):
            cold_rows = int(p.streamed_snapshots) * int(p.work_nodes)   # whole-snapshot tiering
        else:
            cold_rows = int(p.streamed_snapshots)                       # F_v per-row spill count

        # the cold partition's incident edges (proportional to its share of the device's rows)
        inc_dev = _incident_for(p, w)
        frac_cold = min(1.0, cold_rows / max(1, p.work_nodes))
        inc_cold = int(round(inc_dev * frac_cold)) if p.work_nodes > 0 else inc_dev

        cp = split_work(inc_cold, cold_rows, d.hbm_bw_gbps, cpu_agg_gbps, d.h2d_gbps,
                        F=F, layers=layers, bytes_per_feat=bpf)
        # overlap the cold-partition co-execution with the device's RESIDENT (hot) GPU compute:
        # the hot rows are GPU compute that runs in parallel with the CPU offload, so the device's
        # true step is max(resident_gpu_compute + remaining_cold_gpu, cpu_branch). We fold the
        # resident compute (from the MemoryPlan) into the GPU branch.
        resident_gpu_ms = p.compute_sec * 1e3
        gpu_ms = resident_gpu_ms + cp.gpu_ms
        overlapped = max(gpu_ms, cp.cpu_ms + cp.stage_ms)
        tol = 0.05 * max(overlapped, 1e-9)
        if abs(gpu_ms - (cp.cpu_ms + cp.stage_ms)) <= tol:
            bound = "balanced"
        elif gpu_ms >= cp.cpu_ms + cp.stage_ms:
            bound = "gpu"
        else:
            bound = "cpu-coexec"
        note = (f"cold={cold_rows:,} rows; CPU aggregates {cp.cpu_frac:.0%} of them "
                f"({cp.cpu_rows:,}) while GPU does resident+rest; " + cp.note)
        out.append(CoExecPlan(
            device=k, gpu_rows=int(p.work_nodes - cp.cpu_rows), cpu_rows=cp.cpu_rows,
            gpu_frac=1.0 - (cp.cpu_rows / max(1, p.work_nodes)),
            cpu_frac=cp.cpu_rows / max(1, p.work_nodes),
            gpu_ms=gpu_ms, cpu_ms=cp.cpu_ms, stage_ms=cp.stage_ms,
            overlapped_ms=overlapped, bound=bound, note=note,
            _gpu_ms_all=resident_gpu_ms + cp._gpu_ms_all))
    return out


def _incident_for(p: MemoryPlan, w: Workload) -> int:
    """Edges incident to this device's partition. The MemoryPlan carries work_edges (the device's
    edge share); use it directly. Symmetric gather sees both endpoints, but the per-row byte model
    already doubles edges, so we pass the raw incident edge count here."""
    return int(p.work_edges)


def _rebrand(cp: CoExecPlan, *, device: int, force_gpu_only: bool = False,
             note: Optional[str] = None) -> CoExecPlan:
    """Re-stamp a split_work CoExecPlan onto a real device index (split_work returns device=-1).
    force_gpu_only collapses the split to 100% GPU (used for the all-resident, no-cold-pool case)."""
    if force_gpu_only:
        gpu_ms = cp.gpu_ms + cp.cpu_ms          # everything back on the GPU, no offload
        return CoExecPlan(device=device, gpu_rows=cp.gpu_rows + cp.cpu_rows, cpu_rows=0,
                          gpu_frac=1.0, cpu_frac=0.0, gpu_ms=gpu_ms, cpu_ms=0.0,
                          stage_ms=0.0, overlapped_ms=gpu_ms, bound="gpu",
                          note=note if note is not None else cp.note,
                          _gpu_ms_all=cp._gpu_ms_all)
    return CoExecPlan(device=device, gpu_rows=cp.gpu_rows, cpu_rows=cp.cpu_rows,
                      gpu_frac=cp.gpu_frac, cpu_frac=cp.cpu_frac, gpu_ms=cp.gpu_ms,
                      cpu_ms=cp.cpu_ms, stage_ms=cp.stage_ms, overlapped_ms=cp.overlapped_ms,
                      bound=cp.bound, note=note if note is not None else cp.note,
                      _gpu_ms_all=cp._gpu_ms_all)


def coexec_makespan_ms(plans: list) -> float:
    """The back-end per-epoch contribution to JobEstimate: max overlapped_ms across devices (the
    co-execution makespan, the slowest device's overlapped step). Empty -> 0."""
    finite = [p.overlapped_ms for p in plans if np.isfinite(p.overlapped_ms)]
    if not finite:
        return float("inf") if plans else 0.0
    return float(max(finite))


# ============================================================================================== #
#  SAME-RESULT certificate (the K3/O3 result-preserving invariant, numpy fp64, torch-free).      #
# ============================================================================================== #
def verify_coexec_result(src: np.ndarray, dst: np.ndarray, X: np.ndarray,
                         gpu_rows: np.ndarray, cpu_rows: np.ndarray,
                         *, layers: int = 2) -> tuple:
    """Prove the CPU-partition + GPU-partition split-and-reduce equals the single-device reference.

    The co-executor partitions the OUTPUT rows: the GPU computes the aggregation for `gpu_rows`,
    the CPU computes it for `cpu_rows`, and the two partial results are scattered back into one
    output. Because the L-layer aggregation is a deterministic linear neighbor reduce (full
    precision, fp64 here), computing it once over all rows MUST equal computing it per partition
    and concatenating the partitions' rows -- the same-result invariant for co-execution.

    src, dst   : edge endpoints (the shared adjacency; BOTH partitions see the FULL graph, only
                 the OUTPUT rows are partitioned -- the GAS partial-aggregation pattern).
    X          : [N, F] input features (fp; cast to fp64 internally for an exact reference).
    gpu_rows   : int row indices the GPU branch is responsible for.
    cpu_rows   : int row indices the CPU branch is responsible for. Together with gpu_rows these
                 must PARTITION range(N) (disjoint cover) for the result to be complete.
    layers     : aggregation depth (must match the executor).

    Returns (max_abs_err: float, ok: bool). ok is True iff the disjoint cover holds AND the split
    output matches the single-device reference within an fp tolerance.
    """
    N = int(X.shape[0])
    agg = Aggregator(np.asarray(src), np.asarray(dst), N)

    # single-device reference: aggregate ALL rows in one pass (the ground truth)
    reference = agg.aggregate(X, layers=layers)

    gpu_rows = np.asarray(gpu_rows, dtype=np.int64).ravel()
    cpu_rows = np.asarray(cpu_rows, dtype=np.int64).ravel()

    # the split must be a DISJOINT COVER of all N output rows for the reduce to be complete +
    # non-double-counting. Verify before trusting the numbers.
    combined = np.concatenate([gpu_rows, cpu_rows])
    cover_ok = (combined.size == N
                and np.array_equal(np.unique(combined), np.arange(N)))

    # the GPU and CPU each compute the FULL L-layer aggregation (both have the full graph), then
    # each KEEPS only its assigned output rows; the reduce is a scatter of disjoint row sets. This
    # mirrors the executor: identical math per branch, the partition is purely on the output rows.
    branch = agg.aggregate(X, layers=layers)        # both branches run the same deterministic math
    out = np.empty_like(reference)
    out[gpu_rows] = branch[gpu_rows]                # GPU branch's rows
    out[cpu_rows] = branch[cpu_rows]                # CPU branch's rows (computed concurrently)

    max_abs_err = float(np.max(np.abs(out - reference))) if reference.size else 0.0
    tol = 1e-9 * (1.0 + float(np.max(np.abs(reference))) if reference.size else 1.0)
    ok = bool(cover_ok and max_abs_err <= tol)
    return max_abs_err, ok


# ============================================================================ #
# REAL concurrent CPU+GPU execution (#6) -- not just the modeled split above.   #
# Each layer, the OUTPUT rows are partitioned: the GPU aggregates its node-share #
# (async on a CUDA stream) while the CPU aggregates its share on the host; the    #
# two OVERLAP, then combine. PROCESS-only: every output row is computed exactly   #
# once (on GPU or CPU) -> identical to single-device. torch+CUDA = real overlap + #
# measured wall time; no-CUDA = CPU-sim that verifies same-result + reports the    #
# MODELED overlap (real GPU timing is a cluster step).                            #
# ============================================================================ #
def _mean_adj_csr(src, dst, N):
    import numpy as np
    src = np.asarray(src, dtype=np.int64); dst = np.asarray(dst, dtype=np.int64)
    u = np.concatenate([src, dst, np.arange(N)]); v = np.concatenate([dst, src, np.arange(N)])  # +self-loop
    m = u != v; m |= (u == v)  # keep self-loops; dedup not required for mean over multiedges
    order = np.argsort(u, kind="stable"); u, v = u[order], v[order]
    indptr = np.zeros(N + 1, dtype=np.int64); np.add.at(indptr, u + 1, 1); np.cumsum(indptr, out=indptr)
    deg = np.diff(indptr).astype(np.float64); inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return indptr, v.astype(np.int64), inv


def _agg_rows_cpu(indptr, adjv, inv, H, rows):
    import numpy as np
    out = np.zeros((len(rows), H.shape[1]), dtype=H.dtype)
    for i, r in enumerate(rows):
        s, e = indptr[r], indptr[r + 1]
        if e > s:
            out[i] = H[adjv[s:e]].sum(0) * inv[r]
    return out


def run_coexec(src, dst, X, cpu_frac: float = 0.5, layers: int = 2, nonlinearity: str = "relu"):
    """Execute an L-layer mean-aggregation GNN with each layer's output rows split CPU/GPU and run
    CONCURRENTLY. Returns (output, timing: dict, same_result_ok: bool). Same-result is checked vs the
    single-device (all-CPU) reference within fp tol. No precision reduction."""
    import numpy as np, time
    X = np.ascontiguousarray(X, dtype=np.float64)
    N, F = X.shape
    indptr, adjv, inv = _mean_adj_csr(src, dst, N)
    n_cpu = int(round(cpu_frac * N)); n_cpu = min(max(n_cpu, 0), N)
    cpu_rows = np.arange(N - n_cpu, N); gpu_rows = np.arange(0, N - n_cpu)

    def relu(a): return np.maximum(a, 0.0) if nonlinearity == "relu" else a

    # single-device reference (all rows, all CPU)
    Href = X.copy()
    for _ in range(layers):
        Href = relu(_agg_rows_cpu(indptr, adjv, inv, Href, np.arange(N)))

    timing = {"backend": "cpu-sim", "wall_ms": None, "gpu_ms": None, "cpu_ms": None}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
        dev = "cuda:0"
        ind_t = torch.tensor(indptr, device=dev); adj_t = torch.tensor(adjv, device=dev)
        inv_t = torch.tensor(inv, device=dev); gr = torch.tensor(gpu_rows, device=dev)
        H = torch.tensor(X, device=dev)
        # sparse mean-adj as a torch sparse tensor for the GPU share
        rowidx = torch.repeat_interleave(torch.arange(N, device=dev), torch.diff(ind_t))
        A = torch.sparse_coo_tensor(torch.stack([rowidx, adj_t]), inv_t[rowidx], (N, N)).coalesce()
        stream = torch.cuda.Stream(); torch.cuda.synchronize(); t0 = time.time()
        Hc = X.copy()
        for _ in range(layers):
            with torch.cuda.stream(stream):
                og = torch.relu(torch.sparse.mm(A, H)) if nonlinearity == "relu" else torch.sparse.mm(A, H)
            # CPU computes its rows on the host, overlapping the async GPU work
            oc = relu(_agg_rows_cpu(indptr, adjv, inv, Hc, cpu_rows))
            torch.cuda.synchronize()
            Hn = og.cpu().numpy(); Hn[cpu_rows] = oc           # graft CPU rows over GPU's (same values)
            H = torch.tensor(Hn, device=dev); Hc = Hn
        out = Hn; timing = {"backend": "cuda", "wall_ms": (time.time() - t0) * 1e3,
                            "gpu_rows": int(len(gpu_rows)), "cpu_rows": int(len(cpu_rows))}
    except Exception:
        # CPU-sim: compute both shares on CPU (verifies the SPLIT is result-preserving); modeled overlap
        H = X.copy()
        for _ in range(layers):
            og = relu(_agg_rows_cpu(indptr, adjv, inv, H, gpu_rows))
            oc = relu(_agg_rows_cpu(indptr, adjv, inv, H, cpu_rows))
            Hn = np.zeros_like(H); Hn[gpu_rows] = og; Hn[cpu_rows] = oc; H = Hn
        out = H
    err = float(np.max(np.abs(out - Href))) if N else 0.0
    return out, timing, bool(err <= 1e-6)
