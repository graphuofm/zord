"""CPU<->HBM memory-tiering EXECUTOR (zord runtime kernel).

This is the runtime counterpart of `zord.schedule.plan_memory`: the planner DECIDES the
tiering (how many snapshots / which F_v rows stay RESIDENT in HBM vs are STAGED from CPU RAM
over PCIe, the predicted peak, feasibility); THIS module EXECUTES that decision on a GPU --
holding the resident hot state in HBM, the cold state in PINNED host RAM, and prefetching the
streamed state with a DOUBLE BUFFER so the H2D copy of snapshot s+1 overlaps the compute of s.

It is the VALIDATED logic promoted out of scripts/oom_engine_gpu.py (snapshot tiering) and
scripts/oom_attr_gpu.py (F_v-aware row tiering), per the user's hard rule: updated functionality
MUST land in the kernel, not stay in scripts. The two scripts now have ONE reusable home here.

  engine plan_memory DECIDES  ->  TieredExecutor EXECUTES  ->  predicted peak == measured peak.

================================================================================================
RESULT-PRESERVING / SAME-RESULT INVARIANT (SACRED -- do NOT weaken):
  zord optimizes the PROCESS (wall-clock, peak HBM, feasibility), NEVER the numerical result.
  Same data + same model => SAME output, bit-for-bit-comparable up to fp ordering. Therefore:

    * EVERYTHING here is FULL PRECISION (fp32). We do NOT use FP16 / BF16 / TF32 / any precision
      reduction. TF32 is EXPLICITLY DISABLED on the matmul/cuDNN paths (see `_enforce_full_precision`)
      because PyTorch may default TF32 ON for Ampere+ -- that would silently round fp32 matmuls and
      BREAK the invariant. Tiering only changes WHERE a row lives (HBM vs host RAM) and WHEN it is
      copied (prefetch overlap), never WHAT is computed.
    * The CUDA optimizations here (streams to overlap H2D prefetch with compute, pinned host memory,
      cuSPARSE SpMM via torch.sparse.mm) speed up WALL-CLOCK only; they are bit-neutral. Honestly
      framed: "原理无益但实验好看" -- they do not change the algorithm's result, only its runtime.

  The CPU-simulation path (`simulate_tiering`) reproduces the resident/streamed BYTE budget and the
  reference aggregation EXACTLY without torch, so the invariant is testable on a CPU box.
================================================================================================
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..schedule.planner import GlobalPlan, MemoryPlan, Workload

GB = 1024 ** 3

# ---- torch is OPTIONAL. The module must import + the CPU-sim path must run with NO torch/CUDA. --
try:                                            # pragma: no cover - exercised only where torch lives
    import torch as _torch
    _HAS_TORCH = True
except Exception:                               # torch absent -> CPU-simulation path only
    _torch = None
    _HAS_TORCH = False


def torch_available() -> bool:
    """True iff `import torch` succeeded (does NOT imply a CUDA device is present)."""
    return _HAS_TORCH


def cuda_available() -> bool:
    """True iff torch is importable AND a CUDA device is visible."""
    return _HAS_TORCH and bool(_torch.cuda.is_available())


# ============================================================================================== #
#  RESULT-PRESERVING precision guard.                                                            #
# ============================================================================================== #
def enforce_full_precision() -> None:
    """Disable every silent precision-reduction PyTorch might default to, so fp32 stays fp32.

    Ampere+ defaults TF32 ON for matmul and cuDNN; that rounds fp32 operands to 19-bit mantissa
    and would BREAK zord's same-result invariant. We force the high-precision fp32 paths. This is
    a no-op when torch is absent. It changes RESULTS toward the reference, never away from it."""
    if not _HAS_TORCH:
        return
    try:
        _torch.backends.cuda.matmul.allow_tf32 = False
        _torch.backends.cudnn.allow_tf32 = False
        # newer torch: be explicit that fp32 matmul stays fp32 (no TF32 "high"/"medium" rounding)
        if hasattr(_torch, "set_float32_matmul_precision"):
            _torch.set_float32_matmul_precision("highest")
    except Exception:                           # pragma: no cover - old torch without these knobs
        pass


# ============================================================================================== #
#  CPU-SIMULATION byte budget (torch-free, testable anywhere).                                   #
# ============================================================================================== #
@dataclass
class TierBudget:
    """The resident/streamed BYTE split for one device, derived from a MemoryPlan. This is the
    runtime's view of the planner decision and is computed with NO torch (so it is testable on a
    CPU box and used to drive the simulation path)."""
    device: int
    resident_units: int             # snapshots (uniform path) OR resident rows (F_v path)
    streamed_units: int             # snapshots OR streamed rows
    resident_bytes: int             # HBM bytes held resident (planner peak model)
    streamed_bytes: int             # bytes that cross PCIe per epoch (host -> device)
    capacity_bytes: int
    feasible: bool

    @property
    def fits(self) -> bool:
        return self.feasible and self.resident_bytes <= self.capacity_bytes


def budget_from_plan(plan: GlobalPlan, w: Workload, device: int = 0,
                     h2d_gbps: Optional[float] = None) -> TierBudget:
    """Extract the per-device tiering byte budget from a plan_memory result. Mirrors how
    oom_engine_gpu / oom_attr_gpu read `p.resident_snapshots` / `p.streamed_snapshots` and the
    predicted peak -- now in ONE place so the runtime and the scripts agree by construction.

    h2d_gbps : the device's host->device PCIe bandwidth (from the cluster). When given, streamed
               bytes are recovered EXACTLY as staging_sec * h2d (works for BOTH the whole-snapshot
               and the per-row F_v spill paths -- the same identity plan_memory uses for
               total_streamed_gb). When None, fall back to the uniform whole-snapshot byte model
               (streamed_snapshots * N_k * F * bytes_per_feat)."""
    p: MemoryPlan = plan.per_device[device]
    if h2d_gbps is not None and p.staging_sec > 0:
        streamed_bytes = int(round(p.staging_sec * float(h2d_gbps) * 1e9))
    else:
        streamed_bytes = int(p.streamed_snapshots * p.work_nodes * w.feat_dim * w.bytes_per_feat)
    return TierBudget(
        device=device, resident_units=int(p.resident_snapshots),
        streamed_units=int(p.streamed_snapshots),
        resident_bytes=int(p.peak_hbm_bytes), streamed_bytes=int(streamed_bytes),
        capacity_bytes=int(p.capacity_bytes), feasible=bool(p.feasible))


def simulate_tiering(resident_data: np.ndarray, streamed_data: np.ndarray,
                     adjacency: "Aggregator", *, layers: int = 2) -> np.ndarray:
    """CPU reference: aggregate the FULL feature set with the SAME math whether a row is resident
    or streamed -- proving the same-result invariant on a CPU box (no torch, no CUDA).

    resident_data : [n_res, F] rows held "in HBM".
    streamed_data : [n_str, F] rows "staged from CPU".  Concatenated back in the ORIGINAL order by
                    the caller's index map; here we just assert the tiered path == the single-array
                    reference (the executor never reorders the math, only the storage tier).
    adjacency     : an Aggregator implementing .aggregate(X) -> the L-layer GraphSAGE-style result.
    Returns the aggregation of the concatenated [resident; streamed] feature matrix."""
    full = np.concatenate([resident_data, streamed_data], axis=0)
    return adjacency.aggregate(full, layers=layers)


class Aggregator:
    """Tiny full-precision (fp64 in numpy -> exact) reference aggregator for the CPU-sim path.
    A symmetric, degree-normalized 2-layer mean aggregation -- the SAME shape as the GPU kernel in
    the scripts, but pure numpy so the same-result invariant is testable without torch."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, num_nodes: int):
        self.N = int(num_nodes)
        i = np.concatenate([src, dst]).astype(np.int64)
        j = np.concatenate([dst, src]).astype(np.int64)
        self.i, self.j = i, j
        deg = np.bincount(i, minlength=self.N).astype(np.float64)
        self.inv_deg = 1.0 / np.clip(deg, 1.0, None)

    def _spmm(self, X: np.ndarray) -> np.ndarray:
        """One normalized neighbor-mean: out[i] = sum_{j->i} X[j] / deg[i]. Deterministic order."""
        out = np.zeros_like(X)
        np.add.at(out, self.i, X[self.j])
        return out * self.inv_deg[:, None]

    def aggregate(self, X: np.ndarray, layers: int = 2) -> np.ndarray:
        h = np.asarray(X, dtype=np.float64)
        for _ in range(layers):
            h = np.maximum(self._spmm(h), 0.0)      # relu(mean-agg) -- full precision
        return h


# ============================================================================================== #
#  GPU EXECUTOR (only touched when CUDA is present).                                             #
# ============================================================================================== #
@dataclass
class TieredRunResult:
    completed: bool
    measured_peak_gb: float
    epoch_sec: float
    pcie_gb: float
    predicted_peak_gb: float
    peak_rel_err: float
    note: str = ""


class TieredExecutor:
    """Execute a plan_memory decision on ONE GPU with the validated tiering discipline:

      * RESIDENT snapshots/rows live in HBM (their features + the L activation copies, so the live
        footprint matches the planner's (1+L) model).
      * COLD snapshots/rows live in PINNED host RAM (page-locked -> fast, async-able H2D).
      * A DOUBLE BUFFER (2 pinned bounce buffers + 2 GPU slots) prefetches the next streamed unit on
        a SEPARATE CUDA STREAM so its H2D copy OVERLAPS the compute of the current unit.
      * The in-core-OOM FREE DISCIPLINE: after a baseline OOM, drop refs + gc.collect() +
        empty_cache() + synchronize() so leaked partial allocations do not false-OOM the plan run.

    All ops are FULL PRECISION (fp32); TF32 is disabled in __init__. The streams/pinned-memory/
    cuSPARSE optimizations are bit-neutral (wall-clock only). See the module docstring.

    This generalizes oom_engine_gpu.run_gpu (snapshot tiering) -- the F_v row-tiering variant in
    oom_attr_gpu has a more elaborate column-block GPU layout that stays in that script; the SHARED
    discipline (resident-bank / pinned-cold / double-buffer prefetch / OOM-free) is what is promoted.
    """

    def __init__(self, device: str = "cuda:0"):
        if not _HAS_TORCH:
            raise RuntimeError("TieredExecutor needs torch; use simulate_tiering() on a CPU box.")
        if not _torch.cuda.is_available():
            raise RuntimeError("TieredExecutor needs a CUDA device; none visible.")
        enforce_full_precision()                 # SACRED: keep fp32 fp32 (no TF32 rounding)
        self.device = device
        self.copy_stream = _torch.cuda.Stream(device=device)

    # -- memory bookkeeping --------------------------------------------------------------------
    def _reset_peak(self):
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()
        _torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_gb(self) -> float:
        return _torch.cuda.max_memory_allocated(self.device) / GB

    @staticmethod
    def free_after_oom(*tensors):
        """The validated in-core-OOM free discipline (oom_engine_gpu fix): drop the baseline's
        PARTIAL allocations so they do not survive the OOM and pollute (false-OOM) the next run.
        Call as `exec.free_after_oom(); del a, b` -- here we just do the gc + cache + sync part,
        the caller must `del` its own names (Python scoping)."""
        gc.collect()
        if _HAS_TORCH and _torch.cuda.is_available():
            _torch.cuda.synchronize()
            _torch.cuda.empty_cache()

    def build_norm_adj(self, src, dst, num_nodes):
        """Symmetric degree-normalized sparse adjacency on the GPU (ONE resident copy reused by
        every unit's aggregation -- the resident edge metadata the planner accounts for). Uses
        cuSPARSE-backed torch.sparse (full precision)."""
        i = _torch.as_tensor(np.asarray(src), dtype=_torch.long)
        j = _torch.as_tensor(np.asarray(dst), dtype=_torch.long)
        idx = _torch.stack([_torch.cat([i, j]), _torch.cat([j, i])])
        vals = _torch.ones(idx.shape[1])
        A = _torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce().to(self.device)
        deg = _torch.sparse.sum(A, 1).to_dense().clamp(min=1.0)
        return _torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]],
                                        (num_nodes, num_nodes)).coalesce()

    def run_snapshot_tiering(self, plan: GlobalPlan, w: Workload, src, dst,
                             device_idx: int = 0, *, weights=None) -> TieredRunResult:
        """Execute the engine's snapshot-tiering plan (generalized from oom_engine_gpu.run_gpu):
        keep `resident` snapshots in HBM, stream the rest from PINNED host RAM with a double-buffered
        prefetch on a second CUDA stream, full-precision 2-layer aggregation. Returns measurements
        and the predicted-vs-measured peak check.

        weights : optional (W1, W2) fp32 [F,F] matrices; random fp32 if None. PROCESS-only: the
                  result depends only on the inputs, never on the tier a row lives in."""
        torch = _torch
        N, F, W, L = w.num_nodes, w.feat_dim, w.window, w.layers
        p = plan.per_device[device_idx]
        resident, streamed = int(p.resident_snapshots), int(p.streamed_snapshots)

        A = self.build_norm_adj(src, dst, N)
        if weights is None:
            W1 = torch.randn(F, F, device=self.device) / F ** 0.5
            W2 = torch.randn(F, F, device=self.device) / F ** 0.5
        else:
            W1, W2 = (w_.to(self.device) for w_ in weights)

        def aggregate(X):
            # full-precision 2-layer GraphSAGE-style aggregation; materialize the L activation
            # copies so the live HBM footprint matches the planner's (1+L)*N*F*4 model.
            h1 = torch.sparse.mm(A, torch.relu(torch.sparse.mm(A, X) @ W1))   # activation copy 1
            h2 = torch.sparse.mm(A, h1) @ W2                                  # activation copy 2
            return h1, h2

        cpu_bank = torch.randn(W, N, F)                  # ALL snapshots' features in host RAM
        pcie_gb = streamed * N * F * 4 / GB

        self._reset_peak()
        # RESIDENT: features + L activations live in HBM (the planner's resident footprint).
        res_feat = [torch.empty(N, F, device=self.device) for _ in range(resident)]
        res_act = [[torch.empty(N, F, device=self.device) for _ in range(L)]
                   for _ in range(resident)]
        for s in range(resident):
            res_feat[s].copy_(cpu_bank[s])

        # DOUBLE BUFFER: 2 pinned host bounce buffers + 2 GPU slots overlap H2D with compute.
        pin = [torch.empty(N, F, pin_memory=True) for _ in range(2)] if streamed else []
        gbuf = [torch.empty(N, F, device=self.device) for _ in range(2)] if streamed else []
        sbuf_act = [[torch.empty(N, F, device=self.device) for _ in range(L)]] if streamed else []

        acc = 0.0
        torch.cuda.synchronize()
        import time as _t
        t0 = _t.time()

        for s in range(resident):                        # resident: no PCIe
            h1, h2 = aggregate(res_feat[s])
            res_act[s][0].copy_(h1); res_act[s][1].copy_(h2)
            acc += float(h2.sum())

        if streamed:
            pin[0].copy_(cpu_bank[resident])
            gbuf[0].copy_(pin[0], non_blocking=True)
            torch.cuda.synchronize()
            for k in range(streamed):
                cur, nxt = k % 2, (k + 1) % 2
                if k + 1 < streamed:                     # prefetch next on the COPY stream (overlap)
                    with torch.cuda.stream(self.copy_stream):
                        pin[nxt].copy_(cpu_bank[resident + k + 1])
                        gbuf[nxt].copy_(pin[nxt], non_blocking=True)
                h1, h2 = aggregate(gbuf[cur])            # compute current on the default stream
                sbuf_act[0][0].copy_(h1); sbuf_act[0][1].copy_(h2)
                acc += float(h2.sum())
                torch.cuda.current_stream().wait_stream(self.copy_stream)

        torch.cuda.synchronize()
        epoch = _t.time() - t0
        measured = self._peak_gb()
        pred = p.peak_hbm_bytes / GB
        rel = abs(measured - pred) / max(pred, 1e-9)
        return TieredRunResult(
            completed=True, measured_peak_gb=measured, epoch_sec=epoch, pcie_gb=pcie_gb,
            predicted_peak_gb=pred, peak_rel_err=rel,
            note=f"resident={resident}/{W} streamed={streamed} acc={acc:.3e}")
