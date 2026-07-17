"""Runtime layer (zord kernel). Once a Plan is chosen, the runtime EXECUTES it:

  - memtier   : the CPU<->HBM tiering EXECUTOR. Consumes a plan_memory decision, holds the resident
                hot state in GPU HBM, the cold state in PINNED host RAM, and prefetches the streamed
                state with a DOUBLE BUFFER (H2D overlapped with compute on a second CUDA stream).
                Promoted (validated) from scripts/oom_engine_gpu.py + scripts/oom_attr_gpu.py.
                FULL PRECISION -- TF32 disabled -- the same-result invariant is sacred. Import-safe
                with no torch/CUDA (a CPU-simulation byte-budget + reference aggregation path).
  - watermark : the OVERFLOW / BACKPRESSURE policy from §42. High/low watermarks, reactive-vs-
                proactive decision from measured fill/evict rates (reactive iff evict_BW/fill_BW>=1),
                incremental (snapshot-granular) allocation with a between-alloc watermark check (a
                single alloc bigger than free HBM = instant OOM, zero reaction). Pure-python (no
                torch) so it is testable on CPU. Documents buffer-GPU(NVLink ~6.8x) vs CPU-spill(~1.05x).
  - bufferpool: the O2 admission/eviction pool over the KNOWN-FUTURE snapshot schedule (Belady/MRD),
                reporting hit-rate + staged-bytes reduction vs the naive double-buffer. C++ hot path
                (build/bufferpool) with a pure-numpy fallback; torch-free.
  - coexec    : the K3/O3 CPU||GPU co-execution planner -- the CPU does a REAL partial aggregation on
                the cold/spilled rows WHILE the GPU runs the resident partition; result-preserving
                (verify_coexec_result is the fp64 same-result certificate). Pure numpy; torch-free.
  - feature_recombine : the MUST-DO #1 / H2 closure -- the Megatron-style full-layer (aggregate ->
                W-mix all-reduce -> nonlinearity) feature-split recombination + the END-TO-END
                same-result certificate (verify_recombine; max-abs-err <= 1e-4). numpy fp64 reference,
                optional torch fp32 (TF32 disabled).

The static layout decision lives in zord.schedule.plan_memory; this layer carries it out.
"""
from .memtier import (
    TierBudget, budget_from_plan, simulate_tiering, Aggregator,
    TieredExecutor, TieredRunResult,
    enforce_full_precision, torch_available, cuda_available,
)
from .watermark import (
    OverflowMode, SpillTier, WatermarkPolicy, AdmissionResult,
    policy_from_rates, score_spill_tier,
)
from .bufferpool import (
    BufferPool, BufferPoolPlan, pool_from_plan,
    belady_schedule, mrd_schedule, window_access_sequence, have_bufferpool,
)
from .coexec import (
    CoExecPlan, plan_coexec, coexec_makespan_ms, split_work, verify_coexec_result,
)
from .feature_recombine import (
    RecombineSpec, plan_recombine, verify_recombine,
    reference_llayer, sharded_llayer,
    recombine_full_layer, recombine_aggregation,
    split_columns, split_weight_rows,
)

__all__ = [
    # memtier (CPU<->HBM tiering executor)
    "TierBudget", "budget_from_plan", "simulate_tiering", "Aggregator",
    "TieredExecutor", "TieredRunResult",
    "enforce_full_precision", "torch_available", "cuda_available",
    # watermark (overflow / backpressure policy)
    "OverflowMode", "SpillTier", "WatermarkPolicy", "AdmissionResult",
    "policy_from_rates", "score_spill_tier",
    # bufferpool (O2: Belady/MRD pool over the known snapshot future)
    "BufferPool", "BufferPoolPlan", "pool_from_plan",
    "belady_schedule", "mrd_schedule", "window_access_sequence", "have_bufferpool",
    # coexec (K3/O3: CPU||GPU co-execution)
    "CoExecPlan", "plan_coexec", "coexec_makespan_ms", "split_work", "verify_coexec_result",
    # feature_recombine (H2: full-layer feature-split recombination + same-result certificate)
    "RecombineSpec", "plan_recombine", "verify_recombine",
    "reference_llayer", "sharded_llayer",
    "recombine_full_layer", "recombine_aggregation",
    "split_columns", "split_weight_rows",
]
