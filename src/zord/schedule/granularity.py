"""zord GRANULARITY selector -- a USABLE optimal-partition-count K* (BACKLOG D3).

D3 was a faked-complete: the closed-form K* = W*ln2/alpha OVERSHOT the swept argmin by 7-235x.
ROOT CAUSE (found 2026-06-02): that formula minimises  makespan(K) = W_compute/K + alpha*log2(K)
-- which is monotone-ish and pushes K huge -- because it OMITS the PER-PARTITION FIXED OVERHEAD:
every extra partition adds a fixed per-step cost (kernel launch + its share of the all-reduce +
the cut/boundary it introduces), so TOTAL overhead GROWS like o*K. The honest makespan model is

      makespan(K) = W_compute / K   (strong-scaling compute, divides K ways)
                  + o_pp * K         (per-partition fixed overhead -- the term D3 dropped)
                  + alpha * log2(K) + beta   (BSP barrier / all-reduce, alpha MEASURED = 54.5us/hop)

whose interior optimum is bounded (d/dK = -W/K^2 + o_pp + alpha/(K ln2) = 0 -> K* ~ sqrt(W/o_pp)),
NOT W*ln2/alpha. We do NOT trust a closed form: we SWEEP K over a ladder from the EXACT memory
floor K_mem upward and take the argmin of this full model. K_mem itself is exact and usable
(finer is MANDATORY below it for feasibility); K* is the cheap-sweep interior optimum at the
measured alpha. PROCESS-only (a partition count is a placement choice).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import math

# measured NVLink all-reduce per-hop barrier latency (bvk 78950): alpha=54.5us, beta=45us.
ALPHA_MS_DEFAULT = 0.0545
BETA_MS_DEFAULT = 0.045


@dataclass
class KSelection:
    k_star: int                       # the chosen partition count (>= k_mem)
    k_mem: int                        # exact memory floor: smallest K whose per-device working set fits
    makespan_ms: float                # modeled makespan at k_star
    curve: Dict[int, float] = field(default_factory=dict)   # K -> modeled makespan (the swept ladder)
    note: str = ""


def select_k(compute_ms_at_1: float, working_set_bytes: float, cap_bytes: float, *,
             per_partition_overhead_ms: float = 0.05, alpha_ms: float = ALPHA_MS_DEFAULT,
             beta_ms: float = BETA_MS_DEFAULT, k_max: int = 0) -> KSelection:
    """Pick the partition count K.

    compute_ms_at_1 : total compute makespan on ONE device (the work that strong-scales as /K).
    working_set_bytes : total resident working set; cap_bytes : per-device usable HBM.
    per_partition_overhead_ms (o_pp): fixed per-step cost each added partition imposes (launch +
        all-reduce participation + boundary). This is the term D3's closed form omitted; it bounds
        K*. Default 0.05ms (~ a kernel launch + a hop). alpha_ms : MEASURED per-hop barrier.
    Returns K_mem (exact floor), K* (swept interior optimum), and the makespan(K) curve.
    """
    k_mem = max(1, math.ceil(working_set_bytes / max(1.0, cap_bytes)))
    if k_max <= 0:
        k_max = max(k_mem * 64, k_mem + 1)
    # geometric ladder from the memory floor upward (cheap: O(log) evals), + the endpoints
    ladder = set()
    k = k_mem
    while k <= k_max:
        ladder.add(k)
        k = max(k + 1, int(k * 1.4))
    ladder.add(k_max)
    curve: Dict[int, float] = {}
    for K in sorted(ladder):
        comp = compute_ms_at_1 / K                       # compute strong-scales
        overhead = per_partition_overhead_ms * K          # per-partition fixed overhead (the missing term)
        barrier = alpha_ms * math.log2(max(2, K)) + beta_ms
        curve[K] = comp + overhead + barrier
    k_star = min(curve, key=curve.get)
    # the closed-form D3 estimate (for the honest comparison / to show the overshoot it avoids)
    k_closed = max(1, int(round(compute_ms_at_1 * math.log(2) / max(1e-9, alpha_ms))))
    return KSelection(
        k_star=k_star, k_mem=k_mem, makespan_ms=curve[k_star], curve=curve,
        note=(f"K_mem={k_mem} (exact memory floor, finer mandatory below it); K*={k_star} via sweep "
              f"of makespan(K)=W/K + o_pp*K + alpha*log2(K) at measured alpha={alpha_ms}ms/hop, "
              f"o_pp={per_partition_overhead_ms}ms. (D3 closed-form W*ln2/alpha would say "
              f"K={k_closed} -- the overshoot the o_pp*K term corrects.)"))
