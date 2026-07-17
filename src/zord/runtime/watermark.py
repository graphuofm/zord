"""OVERFLOW / BACKPRESSURE policy (zord runtime kernel) -- the §42 data, formalized.

This is the runtime complement to the STATIC plan_memory tiering: plan_memory decides the layout
BEFORE an epoch; this module decides, DURING the epoch, whether HBM overflow can be handled
REACTIVELY (watermark-triggered spill in the reaction window between allocations) or whether the
runtime must reserve headroom PROACTIVELY (admission control) because we fill faster than we evict.

Promoted from scripts/fill_dynamics.py + docs/RESULTS.md §42 (measured on a real H100), per the
user's rule that validated functionality MUST live in the kernel, not in a script.

----------------------------------------------------------------------------------------------------
THE §42 RESULT (the data this policy encodes):
  FILL  rate : the resident working set grows ~3.81 GB / snapshot at ~47.7 GB/s (alloc + a real SpMM).
  EVICT rate : D2H stage-out to pinned CPU ~50.1 GB/s; H2D refill ~53.6 GB/s -- PCIe-bound.
  VIABILITY  : evict_BW / fill_BW = 1.05  ->  REACTIVE watermark/spill is BARELY viable on CPU-PCIe
               (razor-thin 1.05x margin -> ALSO needs a safety headroom; do not run to 100%).
  HARD LIMIT : a SINGLE alloc bigger than current free HBM = INSTANT OOM, ZERO reaction time,
               regardless of bandwidth. => the runtime MUST allocate at SNAPSHOT (incremental)
               granularity and watermark-check BETWEEN allocations. A single oversized alloc has
               no reaction window. This is the feasibility lever: keep each increment < free HBM.

  BUFFER-GPU vs CPU-SPILL hierarchy (the user's idea, §42 implication 2):
    * spilling over NVLink to a buffer GPU (~325 GB/s) gives evict/fill ~= 325/48 ~= 6.8x -- a
      comfortable margin -> the HIGH-MARGIN overflow tier.
    * CPU-spill over PCIe (~50 GB/s) gives ~1.05x -- the thin FALLBACK when no GPU headroom exists.
  The policy below scores both tiers from their bandwidths so the scheduler can pick.

PURE-PYTHON policy object: NO torch needed (it reasons about bytes/sec), so it is testable on CPU.
PROCESS-only: it governs WHERE/WHEN bytes move and whether an alloc is admitted, never WHAT is
computed -- the same-result invariant is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

GB = 1024 ** 3


class OverflowMode(Enum):
    REACTIVE = "reactive"       # watermark-triggered spill in the inter-alloc reaction window
    PROACTIVE = "proactive"     # reserve headroom upfront / admission-control (fill > evict)


class SpillTier(Enum):
    BUFFER_GPU = "buffer-gpu"   # spill over NVLink to a buffer GPU (~6.8x margin) -- high margin
    CPU = "cpu"                 # spill over PCIe to host RAM (~1.05x margin) -- thin fallback


@dataclass
class WatermarkPolicy:
    """High/low watermark thresholds + the reactive-vs-proactive decision, derived from MEASURED
    fill/evict bandwidths (the §42 data). All fractions are of usable HBM capacity.

    high_frac : >= this fraction resident -> START spilling (the high watermark).
    low_frac  : spill DOWN to this fraction, then STOP (the low watermark; hysteresis prevents
                thrashing).
    safety_headroom_frac : never plan to fill above (1 - this); the thin 1.05x CPU margin demands
                we do NOT run to 100% (§42 implication 1).
    """
    capacity_bytes: int
    fill_gbps: float                # rate the resident working set GROWS per increment (GB/s)
    evict_gbps: float               # rate we can stage OUT (D2H / NVLink) (GB/s)
    increment_bytes: int            # bytes per incremental allocation (one snapshot/chunk)
    high_frac: float = 0.85
    low_frac: float = 0.70
    safety_headroom_frac: float = 0.10
    tier: SpillTier = SpillTier.CPU

    # ---- the core §42 decision ---------------------------------------------------------------
    @property
    def viability_ratio(self) -> float:
        """evict_BW / fill_BW. >= 1 -> we stage out at least as fast as we fill (reaction exists)."""
        return self.evict_gbps / self.fill_gbps if self.fill_gbps > 0 else float("inf")

    @property
    def mode(self) -> OverflowMode:
        """REACTIVE iff evict_BW/fill_BW >= 1 AND one increment evicts within one fill-interval.
        Otherwise PROACTIVE (we fill faster than we drain -> reserve headroom / admission-control)."""
        if self.viability_ratio < 1.0:
            return OverflowMode.PROACTIVE
        # can we evict one increment within the time it took to fill one increment?
        fill_interval_s = (self.increment_bytes / GB) / self.fill_gbps if self.fill_gbps > 0 else 0.0
        evict_per_interval_gb = self.evict_gbps * fill_interval_s
        return (OverflowMode.REACTIVE if evict_per_interval_gb >= self.increment_bytes / GB
                else OverflowMode.PROACTIVE)

    @property
    def reactive_viable(self) -> bool:
        return self.mode is OverflowMode.REACTIVE

    @property
    def high_watermark_bytes(self) -> int:
        return int(self.high_frac * self.capacity_bytes)

    @property
    def low_watermark_bytes(self) -> int:
        return int(self.low_frac * self.capacity_bytes)

    @property
    def usable_ceiling_bytes(self) -> int:
        """The highest resident bytes we PLAN to reach -- below 100% by the safety headroom (the
        thin 1.05x CPU margin means we must keep slack)."""
        return int((1.0 - self.safety_headroom_frac) * self.capacity_bytes)

    # ---- the HARD LIMIT: a single alloc bigger than free HBM = instant OOM, zero reaction -----
    def admit_allocation(self, resident_bytes: int, alloc_bytes: int) -> "AdmissionResult":
        """Decide whether the NEXT incremental allocation is admitted GIVEN the current resident
        bytes. Enforces the §42 hard limit: a single alloc bigger than free HBM OOMs INSTANTLY
        (no reaction window regardless of bandwidth). The fix is incremental granularity + a
        between-alloc watermark check -- exactly what this method performs.

        Returns an AdmissionResult: admitted / spill-first / reject, with the bytes to spill."""
        free = self.capacity_bytes - resident_bytes
        ceiling = self.usable_ceiling_bytes
        # 1. HARD LIMIT: the single alloc itself must fit in CURRENT free HBM (else instant OOM).
        if alloc_bytes > free:
            need_free = alloc_bytes - free
            if self.reactive_viable:
                # reactive: spill enough cold bytes NOW to make room (we can drain in the window)
                return AdmissionResult(admit=True, must_spill_bytes=int(need_free),
                                       reason=("single alloc exceeds free HBM -> REACTIVE spill "
                                               f"{need_free/GB:.2f}GB first (viable {self.viability_ratio:.2f}x)"))
            return AdmissionResult(admit=False, must_spill_bytes=int(need_free),
                                   reason=("single alloc exceeds free HBM and fill>evict "
                                           f"({self.viability_ratio:.2f}x) -> REJECT: must reserve "
                                           "headroom PROACTIVELY / shrink the increment"))
        # 2. WATERMARK: would this alloc cross the high watermark / safety ceiling?
        after = resident_bytes + alloc_bytes
        if after > ceiling:
            # spill down toward the low watermark to keep slack below the ceiling
            target = self.low_watermark_bytes
            spill = max(0, after - target)
            mode_ok = self.reactive_viable
            return AdmissionResult(
                admit=mode_ok, must_spill_bytes=int(spill),
                reason=(f"crosses safety ceiling ({ceiling/GB:.1f}GB); "
                        + ("REACTIVE spill to low watermark" if mode_ok
                           else "PROACTIVE: not reactive-viable -> admission denied, reserve upfront")))
        if after > self.high_watermark_bytes:
            spill = max(0, after - self.low_watermark_bytes)
            return AdmissionResult(admit=True, must_spill_bytes=int(spill),
                                   reason="crosses high watermark -> spill to low watermark (hysteresis)")
        return AdmissionResult(admit=True, must_spill_bytes=0, reason="fits below high watermark")

    def summary(self) -> str:
        m = self.mode
        return (f"[watermark] cap={self.capacity_bytes/GB:.1f}GB incr={self.increment_bytes/GB:.2f}GB "
                f"fill={self.fill_gbps:.1f} evict={self.evict_gbps:.1f}GB/s "
                f"viability={self.viability_ratio:.2f}x -> {m.value.upper()}  "
                f"high={self.high_frac:.0%} low={self.low_frac:.0%} "
                f"ceiling={self.usable_ceiling_bytes/GB:.1f}GB tier={self.tier.value}")


@dataclass
class AdmissionResult:
    admit: bool                     # may the allocation proceed (possibly after spilling)?
    must_spill_bytes: int           # bytes to stage out FIRST to make room / honor the watermark
    reason: str


# ---- spill-tier scoring (buffer-GPU over NVLink vs CPU over PCIe) ------------------------------
def score_spill_tier(fill_gbps: float, nvlink_gbps: float = 325.0,
                     pcie_gbps: float = 50.0) -> dict:
    """Score the two overflow tiers from their evict/fill margins (§42 implication 2). Returns the
    recommended tier + both margins. Buffer-GPU (NVLink ~6.8x) is the high-margin tier; CPU-spill
    (PCIe ~1.05x) is the thin fallback. PROCESS-only: this is a placement/bandwidth decision."""
    f = fill_gbps if fill_gbps > 0 else 1e-9
    gpu_margin = nvlink_gbps / f
    cpu_margin = pcie_gbps / f
    rec = SpillTier.BUFFER_GPU if gpu_margin >= 1.5 and gpu_margin > cpu_margin else SpillTier.CPU
    return {"recommended": rec, "buffer_gpu_margin": gpu_margin, "cpu_margin": cpu_margin,
            "note": (f"buffer-GPU/NVLink margin {gpu_margin:.1f}x vs CPU/PCIe {cpu_margin:.2f}x "
                     f"-> {'NVLink buffer GPU (high margin)' if rec is SpillTier.BUFFER_GPU else 'CPU spill (thin fallback)'}")}


def policy_from_rates(capacity_bytes: int, increment_bytes: int,
                      fill_gbps: float, evict_gbps: float,
                      tier: SpillTier = SpillTier.CPU,
                      high_frac: float = 0.85, low_frac: float = 0.70,
                      safety_headroom_frac: float = 0.10) -> WatermarkPolicy:
    """Build a WatermarkPolicy from MEASURED fill/evict rates (the fill_dynamics.py outputs).
    This is the kernel entry the runtime calls with the §42 numbers (or any device's measurements)."""
    return WatermarkPolicy(
        capacity_bytes=int(capacity_bytes), fill_gbps=float(fill_gbps),
        evict_gbps=float(evict_gbps), increment_bytes=int(increment_bytes),
        high_frac=high_frac, low_frac=low_frac,
        safety_headroom_frac=safety_headroom_frac, tier=tier)
