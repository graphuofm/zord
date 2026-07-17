"""zord RUNTIME MEMORY CONTROL LOOP -- the reactive feedback loop that keeps a step-by-step
workload inside HBM under SUDDEN memory spikes, WITHOUT freezing (OOM).

User idea (2026-06-02): "内存 buffer 用一个 loop 来解决内存突然爆满" -- borrow the control-loop
thinking pattern. zord's static plan (plan_memory) is OPEN-LOOP: it sizes the resident/streamed
split once. But a streaming/dynamic workload can spike (a burst of active nodes, a fat snapshot).
This module is the CLOSED-LOOP complement: each tick it MONITORS resident bytes and REACTS --
evict cold rows (to host RAM, or a buffer-GPU over NVLink), and if the input outruns the drain
rate it applies BACKPRESSURE (pause ingestion) instead of OOMing. It always reserves a HEADROOM
margin so a single oversized increment cannot instantly overflow (the §42 hard limit).

This is a thermostat / TCP-congestion-control / Flink-backpressure-style feedback loop:
    monitor -> decide -> act -> (repeat).
It composes with the static plan: the plan sets the baseline resident set; the loop handles the
surprises. PROCESS-only: evicting/spilling/pausing changes WHEN and WHERE rows live, never the
result. Pure-python + numpy-free so it runs and is testable on any box (the bandwidths are
parameters; §42 measured fill=47.7, host-evict=50.1, buffer-GPU/NVLink ~325 GB/s).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

GB = 1024 ** 3


@dataclass
class LoopConfig:
    """Control-loop parameters. Bandwidths in GB/s; bytes in bytes."""
    cap_bytes: float                      # device HBM capacity
    high_frac: float = 0.90               # evict when resident crosses this fraction of cap
    low_frac: float = 0.70                # evict down to this fraction
    headroom_frac: float = 0.08           # always keep this fraction free (single-alloc safety, §42)
    tick_s: float = 0.080                 # control period (= a snapshot interval; §42 ~80ms)
    fill_bw_gbps: float = 47.7            # rate the resident set grows (§42 measured)
    host_evict_bw_gbps: float = 50.1      # D2H spill rate to pinned host RAM (§42 PCIe)
    buffer_gpu_bw_gbps: Optional[float] = None  # if a spare GPU exists: NVLink spill (~325); else None
    host_cap_bytes: float = 256 * GB      # CPU bank size (overflow tier)


@dataclass
class TickResult:
    step: int
    admitted_bytes: float                 # how much of the incoming was admitted this tick
    deferred_bytes: float                 # backpressured (not admitted) this tick
    evicted_bytes: float                  # spilled out this tick
    spill_tier: str                       # "none" | "host" | "buffer_gpu"
    resident_after: float
    backpressure: bool                    # did we pause/slow ingestion?
    oom: bool                             # TRUE only if even backpressure could not save us
    headroom_after: float


@dataclass
class LoopTrace:
    ticks: List[TickResult] = field(default_factory=list)
    @property
    def any_oom(self) -> bool: return any(t.oom for t in self.ticks)
    @property
    def peak_resident(self) -> float: return max((t.resident_after for t in self.ticks), default=0.0)
    @property
    def total_evicted(self) -> float: return sum(t.evicted_bytes for t in self.ticks)
    @property
    def total_deferred(self) -> float: return sum(t.deferred_bytes for t in self.ticks)
    @property
    def backpressure_ticks(self) -> int: return sum(1 for t in self.ticks if t.backpressure)


class MemoryControlLoop:
    """Reactive HBM control loop. Drive it tick-by-tick with the incoming bytes per step; it
    keeps resident <= cap by evicting cold bytes and, when the drain can't keep up, deferring
    (backpressuring) the input. It never returns oom=True unless the WORKING SET ITSELF (resident
    that cannot be evicted) exceeds the cap -- a genuinely infeasible instant, which the static
    planner is responsible for preventing; the loop's job is the transient spike."""

    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
        self.resident = 0.0          # bytes currently in HBM
        self.host = 0.0              # bytes spilled to host RAM
        self.pinned = 0.0            # resident bytes that are HOT (cannot be evicted this tick)
        self.step = 0

    # --- the per-tick drain budget: how many bytes we can spill out in one control period ---
    def _evict_budget(self) -> tuple[float, str]:
        if self.cfg.buffer_gpu_bw_gbps and self.cfg.buffer_gpu_bw_gbps > self.cfg.host_evict_bw_gbps:
            return self.cfg.buffer_gpu_bw_gbps * 1e9 * self.cfg.tick_s, "buffer_gpu"
        return self.cfg.host_evict_bw_gbps * 1e9 * self.cfg.tick_s, "host"

    def tick(self, incoming_bytes: float, hot_bytes: Optional[float] = None) -> TickResult:
        """One control step. incoming_bytes: new resident demand this step. hot_bytes: of the
        CURRENT resident, how much is hot/un-evictable (defaults to a fraction kept resident)."""
        cfg = self.cfg
        self.step += 1
        high = cfg.high_frac * cfg.cap_bytes
        low = cfg.low_frac * cfg.cap_bytes
        usable = (1.0 - cfg.headroom_frac) * cfg.cap_bytes      # never plan above this (headroom)
        evict_cap, tier = self._evict_budget()
        if hot_bytes is None:
            hot_bytes = min(self.resident, low)                 # treat the low-watermark mass as hot

        evicted = 0.0
        # 1) PROACTIVE evict if we're already above the high watermark (make room before admitting).
        if self.resident > high:
            target = self.resident - low
            evictable = max(0.0, self.resident - hot_bytes)
            ev = min(target, evictable, evict_cap)
            self.resident -= ev; self.host += ev; evicted += ev

        # 2) ADMIT the increment only up to the usable ceiling (headroom-protected). The part that
        #    doesn't fit is DEFERRED (backpressure) -- we slow the producer rather than OOM.
        room = max(0.0, usable - self.resident)
        admit = min(incoming_bytes, room)
        deferred = incoming_bytes - admit
        self.resident += admit

        # 3) If we still had to defer, try to evict MORE (within this tick's drain budget) to admit
        #    the rest next tick faster; if even now resident would exceed cap, that's a true OOM
        #    (the planner should have prevented it) -- flag honestly.
        if deferred > 0 and evicted < evict_cap:
            extra_evictable = max(0.0, self.resident - hot_bytes)
            ev2 = min(evict_cap - evicted, extra_evictable, self.resident - low if self.resident > low else 0.0)
            self.resident -= ev2; self.host += ev2; evicted += ev2

        backpressure = deferred > 0
        oom = self.resident > cfg.cap_bytes or self.host > cfg.host_cap_bytes
        return TickResult(
            step=self.step, admitted_bytes=admit, deferred_bytes=deferred, evicted_bytes=evicted,
            spill_tier=(tier if evicted > 0 else "none"), resident_after=self.resident,
            backpressure=backpressure, oom=oom, headroom_after=cfg.cap_bytes - self.resident)

    def run(self, incoming_stream: List[float], hot_stream: Optional[List[float]] = None) -> LoopTrace:
        """Drive the loop over a sequence of per-step incoming byte demands. Deferred (backpressured)
        bytes roll forward to the next step (the producer is paused, not dropped) -- so the loop
        DELAYS work to stay feasible, never loses or OOMs it (unless genuinely infeasible)."""
        tr = LoopTrace()
        carry = 0.0
        for i, inc in enumerate(incoming_stream):
            hot = hot_stream[i] if hot_stream is not None else None
            res = self.tick(inc + carry, hot_bytes=hot)
            carry = res.deferred_bytes                 # backpressure: defer to next tick
            tr.ticks.append(res)
        # drain any remaining carry (producer paused; we keep ticking with no new input)
        guard = 0
        while carry > 1.0 and guard < 100000:
            res = self.tick(carry, hot_bytes=None); carry = res.deferred_bytes
            tr.ticks.append(res); guard += 1
        return tr


def viability(cfg: LoopConfig) -> dict:
    """Static §42-style check: can the loop's evict keep up with fill? ratio = evict_BW/fill_BW.
    >=1 -> reactive spill viable; <1 -> the loop must lean on backpressure (slow the producer)."""
    evict, tier = (cfg.buffer_gpu_bw_gbps, "buffer_gpu") if (cfg.buffer_gpu_bw_gbps and
                  cfg.buffer_gpu_bw_gbps > cfg.host_evict_bw_gbps) else (cfg.host_evict_bw_gbps, "host")
    ratio = evict / max(1e-9, cfg.fill_bw_gbps)
    return {"evict_bw_gbps": evict, "fill_bw_gbps": cfg.fill_bw_gbps, "ratio": ratio,
            "spill_tier": tier, "reactive_viable": ratio >= 1.0,
            "note": ("reactive spill keeps up" if ratio >= 1.0
                     else "evict < fill -> loop relies on backpressure (slows producer, never OOMs)")}
