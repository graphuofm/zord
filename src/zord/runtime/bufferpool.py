"""BUFFER POOL -- admission + eviction over the KNOWN-FUTURE snapshot schedule (zord runtime, O2).

The runtime stages temporal-graph state (per-snapshot node features + the resident adjacency)
between HBM, host RAM and (optionally) NVMe. The ad-hoc executor in `runtime.memtier` uses a
NAIVE DOUBLE BUFFER: two prefetch slots, so the H2D copy of snapshot s+1 overlaps compute of s --
but it re-stages EVERY non-resident snapshot every epoch, paying PCIe even when a snapshot it just
evicted is about to be reused. For a *temporal* graph the access order is KNOWN OFFLINE (the
window/stride schedule over the snapshots), so the cache problem is not an online guess: we can run
BELADY/MIN -- evict the unit whose NEXT use is farthest in the future -- which is the provably
optimal offline policy and is *actually computable* here precisely because the future is the
deterministic snapshot schedule. That is the temporal-graph advantage this module exploits.

  naive double-buffer (2 slots, blind re-stage)   ->   designed Belady/MRD pool (this module)
                                                        + reports predicted hit-rate and the
                                                        staging-bytes reduction vs naive (O2 win).

WHAT THIS MODULE PRODUCES (a PLAN, not an execution): given the per-device resident byte budget
(from a `plan_memory` GlobalPlan / TierBudget) and the snapshot access sequence, it computes the
admissions, evictions, the predicted HIT RATE, and the staged-bytes vs the naive double-buffer.
`BufferPoolPlan.staged_bytes` feeds back into the TierBudget / JobEstimate; the scheduler picks
`belady` (full offline plan over a fixed schedule) vs `mrd` (the practical online variant for an
evolving stream where the far future is unknown).

================================================================================================
RESULT-PRESERVING / PROCESS-ONLY (sacred, per zord's invariant):
  This module decides only WHICH unit is resident and WHEN it is staged -- never WHAT is computed.
  Admission/eviction reorder *storage*, not arithmetic; the aggregation result is byte-identical to
  the all-resident reference. There is NO precision knob here (it is pure cost + bookkeeping). All
  numpy + dataclasses, fully torch-free, so it imports and plans on a CPU box and is unit-testable.
================================================================================================
"""
from __future__ import annotations

import heapq
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Import-safe: planner is pure numpy/dataclasses (no torch). Used only for type hints + pool_from_plan.
from ..schedule.planner import GlobalPlan, Workload

GB = 1024 ** 3


# ============================================================================================== #
#  C++ kernel resolution + invocation (the HOT-PATH simulation; numpy fallback always available). #
# ============================================================================================== #
# Mirrors partition.cpp_kernel.graph_bin_path: env ZORD_BUFFERPOOL_BIN, else <repo>/build/bufferpool.
# Repo root is FOUR levels up from this file (src/zord/runtime/bufferpool.py).
_BELADY = 0
_MRD = 1


def bufferpool_bin_path() -> str:
    """Resolve the bufferpool binary: $ZORD_BUFFERPOOL_BIN, else <repo>/build/bufferpool."""
    env = os.environ.get("ZORD_BUFFERPOOL_BIN")
    if env:
        return env
    here = os.path.abspath(__file__)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    return os.path.join(repo, "build", "bufferpool")


def have_bufferpool() -> bool:
    return os.path.exists(bufferpool_bin_path())


def _simulate_cpp(access: np.ndarray, capacity_units: int, policy: str):
    """Run the C++ bufferpool simulation over the KNOWN access sequence. Returns
    (is_miss int8[L], admissions, evictions) or None so the caller falls back to the numpy loop.

    Binary I/O (little-endian), matching src/zord/cpp/bufferpool.cpp exactly:
      IN  : int64 L; int32 access[L]; int64 capacity_units; int32 policy(0=belady,1=mrd)
      OUT : int64 L; int32 is_miss[L]; int64 admissions; int64 evictions
    Returns None on a missing binary or a nonzero exit so plan_schedule uses the Python policy."""
    binp = bufferpool_bin_path()
    if not os.path.exists(binp):
        return None
    access = np.asarray(access, dtype=np.int64).ravel()
    L = int(access.size)
    if L == 0:
        return np.zeros(0, dtype=np.int8), 0, 0
    pol = _MRD if policy == "mrd" else _BELADY
    with tempfile.TemporaryDirectory(prefix="zord_kernel_") as tmp:
        ipath = os.path.join(tmp, "in.bin")
        opath = os.path.join(tmp, "out.bin")
        with open(ipath, "wb") as fh:
            fh.write(struct.pack("<q", L))
            access.astype(np.int32).tofile(fh)
            fh.write(struct.pack("<q", int(max(1, capacity_units))))
            fh.write(struct.pack("<i", int(pol)))
        r = subprocess.run([binp, ipath, opath], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        with open(opath, "rb") as fh:
            (n,) = struct.unpack("<q", fh.read(8))
            if n != L:
                return None
            is_miss = np.fromfile(fh, dtype=np.int32, count=L).astype(np.int8)
            (admissions,) = struct.unpack("<q", fh.read(8))
            (evictions,) = struct.unpack("<q", fh.read(8))
    if is_miss.size != L:
        return None
    return is_miss, int(admissions), int(evictions)

# Sentinel "next use" for a unit that is never referenced again on the remaining schedule. Belady
# evicts the LARGEST next_use first, so an unreferenced unit (the ideal eviction victim) gets the
# maximum priority. int64-safe (well below 2**63).
_NEVER = np.iinfo(np.int64).max


@dataclass
class BufferSlot:
    """One resident buffer entry. `next_use` = the next position in the access sequence that will
    reference this unit (Belady's lookahead). _NEVER if the unit is not used again."""
    unit: int
    bytes: int
    resident: bool
    next_use: int


@dataclass
class BufferPoolPlan:
    """The buffer-pool decision + the O2 reporting: hit-rate and staged-bytes reduction vs the
    naive double-buffer. This is what the scheduler stores per device and the CLI prints."""
    capacity_bytes: int             # HBM bytes the pool may hold resident (from the TierBudget peak)
    num_slots: int                  # resident slots = capacity_bytes // unit_bytes (>= 1)
    admissions: int                 # units staged INTO the pool over the schedule (cold + re-staged)
    evictions: int                  # units evicted to make room
    hit_rate: float                 # fraction of accesses already resident (no staging needed)
    staged_bytes: int               # bytes crossing PCIe under THIS policy (the misses)
    staged_bytes_naive: int         # bytes the naive double-buffer would stage (the baseline)
    staging_reduction: float        # 1 - staged_bytes/staged_bytes_naive (the O2 win; >=0)
    policy: str                     # "belady" | "mrd"
    per_step_resident: list         # resident-unit-count after each access (occupancy trace)
    note: str = ""

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate

    def summary(self) -> str:
        return (
            f"[bufferpool/{self.policy}] slots={self.num_slots} "
            f"cap={self.capacity_bytes / GB:.2f}GB  hit_rate={self.hit_rate * 100:.1f}%  "
            f"admissions={self.admissions} evictions={self.evictions}\n"
            f"  staged={self.staged_bytes / GB:.2f}GB vs naive double-buffer "
            f"{self.staged_bytes_naive / GB:.2f}GB  -> staging reduction "
            f"{self.staging_reduction * 100:.1f}%"
            + (f"  [{self.note}]" if self.note else ""))


# ============================================================================================== #
#  Helpers: precompute the next-use array (Belady lookahead) for a fixed access sequence.        #
# ============================================================================================== #
def _next_use_array(access: np.ndarray) -> np.ndarray:
    """For each position i in the access sequence, the NEXT position j>i with access[j]==access[i]
    (else _NEVER). This is Belady's lookahead, computed in one O(L) backward pass. Vectorized:
    walk right-to-left, remembering the last-seen position per unit."""
    L = access.size
    nxt = np.full(L, _NEVER, dtype=np.int64)
    last_seen: dict[int, int] = {}
    for i in range(L - 1, -1, -1):
        u = int(access[i])
        if u in last_seen:
            nxt[i] = last_seen[u]
        last_seen[u] = i
    return nxt


# ============================================================================================== #
#  BELADY / MIN -- the optimal offline policy (computable because the schedule is KNOWN).        #
# ============================================================================================== #
def belady_schedule(access_sequence: np.ndarray, capacity_units: int) -> tuple:
    """Optimal-offline (Belady/MIN) cache simulation over the KNOWN snapshot access order.

    On a miss with a full cache, evict the resident unit whose NEXT use is FARTHEST in the future
    (an unreferenced unit -> _NEVER -> evicted first). This is the minimum-miss reference that
    exploits the temporal-graph advantage: the future access order IS the deterministic schedule.

    access_sequence : 1-D int array of unit ids in access order (e.g. snapshot ids per window step).
    capacity_units  : number of resident slots (>= 1).
    Returns (admissions, evictions, hit_rate, per_step_resident).
    """
    access = np.asarray(access_sequence, dtype=np.int64).ravel()
    L = int(access.size)
    cap = max(1, int(capacity_units))
    if L == 0:
        return 0, 0, 1.0, []

    nxt = _next_use_array(access)
    resident: dict[int, int] = {}       # unit -> its current next_use value (for eviction choice)
    admissions = 0
    evictions = 0
    hits = 0
    per_step = []

    for i in range(L):
        u = int(access[i])
        if u in resident:
            hits += 1
            resident[u] = int(nxt[i])           # refresh this unit's next use
        else:
            admissions += 1
            if len(resident) >= cap:
                # evict the unit with the FARTHEST next use (Belady). Linear scan over <=cap slots.
                victim = max(resident, key=lambda k: resident[k])
                del resident[victim]
                evictions += 1
            resident[u] = int(nxt[i])
        per_step.append(len(resident))

    hit_rate = hits / L
    return admissions, evictions, float(hit_rate), per_step


# ============================================================================================== #
#  MRD -- practical ONLINE variant (Most-Reuse-Distance / MRU-by-estimated-reuse).               #
# ============================================================================================== #
def mrd_schedule(access_sequence: np.ndarray, capacity_units: int) -> tuple:
    """Practical ONLINE variant for the evolving-stream case where the far future is NOT a fixed
    schedule. We track each unit's RECENT reuse distance (the gap between its last two accesses) as
    an estimate of its next reuse, and on a miss evict the unit with the LARGEST estimated reuse
    distance (the one predicted to be used least soon) -- Most-Reuse-Distance eviction. A unit seen
    only once has an unknown (assumed large) reuse distance and is preferred for eviction, which
    coincides with MRU (evict the most-recently-used streaming-once unit). This needs no lookahead,
    so it works online; it approximates Belady and is what the scheduler uses for dynamic streams.

    Returns (admissions, evictions, hit_rate, per_step_resident).
    """
    access = np.asarray(access_sequence, dtype=np.int64).ravel()
    L = int(access.size)
    cap = max(1, int(capacity_units))
    if L == 0:
        return 0, 0, 1.0, []

    resident: dict[int, float] = {}     # unit -> estimated reuse distance (larger = evict sooner)
    last_pos: dict[int, int] = {}       # unit -> position of its previous access (any time)
    admissions = 0
    evictions = 0
    hits = 0
    per_step = []
    BIG = float(L)                      # reuse-distance estimate for a never-before-seen unit

    for i in range(L):
        u = int(access[i])
        # update this unit's estimated reuse distance from its observed history (online: backward).
        if u in last_pos:
            reuse_est = float(i - last_pos[u])
        else:
            reuse_est = BIG
        last_pos[u] = i

        if u in resident:
            hits += 1
            resident[u] = reuse_est
        else:
            admissions += 1
            if len(resident) >= cap:
                # evict the unit with the LARGEST estimated reuse distance (used least soon).
                victim = max(resident, key=lambda k: resident[k])
                del resident[victim]
                evictions += 1
            resident[u] = reuse_est
        per_step.append(len(resident))

    hit_rate = hits / L
    return admissions, evictions, float(hit_rate), per_step


# ============================================================================================== #
#  Staging-bytes accounting: this policy vs the naive double-buffer baseline.                    #
# ============================================================================================== #
def _naive_double_buffer_misses(access: np.ndarray, resident_units: int) -> int:
    """Misses under the NAIVE double-buffer that runtime.memtier emulates: the first `resident_units`
    DISTINCT units are pinned resident (the executor's resident bank); every access to any OTHER unit
    is re-staged through the 2 prefetch slots EVERY time it appears (no reuse cache for streamed
    units). This is the baseline the O2 buffer pool must beat. A "miss" = a staged access.

    resident_units : how many distinct units the naive scheme keeps pinned (= the same slot budget;
                     the double-buffer adds 2 prefetch slots on top but those do NOT cache reuse)."""
    access = np.asarray(access, dtype=np.int64).ravel()
    if access.size == 0:
        return 0
    rcap = max(0, int(resident_units))
    # The pinned resident set = the first rcap DISTINCT units encountered (the executor fills its
    # resident bank greedily in schedule order, exactly as memtier loads res_feat[0..resident-1]).
    pinned: set[int] = set()
    misses = 0
    for u in access:
        ui = int(u)
        if ui in pinned:
            continue                            # resident hit, no staging
        if len(pinned) < rcap:
            pinned.add(ui)                      # admit into the pinned bank (a cold load, but the
            #                                     naive scheme ALSO pays this once; counted identically
            #                                     below so the comparison is apples-to-apples) -- we
            #                                     treat the initial fill as NOT a recurring miss.
            continue
        misses += 1                             # streamed unit: re-staged on EVERY access
    return misses


def _per_unit_bytes(access: np.ndarray, unit_bytes) -> np.ndarray:
    """Resolve per-access byte cost. `unit_bytes` may be a scalar (uniform snapshot bytes) or a
    per-unit array indexed by unit id. Returns a [L] array of bytes for each access position."""
    access = np.asarray(access, dtype=np.int64).ravel()
    if unit_bytes is None:
        return np.ones(access.size, dtype=np.float64)
    arr = np.asarray(unit_bytes, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(access.size, float(arr), dtype=np.float64)
    return arr[access]                          # per-unit byte size, gathered in access order


# ============================================================================================== #
#  The BufferPool object: incremental admit/evict + the full-schedule plan + reporting.          #
# ============================================================================================== #
class BufferPool:
    """A buffer pool with a designed admission/eviction policy over a KNOWN snapshot access order.

    Two uses:
      * `plan_schedule(access_sequence)` -- offline: simulate the whole schedule under the chosen
        policy (belady/mrd), report admissions/evictions/hit-rate and the staged-bytes reduction vs
        the naive double-buffer (the O2 deliverable).
      * `admit(unit, next_use)` -- incremental bookkeeping for a streaming executor: returns the
        evicted units (Belady picks the resident unit with the farthest next_use). Pure bookkeeping;
        the caller does the actual H2D copy.
    """

    def __init__(self, capacity_bytes: int, unit_bytes: int, *, policy: str = "belady"):
        if policy not in ("belady", "mrd"):
            raise ValueError(f"policy must be 'belady' or 'mrd', got {policy!r}")
        self.capacity_bytes = int(capacity_bytes)
        self.unit_bytes = max(1, int(unit_bytes))
        self.policy = policy
        # resident slots from the byte budget; at least 1 so a single unit can always be processed.
        self.num_slots = max(1, self.capacity_bytes // self.unit_bytes)
        # incremental state for admit(): unit -> next_use (Belady) or insertion order (mrd fallback).
        self._resident: dict[int, int] = {}

    # -- incremental API (for a streaming executor; pure bookkeeping) --------------------------
    def admit(self, unit: int, next_use: int) -> list:
        """Admit `unit` (with its known `next_use` position) into the pool, evicting if full.
        Returns the list of evicted units. Belady: evict the resident unit whose next_use is the
        FARTHEST in the future. Idempotent on a hit (just refreshes next_use)."""
        unit = int(unit)
        evicted: list = []
        if unit in self._resident:                  # hit -> refresh lookahead, no eviction
            self._resident[unit] = int(next_use)
            return evicted
        if len(self._resident) >= self.num_slots:
            # evict the farthest-next-use unit (Belady). For mrd the caller passes a reuse-distance
            # estimate as `next_use`, so the same "evict max" rule applies -- evict used-least-soon.
            victim = max(self._resident, key=lambda k: self._resident[k])
            del self._resident[victim]
            evicted.append(victim)
        self._resident[unit] = int(next_use)
        return evicted

    @property
    def resident_units(self) -> list:
        return list(self._resident.keys())

    def reset(self) -> None:
        self._resident.clear()

    # -- offline plan over the full known schedule ---------------------------------------------
    def plan_schedule(self, access_sequence: np.ndarray, *,
                      unit_bytes: Optional[np.ndarray] = None) -> BufferPoolPlan:
        """Simulate the whole KNOWN snapshot access order under this pool's policy and report the
        O2 metrics: hit-rate, admissions/evictions, and staged-bytes vs the naive double-buffer.

        access_sequence : 1-D int array of unit ids in access order. For a DTDG window schedule this
                          is the per-window snapshot ids (see pool_from_plan / build access helpers).
        unit_bytes      : OPTIONAL per-unit byte sizes (array indexed by unit id) when snapshots are
                          unequal; defaults to this pool's uniform `self.unit_bytes` for every unit.
        """
        access = np.asarray(access_sequence, dtype=np.int64).ravel()
        L = int(access.size)
        # per-access byte cost (uniform unit_bytes unless a per-unit vector is supplied)
        if unit_bytes is None:
            ubytes = np.full(L, float(self.unit_bytes), dtype=np.float64)
        else:
            ubytes = _per_unit_bytes(access, unit_bytes)

        # ---- HOT PATH: route the per-access simulation through the C++ kernel when present. It
        # returns is_miss[] + admissions/evictions; we do the byte accounting in numpy here so the
        # result is IDENTICAL to the Python policy loops (the C++ reproduces the same miss set,
        # including the eviction tie-break). Falls back to the pure-numpy belady/mrd loops below
        # when the binary is absent or its run fails -- the plan ALWAYS computes (correctly, slower).
        cpp = _simulate_cpp(access, self.num_slots, self.policy)
        cpp_backed = False
        if cpp is not None:
            is_miss, admissions, evictions = cpp
            hits = L - int(admissions)
            hit_rate = (hits / L) if L > 0 else 1.0
            # staged bytes = sum of per-access bytes over the miss positions (is_miss * ubytes).
            staged_bytes = float(np.dot(is_miss.astype(np.float64), ubytes)) if L > 0 else 0.0
            # occupancy after access i: cumulative misses clamped at the slot count (room is filled
            # monotonically until full, then every miss evicts exactly one so the count holds at cap).
            if L > 0:
                cum = np.cumsum(is_miss.astype(np.int64))
                per_step = np.minimum(cum, self.num_slots).tolist()
            else:
                per_step = []
            cpp_backed = True
        else:
            if self.policy == "belady":
                admissions, evictions, hit_rate, per_step = belady_schedule(access, self.num_slots)
            else:
                admissions, evictions, hit_rate, per_step = mrd_schedule(access, self.num_slots)
            # ---- staged bytes under THIS policy: every miss (admission) re-stages that unit's bytes.
            # Recompute miss positions deterministically so byte accounting matches the policy exactly.
            staged_bytes = self._staged_bytes_for_policy(access, ubytes)

        # ---- naive double-buffer baseline: pin the first num_slots distinct units, re-stage every
        # other access. Its staged bytes are the misses' byte costs.
        naive_misses_bytes = self._staged_bytes_naive(access, ubytes)

        reduction = (1.0 - staged_bytes / naive_misses_bytes) if naive_misses_bytes > 0 else 0.0
        reduction = max(0.0, float(reduction))

        note = (f"known-future schedule (L={L}, distinct={len(np.unique(access))}); "
                f"Belady-optimal offline" if self.policy == "belady"
                else f"online MRD over L={L} accesses")
        note += " [cpp]" if cpp_backed else " [numpy]"
        return BufferPoolPlan(
            capacity_bytes=self.capacity_bytes, num_slots=self.num_slots,
            admissions=int(admissions), evictions=int(evictions), hit_rate=float(hit_rate),
            staged_bytes=int(round(staged_bytes)),
            staged_bytes_naive=int(round(naive_misses_bytes)),
            staging_reduction=reduction, policy=self.policy,
            per_step_resident=per_step, note=note)

    # -- internal: byte accounting that mirrors the chosen policy's miss set -------------------
    def _staged_bytes_for_policy(self, access: np.ndarray, ubytes: np.ndarray) -> float:
        """Sum the byte cost of every MISS under this pool's policy. Re-runs the policy's eviction
        rule tracking which accesses are misses, multiplying each miss by its per-access bytes."""
        L = int(access.size)
        cap = self.num_slots
        total = 0.0
        if self.policy == "belady":
            nxt = _next_use_array(access)
            resident: dict[int, int] = {}
            for i in range(L):
                u = int(access[i])
                if u in resident:
                    resident[u] = int(nxt[i])
                else:
                    total += float(ubytes[i])           # MISS -> stage this unit's bytes
                    if len(resident) >= cap:
                        victim = max(resident, key=lambda k: resident[k])
                        del resident[victim]
                    resident[u] = int(nxt[i])
        else:
            resident_r: dict[int, float] = {}
            last_pos: dict[int, int] = {}
            BIG = float(L)
            for i in range(L):
                u = int(access[i])
                reuse_est = float(i - last_pos[u]) if u in last_pos else BIG
                last_pos[u] = i
                if u in resident_r:
                    resident_r[u] = reuse_est
                else:
                    total += float(ubytes[i])
                    if len(resident_r) >= cap:
                        victim = max(resident_r, key=lambda k: resident_r[k])
                        del resident_r[victim]
                    resident_r[u] = reuse_est
        return total

    def _staged_bytes_naive(self, access: np.ndarray, ubytes: np.ndarray) -> float:
        """Bytes the NAIVE double-buffer stages: pin the first num_slots distinct units; every access
        to any other unit re-stages it (no reuse cache for streamed units). The initial pinned fill is
        NOT counted as a recurring miss (the policy path treats cold loads symmetrically -- both pay
        the first touch), so the comparison isolates the REUSE win of Belady/MRD over blind re-stage."""
        L = int(access.size)
        rcap = self.num_slots
        pinned: set[int] = set()
        total = 0.0
        for i in range(L):
            u = int(access[i])
            if u in pinned:
                continue
            if len(pinned) < rcap:
                pinned.add(u)                            # initial pinned fill (not a recurring miss)
                continue
            total += float(ubytes[i])                    # streamed unit re-staged on every access
        return total


# ============================================================================================== #
#  Access-sequence construction from the snapshot/window schedule (the temporal advantage).      #
# ============================================================================================== #
def window_access_sequence(num_snapshots: int, window: int = 1,
                           stride: Optional[int] = None, num_epochs: int = 1) -> np.ndarray:
    """Build the KNOWN snapshot access order from the DTDG window schedule (mirrors
    TemporalGraph.batches: windows of `window` snapshots, advancing by `stride`). Each window step
    accesses its `window` snapshot ids in order; the sequence repeats over `num_epochs`. This is the
    deterministic future that makes Belady computable for a temporal graph.

    Returns a 1-D int64 array of snapshot ids in access order."""
    S = max(1, int(num_snapshots))
    w = max(1, int(window))
    stride = int(stride) if stride else w
    starts = list(range(0, max(1, S - w + 1), max(1, stride)))
    one_epoch: list[int] = []
    for s in starts:
        one_epoch.extend(range(s, min(s + w, S)))
    if not one_epoch:
        one_epoch = list(range(S))
    seq = one_epoch * max(1, int(num_epochs))
    return np.asarray(seq, dtype=np.int64)


# ============================================================================================== #
#  Build a buffer-pool plan for one device from a GlobalPlan (the runtime wiring entry).         #
# ============================================================================================== #
def pool_from_plan(plan: GlobalPlan, w: Workload, device: int = 0, *,
                   num_snapshots: int = 64, window: Optional[int] = None,
                   stride: Optional[int] = None, num_epochs: int = 1,
                   policy: Optional[str] = None) -> BufferPoolPlan:
    """Build the buffer-pool plan for one device from a plan_memory GlobalPlan.

    The pool's CAPACITY is the device's resident HBM budget (peak_hbm_bytes), the UNIT is one
    snapshot's resident state (per-device features + L activations + adjacency), and the ACCESS
    SEQUENCE is the window/snapshot schedule over `num_snapshots` (the known temporal future).
    Returns a BufferPoolPlan whose staged_bytes/staging_reduction feed the TierBudget/JobEstimate.

    policy : "belady" (offline plan over a fixed schedule, the default) or "mrd" (online stream).
             The scheduler passes "mrd" for the dynamic_online / evolving-stream case.
    """
    p = plan.per_device[device]
    win = int(window) if window is not None else int(getattr(w, "window", 1) or 1)
    # one snapshot's resident bytes on THIS device = peak split across the co-resident window. The
    # planner's peak is for `window` co-resident snapshots; a single unit is that divided by window.
    # Fall back to the per-snapshot feature+activation+adj estimate when peak is unavailable.
    if p.peak_hbm_bytes > 0 and win >= 1:
        unit_bytes = max(1, int(p.peak_hbm_bytes // max(1, win)))
    else:
        n_k = max(1, int(p.work_nodes))
        e_k = max(0, int(p.work_edges))
        unit_bytes = int(n_k * w.feat_dim * w.bytes_per_feat * (1 + w.layers)
                         + e_k * w.bytes_per_edge)
        unit_bytes = max(1, unit_bytes)

    # capacity available to the pool = the device's resident peak budget (what stays in HBM).
    capacity_bytes = int(p.peak_hbm_bytes) if p.peak_hbm_bytes > 0 else int(p.capacity_bytes)

    chosen = policy if policy is not None else "belady"
    access = window_access_sequence(num_snapshots, window=win, stride=stride, num_epochs=num_epochs)

    pool = BufferPool(capacity_bytes, unit_bytes, policy=chosen)
    bpp = pool.plan_schedule(access)
    bpp.note = (f"dev{device} {p.name}: unit~{unit_bytes / GB:.2f}GB, "
                f"window={win}, S={num_snapshots}; " + bpp.note)
    return bpp
