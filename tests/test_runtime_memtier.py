"""Tests for the zord runtime kernel (promoted from scripts/oom_*_gpu.py + fill_dynamics.py).

Two invariants, both checkable on a CPU box (no torch/CUDA needed):

  1. SAME-RESULT (memtier CPU-sim path): the tiered execution -- some rows "resident", some
     "streamed" -- reproduces the single-device reference aggregation EXACTLY. Tiering changes
     WHERE a row lives, never WHAT is computed (zord's sacred PROCESS-only invariant). We also
     assert the resident+streamed BYTE budget extracted from a plan_memory result reconstructs the
     full feature set.

  2. WATERMARK policy (watermark.py): reactive-vs-proactive is picked correctly from synthetic
     fill/evict rates (REACTIVE iff evict_BW/fill_BW >= 1 AND one increment drains in one interval),
     and the §42 hard limit (a single alloc bigger than free HBM is never admitted unless we can
     spill first) is enforced.
"""
import numpy as np
import pytest

from zord.profiler.cluster_profile import from_spec
from zord.schedule import Workload, plan_memory
from zord.runtime import (
    Aggregator, simulate_tiering, budget_from_plan,
    WatermarkPolicy, OverflowMode, SpillTier, policy_from_rates, score_spill_tier,
)

GB = 1024 ** 3


# ----------------------------------------------------------------------------- #
# 1. SAME-RESULT invariant: tiered (resident+streamed) == single-device reference #
# ----------------------------------------------------------------------------- #
def _toy_graph(seed=0):
    rng = np.random.default_rng(seed)
    N, E = 200, 1600
    src = rng.integers(0, N, size=E, dtype=np.int64)
    dst = rng.integers(0, N, size=E, dtype=np.int64)
    return src, dst, N


def test_tiered_aggregation_equals_single_device():
    """The CPU-sim tiering path reproduces the single-device reference aggregation EXACTLY:
    splitting rows into a 'resident' slab + a 'streamed' slab and aggregating the concatenation
    must equal aggregating the whole feature matrix in one shot. (Tiering moves WHERE rows live.)

    NOTE: this is the WEAK invariant -- simulate_tiering reconcatenates the full matrix before
    aggregating, so it cannot really fail. The GENUINE slab-path test (a real resident+streamed
    edge split that sums partial results) is test_slab_path_partial_sum_equals_single_device."""
    src, dst, N = _toy_graph()
    F = 32
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, F))
    agg = Aggregator(src, dst, N)

    reference = agg.aggregate(X, layers=2)                       # single-device reference

    # split the SAME rows into resident (first n_res) + streamed (rest), then re-run via the
    # simulate_tiering path -- the executor never reorders the math, only the storage tier.
    for n_res in (0, 1, 73, N - 1, N):
        resident = X[:n_res]
        streamed = X[n_res:]
        tiered = simulate_tiering(resident, streamed, agg, layers=2)
        assert tiered.shape == reference.shape
        # EXACT (same operations, same order) -> identical bit pattern up to fp summation order,
        # which is unchanged here because we concatenate in the SAME order.
        assert np.array_equal(tiered, reference), f"tiering changed the result at n_res={n_res}"


def _slab_aggregate(src, dst, N, X, n_res, layers=2):
    """A GENUINE slab/tiled aggregator: reproduces Aggregator.aggregate WITHOUT ever materializing
    the doubled edge list in one shot. Each layer's neighbor-sum is built by streaming the directed
    edges in TWO tiles -- a RESIDENT tile (the first n_res nodes' outgoing contributions, held in
    HBM) and a STREAMED tile (the rest, staged from CPU) -- accumulating PARTIAL sums into the same
    output buffer, then applying the degree normalization + relu AFTER both partial sums are summed.

    This is a real resident+streamed split that SUMS partial results: it COULD fail if the tiling
    logic double-counted an edge, dropped a tile, normalized before summing, or split the symmetric
    (u,v)+(v,u) edges incorrectly. We build the SAME symmetric directed edge set the Aggregator uses
    (i = [src,dst], j = [dst,src]) and partition the SOURCE node j into resident vs streamed."""
    src = np.asarray(src, np.int64); dst = np.asarray(dst, np.int64)
    i = np.concatenate([src, dst]).astype(np.int64)              # target (same as Aggregator.i)
    j = np.concatenate([dst, src]).astype(np.int64)              # source (same as Aggregator.j)
    deg = np.bincount(i, minlength=N).astype(np.float64)
    inv_deg = 1.0 / np.clip(deg, 1.0, None)

    # partition the DIRECTED edges by whether their SOURCE node is resident (<n_res) or streamed.
    resident_edge = j < n_res
    tiles = [(i[resident_edge], j[resident_edge]),               # RESIDENT slab (in HBM)
             (i[~resident_edge], j[~resident_edge])]             # STREAMED slab (staged from CPU)

    h = np.asarray(X, dtype=np.float64)
    for _ in range(layers):
        partial = np.zeros_like(h)                               # accumulate partial sums per tile
        for ti, tj in tiles:
            np.add.at(partial, ti, h[tj])                        # this tile's contribution only
        h = np.maximum(partial * inv_deg[:, None], 0.0)          # normalize+relu AFTER summing tiles
    return h


def test_slab_path_partial_sum_equals_single_device():
    """GENUINE slab-path same-result test (the §44/KERNEL-AUDIT TODO): aggregate the graph tile-by-
    tile (a real resident+streamed edge split that SUMS partial results) and compare to the single-
    device reference. Unlike simulate_tiering (which reconcatenates the full matrix), this path
    independently re-derives the aggregation from summed partials, so it WOULD FAIL if the tiering /
    partial-sum logic were wrong. Equality is within fp tolerance (the partial-sum REORDER is fp64
    here -> ~machine-eps; allow a small tol, not bit-exact)."""
    src, dst, N = _toy_graph()
    F = 32
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N, F))
    reference = Aggregator(src, dst, N).aggregate(X, layers=2)

    for n_res in (0, 1, 73, N - 1, N):
        slab = _slab_aggregate(src, dst, N, X, n_res, layers=2)
        assert slab.shape == reference.shape
        err = float(np.abs(slab - reference).max())
        # same operations, fp64 partial-sum reorder -> within associativity floor (NOT bit-identical).
        assert err < 1e-9, f"slab-path tiering changed the result at n_res={n_res}: max|err|={err:.2e}"


def test_slab_path_detects_broken_tiling():
    """SANITY: the slab-path test is NOT tautological -- if the tiling DROPS the streamed tile (a
    realistic bug: forgetting to stream the cold slab), the result MUST diverge from the reference.
    This proves test_slab_path_partial_sum_equals_single_device could actually fail."""
    src, dst, N = _toy_graph()
    F = 16
    rng = np.random.default_rng(2)
    X = rng.standard_normal((N, F))
    reference = Aggregator(src, dst, N).aggregate(X, layers=2)

    # a BROKEN slab path: keep ONLY the resident tile, never sum the streamed tile (n_res in the
    # interior so the streamed tile is non-empty and load-bearing).
    i = np.concatenate([src, dst]).astype(np.int64)
    j = np.concatenate([dst, src]).astype(np.int64)
    deg = np.bincount(i, minlength=N).astype(np.float64)
    inv_deg = 1.0 / np.clip(deg, 1.0, None)
    n_res = N // 2
    mask = j < n_res                                             # resident tile ONLY (bug: drop streamed)
    h = np.asarray(X, dtype=np.float64)
    for _ in range(2):
        partial = np.zeros_like(h)
        np.add.at(partial, i[mask], h[j[mask]])
        h = np.maximum(partial * inv_deg[:, None], 0.0)
    assert np.abs(h - reference).max() > 1e-6, ("dropping the streamed tile should change the "
                                                "result -> the slab-path test is genuine, not a no-op")


def test_budget_from_plan_reconstructs_feature_set():
    """budget_from_plan extracts a resident/streamed split from a plan_memory result; resident +
    streamed must account for the device's whole feature mass (no rows lost/double-counted)."""
    cluster = from_spec(hbm_gb=[80.0], agg_bw_gbps=[942.0], interconnect_gbps=325.0, h2d_gbps=57.5)
    # a window that exceeds one GPU -> the planner tiers (some snapshots streamed).
    w = Workload(num_nodes=3_000_000, num_edges=24_000_000, feat_dim=512, window=16)
    plan = plan_memory(cluster, w)
    b = budget_from_plan(plan, w, device=0, h2d_gbps=cluster.devices[0].h2d_gbps)
    assert b.resident_units + b.streamed_units == w.window      # all snapshots accounted for
    assert b.streamed_units > 0                                 # pressure forced tiering
    assert b.resident_bytes <= b.capacity_bytes                 # resident fits HBM (no-OOM)
    assert b.streamed_bytes > 0


def test_fits_budget_no_spill():
    """When the working set fits, the budget has zero streamed units (degenerate single placement)."""
    cluster = from_spec(hbm_gb=[80.0], agg_bw_gbps=[942.0], interconnect_gbps=325.0, h2d_gbps=57.5)
    w = Workload(num_nodes=500_000, num_edges=4_000_000, feat_dim=128, window=2)
    plan = plan_memory(cluster, w)
    b = budget_from_plan(plan, w, device=0, h2d_gbps=cluster.devices[0].h2d_gbps)
    assert b.streamed_units == 0 and b.streamed_bytes == 0
    assert b.fits


# ----------------------------------------------------------------------------- #
# 2. WATERMARK policy: reactive vs proactive from synthetic fill/evict rates      #
# ----------------------------------------------------------------------------- #
def test_watermark_reactive_when_evict_ge_fill():
    """§42: REACTIVE viable iff evict_BW/fill_BW >= 1 (we drain at least as fast as we fill).
    The measured H100/CPU-PCIe case (fill 47.7, evict 50.1 -> 1.05x) is BARELY reactive."""
    cap = int(80 * GB)
    incr = int(3.81 * GB)
    pol = policy_from_rates(cap, incr, fill_gbps=47.7, evict_gbps=50.1, tier=SpillTier.CPU)
    assert pol.viability_ratio == pytest.approx(50.1 / 47.7, rel=1e-9)
    assert pol.mode is OverflowMode.REACTIVE
    assert pol.reactive_viable
    # watermarks ordered + safety ceiling below 100%
    assert pol.low_watermark_bytes < pol.high_watermark_bytes < cap
    assert pol.usable_ceiling_bytes < cap


def test_watermark_proactive_when_fill_exceeds_evict():
    """Fill faster than evict (e.g. lighter compute per byte -> faster fill, or slower PCIe) flips
    the policy to PROACTIVE: must reserve headroom upfront / admission-control."""
    cap = int(80 * GB)
    incr = int(3.81 * GB)
    pol = policy_from_rates(cap, incr, fill_gbps=120.0, evict_gbps=50.0, tier=SpillTier.CPU)
    assert pol.viability_ratio < 1.0
    assert pol.mode is OverflowMode.PROACTIVE
    assert not pol.reactive_viable


def test_single_oversized_alloc_is_instant_oom():
    """§42 HARD LIMIT: a SINGLE alloc bigger than current free HBM has ZERO reaction window. Under a
    PROACTIVE policy it must be REJECTED; under a REACTIVE policy it is admitted only after spilling
    enough to make room (incremental granularity + between-alloc check is the only fix)."""
    cap = int(10 * GB)
    incr = int(4 * GB)
    # proactive policy: fill >> evict -> a too-big alloc is rejected outright
    proactive = policy_from_rates(cap, incr, fill_gbps=200.0, evict_gbps=50.0)
    resident = int(8 * GB)                         # only 2GB free
    res = proactive.admit_allocation(resident_bytes=resident, alloc_bytes=int(4 * GB))
    assert not res.admit                           # 4GB alloc into 2GB free, can't drain -> reject
    assert res.must_spill_bytes > 0

    # reactive policy: same situation but we CAN drain in the window -> admitted after spilling
    reactive = policy_from_rates(cap, incr, fill_gbps=47.7, evict_gbps=50.1)
    res2 = reactive.admit_allocation(resident_bytes=resident, alloc_bytes=int(4 * GB))
    assert res2.admit
    assert res2.must_spill_bytes >= int(4 * GB) - (cap - resident)   # at least the deficit


def test_small_alloc_below_watermark_admitted_no_spill():
    """An increment that lands below the high watermark is admitted with no spill."""
    cap = int(80 * GB)
    incr = int(3.81 * GB)
    pol = policy_from_rates(cap, incr, fill_gbps=47.7, evict_gbps=50.1)
    res = pol.admit_allocation(resident_bytes=int(10 * GB), alloc_bytes=incr)
    assert res.admit and res.must_spill_bytes == 0


def test_buffer_gpu_tier_has_higher_margin_than_cpu():
    """§42 implication: spilling over NVLink to a buffer GPU (~325GB/s) gives evict/fill ~6.8x, a
    comfortable margin vs CPU-PCIe's ~1.05x. score_spill_tier recommends the buffer GPU."""
    s = score_spill_tier(fill_gbps=47.7, nvlink_gbps=325.0, pcie_gbps=50.0)
    assert s["buffer_gpu_margin"] > s["cpu_margin"]
    assert s["buffer_gpu_margin"] == pytest.approx(325.0 / 47.7, rel=1e-9)
    assert s["recommended"] is SpillTier.BUFFER_GPU
