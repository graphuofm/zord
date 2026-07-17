"""Tests for the global memory scheduler (zord core, post-D25)."""
from zord.profiler.cluster_profile import from_spec, hetcluster
from zord.schedule import Workload, plan_memory
from zord.schedule.planner import (ALLOC_OVERHEAD, FWD_TRANSIENT_COPIES,
                                   _runtime_overhead_bytes, _snapshot_state_bytes)

GB = 1024 ** 3


def _cluster():
    return hetcluster(num_h100=1, num_6000ada=1, num_5000ada=1)


def test_fits_is_all_resident():
    """Small window that fits -> degenerates to single in-HBM placement, no streaming."""
    p = plan_memory(_cluster(), Workload(2_000_000, 16_000_000, feat_dim=128, window=2))
    assert p.all_feasible
    assert all(d.streamed_snapshots == 0 for d in p.per_device)
    assert p.bound == "compute"
    assert p.total_streamed_gb == 0.0


def test_pressure_triggers_tiering_and_stays_feasible():
    """Large window exceeds HBM -> some snapshots stream from CPU, but every device still fits."""
    p = plan_memory(_cluster(), Workload(4_000_000, 40_000_000, feat_dim=256, window=16))
    assert p.all_feasible                                   # no-OOM guarantee holds via tiering
    assert any(d.streamed_snapshots > 0 for d in p.per_device)
    for d in p.per_device:
        assert d.peak_hbm_bytes <= d.capacity_bytes         # provably fits
        assert d.resident_snapshots >= 1


def test_bandwidth_proportional_balance_equalizes_compute():
    """Work split proportional to achieved HBM bandwidth -> heterogeneous GPUs finish together."""
    p = plan_memory(_cluster(), Workload(2_000_000, 16_000_000, feat_dim=128, window=2))
    comps = [d.compute_sec for d in p.per_device]
    assert max(comps) / min(comps) < 1.05                   # near-equal makespan, not equal nodes
    nodes = [d.work_nodes for d in p.per_device]
    assert nodes[0] > nodes[2]                              # H100 gets MORE work than RTX5000


def test_reuse_reduces_epoch_time():
    base = plan_memory(_cluster(), Workload(4_000_000, 40_000_000, feat_dim=256, window=16))
    reuse = plan_memory(_cluster(), Workload(4_000_000, 40_000_000, feat_dim=256, window=16,
                                             reuse_frac=0.7))
    assert reuse.makespan_sec < base.makespan_sec * 0.5     # 70% reuse -> well under half the time


def test_prefetch_hides_staging():
    w = Workload(4_000_000, 40_000_000, feat_dim=256, window=16)
    off = plan_memory(_cluster(), w, prefetch=False)
    on = plan_memory(_cluster(), w, prefetch=True)
    assert on.makespan_sec <= off.makespan_sec              # overlap never hurts


def test_infeasible_when_single_snapshot_too_big():
    """A single snapshot bigger than the smallest HBM -> flagged infeasible (needs node tiling)."""
    p = plan_memory(_cluster(), Workload(60_000_000, 600_000_000, feat_dim=1024, window=1))
    assert not p.all_feasible
    assert any(not d.feasible for d in p.per_device)


# ---- the §40-engine-v2 under-prediction fix: predicted_peak must be a CONSERVATIVE UPPER bound ----
# The old formula was peak = (resident + reserve_buffers) * per_snap, which omitted (a) the SECOND
# double-buffer device feature copy held in flight while streaming and (b) the forward-pass
# working-set transients. On the 78903 uniform run that under-predicted 52.9GB vs a measured 67.6GB
# (27.8% UNSAFE). These tests assert those terms are now included; each FAILS under the old formula.

def test_peak_includes_double_buffer_and_activation_transients_when_streaming():
    """When the window must stream, predicted_peak must EXCEED the bare resident-snapshot state by
    at least the double-buffer feature copy + the forward-pass transients. This is the exact term the
    OLD (resident+reserve)*per_snap formula omitted, so the `> old_peak` assertion FAILS under it.

    Config: the REAL 78903 run -- wiki-talk N,E; F=512 W=16 L=2 on a --hbm-gb 60 single GPU."""
    cluster = from_spec(hbm_gb=[60.0], agg_bw_gbps=[942.0], interconnect_gbps=325.0,
                        h2d_gbps=57.5, names=["GPU-60GB"])
    w = Workload(1_140_149, 7_833_140, feat_dim=512, layers=2, window=16)
    p = plan_memory(cluster, w).per_device[0]
    assert p.streamed_snapshots > 0                                  # this config DOES tier+stream

    n_k = p.work_nodes
    per_snap = _snapshot_state_bytes(n_k, p.work_edges, w)
    feat_one_snap = n_k * w.feat_dim * w.bytes_per_feat
    # the OLD formula's value for the SAME shipped resident count (NO runtime overhead at all)
    old_peak = (p.resident_snapshots + 1) * per_snap
    overhead = _runtime_overhead_bytes(feat_one_snap, streaming=True)

    # the runtime overhead is real: 2nd double-buffer feature copy + FWD transients (>= 4 feat copies)
    assert overhead >= (1 + FWD_TRANSIENT_COPIES) * feat_one_snap
    # predicted_peak must now EXCEED what the old under-counting formula produced for this resident set
    assert p.peak_hbm_bytes > old_peak
    # ... by AT LEAST the omitted double-buffer + transient overhead (the conservative add) ...
    assert p.peak_hbm_bytes >= p.resident_snapshots * per_snap + overhead
    # ... yet still provably fit the device (the no-OOM feasibility guarantee, C1).
    assert p.peak_hbm_bytes <= p.capacity_bytes
    assert p.feasible


def test_peak_is_conservative_upper_bound_on_measured_78903():
    """End-to-end: re-cost the REAL 78903 uniform run (wiki-talk, F=512 W=16 L=2, --hbm-gb 60)
    through plan_memory. The OLD formula shipped resident=7 @ predicted 52.9GB, but the executor of
    THAT plan measured 67.6GB on the card (27.8% UNDER -> UNSAFE). The fix must ship a plan whose
    predicted peak is a CONSERVATIVE upper bound on the bytes the executor of that plan will hold,
    AND whose conservative peak fits the modeled cap (feasibility verdict never under-calls memory)."""
    cluster = from_spec(hbm_gb=[60.0], agg_bw_gbps=[942.0], interconnect_gbps=325.0,
                        h2d_gbps=57.5, names=["GPU-60GB"])
    w = Workload(1_140_149, 7_833_140, feat_dim=512, layers=2, window=16)
    p = plan_memory(cluster, w).per_device[0]
    n_k = p.work_nodes
    per_snap = _snapshot_state_bytes(n_k, p.work_edges, w)
    feat_one_snap = n_k * w.feat_dim * w.bytes_per_feat
    # (i) the SHIPPED plan: its predicted peak must dominate the bytes its executor will hold
    #     (resident snapshots + double-buffer + transients, inflated by the allocator margin) ...
    executor_bytes = (1.0 + ALLOC_OVERHEAD) * (
        p.resident_snapshots * per_snap
        + _runtime_overhead_bytes(feat_one_snap, streaming=p.streamed_snapshots > 0))
    assert p.peak_hbm_bytes >= executor_bytes            # predicted >= what the executor will hold
    assert p.peak_hbm_bytes <= p.capacity_bytes          # ... and the conservative peak fits the cap.
    assert p.feasible
    # (ii) APPLES-TO-APPLES vs the run: the OLD formula SHIPPED resident=7 @ 52.9GB but the executor of
    #      THAT plan MEASURED 67.6GB. Re-cost resident=7 through the FIXED model -> it must now be a
    #      conservative upper bound on the measured 67.6GB (so the engine would never ship it at cap=60).
    GB_ = 1024 ** 3
    recosted_old_plan = (1.0 + ALLOC_OVERHEAD) * (
        7 * per_snap + _runtime_overhead_bytes(feat_one_snap, streaming=True))
    assert recosted_old_plan / GB_ >= 67.6               # was 52.9 (27.8% UNDER) -> now >= measured
