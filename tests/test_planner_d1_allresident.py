"""D1 regression: the F_v ALL-RESIDENT (no-spill) predicted peak must be a CONSERVATIVE upper bound
on the MEASURED 70.3 GB (was 63.5 GB, ~10% UNDER -> UNSAFE) -- so the no-OOM C1 guarantee holds on
EVERY path, including the all-resident one. Rebuilds the exact §46-wiki config and asserts
predicted_peak >= 70.3 GB while still fitting the modeled cap. Also asserts the fix does NOT regress
the real-spilling F_v path nor the scalar path.
"""
from zord.profiler.cluster_profile import from_spec
from zord.schedule.planner import (
    Workload, plan_memory,
    test_all_resident_fv_peak_geq_measured as _planner_d1_selftest,
    test_all_resident_fix_does_not_regress_fv_spill_or_scalar as _planner_no_regress_selftest,
)
import numpy as np

GB = 1024 ** 3
MEASURED_GB = 70.3   # §46-wiki real-GPU max_memory_allocated on the all-resident path


def test_all_resident_fv_peak_predicted_geq_703():
    """The §46-wiki all-resident F_v peak: predicted >= measured 70.3 GB, feasible, no real spill."""
    N, E, L, W = 1_140_149, 7_833_140, 2, 1
    cluster = from_spec(hbm_gb=[80.0], agg_bw_gbps=[942.0], interconnect_gbps=325.0,
                        h2d_gbps=57.5, names=["H100-80GB"])
    cap = cluster.devices[0].usable_mem
    sum_fv = (63.5 * GB - E * 12) / (W * (1 + L) * 4.0)       # modeled F_v whose bare bank == 63.5 GB
    Fv = np.full(N, sum_fv / N, dtype=np.float64)
    w = Workload(num_nodes=N, num_edges=E, feat_dim=int(round(sum_fv / N)), window=W, layers=L,
                 reuse_frac=0.0, feat_bytes=Fv, assignment=np.zeros(N, dtype=np.int64))
    mem = plan_memory(cluster, w)
    p = mem.per_device[0]
    assert mem.total_streamed_gb < 0.01                      # genuinely all-resident (no real spill)
    assert p.feasible and p.peak_hbm_bytes <= cap            # provably fits the GPU
    assert p.peak_hbm_bytes / GB >= MEASURED_GB, (
        f"all-resident F_v peak {p.peak_hbm_bytes/GB:.2f}GB UNDER-predicts measured {MEASURED_GB}GB")


def test_planner_d1_module_selftests_pass():
    """Run the planner module's own D1 self-tests (the conservative all-resident peak + no-regression)."""
    _planner_d1_selftest()
    _planner_no_regress_selftest()
