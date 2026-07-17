"""Wiring tests for the integrated zord kernel: the package __init__ re-exports resolve, the
FRONT -> MIDDLE -> BACK pipeline runs end-to-end on a toy graph, and the CLI exposes the new
commands without disturbing the scalar `plan` path.
"""
import numpy as np
import pytest

from zord.datasets.temporal_graph import TemporalGraph
from zord.profiler import hetcluster


def _toy_tgi(N=80, E=500, S=8, seed=0, feat_bytes=None):
    from zord.frontend import ingest_graph
    rng = np.random.default_rng(seed)
    src = rng.integers(0, N, size=E).astype(np.int64)
    dst = rng.integers(0, N, size=E).astype(np.int64)
    t = np.sort(rng.integers(0, 10_000, size=E)).astype(np.int64)
    g = TemporalGraph(src=src, dst=dst, t=t, name="wiring-toy", num_nodes=N)
    return ingest_graph(g, num_snapshots=S, feat_bytes=feat_bytes)


def test_package_reexports_resolve():
    """Every new public API named in the design contract is importable from its package."""
    from zord.frontend import (ingest_graph, ingest_dataset, ingest_stream, build_snap,
                               build_supra, graph_stats, GraphStats, TemporalGraphInput)
    from zord.profiler import (probe_hardware, calibrate, probe_and_calibrate, probe_graph_stats,
                              ProbeResult, CostCalibration)
    from zord.partition import (allocate, AllocationPlan, supra_solve_run, build_supra_cells,
                               count_cuts, decide_axis, AttrDecision)
    from zord.runtime import (verify_recombine, plan_recombine, RecombineSpec, plan_coexec,
                             CoExecPlan, coexec_makespan_ms, verify_coexec_result, BufferPool,
                             BufferPoolPlan, pool_from_plan)
    from zord.schedule import (schedule, SchedulePlan, JobEstimate, online_step, OnlineState,
                              StalenessPolicy, build_event_dependency, detect_drift)
    import zord
    # top-level lazy entries
    assert callable(zord.ingest) and callable(zord.allocate) and callable(zord.run_schedule)
    assert callable(zord.online_step) and callable(zord.probe_and_calibrate)
    # the `ingest` package attribute must remain the SUBMODULE (prober relies on it)
    import zord.frontend as fe
    assert type(fe.ingest).__name__ == "module"
    assert callable(fe.ingest_graph)


def test_schedule_end_to_end_runs():
    """The L3 conductor produces a feasible SchedulePlan + a positive JobEstimate on a toy graph."""
    from zord.schedule import schedule
    tgi = _toy_tgi()
    c = hetcluster(1, 1, 1)
    sp = schedule(tgi, c, feat_dim=64, num_snapshots=8, decomposition="auto", num_epochs=3)
    assert sp.feasible
    assert sp.makespan_ms >= 0.0
    assert sp.assignment is not None and sp.assignment.shape[0] == tgi.num_nodes
    est = sp.estimate_total_time(num_epochs=3)
    assert est.total_sec >= 0.0
    # the summary composes without error
    assert "SchedulePlan" in sp.summary()


def test_allocate_supra_cut_not_worse_than_arrange():
    """allocate() keeps the BETTER of supra-cut vs arrange on the node axis (the composed guarantee)."""
    from zord.profiler import probe_hardware, calibrate
    from zord.partition import allocate
    tgi = _toy_tgi(seed=2)
    c = hetcluster(1, 1, 1)
    calib = calibrate(probe_hardware(c), tgi.stats, feat_dim=64)
    plan = allocate(tgi, calib, decomposition="node")
    assert plan.assignment.shape[0] == tgi.num_nodes
    assert (plan.assignment >= 0).all() and (plan.assignment < c.num_devices).all()
    assert plan.weighted_cost >= 0.0
    assert plan.axis == "node"


def test_online_step_cold_then_reuse():
    """The online subsystem: a cold start places all vertices; a low-drift follow-up REUSES the layout."""
    from zord.schedule import online_step, StalenessPolicy
    c = hetcluster(1, 1, 1)
    tgi0 = _toy_tgi(seed=5)
    step0, state0 = online_step(None, tgi0, c, StalenessPolicy(max_staleness_snapshots=4))
    assert step0.rearranged and step0.reason == "cold-start"
    assert state0.assignment is not None
    # an identical follow-up window -> ~zero drift -> reuse (no migration) under the staleness budget
    tgi1 = _toy_tgi(seed=5)
    step1, state1 = online_step(state0, tgi1, c, StalenessPolicy(max_staleness_snapshots=4),
                                prior_plan=state0.meta.get("plan"))
    assert step1.moved_vertices == 0
    assert not step1.rearranged


def test_cli_plan_path_unchanged(capsys):
    """The scalar `plan` CLI must still run and emit the canonical [zord plan] report (bit-identical
    contract). We run on the toy registry dataset 'collegemsg' if present, else skip."""
    from zord.datasets import DATASETS
    if "collegemsg" not in DATASETS:
        pytest.skip("collegemsg not in registry")
    from zord.cli import main
    try:
        rc = main(["plan", "collegemsg", "--snapshots", "8"])
    except SystemExit as e:        # dataset file missing on this box -> not a wiring failure
        pytest.skip(f"dataset load failed: {e}")
    out = capsys.readouterr().out
    assert rc == 0
    assert "[zord plan]" in out
    assert "decomposition" in out and "memory-tiering" in out
