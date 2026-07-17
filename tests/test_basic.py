"""Basic self-checks. Run: PYTHONPATH=src python -m pytest tests/ -q"""
import numpy as np

import zord
from zord.datasets import TemporalGraph
from zord.profiler import hetcluster
from zord.frontend import Intent, resolve
from zord.partition import ZordPartitioner, PARTITIONERS


def _toy():
    # 6 nodes, 8 timestamped edges
    src = np.array([0, 1, 2, 3, 4, 5, 0, 2])
    dst = np.array([1, 2, 3, 4, 5, 0, 2, 4])
    t = np.array([10, 20, 20, 30, 40, 40, 50, 60])
    return TemporalGraph(src=src, dst=dst, t=t, name="toy")


def test_import():
    assert zord.__version__
    assert len(zord.DATASETS) >= 10


def test_snapshot_conservation():
    g = _toy()
    snaps = g.to_snapshots(num_snapshots=4)
    assert sum(s.num_edges for s in snaps) == g.num_edges


def test_partition_assigns_all_nodes():
    g = _toy()
    c = hetcluster(1, 1, 1)
    for name, P in PARTITIONERS.items():
        if name == "metis":
            try:
                import pymetis  # noqa: F401
            except Exception:
                continue        # optional dep; tested on the cluster
        p = P().partition(g.src, g.dst, g.num_nodes, c)
        assert (p.assignment >= 0).all()
        assert p.nodes_per_device.sum() == g.num_nodes, name


def test_incremental_reuse():
    g = _toy()
    c = hetcluster(1, 1, 1)
    zp = ZordPartitioner()
    p1 = zp.partition(g.src, g.dst, g.num_nodes, c)
    p2 = zp.partition(g.src, g.dst, g.num_nodes, c, prior=p1)
    assert (p1.assignment == p2.assignment).all()


def test_plan_and_preflight():
    g = _toy()
    c = hetcluster(1, 1, 1)
    plan = resolve(g.src, g.dst, g.num_nodes, c, intent=Intent.MIN_TIME)
    assert plan.preflight.feasible          # toy graph trivially fits
    assert plan.preflight.makespan_sec >= 0
