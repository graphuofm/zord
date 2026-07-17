"""ZORD: heterogeneity-aware batch partition-and-place engine + runtime for
dynamic (snapshot) temporal-graph training on heterogeneous GPU clusters.

Core philosophy (see ZORD_VISION.txt):
  - BATCH, not streaming (process a batch of evolving snapshots jointly).
  - INCREMENTAL / low-recompute (reuse prior partition, pay only the delta).
  - HETEROGENEITY-aware (size partitions to each unequal GPU; never OOM the
    smallest card while the biggest idles).
  - GUARANTEED to complete (G1 no-OOM, G2 bounded makespan, G3 no-hang,
    G4 convergence, G5 pre-flight predictability).
"""
from .version import __version__

# Lightweight, dependency-free re-exports. Heavy submodules (torch/profiler)
# are imported lazily by the caller to keep `import zord` cheap and robust.
from .datasets.temporal_graph import TemporalGraph, Snapshot
from .datasets.registry import DATASETS, get_spec

# The kernel pipeline entry points (FRONT -> MIDDLE -> BACK) are exposed LAZILY via
# PEP-562 __getattr__ so `import zord` stays cheap (they pull numpy + the sub-planners
# only on first access). `zord.ingest`, `zord.schedule`, `zord.allocate`,
# `zord.probe_and_calibrate`, `zord.online_step` resolve to the proven module entries.
_LAZY_EXPORTS = {
    # FRONT-END (K1): ingest a temporal graph -> TemporalGraphInput
    "ingest": ("frontend.ingest", "ingest"),
    "ingest_dataset": ("frontend.ingest", "ingest_dataset"),
    "TemporalGraphInput": ("frontend.ingest", "TemporalGraphInput"),
    # PROFILER (L2): probe hardware + graph -> CostCalibration
    "probe_and_calibrate": ("profiler.prober", "probe_and_calibrate"),
    # MIDDLE-END (K2): the composed allocation
    "allocate": ("partition.allocate", "allocate"),
    # L3 CONDUCTOR + online subsystem. NOTE: the conductor function is reached as
    # `zord.schedule.schedule` (the `schedule` subpackage shadows any top-level lazy name),
    # so we expose the SchedulePlan/online entries here (the `schedule` callable lives in the
    # subpackage). We re-export it under the unambiguous name `run_schedule` for top-level use.
    "run_schedule": ("schedule.scheduler", "schedule"),
    "SchedulePlan": ("schedule.scheduler", "SchedulePlan"),
    "online_step": ("schedule.dynamic_online", "online_step"),
}

__all__ = [
    "__version__",
    "TemporalGraph",
    "Snapshot",
    "DATASETS",
    "get_spec",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name):
    """PEP-562 lazy resolution of the kernel pipeline entry points (keeps `import zord` cheap)."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f".{target[0]}", __name__)
    return getattr(mod, target[1])


def __dir__():
    return sorted(__all__)
