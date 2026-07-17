"""zord global memory scheduler (the core, post-D25 pivot).

Plans, at FULL PRECISION, how temporal-GNN training state is laid out across the
heterogeneous memory system -- GPU HBM (capacity + bandwidth) and CPU RAM (staged
over PCIe) -- so that nothing OOMs, heterogeneous GPUs finish together, and
cross-snapshot reuse minimizes recompute. See planner.plan_memory.

The L3 CONDUCTOR (`schedule`, scheduler.py) ties FRONT -> MIDDLE -> BACK into ONE
SchedulePlan; `online_step` (dynamic_online.py) lifts the batch incremental win into a
designed online subsystem (bounded staleness + drift-triggered changed-cone re-arrange).

`planner` / `dynamic` are dependency-light and imported eagerly. `scheduler` and
`dynamic_online` pull the runtime sub-planners (runtime.coexec/bufferpool/feature_recombine),
and the runtime in turn imports ``schedule.planner`` -- a benign cycle that ONLY bites if the
conductor is imported eagerly at package load (``runtime.memtier`` imports ``schedule.planner``,
which would trigger this ``__init__`` and pull ``scheduler`` -> ``runtime`` mid-init). So the
conductor + online-subsystem symbols are exposed LAZILY via PEP-562 ``__getattr__``; they resolve
on first attribute access, by which point ``runtime`` has finished initializing.
``from zord.schedule import schedule`` etc. work exactly as before.
"""
from .planner import (
    Workload, MemoryPlan, GlobalPlan, plan_memory,
    Plan, DevicePlacement, plan,
    DecompositionChoice, choose_decomposition,
)
from .dynamic import plan_incremental, IncrementalPlan, partition_incremental

# scheduler (L3 conductor) + dynamic_online (online subsystem) -- resolved lazily to avoid the
# runtime<->schedule import cycle at package-init time. Names exposed by __getattr__ below.
_SCHEDULER_EXPORTS = ("schedule", "SchedulePlan", "JobEstimate")
_ONLINE_EXPORTS = (
    "online_step", "OnlineState", "OnlineStep", "StalenessPolicy",
    "EventDependencyGraph", "build_event_dependency", "detect_drift",
)

__all__ = [
    "Workload", "MemoryPlan", "GlobalPlan", "plan_memory",
    "Plan", "DevicePlacement", "plan",
    "DecompositionChoice", "choose_decomposition",
    "plan_incremental", "IncrementalPlan", "partition_incremental",
    # L3 conductor (the one coherent FRONT -> MIDDLE -> BACK schedule)
    *_SCHEDULER_EXPORTS,
    # online subsystem (bounded staleness + drift-triggered changed-cone re-arrangement)
    *_ONLINE_EXPORTS,
]


def __getattr__(name):
    """PEP-562 lazy attribute resolution for the conductor / online subsystem (breaks the cycle)."""
    if name in _SCHEDULER_EXPORTS:
        from . import scheduler
        return getattr(scheduler, name)
    if name in _ONLINE_EXPORTS:
        from . import dynamic_online
        return getattr(dynamic_online, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
