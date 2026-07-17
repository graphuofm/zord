"""Profiler: the hardware PARAMETER (ClusterProfile) + the auto-prober/calibration (L2).

`cluster_profile` is dependency-free and imported eagerly. The auto-prober (`prober`)
imports ``partition.cost_model`` for its ``CostParams`` output, and ``partition`` in turn
imports ``profiler.cluster_profile`` -- a benign import cycle IF ``prober`` were imported
eagerly at package load. To keep the cycle from biting (``import zord.partition`` triggers
``profiler.__init__`` while ``partition`` is still initializing), the prober symbols are
exposed LAZILY via PEP-562 ``__getattr__``: they resolve on first attribute access, by which
point ``partition`` has finished initializing. ``from zord.profiler import calibrate`` etc.
work exactly as before.
"""
from .cluster_profile import DeviceProfile, ClusterProfile, hetcluster, from_spec, GB

# prober (L2 auto-prober + calibration) -- resolved lazily to avoid the partition<->profiler
# import cycle at package-init time. Names exposed by __getattr__ below.
_PROBER_EXPORTS = (
    "probe_hardware", "calibrate", "probe_and_calibrate", "probe_graph_stats",
    "ProbeResult", "CostCalibration",
    "measure_hbm_bw_gbps", "measure_link_gbps",
)

__all__ = [
    # cluster profile (the hardware PARAMETER)
    "DeviceProfile", "ClusterProfile", "hetcluster", "from_spec", "GB",
    # auto-prober + calibration (L2: env -> cost-model parameters, no hand-set weight)
    *_PROBER_EXPORTS,
]


def __getattr__(name):
    """PEP-562 lazy attribute resolution for the prober exports (breaks the import cycle)."""
    if name in _PROBER_EXPORTS:
        from . import prober
        return getattr(prober, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
