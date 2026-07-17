"""zord FRONT-END package.

Exposes the original lightweight resolver (``intent``) plus the K1 temporal-graph ingest API.

IMPORTANT (import-binding gotcha, mirrors the note in partition/allocate.py): the ``ingest``
SUBMODULE and the ``ingest`` FUNCTION share a name. A sibling module body (profiler.prober) does
``from ..frontend import ingest as _ingest`` expecting the MODULE and then calls ``_ingest.build_snap``.
So we must NOT rebind the package attribute ``ingest`` to the function -- the submodule binding has to
win. We therefore re-export the *function* under the explicit name ``ingest_graph`` (and it is always
reachable as ``zord.frontend.ingest.ingest`` and the top-level ``zord.ingest``), while leaving the
package attribute ``ingest`` pointing at the submodule. The other ingest entry points (which do NOT
collide with a submodule name) are re-exported directly.
"""
from . import ingest  # keep the package attribute `ingest` bound to the SUBMODULE (prober relies on it)
from .intent import Intent, Plan, resolve
from .ingest import (
    ingest as ingest_graph,          # the FRONT-END entry FUNCTION (named to avoid shadowing the submodule)
    ingest_dataset, ingest_stream, build_snap, build_supra,
    graph_stats, GraphStats, TemporalGraphInput,
)

__all__ = [
    # intent (the original lightweight front-end resolver)
    "Intent", "Plan", "resolve",
    # ingest (K1: the temporal-graph front-end entry + the cheap GraphStats probe)
    "ingest",          # the SUBMODULE (zord.frontend.ingest)
    "ingest_graph",    # the entry FUNCTION (== zord.frontend.ingest.ingest)
    "ingest_dataset", "ingest_stream", "build_snap", "build_supra",
    "graph_stats", "GraphStats", "TemporalGraphInput",
]
