"""Storage-backend seam (the FlashGraph integration point).

You asked: can FlashGraph be zord's storage layer? This defines exactly what zord
needs FROM storage, so any engine (in-memory numpy today; FlashGraph tomorrow) can
plug in. zord is the COMPUTE/PARTITION layer; the backend is the STORE layer.

What zord needs from a temporal-graph store:
  - load_temporal_graph()      -> a TemporalGraph (edges as SoA arrays)
  - range(t0, t1)              -> the edges in a time window  (= snapshot slicing)
  - ingest(src,dst,t,w)        -> append new edges            (= a new batch arriving)

WHY FlashGraph is a strong fit (analysis, since its code isn't here yet):
  FlashGraph's whole pitch is O(1) temporal POINT/RANGE queries (timestamp ->
  virtual-memory offset) + high-throughput append-only INGEST. Those are *exactly*
  zord's two hot operations: (a) slice a window of snapshots for a batch (range),
  and (b) absorb the next batch of events (ingest). A B+-tree store (Teseo/Neo4j,
  FlashGraph's baselines) pays log(N) per access and stalls ingest on rebalancing.
  => FlashGraph maps cleanly onto `range` + `ingest`; recommend it as the
     high-performance backend, with InMemoryBackend as the portable default.
  Caveat: needs FlashGraph's Python bindings; FlashGraphBackend below is the adapter
  stub to fill once that code is available.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ..datasets.temporal_graph import TemporalGraph


@runtime_checkable
class StorageBackend(Protocol):
    def load_temporal_graph(self) -> TemporalGraph: ...
    def range(self, t0: int, t1: int) -> TemporalGraph: ...
    def ingest(self, src, dst, t, w=None) -> None: ...


class InMemoryBackend:
    """Default backend: a TemporalGraph held in RAM (numpy SoA). Range queries via
    binary search on the time-sorted timestamps (what to_snapshots already uses)."""

    def __init__(self, graph: TemporalGraph):
        self.g = graph.sort_by_time()

    def load_temporal_graph(self) -> TemporalGraph:
        return self.g

    def range(self, t0: int, t1: int) -> TemporalGraph:
        lo = int(np.searchsorted(self.g.t, t0, "left"))
        hi = int(np.searchsorted(self.g.t, t1, "left"))
        w = self.g.w[lo:hi] if self.g.w is not None else None
        return TemporalGraph(self.g.src[lo:hi], self.g.dst[lo:hi], self.g.t[lo:hi],
                             w=w, num_nodes=self.g.num_nodes, name=f"{self.g.name}[{t0},{t1})")

    def ingest(self, src, dst, t, w=None) -> None:
        self.g.src = np.concatenate([self.g.src, np.asarray(src, np.int64)])
        self.g.dst = np.concatenate([self.g.dst, np.asarray(dst, np.int64)])
        self.g.t = np.concatenate([self.g.t, np.asarray(t, np.int64)])
        self.g._sorted = False
        self.g.sort_by_time()


class FlashGraphBackend:
    """Adapter to FlashGraph (Ding & X. Zhang) -- the index-free, virtual-memory
    temporal store. STUB: wire to FlashGraph's bindings when available. Its
    range()/append() are the O(1) ops zord wants; this class just translates
    between zord's TemporalGraph and FlashGraph's edge layout.
    """
    def __init__(self, handle=None):
        self.handle = handle
        if handle is None:
            raise NotImplementedError(
                "FlashGraphBackend needs FlashGraph's Python bindings. "
                "When available, map: range()->FlashGraph temporal range query, "
                "ingest()->FlashGraph arena append, load_temporal_graph()->bulk scan.")

    def load_temporal_graph(self) -> TemporalGraph: ...
    def range(self, t0: int, t1: int) -> TemporalGraph: ...
    def ingest(self, src, dst, t, w=None) -> None: ...
