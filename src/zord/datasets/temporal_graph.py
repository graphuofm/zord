"""Canonical in-memory temporal graph + discrete-time snapshotting.

A TemporalGraph is the common format every loader produces and every zord
component consumes. Edges are stored as parallel numpy arrays (SoA layout) so
they are cache-friendly and cheap to slice into snapshots/batches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class Snapshot:
    """One discrete-time snapshot: a contiguous time window [t_start, t_end)
    and the (sorted-by-time) edge index range it covers in the parent graph."""
    index: int
    t_start: int
    t_end: int
    lo: int  # first edge offset (inclusive)
    hi: int  # last edge offset (exclusive)

    @property
    def num_edges(self) -> int:
        return self.hi - self.lo


@dataclass
class TemporalGraph:
    src: np.ndarray            # int64 [E]
    dst: np.ndarray            # int64 [E]
    t: np.ndarray              # int64 [E]  (unix ts or logical time)
    w: Optional[np.ndarray] = None   # float32 [E] optional edge weight
    efeat: Optional[np.ndarray] = None  # float32 [E, Fe] optional edge features (attributed graphs)
    num_nodes: Optional[int] = None
    name: str = "unnamed"
    _sorted: bool = False

    def __post_init__(self) -> None:
        self.src = np.asarray(self.src, dtype=np.int64)
        self.dst = np.asarray(self.dst, dtype=np.int64)
        self.t = np.asarray(self.t, dtype=np.int64)
        if self.w is not None:
            self.w = np.asarray(self.w, dtype=np.float32)
        if self.efeat is not None:
            self.efeat = np.asarray(self.efeat, dtype=np.float32)
            if self.efeat.ndim == 1:
                self.efeat = self.efeat.reshape(-1, 1)  # [E] -> [E, 1]
        if self.num_nodes is None:
            self.num_nodes = int(max(self.src.max(initial=-1),
                                     self.dst.max(initial=-1)) + 1)

    # -- basic stats ---------------------------------------------------------
    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    @property
    def tmin(self) -> int:
        return int(self.t.min()) if self.num_edges else 0

    @property
    def tmax(self) -> int:
        return int(self.t.max()) if self.num_edges else 0

    @property
    def span(self) -> int:
        return self.tmax - self.tmin

    def sort_by_time(self) -> "TemporalGraph":
        """Sort edges chronologically in place (needed for snapshotting)."""
        if self._sorted:
            return self
        order = np.argsort(self.t, kind="stable")
        self.src, self.dst, self.t = self.src[order], self.dst[order], self.t[order]
        if self.w is not None:
            self.w = self.w[order]
        if self.efeat is not None:
            self.efeat = self.efeat[order]  # reorder edge features alongside src/dst/t/w
        self._sorted = True
        return self

    # -- discrete-time batching (the BATCH, not streaming, model) ------------
    def to_snapshots(self, num_snapshots: Optional[int] = None,
                     interval: Optional[int] = None) -> list[Snapshot]:
        """Cut the timeline into discrete snapshots. Provide either a fixed
        number of equal-time-width snapshots, or a fixed time `interval`.
        Returns Snapshot descriptors (no data copy; just index ranges)."""
        if self.num_edges == 0:
            return []
        self.sort_by_time()
        lo_t, hi_t = self.tmin, self.tmax + 1
        if interval is None:
            if num_snapshots is None:
                num_snapshots = 32
            interval = max(1, (hi_t - lo_t + num_snapshots - 1) // num_snapshots)
        edges = []
        boundaries = np.arange(lo_t, hi_t + interval, interval)
        # searchsorted on the sorted timestamps gives O(log E) slicing
        offs = np.searchsorted(self.t, boundaries, side="left")
        idx = 0
        for i in range(len(boundaries) - 1):
            lo, hi = int(offs[i]), int(offs[i + 1])
            if hi <= lo:
                continue  # empty snapshot -> skip (sparse timeline)
            edges.append(Snapshot(index=idx, t_start=int(boundaries[i]),
                                  t_end=int(boundaries[i + 1]), lo=lo, hi=hi))
            idx += 1
        return edges

    def batches(self, snapshots: list[Snapshot], window: int,
                stride: Optional[int] = None) -> Iterator[list[Snapshot]]:
        """Group snapshots into overlapping training windows (a 'batch' of
        snapshots processed jointly). stride defaults to `window` (disjoint)."""
        stride = stride or window
        for start in range(0, max(1, len(snapshots) - window + 1), stride):
            yield snapshots[start:start + window]

    def summary(self) -> dict:
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "span": self.span,
            "has_weight": self.w is not None,
            "edge_feat_dim": int(self.efeat.shape[1]) if self.efeat is not None else 0,
        }
