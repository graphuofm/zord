"""Partition data model + the Partitioner interface.

A Partition assigns every node of a snapshot-batch to a device. We track the
counts and the cross-device edge traffic, which the cost model turns into a
memory footprint (G1) and a makespan estimate (G2).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..profiler.cluster_profile import ClusterProfile


@dataclass
class Partition:
    assignment: np.ndarray         # int32 [num_nodes] -> device id
    num_devices: int
    nodes_per_device: np.ndarray   # int [num_devices]
    edges_per_device: np.ndarray   # int [num_devices] (local edges)
    cross_edges: np.ndarray        # int [num_devices][num_devices] traffic
    halo_per_device: Optional[np.ndarray] = None  # boundary/halo node counts
    meta: dict = field(default_factory=dict)

    @property
    def total_cross_edges(self) -> int:
        m = self.cross_edges.copy()
        np.fill_diagonal(m, 0)
        return int(m.sum())

    def imbalance(self) -> float:
        """max/mean of local edges -> 1.0 is perfectly balanced (by count)."""
        e = self.edges_per_device.astype(float)
        return float(e.max() / e.mean()) if e.mean() > 0 else 1.0


class Partitioner(ABC):
    name = "abstract"

    @abstractmethod
    def partition(self, src: np.ndarray, dst: np.ndarray, num_nodes: int,
                  cluster: ClusterProfile,
                  prior: Optional[Partition] = None) -> Partition:
        """Assign nodes to devices. `prior` enables INCREMENTAL re-partition
        (reuse last batch's assignment, pay only for the delta -- R5)."""
        raise NotImplementedError

    # shared helper: build a Partition from an assignment vector
    @staticmethod
    def _summarize(assignment: np.ndarray, src: np.ndarray, dst: np.ndarray,
                   num_devices: int) -> Partition:
        nodes_per = np.bincount(assignment, minlength=num_devices)
        a_src, a_dst = assignment[src], assignment[dst]
        cross = np.zeros((num_devices, num_devices), dtype=np.int64)
        np.add.at(cross, (a_src, a_dst), 1)
        local = np.diag(cross).copy()
        return Partition(assignment=assignment, num_devices=num_devices,
                         nodes_per_device=nodes_per, edges_per_device=local,
                         cross_edges=cross)
