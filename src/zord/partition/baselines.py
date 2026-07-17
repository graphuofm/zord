"""Baseline partitioners (the comparison floor from ZORD_VISION):
  - HashPartitioner: node_id % P  (the trivial homogeneous baseline)
  - RandomPartitioner
  - CapacityProportionalHash: hetero-aware floor -- shares sized by device
    throughput so the H100 gets more nodes than the RTX5000. This is the
    simplest thing that respects heterogeneity; zord must beat it on
    cross-edge traffic and on incremental cost.
METIS / Fennel / DistDy / CaPGNN-RAPA are heavier baselines wired in later.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Partitioner, Partition
from ..profiler.cluster_profile import ClusterProfile


class HashPartitioner(Partitioner):
    name = "hash"

    def partition(self, src, dst, num_nodes, cluster: ClusterProfile,
                  prior: Optional[Partition] = None) -> Partition:
        P = cluster.num_devices
        assignment = (np.arange(num_nodes, dtype=np.int64) % P).astype(np.int32)
        return self._summarize(assignment, src, dst, P)


class RandomPartitioner(Partitioner):
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def partition(self, src, dst, num_nodes, cluster: ClusterProfile,
                  prior: Optional[Partition] = None) -> Partition:
        P = cluster.num_devices
        rng = np.random.default_rng(self.seed)
        assignment = rng.integers(0, P, size=num_nodes, dtype=np.int64).astype(np.int32)
        return self._summarize(assignment, src, dst, P)


class FennelPartitioner(Partitioner):
    """Classic (homogeneous) Fennel streaming partition: place each node where it
    has the most already-placed neighbors, penalized by an EQUAL load target.
    Topology-aware but heterogeneity-BLIND and bandwidth-BLIND -- the baseline
    that isolates exactly what zord's memory-capacity + per-link-bandwidth
    weighting adds on top of locality."""
    name = "fennel"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def partition(self, src, dst, num_nodes, cluster: ClusterProfile,
                  prior: Optional[Partition] = None) -> Partition:
        from .hetero import _build_csr, _appearance_order
        P = cluster.num_devices
        cap = max(1, num_nodes // P)
        indptr, adj = _build_csr(src, dst, num_nodes)
        assignment = np.full(num_nodes, -1, dtype=np.int32)
        load = np.zeros(P, dtype=np.float64)
        for v in _appearance_order(src, dst, num_nodes):
            nb = adj[indptr[v]:indptr[v + 1]]
            dv = assignment[nb]
            placed = dv[dv >= 0]
            cnt = np.bincount(placed, minlength=P).astype(np.float64) if placed.size else np.zeros(P)
            # maximize neighbors-in-d minus load penalty == minimize (-cnt + pen)
            score = -cnt + self.alpha * (load / cap)
            d = int(score.argmin())
            assignment[v] = d
            load[d] += 1
        return self._summarize(assignment, src, dst, P)


class MetisPartitioner(Partitioner):
    """METIS (pymetis) -- the gold-standard offline balanced min-cut. By default
    it targets EQUAL parts (heterogeneity-blind), so under memory pressure it OOMs
    the small card just like hash/fennel despite excellent cuts -- the strongest
    'classic partitioner' baseline. Pass tpwgts=memory-shares for a heterogeneous
    variant ('metis-hetero')."""
    name = "metis"

    def __init__(self, memory_weighted: bool = False):
        self.memory_weighted = memory_weighted

    def partition(self, src, dst, num_nodes, cluster: ClusterProfile,
                  prior: Optional[Partition] = None, **_) -> Partition:
        import pymetis
        from .hetero import _build_csr
        m = src != dst                       # METIS dislikes self-loops
        indptr, adj = _build_csr(src[m], dst[m], num_nodes)
        P = cluster.num_devices
        kw = {}
        if self.memory_weighted:
            mem = np.array([d.usable_mem for d in cluster.devices], dtype=float)
            kw["tpwgts"] = list(mem / mem.sum())
        xadj, adjncy = indptr.tolist(), adj.tolist()
        try:
            _, membership = pymetis.part_graph(P, xadj=xadj, adjncy=adjncy, **kw)
        except TypeError:
            # pymetis 2023.1.1 lacks tpwgts -> heterogeneous (memory-weighted) METIS
            # is NOT available; fall back to balanced METIS so we don't crash.
            _, membership = pymetis.part_graph(P, xadj=xadj, adjncy=adjncy)
        return self._summarize(np.asarray(membership, dtype=np.int32), src, dst, P)


class CapacityProportionalHash(Partitioner):
    """Assign node-id ranges to devices in proportion to throughput share, so
    faster/bigger GPUs get more nodes. A hetero-aware *baseline* (still
    topology-blind: cuts a lot of edges). zord beats it by being locality- and
    bandwidth-aware AND incremental."""
    name = "caphash"

    def partition(self, src, dst, num_nodes, cluster: ClusterProfile,
                  prior: Optional[Partition] = None) -> Partition:
        shares = np.asarray(cluster.throughput_shares())
        # device boundaries over [0, num_nodes)
        cuts = np.floor(np.cumsum(shares) * num_nodes).astype(np.int64)
        assignment = np.empty(num_nodes, dtype=np.int32)
        lo = 0
        for dev, hi in enumerate(cuts):
            assignment[lo:hi] = dev
            lo = hi
        assignment[lo:] = cluster.num_devices - 1
        return self._summarize(assignment, src, dst, cluster.num_devices)
