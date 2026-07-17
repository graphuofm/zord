"""ZordPartitioner -- heterogeneity-aware, bandwidth-weighted, INCREMENTAL
streaming partitioner (Fennel-family, adapted for heterogeneous GPUs).

For each node v (processed in temporal first-appearance / streaming order), pick
the device d that minimizes:

    score(v, d) = cut_cost(v, d)  +  alpha * (load[d] / capacity[d])

where
  cut_cost(v, d) = sum over already-placed neighbors u of  linkcost[d, dev[u]]
                 = (linkcost @ neighbor_device_histogram)[d]
  linkcost[i, j] = 1 / bandwidth(i, j)   (0 on the diagonal; cutting across a
                   slow inter-node link costs MUCH more than across NVLink)
  capacity[d]    = num_nodes * (usable_mem[d] / total_usable_mem)
                   -> sized by MEASURED MEMORY, not compute (2026-05-30 finding:
                      compute is nearly flat; memory + bandwidth are what bind).
A hard capacity cap keeps every device feasible (G1). Incremental: a prior
assignment is reused for nodes that still exist; only new nodes are placed (R5).

NOTE: pure-Python streaming loop -> fine for up to ~1-2M nodes; for ultra graphs
a numba/C kernel is future work (the cnt/dot math is already vectorized per node).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Partitioner, Partition
from ..profiler.cluster_profile import ClusterProfile


def _build_csr(src, dst, num_nodes):
    """Undirected CSR adjacency for O(deg) neighbor lookup."""
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.add.at(indptr, u + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, v.astype(np.int64)


def _appearance_order(src, dst, num_nodes):
    """Nodes in order of first appearance in the (time-sorted) edge stream."""
    both = np.empty(2 * src.shape[0], dtype=np.int64)
    both[0::2] = src
    both[1::2] = dst
    first = np.full(num_nodes, both.shape[0], dtype=np.int64)
    np.minimum.at(first, both, np.arange(both.shape[0]))
    return np.argsort(first, kind="stable")


class ZordPartitioner(Partitioner):
    name = "zord"

    def __init__(self, alpha: float = 1.0, refine_passes: int = 2,
                 order: str = "auto", migration_budget: float = 0.0):
        self.alpha = alpha            # load-balance weight (high=balance, low=consolidate)
        self.refine_passes = refine_passes
        # seed order, REGIME-DEPENDENT (measured D17): "appearance" (temporal) wins when
        # CONSOLIDATING (a big card holds ~everything); "degree" (hubs first) wins when
        # forced to SPREAD under memory pressure. "auto" picks per cap vs num_nodes.
        self.order = order            # "auto" | "appearance" | "degree"
        # incremental re-balance knob: also re-refine up to this fraction of nodes
        # (the worst stale boundary nodes) per re-partition -> trades a little migration
        # for lower cuts. 0.0 = pure incremental (no old-node movement); 1.0 ~ full refine.
        self.migration_budget = migration_budget

    def partition(self, src, dst, num_nodes, cluster: ClusterProfile,
                  prior: Optional[Partition] = None,
                  capacity: Optional[np.ndarray] = None) -> Partition:
        P = cluster.num_devices
        if capacity is not None:
            # REAL memory limit per device (max #nodes that fit). Gives big cards
            # slack -> consolidate to cut edges; tight pressure -> near-proportional.
            cap = np.minimum(np.asarray(capacity, dtype=np.int64), num_nodes)
            cap = np.maximum(cap, 1)
        else:
            mem = np.array([d.usable_mem for d in cluster.devices], dtype=np.float64)
            cap = np.maximum(1, np.floor(mem / mem.sum() * num_nodes)).astype(np.int64)
        # If the conservative per-device limits can't hold the whole graph, fall
        # back to a MEMORY-PROPORTIONAL split (uses each card's full memory share,
        # as feasible as the hardware allows) -- do NOT dump the overflow onto one
        # card (that OOMs it). Absorb only the rounding remainder on the largest.
        if cap.sum() < num_nodes:
            cap = np.maximum(1, np.floor(cap * (num_nodes / cap.sum()))).astype(np.int64)
            rem = num_nodes - int(cap.sum())
            if rem > 0:
                cap[int(cap.argmax())] += rem

        bw = np.array(cluster.bandwidth, dtype=np.float64)
        with np.errstate(divide="ignore"):
            linkcost = 1.0 / bw
        linkcost[~np.isfinite(linkcost)] = 0.0   # inf bw (same dev) -> 0 cost
        np.fill_diagonal(linkcost, 0.0)

        indptr, adj = _build_csr(src, dst, num_nodes)
        assignment = np.full(num_nodes, -1, dtype=np.int32)
        load = np.zeros(P, dtype=np.int64)

        incremental = prior is not None and prior.assignment is not None
        if incremental:                                           # incremental (R5)
            k = min(num_nodes, prior.assignment.shape[0])
            assignment[:k] = prior.assignment[:k]
            load = np.bincount(assignment[:k][assignment[:k] >= 0], minlength=P).astype(np.int64)
        new_nodes = np.where(assignment < 0)[0]                   # the delta to (re)assign

        order_mode = self.order
        if order_mode == "auto":             # consolidation possible -> appearance; else degree
            order_mode = "appearance" if cap.max() >= num_nodes else "degree"
        if incremental:                      # only the new nodes need assigning;
            order = new_nodes                # appended ids are ~appearance order
        elif order_mode == "degree":         # place hubs first -> better locality when spread
            order = np.argsort(-np.diff(indptr), kind="stable")
        else:
            order = _appearance_order(src, dst, num_nodes)
        alpha = self.alpha
        INF = np.inf
        for v in order:
            if assignment[v] >= 0:
                continue
            nb = adj[indptr[v]:indptr[v + 1]]
            dv = assignment[nb]
            placed = dv[dv >= 0]
            if placed.size:
                cnt = np.bincount(placed, minlength=P).astype(np.float64)
                cut = linkcost.dot(cnt)
            else:
                cut = np.zeros(P)
            score = cut + alpha * (load / cap)
            score = np.where(load >= cap, INF, score)
            d = int(score.argmin())
            if not np.isfinite(score[d]):           # every device full -> headroom
                d = int((cap - load).argmax())
            assignment[v] = d
            load[d] += 1

        # --- cut-recovery: bandwidth-aware boundary refinement under capacity ---
        # INCREMENTAL: when re-partitioning a drifted graph, only re-touch the
        # CHANGED region (new nodes + their 1-hop neighbors) instead of all nodes
        # -> refinement cost ~ O(delta), not O(N) (earns the incremental-time claim).
        if incremental:
            if new_nodes.size:
                nbr_lists = [adj[indptr[v]:indptr[v + 1]] for v in new_nodes]
                dirty = np.unique(np.concatenate([new_nodes] + nbr_lists))
            else:
                dirty = new_nodes
            # migration budget: add the B worst STALE boundary nodes (most cross-edges,
            # not already dirty) so cuts don't drift -- bounded migration of old nodes.
            B = int(self.migration_budget * num_nodes)
            if B > 0:
                a_s, a_d = assignment[src], assignment[dst]
                cross = a_s != a_d
                if cross.any():
                    cross_deg = np.bincount(np.concatenate([src[cross], dst[cross]]),
                                            minlength=num_nodes)
                    in_dirty = np.zeros(num_nodes, dtype=bool); in_dirty[dirty] = True
                    cand = np.where((cross_deg > 0) & (~in_dirty))[0]
                    if cand.size > B:
                        cand = cand[np.argpartition(-cross_deg[cand], B)[:B]]
                    dirty = np.concatenate([dirty, cand])
        else:
            dirty = np.arange(num_nodes)
        self._refine(assignment, indptr, adj, load, cap, linkcost, P, dirty)
        return self._summarize(assignment, src, dst, P)

    def _refine(self, assignment, indptr, adj, load, cap, linkcost, P, dirty):
        for _ in range(self.refine_passes):
            moved = 0
            for v in dirty:
                lo, hi = indptr[v], indptr[v + 1]
                if hi == lo:
                    continue
                cnt = np.bincount(assignment[adj[lo:hi]], minlength=P).astype(np.float64)
                cost = linkcost.dot(cnt)            # weighted cut if v on each device
                cur = int(assignment[v])
                allowed = load < cap
                allowed[cur] = True                 # staying is always allowed
                cost = np.where(allowed, cost, np.inf)
                best = int(cost.argmin())
                if best != cur and cost[best] < cost[cur] - 1e-9:
                    load[cur] -= 1; load[best] += 1
                    assignment[v] = best; moved += 1
            if moved == 0:
                break
