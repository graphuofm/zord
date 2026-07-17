"""DYNAMIC incremental adaptation -- the temporal win from scripts/dynamic_run.py, ported
into the engine (pure numpy; the planner is GPU-free).

When the temporal graph evolves to a new batch, re-partitioning from scratch churns labels
and forces a huge STATE MIGRATION: every vertex that changes device must physically move its
node-memory vector across the interconnect. zord instead REUSES the prior assignment and only
(re)places the CHANGED CONE -- new vertices plus the endpoints of new edges -- under a
MIGRATION BUDGET (at most budget*N old vertices may move), via a vectorized changed-cone label
propagation. This stays balanced (unlike static, whose cut/imbalance degrade as the graph
grows) WHILE migrating little state (unlike from-scratch). The validated result: ~1.56x lower
dynamic total wall-clock vs static and 4-8x less node-memory migrated vs from-scratch.

THE DYNAMIC COST (the thing a static problem never pays):
    state_migration_sec = (#vertices that changed device) * mem_dim * 4 / (link_gbps * 1e9)
where link_gbps is the interconnect-bandwidth PARAMETER. PROCESS-only: we report TIME and
migration-BYTES and feasibility; same graph + same model => same result; never accuracy.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _migration_match(prev: Optional[np.ndarray], cur: np.ndarray, P: int,
                     exact_match_max_dev: int = 6):
    """#vertices that changed device vs prev (over vertices present in BOTH), AFTER
    permutation-matching cur->prev (device LABELS are arbitrary). Returns (moved, remapped_cur)
    so the caller applies the SAME relabel to the node-memory rows it physically migrates."""
    if prev is None:
        return 0, cur
    k = min(len(prev), len(cur))
    if k == 0:
        return 0, cur
    a, b = prev[:k], cur[:k]
    if P <= exact_match_max_dev:
        best_cnt, best_perm = k, np.arange(P)
        for perm in itertools.permutations(range(P)):
            perm = np.asarray(perm, dtype=cur.dtype)
            c = int((a != perm[b]).sum())
            if c < best_cnt:
                best_cnt, best_perm = c, perm
        remap = best_perm
    else:
        overlap = np.zeros((P, P), dtype=np.int64)
        np.add.at(overlap, (b, a), 1)
        remap = overlap.argmax(axis=1)
    remapped = remap[cur].astype(cur.dtype)
    moved = int((prev[:k] != remapped[:k]).sum())
    return moved, remapped


def partition_incremental(src, dst, N, D, prior, new_edge_lo, budget):
    """Reuse `prior`; reassign NEW vertices + the changed cone (endpoints of edges
    [new_edge_lo:]) under a migration budget (<= budget*N old vertices may move), via a
    vectorized changed-cone label propagation. Returns int32 [N] assignment. O(delta)."""
    src = np.asarray(src, dtype=np.int64); dst = np.asarray(dst, dtype=np.int64)
    cap = N // D + 1
    assignment = np.full(N, -1, dtype=np.int32)
    k = min(N, prior.shape[0]) if prior is not None else 0
    if k:
        assignment[:k] = prior[:k]
    load = np.bincount(assignment[assignment >= 0], minlength=D).astype(np.int64)

    new_nodes = np.where(assignment < 0)[0]

    if new_edge_lo < src.size:
        ns, nd = src[new_edge_lo:], dst[new_edge_lo:]
        touched = np.unique(np.concatenate([ns, nd]))
    else:
        touched = np.empty(0, dtype=np.int64)

    is_new = np.zeros(N, dtype=bool)
    is_new[new_nodes] = True
    old_touched = touched[~is_new[touched]]
    B = int(budget * N)
    if B <= 0:
        old_move = np.empty(0, dtype=np.int64)
    elif old_touched.size > B:
        cross = assignment[src] != assignment[dst]
        ends = np.concatenate([src[cross], dst[cross]])
        cross_deg = np.bincount(ends, minlength=N)
        sel = np.argpartition(-cross_deg[old_touched], B - 1)[:B]
        old_move = old_touched[sel]
    else:
        old_move = old_touched

    if old_move.size:
        np.add.at(load, assignment[old_move], -1)
        assignment[old_move] = -1

    to_place = np.concatenate([new_nodes, old_move]).astype(np.int64)
    if to_place.size == 0:
        return assignment

    in_place = np.zeros(N, dtype=bool)
    in_place[to_place] = True
    inc = in_place[src] | in_place[dst]
    isrc, idst = src[inc], dst[inc]
    pn = np.where(in_place[isrc], isrc, idst)
    other = np.where(in_place[isrc], idst, isrc)
    both = in_place[isrc] & in_place[idst]
    pn = np.concatenate([pn, idst[both]])
    other = np.concatenate([other, isrc[both]])

    # vectorized synchronous LPA over the placement set: fixed (already-placed) neighbors
    # anchor harder than tentative placement neighbors, so a community attaches to where it
    # touches the existing partition AND a self-contained new community converges to one device.
    P = to_place.size
    pos = np.full(N, -1, dtype=np.int64)
    pos[to_place] = np.arange(P)
    rows_all = pos[pn]
    nbr_is_fixed = ~in_place[other]
    ANCHOR_W = 4.0
    fmask = nbr_is_fixed & (assignment[other] >= 0)
    fixed_flat = rows_all[fmask] * D + assignment[other][fmask]
    fixed_votes = np.bincount(fixed_flat, minlength=P * D).astype(np.float64) * ANCHOR_W
    tmask = ~nbr_is_fixed
    t_rows = rows_all[tmask]
    t_other_row = pos[other[tmask]]
    tent = (to_place % D).astype(np.int64)
    for _ in range(6):
        flat = t_rows * D + tent[t_other_row]
        votes = fixed_votes + np.bincount(flat, minlength=P * D).astype(np.float64)
        votes = votes.reshape(P, D)
        new_tent = votes.argmax(axis=1)
        no_vote = votes.sum(axis=1) == 0
        new_tent = np.where(no_vote, tent, new_tent)
        if np.array_equal(new_tent, tent):
            break
        tent = new_tent
    assignment[to_place] = tent.astype(np.int32)
    load = np.bincount(assignment[assignment >= 0], minlength=D).astype(np.int64)
    over = load - cap
    while (over > 0).any():
        d_full = int(np.argmax(over))
        d_open = int(np.argmin(load))
        if load[d_open] >= cap:
            break
        movers = to_place[assignment[to_place] == d_full]
        take = movers[: int(min(over[d_full], cap - load[d_open]))]
        if take.size == 0:
            break
        assignment[take] = d_open
        load[d_full] -= take.size
        load[d_open] += take.size
        over = load - cap
    return assignment


def state_migration_sec(moved_vertices: int, mem_dim: int, link_gbps: float,
                        bytes_per_el: int = 4) -> float:
    """Time to ship `moved_vertices` node-memory vectors of width mem_dim over the
    interconnect (the dynamic cost a static problem never pays). link_gbps is a PARAMETER."""
    link = max(link_gbps, 1e-9)
    return (moved_vertices * mem_dim * bytes_per_el) / (link * 1e9)


@dataclass
class IncrementalPlan:
    assignment: np.ndarray         # int32 [N] new assignment (label-matched to prior)
    moved_vertices: int            # vertices that changed device vs prior
    migrated_bytes: int            # node-memory bytes that must cross the interconnect
    migration_sec: float           # predicted migration time (link_gbps parameter)
    new_vertices: int              # genuinely new vertices placed this batch
    note: str = ""


def plan_incremental(src, dst, num_nodes, num_devices, prior,
                     new_edge_lo: int, migration_budget: float,
                     mem_dim: int, link_gbps: float) -> IncrementalPlan:
    """Produce the incremental-migration plan vs the prior batch: reuse + changed-cone LPA
    under the budget, label-match to prior, and cost the resulting state migration.

    prior        : prior batch's int32 assignment (None -> cold start: place all as new).
    new_edge_lo  : first edge offset that is NEW in this batch (edges before are carried over).
    migration_budget : max fraction of OLD vertices that may move (the temporal lever).
    mem_dim      : TGN node-memory vector width m (state moved per migrating vertex).
    link_gbps    : interconnect bandwidth PARAMETER (GB/s).
    """
    N = int(num_nodes); D = int(num_devices)
    if prior is None:
        cur = partition_incremental(src, dst, N, D, None, 0, migration_budget)
        return IncrementalPlan(assignment=cur.astype(np.int32), moved_vertices=0,
                               migrated_bytes=0, migration_sec=0.0,
                               new_vertices=N, note="cold start (no prior batch)")
    cur = partition_incremental(src, dst, N, D, prior, new_edge_lo, migration_budget)
    moved, cur_matched = _migration_match(prior, cur, D)
    migrated_bytes = moved * mem_dim * 4
    mig_sec = state_migration_sec(moved, mem_dim, link_gbps)
    k = min(N, prior.shape[0])
    new_vertices = int(N - k)
    return IncrementalPlan(assignment=cur_matched.astype(np.int32), moved_vertices=moved,
                           migrated_bytes=migrated_bytes, migration_sec=mig_sec,
                           new_vertices=new_vertices,
                           note=f"reuse+changed-cone under budget={migration_budget:.0%}")
