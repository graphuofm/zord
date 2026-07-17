#!/usr/bin/env python
"""INCREMENTAL re-partition vs FROM-SCRATCH on a GROWING temporal graph (the Amap
nightly thesis: don't recompute the partition from scratch every night).

We simulate a temporal graph ARRIVING in S snapshots (edges split by time). After
snapshot s the graph is the CUMULATIVE union of snapshots 0..s -- it only grows.
At each snapshot we re-partition into D devices two ways and report cost + quality:

  FROM-SCRATCH : re-partition the WHOLE cumulative graph. We get a locality order
                 from the C++ kernel (build/graph_algos `lpa` = label propagation,
                 a community ordering) and slice it into D balanced contiguous
                 blocks -> the cut respects the community structure. This is the
                 "recompute nightly" baseline: O(N) work every snapshot.
  INCREMENTAL  : KEEP the prior assignment. Only (re)assign NEW nodes plus nodes
                 whose neighborhood CHANGED this snapshot (the "changed cone" =
                 endpoints of new edges + their 1-hop neighbors), under a MIGRATION
                 BUDGET (max % of nodes allowed to move). Old, untouched nodes are
                 REUSED losslessly -> work is O(delta), not O(N).

Per snapshot we report: scratch_time vs incr_time (speedup), scratch_cut vs
incr_cut (quality gap, % edges cut), and node MIGRATION vs the previous snapshot
(how many nodes changed device == nightly data movement between servers). We also
print the temporal REUSE fraction (reuse@1hop = fraction of already-active nodes
NOT touched this snapshot) -- the ceiling on what incremental can reuse losslessly.

  # synthetic (no dataset mount needed):
  python scripts/incremental_repartition.py --synthetic --nodes 200000 --edges 2000000 \
      --comms 64 --snapshots 8 --devices 4 --migration-budget 0.05
  # real temporal graph (multiple sizes via --dataset):
  python scripts/incremental_repartition.py --dataset askubuntu --snapshots 8 --devices 4 \
      --migration-budget 0.05

PROCESS-only: TIME / cut / migration / feasibility. SAME graph each way; we never
touch accuracy. numpy + the C++ kernel; no networkx, no cluster/SLURM.
"""
import argparse
import itertools
import os
import struct
import subprocess
import time

import numpy as np

# C++ graph kernels (degree|kcore|bfs|lpa|dfs|slashburn|gorder); see reorder_speedup.py
BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")


# --------------------------------------------------------------------------- #
# synthetic temporal graph: C communities, edges TIMESTAMPED so we can snapshot #
# --------------------------------------------------------------------------- #
def gen_synthetic(N, M, C, intra, seed=0):
    """Community-structured temporal graph with TEMPORAL LOCALITY (the property
    real nightly graphs have and that makes incremental win):
      - node ids are community-CONTIGUOUS (block c == one community) so an lpa /
        block partition gets a naturally low cut, and
      - edges arrive in community WAVES: each edge's timestamp is dominated by its
        source community's "phase", so within any snapshot only a few communities
        are active. Most already-seen nodes are therefore UNTOUCHED each snapshot
        -> high reuse@1hop -> the incremental 'changed cone' stays small.
    A small fraction (1-intra) of edges are random cross-community noise (the cut)."""
    rng = np.random.default_rng(seed)
    # community sizes ~ uniform; node ids laid out contiguously per community.
    csize = np.full(C, N // C, dtype=np.int64)
    csize[: N - csize.sum()] += 1
    cstart = np.concatenate([[0], np.cumsum(csize)])           # [C+1] block bounds
    node_comm = np.repeat(np.arange(C), csize)                 # comm of each node id

    m_in = int(M * intra)
    # intra-community edges: pick a community per edge, then two endpoints in it.
    ec = rng.integers(0, C, size=m_in)
    lo = cstart[ec]
    hi = cstart[ec + 1]
    span = np.maximum(1, hi - lo)
    u = lo + (rng.random(m_in) * span).astype(np.int64)
    v = lo + (rng.random(m_in) * span).astype(np.int64)
    # cross-community noise edges (random endpoints -> these are the unavoidable cut)
    mc = M - m_in
    u2 = rng.integers(0, N, size=mc)
    v2 = rng.integers(0, N, size=mc)
    src = np.concatenate([u, u2]).astype(np.int64)
    dst = np.concatenate([v, v2]).astype(np.int64)
    ecomm = node_comm[src]
    # timestamp = community phase (wave) + small within-wave jitter; cross edges get
    # a uniform random time so noise is spread across the whole timeline.
    phase = ecomm.astype(np.float64) / max(C, 1)
    t = phase + rng.random(src.size) * (1.0 / max(C, 1)) * 1.5
    t[m_in:] = rng.random(mc)                                  # noise: anytime
    o = np.argsort(t, kind="stable")
    return src[o], dst[o], N


# --------------------------------------------------------------------------- #
# C++ lpa ordering (from-scratch locality order)                              #
# --------------------------------------------------------------------------- #
def write_edges(path, N, src, dst):
    """Binary edge format the C++ kernel reads: <int64 N><int64 M> then
    interleaved int32 (src,dst) pairs. Matches reorder_speedup.write_edges."""
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", N, src.size))
        inter = np.empty(2 * src.size, dtype=np.int32)
        inter[0::2] = src.astype(np.int32)
        inter[1::2] = dst.astype(np.int32)
        inter.tofile(f)


def cpp_order(edges_path, mode, out_path):
    """Run a C++ ordering kernel; return (newid[N], wall_seconds). newid[v] = the
    new position of node v in the locality order (a permutation)."""
    r = subprocess.run([BIN, edges_path, mode, out_path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpp {mode} failed: {r.stderr.strip()[:300]}")
    with open(out_path, "rb") as f:
        N = struct.unpack("<q", f.read(8))[0]
        newid = np.fromfile(f, dtype=np.int32, count=N)
    return newid


# --------------------------------------------------------------------------- #
# graph helpers                                                                #
# --------------------------------------------------------------------------- #
def build_csr(src, dst, N):
    """Undirected CSR for O(deg) neighbor lookup."""
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    o = np.argsort(u, kind="stable")
    u, v = u[o], v[o]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, u + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, v.astype(np.int64)


def count_cut(assignment, src, dst):
    """Number of edges whose endpoints are on different devices."""
    if src.size == 0:
        return 0
    return int((assignment[src] != assignment[dst]).sum())


def migration(prev, cur, P, exact_match_max_dev=6):
    """Nodes that changed device vs the previous snapshot, over the nodes that
    existed in BOTH snapshots. Device LABELS are arbitrary, so we permutation-
    match cur's labels to prev's before counting (Hungarian via overlap matrix;
    brute force for small P) -- otherwise an arbitrary relabel inflates migration."""
    if prev is None:
        return 0
    k = min(len(prev), len(cur))
    if k == 0:
        return 0
    a, b = prev[:k], cur[:k]
    if P <= exact_match_max_dev:
        best = k
        for perm in itertools.permutations(range(P)):
            pb = np.asarray(perm, dtype=cur.dtype)[b]
            best = min(best, int((a != pb).sum()))
        return best
    # greedy label match for many devices: map each cur-label to the prev-label it
    # most overlaps with (good enough; migration is a diagnostic, not the metric).
    overlap = np.zeros((P, P), dtype=np.int64)
    np.add.at(overlap, (b, a), 1)
    remap = overlap.argmax(axis=1)
    return int((a != remap[b]).sum())


# --------------------------------------------------------------------------- #
# FROM-SCRATCH: lpa community order sliced into D balanced blocks              #
# --------------------------------------------------------------------------- #
def partition_scratch(src, dst, N, D, tmp_edges, tmp_perm):
    """Re-partition the WHOLE cumulative graph. Compute an lpa (label-propagation
    community) order in C++, then cut the order into D equal contiguous blocks.
    Community-adjacent nodes land in the same block -> low cut, balanced load."""
    write_edges(tmp_edges, N, src, dst)
    try:
        newid = cpp_order(tmp_edges, "lpa", tmp_perm)          # newid[v] = rank
    except (RuntimeError, FileNotFoundError) as e:
        # no C++ kernel available: fall back to a degree-descending order in numpy
        # so the experiment still runs (same "order -> block" recipe).
        print(f"  [scratch] lpa unavailable ({str(e)[:60]}); numpy degree fallback")
        indptr, _ = build_csr(src, dst, N)
        deg = np.diff(indptr)
        newid = np.empty(N, dtype=np.int32)
        newid[np.argsort(-deg, kind="stable")] = np.arange(N)
    # block id by rank: rank in [0,N) -> device floor(rank * D / N)
    rank = newid.astype(np.int64)
    assignment = (rank * D // N).astype(np.int32)
    np.clip(assignment, 0, D - 1, out=assignment)
    return assignment


# --------------------------------------------------------------------------- #
# INCREMENTAL: reuse prior; reassign new nodes + changed cone under a budget   #
# --------------------------------------------------------------------------- #
def partition_incremental(src, dst, N, D, prior, new_edge_lo, budget):
    """Keep `prior` assignment for surviving nodes; (re)assign ONLY:
      - NEW nodes (id never placed before), and
      - up to budget*N OLD boundary nodes inside the CHANGED CONE (nodes incident
        to this snapshot's new edges, i.e. whose neighborhood histogram changed).
    Untouched nodes are REUSED losslessly. Work is O(delta) = O(new edges +
    placed nodes), NOT O(N): we never reorder the whole graph and never build a
    full-graph CSR -- neighbor-device histograms are computed only for the small
    set of nodes we actually (re)place, by filtering the incident edges.

    `prior` MUST be provided (cold start = a from-scratch partition; see driver).
    Placement = greedy min-cut: node -> device holding most of its already-placed
    neighbors, with a soft capacity penalty for load balance.
    """
    cap = N // D + 1                                            # soft per-device cap
    assignment = np.full(N, -1, dtype=np.int32)
    k = min(N, prior.shape[0]) if prior is not None else 0
    if k:
        assignment[:k] = prior[:k]
    load = np.bincount(assignment[assignment >= 0], minlength=D).astype(np.int64)

    new_nodes = np.where(assignment < 0)[0]                     # genuinely new ids

    # changed cone = nodes touched by THIS snapshot's edges (neighborhood changed).
    if new_edge_lo < src.size:
        ns, nd = src[new_edge_lo:], dst[new_edge_lo:]
        touched = np.unique(np.concatenate([ns, nd]))
    else:
        touched = np.empty(0, dtype=np.int64)

    # OLD touched nodes are candidates to MOVE; bound by the migration budget,
    # preferring the worst boundary nodes (most cross-device incident edges).
    is_new = np.zeros(N, dtype=bool)
    is_new[new_nodes] = True
    old_touched = touched[~is_new[touched]]
    B = int(budget * N)
    if B <= 0:
        old_move = np.empty(0, dtype=np.int64)
    elif old_touched.size > B:
        cross = assignment[src] != assignment[dst]
        ends = np.concatenate([src[cross], dst[cross]])
        cross_deg = np.bincount(ends, minlength=N)              # bincount >> np.add.at
        sel = np.argpartition(-cross_deg[old_touched], B - 1)[:B]
        old_move = old_touched[sel]
    else:
        old_move = old_touched

    # free the slots of old nodes we will move (so cut/load reflect the move)
    if old_move.size:
        np.add.at(load, assignment[old_move], -1)
        assignment[old_move] = -1

    to_place = np.concatenate([new_nodes, old_move]).astype(np.int64)
    if to_place.size == 0:
        return assignment

    # Build neighbor (node, device) pairs ONLY for the placement set, from the
    # FULL cumulative edge list, but cheaply: keep edges incident to a to_place
    # node, take the OTHER endpoint's current device. This is the O(delta) core.
    in_place = np.zeros(N, dtype=bool)
    in_place[to_place] = True
    es_in_p = in_place[src]
    ed_in_p = in_place[dst]
    inc = es_in_p | ed_in_p                                     # edges touching a placement node
    isrc, idst = src[inc], dst[inc]
    # orient so the FIRST endpoint is the placement node; emit (place_node, other)
    pn = np.where(in_place[isrc], isrc, idst)
    other = np.where(in_place[isrc], idst, isrc)
    # also handle edges where BOTH endpoints are being placed (emit both directions)
    both = in_place[isrc] & in_place[idst]
    pn = np.concatenate([pn, idst[both]])
    other = np.concatenate([other, isrc[both]])

    # Greedy placement loop over to_place. We process in passes so a placement
    # node can "see" earlier-placed siblings; neighbor devices are looked up live.
    INF = np.inf
    order = to_place                                           # new first, then movers
    # pre-sort emitted pairs by placement node for O(deg) slicing
    so = np.argsort(pn, kind="stable")
    pn_s, other_s = pn[so], other[so]
    bounds = np.searchsorted(pn_s, order)
    bounds_hi = np.searchsorted(pn_s, order, side="right")
    for i, v in enumerate(order):
        nbrs = other_s[bounds[i]:bounds_hi[i]]
        dv = assignment[nbrs]
        placed = dv[dv >= 0]
        if placed.size:
            cnt = np.bincount(placed, minlength=D).astype(np.float64)
            cut = -cnt                                          # prefer device with most nbrs
        else:
            cut = np.zeros(D)
        score = cut + (load / cap)                             # + soft load-balance penalty
        score = np.where(load >= cap, INF, score)
        d = int(np.argmin(score))
        if not np.isfinite(score[d]):                          # all full -> most headroom
            d = int(np.argmax(cap - load))
        assignment[v] = d
        load[d] += 1

    return assignment


# --------------------------------------------------------------------------- #
# reuse fraction (the lossless-reuse ceiling)                                  #
# --------------------------------------------------------------------------- #
def reuse_at_1hop(active_before, n_active_before, new_src, new_dst, N):
    """Fraction of already-active nodes with NO incident new edge this snapshot
    (their 1-hop neighborhood is unchanged -> their assignment is reusable as-is).
    Mirrors scripts/reuse_probe.py."""
    if n_active_before == 0:
        return float("nan")
    touched = np.zeros(N, dtype=bool)
    touched[new_src] = True
    touched[new_dst] = True
    return 1.0 - (touched & active_before).sum() / n_active_before


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def load_graph(args):
    if args.dataset:
        from zord.datasets import load
        g = load(args.dataset).sort_by_time()
        return g.src.astype(np.int64), g.dst.astype(np.int64), int(g.num_nodes), g.name
    src, dst, N = gen_synthetic(args.nodes, args.edges, args.comms, args.intra, args.seed)
    return src, dst, N, f"synthetic(N={N},M={src.size},C={args.comms})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src_grp = ap.add_mutually_exclusive_group()
    src_grp.add_argument("--dataset", default="", help="real temporal graph (zord.datasets.load)")
    src_grp.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    ap.add_argument("--nodes", type=int, default=200_000, help="synthetic node count")
    ap.add_argument("--edges", type=int, default=2_000_000, help="synthetic edge count")
    ap.add_argument("--comms", type=int, default=64, help="synthetic community count")
    ap.add_argument("--intra", type=float, default=0.9, help="synthetic intra-community edge frac")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshots", type=int, default=8, help="number of arriving snapshots S")
    ap.add_argument("--devices", type=int, default=4, help="number of devices/partitions D")
    ap.add_argument("--migration-budget", type=float, default=0.05,
                    help="max fraction of nodes incremental may MOVE per snapshot")
    a = ap.parse_args()

    src, dst, N, name = load_graph(a)
    M = src.size
    D = a.devices
    S = a.snapshots
    tmp_edges = f"/tmp/zord_incr_edges_{os.getpid()}.bin"
    tmp_perm = f"/tmp/zord_incr_perm_{os.getpid()}.bin"

    # cumulative edge offsets: snapshot s covers edges [0, bnd[s+1]); equal-size
    # edge buckets over the time-sorted stream (graph grows monotonically).
    bnd = np.linspace(0, M, S + 1).astype(np.int64)

    print(f"INCREMENTAL-REPARTITION dataset={name} N={N:,} M={M:,} "
          f"snapshots={S} devices={D} budget={a.migration_budget:.0%} bin={BIN}")
    print(f"{'snap':>4} {'nodes':>10} {'edges':>11} {'reuse@1h':>9} | "
          f"{'scr_t(s)':>9} {'inc_t(s)':>9} {'speedup':>8} | "
          f"{'scr_cut%':>9} {'inc_cut%':>9} {'gap':>7} | "
          f"{'scr_mig':>9} {'inc_mig':>9}")
    print("-" * 132)

    prev_scratch = prev_incr = None
    prev_hi = 0
    active = np.zeros(N, dtype=bool)
    tot_scr_t = tot_inc_t = 0.0
    tot_scr_mig = tot_inc_mig = 0
    sum_speedup, sum_gap, n_rows = 0.0, 0.0, 0

    for s in range(S):
        hi = int(bnd[s + 1])
        if hi <= prev_hi and s > 0:
            continue
        es, ed = src[:hi], dst[:hi]                            # cumulative graph
        nn = int(max(es.max(initial=-1), ed.max(initial=-1)) + 1)

        # reuse fraction (ceiling on lossless reuse) for the edges that just arrived
        n_active_before = int(active.sum())
        active_before = active.copy()
        reuse1 = reuse_at_1hop(active_before, n_active_before,
                               src[prev_hi:hi], dst[prev_hi:hi], N)

        # ---- FROM SCRATCH: re-partition the whole cumulative graph ----
        t0 = time.time()
        a_scr = partition_scratch(es, ed, nn, D, tmp_edges, tmp_perm)
        t_scr = time.time() - t0

        # ---- INCREMENTAL: reuse prior, reassign new + changed cone under budget ----
        if prev_incr is None:
            # COLD START (first night): you must partition from scratch -- there is
            # nothing to reuse yet. Incremental seeds itself from the scratch result
            # and identical cost/quality; it only diverges (and saves) from snap 1 on.
            a_inc = a_scr.copy()
            t_inc = t_scr
        else:
            t0 = time.time()
            a_inc = partition_incremental(es, ed, nn, D, prev_incr, prev_hi, a.migration_budget)
            t_inc = time.time() - t0

        cut_scr = count_cut(a_scr, es, ed)
        cut_inc = count_cut(a_inc, es, ed)
        cutpct_scr = 100.0 * cut_scr / max(hi, 1)
        cutpct_inc = 100.0 * cut_inc / max(hi, 1)
        mig_scr = migration(prev_scratch, a_scr, D)
        mig_inc = migration(prev_incr, a_inc, D)

        speedup = t_scr / max(t_inc, 1e-9)
        gap = cutpct_inc - cutpct_scr                          # +ve => incremental cuts more
        r1s = f"{reuse1:.3f}" if reuse1 == reuse1 else "  n/a"  # nan-safe
        print(f"{s:>4} {nn:>10,} {hi:>11,} {r1s:>9} | "
              f"{t_scr:>9.3f} {t_inc:>9.3f} {speedup:>7.1f}x | "
              f"{cutpct_scr:>8.2f}% {cutpct_inc:>8.2f}% {gap:>+6.2f}% | "
              f"{mig_scr:>9,} {mig_inc:>9,}")

        tot_scr_t += t_scr
        tot_inc_t += t_inc
        tot_scr_mig += mig_scr
        tot_inc_mig += mig_inc
        if s > 0:
            sum_speedup += speedup
            sum_gap += gap
            n_rows += 1
        prev_scratch, prev_incr = a_scr, a_inc
        active[src[prev_hi:hi]] = True
        active[dst[prev_hi:hi]] = True
        prev_hi = hi

    print("-" * 132)
    print(f"TOTAL re-partition time : scratch {tot_scr_t:7.2f}s  vs  incremental "
          f"{tot_inc_t:7.2f}s   ({tot_scr_t / max(tot_inc_t, 1e-9):.1f}x faster)")
    print(f"TOTAL node migration    : scratch {tot_scr_mig:>12,}  vs  incremental "
          f"{tot_inc_mig:>12,}   ({tot_scr_mig / max(tot_inc_mig, 1):.0f}x more for scratch)")
    if n_rows:
        print(f"MEAN (snaps>0)          : speedup {sum_speedup / n_rows:.1f}x   "
              f"cut quality gap {sum_gap / n_rows:+.2f}% (incremental - scratch)")
    print("TRADEOFF: incremental trades a small cut-quality gap + a bounded migration "
          "budget for an order-of-magnitude cheaper re-partition; the more REUSE@1hop, "
          "the smaller the gap (less of the graph changed -> more reused losslessly).")

    for p in (tmp_edges, tmp_perm):
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    main()
