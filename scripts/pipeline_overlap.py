#!/usr/bin/env python
"""GLOBAL ACCELERATION via 3-STAGE SOFTWARE PIPELINING (D35).

A temporal graph arrives in S snapshots. Processing each snapshot has a strict
dependency CHAIN of three stages:

    LOAD(front)  ->  ARRANGE(middle)  ->  TRAIN(back)
    read/prep        incremental         2-layer GNN
    edges+feats      re-partition        aggregation (SpMM)
    (CPU / I/O)      (CPU / C++ kernel)   (GPU)

The chain within a snapshot is sequential, BUT the stages use DISJOINT hardware
(CPU I/O, CPU+C++ compute, GPU) -- so across snapshots they can be PIPELINED like
a CPU instruction pipeline: while the GPU is TRAINING snapshot t, a background CPU
thread concurrently LOADs t+1 and INCREMENTAL-ARRANGEs t+1. With a double-buffered
handoff, the per-snapshot wall-clock collapses from the SUM of the three stages to
the MAX of them:

    SEQUENTIAL : total = sum_t (load + arrange + train)            ~ (L+A+T)*S
    OVERLAPPED : total ~ max(L, A, T) * S  +  fill/drain           (the pipeline)

    speedup -> (L+A+T) / max(L,A,T)        (3x if perfectly balanced)

The whole point of zord's INCREMENTAL arrange (reuse t-1's assignment + only the
"changed cone" of this snapshot's new edges, via the C++ kernel) is that it is
CHEAP BY DESIGN: arrange < train, so ARRANGE stays OFF the critical path and TRAIN
(the GPU) is the steady-state BOTTLENECK -- exactly where you want the bottleneck.

This is PROCESS-ONLY: we measure WALL-CLOCK of the identical work done two ways;
the numerical result is bit-identical (same SpMM, same assignment). We NEVER touch
accuracy. Single GPU is fine -- the headline is the OVERLAP of the 3 stages, not
multi-GPU. The --devices arg is the partition count the arrange stage splits into.

  # synthetic (no dataset mount needed):
  python scripts/pipeline_overlap.py --nodes 400000 --edges 4000000 --comms 64 \
      --snapshots 8 --feat 128 --devices 4
  # real temporal graph, sliced into S snapshots:
  python scripts/pipeline_overlap.py --dataset askubuntu --snapshots 8 --feat 128 --devices 4
  # sweep the snapshot count to watch overlapped -> max-stage*S:
  python scripts/pipeline_overlap.py --nodes 400000 --edges 4000000 --sweep 4,8,16

numpy + the C++ kernel (build/graph_algos) for arrange + PyTorch (GPU SpMM + CUDA
stream) for train + a real threading.Thread for the CPU load/arrange overlap.
No networkx, no cluster/SLURM.
"""
import argparse
import os
import queue
import struct
import subprocess
import threading
import time

import numpy as np
import torch

# C++ graph kernels (degree|kcore|bfs|lpa|dfs|slashburn|gorder); see reorder_speedup.py
BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")


# --------------------------------------------------------------------------- #
# synthetic temporal graph (community-structured, temporal locality)          #
# --------------------------------------------------------------------------- #
def gen_synthetic(N, M, C, intra, seed=0):
    """Community-structured temporal graph with TEMPORAL LOCALITY: node ids are
    community-contiguous and edges arrive in community waves, so within a snapshot
    only a few communities are active -> the incremental arrange's changed cone
    stays small (the property that keeps arrange cheap). Mirrors the generator in
    incremental_repartition.py."""
    rng = np.random.default_rng(seed)
    csize = np.full(C, N // C, dtype=np.int64)
    csize[: N - csize.sum()] += 1
    cstart = np.concatenate([[0], np.cumsum(csize)])
    node_comm = np.repeat(np.arange(C), csize)

    m_in = int(M * intra)
    ec = rng.integers(0, C, size=m_in)
    lo = cstart[ec]
    hi = cstart[ec + 1]
    span = np.maximum(1, hi - lo)
    u = lo + (rng.random(m_in) * span).astype(np.int64)
    v = lo + (rng.random(m_in) * span).astype(np.int64)
    mc = M - m_in
    u2 = rng.integers(0, N, size=mc)
    v2 = rng.integers(0, N, size=mc)
    src = np.concatenate([u, u2]).astype(np.int64)
    dst = np.concatenate([v, v2]).astype(np.int64)
    ecomm = node_comm[src]
    phase = ecomm.astype(np.float64) / max(C, 1)
    t = phase + rng.random(src.size) * (1.0 / max(C, 1)) * 1.5
    t[m_in:] = rng.random(mc)
    o = np.argsort(t, kind="stable")
    return src[o], dst[o], N


def load_graph(args):
    if args.dataset:
        from zord.datasets import load
        g = load(args.dataset).sort_by_time()
        return g.src.astype(np.int64), g.dst.astype(np.int64), int(g.num_nodes), g.name
    src, dst, N = gen_synthetic(args.nodes, args.edges, args.comms, args.intra, args.seed)
    return src, dst, N, f"synthetic(N={N},M={src.size},C={args.comms})"


# --------------------------------------------------------------------------- #
# C++ ordering kernel (the arrange backend); same I/O contract as the refs    #
# --------------------------------------------------------------------------- #
def write_edges(path, N, src, dst):
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", N, src.size))
        inter = np.empty(2 * src.size, dtype=np.int32)
        inter[0::2] = src.astype(np.int32)
        inter[1::2] = dst.astype(np.int32)
        inter.tofile(f)


def cpp_order(edges_path, mode, out_path):
    r = subprocess.run([BIN, edges_path, mode, out_path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpp {mode} failed: {r.stderr.strip()[:300]}")
    with open(out_path, "rb") as f:
        N = struct.unpack("<q", f.read(8))[0]
        newid = np.fromfile(f, dtype=np.int32, count=N)
    return newid


# =========================================================================== #
# THE THREE STAGES                                                            #
# =========================================================================== #
def stage_load(src, dst, lo, hi, N, F, io_mb_per_s, rng):
    """LOAD(t): read/prepare snapshot t -- this snapshot's NEW edges and the node
    feature rows they touch. We simulate the I/O cost of streaming the snapshot
    off storage (bytes / bandwidth) so the front stage has a realistic, tunable
    duration; the returned arrays are the prepared snapshot. (With --dataset the
    edges are real; only the I/O *delay* is simulated.)"""
    s = src[lo:hi]
    d = dst[lo:hi]
    # materialise per-snapshot node features (the rows this snapshot activates).
    touched = np.unique(np.concatenate([s, d])) if s.size else np.empty(0, dtype=np.int64)
    feats = rng.standard_normal((touched.size, F), dtype=np.float32) if touched.size else \
        np.empty((0, F), dtype=np.float32)
    # simulate storage I/O: edges (2*int64) + feature bytes over a bandwidth budget.
    nbytes = s.size * 16 + feats.nbytes
    if io_mb_per_s > 0:
        delay = nbytes / (io_mb_per_s * 1024 * 1024)
        time.sleep(delay)
    return {"s": s, "d": d, "touched": touched, "feats": feats, "lo": lo, "hi": hi}


def stage_arrange(snap, N, D, prior, prev_hi, budget, tmp_edges, tmp_perm):
    """ARRANGE(t): INCREMENTAL re-partition of the CUMULATIVE graph at t. Reuse the
    prior assignment for untouched nodes; only (re)place NEW nodes + the changed
    cone (endpoints of this snapshot's new edges) under a migration budget. Work is
    O(delta), not O(N) -- cheap BY DESIGN so it stays off the critical path.

    On the COLD snapshot (prior is None) we seed once from a from-scratch C++ lpa
    order (there is nothing to reuse yet); from then on it is pure incremental.
    The cumulative graph slices are attached to `snap` by the driver/producer.
    Returns (assignment, "incr"|"scratch")."""
    es, ed, nn = snap["_cum_src"], snap["_cum_dst"], snap["_cum_nn"]
    if prior is None:
        # cold start: from-scratch lpa order -> D balanced contiguous blocks.
        write_edges(tmp_edges, nn, es, ed)
        try:
            newid = cpp_order(tmp_edges, "lpa", tmp_perm)
        except (RuntimeError, FileNotFoundError):
            # numpy degree-descending fallback (same order->block recipe)
            deg = np.bincount(np.concatenate([es, ed]), minlength=nn)
            newid = np.empty(nn, dtype=np.int32)
            newid[np.argsort(-deg, kind="stable")] = np.arange(nn)
        rank = newid.astype(np.int64)
        assignment = np.clip((rank * D // nn), 0, D - 1).astype(np.int32)
        return assignment, "scratch"
    return partition_incremental(es, ed, nn, D, prior, prev_hi, budget), "incr"


def partition_incremental(src, dst, N, D, prior, new_edge_lo, budget):
    """Keep `prior` for surviving nodes; reassign ONLY new nodes + (budgeted) old
    boundary nodes in the changed cone. O(delta) greedy min-cut placement. Adapted
    from incremental_repartition.partition_incremental."""
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

    so = np.argsort(pn, kind="stable")
    pn_s, other_s = pn[so], other[so]
    bounds = np.searchsorted(pn_s, to_place)
    bounds_hi = np.searchsorted(pn_s, to_place, side="right")
    INF = np.inf
    for i, v in enumerate(to_place):
        nbrs = other_s[bounds[i]:bounds_hi[i]]
        dv = assignment[nbrs]
        placed = dv[dv >= 0]
        if placed.size:
            cnt = np.bincount(placed, minlength=D).astype(np.float64)
            score = -cnt + (load / cap)
        else:
            score = (load / cap).astype(np.float64)
        score = np.where(load >= cap, INF, score)
        d = int(np.argmin(score))
        if not np.isfinite(score[d]):
            d = int(np.argmax(cap - load))
        assignment[v] = d
        load[d] += 1
    return assignment


def make_train_fn(N, F, dev, W1, W2):
    """Build the TRAIN(t) closure: a 2-layer GNN aggregation (normalized-adjacency
    SpMM x2) on the snapshot's subgraph, executed on the GPU. Returns a function
    train(snap, assignment, stream) -> runs on the given CUDA stream, syncs nothing
    (caller controls synchronization for overlap)."""
    def build_csr_gpu(s, d, n):
        # symmetric, degree-normalized adjacency of the snapshot's touched subgraph
        r = np.concatenate([s, d]).astype(np.int64)
        c = np.concatenate([d, s]).astype(np.int64)
        o = np.argsort(r, kind="stable"); r = r[o]; c = c[o]
        counts = np.bincount(r, minlength=n)
        deg = counts.astype(np.float32); deg[deg == 0] = 1.0
        vals = (1.0 / deg[r]).astype(np.float32)
        crow = np.zeros(n + 1, dtype=np.int64); np.cumsum(counts, out=crow[1:])
        return (torch.from_numpy(crow), torch.from_numpy(c), torch.from_numpy(vals))

    def train(snap, stream):
        s, d = snap["s"], snap["d"]
        if s.size == 0:
            return
        crow, col, vals = build_csr_gpu(s, d, N)
        with torch.cuda.stream(stream):
            A = torch.sparse_csr_tensor(crow.to(dev, non_blocking=True),
                                        col.to(dev, non_blocking=True),
                                        vals.to(dev, non_blocking=True),
                                        size=(N, N), device=dev)
            X = torch.empty(N, F, device=dev)
            X.normal_()
            # 2-layer aggregation: A (relu(A X W1)) W2  -- the memory-bound GNN step
            H = torch.sparse.mm(A, X) @ W1
            torch.sparse.mm(A, torch.relu(H)) @ W2
    return train


# =========================================================================== #
# DRIVERS: sequential vs overlapped                                           #
# =========================================================================== #
def run_sequential(snaps_meta, src, dst, N, F, D, budget, dev, train, io_mb,
                   tmp_edges, tmp_perm, rng_seed):
    """(1) SEQUENTIAL: for each t do load; arrange; train -- one after another.
    total = sum_t (load + arrange + train). Also records per-stage mean times."""
    rng = np.random.default_rng(rng_seed)
    stream = torch.cuda.current_stream()
    prior = None
    prev_hi = 0
    tl = ta = tt = 0.0
    is_off_crit = []  # per-snapshot: arrange < train ?
    torch.cuda.synchronize()
    wall0 = time.perf_counter()
    for (lo, hi, cum_hi) in snaps_meta:
        t0 = time.perf_counter()
        snap = stage_load(src, dst, lo, hi, N, F, io_mb, rng)
        t1 = time.perf_counter()
        snap["_cum_src"] = src[:cum_hi]; snap["_cum_dst"] = dst[:cum_hi]
        snap["_cum_nn"] = int(max(src[:cum_hi].max(initial=-1),
                                  dst[:cum_hi].max(initial=-1)) + 1)
        assignment, _ = stage_arrange(snap, N, D, prior, prev_hi, budget,
                                      tmp_edges, tmp_perm)
        t2 = time.perf_counter()
        train(snap, stream)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        tl += t1 - t0; ta += t2 - t1; tt += t3 - t2
        is_off_crit.append((t2 - t1) < (t3 - t2))
        prior = assignment; prev_hi = hi
    torch.cuda.synchronize()
    total = time.perf_counter() - wall0
    S = len(snaps_meta)
    return {
        "total": total,
        "load_mean": tl / S, "arrange_mean": ta / S, "train_mean": tt / S,
        "arrange_off_critical": all(is_off_crit),
        "off_crit_frac": sum(is_off_crit) / S,
    }


def run_overlapped(snaps_meta, src, dst, N, F, D, budget, dev, train, io_mb,
                   tmp_edges, tmp_perm, rng_seed):
    """(2) OVERLAPPED: a SOFTWARE PIPELINE. A background producer THREAD runs the
    front two stages -- load(t)+arrange(t) -- for every snapshot and hands each
    prepared snapshot to the main thread through a bounded (double-buffer) queue.
    The main thread runs TRAIN(t) on a dedicated CUDA STREAM. So while the GPU
    trains t, the CPU thread is already loading+arranging t+1.

    Steady state: per-snapshot wall ~ max(load+arrange_on_cpu, train_on_gpu).
    Because arrange is incremental (cheap) and load is I/O, the CPU producer's
    L+A is meant to be <= the GPU's T -> TRAIN is the bottleneck and the producer
    is fully hidden. total ~ max-stage*S + fill (first load+arrange) + drain (last
    train). The handoff queue holds at most 2 snapshots (the double buffer)."""
    rng = np.random.default_rng(rng_seed)
    q = queue.Queue(maxsize=2)               # double-buffer handoff (main<-producer)
    err_box = {}

    def producer():
        prior = None
        prev_hi = 0
        try:
            for (lo, hi, cum_hi) in snaps_meta:
                snap = stage_load(src, dst, lo, hi, N, F, io_mb, rng)
                snap["_cum_src"] = src[:cum_hi]; snap["_cum_dst"] = dst[:cum_hi]
                snap["_cum_nn"] = int(max(src[:cum_hi].max(initial=-1),
                                          dst[:cum_hi].max(initial=-1)) + 1)
                assignment, _ = stage_arrange(snap, N, D, prior, prev_hi, budget,
                                              tmp_edges, tmp_perm)
                snap["assignment"] = assignment
                # drop the bulky cumulative refs before handoff; train needs only s/d
                for kk in ("_cum_src", "_cum_dst"):
                    snap.pop(kk, None)
                q.put(snap)
                prior = assignment; prev_hi = hi
        except Exception as e:                # surface producer faults to main
            err_box["err"] = e
        finally:
            q.put(None)                       # sentinel: producer done

    comp = torch.cuda.Stream()                # dedicated TRAIN stream (overlaps producer)
    torch.cuda.synchronize()
    wall0 = time.perf_counter()
    th = threading.Thread(target=producer, daemon=True)
    th.start()
    while True:
        snap = q.get()                        # blocks until producer has t ready
        if snap is None:
            break
        train(snap, comp)                     # GPU train(t) overlaps producer's load/arrange(t+1)
        comp.synchronize()                    # finish t before we (logically) retire it
    th.join()
    torch.cuda.synchronize()
    total = time.perf_counter() - wall0
    if "err" in err_box:
        raise err_box["err"]
    return {"total": total}


# =========================================================================== #
# one full run for a given S                                                  #
# =========================================================================== #
def run_for_S(src, dst, N, F, D, S, budget, dev, W1, W2, io_mb, warmup_train):
    M = src.size
    bnd = np.linspace(0, M, S + 1).astype(np.int64)
    # snaps_meta: (lo, hi, cum_hi) for snapshot s -- new edges [lo,hi), cumulative [0,cum_hi)
    snaps_meta = [(int(bnd[s]), int(bnd[s + 1]), int(bnd[s + 1])) for s in range(S)]
    tmp_edges = f"/tmp/zord_pipe_edges_{os.getpid()}_{S}.bin"
    tmp_perm = f"/tmp/zord_pipe_perm_{os.getpid()}_{S}.bin"
    train = make_train_fn(N, F, dev, W1, W2)

    # warmup the GPU (kernels/allocator) so timing is steady-state, not cold.
    if warmup_train:
        warm = {"s": src[:max(1, M // S)], "d": dst[:max(1, M // S)]}
        for _ in range(2):
            train(warm, torch.cuda.current_stream())
        torch.cuda.synchronize()

    seq = run_sequential(snaps_meta, src, dst, N, F, D, budget, dev, train, io_mb,
                         tmp_edges, tmp_perm, rng_seed=100)
    ovl = run_overlapped(snaps_meta, src, dst, N, F, D, budget, dev, train, io_mb,
                         tmp_edges, tmp_perm, rng_seed=100)

    for p in (tmp_edges, tmp_perm):
        try:
            os.remove(p)
        except OSError:
            pass

    L, A, T = seq["load_mean"], seq["arrange_mean"], seq["train_mean"]
    stages = {"load": L, "arrange": A, "train": T}
    bottleneck = max(stages, key=stages.get)
    max_stage = stages[bottleneck]
    ideal = max_stage * S                                   # the pipeline floor
    speedup = seq["total"] / max(ovl["total"], 1e-9)
    return {
        "S": S, "L": L, "A": A, "T": T,
        "seq_total": seq["total"], "ovl_total": ovl["total"],
        "speedup": speedup, "bottleneck": bottleneck, "ideal": ideal,
        "arrange_off_critical": seq["arrange_off_critical"],
        "off_crit_frac": seq["off_crit_frac"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src_grp = ap.add_mutually_exclusive_group()
    src_grp.add_argument("--dataset", default="", help="real temporal graph (zord.datasets.load)")
    src_grp.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    ap.add_argument("--nodes", type=int, default=400_000, help="synthetic node count")
    ap.add_argument("--edges", type=int, default=4_000_000, help="synthetic edge count")
    ap.add_argument("--comms", type=int, default=64, help="synthetic community count")
    ap.add_argument("--intra", type=float, default=0.9, help="synthetic intra-community edge frac")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshots", type=int, default=8, help="number of arriving snapshots S")
    ap.add_argument("--devices", type=int, default=4, help="partition count for the arrange stage")
    ap.add_argument("--feat", type=int, default=128, help="node feature dim F")
    ap.add_argument("--migration-budget", type=float, default=0.05,
                    help="max fraction of nodes the incremental arrange may MOVE per snapshot")
    ap.add_argument("--io-mb-per-s", type=float, default=2048.0,
                    help="simulated LOAD-stage storage bandwidth (MB/s); 0 disables the I/O delay")
    ap.add_argument("--sweep", default="", help="comma list of S values to sweep, e.g. 4,8,16")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("pipeline_overlap requires a CUDA GPU (train stage runs on the GPU).")
    dev = "cuda:0"
    gpu = torch.cuda.get_device_name(0)
    F = a.feat
    D = a.devices

    src, dst, N, name = load_graph(a)
    M = src.size
    W1 = torch.randn(F, F, device=dev) / F ** 0.5
    W2 = torch.randn(F, F, device=dev) / F ** 0.5

    print(f"PIPELINE-OVERLAP gpu='{gpu}' dataset={name} N={N:,} M={M:,} F={F} "
          f"devices(D)={D} budget={a.migration_budget:.0%} io={a.io_mb_per_s:.0f}MB/s bin={BIN}")
    print("stages: LOAD(CPU/IO) -> ARRANGE(CPU/C++ incremental) -> TRAIN(GPU 2-layer SpMM)")
    print(f"{'S':>4} | {'load(ms)':>9} {'arr(ms)':>9} {'train(ms)':>10} | "
          f"{'SEQ(s)':>9} {'OVL(s)':>9} {'ideal(s)':>9} | "
          f"{'speedup':>8} {'bottleneck':>10} {'arr<train':>9}")
    print("-" * 116)

    S_list = [int(x) for x in a.sweep.split(",") if x] if a.sweep else [a.snapshots]
    rows = []
    for S in S_list:
        if S > M:
            print(f"{S:>4} | (skip: S > edges)")
            continue
        r = run_for_S(src, dst, N, F, D, S, a.migration_budget, dev, W1, W2,
                      a.io_mb_per_s, warmup_train=True)
        rows.append(r)
        off = "YES" if r["arrange_off_critical"] else f"{r['off_crit_frac']*100:.0f}%"
        print(f"{r['S']:>4} | {r['L']*1e3:>9.2f} {r['A']*1e3:>9.2f} {r['T']*1e3:>10.2f} | "
              f"{r['seq_total']:>9.3f} {r['ovl_total']:>9.3f} {r['ideal']:>9.3f} | "
              f"{r['speedup']:>7.2f}x {r['bottleneck']:>10} {off:>9}")
    print("-" * 116)

    if rows:
        r0 = rows[0]
        sumstage = r0["L"] + r0["A"] + r0["T"]
        max_stage = {"load": r0["L"], "arrange": r0["A"], "train": r0["T"]}[r0["bottleneck"]]
        bal = sumstage / max(max_stage, 1e-9)
        print(f"PER-STAGE (S={r0['S']}): load={r0['L']*1e3:.2f}ms arrange={r0['A']*1e3:.2f}ms "
              f"train={r0['T']*1e3:.2f}ms  (sum={sumstage*1e3:.2f}ms, max={max_stage*1e3:.2f}ms)")
        print(f"BOTTLENECK stage = '{r0['bottleneck']}'  ->  pipeline ceiling speedup ~ "
              f"(L+A+T)/max = {bal:.2f}x")
        off_msg = ("YES" if r0["arrange_off_critical"]
                   else f"partial ({r0['off_crit_frac']*100:.0f}% of snaps)")
        print(f"INCREMENTAL ARRANGE off critical path (arrange<train every snapshot): {off_msg}")
        print(f"OBSERVED overlap speedup = {r0['speedup']:.2f}x  "
              f"(SEQ {r0['seq_total']:.3f}s -> OVL {r0['ovl_total']:.3f}s)")
        print("HEADLINE: OVERLAPPED total approaches max-stage*S (the pipeline floor) instead of "
              "the SUM of the three stages -> that gap is zord's GLOBAL ACCELERATION (D35). "
              "Sequential pays L+A+T per snapshot; the software pipeline pays only max(L,A,T) "
              "in steady state, plus a one-time fill (first load+arrange) and drain (last train).")


if __name__ == "__main__":
    main()
