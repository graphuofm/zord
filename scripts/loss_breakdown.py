#!/usr/bin/env python
"""WHERE DOES THE WALL-CLOCK GO? -- a per-rank LOSS BREAKDOWN of one distributed (NCCL, P GPUs)
2-layer temporal-GNN step under a given vertex->device ASSIGNMENT. zord's contribution is a
HARDWARE-AGNOSTIC middle-layer SCHEDULER; before it can attack the loss it must SEE which bucket the
wall-clock is lost to. This script MEASURES that decomposition on whatever GPUs/links the job lands on
(it does NOT assume NVLink, PCIe, or any tier -- hardware is a downstream parameter it just times).

PROCESS metric only (TIME / feasibility). The numerical result is identical across policies; only the
vertex->rank assignment changes per-rank edge counts -> per-rank busy time -> the straggler-idle loss.

Each rank's per-step wall-time is split into four mutually-exclusive buckets:
  (1) COMPUTE        -- local aggregation (2 SpMM gathers over this rank's edges), CUDA-Event timed.
  (2) COMM           -- cross-device feature/memory exchange (boundary isend/irecv of the cut rows).
  (3) MEMORY-STALL   -- H2D staging wait, when --stage moves the feature bank out-of-core (else 0).
  (4) SYNC-WAIT / STRAGGLER-IDLE  -- time this rank sits blocked at the step barrier waiting for the
                       SLOWEST rank = makespan - this_rank's_busy_time, where busy = compute+comm+stall.
                       This is the LOAD-IMBALANCE loss the scheduler exists to shrink.

We report, per rank and aggregate: each bucket in ms and as % of makespan, the efficiency
(= mean_busy / makespan), and the DOMINANT loss bucket. Running the {even, hetero-matched, random}
assignment policies side by side makes the hetero-matched policy's smaller straggler-idle bucket
visible (it balances per-rank busy time across unequal GPUs, so the barrier wait collapses).

Sharding / policy / bandwidth-probe logic mirrors scripts/multi_gpu_train.py; staging mirrors
scripts/oom_to_tiered.py; timing mirrors the warmup+synchronize idiom of multi_gpu_nvlink.py but uses
per-phase torch.cuda.Event so COMPUTE and COMM are isolated on the GPU timeline.

  srun --ntasks=P python scripts/loss_breakdown.py --nodes 8000000 --edges 100000000 --feat 128
  srun --ntasks=P python scripts/loss_breakdown.py --dataset askubuntu --feat 256 --policy hetero-matched
  srun --ntasks=P python scripts/loss_breakdown.py --nodes 4000000 --edges 60000000 --policy all --stage
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

N_GATHERS = 2                                        # 2-layer aggregation = 2 SpMM gathers
POLICIES = ["even", "hetero-matched", "random"]      # the assignment policies we contrast
BUCKETS = ["compute", "comm", "memory_stall", "sync_wait"]


# --------------------------------------------------------------------------- NCCL setup (mirrors multi_gpu_train.py)
def env_int(*keys, default=0):
    for k in keys:
        if k in os.environ:
            return int(os.environ[k])
    return default


def setup():
    rank = env_int("SLURM_PROCID", "RANK")
    world = env_int("SLURM_NTASKS", "WORLD_SIZE", default=1)
    local = env_int("SLURM_LOCALID", "LOCAL_RANK")
    if "MASTER_ADDR" not in os.environ:
        nl = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        if "[" in nl:
            base, rest = nl.split("[", 1)
            nl = f"{base}{rest.split('-', 1)[0].rstrip(']')}"
        os.environ["MASTER_ADDR"] = nl
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world, local


# --------------------------------------------------------------------------- per-phase CUDA-Event timing
def event_time(fn, reps=20, warmup=5):
    """Mean GPU-time (ms) of fn over reps, after warmup. Uses torch.cuda.Event pairs so ONLY the work
    inside fn is timed on the device timeline (not host launch overhead, not other phases). NO barrier
    inside the loop: each rank times its OWN local phase -- that is the per-rank busy time we then
    compare against the barrier-defined makespan to recover the straggler-idle loss."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()                                   # all ranks enter the measured region together
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return float(np.mean([starts[i].elapsed_time(ends[i]) for i in range(reps)]))  # ms


# --------------------------------------------------------------------------- measured per-GPU agg bandwidth (multi_gpu_train.py)
def measure_hbm_bw(dev, mb=256, reps=30):
    """Microbenchmark THIS rank's achieved memory bandwidth (a streaming copy ~ what bounds the
    memory-bound aggregation). GB/s. Lets hetero-matched react to the REAL devices in the job; we do
    NOT assume any tier or link -- heterogeneous GPUs simply measure different bw here."""
    n = (mb * 1024 * 1024) // 4
    a = torch.empty(n, dtype=torch.float32, device=dev)
    b = torch.empty(n, dtype=torch.float32, device=dev)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return (2 * n * 4) / dt / 1e9


# --------------------------------------------------------------------------- graph (deterministic, controllable locality)
def gen_graph(N, M, C, intra, seed=0):
    """Synthetic graph with C communities; `intra` fraction of edges stay inside a node's community
    (so a community-aligned partition has a small cut). Returns int64 src/dst (undirected pairs)."""
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, C, size=N).astype(np.int64)
    order = np.argsort(comm, kind="stable")
    bounds = np.searchsorted(comm[order], np.arange(C + 1))
    m_in = int(M * intra)
    u = rng.integers(0, N, size=m_in)
    cu = comm[u]
    lo = bounds[cu].astype(np.int64)
    hi = bounds[cu + 1].astype(np.int64)
    pick = lo + (rng.random(m_in) * np.maximum(1, hi - lo)).astype(np.int64)
    v = order[np.minimum(pick, N - 1)]
    u2 = rng.integers(0, N, size=M - m_in)
    v2 = rng.integers(0, N, size=M - m_in)
    src = np.concatenate([u, u2]).astype(np.int64)
    dst = np.concatenate([v, v2]).astype(np.int64)
    return src, dst


def node_degree(src, dst, N):
    return (np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)).astype(np.int64)


# --------------------------------------------------------------------------- policy -> contiguous bounds over density order
def density_rank(deg):
    """node id -> density rank (0 = densest)."""
    order = np.argsort(-deg, kind="stable")
    rank_of = np.empty(deg.shape[0], dtype=np.int64)
    rank_of[order] = np.arange(deg.shape[0], dtype=np.int64)
    return rank_of, order


def solve_balanced_bounds(deg_cum, bw, N):
    """hetero-matched: boundaries over the density-sorted order so PREDICTED per-rank agg TIME is
    equalized. time_r ~ local_edges_r / bw_r; give rank r an edge budget ~ bw_r, translate that
    cumulative-degree budget into a node boundary. bw ordered strongest-first (== density order)."""
    D = len(bw)
    share = np.asarray(bw, dtype=np.float64) / np.sum(bw)
    target = share * deg_cum[-1]
    bounds = [0]
    acc = 0.0
    for k in range(D - 1):
        acc += target[k]
        nb = int(np.searchsorted(deg_cum, acc, side="left"))
        nb = max(bounds[-1] + 1, min(nb, N))
        bounds.append(nb)
    bounds.append(N)
    return np.array(bounds, dtype=np.int64)


def policy_bounds(policy, N, P, deg, bw_strong_first, seed):
    """Return (rank_of[v], bounds[P+1]); rank r owns slice [bounds[r], bounds[r+1]) of the layout.
      even           : equal-size contiguous shards over density order (capacity/degree-blind baseline).
      hetero-matched : densest slice -> strongest rank, shard sizes solved so predicted per-rank agg
                       TIME is balanced (the placement zord's scheduler emits).
      random         : node ids permuted then equally sharded (locality-destroying control; high cut)."""
    if policy == "random":
        rng = np.random.default_rng(seed + 12345)
        perm = rng.permutation(N)
        rank_of = np.empty(N, dtype=np.int64)
        rank_of[perm] = np.arange(N, dtype=np.int64)
        bounds = np.linspace(0, N, P + 1).astype(np.int64)
        return rank_of, bounds

    rank_of, _order = density_rank(deg)
    if policy == "even":
        bounds = np.linspace(0, N, P + 1).astype(np.int64)
    elif policy == "hetero-matched":
        deg_by_rank = np.empty(N, dtype=np.float64)
        deg_by_rank[rank_of] = deg.astype(np.float64)
        deg_cum = np.cumsum(deg_by_rank)
        bounds = solve_balanced_bounds(deg_cum, bw_strong_first, N)
    else:
        raise ValueError(f"unknown policy {policy!r}")
    bounds = np.minimum(np.maximum.accumulate(bounds), N).astype(np.int64)
    return rank_of, bounds


# --------------------------------------------------------------------------- deterministic features / count exchange
def _feat_rows(global_ids, F, seed, dev):
    """Deterministic feature rows for given global ids (same on every rank), so gathered remote
    features are consistent -- SAME numerical result regardless of policy."""
    g = global_ids.astype(np.int64)
    base = (g.astype(np.float64) * 0.0009765625) % 1.0
    cols = np.arange(F, dtype=np.float64)
    M = np.sin(base[:, None] * (cols[None, :] + 1.0) + seed)
    return torch.from_numpy(M.astype(np.float32)).to(dev)


def _exchange_counts(recv_counts, world, dev):
    """All-to-all the integer request counts so each rank learns how many rows others want FROM it."""
    recv_t = torch.from_numpy(recv_counts.astype(np.int64)).to(dev)
    send_t = torch.empty_like(recv_t)
    dist.all_to_all_single(send_t, recv_t)
    return send_t.cpu().numpy()


# --------------------------------------------------------------------------- one policy: build shard, time each bucket
def run_policy(policy, rank, world, dev, N, F, src, dst, deg, bw_strong_first, seed, stage):
    """All ranks derive the SAME global bounds (deterministic from deg+bw+seed), then this rank
    materializes its own shard and CUDA-Event-times its compute / comm / memory-stall phases. Returns a
    stats tensor; the caller gathers them and recovers the barrier-defined makespan + sync-wait loss."""
    rank_of, bounds = policy_bounds(policy, N, world, deg, bw_strong_first, seed)
    part = np.searchsorted(bounds, rank_of, side="right") - 1
    part = part.clip(0, world - 1).astype(np.int64)

    my_nodes = np.nonzero(part == rank)[0]
    nl = int(my_nodes.size)
    g2l = np.full(N, -1, dtype=np.int64)
    g2l[my_nodes] = np.arange(nl, dtype=np.int64)

    # undirected: an edge contributes to this rank's local aggregation if its source is local.
    es = np.concatenate([src, dst])
    ed = np.concatenate([dst, src])
    mine = part[es] == rank
    e_src = es[mine]
    e_dst = ed[mine]
    rows = g2l[e_src]

    nbr_part = part[e_dst]
    is_remote = nbr_part != rank
    remote_ids = np.unique(e_dst[is_remote])
    n_remote = int(remote_ids.size)

    # feature buffer layout: [ local shard rows (nl) | remote fetched rows (n_remote) ]
    rid2slot = np.full(N, -1, dtype=np.int64)
    rid2slot[remote_ids] = np.arange(n_remote, dtype=np.int64) + nl
    cols = np.where(is_remote, rid2slot[e_dst], g2l[e_dst])
    nbuf = nl + n_remote

    gen = torch.Generator().manual_seed(1234 + seed)
    Xbuf = torch.empty(max(1, nbuf), F, device=dev)
    if nl:
        Xbuf[:nl] = _feat_rows(my_nodes, F, seed, dev)
    if n_remote:
        Xbuf[nl:nbuf] = _feat_rows(remote_ids, F, seed, dev)

    A = torch.sparse_coo_tensor(
        torch.stack([torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)]),
        torch.ones(rows.size, device=dev),
        (max(1, nl), max(1, nbuf)),
    ).coalesce()
    W1 = torch.randn(F, F, generator=gen, device="cpu").to(dev) / F ** 0.5
    W2 = torch.randn(F, F, generator=gen, device="cpu").to(dev) / F ** 0.5

    # ---- (1) COMPUTE: local 2-layer aggregation (2 SpMM gathers) ----
    def compute():
        h = torch.relu(torch.sparse.mm(A, Xbuf) @ W1)
        h2buf = torch.zeros(max(1, nbuf), F, device=dev)
        h2buf[:nl] = h
        return torch.relu(torch.sparse.mm(A, h2buf) @ W2)
    t_compute = event_time(compute)

    # ---- (2) COMM: boundary exchange -- ship the n_remote feature rows this rank needs (the cut) ----
    remote_owner = part[remote_ids] if n_remote else np.empty(0, dtype=np.int64)
    recv_counts = np.bincount(remote_owner, minlength=world).astype(np.int64)
    send_counts = _exchange_counts(recv_counts, world, dev)
    recv_bufs = [torch.empty(max(1, int(recv_counts[p])), F, device=dev) for p in range(world)]
    send_bufs = [torch.empty(max(1, int(send_counts[p])), F, device=dev) for p in range(world)]

    def boundary():
        reqs = []
        for p in range(world):
            if p == rank:
                continue
            if recv_counts[p] > 0:
                reqs.append(dist.irecv(recv_bufs[p], src=p))
            if send_counts[p] > 0:
                reqs.append(dist.isend(send_bufs[p], dst=p))
        for r in reqs:
            r.wait()
    t_comm = event_time(boundary) if world > 1 else 0.0

    # ---- (3) MEMORY-STALL: if --stage, this rank's feature shard lives in CPU RAM and is copied H2D
    #          each step through a pinned bounce buffer (out-of-core staging). The H2D wait is the
    #          memory-stall loss; 0 when in-core. Mirrors the tiered-blocking path in oom_to_tiered.py.
    if stage and nl:
        cpu_shard = torch.empty(nl, F, pin_memory=True)
        cpu_shard.copy_(Xbuf[:nl].cpu())
        dst_slice = Xbuf[:nl]

        def h2d():
            dst_slice.copy_(cpu_shard.to(dev, non_blocking=False))
        t_stall = event_time(h2d)
    else:
        t_stall = 0.0

    cut = int(is_remote.sum())
    local_e = int((~is_remote).sum())
    return torch.tensor(
        [t_compute, t_comm, t_stall, float(nl), float(local_e), float(cut), float(n_remote),
         float(bw_strong_first[rank] if rank < len(bw_strong_first) else 0.0)],
        device=dev, dtype=torch.float64,
    )


# --------------------------------------------------------------------------- reporting
def report_policy(pol, rows, world):
    """rows[r] = [compute, comm, stall, nl, local_e, cut, n_remote, bw] (seconds-as-ms already in ms).
    Recover the makespan (the step barrier releases when the SLOWEST rank's busy time finishes) and the
    per-rank sync-wait = makespan - busy. Returns the makespan + dominant aggregate bucket."""
    comp = np.array([r[0] for r in rows])              # ms
    comm = np.array([r[1] for r in rows])              # ms
    stall = np.array([r[2] for r in rows])             # ms
    busy = comp + comm + stall                         # per-rank busy time
    makespan = float(busy.max())                       # barrier := slowest rank finishes
    sync_wait = makespan - busy                        # straggler-idle loss per rank (>= 0)

    print(f"  [{pol}]  makespan={makespan:8.2f}ms  efficiency(mean_busy/makespan)={busy.mean()/makespan*100:5.1f}%")
    for r in range(world):
        nl, le, cut, nr, bw = rows[r][3], rows[r][4], rows[r][5], rows[r][6], rows[r][7]
        pc = lambda x: x / makespan * 100.0
        print(f"      rank{r}: nodes={int(nl):>11,d} local_edges={int(le):>12,d} cut={int(cut):>11,d} "
              f"remote_nbrs={int(nr):>9,d} bw={bw:5.0f}GB/s")
        print(f"             compute={comp[r]:8.2f}ms({pc(comp[r]):4.1f}%) "
              f"comm={comm[r]:7.2f}ms({pc(comm[r]):4.1f}%) "
              f"mem_stall={stall[r]:7.2f}ms({pc(stall[r]):4.1f}%) "
              f"SYNC_WAIT={sync_wait[r]:8.2f}ms({pc(sync_wait[r]):4.1f}%)")

    agg = {
        "compute": comp.sum(), "comm": comm.sum(),
        "memory_stall": stall.sum(), "sync_wait": sync_wait.sum(),
    }
    total = sum(agg.values()) or 1.0
    dom = max(agg, key=agg.get)
    print("      aggregate over ranks: " +
          "  ".join(f"{b}={agg[b]:7.2f}ms({agg[b]/total*100:4.1f}%)" for b in BUCKETS) +
          f"  | DOMINANT LOSS = {dom}")
    return makespan, dom, busy.mean() / makespan


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8_000_000)
    ap.add_argument("--edges", type=int, default=100_000_000)
    ap.add_argument("--comms", type=int, default=4000)
    ap.add_argument("--intra", type=float, default=0.9,
                    help="frac of edges kept inside a node's community (intra-shard locality)")
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--dataset", default="", help="real temporal graph name (else synthetic)")
    ap.add_argument("--policy", default="all", choices=POLICIES + ["all"])
    ap.add_argument("--stage", action="store_true",
                    help="out-of-core: keep each rank's feature shard in CPU RAM, copy H2D per step "
                         "(turns on the MEMORY-STALL bucket; default in-core -> stall=0)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rank, world, local = setup()
    dev = f"cuda:{local}"
    F = a.feat
    policies = POLICIES if a.policy == "all" else [a.policy]

    # ---- MEASURE each rank's real aggregation bandwidth (hardware-agnostic; whatever GPU it is) ----
    my_bw = measure_hbm_bw(dev)
    bw_dev = torch.tensor([my_bw], device=dev, dtype=torch.float64)
    bw_all = [torch.empty(1, device=dev, dtype=torch.float64) for _ in range(world)]
    dist.all_gather(bw_all, bw_dev)
    bw_by_rank = np.array([b.item() for b in bw_all], dtype=np.float64)
    bw_strong_first = bw_by_rank                       # ranks assumed launched strong->weak (else still valid)

    # ---- build the SAME graph on every rank (deterministic) ----
    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes
        src = g.src.astype(np.int64)
        dst = g.dst.astype(np.int64)
        M = src.size
        gname = g.name
    else:
        N, M = a.nodes, a.edges
        src, dst = gen_graph(N, M, a.comms, a.intra, seed=a.seed)
        gname = f"synthetic(comms={a.comms},intra={a.intra})"
    deg = node_degree(src, dst, N)
    if rank == 0:
        print(f"LOSS-BREAKDOWN gpu='{torch.cuda.get_device_name(0)}' world={world} dataset={gname} "
              f"N={N} M={M} F={F} seed={a.seed} stage={a.stage} graph_build={time.time()-t0:.1f}s")
        print("  measured agg bw by rank (hardware just as measured): " +
              " ".join(f"r{r}={bw_by_rank[r]:.0f}GB/s" for r in range(world)) +
              ("  [heterogeneous]" if bw_by_rank.max() / max(bw_by_rank.min(), 1e-9) > 1.15 else "  [homogeneous]"))
        print("  buckets: (1)COMPUTE local SpMM  (2)COMM boundary exchange  (3)MEMORY-STALL H2D  "
              "(4)SYNC-WAIT = makespan - busy (load-imbalance/straggler-idle)\n")

    # ---- run each policy, gather per-rank stats, decompose at rank 0 ----
    results = {}
    for pol in policies:
        stats = run_policy(pol, rank, world, dev, N, F, src, dst, deg, bw_strong_first, a.seed, a.stage)
        gathered = [torch.empty_like(stats) for _ in range(world)]
        dist.all_gather(gathered, stats)
        dist.barrier()
        if rank == 0:
            results[pol] = [g.cpu().numpy() for g in gathered]

    if rank == 0:
        print("  per-policy LOSS BREAKDOWN (ms + % of makespan):")
        summary = {}
        for pol in policies:
            mk, dom, eff = report_policy(pol, results[pol], world)
            summary[pol] = (mk, dom, eff)
            print()

        print("  ==== SUMMARY: makespan / efficiency / dominant loss bucket ====")
        for pol in policies:
            mk, dom, eff = summary[pol]
            print(f"    {pol:<16} makespan={mk:8.2f}ms  efficiency={eff*100:5.1f}%  dominant_loss={dom}")
        if "hetero-matched" in summary and "even" in summary:
            hm = summary["hetero-matched"][0]
            ev = summary["even"][0]
            print(f"    => hetero-matched makespan is {ev/hm:5.2f}x of/under even; the saving is the "
                  f"SHRUNK SYNC-WAIT (straggler-idle) bucket -- balancing per-rank busy time across "
                  f"unequal GPUs collapses the barrier wait.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
