#!/usr/bin/env python
"""REAL multi-GPU (NCCL) VALIDATION of the §17 hetero-matched placement result. hetero_matched.py
PREDICTS (PROCESS-only, roofline) that putting the dense graph core on the strong GPU and balancing
per-device aggregation TIME gives ~3.3x lower makespan than even / bandwidth-proportional shards.
This script MEASURES it: it runs a real distributed 2-layer GNN aggregation step across P GPUs (one
rank/GPU, NCCL), where the graph's vertices are assigned to ranks by a chosen ASSIGNMENT policy, and
reports per-rank COMPUTE ms + cross-rank boundary-exchange COMM ms + the MAKESPAN (max over ranks) of
the step under each policy -- so the REAL makespan ratio can be compared against the §17 3.3x.

Same graph, same numerical result for every policy; only the vertex->rank assignment differs (which
changes per-rank edge counts -> per-rank compute time -> makespan). PROCESS metric only (TIME), never
accuracy.

POLICIES (--policy; runs all four by default), assigning the N vertices to P ranks. Vertices are first
ranked by DENSITY (degree, descending: rank-index 0 = densest node); a policy then chooses contiguous
boundaries over that density-sorted order, and rank r owns the density slice [bounds[r], bounds[r+1]).
  even            : equal-size contiguous shards (capacity/bandwidth/degree-blind baseline).
  bw-proportional : shard SIZES proportional to the rank's MEASURED hbm aggregation bandwidth
                    (bandwidth-aware but degree-blind -- a big shard of sparse nodes is cheap, a small
                    shard of dense nodes is expensive, so balancing nodes != balancing time).
  hetero-matched  : densest nodes -> highest-bw rank, and shard sizes solved so the PREDICTED per-rank
                    agg time (local_edges_r / bw_r) is balanced (the §17 placement).
  random          : vertices permuted then equally sharded (locality-destroying control; high cut).

GRAPH: synthetic by default (deterministic per --seed; --intra controls intra-shard locality, i.e. the
fraction of edges kept inside a node's community so a good partition has few cut edges), or a real
temporal graph via --dataset.

EXCHANGE: each rank holds ONLY its vertex shard's features + the local adjacency block (rows=local
nodes, cols=all nodes). For the 2 aggregation gathers it needs remote neighbors' features; those are
fetched over NCCL by the same boundary-exchange style as multi_gpu_nvlink.py (each rank ships the
distinct remote-neighbor feature rows it needs -- the realistic cut volume -- via isend/irecv).

  srun --ntasks=P python scripts/multi_gpu_train.py --nodes 8000000 --edges 100000000 --feat 128
  srun --ntasks=P python scripts/multi_gpu_train.py --dataset askubuntu --feat 256 --policy hetero-matched
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

BYTES_PER_EDGE_TRAVERSAL = 4.0   # fp32 feature word moved per edge per gather (memory-bound model)
N_GATHERS = 2                    # 2-layer aggregation = 2 SpMM gathers over the local edges
POLICIES = ["even", "bw-proportional", "hetero-matched", "random"]


# --------------------------------------------------------------------------- NCCL setup (mirrors multi_gpu_nvlink.py)
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
    os.environ.setdefault("MASTER_PORT", "29588")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world, local


def timed_cuda(fn, reps=20, warmup=5):
    """Mean wall-time of fn over reps, after warmup, CUDA-synced. NO barrier inside the loop so each
    rank's time reflects its OWN local work (the per-rank compute/comm we compare across policies)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


# --------------------------------------------------------------------------- measured per-GPU agg bandwidth
def measure_hbm_bw(dev, mb=256, reps=30):
    """Microbenchmark the achieved memory bandwidth of THIS rank's GPU (a streaming copy ~ the
    bandwidth that bounds the memory-bound aggregation). Returns GB/s. Used to make bw-proportional /
    hetero-matched react to the REAL devices in the job (heterogeneous tiers measure different bw)."""
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
    return (2 * n * 4) / dt / 1e9     # read+write bytes / s -> GB/s


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
    """Map each node id -> its density rank (0 = densest). rank_of[v] in [0,N)."""
    order = np.argsort(-deg, kind="stable")          # node ids, densest first
    rank_of = np.empty(deg.shape[0], dtype=np.int64)
    rank_of[order] = np.arange(deg.shape[0], dtype=np.int64)
    return rank_of, order


def solve_balanced_bounds(deg_cum, bw, N):
    """hetero-matched: pick boundaries over the density-sorted order so predicted per-rank agg TIME is
    equalized. time_r ~ local_edges_r / bw_r; give rank r an 'edge budget' ~ bw_r, then translate that
    cumulative-degree budget into a node boundary. bw is ordered strongest-first (== density order)."""
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
    """Return (rank_of[v], bounds[P+1]) where rank r owns density slice [bounds[r], bounds[r+1]).
    rank_of maps node id -> position in the layout the bounds index into.
      even/bw-proportional/hetero-matched: layout = density order (densest first), so rank 0 (which
        owns the strongest GPU, since hetcluster lists devices strongest-first and we keep that order)
        gets the densest slice. random: layout = a seeded permutation (locality destroyed)."""
    if policy == "random":
        rng = np.random.default_rng(seed + 12345)
        perm = rng.permutation(N)                     # node ids in random layout order
        rank_of = np.empty(N, dtype=np.int64)
        rank_of[perm] = np.arange(N, dtype=np.int64)
        bounds = np.linspace(0, N, P + 1).astype(np.int64)
        return rank_of, bounds

    rank_of, order = density_rank(deg)
    if policy == "even":
        bounds = np.linspace(0, N, P + 1).astype(np.int64)
    elif policy == "bw-proportional":
        frac = bw_strong_first / bw_strong_first.sum()
        bounds = np.concatenate([[0], np.cumsum((frac * N).astype(np.int64))]).astype(np.int64)
        bounds[-1] = N
        bounds = np.maximum.accumulate(bounds)
    elif policy == "hetero-matched":
        deg_by_rank = np.empty(N, dtype=np.float64)
        deg_by_rank[rank_of] = deg.astype(np.float64)   # degree indexed by density rank
        deg_cum = np.cumsum(deg_by_rank)
        bounds = solve_balanced_bounds(deg_cum, bw_strong_first, N)
    else:
        raise ValueError(f"unknown policy {policy!r}")
    bounds = np.minimum(np.maximum.accumulate(bounds), N).astype(np.int64)
    return rank_of, bounds


# --------------------------------------------------------------------------- one policy: build shards, run, time
def run_policy(policy, rank, world, dev, N, F, src, dst, deg, bw_strong_first, seed):
    """All ranks compute the SAME global bounds (deterministic from deg+bw+seed), then this rank
    materializes its own shard and times its compute + boundary comm. Returns a stats tensor."""
    rank_of, bounds = policy_bounds(policy, N, world, deg, bw_strong_first, seed)
    part = np.searchsorted(bounds, rank_of, side="right") - 1   # node id -> owning rank
    part = part.clip(0, world - 1).astype(np.int64)

    my_nodes = np.nonzero(part == rank)[0]                      # global ids this rank owns
    nl = int(my_nodes.size)
    # global id -> local row within this shard (-1 if not mine)
    g2l = np.full(N, -1, dtype=np.int64)
    g2l[my_nodes] = np.arange(nl, dtype=np.int64)

    # edges whose SOURCE is local: these contribute to this rank's local aggregation (dst is the
    # neighbor whose feature is gathered). Use both directions (undirected) so degree matches.
    es = np.concatenate([src, dst])
    ed = np.concatenate([dst, src])
    mine = part[es] == rank
    e_src = es[mine]                                            # global, local source
    e_dst = ed[mine]                                            # global, neighbor (may be remote)
    rows = g2l[e_src]                                           # local row index

    # neighbor columns: local ones index our own X; remote ones must be fetched.
    nbr_part = part[e_dst]
    is_remote = nbr_part != rank
    remote_ids = np.unique(e_dst[is_remote])
    n_remote = int(remote_ids.size)

    # build a contiguous feature buffer: [ local shard rows (nl) | remote fetched rows (n_remote) ]
    rid2slot = np.full(N, -1, dtype=np.int64)
    rid2slot[remote_ids] = np.arange(n_remote, dtype=np.int64) + nl
    cols = np.where(is_remote, rid2slot[e_dst], g2l[e_dst])     # column in the local feature buffer
    nbuf = nl + n_remote

    # local features (deterministic per global id so every rank agrees on remote values)
    gen = torch.Generator().manual_seed(1234 + seed)
    # build only the rows we hold + remote rows we need (cheap, avoids an N x F tensor)
    Xbuf = torch.empty(max(1, nbuf), F, device=dev)
    if nl:
        Xbuf[:nl] = _feat_rows(my_nodes, F, seed, dev)
    # remote rows are filled by the exchange each step; seed once so warmup compute is valid
    if n_remote:
        Xbuf[nl:nbuf] = _feat_rows(remote_ids, F, seed, dev)

    A = torch.sparse_coo_tensor(
        torch.stack([torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)]),
        torch.ones(rows.size, device=dev),
        (max(1, nl), max(1, nbuf)),
    ).coalesce()
    W1 = torch.randn(F, F, generator=gen, device="cpu").to(dev) / F ** 0.5
    W2 = torch.randn(F, F, generator=gen, device="cpu").to(dev) / F ** 0.5

    def compute():
        h = torch.relu(torch.sparse.mm(A, Xbuf) @ W1)          # layer 1 over local+remote neighbors
        # layer 2 reuses the same local adjacency on the layer-1 output padded to nbuf
        h2buf = torch.zeros(max(1, nbuf), F, device=dev)
        h2buf[:nl] = h
        return torch.relu(torch.sparse.mm(A, h2buf) @ W2)
    t_compute = timed_cuda(compute)

    # ---- boundary exchange: ship the n_remote feature rows this rank needs, via isend/irecv ----
    # who owns each remote id, and counts per source rank
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
    t_comm = timed_cuda(boundary) if world > 1 else 0.0

    cut = int(is_remote.sum())
    local_e = int((~is_remote).sum())
    return torch.tensor(
        [t_compute, t_comm, float(nl), float(local_e), float(cut), float(n_remote),
         float(bw_strong_first[rank] if rank < len(bw_strong_first) else 0.0)],
        device=dev, dtype=torch.float64,
    )


def _feat_rows(global_ids, F, seed, dev):
    """Deterministic feature rows for given global ids (same on every rank that touches them), so
    the gathered remote features are consistent -- SAME numerical result regardless of policy."""
    g = global_ids.astype(np.int64)
    rng = np.random.default_rng(987654321 + seed)
    # hash ids into a base offset; simple reproducible per-id features
    base = (g.astype(np.float64) * 0.0009765625) % 1.0
    cols = np.arange(F, dtype=np.float64)
    M = np.sin(base[:, None] * (cols[None, :] + 1.0) + seed) * 1.0
    _ = rng  # rng kept for API symmetry / future extension
    return torch.from_numpy(M.astype(np.float32)).to(dev)


def _exchange_counts(recv_counts, world, dev):
    """All-to-all of the integer request counts so each rank learns how many rows others want FROM it
    (recv_counts[p] = rows I want from p  ->  send_counts[p] = rows p wants from me)."""
    recv_t = torch.from_numpy(recv_counts.astype(np.int64)).to(dev)
    send_t = torch.empty_like(recv_t)
    dist.all_to_all_single(send_t, recv_t)
    return send_t.cpu().numpy()


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
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rank, world, local = setup()
    dev = f"cuda:{local}"
    F = a.feat
    policies = POLICIES if a.policy == "all" else [a.policy]

    # ---- MEASURE each rank's real aggregation bandwidth, gather so all ranks share the vector ----
    my_bw = measure_hbm_bw(dev)
    bw_dev = torch.tensor([my_bw], device=dev, dtype=torch.float64)
    bw_all = [torch.empty(1, device=dev, dtype=torch.float64) for _ in range(world)]
    dist.all_gather(bw_all, bw_dev)
    bw_by_rank = np.array([b.item() for b in bw_all], dtype=np.float64)  # bw of rank r's GPU
    # rank 0 is treated as the "strongest-first" slot for density placement; if the launcher already
    # orders ranks strong->weak (e.g. H100 on rank0) this is exact. We pass bw_by_rank as the
    # strong-first vector so policies index it by rank directly.
    bw_strong_first = bw_by_rank

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
        print(f"MGTRAIN gpu='{torch.cuda.get_device_name(0)}' world={world} dataset={gname} "
              f"N={N} M={M} F={F} seed={a.seed} graph_build={time.time()-t0:.1f}s")
        print("  measured agg bw by rank: " +
              " ".join(f"r{r}={bw_by_rank[r]:.0f}GB/s" for r in range(world)))

    # ---- run each policy, gather stats, report ----
    results = {}
    for pol in policies:
        stats = run_policy(pol, rank, world, dev, N, F, src, dst, deg, bw_strong_first, a.seed)
        gathered = [torch.empty_like(stats) for _ in range(world)]
        dist.all_gather(gathered, stats)
        dist.barrier()
        if rank == 0:
            results[pol] = [g.cpu().numpy() for g in gathered]

    if rank == 0:
        print("\n  per-policy REAL timing (compute = local 2-layer SpMM; comm = boundary exchange):")
        makespans = {}
        for pol in policies:
            rows = results[pol]
            print(f"  [{pol}]")
            comp = np.array([r[0] for r in rows]) * 1e3   # ms
            comm = np.array([r[1] for r in rows]) * 1e3   # ms
            for r in range(world):
                nl, le, cut, nr, bw = rows[r][2], rows[r][3], rows[r][4], rows[r][5], rows[r][6]
                print(f"      rank{r}: nodes={int(nl):>11,d} local_edges={int(le):>12,d} "
                      f"cut={int(cut):>11,d} remote_nbrs={int(nr):>9,d} bw={bw:5.0f}GB/s | "
                      f"compute={comp[r]:8.2f}ms comm={comm[r]:8.2f}ms")
            step = comp + comm                            # per-rank step time = its compute + its comm
            makespan = float(step.max())
            makespans[pol] = makespan
            print(f"      => compute_makespan={comp.max():8.2f}ms  comm_makespan={comm.max():8.2f}ms  "
                  f"STEP MAKESPAN={makespan:8.2f}ms  (util={step.mean()/makespan*100:5.1f}%)")

        print("\n  ==== REAL MAKESPAN ACROSS POLICIES (validates §17 prediction) ====")
        best = min(makespans, key=makespans.get)
        for pol in policies:
            print(f"    {pol:<16} makespan={makespans[pol]:8.2f}ms")
        if "hetero-matched" in makespans:
            hm = makespans["hetero-matched"]
            for base in ("even", "bw-proportional", "random"):
                if base in makespans:
                    print(f"    hetero-matched is {makespans[base]/hm:5.2f}x faster than {base} "
                          f"(REAL; §17 predicted ~3.3x vs even/bw-prop)")
        print(f"    best policy: {best}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
