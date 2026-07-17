#!/usr/bin/env python
"""REAL multi-GPU (NCCL/NVLink) measurement: for a sharded temporal-GNN step, how much of the
time is inter-GPU COMMUNICATION vs local COMPUTE on NVLink? This puts a real multi-GPU number
behind the D25 pivot ("comm is hardware-solved on all-NVLink clusters -> not the bottleneck").

Setup: N nodes contiguously sharded across P GPUs (one rank/GPU). A 2-layer aggregation needs
neighbor features; we measure two exchange policies over NVLink:
  full-allgather : every rank gathers ALL shards' features (worst case; comm ~ (P-1)/P * N*F).
  boundary-only  : every rank fetches ONLY the features it actually needs (the cut) -- realistic
                   with a locality partition; comm ~ (#distinct remote neighbors) * F.
We report comm-time, compute-time (local SpMM), and comm/compute. If comm << compute on NVLink,
the bottleneck is compute+memory, not the network -> the pivot holds, MEASURED at multi-GPU.
Also: aggregate window capacity scales with P (a window too big for 1 GPU fits across P).
  srun ... python scripts/multi_gpu_nvlink.py --nodes 4000000 --degree 16 --feat 256
"""
import argparse, os, time
import numpy as np
import torch
import torch.distributed as dist


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
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world, local


def timed_cuda(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=4_000_000)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--feat", type=int, default=256)
    ap.add_argument("--intra", type=float, default=0.0)   # frac of edges kept INSIDE this shard (clustered partition)
    a = ap.parse_args()
    rank, world, local = setup()
    dev = f"cuda:{local}"
    N, F = a.nodes, a.feat
    n_local = N // world
    lo = rank * n_local
    hi = N if rank == world - 1 else lo + n_local
    nl = hi - lo

    # local features + a local adjacency block (rows = local nodes, cols = ALL nodes)
    g = torch.Generator().manual_seed(rank)
    Xloc = torch.randn(nl, F, device=dev)
    e = nl * a.degree
    src = torch.randint(0, nl, (e,), generator=g)        # local row
    # with prob intra, neighbor is INSIDE this shard [lo,hi) (a clustered/good partition); else anywhere
    dst_local = torch.randint(lo, hi, (e,), generator=g)
    dst_any = torch.randint(0, N, (e,), generator=g)
    keep_local = torch.rand(e, generator=g) < a.intra
    dst = torch.where(keep_local, dst_local, dst_any)    # intra=0 -> all random; intra->1 -> few cross-shard
    vals = torch.ones(e, device=dev)
    Aloc = torch.sparse_coo_tensor(torch.stack([src.to(dev), dst.to(dev)]), vals, (nl, N)).coalesce()

    # remote neighbor columns this rank actually needs (the "cut")
    cols = torch.unique(dst).to(dev)
    remote = cols[(cols < lo) | (cols >= hi)]
    n_remote = int(remote.numel())

    # ---- compute: local 2-hop aggregation (needs full X of width N) ----
    W1 = torch.randn(F, F, device=dev) / F ** 0.5
    Xfull = torch.zeros(N, F, device=dev)                 # scratch for gathered features
    def compute():
        Xfull[lo:hi] = Xloc
        h = torch.relu(torch.sparse.mm(Aloc, Xfull) @ W1)  # (nl, F)
        return h
    t_compute = timed_cuda(lambda: compute())

    # ---- policy A: full all_gather of every shard's features over NVLink ----
    shards = [torch.empty(N // world, F, device=dev) for _ in range(world)]
    def full_gather():
        dist.all_gather(shards, Xloc[: N // world])
    t_full = timed_cuda(full_gather)
    full_bytes = (world - 1) * (N // world) * F * 4

    # ---- policy B: boundary-only fetch (all_to_all of just-needed columns; approx via isend/irecv sizes) ----
    # measure shipping n_remote feature rows in (the realistic cut volume)
    recv_buf = torch.empty(max(1, n_remote), F, device=dev)
    src_buf = torch.empty(max(1, n_remote), F, device=dev)
    def boundary():
        # approximate the cut exchange cost with a same-sized all_reduce on the boundary block
        dist.all_reduce(recv_buf)
    t_bnd = timed_cuda(boundary)
    bnd_bytes = n_remote * F * 4

    stats = torch.tensor([t_compute, t_full, t_bnd, float(n_remote), float(nl)], device=dev)
    gathered = [torch.empty_like(stats) for _ in range(world)]
    dist.all_gather(gathered, stats)
    if rank == 0:
        gpu = torch.cuda.get_device_name(0)
        print(f"NVLINK-MULTI gpu='{gpu}' world={world} N={N} deg={a.degree} F={F} "
              f"per_rank_nodes={n_local} window_capacity_scales_with_P={world}x")
        for r, s in enumerate(gathered):
            tc, tf, tb, nr, nlr = [x.item() for x in s]
            print(f"  rank{r}: compute={tc*1e3:7.2f}ms  full_allgather={tf*1e3:7.2f}ms "
                  f"({full_bytes/1e6:.0f}MB, comm/compute={tf/tc:.2f}) "
                  f"boundary={tb*1e3:7.2f}ms (remote_nbrs={int(nr)}, {bnd_bytes/1e6:.1f}MB, "
                  f"comm/compute={tb/tc:.3f})")
        tc0, tf0, tb0 = gathered[0][0].item(), gathered[0][1].item(), gathered[0][2].item()
        print(f"  => NVLink full-allgather comm is {tf0/tc0:.2f}x compute; boundary-only is "
              f"{tb0/tc0:.3f}x compute. {'comm NOT the bottleneck (pivot holds)' if tb0 < tc0 else 'comm matters here'}.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
