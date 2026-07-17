#!/usr/bin/env python
"""End-to-end 2-node measured iteration time: does zord's lower cross-partition edge
count translate into lower REAL per-step time (cross-rank feature exchange over gloo +
local GraphSAGE), beyond the modeled makespan? Compares zord-placement vs hash-placement.

Launch via SLURM: 2 nodes, 1 GPU each (srun --ntasks=2). gloo float32 only.
  python scripts/end_to_end.py wiki-talk --scheme zord --feat-dim 256 --steps 20
"""
import argparse, os, time
import numpy as np
import torch
import torch.distributed as dist

from zord.datasets import load
from zord.profiler import ClusterProfile, DeviceProfile, GB
from zord.partition import ZordPartitioner, HashPartitioner
from zord.partition.cost_model import CostParams, max_nodes_per_device


def setup():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    if "MASTER_ADDR" not in os.environ:
        nl = os.environ.get("SLURM_NODELIST", "")
        if "[" in nl:
            base, rest = nl.split("[", 1); nl = f"{base}{rest.split('-',1)[0].rstrip(']')}"
        os.environ["MASTER_ADDR"] = nl or "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group("gloo")
    return rank, world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--scheme", required=True, choices=["zord", "hash"])
    ap.add_argument("--feat-dim", type=int, default=256); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--mem-gb", type=float, default=32.0,
                    help="per-device memory (GB); set small to FORCE zord to spread (balanced regime)")
    a = ap.parse_args()
    rank, world = setup()
    assert world == 2, "this benchmark expects exactly 2 ranks (2 nodes)"
    F = a.feat_dim
    g = load(a.dataset).sort_by_time()       # both ranks load the same graph (deterministic)
    # 2-device cluster, one GPU per NODE (so cross-rank = inter-node gloo @ measured 0.12 GB/s)
    mb = int(a.mem_gb * GB); rsv = int(0.1 * mb)
    c = ClusterProfile([DeviceProfile(0, "gpu0", mb, 1.0, node=0, mem_reserved=rsv),
                        DeviceProfile(1, "gpu1", mb, 1.0, node=1, mem_reserved=rsv)])
    cp = CostParams(feat_dim=F, window=1)
    if a.scheme == "zord":
        cap = max_nodes_per_device(c, cp, avg_degree=g.num_edges / max(g.num_nodes, 1))
        part = ZordPartitioner().partition(g.src, g.dst, g.num_nodes, c, capacity=cap)
    else:
        part = HashPartitioner().partition(g.src, g.dst, g.num_nodes, c)
    assign = part.assignment
    # cross-partition edges -> boundary nodes each rank must SEND to the other
    cross = assign[g.src] != assign[g.dst]
    send_nodes = np.unique(np.concatenate([g.src[cross], g.dst[cross]])[
        (assign[np.concatenate([g.src[cross], g.dst[cross]])] == rank)])
    k_send = int(send_nodes.shape[0])
    other = 1 - rank
    # exchange sizes so each rank can size its recv buffer
    sizes = [torch.zeros(1, dtype=torch.int64) for _ in range(2)]
    dist.all_gather(sizes, torch.tensor([k_send], dtype=torch.int64))
    k_recv = int(sizes[other].item())
    if rank == 0:
        print(f"E2E scheme={a.scheme} cross_edges={part.total_cross_edges} "
              f"send0={k_send} recv0={k_recv} nodes/dev={part.nodes_per_device.tolist()}", flush=True)

    dev = "cuda:0"
    sendbuf = torch.ones(max(k_send, 1) * F, dtype=torch.float32)
    recvbuf = torch.empty(max(k_recv, 1) * F, dtype=torch.float32)
    # tiny local GraphSAGE proxy on this rank's local nodes
    nloc = int((assign == rank).sum())
    X = torch.randn(max(nloc, 1), F, device=dev); W = torch.randn(F, F, device=dev)

    def step():
        # cross-rank boundary feature exchange (the comm zord reduces)
        if rank == 0:
            if k_send: dist.send(sendbuf, dst=1)
            if k_recv: dist.recv(recvbuf, src=1)
        else:
            if k_recv: dist.recv(recvbuf, src=0)
            if k_send: dist.send(sendbuf, dst=0)
        _ = (X @ W); torch.cuda.synchronize()      # local compute

    for _ in range(3):
        step()
    dist.barrier(); t0 = time.time()
    for _ in range(a.steps):
        step()
    dist.barrier(); dt = (time.time() - t0) / a.steps
    if rank == 0:
        print(f"E2E_RESULT scheme={a.scheme} cross_edges={part.total_cross_edges} "
              f"step_ms={dt*1e3:.1f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
