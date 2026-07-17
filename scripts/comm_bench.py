#!/usr/bin/env python
"""2-node gloo point-to-point bandwidth microbench -> calibrates the inter-node
bandwidth constant in ClusterProfile (currently assumed ~1.25 GB/s). This is the
'cost of cutting across a slow link' that zord's bandwidth-weighted partitioner
optimizes; measuring it makes the comm model data-backed (R1).
Launch via SLURM: srun (2 tasks, 1 per node). gloo float32 only (per CLUSTER notes).
"""
import os, time, json
import torch
import torch.distributed as dist


def setup():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    if "MASTER_ADDR" not in os.environ:
        nl = os.environ.get("SLURM_NODELIST", "")
        if "[" in nl:
            base, rest = nl.split("[", 1)
            nl = f"{base}{rest.split('-', 1)[0].rstrip(']')}"
        os.environ["MASTER_ADDR"] = nl or "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ["RANK"] = str(rank); os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group("gloo")
    return rank, world


def main():
    rank, world = setup()
    if rank == 0:
        print(f"COMMBENCH world={world} master={os.environ['MASTER_ADDR']}")
    for mb in [1, 10, 50, 200]:
        n = mb * 1024 * 1024 // 4
        t = torch.ones(n, dtype=torch.float32)      # gloo: float32 only
        dist.barrier()
        t0 = time.time()
        for _ in range(5):
            if rank == 0:
                dist.send(t, dst=1)
            elif rank == 1:
                dist.recv(t, src=0)
        dist.barrier()
        dt = (time.time() - t0) / 5
        if rank == 0:
            gbps = (n * 4) / dt / 1e9
            print(f"COMMBENCH {mb:>4}MB send/recv: {dt*1e3:7.1f} ms -> {gbps:5.2f} GB/s")
    if rank == 0:
        print("COMMBENCH DONE")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
