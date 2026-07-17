#!/usr/bin/env python
"""LIVE GPU-OOM probe: actually allocate (on a real GPU) the feature tensor that
one device's partition would need, under a given partitioner, and observe real
CUDA OOM vs success. Turns zord's *analytical* G1 (no-OOM) into a *measured* one.

Run inside a SLURM job pinned to the GPU tier you want to test (e.g. rtx_5000=32GB,
which is device index 2 in hetcluster(1,1,1)):
  python scripts/oom_probe.py wiki-talk --scheme hash --device 2 --feat-dim 1024 --window 24
  python scripts/oom_probe.py wiki-talk --scheme zord --device 2 --feat-dim 1024 --window 24
"""
import argparse
import torch

from zord.datasets import load
from zord.profiler import hetcluster
from zord.partition import PARTITIONERS
from zord.partition.cost_model import CostParams, max_nodes_per_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--scheme", required=True, choices=list(PARTITIONERS))
    ap.add_argument("--device", type=int, default=2, help="which logical device's partition to materialize")
    ap.add_argument("--feat-dim", type=int, default=1024)
    ap.add_argument("--window", type=int, default=24)
    a = ap.parse_args()

    g = load(a.dataset).sort_by_time()
    c = hetcluster(1, 1, 1)                       # H100 / RTX6000 / RTX5000  (idx 0/1/2)
    cp = CostParams(feat_dim=a.feat_dim, window=a.window)
    kw = {"capacity": max_nodes_per_device(c, cp, avg_degree=g.num_edges / max(g.num_nodes, 1))} if a.scheme == "zord" else {}
    part = PARTITIONERS[a.scheme]().partition(g.src, g.dst, g.num_nodes, c, **kw)

    n_k = int(part.nodes_per_device[a.device])
    want_gb = n_k * a.window * a.feat_dim * 4 / 1e9
    gpu = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"PROBE dataset={g.name} scheme={a.scheme} logical_dev={a.device} "
          f"nodes_on_dev={n_k:,} want={want_gb:.1f}GB on {gpu} ({total_gb:.0f}GB)")
    try:
        # the feature working set this device must hold (window x feat per node)
        feat = torch.empty((n_k * a.window, a.feat_dim), dtype=torch.float32, device="cuda:0")
        feat.fill_(1.0); torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"RESULT scheme={a.scheme} -> OK   (allocated {want_gb:.1f}GB, peak {peak:.1f}GB)")
    except RuntimeError as e:
        msg = str(e).splitlines()[0][:90]
        print(f"RESULT scheme={a.scheme} -> CUDA_OOM ({want_gb:.1f}GB needed on {total_gb:.0f}GB): {msg}")


if __name__ == "__main__":
    main()
