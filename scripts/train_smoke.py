#!/usr/bin/env python
"""Minimal end-to-end runtime smoke: a real full-batch GraphSAGE (mean-aggregator,
sparse A@X) forward+backward over a snapshot-batch, on one GPU. Proves the
partition->place->train path runs, and MEASURES real per-step time + peak memory.
Run on each tier (h100 / rtx_6000 / rtx_5000) to get the REAL GNN-kernel throughput
ratio (refines the profiler, whose fp32-matmul r_k understated the H100).

  python scripts/train_smoke.py askubuntu --feat-dim 128 --layers 2 --steps 30
"""
import argparse, json, os, time
import torch

from zord.datasets import load


def build_norm_adj(src, dst, n, device):
    import numpy as np
    i = torch.tensor(np.concatenate([src, dst]), dtype=torch.long)
    j = torch.tensor(np.concatenate([dst, src]), dtype=torch.long)
    idx = torch.stack([i, j])
    vals = torch.ones(idx.shape[1])
    A = torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce().to(device)
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    rows = A.indices()[0]
    Aval = A.values() / deg[rows]                  # row-normalized mean aggregator
    return torch.sparse_coo_tensor(A.indices(), Aval, (n, n)).coalesce()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda:0"
    g = load(a.dataset).sort_by_time()
    n, F = g.num_nodes, a.feat_dim
    A = build_norm_adj(g.src, g.dst, n, dev)
    X = torch.randn(n, F, device=dev)
    Ws = [torch.randn(F, F, device=dev, requires_grad=True) for _ in range(a.layers)]
    opt = torch.optim.Adam(Ws, lr=1e-3)

    def step():
        opt.zero_grad()
        h = X
        for k, W in enumerate(Ws):
            h = torch.sparse.mm(A, h) @ W
            if k < len(Ws) - 1:
                h = torch.relu(h)
        loss = h.pow(2).mean()
        loss.backward(); opt.step()
        return float(loss)

    for _ in range(3):           # warmup (kernels, allocator)
        step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / a.steps
    peak = torch.cuda.max_memory_allocated() / 1e9
    gpu = torch.cuda.get_device_name(0)
    out = {"gpu": gpu, "dataset": g.name, "nodes": n, "edges": g.num_edges,
           "feat_dim": F, "layers": a.layers, "step_ms": round(dt * 1e3, 2),
           "peak_gb": round(peak, 2), "jobid": os.environ.get("SLURM_JOB_ID", "local")}
    print("TRAIN_SMOKE", json.dumps(out))
    p = a.out or f"$ZORD_DATA/results/train_{gpu.replace(' ', '_')}_{out['jobid']}.json"
    try:
        json.dump(out, open(p, "w"), indent=2); print("wrote", p)
    except Exception as e:
        print("(no out file)", e)


if __name__ == "__main__":
    main()
