#!/usr/bin/env python
"""COMM-FREE single-GPU roofline: is a temporal-GNN step HBM-BANDWIDTH-bound (aggregation)
or COMPUTE-bound (dense matmul)? This drives zord's NON-comm thesis: on an all-NVLink cluster
the binding cost is the MEMORY SYSTEM (HBM capacity + HBM bandwidth + PCIe staging), not the
network. We decompose one GraphSAGE layer into:
  - AGG  = sparse.mm(A, X)   -- gathers neighbor features; bytes ~ (nnz + N*F); bandwidth-bound
  - DENSE= X @ W             -- FLOPs ~ N*F^2; tensor-core / compute-bound
and scan feat_dim + dtype (fp32/fp16). If AGG time scales ~linearly with F (bytes) and fp16
halves it, AGG is bandwidth-bound -> int8/fp16 (fewer bytes) is a direct SPEED lever, not just
a memory one. Run on each GPU tier to quantify the HBM-bandwidth heterogeneity that should
drive zord's work-balance (compute/bandwidth-proportional, not equal).
  python scripts/kernel_roofline.py askubuntu
"""
import argparse, time
import numpy as np
import torch

from zord.datasets import load

# spec-sheet peak HBM bandwidth (GB/s) for achieved/peak context
PEAK_BW = {"H100": 3350.0, "RTX 6000 Ada": 960.0, "RTX 5000 Ada": 576.0,
           "A100": 2039.0, "RTX A6000": 768.0}


def peak_for(name):
    for k, v in PEAK_BW.items():
        if k.replace(" ", "").lower() in name.replace(" ", "").lower():
            return v
    return None


def build_adj(src, dst, n, dev):
    i = torch.tensor(np.concatenate([src, dst]), dtype=torch.long)
    j = torch.tensor(np.concatenate([dst, src]), dtype=torch.long)
    A = torch.sparse_coo_tensor(torch.stack([i, j]), torch.ones(i.shape[0]), (n, n)).coalesce().to(dev)
    deg = torch.sparse.sum(A, 1).to_dense().clamp(min=1.0)
    return torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]], (n, n)).coalesce()


def timed(fn, reps=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--dims", default="16,32,64,128,256,512,1024")
    a = ap.parse_args()
    dev = "cuda:0"
    name = torch.cuda.get_device_name(0)
    peak = peak_for(name)
    g = load(a.dataset).sort_by_time()
    N, E = g.num_nodes, g.num_edges
    nnz = 2 * E
    A32 = build_adj(g.src, g.dst, N, dev)
    print(f"ROOFLINE gpu='{name}' peak_HBM_GBs={peak} dataset={g.name} N={N} E={E} nnz={nnz}")

    for F in [int(x) for x in a.dims.split(",")]:
        for dt in (torch.float32, torch.float16):
            tag = "fp32" if dt == torch.float32 else "fp16"
            try:
                X = torch.randn(N, F, device=dev, dtype=dt)
                W = torch.randn(F, F, device=dev, dtype=dt)
                A = A32 if dt == torch.float32 else \
                    torch.sparse_coo_tensor(A32.indices(), A32.values().half(), (N, N)).coalesce()
                t_agg = timed(lambda: torch.sparse.mm(A, X))
                t_dense = timed(lambda: X @ W)
                elem = 4 if dt == torch.float32 else 2
                # AGG bytes ~ read A vals (nnz*elem) + gather src rows (nnz*F*elem) + write (N*F*elem)
                agg_bytes = nnz * elem + nnz * F * elem + N * F * elem
                agg_gbs = agg_bytes / t_agg / 1024 ** 3
                dense_tflops = (2.0 * N * F * F) / t_dense / 1e12
                frac = f" agg/peak={agg_gbs/peak*100:4.1f}%" if peak else ""
                print(f"  F={F:<5} {tag}  agg={t_agg*1e3:7.3f}ms ({agg_gbs:7.1f} GB/s{frac})  "
                      f"dense={t_dense*1e3:7.3f}ms ({dense_tflops:6.1f} TFLOP/s)  "
                      f"agg/dense={t_agg/t_dense:5.1f}x")
            except Exception as e:
                print(f"  F={F:<5} {tag}  SKIP ({type(e).__name__}: {str(e)[:60]})")


if __name__ == "__main__":
    main()
