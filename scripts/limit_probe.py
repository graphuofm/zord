#!/usr/bin/env python
"""SINGLE-GPU FEASIBILITY FRONTIER (E2): the largest graph that BUILDS + AGGREGATES (a 2-layer GNN
forward) on ONE GPU before CUDA OOM. This maps zord's "submit-and-it-completes" boundary -- the
process-level capacity frontier, no accuracy involved.

The graph is generated DIRECTLY ON THE GPU (torch.randint(..., device='cuda') for src/dst) so we are
NOT bounded by host RAM and can probe huge edge counts. At each size E in {50M,100M,200M,400M,800M,
1600M} (N = E/16, ~degree 32 symmetric) we:
  1. build the sparse, row-normalized adjacency on GPU,
  2. allocate features X (N x F fp32),
  3. run a 2-layer aggregation  torch.sparse.mm(A, relu(torch.sparse.mm(A, X) @ W1)).
Each stage is wrapped in try/except RuntimeError; on CUDA OOM we record the failure, free, and report
the LAST feasible size and its peak memory. Per size we print: built? agg ms? peak GB? free GB?

  python scripts/limit_probe.py --feat 128
"""
import argparse, time
import numpy as np
import torch


def reset_peak(dev):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)


def peak_gb(dev):
    return torch.cuda.max_memory_allocated(dev) / 1024 ** 3


def free_gb(dev):
    return torch.cuda.mem_get_info(dev)[0] / 1024 ** 3


def is_oom(e):
    return "out of memory" in str(e).lower()


def build_adj_gpu(N, E, dev, seed=0):
    """Build a symmetric, row-normalized CSR adjacency ENTIRELY on the GPU.

    src/dst are sampled with torch.randint on-device (no host array of E ints), then symmetrized and
    sorted into CSR via a COO->coalesce->CSR conversion -- all device-resident."""
    g = torch.Generator(device=dev).manual_seed(seed)
    i = torch.randint(0, N, (E,), generator=g, device=dev, dtype=torch.int64)
    j = torch.randint(0, N, (E,), generator=g, device=dev, dtype=torch.int64)
    rows = torch.cat([i, j]); cols = torch.cat([j, i])                # symmetrize
    del i, j
    idx = torch.stack([rows, cols])
    del rows, cols
    A = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1], device=dev), (N, N)).coalesce()
    del idx
    deg = torch.sparse.sum(A, 1).to_dense().clamp(min=1.0)
    vals = A.values() / deg[A.indices()[0]]
    A = torch.sparse_coo_tensor(A.indices(), vals, (N, N)).coalesce().to_sparse_csr()
    return A


def timed(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--edges", type=int, nargs="*",
                    default=[50_000_000, 100_000_000, 200_000_000, 400_000_000,
                             800_000_000, 1_600_000_000])
    ap.add_argument("--ratio", type=int, default=16, help="N = E / ratio (avg degree ~ 2*ratio)")
    a = ap.parse_args()
    dev = "cuda:0"
    F = a.feat
    gpu = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3
    print(f"LIMIT-PROBE gpu='{gpu}' total_HBM={total_gb:.1f}GB F={F} ratio={a.ratio} "
          f"sizes={[e//1_000_000 for e in a.edges]}M-edges")

    last_feasible = None                                              # (E, N, peak_gb, agg_ms)

    for E in a.edges:
        N = max(1, E // a.ratio)
        adj_gb = (2 * E) * (8 + 8 + 4) / 1024 ** 3                    # COO rough: 2 idx (i64) + val (f32)
        feat_gb = N * F * 4 / 1024 ** 3
        print(f"\n=== E={E/1e6:.0f}M  N={N/1e6:.2f}M  est_adj(coo)~{adj_gb:.1f}GB  X~{feat_gb:.1f}GB "
              f"free={free_gb(dev):.1f}GB ===")
        reset_peak(dev)

        # ---- stage 1: build adjacency on GPU ----
        built = False; A = None
        try:
            t0 = time.time()
            A = build_adj_gpu(N, E, dev)
            torch.cuda.synchronize()
            built = True
            print(f"  build   : OK   {time.time()-t0:.2f}s  peak={peak_gb(dev):.1f}GB  free={free_gb(dev):.1f}GB")
        except RuntimeError as e:
            print(f"  build   : FAILED ({'CUDA-OOM' if is_oom(e) else type(e).__name__})")
            A = None
            torch.cuda.empty_cache()
            if not is_oom(e):
                raise
            break                                                     # bigger sizes only get worse

        # ---- stage 2: allocate features ----
        X = None; W1 = None
        try:
            X = torch.randn(N, F, device=dev)
            W1 = torch.randn(F, F, device=dev) / F ** 0.5
            print(f"  alloc X : OK   peak={peak_gb(dev):.1f}GB  free={free_gb(dev):.1f}GB")
        except RuntimeError as e:
            print(f"  alloc X : FAILED ({'CUDA-OOM' if is_oom(e) else type(e).__name__})  "
                  f"-- adjacency built but features do not fit")
            del A
            torch.cuda.empty_cache()
            if not is_oom(e):
                raise
            break

        # ---- stage 3: 2-layer aggregation ----
        try:
            def fwd():
                return torch.sparse.mm(A, torch.relu(torch.sparse.mm(A, X) @ W1))
            agg = timed(fwd)
            pk = peak_gb(dev)
            bw = (2 * (2 * E) * F * 4) / agg / 1024 ** 3              # 2 gather layers over 2E entries
            print(f"  agg(2L) : OK   {agg*1e3:8.2f}ms  bw={bw:7.1f}GB/s  peak={pk:.1f}GB  free={free_gb(dev):.1f}GB")
            last_feasible = (E, N, pk, agg * 1e3)
        except RuntimeError as e:
            print(f"  agg(2L) : FAILED ({'CUDA-OOM' if is_oom(e) else type(e).__name__})  "
                  f"-- built+allocated but forward OOMs")
            del A, X, W1
            torch.cuda.empty_cache()
            if not is_oom(e):
                raise
            break

        del A, X, W1
        torch.cuda.empty_cache()

    print("\n========================================================")
    if last_feasible is None:
        print("FRONTIER: NO size completed the full build+alloc+agg pipeline on this GPU.")
    else:
        E, N, pk, ms = last_feasible
        print(f"FRONTIER (single-GPU 'submit-and-it-completes'): largest COMPLETED size = "
              f"E={E/1e6:.0f}M edges, N={N/1e6:.2f}M nodes, F={F}")
        print(f"  at peak HBM={pk:.1f}GB of {total_gb:.1f}GB total, agg={ms:.2f}ms/forward")
    print("Beyond the frontier, zord must spill the bank to CPU RAM (see oom_to_tiered.py) or shard "
          "across GPUs -- the single-GPU capacity wall this probe measures.")


if __name__ == "__main__":
    main()
