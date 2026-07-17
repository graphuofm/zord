#!/usr/bin/env python
"""GLOBAL MEMORY SCHEDULING, full precision: a temporal-GNN working set (W snapshots x N x F
fp32 embeddings) that does NOT fit a single GPU's HBM. We show zord's core move -- decide what
stays resident in HBM vs what is STAGED from CPU RAM over PCIe, with prefetch overlapping the
H2D copy behind compute -- turns a guaranteed OOM into a completed full-precision epoch.

Three modes on the SAME workload:
  in-core         : allocate the whole bank on GPU  -> OOMs (the baseline failure)
  tiered-blocking : bank in CPU; per-snapshot H2D copy THEN compute (serialized)
  tiered-prefetch : double-buffer; H2D of snapshot s+1 overlaps compute of s (2 CUDA streams)

Reports completed?/peak HBM/epoch time/PCIe GB for each. This is the NON-comm, NON-compression
thesis: the binding cost is the MEMORY SYSTEM (HBM capacity + CPU<->HBM PCIe), scheduled globally.
  python scripts/oom_to_tiered.py --nodes 2000000 --degree 8 --feat 256 --window 16
"""
import argparse, time
import numpy as np
import torch


def synth_adj(n, deg, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    e = n * deg
    i = torch.randint(0, n, (e,), generator=g)
    j = torch.randint(0, n, (e,), generator=g)
    idx = torch.stack([torch.cat([i, j]), torch.cat([j, i])])
    A = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (n, n)).coalesce().to(dev)
    deg_v = torch.sparse.sum(A, 1).to_dense().clamp(min=1.0)
    return torch.sparse_coo_tensor(A.indices(), A.values() / deg_v[A.indices()[0]], (n, n)).coalesce()


def reset_peak(dev):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)


def peak_gb(dev):
    return torch.cuda.max_memory_allocated(dev) / 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=2_000_000)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--feat", type=int, default=256)
    ap.add_argument("--window", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda:0"
    N, F, W = a.nodes, a.feat, a.window
    name = torch.cuda.get_device_name(0)
    bank_gb = W * N * F * 4 / 1024 ** 3
    print(f"TIERED gpu='{name}' N={N} deg={a.degree} F={F} window={W} "
          f"bank(fp32)={bank_gb:.1f}GB free_HBM={torch.cuda.mem_get_info(dev)[0]/1024**3:.1f}GB")

    A = synth_adj(N, a.degree, dev)
    W1 = torch.randn(F, F, device=dev) / F ** 0.5
    W2 = torch.randn(F, F, device=dev) / F ** 0.5

    def layer(X):                      # one GraphSAGE-style 2-hop forward on a resident snapshot
        return torch.sparse.mm(A, torch.relu(torch.sparse.mm(A, X) @ W1)) @ W2

    # ---- mode 1: in-core (allocate whole bank on GPU) ----
    reset_peak(dev)
    try:
        bank = torch.empty(W, N, F, device=dev)          # <- the OOM point at scale
        bank.normal_()
        t0 = time.time(); acc = 0.0
        for s in range(W):
            acc += layer(bank[s]).sum().item()
        torch.cuda.synchronize()
        print(f"  in-core         : COMPLETED  peak={peak_gb(dev):.1f}GB  time={time.time()-t0:.2f}s")
        del bank
    except RuntimeError as e:
        msg = "OOM" if "out of memory" in str(e).lower() else type(e).__name__
        print(f"  in-core         : FAILED ({msg})  -- baseline cannot run this config")
        torch.cuda.empty_cache()

    # CPU-resident bank (pageable); a single reused PINNED bounce buffer for fast H2D
    cpu_bank = torch.randn(W, N, F)                      # full precision, lives in CPU RAM
    pcie_gb = W * N * F * 4 / 1024 ** 3

    # ---- mode 2: tiered-blocking ----
    reset_peak(dev)
    buf = torch.empty(N, F, pin_memory=True)
    t0 = time.time(); acc = 0.0
    for s in range(W):
        buf.copy_(cpu_bank[s])
        Xg = buf.to(dev, non_blocking=False)
        acc += layer(Xg).sum().item()
    torch.cuda.synchronize(); t_block = time.time() - t0
    print(f"  tiered-blocking : COMPLETED  peak={peak_gb(dev):.1f}GB  time={t_block:.2f}s  PCIe={pcie_gb:.1f}GB")

    # ---- mode 3: tiered-prefetch (double buffer, copy stream overlaps compute) ----
    reset_peak(dev)
    cpy = torch.cuda.Stream()
    bufs = [torch.empty(N, F, pin_memory=True) for _ in range(2)]
    gpu = [torch.empty(N, F, device=dev) for _ in range(2)]
    bufs[0].copy_(cpu_bank[0]); gpu[0].copy_(bufs[0], non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.time(); acc = 0.0
    for s in range(W):
        cur = s % 2; nxt = (s + 1) % 2
        if s + 1 < W:                                    # prefetch next on copy stream
            with torch.cuda.stream(cpy):
                bufs[nxt].copy_(cpu_bank[s + 1])
                gpu[nxt].copy_(bufs[nxt], non_blocking=True)
        acc += layer(gpu[cur]).sum().item()              # compute current on default stream
        torch.cuda.current_stream().wait_stream(cpy)
    torch.cuda.synchronize(); t_pre = time.time() - t0
    print(f"  tiered-prefetch : COMPLETED  peak={peak_gb(dev):.1f}GB  time={t_pre:.2f}s  PCIe={pcie_gb:.1f}GB")
    print(f"  => prefetch hides {max(0.0,(t_block-t_pre))/t_block*100:.1f}% of PCIe staging vs blocking")


if __name__ == "__main__":
    main()
