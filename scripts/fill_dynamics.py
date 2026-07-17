#!/usr/bin/env python
"""HBM FILL DYNAMICS -- measure the data-flow timescales that decide whether DYNAMIC overflow
handling (watermark / spill / backpressure) is even VIABLE, or whether we must reserve headroom
PROACTIVELY (user question 2026-05-31: "is the card full in one 'pop', or is there reaction time?
the data transfer may not be that fast").

The reaction window exists only BETWEEN incremental allocations. We measure, on a real GPU:
  1. FILL rate    : grow the working set one SNAPSHOT (N x F fp32) at a time -> GB per snapshot and
                    ms per snapshot (allocate + a real aggregation). fill_GBps = GB / sec.
  2. EVICT rate   : D2H copy a snapshot GPU->CPU(pinned) -> stage-out GB/s (PCIe). Also H2D (refill).
  3. REACTION test: reaction_window_ms = headroom_GB / fill_GBps; in that window can we evict the
                    next increment? VIABLE (reactive watermark works) iff evict_GBps >= fill_GBps
                    (we stage out at least as fast as we fill) AND one snapshot evicts within one
                    snapshot-interval. Else PROACTIVE-only (reserve headroom upfront / admission ctrl).
PROCESS-only (timescales/bandwidth; never accuracy). NEVER networkx.
  python scripts/fill_dynamics.py --nodes 1000000 --feat 512 --snapshots 40
"""
import argparse, time
import numpy as np

GB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=1_000_000)
    ap.add_argument("--feat", type=int, default=512)
    ap.add_argument("--degree", type=int, default=8)
    ap.add_argument("--snapshots", type=int, default=40, help="keep allocating until OOM or this many")
    a = ap.parse_args()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA")
    except Exception as e:
        print(f"[no GPU here: {e}] submit on an H100; this measures real fill/evict timescales.")
        return
    dev = "cuda:0"
    N, F = a.nodes, a.feat
    snap_gb = N * F * 4 / GB
    total = torch.cuda.get_device_properties(0).total_memory / GB
    free0 = torch.cuda.mem_get_info(dev)[0] / GB
    name = torch.cuda.get_device_name(0)
    print(f"FILL-DYNAMICS gpu='{name}' total={total:.1f}GB free={free0:.1f}GB  N={N:,} F={F}  "
          f"snapshot={snap_gb:.2f}GB")

    # a small sparse adjacency for a REAL aggregation per snapshot (so fill includes compute, not
    # just malloc) -- this is what makes the per-snapshot INTERVAL realistic.
    g = torch.Generator().manual_seed(0)
    e = N * a.degree
    src = torch.randint(0, N, (e,), generator=g)
    dst = torch.randint(0, N, (e,), generator=g)
    idx = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    A = torch.sparse_coo_tensor(idx, torch.ones(2 * e), (N, N)).coalesce().to(dev)
    W = torch.randn(F, F, device=dev) / F ** 0.5

    # ---- 1. FILL rate: allocate + aggregate one snapshot at a time, time each ----
    bank = []
    fill_ms = []
    print("  -- filling one snapshot at a time (allocate + real SpMM aggregation) --")
    for s in range(a.snapshots):
        torch.cuda.synchronize(); t0 = time.time()
        try:
            x = torch.randn(N, F, device=dev)            # allocate the snapshot
            h = torch.relu(torch.sparse.mm(A, x) @ W)     # a real aggregation (keeps it live)
            bank.append(x); bank.append(h)                # hold -> grows the working set
            torch.cuda.synchronize()
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                used = torch.cuda.memory_allocated(dev) / GB
                print(f"  snapshot {s}: CUDA-OOM at ~{used:.1f}GB used "
                      f"(free was {torch.cuda.mem_get_info(dev)[0]/GB:.2f}GB) -- a SINGLE alloc of "
                      f"{2*snap_gb:.2f}GB had no room -> ZERO reaction time at this granularity.")
                break
            raise
        dt = (time.time() - t0) * 1e3
        fill_ms.append(dt)
        if s < 3 or s % 5 == 0:
            used = torch.cuda.memory_allocated(dev) / GB
            print(f"  snapshot {s:2d}: +{2*snap_gb:.2f}GB in {dt:6.1f}ms  used={used:5.1f}GB  "
                  f"free={torch.cuda.mem_get_info(dev)[0]/GB:5.1f}GB")
    med_fill_ms = float(np.median(fill_ms)) if fill_ms else float("nan")
    fill_gbps = (2 * snap_gb) / (med_fill_ms / 1e3) if fill_ms else float("nan")

    # ---- 2. EVICT rate: D2H (stage-out to CPU pinned) and H2D (refill) for one snapshot ----
    if bank:
        x = bank[0]
        cpu = torch.empty(N, F, pin_memory=True)
        torch.cuda.synchronize(); t0 = time.time()
        cpu.copy_(x, non_blocking=False); torch.cuda.synchronize()
        d2h_ms = (time.time() - t0) * 1e3
        gpu2 = torch.empty(N, F, device=dev)
        torch.cuda.synchronize(); t0 = time.time()
        gpu2.copy_(cpu, non_blocking=False); torch.cuda.synchronize()
        h2d_ms = (time.time() - t0) * 1e3
        d2h_gbps = snap_gb / (d2h_ms / 1e3)
        h2d_gbps = snap_gb / (h2d_ms / 1e3)
    else:
        d2h_gbps = h2d_gbps = d2h_ms = h2d_ms = float("nan")

    # ---- 3. REACTION analysis ----
    print("\n  ================= REACTION-WINDOW ANALYSIS =================")
    print(f"  FILL  : {2*snap_gb:.2f}GB / snapshot, median {med_fill_ms:.1f}ms/snapshot "
          f"-> fill rate = {fill_gbps:.1f} GB/s")
    print(f"  EVICT : D2H(stage-out to CPU) {d2h_gbps:.1f} GB/s ({d2h_ms:.1f}ms/snapshot); "
          f"H2D(refill) {h2d_gbps:.1f} GB/s")
    if fill_ms:
        # between two snapshot allocations we have ~med_fill_ms; in that interval we can evict:
        evict_per_interval = d2h_gbps * (med_fill_ms / 1e3)
        ratio = d2h_gbps / fill_gbps if fill_gbps else float("nan")
        viable = (ratio >= 1.0) and (evict_per_interval >= 2 * snap_gb)
        print(f"  per-snapshot INTERVAL = {med_fill_ms:.1f}ms; in it we can stage out "
              f"{evict_per_interval:.2f}GB vs the {2*snap_gb:.2f}GB we just added.")
        print(f"  VIABILITY ratio (evict_BW / fill_BW) = {ratio:.2f}  -> "
              f"{'REACTIVE watermark/spill IS viable (we evict >= we fill; reaction time exists)' if viable else 'REACTIVE NOT enough (we fill faster than we evict) -> must RESERVE HEADROOM PROACTIVELY / admission-control'}")
        print(f"  NOTE: a SINGLE alloc bigger than current free = INSTANT OOM (zero reaction) regardless"
              f" -- so the runtime MUST allocate at snapshot granularity + watermark-check BETWEEN allocs.")


if __name__ == "__main__":
    main()
