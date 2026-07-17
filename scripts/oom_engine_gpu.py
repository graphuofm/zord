#!/usr/bin/env python
"""ENGINE-DECIDED tiering, EXECUTED on a real GPU, on a REAL graph -- the unified feasibility
headline (§40 the engine DECIDES + §40-empirical the GPU EXECUTES, no hardcoded split).

This is the honest end-to-end claim:

  1. A REAL temporal graph (default `wiki-talk`, N~1.14M, E~7.8M, from zord.datasets.load) gives
     the workload's N and E. With F=512, W=16 (the §40-empirical config) the FULL working set --
     W co-resident snapshots, each (1+L) fp32 copies of [N,F] features+activations plus the
     resident adjacency -- is ~105GB, which CLEARLY exceeds one H100's 80GB HBM.

  2. The REAL zord engine, zord.schedule.plan_memory(cluster, workload), DECIDES the tiering:
     how many of the W snapshots stay RESIDENT in HBM vs are STREAMED from CPU RAM over PCIe,
     the PREDICTED peak HBM, and feasibility. The split is NOT hardcoded -- it comes from the
     planner's roofline + capacity model. (vs scripts/oom_to_tiered.py whose split is hardcoded.)

  3. On ONE GPU (torch) we then:
       (a) IN-CORE baseline   -- try to hold the WHOLE working set on one GPU -> CUDA-OOM.
       (b) EXECUTE THE PLAN   -- keep the engine-decided `resident` snapshots in HBM, STREAM the
           remaining `streamed` snapshots from CPU over PCIe with double-buffered prefetch (the
           H2D copy of snapshot s+1 overlaps the aggregation of s on a second CUDA stream), run
           the full-precision 2-layer aggregation -> COMPLETES.

  4. KEY (vs oom_to_tiered.py): the resident/streamed split is taken FROM zord.schedule.plan_memory
     (the ENGINE), and we VERIFY the engine's PREDICTED peak HBM matches the MEASURED peak on the
     GPU -- i.e. the planner's prediction is accurate, so the plan it ships is the plan that runs.

PROCESS-only: full precision (fp32), same data + model => same result; we report time / memory /
PCIe / feasibility, NEVER accuracy. NEVER networkx. Runs on 1 GPU; CUDA-absent is guarded so the
file imports + py_compiles on a CPU box (the real numbers come from the H100 submission).

  python scripts/oom_engine_gpu.py --dataset wiki-talk --feat 512 --window 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# zord lives under src/ (editable layout); make it importable without an install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# THE ENGINE. grep-able: this script imports + calls zord.schedule.plan_memory.
from zord.schedule import plan_memory                      # noqa: E402  (the engine)
from zord.schedule.planner import Workload, _snapshot_state_bytes   # noqa: E402
from zord.profiler.cluster_profile import from_spec, hetcluster          # noqa: E402

GB = 1024 ** 3


# ----------------------------------------------------------------------------- #
# 1. REAL graph -> Workload (N, E from the real temporal stream; F, W from args) #
# ----------------------------------------------------------------------------- #
def load_workload(dataset: str, feat: int, window: int, layers: int,
                  reuse_frac: float, synth_nodes: int, synth_degree: int):
    """Return (Workload, N, E, src, dst, label). Prefer a REAL graph via zord.datasets.load; if
    it is unavailable on this box (not staged / no network), fall back to a synthetic stream of
    the requested scale so the script still RUNS for py_compile / smoke -- the engine call and the
    GPU execution are identical either way. The HEADLINE config uses the real graph."""
    if dataset != "synthetic":
        try:
            from zord.datasets import load
            g = load(dataset).sort_by_time()
            N, E = int(g.num_nodes), int(g.num_edges)
            print(f"[real graph] dataset={dataset} N={N:,} E={E:,} "
                  f"(loaded via zord.datasets.load)")
            w = Workload(num_nodes=N, num_edges=E, feat_dim=feat, layers=layers,
                         window=window, reuse_frac=reuse_frac)
            return w, N, E, np.asarray(g.src), np.asarray(g.dst), dataset
        except Exception as e:  # not staged here -> synthetic fallback (same downstream path)
            print(f"[warn] could not load real dataset {dataset!r} ({type(e).__name__}: {e}); "
                  f"falling back to synthetic N={synth_nodes:,} deg={synth_degree}")
    N = synth_nodes
    E = N * synth_degree
    rng = np.random.default_rng(0)
    src = rng.integers(0, N, size=E, dtype=np.int64)
    dst = rng.integers(0, N, size=E, dtype=np.int64)
    print(f"[synthetic graph] N={N:,} E={E:,} deg={synth_degree}")
    w = Workload(num_nodes=N, num_edges=E, feat_dim=feat, layers=layers,
                 window=window, reuse_frac=reuse_frac)
    return w, N, E, src, dst, "synthetic"


# ----------------------------------------------------------------------------- #
# GPU helpers (only touched when CUDA is present)                               #
# ----------------------------------------------------------------------------- #
def _reset_peak(torch, dev):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)


def _peak_gb(torch, dev):
    return torch.cuda.max_memory_allocated(dev) / GB


def build_norm_adj(torch, src, dst, N, dev):
    """Symmetric, degree-normalized sparse adjacency on the GPU (one resident copy, reused by
    every snapshot's aggregation -- this is the resident edge metadata the planner accounts for)."""
    i = torch.as_tensor(src, dtype=torch.long)
    j = torch.as_tensor(dst, dtype=torch.long)
    idx = torch.stack([torch.cat([i, j]), torch.cat([j, i])])
    vals = torch.ones(idx.shape[1])
    A = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce().to(dev)
    deg = torch.sparse.sum(A, 1).to_dense().clamp(min=1.0)
    return torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]],
                                   (N, N)).coalesce()


def run_gpu(plan, w, N, E, src, dst, layers, feat):
    """Execute the ENGINE'S plan on one GPU: (a) in-core baseline (expect OOM), (b) the
    engine-decided resident/streamed split with prefetch (expect COMPLETE), then VERIFY the
    PREDICTED peak == MEASURED peak. Returns a dict of measurements (or None if no CUDA)."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
    except Exception as e:
        print(f"\n[no GPU here] {type(e).__name__}: {e} -- skipping GPU execution. "
              f"Submit on an H100 to get the measured numbers.\n"
              f"The engine plan above is what the GPU run will execute.")
        return None

    dev = "cuda:0"
    F, W = feat, w.window
    p = plan.per_device[0]
    resident, streamed = p.resident_snapshots, p.streamed_snapshots
    name = torch.cuda.get_device_name(0)
    free0 = torch.cuda.mem_get_info(dev)[0] / GB
    full_ws_gb = _snapshot_state_bytes(N, E, w) * W / GB
    print(f"\n[GPU] '{name}' free_HBM={free0:.1f}GB  full_working_set={full_ws_gb:.1f}GB  "
          f"engine: resident={resident}/{W} streamed={streamed} "
          f"predicted_peak={p.peak_hbm_bytes/GB:.1f}GB")

    A = build_norm_adj(torch, src, dst, N, dev)            # resident edge metadata (one copy)
    W1 = torch.randn(F, F, device=dev) / F ** 0.5
    W2 = torch.randn(F, F, device=dev) / F ** 0.5

    def aggregate(X):
        """One full-precision 2-layer GraphSAGE-style aggregation. We MATERIALIZE the L activation
        copies (h1, h2) so the live HBM footprint == the planner's (1+L)*N*F*4 per-snapshot model."""
        h1 = torch.sparse.mm(A, torch.relu(torch.sparse.mm(A, X) @ W1))   # activation copy 1
        h2 = torch.sparse.mm(A, h1) @ W2                                   # activation copy 2
        return h1, h2

    out = {"gpu": name, "in_core_oom": None, "completed": None,
           "predicted_peak_gb": p.peak_hbm_bytes / GB, "measured_peak_gb": None,
           "epoch_sec": None, "pcie_gb": None}

    # ---- (a) IN-CORE baseline: hold the WHOLE working set on the GPU -> expect CUDA-OOM ----
    # The "working set" is the SAME footprint the engine models: each of the W co-resident snapshots
    # keeps its features [N,F] AND its L activation copies [N,F] live (the windowed back-prop state).
    # That is (1+L)*N*F*4 * W bytes -- the planner's _snapshot_state_bytes summed over the window --
    # which at the headline config is ~105GB and CANNOT fit one 80GB GPU. (A naive features-only
    # torch.empty(W,N,F) would understate the working set and could fit; we hold the activations too
    # so the baseline allocation matches the engine's working-set definition exactly -> honest OOM.)
    _reset_peak(torch, dev)
    try:
        feat_bank = [torch.empty(N, F, device=dev) for _ in range(W)]          # W feature copies
        act_bank = [[torch.empty(N, F, device=dev) for _ in range(layers)]      # W * L activations
                    for _ in range(W)]
        acc = 0.0
        for s in range(W):
            feat_bank[s].normal_()
            h1, h2 = aggregate(feat_bank[s])
            act_bank[s][0].copy_(h1); act_bank[s][1].copy_(h2)   # keep activations live (window state)
            acc += float(h2.sum())
        torch.cuda.synchronize()
        out["in_core_oom"] = False
        print(f"  in-core   : COMPLETED (working set fit -- raise --feat/--window to force OOM) "
              f"peak={_peak_gb(torch, dev):.1f}GB")
        del feat_bank, act_bank, h1, h2
    except RuntimeError as e:
        is_oom = "out of memory" in str(e).lower()
        out["in_core_oom"] = bool(is_oom)
        print(f"  in-core   : FAILED ({'CUDA-OOM' if is_oom else type(e).__name__}) "
              f"-- baseline cannot hold the full {full_ws_gb:.0f}GB working set in {free0:.0f}GB HBM")
        # CRITICAL: drop refs to the in-core baseline's PARTIAL allocations (a completed feat_bank
        # ~= W*N*F*4 can survive the OOM and POLLUTE the engine-plan run below -> false OOM). Set to
        # None + gc + empty_cache so the engine-plan executes on a CLEAN GPU. (Fix: the prior
        # "engine peak undercounts double-buffer" diagnosis was WRONG -- the real cause was this leak.)
        import gc
        feat_bank = act_bank = h1 = h2 = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ---- (b) EXECUTE THE ENGINE'S PLAN: resident snapshots in HBM, stream the rest w/ prefetch --
    # CPU-resident bank holds ALL W snapshots' features (full precision, in CPU RAM).
    cpu_bank = torch.randn(W, N, F)                        # CPU RAM (pageable)
    pcie_gb = streamed * N * F * 4 / GB                    # only the streamed snapshots cross PCIe

    _reset_peak(torch, dev)
    # RESIDENT: keep `resident` snapshots' features + their L activation copies live in HBM, so the
    # resident footprint matches the planner's (1+L)*N*F*4 * resident model exactly.
    res_feat = [torch.empty(N, F, device=dev) for _ in range(resident)]
    res_act = [[torch.empty(N, F, device=dev) for _ in range(layers)] for _ in range(resident)]
    for s in range(resident):
        res_feat[s].copy_(cpu_bank[s], non_blocking=False)

    # STREAMING double-buffer (reserve_buffers=1 -> 1 extra snapshot slot + its activations live in
    # HBM while we prefetch). Two pinned bounce buffers + two GPU slots overlap H2D with compute.
    cpy = torch.cuda.Stream()
    pin = [torch.empty(N, F, pin_memory=True) for _ in range(2)] if streamed else []
    gbuf = [torch.empty(N, F, device=dev) for _ in range(2)] if streamed else []
    sbuf_act = [[torch.empty(N, F, device=dev) for _ in range(layers)]
                for _ in range(1)] if streamed else []      # activations for the streamed slot

    acc = 0.0
    torch.cuda.synchronize()
    t0 = time.time()

    # process the RESIDENT snapshots (no PCIe; already in HBM)
    for s in range(resident):
        h1, h2 = aggregate(res_feat[s])
        res_act[s][0].copy_(h1); res_act[s][1].copy_(h2)    # hold activations live (footprint)
        acc += float(h2.sum())

    # process the STREAMED snapshots with double-buffered prefetch over PCIe
    if streamed:
        # prime: stage the first streamed snapshot
        pin[0].copy_(cpu_bank[resident])
        gbuf[0].copy_(pin[0], non_blocking=True)
        torch.cuda.synchronize()
        for k in range(streamed):
            cur, nxt = k % 2, (k + 1) % 2
            s = resident + k
            if k + 1 < streamed:                            # prefetch next streamed snapshot
                with torch.cuda.stream(cpy):
                    pin[nxt].copy_(cpu_bank[resident + k + 1])
                    gbuf[nxt].copy_(pin[nxt], non_blocking=True)
            h1, h2 = aggregate(gbuf[cur])                   # compute current (default stream)
            sbuf_act[0][0].copy_(h1); sbuf_act[0][1].copy_(h2)
            acc += float(h2.sum())
            torch.cuda.current_stream().wait_stream(cpy)    # ensure prefetch landed before reuse

    torch.cuda.synchronize()
    epoch = time.time() - t0
    measured_peak = _peak_gb(torch, dev)
    out.update(completed=True, measured_peak_gb=measured_peak, epoch_sec=epoch, pcie_gb=pcie_gb)
    print(f"  engine-plan: COMPLETED  measured_peak={measured_peak:.1f}GB  "
          f"epoch={epoch:.2f}s  PCIe={pcie_gb:.1f}GB  (acc={acc:.3e})")

    # ---- (4) VERIFY predicted peak == measured peak (planner accuracy) ----
    pred = p.peak_hbm_bytes / GB
    rel = abs(measured_peak - pred) / max(pred, 1e-9)
    out["peak_rel_err"] = rel
    verdict = "MATCH" if rel <= 0.15 else "MISMATCH"
    print(f"  VERIFY peak: predicted={pred:.1f}GB  measured={measured_peak:.1f}GB  "
          f"rel_err={rel*100:.1f}%  -> {verdict} "
          f"(planner's predicted peak is {'accurate' if verdict=='MATCH' else 'OFF'})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="wiki-talk",
                    help="real graph via zord.datasets.load (default wiki-talk: N~1.14M, E~7.8M); "
                         "'synthetic' to force the synthetic fallback")
    ap.add_argument("--feat", type=int, default=512, help="feature dim F (default 512, §40-empirical)")
    ap.add_argument("--window", type=int, default=16, help="co-resident snapshots W (default 16)")
    ap.add_argument("--layers", type=int, default=2, help="GraphSAGE depth L (activation copies)")
    ap.add_argument("--reuse-frac", type=float, default=0.0, help="rho: unchanged-node reuse")
    ap.add_argument("--hbm-gb", type=float, default=80.0, help="single-GPU HBM capacity (engine model)")
    ap.add_argument("--agg-bw", type=float, default=942.0, help="achieved HBM agg bandwidth GB/s")
    ap.add_argument("--h2d-gbps", type=float, default=57.5, help="host->device PCIe GB/s")
    ap.add_argument("--synth-nodes", type=int, default=4_000_000, help="synthetic fallback N")
    ap.add_argument("--synth-degree", type=int, default=8, help="synthetic fallback avg degree")
    a = ap.parse_args()

    # 1. REAL graph -> Workload (N, E real; F, W from args)
    w, N, E, src, dst, label = load_workload(
        a.dataset, a.feat, a.window, a.layers, a.reuse_frac, a.synth_nodes, a.synth_degree)

    # ---- a single-GPU cluster, built from an EXPLICIT spec (the planner is not HetCluster-tied) ----
    cluster = from_spec(hbm_gb=[a.hbm_gb], agg_bw_gbps=[a.agg_bw],
                        interconnect_gbps=325.0, h2d_gbps=a.h2d_gbps,
                        names=[f"GPU-{a.hbm_gb:g}GB"])
    cap = cluster.devices[0].usable_mem
    per_snap = _snapshot_state_bytes(N, E, w)
    full_ws = per_snap * w.window
    print(f"[cluster] 1x {cluster.devices[0].name}  usable_HBM={cap/GB:.1f}GB  "
          f"agg_bw={a.agg_bw:g}GB/s  h2d={a.h2d_gbps:g}GB/s")
    print(f"[workload] N={N:,} E={E:,} F={a.feat} L={a.layers} W={a.window}  "
          f"per_snapshot_state={per_snap/GB:.2f}GB  FULL_working_set={full_ws/GB:.1f}GB  "
          f"({'EXCEEDS' if full_ws > cap else 'fits in'} the {cap/GB:.0f}GB GPU)")

    # 2. THE ENGINE DECIDES the tiering plan (resident vs streamed, predicted peak, feasibility)
    plan = plan_memory(cluster, w, prefetch=True)
    print("\n[engine] zord.schedule.plan_memory ->")
    print(plan.summary())
    p = plan.per_device[0]
    print(f"[engine decision] resident={p.resident_snapshots}/{w.window}  "
          f"streamed={p.streamed_snapshots}  predicted_peak_HBM={p.peak_hbm_bytes/GB:.1f}GB  "
          f"feasible={p.feasible}  predicted_epoch={p.epoch_sec*1e3:.0f}ms  "
          f"streamed/epoch={plan.total_streamed_gb:.1f}GB  bound={plan.bound}")
    if p.streamed_snapshots == 0 and p.feasible:
        print("[note] the engine kept ALL snapshots resident (working set fit) -- increase "
              "--feat or --window to push the working set past HBM and exercise streaming.")

    # 3+4. EXECUTE the engine's plan on the GPU + VERIFY predicted vs measured peak
    run_gpu(plan, w, N, E, src, dst, a.layers, a.feat)


if __name__ == "__main__":
    main()
