#!/usr/bin/env python
"""ATTRIBUTE-DIMENSION COST SCALING (process-only): how does the memory-bound temporal-GNN
aggregation cost grow with the ATTRIBUTE (feature) dimension F? The 2-layer SpMM gather moves
~ (nnz * F) feature words per layer, so a memory-bound kernel should be LINEAR in F. We confirm
that story and FIT the slope = "Delta-cost per attribute" (ms per +1 dim) plus its linearity R^2.

Same graph, same result; we only vary F (the attribute width). This quantifies what it costs to
carry richer node/edge attributes through aggregation -- the lever for attribute-aware partitioning.

The graph is either:
  - a REAL attributed dataset (--dataset, e.g. jodie-wikipedia / tgbl-wiki): we additionally probe
    its NATIVE edge-feature dim Fe and sub/super-sets of it (Fe/4 .. 4*Fe via slice/tile), so the
    sweep is anchored on the real attribute width; OR
  - structural edges (--dataset of a non-attributed graph, or synthetic) with SYNTHETIC features of
    the swept dim F.

Graph build (CSR adjacency) and the 2-layer aggregation are timed in PyTorch on GPU; numpy builds
the CSR. NEVER networkx.
  python scripts/attribute_cost.py --dataset jodie-wikipedia
  python scripts/attribute_cost.py --nodes 2000000 --edges 30000000 --dims 16,32,64,128,256,512,1024
"""
import argparse, time
import numpy as np
import torch

# spec-sheet peak HBM bandwidth (GB/s) for an achieved/peak context line.
PEAK_BW = {"H100": 3350.0, "RTX 6000 Ada": 960.0, "RTX 5000 Ada": 576.0,
           "A100": 2039.0, "RTX A6000": 768.0}


def peak_for(name):
    for k, v in PEAK_BW.items():
        if k.replace(" ", "").lower() in name.replace(" ", "").lower():
            return v
    return None


def gen_edges(N, M, seed=0):
    """Plain Erdos-Renyi structural edges (attribute cost is order-independent here)."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, N, size=M).astype(np.int64)
    dst = rng.integers(0, N, size=M).astype(np.int64)
    return src, dst


def build_csr(src, dst, N, dev):
    """Symmetric row-normalized adjacency as a CSR tensor on `dev` (matches reorder_speedup)."""
    r = np.concatenate([src, dst]).astype(np.int64)
    c = np.concatenate([dst, src]).astype(np.int64)
    o = np.argsort(r, kind="stable"); r = r[o]; c = c[o]
    counts = np.bincount(r, minlength=N)
    deg = counts.astype(np.float32); deg[deg == 0] = 1.0
    vals = (1.0 / deg[r]).astype(np.float32)
    crow = np.zeros(N + 1, dtype=np.int64); np.cumsum(counts, out=crow[1:])
    return torch.sparse_csr_tensor(torch.from_numpy(crow), torch.from_numpy(c),
                                   torch.from_numpy(vals), size=(N, N), device=dev)


def timed(fn, reps=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def linfit(xs, ys):
    """Least-squares ys ~ a + b*xs; return (slope b, intercept a, R^2)."""
    x = np.asarray(xs, dtype=np.float64); y = np.asarray(ys, dtype=np.float64)
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(b), float(a), r2


def dims_for(arg_dims, native_fe):
    """Build the F sweep. If a native edge-feature dim exists, anchor the sweep on it
    (Fe/4, Fe/2, Fe, 2*Fe, 4*Fe) merged with the requested base dims; else use base dims."""
    base = sorted({int(x) for x in arg_dims.split(",") if x})
    if native_fe and native_fe > 0:
        anchored = {max(1, native_fe // 4), max(1, native_fe // 2), native_fe,
                    native_fe * 2, native_fe * 4}
        base = sorted(set(base) | anchored)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")                          # real graph (attributed or not)
    ap.add_argument("--nodes", type=int, default=2_000_000)           # synthetic fallback
    ap.add_argument("--edges", type=int, default=30_000_000)
    ap.add_argument("--dims", default="16,32,64,128,256,512,1024")
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()
    dev = "cuda:0"
    gpu = torch.cuda.get_device_name(0)
    peak = peak_for(gpu)

    native_fe = 0
    real_efeat = None
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int64); dst = g.dst.astype(np.int64); M = src.size
        if g.efeat is not None:
            native_fe = int(g.efeat.shape[1]); real_efeat = g.efeat
        print(f"ATTR-COST gpu='{gpu}' peak_HBM_GBs={peak} dataset={g.name} N={N} M={M} "
              f"native_edge_Fe={native_fe}")
    else:
        N, M = a.nodes, a.edges
        src, dst = gen_edges(N, M)
        print(f"ATTR-COST gpu='{gpu}' peak_HBM_GBs={peak} SYNTHETIC N={N} M={M} native_edge_Fe=0")

    nnz = 2 * M
    A = build_csr(src, dst, N, dev)
    dims = dims_for(a.dims, native_fe)
    print(f"  sweeping F over {dims}  (nnz={nnz}; 2-layer agg: relu(A X) -> A((.)W))")

    Fs, aggs = [], []
    for F in dims:
        try:
            # Node-feature matrix X [N, F]. For a real attributed dataset, materialize X at the
            # NATIVE Fe from a per-node pooled edge feature, then slice/tile to width F so the cost
            # reflects the real attribute distribution; else random features.
            if real_efeat is not None:
                Xn = node_pool(src, dst, real_efeat, N)            # [N, Fe]
                X = fit_width(Xn, F).to(dev)
            else:
                X = torch.randn(N, F, device=dev)
            W = torch.randn(F, F, device=dev) / F ** 0.5
            agg = timed(lambda: torch.sparse.mm(A, torch.relu(torch.sparse.mm(A, X) @ W)),
                        reps=a.reps)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            torch.cuda.reset_peak_memory_stats()
            # 2 gathers, each moves nnz*F feature words (fp32) -> bytes for achieved bandwidth.
            bw = (2 * nnz * F * 4) / agg / 1024 ** 3
            frac = f" ({bw/peak*100:4.1f}% peak)" if peak else ""
            tag = "  <-NATIVE" if F == native_fe else ""
            print(f"  F={F:<5} agg(2-layer)={agg*1e3:8.3f}ms  bw={bw:7.1f} GB/s{frac}  "
                  f"peak_mem={peak_mem:6.2f}GB{tag}")
            Fs.append(F); aggs.append(agg * 1e3)
            del X, W; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  F={F:<5} SKIP ({type(e).__name__}: {str(e)[:70]})")

    slope, intercept, r2 = linfit(Fs, aggs)
    print(f"  => FIT agg_ms ~= {intercept:.3f} + {slope:.5f}*F   "
          f"(Delta-cost/attribute = {slope*1e3:.3f} us per +1 dim)  R^2={r2:.4f}")
    if r2 == r2 and r2 > 0.97:
        print("  => LINEAR-in-F confirmed (R^2>0.97): aggregation is memory-bound; "
              "attribute width is a direct, linear cost lever.")


def node_pool(src, dst, efeat, N):
    """Pool edge features to nodes (mean over incident edges) -> a per-node attribute [N, Fe].
    numpy scatter-add (no networkx)."""
    Fe = efeat.shape[1]
    acc = np.zeros((N, Fe), dtype=np.float64)
    cnt = np.zeros(N, dtype=np.float64)
    np.add.at(acc, src, efeat); np.add.at(cnt, src, 1.0)
    np.add.at(acc, dst, efeat); np.add.at(cnt, dst, 1.0)
    cnt[cnt == 0] = 1.0
    return torch.from_numpy((acc / cnt[:, None]).astype(np.float32))


def fit_width(Xn, F):
    """Slice (sub-set) or tile (super-set) a [N, Fe] matrix to width F."""
    Fe = Xn.shape[1]
    if F == Fe:
        return Xn
    if F < Fe:
        return Xn[:, :F].contiguous()
    reps = (F + Fe - 1) // Fe
    return Xn.repeat(1, reps)[:, :F].contiguous()


if __name__ == "__main__":
    main()
