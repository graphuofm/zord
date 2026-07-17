#!/usr/bin/env python
"""CACHE-RESIDENT TILING of the GNN aggregation (D8): the SpMM aggregation is memory-bound, so its
speed is set by where its working set lives. RESULTS 9b: when a feature matrix fits in L2 (~32MB on
RTX5000, ~50MB on H100/RTX6000) the aggregation reaches ~1700+ GB/s vs ~683 GB/s HBM-resident.

This harness PROCESS-measures (TIME / bandwidth; the RESULT is identical -- same A, same X, same W)
whether we can recover that cache bandwidth by TILING the aggregation over destination-node blocks
sized so each tile's gathered features stay L2-resident. tile = L2_bytes / (F * 4) rows.

  UNTILED : torch.sparse.mm(A, relu(torch.sparse.mm(A, X) @ W1))   -- whole X streams from HBM
  TILED   : same math, but the destination rows are processed in blocks of `tile` so the gathered
            source features of each block fit L2; loop tiles, accumulate into the output.

With a LOCAL node ordering (e.g. the C++ lpa ordering in build/graph_algos) the neighbors of a
contiguous destination block are themselves contiguous, so a tile touches a small, cache-resident
slice of X and tiling helps MORE. We optionally reorder first (--reorder lpa); identity ordering is
the conservative (worst) case for tiling.

Sweeps F in {64,128,256,512} x a couple graph sizes; reports agg ms + achieved GB/s untiled vs
tiled and whether tiling approaches cache bandwidth.
  python scripts/cache_tiling.py --nodes 4000000 --edges 64000000 --l2-mb 32
  python scripts/cache_tiling.py --dataset hetcluster --l2-mb 50 --reorder lpa
"""
import argparse, os, struct, subprocess, time
import numpy as np
import torch

BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")


def gen_graph(N, M, seed=0):
    """Synthetic graph with LOCALITY: most edges connect nearby node ids (banded), so a
    contiguous destination tile has mostly-contiguous neighbors -- the regime where tiling can win."""
    rng = np.random.default_rng(seed)
    band = max(1, N // 100)                                   # locality radius (~1% of N)
    m_loc = int(M * 0.9)
    u = rng.integers(0, N, size=m_loc)
    off = (rng.standard_normal(m_loc) * band).astype(np.int64)
    v = np.clip(u + off, 0, N - 1)
    u2 = rng.integers(0, N, size=M - m_loc); v2 = rng.integers(0, N, size=M - m_loc)
    src = np.concatenate([u, u2]).astype(np.int32)
    dst = np.concatenate([v, v2]).astype(np.int32)
    return src, dst


def write_edges(path, N, src, dst):
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", N, src.size))
        inter = np.empty(2 * src.size, dtype=np.int32); inter[0::2] = src; inter[1::2] = dst
        inter.tofile(f)


def cpp_order(edges_path, mode, out_path):
    """Run the C++ ordering kernel (binary fmt in: int64 N,int64 M,2M int32; out: int64 N + N int32)."""
    t0 = time.time()
    r = subprocess.run([BIN, edges_path, mode, out_path], capture_output=True, text=True)
    cost = time.time() - t0
    if r.returncode != 0:
        print(f"  [cpp {mode}] FAILED ({r.stderr.strip()[:160]}) -- keeping identity order")
        return None, cost
    with open(out_path, "rb") as f:
        N = struct.unpack("<q", f.read(8))[0]
        newid = np.fromfile(f, dtype=np.int32, count=N)
    return newid, cost


def build_csr(src, dst, N, dev):
    """Symmetric, row-normalized adjacency as a sorted CSR tensor on the device."""
    r = np.concatenate([src, dst]).astype(np.int64); c = np.concatenate([dst, src]).astype(np.int64)
    o = np.argsort(r, kind="stable"); r = r[o]; c = c[o]
    counts = np.bincount(r, minlength=N)
    deg = counts.astype(np.float32); deg[deg == 0] = 1.0
    vals = (1.0 / deg[r]).astype(np.float32)
    crow = np.zeros(N + 1, dtype=np.int64); np.cumsum(counts, out=crow[1:])
    A = torch.sparse_csr_tensor(torch.from_numpy(crow), torch.from_numpy(c),
                                torch.from_numpy(vals), size=(N, N), device=dev)
    return A, counts


def build_row_block_csr(crow_np, col_np, val_np, lo, hi, N, dev):
    """Slice rows [lo:hi) out of a CSR (numpy arrays) into a (hi-lo) x N device CSR sub-matrix.
    This is the per-tile destination block: it gathers only the neighbors of those rows."""
    c0 = int(crow_np[lo]); c1 = int(crow_np[hi])
    sub_crow = (crow_np[lo:hi + 1] - c0).astype(np.int64)
    sub_col = col_np[c0:c1].astype(np.int64)
    sub_val = val_np[c0:c1].astype(np.float32)
    return torch.sparse_csr_tensor(torch.from_numpy(sub_crow), torch.from_numpy(sub_col),
                                   torch.from_numpy(sub_val), size=(hi - lo, N), device=dev)


def timed(fn, reps=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")                      # real temporal graph (else synthetic)
    ap.add_argument("--nodes", type=int, default=4_000_000)
    ap.add_argument("--edges", type=int, default=64_000_000)
    ap.add_argument("--l2-mb", type=float, default=32.0,          # L2 capacity target (RTX5000~32, H100~50)
                    help="L2 cache size in MB used to size the destination tile")
    ap.add_argument("--reorder", default="",                      # "" | lpa | bfs | kcore | degree (C++)
                    help="optional C++ locality reordering before tiling (lpa recommended)")
    a = ap.parse_args()
    dev = "cuda:0"
    gpu = torch.cuda.get_device_name(0)
    L2 = a.l2_mb * 1024 ** 2

    # ---- load / generate the graph (numpy edge lists) ----
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int32); dst = g.dst.astype(np.int32); M = src.size
        sizes = [(N, M)]
        print(f"CACHE-TILING gpu='{gpu}' dataset={g.name} N={N} M={M} L2={a.l2_mb:.0f}MB reorder={a.reorder or 'identity'}")
    else:
        N, M = a.nodes, a.edges
        sizes = [(N, M), (N * 2, M * 2)]                          # a couple graph sizes
        print(f"CACHE-TILING gpu='{gpu}' SYNTHETIC base N={N} M={M} L2={a.l2_mb:.0f}MB reorder={a.reorder or 'identity'}")

    feats = [64, 128, 256, 512]

    for (Ng, Mg) in sizes:
        if a.dataset:
            s, d = src, dst
        else:
            s, d = gen_graph(Ng, Mg)

        # optional C++ locality reordering: makes a destination tile's neighbors contiguous
        order_cost = 0.0
        if a.reorder:
            ep = "/tmp/zord_tile_edges.bin"; write_edges(ep, Ng, s, d)
            newid, order_cost = cpp_order(ep, a.reorder, f"/tmp/zord_tile_perm_{a.reorder}.bin")
            if newid is not None:
                s = newid[s]; d = newid[d]

        nnz = 2 * Mg                                              # symmetric -> 2 entries per edge
        feat_bytes_per_row = 4                                    # fp32
        print(f"\n=== graph N={Ng} M={Mg} nnz={nnz} order_cost={order_cost:.2f}s ===")

        A, _counts = build_csr(s, d, Ng, dev)
        crow_np = A.crow_indices().cpu().numpy()
        col_np = A.col_indices().cpu().numpy()
        val_np = A.values().cpu().numpy()

        for F in feats:
            tile = max(1, int(L2 // (F * feat_bytes_per_row)))    # rows whose gathered feats fit L2
            tile = min(tile, Ng)
            n_tiles = (Ng + tile - 1) // tile
            W1 = torch.randn(F, F, device=dev) / F ** 0.5
            X = torch.randn(Ng, F, device=dev)
            x_mb = Ng * F * 4 / 1024 ** 2

            # ---- (a) UNTILED: whole-X 2-layer aggregation ----
            def untiled():
                return torch.sparse.mm(A, torch.relu(torch.sparse.mm(A, X) @ W1))
            t_un = timed(untiled)
            bw_un = (2 * nnz * F * 4) / t_un / 1024 ** 3          # 2 gather layers over nnz entries

            # ---- (b) TILED: layer1 over full X, then layer2 by destination row blocks ----
            # H = relu(A @ X @ W1)  (the cheap, dense-feature first hop) computed once.
            # Then the second aggregation A @ H is split into destination tiles; each tile only
            # gathers H rows of its neighbors, the cache-resident working set.
            blocks = []
            for lo in range(0, Ng, tile):
                hi = min(lo + tile, Ng)
                blocks.append((lo, hi, build_row_block_csr(crow_np, col_np, val_np, lo, hi, Ng, dev)))

            out = torch.empty(Ng, F, device=dev)

            def tiled():
                H = torch.relu(torch.sparse.mm(A, X) @ W1)        # first hop (shared)
                for lo, hi, Ab in blocks:
                    out[lo:hi] = torch.sparse.mm(Ab, H)           # cache-resident destination block
                return out
            t_ti = timed(tiled)
            bw_ti = (2 * nnz * F * 4) / t_ti / 1024 ** 3

            cache_resident = tile * F * 4 / 1024 ** 2 <= a.l2_mb
            print(f"  F={F:<4} tile={tile:>9} ({n_tiles} tiles, {tile*F*4/1024**2:6.1f}MB/tile, "
                  f"X={x_mb:7.1f}MB)  untiled={t_un*1e3:8.2f}ms {bw_un:7.1f}GB/s | "
                  f"tiled={t_ti*1e3:8.2f}ms {bw_ti:7.1f}GB/s | "
                  f"speedup={t_un/t_ti:4.2f}x  tile_fits_L2={cache_resident}")

            del X, W1, blocks, out
            torch.cuda.empty_cache()

        del A
        torch.cuda.empty_cache()

    print("\n=> Tiling wins when (i) X exceeds L2 (so untiled streams HBM at ~683GB/s) AND (ii) node "
          "order is local (use --reorder lpa) so each tile's gathered features stay L2-resident, "
          "pushing achieved bandwidth toward the ~1700GB/s cache regime.")


if __name__ == "__main__":
    main()
