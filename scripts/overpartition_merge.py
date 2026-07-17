#!/usr/bin/env python
"""OVER-PARTITION THEN MERGE (D2): a PROCESS experiment -- does splitting the graph into MANY more
parts than devices (K >> D) and then greedily MERGING them back to D balanced parts cut FEWER edges
(and give better aggregation locality) than a DIRECT D-way split? Same graph, same result; only the
vertex->device assignment differs, which changes the SpatialCut (cross-device edges) and the memory
ACCESS LOCALITY of the per-device SpMM aggregation.

Pipeline (graph algorithms in the C++ kernel build/graph_algos; aggregation timed in PyTorch):
  1. C++ lpa ordering -> lpa_rank[v] (a locality-preserving 1-D layout that groups communities).
  2. OVER-PARTITION: block[v] = lpa_rank[v]*K//N  (K = mult*D contiguous blocks of the lpa layout).
  3. MERGE: build the KxK inter-block edge-weight matrix, then agglomeratively merge the
     highest-affinity (most shared edges) block-groups, subject to size <= cap*N/D, until D remain.
  4. DIRECT baseline: block[v] = lpa_rank[v]*D//N  (a straight D-way slice of the SAME layout).
Reports for BOTH assignments: SpatialCut (edges crossing the D parts), part sizes (balance), timed
2-layer agg ms, and achieved aggregation bandwidth -- so we can see whether over-partition+merge wins.
  python scripts/overpartition_merge.py --nodes 8000000 --edges 100000000 --comms 4000 --devices 4 --feat 128
  python scripts/overpartition_merge.py --dataset <name> --devices 4 --feat 128
"""
import argparse, os, struct, subprocess, time
import numpy as np
import torch

BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")


def gen_graph(N, M, C, intra, seed=0):
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, C, size=N).astype(np.int64)
    order = np.argsort(comm, kind="stable")
    bounds = np.searchsorted(comm[order], np.arange(C + 1))
    m_in = int(M * intra)
    u = rng.integers(0, N, size=m_in); cu = comm[u]
    lo = bounds[cu].astype(np.int64); hi = bounds[cu + 1].astype(np.int64)
    pick = lo + (rng.random(m_in) * np.maximum(1, hi - lo)).astype(np.int64)
    v = order[np.minimum(pick, N - 1)]
    u2 = rng.integers(0, N, size=M - m_in); v2 = rng.integers(0, N, size=M - m_in)
    return (np.concatenate([u, u2]).astype(np.int32), np.concatenate([v, v2]).astype(np.int32), comm)


def write_edges(path, N, src, dst):
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", N, src.size))
        inter = np.empty(2 * src.size, dtype=np.int32); inter[0::2] = src; inter[1::2] = dst
        inter.tofile(f)


def cpp_order(edges_path, mode, out_path):
    t0 = time.time()
    r = subprocess.run([BIN, edges_path, mode, out_path], capture_output=True, text=True)
    cost = time.time() - t0
    if r.returncode != 0:
        print(f"  [cpp {mode}] FAILED: {r.stderr.strip()[:200]}"); return None, cost
    with open(out_path, "rb") as f:
        N = struct.unpack("<q", f.read(8))[0]
        newid = np.fromfile(f, dtype=np.int32, count=N)
    return newid, cost


def build_csr(src, dst, N, dev):
    r = np.concatenate([src, dst]).astype(np.int64); c = np.concatenate([dst, src]).astype(np.int64)
    o = np.argsort(r, kind="stable"); r = r[o]; c = c[o]
    counts = np.bincount(r, minlength=N)
    deg = counts.astype(np.float32); deg[deg == 0] = 1.0
    vals = (1.0 / deg[r]).astype(np.float32)
    crow = np.zeros(N + 1, dtype=np.int64); np.cumsum(counts, out=crow[1:])
    return torch.sparse_csr_tensor(torch.from_numpy(crow), torch.from_numpy(c),
                                   torch.from_numpy(vals), size=(N, N), device=dev)


def timed(fn, reps=15, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps


def spatial_cut(part, src, dst):
    """# undirected edges whose endpoints land in different parts."""
    return int(np.count_nonzero(part[src] != part[dst]))


def block_edge_matrix(block, src, dst, K):
    """KxK symmetric inter-block edge-weight matrix W (undirected: each edge counted once per
    unordered block pair; diagonal = intra-block edges). Built with bincount on the K*bu+bv key."""
    bu = block[src].astype(np.int64); bv = block[dst].astype(np.int64)
    lo = np.minimum(bu, bv); hi = np.maximum(bu, bv)
    key = lo * K + hi
    W = np.bincount(key, minlength=K * K).reshape(K, K).astype(np.float64)
    # fold upper triangle onto a symmetric matrix (diagonal already correct from lo==hi)
    Wsym = W + W.T
    np.fill_diagonal(Wsym, np.diag(W))
    return Wsym


def greedy_merge(W, sizes, D, cap):
    """Agglomeratively merge K blocks (affinity = shared cross edges W[i][j]) down to D groups,
    never letting a merged group exceed `cap`. Returns label[k] in [0,D) for each original block.
    Pure-numpy O(K^2 * merges); K is small (mult*D), so this is cheap relative to the graph work."""
    K = W.shape[0]
    W = W.copy(); np.fill_diagonal(W, 0.0)          # only inter-group affinity drives merging
    sizes = sizes.astype(np.float64).copy()
    alive = list(range(K))
    members = {k: [k] for k in range(K)}            # group id -> original block ids
    n_groups = K
    while n_groups > D:
        best = (-1.0, -1, -1)
        for ai in range(len(alive)):
            i = alive[ai]
            for aj in range(ai + 1, len(alive)):
                j = alive[aj]
                if sizes[i] + sizes[j] > cap:
                    continue
                aff = W[i, j]
                if aff > best[0]:
                    best = (aff, i, j)
        if best[1] < 0:
            # no feasible merge under the cap; relax to the smallest-combined feasible-ish pair
            best = (-1.0, -1, -1)
            for ai in range(len(alive)):
                i = alive[ai]
                for aj in range(ai + 1, len(alive)):
                    j = alive[aj]
                    score = -(sizes[i] + sizes[j])   # prefer keeping groups small
                    if score > best[0] or best[1] < 0:
                        best = (score, i, j)
        _, i, j = best
        # merge j into i: combine affinity rows/cols and sizes
        W[i, :] += W[j, :]; W[:, i] += W[:, j]
        W[i, i] = 0.0
        W[j, :] = 0.0; W[:, j] = 0.0
        sizes[i] += sizes[j]; sizes[j] = 0.0
        members[i].extend(members[j]); del members[j]
        alive.remove(j)
        n_groups -= 1
    label = np.empty(W.shape[0], dtype=np.int64)
    for new_id, g in enumerate(alive):
        for k in members[g]:
            label[k] = new_id
    return label


def agg_and_report(name, part, src, dst, N, F, dev, W1, nnz, base_agg):
    cut = spatial_cut(part, src, dst)
    counts = np.bincount(part, minlength=int(part.max()) + 1)
    bal = counts.max() / max(1.0, counts.mean())
    A = build_csr(src, dst, N, dev)
    X = torch.randn(N, F, device=dev)
    agg = timed(lambda: torch.sparse.mm(A, torch.relu(torch.sparse.mm(A, X) @ W1)))
    bw = (2 * nnz * F * 4) / agg / 1024 ** 3
    spd = (base_agg / agg) if base_agg else 1.0
    print(f"  {name:<22} cut={cut:>12,d} ({cut / nnz * 100:5.1f}% edges)  parts={counts.tolist()}  "
          f"imbal={bal:4.2f}  agg(2-layer)={agg * 1e3:8.2f}ms  bw={bw:7.1f} GB/s  speedup={spd:4.2f}x")
    del A, X; torch.cuda.empty_cache()
    return cut, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8_000_000)
    ap.add_argument("--edges", type=int, default=100_000_000)
    ap.add_argument("--comms", type=int, default=4000)
    ap.add_argument("--intra", type=float, default=0.9)
    ap.add_argument("--devices", type=int, default=4)                        # D = target #parts
    ap.add_argument("--mult", type=int, default=16)                         # K = mult * D over-parts
    ap.add_argument("--cap", type=float, default=1.1)                       # merged size <= cap*N/D
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--dataset", default="")                                # real temporal graph (else synthetic)
    a = ap.parse_args()
    dev = "cuda:0"; F = a.feat; D = a.devices; K = a.mult * D
    gpu = torch.cuda.get_device_name(0)
    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int32); dst = g.dst.astype(np.int32); M = src.size
        print(f"OVERPART gpu='{gpu}' dataset={g.name} N={N} M={M} D={D} K={K} cap={a.cap} F={F} bin={BIN}")
    else:
        N, M = a.nodes, a.edges
        print(f"OVERPART gpu='{gpu}' SYNTHETIC N={N} M={M} comms={a.comms} intra={a.intra} "
              f"D={D} K={K} cap={a.cap} F={F} bin={BIN}")
        src, dst, _ = gen_graph(N, M, a.comms, a.intra)
    edges_path = "/tmp/zord_op_edges.bin"; write_edges(edges_path, N, src, dst)
    print(f"  loaded/generated+wrote graph in {time.time()-t0:.1f}s")

    # 1) C++ lpa layout (shared by both assignments).
    lpa_rank, lpa_cost = cpp_order(edges_path, "lpa", "/tmp/zord_op_perm_lpa.bin")
    if lpa_rank is None:
        print("  ABORT: lpa ordering failed."); return
    print(f"  lpa layout computed in {lpa_cost:.2f}s (C++)")

    W1 = torch.randn(F, F, device=dev) / F ** 0.5
    nnz = 2 * M
    cap_nodes = a.cap * N / D

    # 2) DIRECT D-way split of the lpa layout (baseline).
    direct = (lpa_rank.astype(np.int64) * D // N).clip(0, D - 1)
    base_cut, base_agg = agg_and_report("direct-lpa-D", direct, src, dst, N, F, dev, W1, nnz, None)

    # 3) OVER-PARTITION into K blocks, then GREEDY MERGE down to D.
    t_op = time.time()
    block = (lpa_rank.astype(np.int64) * K // N).clip(0, K - 1)
    blk_sizes = np.bincount(block, minlength=K).astype(np.float64)
    Wmat = block_edge_matrix(block, src, dst, K)
    merge_label = greedy_merge(Wmat, blk_sizes, D, cap_nodes)              # block-group -> [0,D)
    merged = merge_label[block]                                            # vertex -> [0,D)
    op_cost = time.time() - t_op
    inter_before = (Wmat.sum() - np.trace(Wmat)) / 2.0
    print(f"  over-partition+merge built in {op_cost:.2f}s (K={K} blocks, KxK affinity + greedy agglomerate; "
          f"inter-block edges before merge={inter_before/nnz*100:5.1f}% of edges)")
    op_cut, op_agg = agg_and_report("overpart-merge-D", merged, src, dst, N, F, dev, W1, nnz, base_agg)

    # 4) Verdict.
    dcut = (base_cut - op_cut) / max(1, base_cut) * 100
    dagg = (base_agg - op_agg) / base_agg * 100
    print(f"  => overpart+merge vs direct: cut {dcut:+.1f}%  agg-time {dagg:+.1f}%  "
          f"({'WINS' if op_cut < base_cut else 'loses'} on cut, "
          f"{'faster' if op_agg < base_agg else 'slower'} on agg)")


if __name__ == "__main__":
    main()
