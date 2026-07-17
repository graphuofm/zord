#!/usr/bin/env python
"""COUNTER-INTUITIVE process tradeoff at SCALE (D29): does investing MORE in partition/layout
(k-core/locality reordering, computed in C++) make the memory-bound GNN aggregation FASTER -- enough
to win after amortizing the ordering cost over training epochs? SAME graph, SAME result (D28); only
node ORDER differs, which changes memory ACCESS LOCALITY of the SpMM aggregation.

Graph algorithms (degree/kcore/bfs orderings) run in the C++ kernel (build/graph_algos); the GNN
aggregation is timed in PyTorch. Structured temporal graph (communities) so locality is exploitable.
Reports per ordering: aggregation ms, achieved bandwidth, and the ORDERING COMPUTE COST (the upfront
investment). Net win = order_cost + N_epochs * agg_time vs the cheap (random) baseline.
  python scripts/reorder_speedup.py --nodes 8000000 --edges 100000000 --comms 4000 --feat 128
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8_000_000)
    ap.add_argument("--edges", type=int, default=100_000_000)
    ap.add_argument("--comms", type=int, default=4000)
    ap.add_argument("--intra", type=float, default=0.9)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--dataset", default="")                                # real temporal graph (else synthetic)
    a = ap.parse_args()
    dev = "cuda:0"; F = a.feat
    gpu = torch.cuda.get_device_name(0)
    t0 = time.time()
    comm = None
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int32); dst = g.dst.astype(np.int32); M = src.size
        print(f"REORDER gpu='{gpu}' dataset={g.name} N={N} M={M} F={F} bin={BIN}")
    else:
        N, M = a.nodes, a.edges
        print(f"REORDER gpu='{gpu}' SYNTHETIC N={N} M={M} comms={a.comms} intra={a.intra} F={F} bin={BIN}")
        src, dst, comm = gen_graph(N, M, a.comms, a.intra)
    edges_path = "/tmp/zord_edges.bin"; write_edges(edges_path, N, src, dst)
    print(f"  loaded/generated+wrote graph in {time.time()-t0:.1f}s")

    # orderings: name -> (newid, ordering_cost_seconds)
    rng = np.random.default_rng(1)
    orders = {}
    orders["identity"] = (np.arange(N, dtype=np.int32), 0.0)                # NATURAL order (control)
    orders["random"] = (rng.permutation(N).astype(np.int32), 0.0)           # scrambled baseline
    if comm is not None:
        t = time.time(); co = np.argsort(comm, kind="stable"); nid = np.empty(N, np.int32); nid[co] = np.arange(N)
        orders["community"] = (nid, time.time() - t)                        # oracle locality (synthetic only)
    for mode in ("degree", "kcore", "bfs", "lpa", "dfs", "slashburn", "gorder"):   # C++ kernels (full set)
        nid, cost = cpp_order(edges_path, mode, f"/tmp/zord_perm_{mode}.bin")
        if nid is not None: orders[mode] = (nid, cost)

    W1 = torch.randn(F, F, device=dev) / F ** 0.5
    nnz = 2 * M
    base = None
    for name, (newid, cost) in orders.items():
        s2 = newid[src]; d2 = newid[dst]
        A = build_csr(s2, d2, N, dev)
        X = torch.randn(N, F, device=dev)
        agg = timed(lambda: torch.sparse.mm(A, torch.relu(torch.sparse.mm(A, X) @ W1)))
        bw = (2 * nnz * F * 4) / agg / 1024 ** 3                            # 2 layers of gather
        if base is None: base = agg
        print(f"  order={name:<10} order_cost={cost:6.2f}s  agg(2-layer)={agg*1e3:8.2f}ms  "
              f"bw={bw:7.1f} GB/s  speedup_vs_random={base/agg:4.2f}x")
        del A, X; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
