#!/usr/bin/env python
"""HETEROGENEITY-MATCHED ASYMMETRIC CUT (D7): a PROCESS experiment -- on a HETEROGENEOUS cluster
(H100 / RTX6000Ada / RTX5000Ada, with very different HBM aggregation bandwidth and capacity), should
we give the strong device the DENSE graph core and the weak devices the sparse periphery, with
per-device NODE COUNTS chosen so the predicted per-device aggregation TIME is balanced? Same graph,
same result; only the region->device assignment differs, which changes the predicted makespan
(= max over devices of local-agg time) of one memory-bound aggregation step.

The aggregation is memory-bound, so we model per-device step time as a roofline:
    time_k = (local_edges_k * F * BYTES_PER_EDGE_TRAVERSAL) / hbm_bw_k
Densest nodes (highest degree / highest k-core, ranked by the C++ kernel build/graph_algos) go to the
strongest device; per-device node counts are solved so all device times match (subject to capacity).

Compares THREE region->device assignments and reports per-device edges / predicted ms + makespan:
  even           : equal node counts (capacity/bandwidth-blind baseline)
  bw-proportional: node counts proportional to hbm_bw (bandwidth-aware but degree-blind)
  hetero-matched : dense core -> strong device, node counts solved to BALANCE predicted agg time
  python scripts/hetero_matched.py --nodes 8000000 --edges 100000000 --comms 4000 --feat 128
  python scripts/hetero_matched.py --dataset <name> --feat 128
"""
import argparse, os, struct, subprocess, time
import numpy as np

BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")
BYTES_PER_EDGE_TRAVERSAL = 4.0   # fp32 feature word moved per edge per gather (memory-bound model)
N_GATHERS = 2                    # 2-layer aggregation does 2 SpMM gathers over the local edges


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
    return (np.concatenate([u, u2]).astype(np.int32), np.concatenate([v, v2]).astype(np.int32))


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


def node_degree(src, dst, N):
    return (np.bincount(src.astype(np.int64), minlength=N) +
            np.bincount(dst.astype(np.int64), minlength=N)).astype(np.int64)


def edges_per_part_by_segments(rank, src, dst, deg, N, bounds):
    """Given a 1-D ranking rank[v] in [0,N) and contiguous rank boundaries `bounds`
    (len D+1), return per-part node counts, INCIDENT-edge gather work, local edges, and cut.
    FIX (auditor C1): the memory-bound aggregation gathers EVERY incident edge of a part's
    nodes (incl. cross-part/remote neighbors -- those feature rows still get fetched), so agg
    work = sum of node DEGREE in the part (incident edges), NOT just intra-part edges. A cut
    edge is gathered by BOTH endpoints' parts -> correctly counted in both incidents."""
    D = len(bounds) - 1
    part = np.searchsorted(bounds, rank, side="right") - 1   # rank in [bounds[p],bounds[p+1]) -> p
    part = part.clip(0, D - 1).astype(np.int64)
    counts = np.bincount(part, minlength=D).astype(np.int64)
    incident = np.bincount(part, weights=deg.astype(np.float64), minlength=D).astype(np.int64)  # FULL gather work
    pu = part[src]; pv = part[dst]
    local = pu[pu == pv]
    local_edges = np.bincount(local, minlength=D).astype(np.int64)
    cut = int(np.count_nonzero(pu != pv))
    return counts, incident, local_edges, cut, part


def predict_times(local_edges, bw_gbps, F):
    """Predicted per-device agg ms from the roofline memory model."""
    bytes_moved = local_edges.astype(np.float64) * F * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS
    return bytes_moved / (bw_gbps * 1e9) * 1e3   # ms


def report(label, counts, local_edges, cut, times_ms, devs, M):
    nnz = 2 * M
    print(f"  [{label}]")
    for k, d in enumerate(devs):
        print(f"      dev{k} {d.name:<16} nodes={counts[k]:>12,d}  local_edges={local_edges[k]:>12,d}  "
              f"hbm={d.hbm_bw_gbps:6.0f}GB/s  pred_agg={times_ms[k]:8.2f}ms")
    makespan = float(times_ms.max())
    busy = float(times_ms.mean())
    print(f"      cut={cut:>12,d} ({cut/nnz*100:5.1f}% edges)  makespan={makespan:8.2f}ms  "
          f"util={busy/makespan*100:5.1f}%  (lower makespan is better)")
    return makespan


def solve_balanced_bounds(deg_sorted_cum, total_local_target, bw, usable_nodes, N):
    """Pick rank boundaries (over the DENSITY-sorted layout: rank 0 = densest -> strongest device)
    so predicted per-device agg TIME is equalized. Time_k ~ local_edges_k / bw_k. We allocate
    'edge budget' to each device proportional to bw_k (so edges_k/bw_k is constant), then translate
    that edge budget into a node-count boundary using the cumulative degree of the sorted layout.
    Respects per-device node capacity (usable_nodes)."""
    D = len(bw)
    share = np.asarray(bw, dtype=np.float64) / np.sum(bw)        # edge budget fraction per device
    target_edges = share * deg_sorted_cum[-1]                    # ~ 2*local edges in degree units
    bounds = [0]
    acc = 0.0
    idx = 0
    for k in range(D - 1):
        acc += target_edges[k]
        # advance node boundary until cumulative degree reaches the running edge target
        nb = int(np.searchsorted(deg_sorted_cum, acc, side="left"))
        nb = max(bounds[-1] + 1, min(nb, N))
        # capacity clamp: device k cannot hold more than usable_nodes[k] nodes
        nb = min(nb, bounds[-1] + max(1, int(usable_nodes[k])))
        bounds.append(nb)
    bounds.append(N)
    return np.array(bounds, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8_000_000)
    ap.add_argument("--edges", type=int, default=100_000_000)
    ap.add_argument("--comms", type=int, default=4000)
    ap.add_argument("--intra", type=float, default=0.9)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--rank-by", default="degree", choices=["degree", "kcore"])
    ap.add_argument("--dataset", default="")                                # real temporal graph (else synthetic)
    a = ap.parse_args()
    F = a.feat
    from zord.profiler.cluster_profile import hetcluster
    cluster = hetcluster()
    devs = cluster.devices
    D = len(devs)
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    # per-device node capacity: usable HBM / (one fp32 feature row F*4 bytes), a generous upper bound
    usable_nodes = np.array([d.usable_mem / (F * 4) for d in devs], dtype=np.float64)

    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int32); dst = g.dst.astype(np.int32); M = src.size
        print(f"HETERO dataset={g.name} N={N} M={M} F={F} D={D} rank-by={a.rank_by} bin={BIN}")
    else:
        N, M = a.nodes, a.edges
        print(f"HETERO SYNTHETIC N={N} M={M} comms={a.comms} intra={a.intra} F={F} D={D} "
              f"rank-by={a.rank_by} bin={BIN}")
        src, dst = gen_graph(N, M, a.comms, a.intra)
    print("  devices: " + " | ".join(
        f"{d.name} bw={d.hbm_bw_gbps:.0f}GB/s mem={d.usable_mem/1024**3:.0f}GB r={d.throughput:.2f}"
        for d in devs))
    print(f"  loaded/generated graph in {time.time()-t0:.1f}s")

    edges_path = "/tmp/zord_hm_edges.bin"; write_edges(edges_path, N, src, dst)

    # DENSITY RANKING from the C++ kernel: rank 0 = densest node -> assigned to the strongest device.
    # degree mode already emits rank 0 = highest degree. kcore emits rank 0 = lowest core (peeled
    # first), so we REVERSE it to make rank 0 = highest core (densest).
    crank, cost = cpp_order(edges_path, a.rank_by, f"/tmp/zord_hm_perm_{a.rank_by}.bin")
    if crank is None:
        print("  ABORT: density ranking failed."); return
    rank = crank.astype(np.int64)
    if a.rank_by == "kcore":
        rank = (N - 1) - rank                          # highest core -> rank 0 (densest first)
    print(f"  density ranking ({a.rank_by}) computed in {cost:.2f}s (C++)")

    deg = node_degree(src, dst, N)
    # cumulative degree along the density-sorted layout (rank order): index = rank, value = node deg
    deg_by_rank = np.empty(N, dtype=np.float64)
    deg_by_rank[rank] = deg.astype(np.float64)
    deg_cum = np.cumsum(deg_by_rank)                   # deg_cum[r] = sum of degrees of ranks 0..r

    # Devices are listed strongest-first in hetcluster() (H100, then 6000, then 5000). Ensure the
    # strongest device (max bw) owns the lowest ranks (densest core) by ordering bounds along rank.
    order_strong = np.argsort(-bw)                     # device indices, strongest first
    bw_sorted = bw[order_strong]
    usable_sorted = usable_nodes[order_strong]

    # ---- assignment 1: EVEN (equal node counts) ----
    even_bounds_sorted = np.linspace(0, N, D + 1).astype(np.int64)
    counts_s, inc_s, le_s, cut, _ = edges_per_part_by_segments(rank, src, dst, deg, N, even_bounds_sorted)
    # map sorted-segment results back to physical device ids
    counts = np.empty(D, np.int64); le = np.empty(D, np.int64); inc = np.empty(D, np.int64)
    counts[order_strong] = counts_s; le[order_strong] = le_s; inc[order_strong] = inc_s
    t_even = predict_times(inc, bw, F)   # incident gather work (auditor C1 fix)
    mk_even = report("even (equal nodes)", counts, le, cut, t_even, devs, M)

    # ---- assignment 2: BANDWIDTH-PROPORTIONAL (node counts ~ hbm_bw) ----
    frac = bw_sorted / bw_sorted.sum()
    bp_bounds_sorted = np.concatenate([[0], np.cumsum((frac * N).astype(np.int64))])
    bp_bounds_sorted[-1] = N
    bp_bounds_sorted = np.maximum.accumulate(bp_bounds_sorted).astype(np.int64)
    counts_s, inc_s, le_s, cut, _ = edges_per_part_by_segments(rank, src, dst, deg, N, bp_bounds_sorted)
    counts[order_strong] = counts_s; le[order_strong] = le_s; inc[order_strong] = inc_s
    t_bp = predict_times(inc, bw, F)
    mk_bp = report("bw-proportional (nodes ~ hbm_bw)", counts, le, cut, t_bp, devs, M)

    # ---- assignment 3: HETERO-MATCHED (dense core -> strong dev; node counts balance agg TIME) ----
    hm_bounds_sorted = solve_balanced_bounds(deg_cum, deg_cum[-1], bw_sorted, usable_sorted, N)
    counts_s, inc_s, le_s, cut, _ = edges_per_part_by_segments(rank, src, dst, deg, N, hm_bounds_sorted)
    counts[order_strong] = counts_s; le[order_strong] = le_s; inc[order_strong] = inc_s
    t_hm = predict_times(inc, bw, F)
    mk_hm = report("hetero-matched (core->strong, time-balanced)", counts, le, cut, t_hm, devs, M)

    best = min(mk_even, mk_bp, mk_hm)
    print(f"  => makespan  even={mk_even:.2f}ms  bw-prop={mk_bp:.2f}ms  hetero-matched={mk_hm:.2f}ms  "
          f"(best={'hetero-matched' if mk_hm == best else 'bw-prop' if mk_bp == best else 'even'}; "
          f"hetero-matched is {mk_even/mk_hm:.2f}x vs even, {mk_bp/mk_hm:.2f}x vs bw-prop)")


if __name__ == "__main__":
    main()
