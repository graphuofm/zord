#!/usr/bin/env python
"""ATTRIBUTE-AWARE PLACEMENT on a HETEROGENEOUS cluster (process-only, predicted/measured cost):
when nodes carry HETEROGENEOUS attribute widths F_v (some nodes are attribute-heavy, e.g. rich edge
features pooled to the node, or a long feature vector), does placing the attribute-heavy nodes on the
HIGH-MEMORY / HIGH-BANDWIDTH GPUs (and balancing predicted aggregation TIME) beat structure-only and
even placement? Same graph, same result; only node->device assignment differs.

The aggregation is memory-bound, so per-node aggregation cost is modeled as a roofline:
    bytes_v   = deg_v * F_v * BYTES_PER_WORD * N_GATHERS         # gather F_v words per incident edge
    time_dev  = (sum_{v in dev} bytes_v) / hbm_bw_dev
A node's "weight" is its attribute-scaled traffic deg_v * F_v (NOT just degree). The capacity bound
is attribute-scaled too: node v needs F_v * BYTES_PER_WORD of VRAM for its feature row.

We compare THREE node->device assignments and report per-device {nodes, attr-traffic, predicted ms},
the makespan (= max device time), and the cut (cross-device edges):
  even               : equal NODE counts (capacity/bandwidth/attribute-blind)
  structure-only     : densest nodes (degree/kcore via C++ build/graph_algos) -> strongest device,
                       node counts balance predicted time using DEGREE only (attribute-blind)
  attr-memory-matched: attribute-heavy nodes -> high-mem/high-bw device; node counts balance
                       predicted AGG TIME using attribute-scaled traffic deg_v*F_v, respecting the
                       attribute-scaled VRAM capacity.

Per-node attribute width F_v is either REAL (pooled native edge-feature dim / norm from an attributed
dataset, scaled to a width band) or SYNTHETIC heterogeneous (--attr-skew controls heterogeneity).
Graph algorithms run in the C++ kernel (build/graph_algos); placement math is numpy. NEVER networkx.
  python scripts/attr_aware_partition.py --dataset jodie-wikipedia --feat 172
  python scripts/attr_aware_partition.py --nodes 4000000 --edges 60000000 --feat 128 --attr-skew 3.0
"""
import argparse, os, struct, subprocess, time
import numpy as np

BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")
BYTES_PER_WORD = 4.0     # fp32 feature word moved per edge per gather (memory-bound model)
N_GATHERS = 2            # 2-layer aggregation does 2 SpMM gathers over local edges


def gen_graph(N, M, C, intra, seed=0):
    """Community-structured edges so a locality ordering is meaningful (matches hetero_matched)."""
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


def real_attr_width(g, N, feat_band):
    """Per-node attribute width F_v from a REAL attributed graph: pool incident edge-feature L2-norm
    to each node, then map the norm distribution onto a width band [Fe_min .. feat_band]. Nodes with
    richer/larger features get a wider attribute vector. Falls back to None if no edge features."""
    if g.efeat is None:
        return None
    enorm = np.linalg.norm(g.efeat.astype(np.float64), axis=1)   # [E]
    acc = np.zeros(N, dtype=np.float64); cnt = np.zeros(N, dtype=np.float64)
    s = g.src.astype(np.int64); d = g.dst.astype(np.int64)
    np.add.at(acc, s, enorm); np.add.at(cnt, s, 1.0)
    np.add.at(acc, d, enorm); np.add.at(cnt, d, 1.0)
    cnt[cnt == 0] = 1.0
    pooled = acc / cnt
    lo, hi = np.percentile(pooled, 1), np.percentile(pooled, 99)
    if hi <= lo:
        return np.full(N, feat_band, dtype=np.int64)
    frac = np.clip((pooled - lo) / (hi - lo), 0.0, 1.0)
    fmin = max(1, feat_band // 8)
    return (fmin + frac * (feat_band - fmin)).round().astype(np.int64)


def synth_attr_width(N, feat_band, skew, seed=0):
    """Synthetic HETEROGENEOUS per-node attribute width: a power-law-ish band so some nodes are
    attribute-heavy (width feat_band) and most are light. `skew`>1 makes it more heterogeneous."""
    rng = np.random.default_rng(seed)
    fmin = max(1, feat_band // 8)
    u = rng.random(N) ** max(1e-6, skew)          # skew>1 pushes mass toward 0 (light) with a heavy tail
    return (fmin + u * (feat_band - fmin)).round().astype(np.int64)


def predict_times(traffic_words, bw_gbps):
    """Predicted per-device agg ms: traffic_words is sum of deg_v*F_v over the device's nodes."""
    bytes_moved = traffic_words.astype(np.float64) * BYTES_PER_WORD * N_GATHERS
    return bytes_moved / (bw_gbps * 1e9) * 1e3    # ms


def cut_for(part, src, dst):
    pu = part[src.astype(np.int64)]; pv = part[dst.astype(np.int64)]
    return int(np.count_nonzero(pu != pv))


def assign_by_balance(rank, weight_by_node, bw_sorted, mem_by_node, cap_sorted_bytes, N):
    """Greedily cut the RANK-sorted node order (rank 0 -> strongest/first device) into D contiguous
    segments whose per-device WEIGHT (sum of weight_by_node) is proportional to bw, so weight/bw is
    equalized; CLAMP each boundary so a device's VRAM-byte load (sum of mem_by_node) does not exceed
    its capacity cap_sorted_bytes. Returns rank boundaries (len D+1)."""
    D = len(bw_sorted)
    # weight + memory along the rank-sorted layout
    w_by_rank = np.empty(N, dtype=np.float64)
    w_by_rank[rank] = weight_by_node.astype(np.float64)
    wcum = np.cumsum(w_by_rank)
    m_by_rank = np.empty(N, dtype=np.float64)
    m_by_rank[rank] = mem_by_node.astype(np.float64)
    mcum = np.concatenate([[0.0], np.cumsum(m_by_rank)])   # mcum[i] = VRAM bytes of ranks [0,i)
    total = wcum[-1] if wcum[-1] > 0 else 1.0
    share = np.asarray(bw_sorted, dtype=np.float64) / np.sum(bw_sorted)
    bounds = [0]; acc = 0.0
    for k in range(D - 1):
        acc += share[k] * total
        nb = int(np.searchsorted(wcum, acc, side="left"))
        nb = max(bounds[-1] + 1, min(nb, N))
        # capacity clamp: device k can hold at most cap_sorted_bytes[k] VRAM bytes of feature rows
        cap_bytes = cap_sorted_bytes[k]
        max_nb = int(np.searchsorted(mcum, mcum[bounds[-1]] + cap_bytes, side="right")) - 1
        nb = max(bounds[-1] + 1, min(nb, max(bounds[-1] + 1, max_nb)))
        bounds.append(nb)
    bounds.append(N)
    return np.array(bounds, dtype=np.int64)


def segment_stats(rank, bounds, src, dst, N, traffic_by_node, mem_by_node):
    """For contiguous rank segments (boundaries `bounds`), return per-segment node counts, summed
    attribute traffic (deg*F), summed VRAM bytes (F*word), the part label per node, and cut edges."""
    D = len(bounds) - 1
    part = np.searchsorted(bounds, rank, side="right") - 1
    part = part.clip(0, D - 1).astype(np.int64)
    counts = np.bincount(part, minlength=D).astype(np.int64)
    traffic = np.bincount(part, weights=traffic_by_node, minlength=D)
    mem = np.bincount(part, weights=mem_by_node, minlength=D)
    cut = cut_for(part, src, dst)
    return counts, traffic, mem, part, cut


def report(label, counts, traffic, mem_bytes, cap_bytes, cut, times_ms, devs, M):
    nnz = 2 * M
    print(f"  [{label}]")
    over = False
    for k, d in enumerate(devs):
        cap = cap_bytes[k]
        oc = mem_bytes[k] > cap
        over = over or oc
        flag = "  !!OVER-CAP" if oc else ""
        print(f"      dev{k} {d.name:<16} nodes={counts[k]:>11,d}  attr_traffic={traffic[k]:>16,.0f}  "
              f"hbm={d.hbm_bw_gbps:6.0f}GB/s  feat_mem={mem_bytes[k]/1024**3:6.2f}/{cap/1024**3:5.1f}GB"
              f"  pred_agg={times_ms[k]:8.2f}ms{flag}")
    makespan = float(times_ms.max())
    busy = float(times_ms.mean())
    print(f"      cut={cut:>12,d} ({cut/nnz*100:5.1f}% edges)  makespan={makespan:8.2f}ms  "
          f"util={busy/makespan*100:5.1f}%  fits={'NO' if over else 'yes'}  (lower makespan is better)")
    return makespan, over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=4_000_000)
    ap.add_argument("--edges", type=int, default=60_000_000)
    ap.add_argument("--comms", type=int, default=2000)
    ap.add_argument("--intra", type=float, default=0.9)
    ap.add_argument("--feat", type=int, default=128)               # attribute-width band (max F_v)
    ap.add_argument("--attr-skew", type=float, default=3.0)        # synthetic attribute heterogeneity
    ap.add_argument("--rank-by", default="degree", choices=["degree", "kcore"])
    ap.add_argument("--dataset", default="")                        # real (attributed) temporal graph
    a = ap.parse_args()
    F = a.feat

    from zord.profiler.cluster_profile import hetcluster
    cluster = hetcluster()
    devs = cluster.devices
    D = len(devs)
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    usable_bytes = np.array([d.usable_mem for d in devs], dtype=np.float64)

    t0 = time.time()
    g = None
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int32); dst = g.dst.astype(np.int32); M = src.size
        fe = int(g.efeat.shape[1]) if g.efeat is not None else 0
        print(f"ATTR-PART dataset={g.name} N={N} M={M} feat_band={F} native_Fe={fe} D={D} "
              f"rank-by={a.rank_by} bin={BIN}")
    else:
        N, M = a.nodes, a.edges
        src, dst = gen_graph(N, M, a.comms, a.intra)
        print(f"ATTR-PART SYNTHETIC N={N} M={M} comms={a.comms} feat_band={F} attr_skew={a.attr_skew} "
              f"D={D} rank-by={a.rank_by} bin={BIN}")
    print("  devices: " + " | ".join(
        f"{d.name} bw={d.hbm_bw_gbps:.0f}GB/s mem={d.usable_mem/1024**3:.0f}GB" for d in devs))

    # ---- per-node attribute width F_v (heterogeneous) ----
    Fv = None
    if g is not None:
        Fv = real_attr_width(g, N, F)
    if Fv is None:
        Fv = synth_attr_width(N, F, a.attr_skew)
        if a.dataset:
            print("  (dataset has no edge features -> using SYNTHETIC heterogeneous attribute widths)")
    print(f"  loaded/generated graph + attr widths in {time.time()-t0:.1f}s  "
          f"F_v: min={Fv.min()} med={int(np.median(Fv))} max={Fv.max()} mean={Fv.mean():.1f}")

    deg = node_degree(src, dst, N)
    traffic_by_node = deg.astype(np.float64) * Fv.astype(np.float64)   # attribute-scaled traffic
    mem_by_node = Fv.astype(np.float64) * BYTES_PER_WORD               # per-node feature-row VRAM
    cap_bytes = usable_bytes                                           # device VRAM budget (bytes)

    edges_path = "/tmp/zord_attr_edges.bin"; write_edges(edges_path, N, src, dst)

    # density ranking from C++ (rank 0 = densest); reverse kcore so highest core = rank 0.
    crank, cost = cpp_order(edges_path, a.rank_by, f"/tmp/zord_attr_perm_{a.rank_by}.bin")
    if crank is None:
        print("  ABORT: density ranking failed."); return
    deg_rank = crank.astype(np.int64)
    if a.rank_by == "kcore":
        deg_rank = (N - 1) - deg_rank
    print(f"  density ranking ({a.rank_by}) computed in {cost:.2f}s (C++)")

    # attribute ranking: rank 0 = attribute-HEAVIEST node (by attribute-scaled traffic deg*F).
    attr_rank = np.empty(N, dtype=np.int64)
    attr_rank[np.argsort(-traffic_by_node, kind="stable")] = np.arange(N)

    # devices strongest-first by bandwidth so rank 0 (densest/heaviest) -> strongest device.
    order_strong = np.argsort(-bw)
    bw_sorted = bw[order_strong]
    cap_sorted = cap_bytes[order_strong]

    def to_phys(arr_sorted):
        out = np.empty(D, dtype=arr_sorted.dtype); out[order_strong] = arr_sorted; return out

    results = {}

    # ---- 1: EVEN (equal node counts) ----
    even_bounds = np.linspace(0, N, D + 1).astype(np.int64)
    # use the density rank just to define a stable contiguous identity layout for even cut
    counts_s, traf_s, mem_s, part, cut = segment_stats(deg_rank, even_bounds, src, dst, N,
                                                        traffic_by_node, mem_by_node)
    counts = to_phys(counts_s); traf = to_phys(traf_s); mem = to_phys(mem_s)
    t_even = predict_times(traf, bw)
    mk_even, ov_even = report("even (equal nodes)", counts, traf, mem, cap_bytes, cut, t_even, devs, M)
    results["even"] = (mk_even, ov_even)

    # ---- 2: STRUCTURE-ONLY (dense core -> strong dev; balance by DEGREE, attribute-blind) ----
    so_bounds_sorted = assign_by_balance(deg_rank, deg.astype(np.float64), bw_sorted,
                                          mem_by_node, cap_sorted, N)
    counts_s, traf_s, mem_s, part, cut = segment_stats(deg_rank, so_bounds_sorted, src, dst, N,
                                                        traffic_by_node, mem_by_node)
    counts = to_phys(counts_s); traf = to_phys(traf_s); mem = to_phys(mem_s)
    t_so = predict_times(traf, bw)
    mk_so, ov_so = report("structure-only (degree-balanced, attribute-blind)",
                          counts, traf, mem, cap_bytes, cut, t_so, devs, M)
    results["structure-only"] = (mk_so, ov_so)

    # ---- 3: ATTR-MEMORY-MATCHED (heavy attrs -> strong/high-mem dev; balance attr traffic) ----
    am_bounds_sorted = assign_by_balance(attr_rank, traffic_by_node, bw_sorted,
                                         mem_by_node, cap_sorted, N)
    counts_s, traf_s, mem_s, part, cut = segment_stats(attr_rank, am_bounds_sorted, src, dst, N,
                                                        traffic_by_node, mem_by_node)
    counts = to_phys(counts_s); traf = to_phys(traf_s); mem = to_phys(mem_s)
    t_am = predict_times(traf, bw)
    mk_am, ov_am = report("attr-memory-matched (attr-heavy->strong, traffic-balanced)",
                          counts, traf, mem, cap_bytes, cut, t_am, devs, M)
    results["attr-memory-matched"] = (mk_am, ov_am)

    # ---- verdict ----
    feasible = {k: v[0] for k, v in results.items() if not v[1]}
    print(f"  => makespan  even={mk_even:.2f}ms  structure-only={mk_so:.2f}ms  "
          f"attr-memory-matched={mk_am:.2f}ms")
    best = min(feasible, key=feasible.get) if feasible else min(results, key=lambda k: results[k][0])
    print(f"     attribute-aware speedup vs even={mk_even/mk_am:.2f}x  vs structure-only="
          f"{mk_so/mk_am:.2f}x   best(feasible)={best}")
    cv = Fv.std() / Fv.mean() if Fv.mean() else 0.0
    print(f"     attribute heterogeneity (CoV of F_v)={cv:.2f}: "
          + ("HIGH -> attribute-aware placement matters" if cv > 0.3
             else "LOW -> structure-only ~ attribute-aware (attrs uniform)"))


if __name__ == "__main__":
    main()
