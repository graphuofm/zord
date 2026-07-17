#!/usr/bin/env python
"""PARTITION-QUALITY-vs-COST tradeoff bench (D1): does spending MORE on partitioning buy
enough edge-CUT reduction to amortize over training epochs?

SAME graph, MANY partitioners; per partitioner we report:
  - cut%      : fraction of (undirected) edges whose endpoints land on DIFFERENT devices
                (the comm-driving quantity; lower is better).
  - balance   : max/avg part size BY NODE COUNT (1.00 == perfectly even; the load skew).
  - edge_imb  : max/avg LOCAL-edge count (compute skew -- the straggler proxy).
  - part_s    : PARTITION TIME -- the "spend more on partitioning" cost we are trading against.
  - comm_pred : predicted boundary-feature comm time on the measured HetCluster profile (the
                cost model's bandwidth-WEIGHTED cut: cutting a slow Ethernet link costs
                ~2700x an NVLink one, so cut% alone undersells a topology-aware split).

Partitioners: hash, caphash, fennel (zord.partition.baselines); LDG and spectral/Fiedler
(implemented inline); METIS (pymetis, skipped if missing); and the C++ `lpa` label-propagation
ORDERING sliced into D contiguous blocks (skipped if the build/graph_algos binary is absent).
Optional deps (scipy, pymetis, the C++ binary) degrade gracefully -- a missing one is reported
as a SKIP line, never a crash.

  # synthetic (default): community-structured temporal graph
  python scripts/partitioner_bench.py --nodes 200000 --edges 2000000 --comms 64 --intra 0.9 --devices 3
  # real staged dataset
  python scripts/partitioner_bench.py --dataset mathoverflow --devices 3
"""
import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np

# Make `zord` importable when run straight from the repo (scripts/ is not a package
# and zord may not be pip-installed in the run env). Harmless if it's already importable.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from zord.partition.base import Partitioner          # _summarize: cut/balance bookkeeping
from zord.partition.baselines import (HashPartitioner, CapacityProportionalHash,
                                      FennelPartitioner, MetisPartitioner)
from zord.partition.hetero import _build_csr
from zord.partition.cost_model import CostParams, device_comm_sec

# C++ graph kernels (binary format: int64 N, int64 M, then 2*M int32 interleaved src,dst;
# output int64 N + N int32 newid). Built via:
#   g++ -O3 -std=c++17 -fopenmp src/zord/cpp/graph_algos.cpp -o build/graph_algos
_BIN = os.environ.get(
    "ZORD_GRAPH_BIN",
    os.path.join(os.path.dirname(_SRC), "build", "graph_algos"))


# ---------------------------------------------------------------------------
# synthetic graph: community-structured so a locality-aware split has cut to win.
# (mirrors scripts/reorder_speedup.py gen_graph: `intra` fraction of edges stay
#  inside a node's community, the rest are uniform-random noise.)
# ---------------------------------------------------------------------------
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
    src = np.concatenate([u, u2]).astype(np.int64)
    dst = np.concatenate([v, v2]).astype(np.int64)
    return src, dst, comm


# ---------------------------------------------------------------------------
# inline partitioners not in baselines
# ---------------------------------------------------------------------------
def ldg_assign(src, dst, N, P):
    """Linear Deterministic Greedy streaming partition (Stanton & Kliot, KDD'12).
    For each node in first-appearance order, place it on the device holding the
    most of its already-placed neighbors, weighted by remaining capacity:
        score(d) = neighbors_on(d) * (1 - load[d] / cap)
    Cheap, single-pass, locality-aware -- the streaming-greedy point between
    capacity-blind hash and offline METIS."""
    indptr, adj = _build_csr(src, dst, N)
    # first-appearance order over the (time-sorted) edge stream
    both = np.empty(2 * src.shape[0], dtype=np.int64)
    both[0::2] = src; both[1::2] = dst
    first = np.full(N, both.shape[0], dtype=np.int64)
    np.minimum.at(first, both, np.arange(both.shape[0]))
    order = np.argsort(first, kind="stable")

    cap = max(1.0, N / P)
    assignment = np.full(N, -1, dtype=np.int32)
    load = np.zeros(P, dtype=np.float64)
    for v in order:
        nb = adj[indptr[v]:indptr[v + 1]]
        dv = assignment[nb]
        placed = dv[dv >= 0]
        cnt = (np.bincount(placed, minlength=P).astype(np.float64)
               if placed.size else np.zeros(P))
        weight = 1.0 - load / cap            # penalize near-full devices
        score = cnt * weight
        # break ties (e.g. an isolated/first node: all cnt==0) toward the emptiest device
        d = int(np.lexsort((-(-load), score))[-1]) if placed.size else int(load.argmin())
        assignment[v] = d
        load[d] += 1
    return assignment


def spectral_assign(src, dst, N, P, max_nodes=2_000_000):
    """Spectral / Fiedler partition: the bottom non-trivial eigenvectors of the
    NORMALIZED Laplacian L_sym = I - D^-1/2 A D^-1/2 embed nodes so that
    well-connected groups cluster; we split that embedding into P parts.
      P == 2 -> sign of the Fiedler vector (classic recursive-bisection base case).
      P  > 2 -> k-means on the bottom-(P) eigenvectors (spectral clustering), with a
                quantile fallback if sklearn is absent.
    Capped to < ~max_nodes nodes (eigsh is the expensive partitioner -- exactly the
    high-cost end of the tradeoff). Raises if scipy is missing (caller -> SKIP)."""
    if N >= max_nodes:
        raise RuntimeError(f"spectral capped at <{max_nodes:,} nodes (N={N:,})")
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    m = src != dst                                  # drop self-loops
    s, d = src[m], dst[m]
    # symmetric adjacency (de-dup not needed; eigvecs are robust to multiplicity)
    rows = np.concatenate([s, d]); cols = np.concatenate([d, s])
    data = np.ones(rows.shape[0], dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    dinv = sp.diags(1.0 / np.sqrt(deg))
    L = sp.identity(N) - dinv @ A @ dinv            # normalized Laplacian (symmetric)
    k = min(P, N - 2)
    # smallest eigenvalues -> sigma=0 shift-invert is fastest but needs a factorization;
    # 'SA' (smallest algebraic) is the robust default for L_sym (eigvals in [0, 2]).
    vals, vecs = eigsh(L, k=max(2, k + 1), which="SA")
    order = np.argsort(vals)
    vecs = vecs[:, order]
    if P == 2:
        fiedler = vecs[:, 1]                         # skip the trivial constant vec
        return (fiedler >= 0).astype(np.int32)
    emb = vecs[:, 1:k + 1]                           # drop trivial; keep next k-1..
    try:
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=P, n_init=4, random_state=0).fit_predict(emb)
        return lab.astype(np.int32)
    except Exception:
        # no sklearn: quantile-bucket the Fiedler vector into P balanced parts
        f = vecs[:, 1]
        ranks = np.argsort(np.argsort(f))
        return (ranks * P // N).astype(np.int32)


def lpa_blocks_assign(src, dst, N, P, bin_path, tmp_dir="/tmp"):
    """Run the C++ `lpa` (label-propagation) ORDERING kernel and slice the resulting
    node permutation into P CONTIGUOUS, EQUAL-SIZE blocks -> a partition. LPA groups
    same-cluster nodes adjacently in the new id space, so contiguous blocks keep most
    intra-cluster edges local. Returns (assignment, cpp_seconds). Raises if the binary
    is missing or fails (caller -> SKIP)."""
    if not os.path.exists(bin_path):
        raise RuntimeError(f"C++ binary not found: {bin_path} (build graph_algos)")
    edges_path = os.path.join(tmp_dir, "zord_pbench_edges.bin")
    out_path = os.path.join(tmp_dir, "zord_pbench_lpa.bin")
    s32 = src.astype(np.int32); d32 = dst.astype(np.int32)
    with open(edges_path, "wb") as f:
        f.write(struct.pack("<qq", N, s32.size))
        inter = np.empty(2 * s32.size, dtype=np.int32)
        inter[0::2] = s32; inter[1::2] = d32
        inter.tofile(f)
    t0 = time.time()
    r = subprocess.run([bin_path, edges_path, "lpa", out_path],
                       capture_output=True, text=True)
    cost = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"cpp lpa failed: {r.stderr.strip()[:200]}")
    with open(out_path, "rb") as f:
        n = struct.unpack("<q", f.read(8))[0]
        newid = np.fromfile(f, dtype=np.int32, count=n)        # newid[old] = rank
    # contiguous blocks over the LPA-induced rank space:  device = rank * P // N
    assignment = (newid.astype(np.int64) * P // N).astype(np.int32)
    return assignment, cost


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def cut_fraction(part):
    """% of undirected edges crossing devices. _summarize's cross_edges counts the
    directed (src->dst) edge stream once each; off-diagonal sum / total == cut frac."""
    total = int(part.cross_edges.sum())
    return (part.total_cross_edges / total) if total else 0.0


def node_balance(part):
    """max/avg part size by NODE count (1.0 == perfectly even load)."""
    n = part.nodes_per_device.astype(float)
    return float(n.max() / n.mean()) if n.mean() > 0 else 1.0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="", help="staged real temporal graph (else synthetic)")
    ap.add_argument("--nodes", type=int, default=200_000, help="synthetic node count")
    ap.add_argument("--edges", type=int, default=2_000_000, help="synthetic edge count")
    ap.add_argument("--comms", type=int, default=64, help="synthetic community count")
    ap.add_argument("--intra", type=float, default=0.9, help="synthetic intra-community edge fraction")
    ap.add_argument("--devices", "-D", type=int, default=3, help="number of devices (parts)")
    ap.add_argument("--feat-dim", type=int, default=128, help="feature dim for comm-cost prediction")
    ap.add_argument("--window", type=int, default=1, help="snapshots per batch (comm-cost scale)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spectral-cap", type=int, default=2_000_000,
                    help="skip spectral above this node count (eigsh is expensive)")
    a = ap.parse_args()

    D = a.devices
    # cluster profile drives the bandwidth-WEIGHTED comm-cost prediction. Spread the D
    # devices across the 3 measured HetCluster tiers so the (NVLink vs Ethernet) link skew
    # is exercised; the cut% / balance numbers themselves are profile-independent.
    cluster = _build_cluster(D)
    cp = CostParams(feat_dim=a.feat_dim, window=a.window)

    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        src, dst, N = g.src, g.dst, g.num_nodes
        name = g.name
    else:
        N = a.nodes
        src, dst, _ = gen_graph(N, a.edges, a.comms, a.intra, seed=a.seed)
        name = f"synth(comms={a.comms},intra={a.intra})"
    M = int(src.shape[0])
    shares = np.asarray(cluster.throughput_shares())
    print(f"PARTITIONER-BENCH  graph={name}  N={N:,}  M={M:,}  devices={D}  "
          f"feat_dim={a.feat_dim}  window={a.window}")
    print(f"  cluster={[d.name for d in cluster.devices]}  "
          f"throughput_shares={np.round(shares, 3).tolist()}")
    print(f"  loaded/generated graph in {time.time()-t0:.2f}s\n")

    # partitioner registry for THIS bench: name -> (callable -> (assignment, extra_cost_s))
    # extra_cost captures out-of-process time (e.g. the C++ subprocess) not seen by the
    # in-process timer; for pure-Python ones it is 0.
    def via(P_cls, **kw):
        def run():
            part = P_cls().partition(src, dst, N, cluster, **kw)
            return part.assignment, 0.0
        return run

    runners = {
        "hash":     via(HashPartitioner),
        "caphash":  via(CapacityProportionalHash),
        "fennel":   via(FennelPartitioner),
        "ldg":      lambda: (ldg_assign(src, dst, N, D), 0.0),
        "metis":    via(MetisPartitioner),
        "spectral": lambda: (spectral_assign(src, dst, N, D, max_nodes=a.spectral_cap), 0.0),
        "lpa-block": lambda: lpa_blocks_assign(src, dst, N, D, _BIN),
    }

    hdr = (f"  {'partitioner':<11} {'cut%':>7} {'balance':>8} {'edge_imb':>8} "
           f"{'part_s':>8} {'comm_pred_ms':>12}  note")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = {}
    base_cut = None
    for pname, run in runners.items():
        t = time.time()
        try:
            assignment, extra = run()
        except ImportError as e:
            print(f"  {pname:<11} {'--':>7} {'--':>8} {'--':>8} {'--':>8} {'--':>12}  "
                  f"SKIP (missing dep: {str(e)[:40]})")
            continue
        except Exception as e:
            print(f"  {pname:<11} {'--':>7} {'--':>8} {'--':>8} {'--':>8} {'--':>12}  "
                  f"SKIP ({str(e)[:60]})")
            continue
        part_s = (time.time() - t) + extra
        assignment = np.asarray(assignment, dtype=np.int32)
        part = Partitioner._summarize(assignment, src, dst, D)
        cutf = cut_fraction(part)
        bal = node_balance(part)
        eimb = part.imbalance()
        comm_ms = sum(device_comm_sec(part, i, cluster, cp)
                      for i in range(D)) * 1e3
        if base_cut is None:
            base_cut = cutf
        results[pname] = dict(cut=cutf, balance=bal, edge_imb=eimb,
                              part_s=part_s, comm_ms=comm_ms)
        note = f"cut-reduction vs hash: {(1 - cutf / base_cut) * 100:+.0f}%" if base_cut else ""
        print(f"  {pname:<11} {cutf*100:>6.2f}% {bal:>8.3f} {eimb:>8.3f} "
              f"{part_s:>8.3f} {comm_ms:>12.3f}  {note}")

    # -- the D1 takeaway: cut-reduction-per-second-spent, and the epoch break-even --
    if "hash" in results:
        print()
        h = results["hash"]
        print("  TRADEOFF (vs cheap hash baseline):")
        for pname, r in results.items():
            if pname == "hash":
                continue
            dcut = h["cut"] - r["cut"]                       # cut-fraction reduction
            dcomm = h["comm_ms"] - r["comm_ms"]             # predicted comm saved per batch (ms)
            dpart = r["part_s"] - h["part_s"]              # extra partition cost (s)
            # epochs to break even: extra partition time / comm saved per epoch (1 batch ~ 1 step)
            if dcomm > 1e-9:
                breakeven = (dpart * 1e3) / dcomm
                be = f"break-even ~{breakeven:,.0f} batches"
            elif dcomm < -1e-9:
                be = "never (comm WORSE than hash)"
            else:
                be = "comm ~= hash"
            print(f"    {pname:<11} cut {dcut*100:+5.1f}pp  comm {dcomm:+8.3f} ms/batch  "
                  f"+{dpart:6.3f}s partition  ->  {be}")


def _build_cluster(D):
    """Build an HetCluster profile with EXACTLY D devices, round-robin across the 3
    measured tiers (H100 / RTX6000Ada / RTX5000Ada) so heterogeneity + the
    intra/inter-node bandwidth skew are present for the comm-cost prediction."""
    from zord.profiler.cluster_profile import DeviceProfile, ClusterProfile, _MEASURED, GB
    tiers = ["H100-80GB", "RTX6000Ada-48GB", "RTX5000Ada-32GB"]
    devs = []
    for i in range(D):
        key = tiers[i % len(tiers)]
        m = _MEASURED[key]
        devs.append(DeviceProfile(i, key, int(m["mem_gb"] * GB), throughput=m["r"],
                                  node=i % len(tiers), h2d_gbps=m["h2d"],
                                  hbm_bw_gbps=m["hbm"], measured=True))
    return ClusterProfile(devices=devs)


if __name__ == "__main__":
    main()
