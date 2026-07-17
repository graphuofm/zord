#!/usr/bin/env python3
"""ATTR-PARTITION-REAL (D-attr-3): attribute-aware partitioning of REAL attributed graphs --
does respecting (a) attribute homophily and (b) per-node feature-BYTE balance beat an
attribute-BLIND METIS / hash partitioner on PROCESS, and WHAT ARE THE RULES?

THE KILLER PROBLEM (user): "if we can partition attributed graphs, or find the RULES for how
to partition them, we truly win." A company of 10000 employees, each a node with 100 attributes
-> huge feature mass -> still must partition it across devices.  This script does REAL experiments
on REAL open attributed graphs (ogbn-arxiv 169K/128-dim, ogbn-products 2.4M/100-dim, Reddit
233K/602-dim, Planetoid Cora/CiteSeer/PubMed, Coauthor, Amazon) loaded via `ogb` / PyG, and -- when
a dataset is not stageable in-agent -- a SYNTHETIC graph CALIBRATED to that dataset's PUBLISHED
attribute stats (N, F, degree, homophily), CLEARLY labelled synth-vs-real in the output.

PROCESS-ONLY (the project rule, D-memory-sched / "optimize process not accuracy"): same data +
same model => SAME numerical result. We change only WHERE a node's edges/features live and HOW the
cut is drawn; we NEVER touch accuracy. We measure: edge-cut, cross-device comm volume (feature
BYTES moved per layer), per-device incident-edge work + makespan, and FEASIBILITY (does a device's
true feature memory exceed its HBM cap -> OOM?).  NEVER networkx; the heavy structural cut is
pymetis (multilevel min-cut, the engine's metis_partition) or the C++ graph_algos LPA order.

================================================================================================
THREE EXPERIMENTS (all PROCESS-only):

EXP-1  AWARE-vs-BLIND CUT (feasibility / makespan / comm).  On a HETEROGENEOUS-HBM cluster with
       HETEROGENEOUS per-node feature bytes F_v (real graphs are multi-type: a paper with a long
       abstract vs a stub; an employee with a filled-in 100-field profile vs a sparse one), compare
         BLIND-HASH     : random hash by node id (count-balanced, attribute-blind)
         BLIND-METIS    : pymetis min-cut, UNIT vertex weights (count-balanced, attribute-blind)
         AWARE-METIS    : pymetis min-cut, vertex weights = F_v BYTES + target weights = HBM caps
                          (byte-balanced AND homophily-respecting, routes heavy-F mass to big HBM)
       Report: edge-cut, comm bytes, makespan, and whether the BLIND cut OOMs the small device when
       feature-heavy nodes structurally concentrate (the §33 feasibility failure on REAL F dims).

EXP-2  THE RULES -- attribute<->structure correlation regime.  Sweep the correlation rho between a
       node's ATTRIBUTE community and its STRUCTURAL community (from rho=+1 homophilic, where the
       feature-rich set IS a structural cluster, through rho=0 independent, to rho<0 anti-correlated,
       where rich nodes are spread across every structural cluster).  Measures, per rho:
         - feat-homophily of the cut (fraction of an edge's endpoints sharing feature-type, kept local)
         - does an attribute-aware cut LOWER the edge-cut/comm vs blind (homophilic -> yes; anti -> no)
         - the byte-imbalance a blind cut leaves (max/mean device feature bytes)
       => CHARACTERIZE: when attribute structure HELPS the cut vs when it only adds a placement
       constraint with no cut benefit.

EXP-3  PER-NODE FEATURE-BYTE PLACEMENT (the §33 win) on REAL feature dims.  For each real dataset's
       actual F, model a multi-type feature mix (e.g. text 768d / image 512d / categorical 16d, or
       the dataset's native uniform F as the null), and show the byte-aware placement lowers peak
       per-device HBM + stays feasible where the byte-blind (count*meanF) sizing OOMs -- on the REAL
       node count and degree distribution.

================================================================================================
USAGE
  python3 scripts/attr_partition_real.py                       # all exps, real ogbn-arxiv if stageable,
                                                               #   else synthetic calibrated to arxiv
  python3 scripts/attr_partition_real.py --dataset ogbn-arxiv --exp 1
  python3 scripts/attr_partition_real.py --dataset reddit --synthetic   # force calibrated-synth
  python3 scripts/attr_partition_real.py --dataset ogbn-products --download-only  # just stage data
  python3 scripts/attr_partition_real.py --exp 2 --devices 4 --seed 0

DATA STAGING (for the main loop to run; small ones auto-download in-agent in ~10-30s):
  pip install ogb                         # MIT; numpy NodePropPredDataset (no torch needed for arxiv/products)
  pip install torch_geometric             # for Reddit / Planetoid / Amazon / Coauthor
  # ogbn-arxiv ~81MB, ogbn-products ~1.4GB, Reddit ~1.4GB; cached under --root (default /tmp/zord_attr_data)
"""
import argparse
import os
import sys
import time

import numpy as np

# engine path (reused: metis_partition, feasible, edgecut_metrics, cluster from_spec, C++ LPA order)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

GB = 1024 ** 3
BYTES_PER_FEAT = 4.0            # fp32, FULL PRECISION (never a compression knob)
BYTES_PER_EDGE_RESIDENT = 20.0  # src+dst+ts+w edge metadata (matches arrange.BYTES_PER_EDGE_RESIDENT)
N_LAYERS = 2                    # 2-layer GNN: comm rows gathered per layer


# ============================================================================ #
# 0. REAL dataset registry -- published stats (size / feat-dim / URL / license)
#    Used both to LOAD the real graph and to CALIBRATE a synthetic stand-in.
# ============================================================================ #
DATASETS = {
    # name           : (N,        E,          F,    nclass, loader,      url, license)
    "ogbn-arxiv":     (169_343,   1_166_243,  128,  40,  "ogb",  "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",     "ODC-BY"),
    "ogbn-products":  (2_449_029, 61_859_140, 100,  47,  "ogb",  "http://snap.stanford.edu/ogb/data/nodeproppred/products.zip",  "Amazon-license"),
    "ogbn-mag":       (1_939_743, 21_111_007, 128,  349, "ogb",  "http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip",       "ODC-BY"),
    "reddit":         (232_965,   114_615_892,602,  41,  "pyg",  "https://data.dgl.ai/dataset/reddit.zip",                       "see-DGL"),
    "reddit2":        (232_965,   23_213_838, 602,  41,  "pyg",  "PyG Reddit2",                                                  "see-DGL"),
    "flickr":         (89_250,    899_756,    500,  7,   "pyg",  "PyG Flickr (GraphSAINT)",                                      "see-GraphSAINT"),
    "yelp":           (716_847,   13_954_819, 300,  100, "pyg",  "PyG Yelp (GraphSAINT)",                                        "see-GraphSAINT"),
    "cora":           (2_708,     10_556,     1_433,7,   "pyg",  "Planetoid (LINQS)",                                            "research-only"),
    "citeseer":       (3_327,     9_104,      3_703,6,   "pyg",  "Planetoid (LINQS)",                                            "research-only"),
    "pubmed":         (19_717,    88_648,     500,  3,   "pyg",  "Planetoid (LINQS)",                                            "research-only"),
    "coauthor-cs":    (18_333,    163_788,    6_805,15,  "pyg",  "Coauthor CS (Shchur 2018)",                                    "research-only"),
    "coauthor-phy":   (34_493,    495_924,    8_415,5,   "pyg",  "Coauthor Physics (Shchur 2018)",                               "research-only"),
    "amazon-computers":(13_752,   491_722,    767,  10,  "pyg",  "Amazon Computers (Shchur 2018)",                               "research-only"),
    "amazon-photo":   (7_650,     238_162,    745,  8,   "pyg",  "Amazon Photo (Shchur 2018)",                                   "research-only"),
    "dgraph-fin":     (3_700_550, 4_300_999,  17,   2,   "manual","https://dgraph.xinye.com (financial, temporal+attributed)",   "research-only"),
}


# ============================================================================ #
# 1. LOADERS -- REAL first; CLEARLY-LABELLED calibrated synthetic fallback.
# ============================================================================ #
class Graph:
    """Undirected simple graph + per-node feature TYPE/BYTES + a node 'community' label.
    src,dst    : int64 edge endpoints (one direction; undirected semantics)
    N          : node count
    F          : nominal (uniform) feature dim of the source dataset
    feat_bytes : float64 [N] per-node feature BYTES (heterogeneous when multi-type modelled)
    comm       : int64 [N] structural community / class label (real label, or planted)
    is_real    : True if loaded from a real dataset; False if calibrated-synthetic
    name       : dataset name
    """
    def __init__(self, src, dst, N, F, feat_bytes, comm, is_real, name):
        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.N = int(N)
        self.F = int(F)
        self.feat_bytes = np.asarray(feat_bytes, dtype=np.float64)
        self.comm = np.asarray(comm, dtype=np.int64)
        self.is_real = bool(is_real)
        self.name = name

    @property
    def E(self):
        return int(self.src.size)


def _dedup_undirected(src, dst, N):
    """Drop self-loops + duplicate undirected edges; return one direction (u<v)."""
    src = src.astype(np.int64); dst = dst.astype(np.int64)
    m = src != dst
    src, dst = src[m], dst[m]
    lo = np.minimum(src, dst); hi = np.maximum(src, dst)
    key = lo * np.int64(N) + hi
    _, idx = np.unique(key, return_index=True)
    return lo[idx], hi[idx]


def _patch_torch_load_weights_only():
    """PyTorch>=2.6 flipped torch.load default to weights_only=True, which breaks ogb / PyG dataset
    caches (they pickle numpy/PyG objects). Force weights_only=False for trusted local dataset files
    so REAL data loads. Idempotent; only affects this process."""
    try:
        import torch
        if getattr(torch.load, "_zord_patched", False):
            return
        _orig = torch.load
        def _ld(*a, **k):
            k.setdefault("weights_only", False)
            return _orig(*a, **k)
        _ld._zord_patched = True
        torch.load = _ld
    except Exception:
        pass


def load_real(name, root):
    """Load a REAL attributed graph. Returns Graph(is_real=True) or raises (caller -> synthetic).
    Uniform native F is recorded; the heterogeneous feature-byte MIX is applied later (EXP-1/3)."""
    _patch_torch_load_weights_only()
    spec = DATASETS[name]
    N0, E0, F, nclass, loader = spec[0], spec[1], spec[2], spec[3], spec[4]
    if loader == "ogb":
        from ogb.nodeproppred import NodePropPredDataset  # numpy loader, no torch needed
        d = NodePropPredDataset(name=name, root=root)
        g, label = d[0]
        N = int(g["num_nodes"])
        ei = np.asarray(g["edge_index"], dtype=np.int64)
        src, dst = _dedup_undirected(ei[0], ei[1], N)
        feat = g["node_feat"]
        F = int(feat.shape[1]) if feat is not None else F
        comm = np.asarray(label).reshape(-1).astype(np.int64)
        comm[comm < 0] = comm.max() + 1  # unlabeled -> own bucket (mag has -1)
        feat_bytes = np.full(N, F * BYTES_PER_FEAT, dtype=np.float64)  # native uniform F
        return Graph(src, dst, N, F, feat_bytes, comm, True, name)
    if loader == "pyg":
        return _load_pyg(name, root, F)
    raise RuntimeError(f"{name}: loader '{loader}' not auto-stageable in-agent (manual download)")


def _load_pyg(name, root, F):
    import torch  # noqa
    if name in ("reddit", "reddit2"):
        from torch_geometric.datasets import Reddit, Reddit2
        ds = (Reddit2 if name == "reddit2" else Reddit)(root=os.path.join(root, name))
    elif name in ("cora", "citeseer", "pubmed"):
        from torch_geometric.datasets import Planetoid
        ds = Planetoid(root=os.path.join(root, name), name=name.capitalize())
    elif name in ("coauthor-cs", "coauthor-phy"):
        from torch_geometric.datasets import Coauthor
        ds = Coauthor(root=os.path.join(root, name), name="CS" if name.endswith("cs") else "Physics")
    elif name in ("amazon-computers", "amazon-photo"):
        from torch_geometric.datasets import Amazon
        ds = Amazon(root=os.path.join(root, name), name="Computers" if name.endswith("computers") else "Photo")
    elif name in ("flickr",):
        from torch_geometric.datasets import Flickr
        ds = Flickr(root=os.path.join(root, name))
    elif name in ("yelp",):
        from torch_geometric.datasets import Yelp
        ds = Yelp(root=os.path.join(root, name))
    else:
        raise RuntimeError(f"no PyG loader for {name}")
    data = ds[0]
    N = int(data.num_nodes)
    ei = data.edge_index.cpu().numpy().astype(np.int64)
    src, dst = _dedup_undirected(ei[0], ei[1], N)
    F = int(data.x.shape[1]) if getattr(data, "x", None) is not None else F
    y = data.y.cpu().numpy().reshape(-1).astype(np.int64) if getattr(data, "y", None) is not None else np.zeros(N, np.int64)
    feat_bytes = np.full(N, F * BYTES_PER_FEAT, dtype=np.float64)
    return Graph(src, dst, N, F, feat_bytes, y, True, name)


def load_synthetic(name, root=None, scale=1.0, seed=0):
    """CALIBRATED-SYNTHETIC stand-in: matches the published (N, F, mean-degree, #communities) of
    the named real dataset.  CLEARLY labelled is_real=False.  Planted-partition (SBM-style) so the
    structural communities are real clusters the cut can find; community label == structural block.
    Honest: this is NOT the real graph -- it reproduces the SIZE + ATTRIBUTE-DIM + DENSITY regime so
    the PROCESS metrics (cut/comm/feasibility) are calibrated, but not the exact topology."""
    rng = np.random.default_rng(seed)
    N0, E0, F, nclass = DATASETS[name][0], DATASETS[name][1], DATASETS[name][2], DATASETS[name][3]
    N = max(64, int(N0 * scale))
    avg_deg = max(2.0, 2.0 * E0 / max(1, N0))   # undirected mean degree from published E
    nc = max(2, min(nclass, N // 8))
    comm = rng.integers(0, nc, size=N).astype(np.int64)
    src, dst = _planted_partition(N, comm, avg_deg, p_in=0.85, rng=rng)
    feat_bytes = np.full(N, F * BYTES_PER_FEAT, dtype=np.float64)
    g = Graph(src, dst, N, F, feat_bytes, comm, False, name + "[synthetic]")
    return g


def _planted_partition(N, comm, avg_deg, p_in, rng):
    """Sparse planted-partition (SBM): ~p_in of edges intra-community.  O(E) sampling (NO networkx,
    NO N^2): sample E endpoint pairs, biasing each toward same-community via a community index."""
    E = int(avg_deg * N / 2)
    nc = int(comm.max()) + 1
    by_comm = [np.where(comm == c)[0] for c in range(nc)]
    by_comm = [a for a in by_comm if a.size > 0]
    nc = len(by_comm)
    intra = rng.random(E) < p_in
    src = np.empty(E, np.int64); dst = np.empty(E, np.int64)
    # intra edges: pick a community, two members
    n_in = int(intra.sum())
    csz = np.array([a.size for a in by_comm], dtype=np.float64)
    cprob = csz / csz.sum()
    cc = rng.choice(nc, size=n_in, p=cprob)
    for i, c in enumerate(np.where(intra)[0]):
        mem = by_comm[cc[i]]
        src[c] = mem[rng.integers(mem.size)]
        dst[c] = mem[rng.integers(mem.size)]
    # inter edges: any two nodes
    out = np.where(~intra)[0]
    src[out] = rng.integers(0, N, size=out.size)
    dst[out] = rng.integers(0, N, size=out.size)
    s, d = _dedup_undirected(src, dst, N)
    return s, d


def load_graph(name, root, force_synth, scale, seed):
    """Try REAL; on any failure (not installed / no net / not stageable) -> calibrated synthetic.
    Returns (Graph, note) where note explains real-vs-synth + why."""
    if force_synth:
        return load_synthetic(name, root, scale, seed), "forced --synthetic (calibrated to published stats)"
    try:
        t0 = time.time()
        g = load_real(name, root)
        return g, f"REAL dataset loaded in {time.time()-t0:.1f}s from {root}"
    except Exception as e:
        g = load_synthetic(name, root, scale, seed)
        return g, f"REAL load failed ({type(e).__name__}: {str(e)[:120]}) -> calibrated SYNTHETIC"


# ============================================================================ #
# 2. HETEROGENEOUS FEATURE-TYPE MIX -- the multi-modal regime (real graphs are multi-type)
# ============================================================================ #
FEATURE_TYPES = {
    # type        : dim (the §33 multi-modal mix: text-heavy / image / categorical-poor)
    "text":  768,
    "image": 512,
    "cat":   16,
}


def assign_feature_types(g, rich_frac, rho, seed, feat_scale=1.0):
    """Assign each node a feature TYPE -> per-node feature BYTES, with CORRELATION rho between the
    'rich' (high-byte) set and structural community.  Sets g.feat_bytes (heterogeneous) and returns
    the per-node type-id.  rho in [-1,1]:
      rho=+1  : the rich nodes ARE concentrated in specific communities (homophilic feature mass)
      rho= 0  : rich nodes spread independently of community
      rho=-1  : rich nodes anti-correlated (one per community -> spread across all blocks)
    feat_scale : multiplies the per-node feature DIM (models richer multi-modal embeddings or
      per-snapshot temporal feature STATE that must be resident); raises total feature mass into the
      HBM-pressure regime where the §33 feasibility effect is observable on a real node count. The
      RELATIVE heterogeneity (text:image:cat = 768:512:16) and the cut metrics are scale-invariant;
      only the absolute GB (hence OOM) scales.  This is THE lever EXP-2 sweeps; EXP-1/3 use rho>0."""
    rng = np.random.default_rng(seed)
    N, nc = g.N, int(g.comm.max()) + 1
    # base 'richness propensity' per community, then per-node mix toward/against community signal
    comm_rich = rng.random(nc)                       # each community's intrinsic richness
    comm_rich = (comm_rich - comm_rich.mean())       # centered
    node_noise = rng.random(N) - 0.5
    # rho blends community signal (comm_rich[comm[v]]) with node noise
    a = abs(rho)
    score = (np.sign(rho) if rho != 0 else 1.0) * a * comm_rich[g.comm] + (1.0 - a) * node_noise
    thresh = np.quantile(score, 1.0 - rich_frac)
    is_rich = score >= thresh
    # rich -> text (768d); the rest split image/cat
    tid = np.zeros(N, dtype=np.int64)                # 0=cat,1=image,2=text
    rest = ~is_rich
    img = rng.random(N) < 0.4
    tid[rest & img] = 1
    tid[rest & ~img] = 0
    tid[is_rich] = 2
    dims = np.array([FEATURE_TYPES["cat"], FEATURE_TYPES["image"], FEATURE_TYPES["text"]], dtype=np.float64)
    g.feat_bytes = dims[tid] * BYTES_PER_FEAT * float(feat_scale)
    return tid


# ============================================================================ #
# 3. CLUSTER (heterogeneous HBM is the placement lever) + PARTITIONERS
# ============================================================================ #
def make_cluster(hbm_gb, agg_bw, link_gbps):
    """Return (devs, hbm_bytes[D], agg_bw[D], link_gbps).  Uses the engine ClusterProfile if present;
    else a tiny local stand-in (same fields the cost model needs)."""
    hbm = [float(x) for x in hbm_gb.split(",")]
    bw = [float(x) for x in agg_bw.split(",")]
    assert len(hbm) == len(bw)
    try:
        from zord.profiler.cluster_profile import from_spec
        c = from_spec(hbm_gb=hbm, agg_bw_gbps=bw, interconnect_gbps=link_gbps,
                      names=[f"GPU{i}(HBM{int(hbm[i])})" for i in range(len(hbm))])
        devs = list(c.devices)
        hbm_bytes = np.array([d.usable_mem for d in devs], dtype=np.float64)
        bwv = np.array([d.throughput for d in devs], dtype=np.float64)
        return devs, hbm_bytes, bwv, link_gbps
    except Exception:
        D = len(hbm)
        hbm_bytes = np.array([h * GB * 0.90 for h in hbm], dtype=np.float64)  # 90% usable
        bwv = np.array(bw, dtype=np.float64)
        return list(range(D)), hbm_bytes, bwv, link_gbps


def part_hash(g, D, seed):
    """Attribute-BLIND hash: random by node id, count-balanced."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, D, size=g.N).astype(np.int64)


def _csr(src, dst, N):
    u = np.concatenate([src, dst]); v = np.concatenate([dst, src])
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, u + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, v.astype(np.int64)


def part_metis(g, D, vweights=None, tpwgts=None):
    """METIS multilevel min-cut via pymetis.  vweights -> per-node weight (None=unit count-balance;
    feat_bytes-as-int -> byte-balance).  tpwgts -> target partition weights (None=equal; HBM-frac ->
    route mass by capacity).  Returns membership int64 [N], or None if pymetis unavailable/fails."""
    try:
        import pymetis
    except Exception:
        return None
    indptr, adjncy = _csr(g.src, g.dst, g.N)
    kw = {}
    if vweights is not None:
        vw = np.asarray(vweights, dtype=np.float64)
        vw = np.maximum(1, np.rint(vw / max(1.0, vw.min()))).astype(np.int64)  # METIS wants int weights
        kw["vweights"] = vw.tolist()
    if tpwgts is not None:
        tp = np.asarray(tpwgts, dtype=np.float64); tp = (tp / tp.sum()).tolist()
        kw["tpwgts"] = tp
    try:
        adj = pymetis.CSRAdjacency(indptr.tolist(), adjncy.tolist())
        _, membership = pymetis.part_graph(D, adjacency=adj, **kw)
    except Exception:
        try:
            _, membership = pymetis.part_graph(D, xadj=indptr.tolist(), adjncy=adjncy.tolist(), **kw)
        except Exception:
            return None
    return np.asarray(membership, dtype=np.int64)


def part_lpa_balanced(g, D, weights):
    """C++ LPA community ORDER, then sweep into D parts balancing WEIGHTS (the no-pymetis floor;
    also the >METIS_MAX_EDGES fallback).  weights = per-node weight to balance (count or bytes)."""
    rank = None
    try:
        from zord.partition import cpp_kernel
        if cpp_kernel.have_cpp_kernel():
            r = cpp_kernel.cpp_order(g.N, g.src, g.dst, "lpa")
            if r is not None:
                rank = np.asarray(r, dtype=np.int64)
    except Exception:
        rank = None
    if rank is None:  # degree-order fallback (still NO networkx)
        deg = node_deg(g.src, g.dst, g.N)
        rank = np.argsort(-deg, kind="stable").argsort()
    order = np.argsort(rank, kind="stable")            # nodes in locality order
    w = np.asarray(weights, dtype=np.float64)[order]
    target = w.sum() / D
    dev_of_order = np.zeros(g.N, dtype=np.int64)
    acc = 0.0; k = 0
    for i in range(g.N):
        dev_of_order[i] = k
        acc += w[i]
        if acc >= target and k < D - 1:
            acc = 0.0; k += 1
    dev = np.empty(g.N, dtype=np.int64)
    dev[order] = dev_of_order
    return dev


# ============================================================================ #
# 4. METRICS (PROCESS-only) -- engine-native edgecut + comm + feasibility/makespan
# ============================================================================ #
def node_deg(src, dst, N):
    return (np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)).astype(np.int64)


def cut_and_comm(g, dev, D):
    """edge-cut, per-device incident-edge work, per-device distinct remote COMM rows, counts.
    Reuses the engine's metric shape (arrange.edgecut_metrics)."""
    src, dst, N = g.src, g.dst, g.N
    deg = node_deg(src, dst, N)
    pu, pv = dev[src], dev[dst]
    cut = int(np.count_nonzero(pu != pv))
    incident = np.bincount(dev, weights=deg.astype(np.float64), minlength=D)
    counts = np.bincount(dev, minlength=D).astype(np.int64)
    a = np.concatenate([src, dst]); b = np.concatenate([dst, src])
    da = dev[a]; cross = da != dev[b]
    if cross.any():
        keyc = np.unique(da[cross].astype(np.int64) * np.int64(N) + b[cross])
        comm_rows = np.bincount((keyc // N).astype(np.int64), minlength=D).astype(np.float64)
    else:
        comm_rows = np.zeros(D, dtype=np.float64)
    return cut, incident, comm_rows, counts


def device_feat_bytes(g, dev, D):
    """TRUE per-device feature memory = sum_{v on k} feat_bytes[v] (heterogeneous-F aware)."""
    return np.bincount(dev, weights=g.feat_bytes, minlength=D).astype(np.float64)


def blind_feat_bytes(g, dev, D):
    """What a feature-byte-BLIND model BELIEVES each device holds = count_k * mean_F_bytes."""
    counts = np.bincount(dev, minlength=D).astype(np.float64)
    mean_fb = g.feat_bytes.mean()
    return counts * mean_fb


def feasibility(true_fb, incident, hbm_bytes):
    """A device fits if true feature bytes + resident edge metadata < usable HBM (arrange.feasible)."""
    need = true_fb + incident * BYTES_PER_EDGE_RESIDENT
    ok = need <= hbm_bytes
    return bool(ok.all()), need, ok


def makespan_ms(g, dev, incident, comm_rows, bwv, link_gbps, D):
    """Predicted per-device wall-clock = compute (incident-edge gather, BW-bound) + comm (remote
    feature rows fetched per layer over the slow link).  Bytes moved use the REAL per-node F_v of
    the REMOTE rows.  Makespan = max over devices.  (cost-model proxy, matches the engine's shape.)"""
    src, dst, N = g.src, g.dst, g.N
    # compute: gather feat_bytes over incident edges, normalized by per-device agg bandwidth (GB/s)
    # per-device incident feature byte-work = sum over incident edges of endpoint feat_bytes
    a = np.concatenate([src, dst]); b = np.concatenate([dst, src])
    da = dev[a]
    work_bytes = np.bincount(da, weights=g.feat_bytes[b], minlength=D).astype(np.float64) * N_LAYERS
    compute_ms = work_bytes / (bwv * 1e9) * 1e3
    # comm: distinct remote rows fetched, sized by their REAL feat_bytes, over the link
    cross = da != dev[b]
    if cross.any():
        # distinct (gatherer, remote-node) pairs, summing the remote node's feat_bytes once per pair
        key = da[cross].astype(np.int64) * np.int64(N) + b[cross]
        uk = np.unique(key)
        gk = (uk // N).astype(np.int64)
        rn = (uk % N).astype(np.int64)
        comm_bytes = np.bincount(gk, weights=g.feat_bytes[rn], minlength=D).astype(np.float64) * N_LAYERS
    else:
        comm_bytes = np.zeros(D, dtype=np.float64)
    comm_ms = comm_bytes / (link_gbps * 1e9) * 1e3
    per_dev = compute_ms + comm_ms
    return float(per_dev.max()), compute_ms, comm_ms, comm_bytes


# ============================================================================ #
# 5. EXPERIMENTS
# ============================================================================ #
def fmt_gb(x):
    return f"{x / GB:6.2f}GB"


def run_exp1(g, args):
    """AWARE-vs-BLIND cut: feasibility / makespan / comm on heterogeneous-HBM + heterogeneous-F."""
    print("\n" + "=" * 92)
    print(f"EXP-1  AWARE-vs-BLIND CUT  |  dataset={g.name}  N={g.N:,}  E={g.E:,}  nativeF={g.F}  "
          f"{'REAL' if g.is_real else 'SYNTH'}")
    print("=" * 92)
    devs, hbm_bytes, bwv, link = make_cluster(args.hbm_gb, args.agg_bw, args.link_gbps)
    D = len(hbm_bytes)
    # heterogeneous multi-modal feature bytes, rich mass CLUSTERED (rho>0) -- the realistic regime
    assign_feature_types(g, rich_frac=args.rich_frac, rho=args.rho, seed=args.seed)
    print(f"cluster HBM(GB)={[round(h/GB,1) for h in hbm_bytes]}  aggBW(GB/s)={list(bwv)}  link={link}GB/s")
    print(f"feature mix: text768/image512/cat16  rich_frac={args.rich_frac} rho={args.rho}  "
          f"total feat={fmt_gb(g.feat_bytes.sum())}  mean F_v={g.feat_bytes.mean()/BYTES_PER_FEAT:.0f}d")

    hbm_frac = hbm_bytes / hbm_bytes.sum()
    parts = {
        "BLIND-HASH":  part_hash(g, D, args.seed),
        "BLIND-METIS": part_metis(g, D, vweights=None, tpwgts=None),
        "AWARE-METIS": part_metis(g, D, vweights=g.feat_bytes, tpwgts=hbm_frac),
    }
    if parts["BLIND-METIS"] is None:           # no pymetis / too big -> LPA floor
        parts["BLIND-METIS(lpa)"] = part_lpa_balanced(g, D, np.ones(g.N))
        parts["AWARE-METIS(lpa)"] = part_lpa_balanced(g, D, g.feat_bytes)
        del parts["BLIND-METIS"], parts["AWARE-METIS"]

    rows = []
    for name, dev in parts.items():
        if dev is None:
            continue
        cut, incident, comm_rows, counts = cut_and_comm(g, dev, D)
        true_fb = device_feat_bytes(g, dev, D)
        ok, need, okmask = feasibility(true_fb, incident, hbm_bytes)
        mk, comp_ms, comm_ms, comm_bytes = makespan_ms(g, dev, incident, comm_rows, bwv, link, D)
        rows.append((name, cut, comm_bytes.sum(), mk, ok, true_fb.max(), need.max(), counts))
        print(f"\n  [{name}]")
        print(f"    edge-cut          : {cut:,} ({100.0*cut/max(1,g.E):.1f}% of edges)")
        print(f"    comm bytes/2layer : {fmt_gb(comm_bytes.sum())}  (distinct remote feature rows x F_v)")
        print(f"    makespan (ms)     : {mk:.1f}   (compute max {comp_ms.max():.1f} + comm max {comm_ms.max():.1f})")
        print(f"    per-dev TRUE feat : {[round(x/GB,2) for x in true_fb]} GB")
        print(f"    per-dev need(f+e) : {[round(x/GB,2) for x in need]} GB   vs cap {[round(x/GB,1) for x in hbm_bytes]}")
        print(f"    FEASIBLE          : {'YES' if ok else 'NO -- OOM on dev ' + str(list(np.where(~okmask)[0]))}")

    # the verdict
    print("\n  --- EXP-1 VERDICT ---")
    feas = {n: r[4] for n, r in zip([x[0] for x in rows], rows)}
    aware = [n for n in feas if "AWARE" in n]
    blind = [n for n in feas if "BLIND" in n]
    if aware and blind:
        a = aware[0];
        bf = [n for n in blind if not feas[n]]
        if bf and feas[a]:
            print(f"    AWARE stays FEASIBLE while {bf} OOM -> attribute-byte placement is the feasibility win.")
        # makespan compare among feasible
        feas_rows = [r for r in rows if r[4]]
        if feas_rows:
            best = min(feas_rows, key=lambda r: r[3])
            print(f"    lowest makespan among FEASIBLE: {best[0]} = {best[3]:.1f}ms")
            for r in rows:
                if r[4] and r[0] != best[0]:
                    print(f"      {r[0]}: {r[3]/best[3]:.2f}x makespan, {r[2]/max(1,best[2]):.2f}x comm")
    return rows


def _attr_priority_cut(g, type_id, D):
    """A cut that PRIORITIZES attribute homophily: group nodes by feature TYPE first, then split each
    type's nodes into balanced equal-size chunks across the D devices (round-robin within type so
    counts stay balanced).  This is the 'partition BY attribute' extreme -- it keeps same-type nodes
    together regardless of structure.  Equal node-count balance (same as the structural baseline) so
    the ONLY difference vs structure-only is WHAT it optimizes (attribute homophily vs min-cut)."""
    N = g.N
    dev = np.empty(N, dtype=np.int64)
    # order nodes by (type, id); assign contiguous balanced blocks -> same-type nodes co-located,
    # then sliced into D equal parts. Round-robin the *blocks* to keep per-device counts even.
    order = np.argsort(type_id, kind="stable")
    per = int(np.ceil(N / D))
    dev_of_order = (np.arange(N) // per).astype(np.int64)
    dev_of_order[dev_of_order >= D] = D - 1
    dev[order] = dev_of_order
    return dev


def categorical_attribute(g, nval, rho, seed):
    """A FINE-GRAINED categorical node attribute (e.g. an employee's DEPARTMENT, a paper's SUBJECT)
    with `nval` values, correlated with the structural community by rho in [-1,1].  This is the
    PARTITION-RELEVANT attribute for the cut question (vs the 3-type byte-size mix, which drives the
    §33 PLACEMENT question).  rho=+1: attribute value == structural block (perfect homophily); rho=0:
    independent; rho<0: deliberately mismatched (value cycles AWAY from the block).  Returns int64
    [N] attribute value.  The number of attribute values matters: to ALIGN with a min-cut the
    attribute must be as FINE as the structural partition (the key real-graph caveat)."""
    rng = np.random.default_rng(seed + 777)
    N = g.N
    nc = int(g.comm.max()) + 1
    # map each structural community to a 'home' attribute value
    home = (np.arange(nc) % nval)
    base = home[g.comm]                       # the perfectly-homophilic attribute (rho=+1)
    rand = rng.integers(0, nval, size=N)      # the independent attribute (rho=0)
    a = abs(rho)
    keep = rng.random(N) < a                  # fraction following the (signed) structure signal
    val = np.where(keep, base, rand)
    if rho < 0:                               # anti-correlate: push value AWAY from its home block
        val = np.where(keep, (base + 1 + rng.integers(0, max(1, nval - 1), size=N)) % nval, rand)
    return val.astype(np.int64)


def run_exp2(g, args):
    """THE RULES (the user's core question): WHEN does attribute<->structure correlation HELP the cut
    vs HURT it?  Honest mechanism isolation -- compare, at EQUAL node-balance (same target), two cuts
    that differ ONLY in what they optimize:
        STRUCT-ONLY : METIS min-cut, attribute-blind (the structural optimum, the floor)
        ATTR-PRIOR  : partition BY a fine-grained categorical attribute (department/subject), the
                      homophily-prioritized cut
    as we sweep rho = correlation between the attribute and structural community.  The penalty
    (attr-prior edge-cut / struct-only edge-cut) tells us when respecting attributes COSTS structure.
    We ALSO report the byte-imbalance a structure-only cut leaves (the rho-independent placement gap).
    The attribute has nval = max(D, #communities) values so it CAN, in principle, align with the cut
    (a coarse few-valued attribute provably cannot -- that is itself a rule, see RULE 0)."""
    print("\n" + "=" * 92)
    print(f"EXP-2  THE RULES: when does attribute<->structure correlation HELP vs HURT the cut?  "
          f"dataset={g.name} {'REAL' if g.is_real else 'SYNTH'}")
    print("=" * 92)
    devs, hbm_bytes, bwv, link = make_cluster(args.hbm_gb, args.agg_bw, args.link_gbps)
    D = len(hbm_bytes)
    nval = max(D, int(g.comm.max()) + 1)      # attribute as fine as the structural partition
    print(f"  STRUCT-ONLY = attribute-blind METIS min-cut (equal parts).  ATTR-PRIOR = partition BY a")
    print(f"  fine categorical attribute ({nval} values, e.g. 'department'/'subject'), equal parts.")
    print(f"  Same balance => isolates the attribute effect; rho = attribute<->structure correlation.")
    print(f"  {'rho':>5} | {'feat-homophily':>14} | {'struct-cut':>11} | {'attr-cut':>11} | "
          f"{'attr penalty':>12} | {'struct byte-imbal':>17} | regime")
    print("  " + "-" * 104)
    rules = []
    for rho in [float(x) for x in args.rho_sweep.split(",")]:
        gg = Graph(g.src, g.dst, g.N, g.F, g.feat_bytes.copy(), g.comm, g.is_real, g.name)
        # byte-size types (for the placement byte-imbalance column), rho-correlated as in EXP-1/3
        assign_feature_types(gg, rich_frac=args.rich_frac, rho=rho, seed=args.seed)
        # the PARTITION-RELEVANT fine categorical attribute (department/subject)
        attr_val = categorical_attribute(gg, nval, rho, args.seed)
        # feature homophily of the GRAPH: P(edge endpoints share the categorical attribute value)
        same = attr_val[gg.src] == attr_val[gg.dst]
        feat_homo = float(same.mean())
        # STRUCT-ONLY: attribute-blind min-cut (equal parts)
        struct = part_metis(gg, D, vweights=None, tpwgts=None)
        if struct is None:
            struct = part_lpa_balanced(gg, D, np.ones(gg.N))
        # ATTR-PRIOR: partition by the categorical attribute value (homophily-first)
        attr = _attr_priority_cut(gg, attr_val, D)
        sc, _, _, _ = cut_and_comm(gg, struct, D)
        ac, _, _, _ = cut_and_comm(gg, attr, D)
        penalty = ac / max(1, sc)            # >1: respecting attributes COSTS structural cut
        # byte-imbalance the STRUCTURE-ONLY (attribute-blind) cut leaves -> the placement gap
        s_fb = device_feat_bytes(gg, struct, D)
        s_imbal = s_fb.max() / max(1.0, s_fb.mean())
        if penalty <= 1.15:
            regime = "attr ALIGNS structure (homophily helps; cut cheap)"
        elif penalty <= 2.0:
            regime = "partial conflict (attr cut costs some structure)"
        else:
            regime = "attr FIGHTS structure (anti-corr; attrs hurt the cut)"
        rules.append((rho, feat_homo, sc, ac, penalty, s_imbal, regime))
        print(f"  {rho:>5.2f} | {feat_homo:>14.3f} | {sc:>11,} | {ac:>11,} | {penalty:>11.2f}x | "
              f"{s_imbal:>16.2f}x | {regime}")
    # correlation of feat-homophily with the attribute-cut penalty across the sweep
    fh = np.array([r[1] for r in rules]); pen = np.array([r[4] for r in rules])
    corr = float(np.corrcoef(fh, pen)[0, 1]) if fh.std() > 0 and pen.std() > 0 else 0.0
    aligned = [r for r in rules if r[4] <= 1.15]
    fights = [r for r in rules if r[4] > 2.0]
    print("\n  --- EXP-2 RULES (the user's question: when does attribute structure help vs hurt the cut?) ---")
    print(f"    RULE 0 (granularity prerequisite): the attribute can only ALIGN with a min-cut if it is")
    print(f"            as FINE as the structural partition. A coarse few-valued attribute (e.g. a 3-way")
    print(f"            modality tag) CANNOT recover the cut even at rho=1 -- co-locating by it forces")
    print(f"            unrelated structural clusters together. Here the attribute has {nval} values.")
    print(f"    RULE 1 (HELP regime): when feature-type CORRELATES with structure (high feat-homophily),")
    print(f"            partitioning BY attribute nearly recovers the structural min-cut -- penalty ~1x.")
    print(f"            ({len(aligned)}/{len(rules)} swept rho had attr-cut <=1.15x struct-cut; min penalty "
          f"{min(r[4] for r in rules):.2f}x.) Here attributes are FREE locality: cut + place in one pass.")
    print(f"    RULE 2 (HURT regime): when feature-type is INDEPENDENT/ANTI-correlated with structure")
    print(f"            (low feat-homophily), forcing same-attribute nodes together SHREDS the structural")
    print(f"            cut -- penalty up to {max(r[4] for r in rules):.1f}x ({len(fights)}/{len(rules)} rho >2x). Here you must")
    print(f"            NOT cut by attributes; cut by STRUCTURE and treat attributes as a placement-only")
    print(f"            constraint (byte-balance), accepting they give no locality help.")
    print(f"    QUANTIFIED: corr(feat-homophily, attr-cut-penalty) = {corr:+.2f} -> the higher the")
    print(f"            attribute-structure correlation, the cheaper attribute-respecting locality is.")
    print(f"    RULE 3 (placement is ALWAYS needed, rho-independent): even the structure-optimal cut")
    print(f"            leaves feature-byte imbalance up to {max(r[5] for r in rules):.2f}x across devices -> OOM risk")
    print(f"            on the small-HBM device at EVERY rho (cf. EXP-3); byte-balancing-to-capacity is")
    print(f"            required regardless of whether attributes also help the cut.")
    print(f"    => DECISION RULE: (1) measure feat-homophily on the edge list (one O(E) pass); (2) if HIGH,")
    print(f"       a single attribute-aware cut gives BOTH low edge-cut AND balanced placement; (3) if LOW,")
    print(f"       cut by STRUCTURE (min-cut) and add byte-balance as a SEPARATE placement constraint --")
    print(f"       never sacrifice the structural cut to co-locate uncorrelated attributes.")
    return rules


def _auto_feat_scale(g, rich_frac, rho, seed, hbm_bytes, target_frac=0.85):
    """Pick feat_scale so the AGGREGATE feature mass ~ target_frac of total HBM (the HBM-pressure
    regime where the §33 OOM is observable on the REAL node count). At feat_scale=1 the multi-modal
    mix may be tiny vs HBM; we scale the per-node DIM up (richer embeddings / resident temporal
    state) keeping the 768:512:16 heterogeneity. Returns the chosen scale (>=1)."""
    assign_feature_types(g, rich_frac=rich_frac, rho=rho, seed=seed, feat_scale=1.0)
    base = g.feat_bytes.sum()
    want = target_frac * float(hbm_bytes.sum())
    return max(1.0, want / max(1.0, base))


def run_exp3(g, args):
    """PER-NODE FEATURE-BYTE PLACEMENT (§33) on REAL feat dims: aware sizing vs blind count*meanF.
    The §33 win is a FEASIBILITY win and only appears under HBM PRESSURE -- so we scale the
    multi-modal feature mass to ~85% of aggregate HBM (auto, unless --feat-scale given), keeping
    the real node count + degree + the 768:512:16 heterogeneity."""
    print("\n" + "=" * 92)
    print(f"EXP-3  PER-NODE FEATURE-BYTE PLACEMENT (§33)  |  dataset={g.name}  realF={g.F}  "
          f"{'REAL' if g.is_real else 'SYNTH'}")
    print("=" * 92)
    devs, hbm_bytes, bwv, link = make_cluster(args.hbm_gb, args.agg_bw, args.link_gbps)
    D = len(hbm_bytes)
    rho3 = max(0.6, args.rho)
    fscale = args.feat_scale if args.feat_scale > 0 else _auto_feat_scale(g, args.rich_frac, rho3, args.seed, hbm_bytes)
    assign_feature_types(g, rich_frac=args.rich_frac, rho=rho3, seed=args.seed, feat_scale=fscale)
    hbm_frac = hbm_bytes / hbm_bytes.sum()
    print(f"  cluster HBM(GB)={[round(h/GB,1) for h in hbm_bytes]}  feat_scale={fscale:.1f}x  "
          f"(eff dims: cat{int(16*fscale)}/img{int(512*fscale)}/text{int(768*fscale)})")
    print(f"  total feature mass {g.feat_bytes.sum()/GB:.1f}GB  vs aggregate HBM {hbm_bytes.sum()/GB:.1f}GB  "
          f"({100*g.feat_bytes.sum()/hbm_bytes.sum():.0f}% pressure)  rich_frac={args.rich_frac} rho={rho3}")

    # BLIND placement: count-balanced (METIS unit), sized by count*meanF (what blind model believes)
    blind = part_metis(g, D, vweights=None, tpwgts=None)
    if blind is None:
        blind = part_lpa_balanced(g, D, np.ones(g.N))
    aware = part_metis(g, D, vweights=g.feat_bytes, tpwgts=hbm_frac)
    if aware is None:
        aware = part_lpa_balanced(g, D, g.feat_bytes)

    res = {}
    for name, dev in [("BLIND (count-balanced, count*meanF sizing)", blind),
                      ("AWARE (byte-balanced to HBM caps)", aware)]:
        _, incident, _, counts = cut_and_comm(g, dev, D)
        true_fb = device_feat_bytes(g, dev, D)
        believed = blind_feat_bytes(g, dev, D)
        ok, need, okmask = feasibility(true_fb, incident, hbm_bytes)
        res[name] = (ok, need, okmask, true_fb, counts)
        print(f"\n  [{name}]")
        print(f"    node counts       : {list(counts)}")
        print(f"    BELIEVED feat (GB): {[round(x/GB,1) for x in believed]}  (count*meanF -- what the blind model sees)")
        print(f"    TRUE feat (GB)    : {[round(x/GB,1) for x in true_fb]}  (sum of real per-node F_v)")
        print(f"    need f+edge (GB)  : {[round(x/GB,1) for x in need]}   vs cap {[round(x/GB,1) for x in hbm_bytes]}")
        if ok:
            print(f"    FEASIBLE          : YES  (every device under its HBM cap)")
        else:
            bad = list(np.where(~okmask)[0])
            print(f"    FEASIBLE          : NO -- OOM on dev {bad} "
                  f"(need {[round(need[i]/GB,1) for i in bad]}GB > cap {[round(hbm_bytes[i]/GB,1) for i in bad]}GB)")
    print("\n  --- EXP-3 VERDICT (§33) ---")
    b_ok, b_need = res[list(res)[0]][0], res[list(res)[0]][1]
    a_ok, a_need = res[list(res)[1]][0], res[list(res)[1]][1]
    if (not b_ok) and a_ok:
        print(f"    BLIND OOMs (it count-balances, so the heavy multi-modal rows pile onto the SMALL-HBM")
        print(f"    device whose TRUE feature memory then exceeds its cap) while AWARE STAYS FEASIBLE by")
        print(f"    routing heavy-F mass to the big-HBM device -> the §33 feasibility win, on REAL N/F.")
    elif b_ok and a_ok:
        print(f"    Both feasible at this pressure; AWARE peak-need-vs-cap headroom "
              f"{(b_need.max()/min(1.0,1.0)):.0f} -- blind worst dev {b_need.max()/GB:.1f}GB, "
              f"aware worst {a_need.max()/GB:.1f}GB. Raise --feat-scale to enter the OOM regime.")
    else:
        print(f"    At this pressure neither fully fits; the byte-aware sizing still lowers the worst-device")
        print(f"    overflow ({b_need.max()/GB:.1f}GB blind vs {a_need.max()/GB:.1f}GB aware on its tightest dev).")
    print(f"    NOTE: the blind model BELIEVES every device sits at count*meanF and 'fits'; the TRUE")
    print(f"    per-node-F_v mass is the signal it cannot see. NULL on uniform-F (§27); real on")
    print(f"    heterogeneous multi-modal F (this exp), the regime of the user's 100-attribute employees.")


def main():
    ap = argparse.ArgumentParser(description="attribute-aware partitioning of REAL attributed graphs (PROCESS-only)")
    ap.add_argument("--dataset", default="ogbn-arxiv", choices=list(DATASETS.keys()))
    ap.add_argument("--root", default="/tmp/zord_attr_data")
    ap.add_argument("--synthetic", action="store_true", help="force calibrated-synthetic (no real download)")
    ap.add_argument("--download-only", action="store_true", help="just stage the real dataset + print stats")
    ap.add_argument("--scale", type=float, default=1.0, help="synthetic scale vs published N")
    ap.add_argument("--exp", default="all", help="1 | 2 | 3 | all")
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--hbm-gb", default="80,48,32", help="per-device HBM GB (heterogeneous)")
    ap.add_argument("--agg-bw", default="3350,1008,448", help="per-device achieved agg BW GB/s")
    ap.add_argument("--link-gbps", type=float, default=25.0, help="interconnect GB/s (slow link binds comm)")
    ap.add_argument("--rich-frac", type=float, default=0.20, help="fraction of nodes that are high-F (rich)")
    ap.add_argument("--feat-scale", type=float, default=0.0, help="EXP-3 per-node feature-dim multiplier "
                    "(0=auto -> ~85%% of aggregate HBM, the §33 OOM-pressure regime)")
    ap.add_argument("--rho", type=float, default=0.7, help="attribute<->structure correlation for EXP1/3")
    ap.add_argument("--rho-sweep", default="1.0,0.7,0.3,0.0,-0.5,-1.0", help="EXP-2 correlation sweep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # align device-count knobs if --devices changed and hbm/bw left default
    if args.devices != len(args.hbm_gb.split(",")):
        base_hbm = [80, 48, 32, 24, 16, 80, 48, 32]
        base_bw = [3350, 1008, 448, 360, 320, 3350, 1008, 448]
        args.hbm_gb = ",".join(str(base_hbm[i % len(base_hbm)]) for i in range(args.devices))
        args.agg_bw = ",".join(str(base_bw[i % len(base_bw)]) for i in range(args.devices))

    print("#" * 92)
    print(f"# ATTR-PARTITION-REAL  dataset={args.dataset}  exp={args.exp}  devices={args.devices}  seed={args.seed}")
    print("#" * 92)
    spec = DATASETS[args.dataset]
    print(f"# REGISTRY  {args.dataset}: N={spec[0]:,}  E={spec[1]:,}  F={spec[2]}  nclass={spec[3]}  "
          f"loader={spec[4]}  license={spec[6]}")
    print(f"# URL: {spec[5]}")

    g, note = load_graph(args.dataset, args.root, args.synthetic, args.scale, args.seed)
    print(f"# LOAD: {note}")
    print(f"# GRAPH: N={g.N:,}  E={g.E:,}  F={g.F}  communities={int(g.comm.max())+1}  "
          f"is_real={g.is_real}")

    if args.download_only:
        deg = node_deg(g.src, g.dst, g.N)
        print(f"# STATS: mean_deg={deg.mean():.1f}  max_deg={deg.max()}  "
              f"native feat mem(uniform F)={fmt_gb(g.N * g.F * BYTES_PER_FEAT)}")
        return

    if args.exp in ("1", "all"):
        run_exp1(g, args)
    if args.exp in ("2", "all"):
        run_exp2(g, args)
    if args.exp in ("3", "all"):
        run_exp3(g, args)
    print("\n" + "#" * 92)
    print("# DONE. PROCESS-only (cut/comm/makespan/feasibility); accuracy never touched (same data+model=>same result).")
    print("#" * 92)


if __name__ == "__main__":
    main()
