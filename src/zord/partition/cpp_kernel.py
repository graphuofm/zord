"""C++ graph-kernel interface (the locality/density rankings ZORD's arrange needs).

The validated experiments (scripts/sota_compare.py, hetero_matched.py, dynamic_run.py)
all reach the SAME C++ binary `build/graph_algos` for the heavy structural passes --
label-propagation community order (`lpa`), k-core degeneracy order (`kcore`), and
degree order (`degree`) -- because those must scale to 100M-edge graphs (NEVER
networkx; per the project rule, graph algos live in C++). This module is the single
engine-side wrapper around that binary, with a pure-numpy fallback so the planner still
runs (correctly, just slower) when the binary is absent. The binary I/O format matches
the scripts exactly: int64 N, int64 M, then 2*M interleaved int32 (src,dst); out = int64 N
followed by N int32 `newid[old]->rank`.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile

import numpy as np


def graph_bin_path() -> str:
    """Resolve the graph_algos binary: $ZORD_GRAPH_BIN, else <repo>/build/graph_algos.
    Repo root is three levels up from this file (src/zord/partition/cpp_kernel.py)."""
    env = os.environ.get("ZORD_GRAPH_BIN")
    if env:
        return env
    here = os.path.abspath(__file__)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    return os.path.join(repo, "build", "graph_algos")


def have_cpp_kernel() -> bool:
    return os.path.exists(graph_bin_path())


def _write_edges(path: str, num_nodes: int, src: np.ndarray, dst: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", int(num_nodes), int(src.size)))
        inter = np.empty(2 * src.size, dtype=np.int32)
        inter[0::2] = src.astype(np.int32)
        inter[1::2] = dst.astype(np.int32)
        inter.tofile(f)


def cpp_order(num_nodes: int, src: np.ndarray, dst: np.ndarray, mode: str):
    """Run a C++ graph_algos `mode` and return newid[old]->rank (int64 [N]) or None.

    Returns None (caller falls back) if the binary is missing or the run fails.
    `mode` in {degree, kcore, bfs, lpa, dfs, slashburn, gorder}.
    """
    binp = graph_bin_path()
    if not os.path.exists(binp):
        return None
    with tempfile.TemporaryDirectory(prefix="zord_kernel_") as tmp:
        epath = os.path.join(tmp, "edges.bin")
        opath = os.path.join(tmp, "out.bin")
        _write_edges(epath, num_nodes, src, dst)
        r = subprocess.run([binp, epath, mode, opath], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        with open(opath, "rb") as f:
            n = struct.unpack("<q", f.read(8))[0]
            newid = np.fromfile(f, dtype=np.int32, count=n)
    return newid.astype(np.int64)


# --------------------------------------------------------------------------- #
# pure-numpy fallbacks (correct, just not 100M-scale). Used when the binary is #
# absent so the planner ALWAYS runs (the planner itself is pure CPU/numpy).    #
# --------------------------------------------------------------------------- #
def node_degree(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    return (np.bincount(src, minlength=num_nodes)
            + np.bincount(dst, minlength=num_nodes)).astype(np.int64)


def lpa_rank(num_nodes: int, src: np.ndarray, dst: np.ndarray, iters: int = 5):
    """Label-propagation community order -> rank[old]. Prefers the C++ `lpa`; falls
    back to a numpy synchronous LPA (mode of neighbor labels) then orders by label."""
    r = cpp_order(num_nodes, src, dst, "lpa")
    if r is not None:
        return r
    N = num_nodes
    u = np.concatenate([src, dst]).astype(np.int64)
    v = np.concatenate([dst, src]).astype(np.int64)
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.add.at(indptr, u + 1, 1)
    np.cumsum(indptr, out=indptr)
    lab = np.arange(N, dtype=np.int64)
    for _ in range(iters):
        nb_lab = lab[v]
        # weighted (count) mode of neighbor labels per node, via sort-segment-argmax
        key = u * np.int64(N) + nb_lab
        key.sort()
        # collapse to (node, label) groups
        new_grp = np.empty(key.size, dtype=bool)
        new_grp[0] = True
        new_grp[1:] = key[1:] != key[:-1]
        gid = np.cumsum(new_grp) - 1
        gcount = np.bincount(gid)
        g_node = (key[new_grp] // N)
        g_lab = (key[new_grp] % N)
        # per node, pick label of the max-count group
        node_new = np.empty(g_node.size, dtype=bool)
        node_new[0] = True
        node_new[1:] = g_node[1:] != g_node[:-1]
        nid = np.cumsum(node_new) - 1
        best = np.full(int(nid[-1]) + 1, -1, dtype=np.int64) if g_node.size else np.empty(0, np.int64)
        starts = np.nonzero(node_new)[0]
        ends = np.concatenate([starts[1:], [g_node.size]])
        for s_, e_ in zip(starts, ends):
            j = s_ + int(np.argmax(gcount[s_:e_]))
            best[nid[s_]] = g_lab[j]
        new_lab = lab.copy()
        nodes_with_nb = g_node[starts]
        new_lab[nodes_with_nb] = best
        if np.array_equal(new_lab, lab):
            break
        lab = new_lab
    seq = np.argsort(lab, kind="stable")           # cluster-grouped node order
    rank = np.empty(N, dtype=np.int64)
    rank[seq] = np.arange(N)
    return rank


def coreness_cpp(src: np.ndarray, dst: np.ndarray, num_nodes: int):
    """Per-node coreness VALUES via the C++ `kcorevals` mode (Batagelj-Zaversnik, O(M)).

    Returns core[v] (int64 [N]) or None so the caller can fall back to the numpy peel.
    None is returned when the binary is missing/old (unknown mode -> nonzero exit) OR the
    run fails. To match the numpy peel EXACTLY we feed the binary the same canonicalized
    edge set: self-loops dropped and undirected duplicates removed (the C++ side builds CSR
    straight from the edges it is given, so parallel edges / self-loops would otherwise
    inflate degrees and change core numbers)."""
    binp = graph_bin_path()
    if not os.path.exists(binp):
        return None
    N = num_nodes
    s = src.astype(np.int64); d = dst.astype(np.int64)
    m = s != d
    a = np.minimum(s[m], d[m]); b = np.maximum(s[m], d[m])
    if a.size == 0:
        return np.zeros(N, dtype=np.int64)
    key = np.unique(a * np.int64(N) + b)
    a = key // N; b = key % N
    with tempfile.TemporaryDirectory(prefix="zord_kernel_") as tmp:
        epath = os.path.join(tmp, "edges.bin")
        opath = os.path.join(tmp, "out.bin")
        _write_edges(epath, N, a, b)
        r = subprocess.run([binp, epath, "kcorevals", opath], capture_output=True, text=True)
        if r.returncode != 0:
            return None  # binary too old (unknown mode) or run failure -> numpy fallback
        with open(opath, "rb") as f:
            n = struct.unpack("<q", f.read(8))[0]
            core = np.fromfile(f, dtype=np.int32, count=n)
    if core.size != N:
        return None
    return core.astype(np.int64)


def coreness(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    """Per-node coreness VALUES (not order) for the planner's vertex-cut SWEEP quantiles.

    PREFERS the C++ `kcorevals` Batagelj-Zaversnik path (O(M); scales to 100M-edge clustered
    graphs) when ZORD_GRAPH_BIN is set / the binary exists, FALLING BACK to the vectorized
    numpy peel below when the binary is missing or its mode is unavailable. Both paths produce
    IDENTICAL core numbers (the C++ side is fed the same deduplicated, self-loop-free edges).
    The numpy peel is O(M * rounds) and on highly clustered graphs `rounds` is large -- it
    hung 46min on an 8M-node/100M-edge intra=0.9 graph -- which is why the C++ path is preferred."""
    if os.environ.get("ZORD_GRAPH_BIN") or have_cpp_kernel():
        cc = coreness_cpp(src, dst, num_nodes)
        if cc is not None:
            return cc
    N = num_nodes
    src = src.astype(np.int64); dst = dst.astype(np.int64)
    m = src != dst
    a = np.minimum(src[m], dst[m]); b = np.maximum(src[m], dst[m])
    if a.size == 0:
        return np.zeros(N, dtype=np.int64)
    key = np.unique(a * np.int64(N) + b)
    a = key // N; b = key % N
    r = np.concatenate([a, b]); c = np.concatenate([b, a])
    o = np.argsort(r, kind="stable"); r = r[o]; c = c[o]
    off = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(r, minlength=N), out=off[1:])
    core = np.zeros(N, dtype=np.int64)
    cur = (off[1:] - off[:-1]).copy()
    removed = np.zeros(N, dtype=bool)
    alive = N; k = 0
    while alive > 0:
        amin = int(cur[~removed].min())
        if amin > k:
            k = amin
        peel = (~removed) & (cur <= k)
        while peel.any():
            pnodes = np.nonzero(peel)[0]
            core[pnodes] = k
            removed[pnodes] = True
            alive -= pnodes.size
            if alive == 0:
                break
            slots = _expand_ranges(off[pnodes], off[pnodes + 1])
            nbr = c[slots]
            nbr = nbr[~removed[nbr]]
            if nbr.size:
                cur -= np.bincount(nbr, minlength=N)
            peel = (~removed) & (cur <= k)
    return core


def _expand_ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Vectorized concat of arange(starts[i], ends[i]) with no Python loop."""
    lens = ends - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    idx = np.cumsum(lens)[:-1]
    out[0] = starts[0]
    out[idx] = starts[1:] - ends[:-1] + 1
    return np.cumsum(out)


# ---- zord's OWN multilevel k-way partitioner (build/multilevel) -- a real METIS-quality
#      min-cut in C++ (no pymetis, no superlinear size-gate; runs at 1B edges). Verified
#      cut == pymetis (1.00x) on SBM. -------------------------------------------------------
def multilevel_bin_path() -> str:
    import os
    from pathlib import Path
    env = os.environ.get("ZORD_MULTILEVEL_BIN")
    return env if env else str(Path(__file__).resolve().parents[3] / "build" / "multilevel")


def have_multilevel() -> bool:
    import os
    return os.path.exists(multilevel_bin_path())


def multilevel_partition(src, dst, num_nodes, num_parts, ubfactor: float = 1.03, ratio=None,
                         vwgt=None, ewgt=None, init=None):
    """zord's own multilevel k-way min-cut. Returns membership[N] in [0, num_parts).
    Coarsen (HEM) -> greedy k-way init -> k-way FM refine -> uncoarsen, all in C++.
    ratio (OPTIONAL): per-part target share (len num_parts, e.g. GPU throughput shares) -> the
    HETEROGENEITY-AWARE solution that balances COMPUTE (proportional part sizes) rather than node
    count. None -> equal split (the original behaviour) -- an additional solution, not a replace.
    ATTRIBUTE-AWARE (the temporal-GNN point: nodes AND edges carry features):
      vwgt (OPTIONAL, len N): per-node FEATURE BYTES (F_v*4) -> balance FEATURE MEMORY, not node
                              count; pair with ratio=HBM caps to land heavy-attribute nodes on
                              big-HBM cards. None -> unit (count balance, original behaviour).
      ewgt (OPTIONAL, len M): per-edge FEATURE BYTES (F_e) -> the min-cut minimizes the TRUE
                              boundary feature COMM (a cut edge moves F_n+F_e). None -> unit."""
    import os, struct, subprocess, tempfile
    binp = multilevel_bin_path()
    if not os.path.exists(binp):
        raise FileNotFoundError(binp)
    src = np.ascontiguousarray(src, dtype=np.int32)
    dst = np.ascontiguousarray(dst, dtype=np.int32)
    N, M, D = int(num_nodes), int(src.size), int(num_parts)
    has_ratio = 1 if ratio is not None else 0
    ratio_arr = np.ascontiguousarray(ratio, dtype="<f8") if has_ratio else None
    if has_ratio and ratio_arr.size != D:
        raise ValueError(f"ratio must have len num_parts={D}, got {ratio_arr.size}")
    with tempfile.TemporaryDirectory() as td:
        ip, op = os.path.join(td, "i"), os.path.join(td, "o")
        with open(ip, "wb") as fh:
            fh.write(struct.pack("<2q", N, M)); fh.write(src.tobytes()); fh.write(dst.tobytes())
            fh.write(struct.pack("<i", D)); fh.write(struct.pack("<d", float(ubfactor)))
            fh.write(struct.pack("<i", has_ratio))
            if has_ratio:
                fh.write(ratio_arr.tobytes())
            has_vwgt = 1 if vwgt is not None else 0
            fh.write(struct.pack("<i", has_vwgt))
            if has_vwgt:
                vw = np.ascontiguousarray(vwgt, dtype="<i8")
                if vw.size != N:
                    raise ValueError(f"vwgt must have len num_nodes={N}, got {vw.size}")
                fh.write(vw.tobytes())
            has_ewgt = 1 if ewgt is not None else 0
            fh.write(struct.pack("<i", has_ewgt))
            if has_ewgt:
                ew = np.ascontiguousarray(ewgt, dtype="<i8")
                if ew.size != M:
                    raise ValueError(f"ewgt must have len num_edges={M}, got {ew.size}")
                fh.write(ew.tobytes())
            has_init = 1 if init is not None else 0
            fh.write(struct.pack("<i", has_init))
            if has_init:   # POLISH: refine this partition directly (boundary FM, no coarsening)
                ip_arr = np.ascontiguousarray(init, dtype="<i4")
                if ip_arr.size != N:
                    raise ValueError(f"init must have len num_nodes={N}, got {ip_arr.size}")
                fh.write(ip_arr.tobytes())
        r = subprocess.run([binp, ip, op], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
        out = open(op, "rb").read()
    Nout = struct.unpack_from("<q", out, 0)[0]
    return np.frombuffer(out, dtype="<i4", count=Nout, offset=8).astype(np.int64)


# ---- streaming partitioners (build/streaming): single-pass Fennel/LDG (edge-cut) + HDRF
#      (vertex-cut). O(M), bounded memory -- the tool when the graph is a STREAM / doesn't fit /
#      arrives online. Quality < multilevel (no global view); value is single-pass/online. -------
def streaming_bin_path() -> str:
    import os
    from pathlib import Path
    env = os.environ.get("ZORD_STREAMING_BIN")
    return env if env else str(Path(__file__).resolve().parents[3] / "build" / "streaming")


def have_streaming() -> bool:
    import os
    return os.path.exists(streaming_bin_path())


def streaming_partition(src, dst, num_nodes, num_parts, mode: str = "fennel", param: float = 0.0,
                        order=None, vwgt=None):
    """Single-pass streaming partition. mode in {fennel, ldg, hdrf}. Returns (membership, info):
    fennel/ldg -> membership[N] (edge-cut, vertex->part); hdrf -> edge_part[M] (vertex-cut,
    edge->part). param = gamma(fennel)/slack(ldg)/lambda(hdrf), 0 -> default.
    order (OPTIONAL, fennel/ldg): a node ARRIVAL ORDER to stream in (e.g. lpa_rank) for a much
    better cut than id-order -- an additional solution, NOT a replacement (None -> id-order)."""
    import os, struct, subprocess, tempfile
    binp = streaming_bin_path()
    if not os.path.exists(binp):
        raise FileNotFoundError(binp)
    mid = {"fennel": 0, "ldg": 1, "hdrf": 2}[mode]
    src = np.ascontiguousarray(src, dtype=np.int32); dst = np.ascontiguousarray(dst, dtype=np.int32)
    N, M, D = int(num_nodes), int(src.size), int(num_parts)
    has_order = 1 if order is not None else 0
    order_arr = np.ascontiguousarray(order, dtype=np.int32) if has_order else None
    has_vwgt = 1 if vwgt is not None else 0
    vwgt_arr = np.ascontiguousarray(vwgt, dtype="<i8") if has_vwgt else None
    if has_vwgt and vwgt_arr.size != N:
        raise ValueError(f"vwgt must have len num_nodes={N}, got {vwgt_arr.size}")
    with tempfile.TemporaryDirectory() as td:
        ip, op = os.path.join(td, "i"), os.path.join(td, "o")
        with open(ip, "wb") as fh:
            fh.write(struct.pack("<2q", N, M)); fh.write(src.tobytes()); fh.write(dst.tobytes())
            fh.write(struct.pack("<ii", D, mid)); fh.write(struct.pack("<d", float(param)))
            fh.write(struct.pack("<i", has_order))
            if has_order:
                fh.write(order_arr.tobytes())
            fh.write(struct.pack("<i", has_vwgt))
            if has_vwgt:
                fh.write(vwgt_arr.tobytes())
        r = subprocess.run([binp, ip, op], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
        out = open(op, "rb").read()
    cnt = struct.unpack_from("<q", out, 4)[0]
    arr = np.frombuffer(out, dtype="<i4", count=cnt, offset=12).astype(np.int64)
    info = next((l for l in r.stderr.decode("utf-8", "replace").splitlines() if l.startswith("STAT")), "")
    return arr, info


# ---- spectral (build/spectral): Fiedler vector + algebraic connectivity lambda2 (deflated power
#      iteration on the normalized adjacency). Gives the REAL lambda2 for the THEORY.md Cheeger
#      lower bound h>=lambda2/2 (was approximated), and a spectral bipartition. Verified vs scipy
#      eigsh: lambda2 relerr ~2% (power-iter slight under-estimate keeps the lower bound valid). --
def spectral_bin_path() -> str:
    import os
    from pathlib import Path
    env = os.environ.get("ZORD_SPECTRAL_BIN")
    return env if env else str(Path(__file__).resolve().parents[3] / "build" / "spectral")


def have_spectral() -> bool:
    import os
    return os.path.exists(spectral_bin_path())


def spectral_lambda2(src, dst, num_nodes, iters: int = 200, seed: int = 0, vwgt=None):
    """Algebraic connectivity lambda2 (2nd-smallest eigenvalue of the normalized Laplacian) +
    the Fiedler bipartition. Returns (lambda2, cheeger_lb=lambda2/2, part[N] in {0,1}).
    vwgt (OPTIONAL, len N): per-node FEATURE BYTES -> split the Fiedler order at the WEIGHTED
    median so both sides hold equal FEATURE MEMORY (attribute-aware). None -> sign split."""
    import os, struct, subprocess, tempfile
    binp = spectral_bin_path()
    if not os.path.exists(binp):
        raise FileNotFoundError(binp)
    src = np.ascontiguousarray(src, dtype=np.int32); dst = np.ascontiguousarray(dst, dtype=np.int32)
    N, M = int(num_nodes), int(src.size)
    has_vw = 1 if vwgt is not None else 0
    vw = np.ascontiguousarray(vwgt, dtype="<i8") if has_vw else None
    if has_vw and vw.size != N:
        raise ValueError(f"vwgt must have len num_nodes={N}, got {vw.size}")
    with tempfile.TemporaryDirectory() as td:
        ip, op = os.path.join(td, "i"), os.path.join(td, "o")
        with open(ip, "wb") as fh:
            fh.write(struct.pack("<2q", N, M)); fh.write(src.tobytes()); fh.write(dst.tobytes())
            fh.write(struct.pack("<ii", int(iters), int(seed)))
            fh.write(struct.pack("<i", has_vw))
            if has_vw:
                fh.write(vw.tobytes())
        r = subprocess.run([binp, ip, op], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[-300:])
        out = open(op, "rb").read()
    Nout = struct.unpack_from("<q", out, 0)[0]
    lam2, clb = struct.unpack_from("<2d", out, 8)
    part = np.frombuffer(out, dtype="<i4", count=Nout, offset=24).astype(np.int64)
    return float(lam2), float(clb), part
