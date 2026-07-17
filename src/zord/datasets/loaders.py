"""Loaders: turn a staged dataset file into a canonical TemporalGraph.

Supported formats:
  - snap_edgelist : whitespace "src dst unixts" (.txt or .txt.gz), '#' comments
  - bitcoin_csv   : "source,target,rating,time" (.csv or .csv.gz)
  - tgb_linkprop  : TGB LinkPropPredDataset (needs py-tgb installed); captures edge_feat
  - jodie         : JODIE CSV (user,item,ts,label, then comma-sep edge features)

`load(name)` dispatches via the registry and prefers the HetCluster staged path,
falling back to downloading from the spec url.
"""
from __future__ import annotations

import gzip
import os
import urllib.request
from typing import Optional

import numpy as np

from .registry import get_spec, DatasetSpec
from .temporal_graph import TemporalGraph


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def load_snap_edgelist(path: str, name: str = "snap", remap: bool = True) -> TemporalGraph:
    src_l, dst_l, ts_l = [], [], []
    with _open(path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            p = line.split()
            if len(p) < 3:
                continue
            src_l.append(int(p[0])); dst_l.append(int(p[1])); ts_l.append(int(float(p[2])))
    src = np.asarray(src_l, dtype=np.int64)
    dst = np.asarray(dst_l, dtype=np.int64)
    ts = np.asarray(ts_l, dtype=np.int64)
    nn = None
    if remap:
        src, dst, nn = _remap_nodes(src, dst)
    return TemporalGraph(src=src, dst=dst, t=ts, num_nodes=nn, name=name)


def load_bitcoin_csv(path: str, name: str = "bitcoin", remap: bool = True) -> TemporalGraph:
    src_l, dst_l, w_l, ts_l = [], [], [], []
    with _open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 4:
                continue
            src_l.append(int(p[0])); dst_l.append(int(p[1]))
            w_l.append(float(p[2])); ts_l.append(int(float(p[3])))
    src = np.asarray(src_l, dtype=np.int64); dst = np.asarray(dst_l, dtype=np.int64)
    w = np.asarray(w_l, dtype=np.float32); ts = np.asarray(ts_l, dtype=np.int64)
    nn = None
    if remap:
        src, dst, nn = _remap_nodes(src, dst)
    return TemporalGraph(src=src, dst=dst, t=ts, w=w, num_nodes=nn, name=name)


def load_tgb(name: str, root: Optional[str] = None) -> TemporalGraph:
    from tgb.linkproppred.dataset import LinkPropPredDataset  # lazy
    d = LinkPropPredDataset(name=name, root=root or ".", preprocess=True)
    data = d.full_data
    src = np.asarray(data["sources"], dtype=np.int64)
    dst = np.asarray(data["destinations"], dtype=np.int64)
    ts = np.asarray(data["timestamps"], dtype=np.int64)
    # Many TGB link-prop datasets (e.g. tgbl-wiki) ship per-edge features under
    # data["edge_feat"]; capture them as efeat [E, Fe] when present (was DROPPED before).
    efeat = None
    ef = data.get("edge_feat") if isinstance(data, dict) else None
    if ef is not None:
        ef = np.asarray(ef, dtype=np.float32)
        if ef.ndim == 1:
            ef = ef.reshape(-1, 1)
        if ef.shape[0] == src.shape[0]:   # only attach if it aligns with the edges
            efeat = ef
    return TemporalGraph(src=src, dst=dst, t=ts, efeat=efeat, name=name)


def load_jodie(name: str, path: Optional[str] = None) -> TemporalGraph:
    """Load a JODIE-format temporal interaction dataset (wikipedia/reddit/mooc/lastfm).

    CSV layout (one header row, then rows):
        user_id, item_id, timestamp, state_label, feat_0, feat_1, ..., feat_{Fe-1}
    Users and items are mapped into a SHARED contiguous node-id space (users first,
    then items), the trailing comma-separated columns become edge features efeat [E, Fe].
    `path` may point at a staged/downloaded <name>.csv (.csv or .csv.gz); otherwise the
    caller is expected to have downloaded it from the spec url.
    """
    users, items, ts_l, feats = [], [], [], []
    with _open(path) as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                # JODIE files carry a header row (user_id,item_id,timestamp,state_label,...)
                head = line.split(",", 4)
                if head and not _is_number(head[0]):
                    continue  # skip header
            p = line.split(",")
            if len(p) < 4:
                continue
            users.append(int(float(p[0]))); items.append(int(float(p[1])))
            ts_l.append(int(float(p[2])))
            feats.append([float(x) for x in p[4:]])  # cols after state_label = edge features
    u = np.asarray(users, dtype=np.int64)
    it = np.asarray(items, dtype=np.int64)
    ts = np.asarray(ts_l, dtype=np.int64)
    # shared node space: users in [0, nU), items in [nU, nU+nI)
    uu, u_inv = np.unique(u, return_inverse=True)
    ii, i_inv = np.unique(it, return_inverse=True)
    nU = int(uu.shape[0])
    src = u_inv.astype(np.int64)
    dst = (i_inv + nU).astype(np.int64)
    nn = nU + int(ii.shape[0])
    efeat = None
    if feats and len(feats[0]) > 0:
        Fe = len(feats[0])
        if all(len(r) == Fe for r in feats):       # well-formed rectangular feature block
            efeat = np.asarray(feats, dtype=np.float32)
    return TemporalGraph(src=src, dst=dst, t=ts, efeat=efeat, num_nodes=nn, name=name)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _remap_nodes(src: np.ndarray, dst: np.ndarray):
    """Remap (possibly sparse) node ids to a contiguous [0, n) range."""
    uniq, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    e = src.shape[0]
    return inv[:e].astype(np.int64), inv[e:].astype(np.int64), int(uniq.shape[0])


def load(name: str, prefer: str = "hetcluster") -> TemporalGraph:
    spec: DatasetSpec = get_spec(name)
    path = spec.hetcluster_path
    if spec.fmt == "tgb_linkprop":
        return load_tgb(spec.name, root=path)
    # file-based formats: use staged path if present, else download
    if not (path and os.path.exists(path)):
        path = _ensure_local(spec)
    if spec.fmt == "snap_edgelist":
        return load_snap_edgelist(path, name=spec.name)
    if spec.fmt == "bitcoin_csv":
        return load_bitcoin_csv(path, name=spec.name)
    if spec.fmt == "jodie":
        return load_jodie(spec.name, path=path)
    raise ValueError(f"no loader for fmt {spec.fmt!r} (dataset {name!r})")


def _ensure_local(spec: DatasetSpec, cache_dir: str = "~/.cache/zord") -> str:
    if not spec.url:
        raise FileNotFoundError(f"{spec.name}: not staged and no url to download")
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, os.path.basename(spec.url))
    if not os.path.exists(dst):
        urllib.request.urlretrieve(spec.url, dst)
    return dst
