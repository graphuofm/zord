#!/usr/bin/env python3
"""EXP — ATTRIBUTE-AWARE partitioning, multi-scale (24k -> 63.5M -> 1B), BLIND vs AWARE vs SOTA.

The zord thesis under test: a temporal-GNN graph carries features on NODES and EDGES; the memory +
comm cost is dominated by those FEATURES, not the structure. An attribute-BLIND partitioner (count-
balanced METIS/hash) minimizes the structural cut but leaves FEATURE MEMORY imbalanced -> the heavy-
feature device OOMs. zord's attribute-AWARE kernel balances FEATURE BYTES (vertex weight = F_v) and
cuts on FEATURE COMM (edge weight = F_e). PROCESS-only: placement changes, the GNN result does not.

ROBUST BY DESIGN (user req): every (dataset, method) runs in an ISOLATED subprocess with a wall-clock
TIMEOUT. A hang / OOM / crash is recorded as TIMEOUT / FAIL and the campaign MOVES ON to the next --
never blocks the whole run. Results streamed to CSV + JSONL as they complete.

REAL data for quality (staged temporal graphs incl. JODIE 172-dim EDGE features); a clearly-LABELLED
RMAT (Graph500) point at billion-edge scale for FEASIBILITY/throughput only (never a quality claim).

  # full campaign (orchestrator)
  python3 scripts/exp_attr_partition.py --parts 8 --out results/attr_partition
  # one cell (worker; used internally, also runnable by hand)
  python3 scripts/exp_attr_partition.py --worker --dataset jodie-wikipedia --method zord-mc-aware --parts 8
"""
import argparse, json, os, struct, subprocess, sys, time, resource

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import numpy as np

_REPO = os.path.dirname(_SRC)
FEAT_BYTES = 4            # fp32 per feature dim
DEFAULT_FEAT_DIM = 128    # node embedding width when a dataset has no native node features

# ----------------------------------------------------------------------------- campaign
# (name, kind, params, timeout_s).  edges noted for context; timeouts scale with size.
CAMPAIGN = [
    ("bitcoin-alpha",   "real", {},                               120),
    ("collegemsg",      "real", {},                               120),
    ("jodie-wikipedia", "real", {},                               180),   # 172-dim EDGE features (real)
    ("mathoverflow",    "real", {},                               300),
    ("jodie-reddit",    "real", {},                               420),   # 172-dim EDGE features (real)
    ("askubuntu",       "real", {},                               600),
    ("superuser",       "real", {},                               900),
    ("wiki-talk",       "real", {},                              1800),   # 7.8M
    ("stackoverflow",   "real", {},                              3600),   # 63.5M
    ("rmat-100M",       "rmat", {"nlog2": 25, "m": 100_000_000},  3600),  # scalability (LABELLED synth)
    ("rmat-1B",         "rmat", {"nlog2": 28, "m": 1_000_000_000},7200),  # BILLION-edge feasibility
]
METHODS = ["hash", "metis-blind", "metis-aware",
           "zord-mc-blind", "zord-mc-aware", "zord-stream-aware", "hdrf"]


# ----------------------------------------------------------------------------- helpers
def peak_rss_mb():
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(s, c) / 1024.0, 1)          # ru_maxrss is KB on Linux


def build_fv(N, feat_dim, mode, seed):
    """Per-node feature DIMS F_v. uniform -> every node feat_dim (the real null for uniform-F data).
    multimodal -> a MODELED heterogeneous mix (80% thin feat_dim, 15% x4, 5% x16) on the REAL graph
    structure -- CLEARLY LABELLED; exposes the thick/thin feasibility win (real open data is uniform-F)."""
    if mode == "uniform":
        return np.full(N, feat_dim, dtype=np.int64)
    rng = np.random.default_rng(seed)
    r = rng.random(N)
    fv = np.full(N, feat_dim, dtype=np.int64)
    fv[r > 0.80] = feat_dim * 4
    fv[r > 0.95] = feat_dim * 16
    return fv


def load_graph(args):
    """-> (src, dst, N, fe_dim).  fe_dim = native EDGE-feature dim (0 if none)."""
    if args.dataset.startswith("rmat"):
        binp = os.environ.get("ZORD_RMAT_BIN", os.path.join(_REPO, "build", "rmat"))
        out = os.path.join(args.tmp, "rmat.bin")
        subprocess.run([binp, str(args.rmat_nlog2), str(args.rmat_m), "1", out], check=True)
        with open(out, "rb") as f:
            N = struct.unpack("<q", f.read(8))[0]; M = struct.unpack("<q", f.read(8))[0]
            src = np.fromfile(f, dtype="<i4", count=M); dst = np.fromfile(f, dtype="<i4", count=M)
        try: os.remove(out)
        except OSError: pass
        return src, dst, N, 0
    from zord.datasets import load
    g = load(args.dataset).sort_by_time()
    fe = getattr(g, "efeat", None)
    fe_dim = int(fe.shape[1]) if (fe is not None and getattr(fe, "ndim", 0) == 2) else 0
    return np.asarray(g.src), np.asarray(g.dst), int(g.num_nodes), fe_dim


def metis_part(src, dst, N, D, vweights=None):
    import pymetis
    from zord.partition.hetero import _build_csr
    indptr, adj = _build_csr(np.asarray(src, np.int64), np.asarray(dst, np.int64), N)
    kw = {}
    if vweights is not None:
        kw["vweights"] = np.maximum(1, np.asarray(vweights, np.int64)).tolist()
    _, membership = pymetis.part_graph(D, xadj=indptr.tolist(), adjncy=adj.tolist(), **kw)
    return np.asarray(membership, dtype=np.int64)


def run_method(method, src, dst, N, D, fv_dims, fe_dim):
    """Return (assignment, kind). kind 'vertex' (node->part) or 'edge' (edge->part, vertex-cut)."""
    from zord.partition import cpp_kernel as CK
    fv_bytes = (fv_dims * FEAT_BYTES).astype(np.int64)
    if method == "hash":
        return (np.arange(N, dtype=np.int64) % D), "vertex"
    if method == "metis-blind":
        return metis_part(src, dst, N, D, None), "vertex"
    if method == "metis-aware":
        return metis_part(src, dst, N, D, fv_dims), "vertex"          # vweights ~ feature dims
    if method == "zord-mc-blind":
        return CK.multilevel_partition(src, dst, N, D), "vertex"
    if method == "zord-mc-aware":
        ewgt = None
        if fe_dim > 0:
            ewgt = np.full(int(np.asarray(src).size), fe_dim * FEAT_BYTES, dtype=np.int64)
        return CK.multilevel_partition(src, dst, N, D, vwgt=fv_bytes, ewgt=ewgt), "vertex"
    if method == "zord-stream-aware":
        part, _ = CK.streaming_partition(src, dst, N, D, mode="fennel", vwgt=fv_bytes)
        return part, "vertex"
    if method == "hdrf":
        epart, _ = CK.streaming_partition(src, dst, N, D, mode="hdrf")
        return epart, "edge"
    raise ValueError(f"unknown method {method}")


def compute_metrics(part, kind, src, dst, N, D, fv_dims, fe_dim, cap_bytes):
    fv_bytes = (fv_dims * FEAT_BYTES).astype(np.float64)
    m = {}
    if kind == "vertex":
        part = np.asarray(part, np.int64)
        cross = part[src] != part[dst]
        m["cut"] = int(cross.sum())
        # feature COMM per cut edge: remote node row (F_v of dst) + edge feature (F_e). bytes.
        fe_b = fe_dim * FEAT_BYTES
        m["feat_comm_mb"] = round(float(fv_bytes[dst[cross]].sum() + fe_b * int(cross.sum())) / 1e6, 2)
        cnt = np.bincount(part, minlength=D).astype(np.float64)
        fmem = np.bincount(part, weights=fv_bytes, minlength=D)
        m["balance_count"] = round(float(cnt.max() / max(1.0, cnt.mean())), 3)
        m["balance_featmem"] = round(float(fmem.max() / max(1.0, fmem.mean())), 3)
        m["peak_featmem_gb"] = round(float(fmem.max()) / 1e9, 3)
        m["feasible"] = bool(fmem.max() <= cap_bytes)
    else:  # edge / vertex-cut (hdrf)
        epart = np.asarray(part, np.int64)
        M = epart.size
        # replication factor: avg # distinct parts a vertex's edges touch
        key = np.concatenate([src, dst]).astype(np.int64) * D + np.concatenate([epart, epart])
        repl = np.unique(key).size / max(1, N)
        ebal = np.bincount(epart, minlength=D).max() / max(1.0, M / D)
        m["cut"] = None; m["feat_comm_mb"] = None
        m["balance_count"] = round(float(ebal), 3)          # edge balance here
        m["balance_featmem"] = None
        m["replication_factor"] = round(float(repl), 3)
        m["feasible"] = None
    return m


# ----------------------------------------------------------------------------- worker
def worker(args):
    t0 = time.time()
    src, dst, N, fe_dim = load_graph(args)
    src = np.ascontiguousarray(src, np.int64); dst = np.ascontiguousarray(dst, np.int64)
    M = int(src.size)
    fv_dims = build_fv(N, args.feat_dim, args.fv, args.seed)
    cap_bytes = args.cap_gb * (1024 ** 3)
    load_s = time.time() - t0
    t1 = time.time()
    part, kind = run_method(args.method, src, dst, N, args.parts, fv_dims, fe_dim)
    part_s = time.time() - t1
    met = compute_metrics(part, kind, src, dst, N, args.parts, fv_dims, fe_dim, cap_bytes)
    rec = dict(dataset=args.dataset, method=args.method, parts=args.parts, N=int(N), M=M,
               fe_dim=fe_dim, feat_dim=args.feat_dim, fv_mode=args.fv, kind=kind,
               load_s=round(load_s, 2), part_s=round(part_s, 3), peak_rss_mb=peak_rss_mb(),
               status="OK", **met)
    print("ZORDJSON " + json.dumps(rec))


# ----------------------------------------------------------------------------- orchestrator
def orchestrator(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    csv_p, jsonl_p = args.out + ".csv", args.out + ".jsonl"
    cols = ["dataset", "method", "parts", "N", "M", "fe_dim", "fv_mode", "kind", "cut",
            "feat_comm_mb", "balance_count", "balance_featmem", "peak_featmem_gb", "feasible",
            "replication_factor", "part_s", "peak_rss_mb", "status", "note"]
    only_ds = set(args.datasets.split(",")) if args.datasets else None
    only_m = set(args.methods.split(",")) if args.methods else None
    with open(csv_p, "w") as cf:
        cf.write(",".join(cols) + "\n")
    print(f"[campaign] parts={args.parts} fv={args.fv} cap={args.cap_gb}GB -> {csv_p}\n")
    print(f"{'dataset':16} {'method':18} {'status':8} {'cut':>12} {'feat_comm_mb':>12} "
          f"{'bal_fmem':>9} {'peakF_gb':>9} {'feas':>5} {'part_s':>9}")
    for name, kind, params, timeout in CAMPAIGN:
        if only_ds and name not in only_ds:
            continue
        for method in METHODS:
            if only_m and method not in only_m:
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--dataset", name,
                   "--method", method, "--parts", str(args.parts), "--feat-dim", str(args.feat_dim),
                   "--fv", args.fv, "--cap-gb", str(args.cap_gb), "--tmp", args.tmp]
            if kind == "rmat":
                cmd += ["--rmat-nlog2", str(params["nlog2"]), "--rmat-m", str(params["m"])]
            rec = dict(dataset=name, method=method, parts=args.parts, status="?", note="")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                line = next((l for l in r.stdout.splitlines() if l.startswith("ZORDJSON ")), None)
                if line:
                    rec = json.loads(line[len("ZORDJSON "):])
                else:
                    rec["status"] = "FAIL"; rec["note"] = (r.stderr.strip().splitlines() or [""])[-1][:120]
            except subprocess.TimeoutExpired:
                rec["status"] = "TIMEOUT"; rec["note"] = f"> {timeout}s"
            except Exception as e:                                   # never let one cell kill the run
                rec["status"] = "FAIL"; rec["note"] = str(e)[:120]
            with open(jsonl_p, "a") as jf:
                jf.write(json.dumps(rec) + "\n")
            with open(csv_p, "a") as cf:
                cf.write(",".join(str(rec.get(c, "")) for c in cols) + "\n")
            print(f"{name:16} {method:18} {rec['status']:8} {str(rec.get('cut','')):>12} "
                  f"{str(rec.get('feat_comm_mb','')):>12} {str(rec.get('balance_featmem','')):>9} "
                  f"{str(rec.get('peak_featmem_gb','')):>9} {str(rec.get('feasible','')):>5} "
                  f"{str(rec.get('part_s','')):>9}")
    print(f"\n[campaign] done -> {csv_p} , {jsonl_p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="run ONE (dataset,method) cell + print JSON")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--method", default="")
    ap.add_argument("--parts", "-D", type=int, default=8)
    ap.add_argument("--feat-dim", type=int, default=DEFAULT_FEAT_DIM)
    ap.add_argument("--fv", choices=["uniform", "multimodal"], default="multimodal",
                    help="per-node feature-size model (multimodal is MODELED + labelled)")
    ap.add_argument("--cap-gb", type=float, default=32.0, help="per-device HBM cap for feasibility (smallest card)")
    ap.add_argument("--rmat-nlog2", type=int, default=25)
    ap.add_argument("--rmat-m", type=int, default=100_000_000)
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("--datasets", default="", help="comma list to restrict (orchestrator)")
    ap.add_argument("--methods", default="", help="comma list to restrict (orchestrator)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/attr_partition")
    args = ap.parse_args()
    if args.worker:
        worker(args)
    else:
        orchestrator(args)


if __name__ == "__main__":
    main()
