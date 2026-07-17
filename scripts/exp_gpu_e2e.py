#!/usr/bin/env python3
"""EXP — GPU END-TO-END: real multi-GPU execution of a partitioned L-layer GNN aggregation.

Measures, on REAL GPUs (homogeneous single-node N-GPU; heterogeneous via the 2-node phase):
  - step_ms        : median wall-clock per training-step forward aggregation (CUDA-event timed)
  - peak_mb        : per-GPU torch.cuda.max_memory_allocated (the feasibility quantity)
  - same_result    : multi-GPU output == single-device reference (max-abs-diff, fp32 tol)
  - halo_rows      : boundary feature rows actually exchanged per layer (the cut, realized)
for each PARTITION METHOD (hash / metis-aware / zord-mc-aware / zord-stream-aware) x GPU count.

PROCESS-only: the aggregation result is identical across all placements (certified per cell);
only time/memory/feasibility change. Robust by design: the orchestrator runs each dataset in a
subprocess with a TIMEOUT; a hang/OOM is recorded and the campaign moves on. Results stream to
CSV+JSONL as cells complete. Industrial niceties: dataset .npz cache (parse the .gz once),
chunk-free cuSPARSE SpMM (no E x F materialization), auto GPU detection.

  # on a GPU node (inside SLURM):
  python3 scripts/exp_gpu_e2e.py --out results/gpu_e2e_rtx6000 --ngpus 1,4,8 --feat-dim 256
  # one worker by hand:
  python3 scripts/exp_gpu_e2e.py --worker --dataset wiki-talk --ngpus 1,4,8 --feat-dim 256
"""
import argparse, json, os, signal, subprocess, sys, time

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import numpy as np

CACHE = os.environ.get("ZORD_DATA_CACHE", "/tmp/zord_cache")   # node-local: NEVER a quota'd share
METHODS = ["hash", "metis-aware", "zord-mc-aware", "zord-stream-aware", "zord-stream-mem",
           "zord-polish", "kaminpar-aware", "tensor-split"]
CAMPAIGN = [  # (dataset, per-dataset worker timeout seconds)
    ("jodie-wikipedia", 900),
    ("wiki-talk", 2400),
    ("stackoverflow", 5400),
    ("rmat-100M", 7200),      # 100M edges: beyond METIS's reach; zord kernels only
]
RMAT = {"rmat-100M": (25, 100_000_000)}   # name -> (nlog2, edges); generated on the node
WARMUP, TIMED = 3, 10


def load_cached(name):
    """Parse the staged dataset once; npz-cache src/dst/N for fast reloads."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{name}.npz")
    if name in RMAT and not os.path.exists(p):
        import subprocess as sp, struct as st
        nlog2, m = RMAT[name]
        binp = os.environ.get("ZORD_RMAT_BIN",
                              os.path.join(os.path.dirname(_SRC), "build", "rmat"))
        raw = os.path.join(CACHE, f"{name}.bin")
        sp.run([binp, str(nlog2), str(m), "1", raw], check=True)
        with open(raw, "rb") as fh:
            N = st.unpack("<q", fh.read(8))[0]; M = st.unpack("<q", fh.read(8))[0]
            src = np.fromfile(fh, dtype="<i4", count=M).astype(np.int64)
            dst = np.fromfile(fh, dtype="<i4", count=M).astype(np.int64)
        os.remove(raw)
        try:
            np.savez(p, src=src, dst=dst, N=np.int64(N))
        except OSError:
            pass
        return src, dst, int(N)
    if os.path.exists(p):
        try:
            z = np.load(p)
            return z["src"], z["dst"], int(z["N"])
        except Exception as e:            # truncated/corrupt cache (e.g. quota hit mid-write)
            print(f"[cache] corrupt, re-parsing ({e})", file=sys.stderr)
            try: os.remove(p)
            except OSError: pass
    from zord.datasets import load
    g = load(name).sort_by_time()
    src = np.ascontiguousarray(g.src, np.int64); dst = np.ascontiguousarray(g.dst, np.int64)
    try:
        np.savez(p, src=src, dst=dst, N=np.int64(g.num_nodes))
    except OSError as e:                      # disk quota / read-only cache dir: cache is an
        print(f"[cache] skip ({e})", file=sys.stderr)   # optimization, never a failure
        try: os.remove(p)
        except OSError: pass
    return src, dst, int(g.num_nodes)


def build_fv(N, feat_dim, seed=0):
    """MODELED multimodal per-node feature dims (80% F, 15% 4F, 5% 16F) -- labelled, same as the
    partition campaign so the partitions answer the same question. GPU tensors still use uniform
    feat_dim rows (the *placement* is what varies); fv drives the AWARE partitions only."""
    rng = np.random.default_rng(seed)
    r = rng.random(N)
    fv = np.full(N, feat_dim, dtype=np.int64)
    fv[r > 0.80] = feat_dim * 4
    fv[r > 0.95] = feat_dim * 16
    return fv


def make_partition(method, src, dst, N, D, fv_dims):
    from zord.partition import cpp_kernel as CK
    fvb = (fv_dims * 4).astype(np.int64)
    if D == 1:
        return np.zeros(N, np.int64)
    if method == "hash":
        return np.arange(N, dtype=np.int64) % D
    if method == "metis-aware":
        import pymetis
        from zord.partition.hetero import _build_csr
        indptr, adj = _build_csr(src, dst, N)
        _, mem = pymetis.part_graph(D, xadj=indptr.tolist(), adjncy=adj.tolist(),
                                    vweights=np.maximum(1, fv_dims).tolist())
        return np.asarray(mem, np.int64)
    if method == "zord-mc-aware":
        return CK.multilevel_partition(src, dst, N, D, vwgt=fvb)
    if method == "zord-stream-aware":
        part, _ = CK.streaming_partition(src, dst, N, D, mode="fennel", vwgt=fvb)
        return np.asarray(part, np.int64)
    if method == "zord-stream-mem":
        # COMBINED weight: feature bytes + resident adjacency bytes (20.deg) -- balances the TOTAL
        # per-device memory AND (via deg) the incident-edge compute, the straggler fix the
        # 8xRTX6000 stackoverflow run exposed (stream-aware: fewest halo rows yet slowest step).
        deg = np.bincount(np.concatenate([src, dst]), minlength=N).astype(np.int64)
        w = fvb + 20 * deg
        part, _ = CK.streaming_partition(src, dst, N, D, mode="fennel", vwgt=w)
        return np.asarray(part, np.int64)
    if method == "zord-polish":
        # PORTFOLIO POLISH: start from the strongest available base partition and refine it with
        # zord's boundary FM under the unified byte account. METIS is a baseline, so zord treats
        # it as a warm start where it is feasible (<=80M edges); beyond that, polish stream-mem.
        deg = np.bincount(np.concatenate([src, dst]), minlength=N).astype(np.int64)
        w = fvb + 20 * deg
        if src.size <= 80_000_000:
            base = make_partition("metis-aware", src, dst, N, D, fv_dims)
        else:
            base = make_partition("zord-stream-mem", src, dst, N, D, fv_dims)
        part = CK.multilevel_partition(src, dst, N, D, vwgt=w, init=base)
        return np.asarray(part, np.int64)
    if method == "kaminpar-aware":
        # KaMinPar (deep multilevel, shared-memory parallel) with BYTE node weights -- the modern
        # scalable high-quality baseline reviewers ask for. Python bindings load METIS-format files.
        import kaminpar
        path = os.path.join(CACHE, f"kmp_{N}_{src.size}.metis")
        if not os.path.exists(path):
            write_metis_graph(path, src, dst, N, vwgt=np.maximum(1, fv_dims))
        g = kaminpar.load_graph(path, kaminpar.GraphFileFormat.METIS)
        inst = kaminpar.KaMinPar(int(os.environ.get("OMP_NUM_THREADS", "16")),
                                 kaminpar.default_context())
        try:
            p = inst.compute_partition(g, D, 0.03)
        except TypeError:
            p = inst.compute_partition(g, D)
        return np.asarray(p, np.int64)
    raise ValueError(method)


def write_metis_graph(path, src, dst, N, vwgt=None):
    """Write the graph in METIS ASCII format (1-indexed, symmetric, self-loops dropped,
    parallel edges deduped; fmt=10 when node weights are given)."""
    s = np.asarray(src, np.int64); d = np.asarray(dst, np.int64)
    m = s != d
    u = np.concatenate([s[m], d[m]]); v = np.concatenate([d[m], s[m]])
    key = u * np.int64(N) + v
    key = np.unique(key)                       # dedupe
    uu = (key // N).astype(np.int64); vv = (key % N).astype(np.int64)
    order = np.argsort(uu, kind="stable")
    uu, vv = uu[order], vv[order]
    indptr = np.zeros(N + 1, np.int64); np.add.at(indptr, uu + 1, 1); np.cumsum(indptr, out=indptr)
    M2 = key.size // 2                         # undirected edge count
    with open(path, "w") as fh:
        fh.write(f"{N} {M2} 10\n" if vwgt is not None else f"{N} {M2}\n")
        vv1 = (vv + 1).astype(str)
        for i in range(N):
            nbrs = " ".join(vv1[indptr[i]:indptr[i + 1]])
            if vwgt is not None:
                fh.write(f"{int(vwgt[i])} {nbrs}\n" if nbrs else f"{int(vwgt[i])}\n")
            else:
                fh.write(nbrs + "\n")


def run_cell(src, dst, N, part, D, F, layers, seed, W=None):
    """Execute the partitioned L-layer mean aggregation on D GPUs; return real measurements.
    W (optional, [F,F]): per-layer weight mix H <- relu((A@H)@W) -- the real-GNN form. Replicated
    on every GPU, so NODE-split pays no extra comm for it (tensor-split must all-reduce)."""
    import torch
    devs = [torch.device(f"cuda:{i}") for i in range(D)]
    # symmetric + self-loop edge set; aggregating-row owner = part[u]
    u = np.concatenate([src, dst, np.arange(N)]); v = np.concatenate([dst, src, np.arange(N)])
    part = np.asarray(part, np.int64)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, F)).astype(np.float32)

    locals_g, lidx = [], np.full(N, -1, np.int64)
    for d in range(D):
        g = np.nonzero(part == d)[0]; locals_g.append(g); lidx[g] = np.arange(g.size)

    A, tables, feats, halo_plan, halo_rows = [], [], [], [], 0
    pair_np = []       # per dest d: list of (owner o, send-idx np on o, table-pos np on d)
    for d in range(D):
        m = part[u] == d
        gu, gv = u[m], v[m]
        nl = locals_g[d].size
        hg = np.unique(gv[part[gv] != d]); nh = hg.size; halo_rows += nh
        tpos = np.full(N, -1, np.int64); tpos[locals_g[d]] = np.arange(nl)
        tpos[hg] = nl + np.arange(nh)
        ei, ej = lidx[gu], tpos[gv]
        o = np.argsort(ei, kind="stable"); ei, ej = ei[o], ej[o]
        deg = np.bincount(ei, minlength=nl).astype(np.float32)
        crow = np.zeros(nl + 1, np.int64); np.add.at(crow, ei + 1, 1); np.cumsum(crow, out=crow)
        vals = (1.0 / np.maximum(deg, 1.0))[ei].astype(np.float32)
        t = devs[d]
        A.append(torch.sparse_csr_tensor(torch.from_numpy(crow).to(t), torch.from_numpy(ej).to(t),
                                         torch.from_numpy(vals).to(t), size=(nl, nl + nh)))
        tables.append(torch.empty((nl + nh, F), dtype=torch.float32, device=t))
        feats.append(torch.from_numpy(X[locals_g[d]]).to(t))
        plan, pnp = [], []
        for o_ in range(D):
            if o_ == d: continue
            sel = hg[part[hg] == o_]
            if sel.size:      # exactly ONE index_select + ONE copy + ONE index_copy_ per (o,d) pair
                plan.append((o_, torch.from_numpy(lidx[sel]).to(devs[o_]),
                             torch.from_numpy(tpos[sel]).to(t)))
                pnp.append((o_, lidx[sel], tpos[sel]))
        halo_plan.append(plan); pair_np.append(pnp)

    Wt = [torch.from_numpy(W).to(devs[d]) for d in range(D)] if W is not None else None

    # ---- EXCHANGE ENGINE (method-independent: every partition runs the same path). The measured
    # 8-GPU exchange was OVERHEAD-bound, not bandwidth-bound (results/gpu_e2e_prof.csv: the bytes
    # need ~ms at PCIe rate; measured tens of ms): D*(D-1) pairwise copies issued serially on the
    # devices' default streams, each device-to-device copy staged by the DRIVER through the host
    # when P2P is off (global serialization). Fix, per mode (ZORD_EXCHANGE=auto|direct|pinned|
    # legacy; auto = direct if all-pairs P2P else pinned):
    #   direct: issue every inter-GPU copy on the OWNER's dedicated side stream (concurrent copy
    #           engines), scatter on the dest default stream (torch event-orders it), ONE host
    #           sync for the whole exchange.
    #   pinned: stage EXPLICITLY through pinned host buffers -- per owner ONE concatenated gather
    #           + ONE async D2H; per (owner,dest) ONE async H2D + ONE scatter; D2H/H2D overlap on
    #           separate copy engines across all GPUs. This is the no-P2P fast path.
    mode = os.environ.get("ZORD_EXCHANGE", "auto")
    p2p = (all(torch.cuda.can_device_access_peer(i, j)
               for i in range(D) for j in range(D) if i != j) if D > 1 else True)
    if mode == "auto":
        mode = "direct" if p2p else "pinned"
    if D > 1 and not getattr(run_cell, "_p2p_reported", False):
        run_cell._p2p_reported = True
        npairs = sum(1 for i in range(D) for j in range(D)
                     if i != j and torch.cuda.can_device_access_peer(i, j))
        print(f"[exchange] D={D} p2p_all_pairs={p2p} accessible_pairs={npairs}/{D*(D-1)} "
              f"mode={mode}", file=sys.stderr, flush=True)
    cs = [torch.cuda.Stream(device=devs[i]) for i in range(D)] if (D > 1 and mode != "legacy") else None
    hs = [torch.cuda.Stream(device=devs[i]) for i in range(D)] if (D > 1 and mode == "pinned") else None
    send_idx, host_buf, owner_ev = [None] * D, [None] * D, [None] * D
    recv_plan = [[] for _ in range(D)]
    if D > 1 and mode == "pinned":
        per_owner = [[] for _ in range(D)]
        for d in range(D):
            for o_, snp, rnp in pair_np[d]:
                per_owner[o_].append((d, snp, rnp))
        for o_ in range(D):
            if not per_owner[o_]:
                continue
            cat = np.concatenate([snp for _, snp, _ in per_owner[o_]])
            send_idx[o_] = torch.from_numpy(cat).to(devs[o_])
            host_buf[o_] = torch.empty((cat.size, F), dtype=torch.float32, pin_memory=True)
            owner_ev[o_] = torch.cuda.Event()
            off = 0
            for d, snp, rnp in per_owner[o_]:
                recv_plan[d].append((o_, off, off + snp.size, torch.from_numpy(rnp).to(devs[d]),
                                     torch.empty((snp.size, F), dtype=torch.float32, device=devs[d])))
                off += snp.size

    def _exchange_legacy():
        for d in range(D):
            tables[d][:feats[d].shape[0]] = feats[d]
            for o_, si, di in halo_plan[d]:
                tables[d].index_copy_(0, di, feats[o_].index_select(0, si).to(devs[d], non_blocking=True))

    def _exchange_direct():
        for o_ in range(D):                       # side streams see the freshest feats
            cs[o_].wait_stream(torch.cuda.default_stream(devs[o_]))
        for d in range(D):
            tables[d][:feats[d].shape[0]] = feats[d]
        pend = []
        for d in range(D):                        # ALL copies in flight before any scatter
            for o_, si, di in halo_plan[d]:
                with torch.cuda.stream(cs[o_]):   # copy runs on the OWNER's side stream
                    pend.append((d, di, feats[o_].index_select(0, si).to(devs[d], non_blocking=True)))
        for d, di, buf in pend:                   # dest default stream: event-ordered after copy
            tables[d].index_copy_(0, di, buf)

    def _exchange_pinned():
        for o_ in range(D):
            if send_idx[o_] is not None:
                cs[o_].wait_stream(torch.cuda.default_stream(devs[o_]))
        for d in range(D):
            tables[d][:feats[d].shape[0]] = feats[d]
        for o_ in range(D):                       # phase 1: ONE gather + ONE D2H per owner
            if send_idx[o_] is None:
                continue
            with torch.cuda.stream(cs[o_]):
                host_buf[o_].copy_(feats[o_].index_select(0, send_idx[o_]), non_blocking=True)
                owner_ev[o_].record(cs[o_])
        for d in range(D):                        # phase 2: per-pair H2D on the dest side stream
            for o_, s, e, di, db in recv_plan[d]:
                hs[d].wait_event(owner_ev[o_])
                with torch.cuda.stream(hs[d]):
                    db.copy_(host_buf[o_][s:e], non_blocking=True)
        for d in range(D):                        # phase 3: scatter after the dest's H2Ds
            torch.cuda.default_stream(devs[d]).wait_stream(hs[d])
            for o_, s, e, di, db in recv_plan[d]:
                tables[d].index_copy_(0, di, db)

    exchange = ({"legacy": _exchange_legacy, "direct": _exchange_direct,
                 "pinned": _exchange_pinned}[mode] if D > 1 else _exchange_legacy)

    phase_ms = {"exchange": 0.0, "compute": 0.0}

    def step():
        t0 = time.perf_counter()
        exchange()
        for d in range(D):
            torch.cuda.synchronize(devs[d])
        t1 = time.perf_counter()
        for d in range(D):
            h = torch.sparse.mm(A[d], tables[d])
            feats[d] = torch.relu(h @ Wt[d] if Wt is not None else h)
        for d in range(D):
            torch.cuda.synchronize(devs[d])
        phase_ms["exchange"] += (t1 - t0) * 1e3
        phase_ms["compute"] += (time.perf_counter() - t1) * 1e3

    for d in range(D):
        torch.cuda.reset_peak_memory_stats(devs[d])
    for _ in range(WARMUP):
        for _ in range(layers): step()
        for d in range(D): feats[d] = torch.from_numpy(X[locals_g[d]]).to(devs[d])  # reset
    times = []
    for _ in range(TIMED):
        for d in range(D): feats[d] = torch.from_numpy(X[locals_g[d]]).to(devs[d])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(layers): step()
        times.append((time.perf_counter() - t0) * 1e3)
    peak = [round(torch.cuda.max_memory_allocated(devs[d]) / 2**20, 1) for d in range(D)]

    # single-device fp32 reference on cuda:0 (same op order class; certifies the PLACEMENT).
    # At the feasibility frontier (high F) the REFERENCE itself may not fit one GPU -- that IS the
    # frontier datapoint: record ref_oom honestly instead of dying.
    try:
        ref = run_reference(u, v, N, X, layers, W)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(step_ms=round(float(np.median(times)), 2), peak_mb=peak, peak_mb_max=max(peak),
                    halo_rows=int(halo_rows), maxdiff=None, same_result=None, xmode=mode,
                    note="single-GPU reference OOM (the frontier: multi-GPU runs, 1-GPU cannot)")
    out = np.empty((N, F), np.float32)
    for d in range(D):
        out[locals_g[d]] = feats[d].cpu().numpy()
    maxdiff = float(np.max(np.abs(out - ref))) if N else 0.0
    nsteps = max(1, (WARMUP + TIMED) * layers)
    return dict(step_ms=round(float(np.median(times)), 2), peak_mb=peak, peak_mb_max=max(peak),
                halo_rows=int(halo_rows), maxdiff=maxdiff, same_result=bool(maxdiff <= 1e-3),
                xmode=mode,
                exchange_ms=round(phase_ms["exchange"] / nsteps * layers, 2),
                compute_ms=round(phase_ms["compute"] / nsteps * layers, 2))


def run_cell_tensor(src, dst, N, D, F, layers, seed, W=None):
    """TENSOR-SPLIT (NeutronTP-style column parallelism) execution: every GPU holds the FULL
    adjacency (replicated) + F/D feature COLUMNS. Pure aggregation+relu is column-independent ->
    ZERO inter-GPU traffic per layer; the cost is the replicated adjacency (peak memory) and no
    compute scaling (every GPU traverses all E edges). The regime contrast to node-split: wins
    while A fits per-GPU and comm dominates; dies (OOM) when the graph outgrows one GPU's HBM --
    exactly the axis boundary zord's choose_axis prices. Same-result certified like node-split."""
    import torch
    devs = [torch.device(f"cuda:{i}") for i in range(D)]
    u = np.concatenate([src, dst, np.arange(N)]); v = np.concatenate([dst, src, np.arange(N)])
    o = np.argsort(u, kind="stable"); us, vs = u[o], v[o]
    deg = np.bincount(us, minlength=N).astype(np.float32)
    crow = np.zeros(N + 1, np.int64); np.add.at(crow, us + 1, 1); np.cumsum(crow, out=crow)
    vals = (1.0 / np.maximum(deg, 1.0))[us].astype(np.float32)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, F)).astype(np.float32)
    cols = np.array_split(np.arange(F), D)
    A, feats = [], []
    for d in range(D):
        t = devs[d]
        A.append(torch.sparse_csr_tensor(torch.from_numpy(crow).to(t), torch.from_numpy(vs).to(t),
                                         torch.from_numpy(vals).to(t), size=(N, N)))
        feats.append(torch.from_numpy(np.ascontiguousarray(X[:, cols[d]])).to(t))
    for d in range(D):
        torch.cuda.reset_peak_memory_stats(devs[d])

    # With W (Megatron row-parallel): each GPU computes the FULL-width partial (A@X_d)@W[cols_d,:],
    # partials are ALL-REDUCED (naive: summed on dev0, fp32), relu once, columns re-scattered. The
    # all-reduce is the price column-parallelism pays for the W-mix -- the fair-fight term.
    Wd = ([torch.from_numpy(np.ascontiguousarray(W[cols[d], :])).to(devs[d]) for d in range(D)]
          if W is not None else None)

    def sweep():
        if Wd is None:
            for d in range(D):
                feats[d] = torch.relu(torch.sparse.mm(A[d], feats[d]))
        else:
            partials = [torch.sparse.mm(A[d], feats[d]) @ Wd[d] for d in range(D)]
            z = partials[0].to(devs[0])
            for d in range(1, D):
                z = z + partials[d].to(devs[0], non_blocking=False)
            z = torch.relu(z)
            for d in range(D):
                feats[d] = z[:, cols[d][0]:cols[d][-1] + 1].to(devs[d]).contiguous()
        for d in range(D):
            torch.cuda.synchronize(devs[d])

    for _ in range(WARMUP):
        for _ in range(layers): sweep()
        for d in range(D): feats[d] = torch.from_numpy(np.ascontiguousarray(X[:, cols[d]])).to(devs[d])
    times = []
    for _ in range(TIMED):
        for d in range(D): feats[d] = torch.from_numpy(np.ascontiguousarray(X[:, cols[d]])).to(devs[d])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(layers): sweep()
        times.append((time.perf_counter() - t0) * 1e3)
    peak = [round(torch.cuda.max_memory_allocated(devs[d]) / 2**20, 1) for d in range(D)]
    try:
        ref = run_reference(u, v, N, X, layers, W)
    except torch.cuda.OutOfMemoryError:   # frontier: the full-width reference doesn't fit one GPU
        torch.cuda.empty_cache()
        return dict(step_ms=round(float(np.median(times)), 2), peak_mb=peak, peak_mb_max=max(peak),
                    halo_rows=0, maxdiff=None, same_result=None,
                    note="single-GPU reference OOM (tensor-split itself ran)")
    out = np.empty((N, F), np.float32)
    for d in range(D):
        out[:, cols[d]] = feats[d].cpu().numpy()
    maxdiff = float(np.max(np.abs(out - ref))) if N else 0.0
    return dict(step_ms=round(float(np.median(times)), 2), peak_mb=peak, peak_mb_max=max(peak),
                halo_rows=0, maxdiff=maxdiff, same_result=bool(maxdiff <= 1e-3))


def run_cell_train(src, dst, N, part, D, F, layers, seed, steps=10, lr=0.01):
    """END-TO-END TRAINING step on D GPUs: forward (L layers, relu((A@H)@W_l)) -> MSE loss against
    a fixed seeded target -> backward (gradients cross devices through the halo copies) ->
    all-reduce of the per-replica dW -> synchronous SGD. Reports the median full-training-step
    time and a LOSS-TRAJECTORY certificate against single-device training (same seeds/order).
    Autograd-safe halo assembly: the per-device neighbor table is built functionally
    (cat + index_select) so gradients flow back to remote owners."""
    import torch
    devs = [torch.device(f"cuda:{i}") for i in range(D)]
    u = np.concatenate([src, dst, np.arange(N)]); v = np.concatenate([dst, src, np.arange(N)])
    part = np.asarray(part, np.int64)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, F)).astype(np.float32)
    Y = rng.standard_normal((N, F)).astype(np.float32)          # fixed regression target
    W0 = [(rng.standard_normal((F, F)) / np.sqrt(F)).astype(np.float32) for _ in range(layers)]

    locals_g, lidx = [], np.full(N, -1, np.int64)
    for d in range(D):
        g = np.nonzero(part == d)[0]; locals_g.append(g); lidx[g] = np.arange(g.size)

    A, Xd, Yd, plans, perms = [], [], [], [], []
    for d in range(D):
        m = part[u] == d
        gu, gv = u[m], v[m]
        nl = locals_g[d].size
        hg = np.unique(gv[part[gv] != d]); nh = hg.size
        tpos = np.full(N, -1, np.int64); tpos[locals_g[d]] = np.arange(nl)
        tpos[hg] = nl + np.arange(nh)
        ei, ej = lidx[gu], tpos[gv]
        o = np.argsort(ei, kind="stable"); ei, ej = ei[o], ej[o]
        deg = np.bincount(ei, minlength=nl).astype(np.float32)
        crow = np.zeros(nl + 1, np.int64); np.add.at(crow, ei + 1, 1); np.cumsum(crow, out=crow)
        vals = (1.0 / np.maximum(deg, 1.0))[ei].astype(np.float32)
        t = devs[d]
        A.append(torch.sparse_csr_tensor(torch.from_numpy(crow).to(t), torch.from_numpy(ej).to(t),
                                         torch.from_numpy(vals).to(t), size=(nl, nl + nh)))
        Xd.append(torch.from_numpy(X[locals_g[d]]).to(t))       # inputs: fixed, no grad
        Yd.append(torch.from_numpy(Y[locals_g[d]]).to(t))
        # halo assembly plan: pieces per owner (in ascending owner order), then a permutation
        # mapping [local | piece_0 | piece_1 | ...] -> table position order
        plan, cat_pos = [], [np.arange(nl)]
        for o_ in range(D):
            if o_ == d: continue
            sel = hg[part[hg] == o_]
            if sel.size:
                plan.append((o_, torch.from_numpy(lidx[sel]).to(devs[o_])))
                cat_pos.append(tpos[sel])
        inv = np.argsort(np.concatenate(cat_pos), kind="stable")
        plans.append(plan); perms.append(torch.from_numpy(inv).to(t))

    Wt = [[torch.tensor(W0[l], device=devs[d], requires_grad=True) for l in range(layers)]
          for d in range(D)]

    def train_step():
        for d in range(D):
            for w in Wt[d]:
                if w.grad is not None: w.grad = None
        H = [Xd[d] for d in range(D)]
        for l in range(layers):
            tables = []
            for d in range(D):
                pieces = [H[d]] + [H[o_].index_select(0, si).to(devs[d]) for o_, si in plans[d]]
                tables.append(torch.cat(pieces, 0).index_select(0, perms[d]))
            H = [torch.relu(torch.sparse.mm(A[d], tables[d]) @ Wt[d][l]) for d in range(D)]
        loss = sum(((H[d] - Yd[d]) ** 2).sum().to(devs[0]) for d in range(D)) / (N * F)
        loss.backward()
        with torch.no_grad():                                    # all-reduce dW; identical SGD
            for l in range(layers):
                gsum = sum(Wt[d][l].grad.to(devs[0]) for d in range(D))
                for d in range(D):
                    Wt[d][l] -= lr * gsum.to(devs[d])
        for d in range(D):
            torch.cuda.synchronize(devs[d])
        return float(loss.detach().cpu())

    for d in range(D):
        torch.cuda.reset_peak_memory_stats(devs[d])
    losses, times = [], []
    for it in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses.append(train_step())
        times.append((time.perf_counter() - t0) * 1e3)
    peak = [round(torch.cuda.max_memory_allocated(devs[d]) / 2**20, 1) for d in range(D)]

    # single-device reference training (same init/order) for the loss-trajectory certificate
    try:
        t = devs[0]
        o = np.argsort(u, kind="stable"); us, vs = u[o], v[o]
        degf = np.bincount(us, minlength=N).astype(np.float32)
        crow = np.zeros(N + 1, np.int64); np.add.at(crow, us + 1, 1); np.cumsum(crow, out=crow)
        vals = (1.0 / np.maximum(degf, 1.0))[us].astype(np.float32)
        Af = torch.sparse_csr_tensor(torch.from_numpy(crow).to(t), torch.from_numpy(vs).to(t),
                                     torch.from_numpy(vals).to(t), size=(N, N))
        Xf = torch.from_numpy(X).to(t); Yf = torch.from_numpy(Y).to(t)
        Wf = [torch.tensor(W0[l], device=t, requires_grad=True) for l in range(layers)]
        ref_losses = []
        for it in range(steps):
            for w in Wf:
                if w.grad is not None: w.grad = None
            H = Xf
            for l in range(layers):
                H = torch.relu(torch.sparse.mm(Af, H) @ Wf[l])
            loss = ((H - Yf) ** 2).sum() / (N * F)
            loss.backward()
            with torch.no_grad():
                for w in Wf: w -= lr * w.grad
            ref_losses.append(float(loss.detach().cpu()))
        del Af, Xf, Yf, Wf; torch.cuda.empty_cache()
        loss_dev = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(losses, ref_losses))
        same = bool(loss_dev <= 1e-3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        loss_dev, same = None, None
    return dict(step_ms=round(float(np.median(times)), 2), peak_mb=peak, peak_mb_max=max(peak),
                halo_rows=0, maxdiff=loss_dev, same_result=same,
                note=f"TRAIN fwd+bwd+SGD, {steps} steps; maxdiff = max relative loss deviation")


def run_reference(u, v, N, X, layers, W=None):
    import torch
    t = torch.device("cuda:0")
    o = np.argsort(u, kind="stable"); us, vs = u[o], v[o]
    deg = np.bincount(us, minlength=N).astype(np.float32)
    crow = np.zeros(N + 1, np.int64); np.add.at(crow, us + 1, 1); np.cumsum(crow, out=crow)
    vals = (1.0 / np.maximum(deg, 1.0))[us].astype(np.float32)
    A = torch.sparse_csr_tensor(torch.from_numpy(crow).to(t), torch.from_numpy(vs).to(t),
                                torch.from_numpy(vals).to(t), size=(N, N))
    H = torch.from_numpy(X).to(t)
    Wt = torch.from_numpy(W).to(t) if W is not None else None
    for _ in range(layers):
        h = torch.sparse.mm(A, H)
        H = torch.relu(h @ Wt if Wt is not None else h)
    r = H.cpu().numpy(); del H, A; import torch as _t; _t.cuda.empty_cache()
    return r


def worker(args):
    import torch, traceback
    try:
        src, dst, N = load_cached(args.dataset)
    except Exception as e:
        traceback.print_exc()
        print("ZORDJSON " + json.dumps(dict(dataset=args.dataset, status="FAIL",
                                            note=f"load: {type(e).__name__}: {e}"[:140])), flush=True)
        return
    fv = build_fv(N, args.feat_dim)
    W = None
    if args.wmix:   # per-layer weight mix, scaled to keep activations bounded through L layers
        W = (np.random.default_rng(7).standard_normal((args.feat_dim, args.feat_dim))
             / np.sqrt(args.feat_dim)).astype(np.float32)
    avail = torch.cuda.device_count()
    gname = torch.cuda.get_device_name(0) if avail else "none"
    for ng in [int(x) for x in args.ngpus.split(",")]:
        if ng > avail:
            print("ZORDJSON " + json.dumps(dict(dataset=args.dataset, ngpu=ng, status="SKIP",
                                                note=f"only {avail} GPUs"))); continue
        methods = [m for m in (["single"] if ng == 1 else METHODS)
                   if not args.methods or m in args.methods.split(",")]
        for method in methods:
            rec = dict(dataset=args.dataset, gpu=gname, ngpu=ng, method=method,
                       feat_dim=args.feat_dim, layers=args.layers, N=int(N), M=int(src.size))
            try:
                rec["wmix"] = bool(W is not None)
                if method == "tensor-split":
                    rec["part_s"] = 0.0     # column split needs no graph partition
                    rec.update(run_cell_tensor(src, dst, N, ng, args.feat_dim, args.layers, seed=0, W=W))
                elif args.train:
                    t0 = time.time()
                    part = make_partition("hash" if method == "single" else method, src, dst, N, ng, fv)
                    rec["part_s"] = round(time.time() - t0, 2)
                    rec.update(run_cell_train(src, dst, N, part, ng, args.feat_dim, args.layers, seed=0))
                else:
                    t0 = time.time()
                    part = make_partition("hash" if method == "single" else method, src, dst, N, ng, fv)
                    rec["part_s"] = round(time.time() - t0, 2)
                    rec.update(run_cell(src, dst, N, part, ng, args.feat_dim, args.layers, seed=0, W=W))
                rec["status"] = "OK"
            except torch.cuda.OutOfMemoryError as e:
                rec["status"] = "OOM"; rec["note"] = str(e)[:100]; torch.cuda.empty_cache()
            except Exception as e:
                rec["status"] = "FAIL"; rec["note"] = f"{type(e).__name__}: {e}"[:140]
            print("ZORDJSON " + json.dumps(rec), flush=True)


def orchestrator(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["dataset", "gpu", "ngpu", "method", "feat_dim", "N", "M", "part_s", "step_ms",
            "peak_mb_max", "halo_rows", "exchange_ms", "compute_ms", "maxdiff", "same_result",
            "xmode", "status", "note"]
    with open(args.out + ".csv", "w") as f:
        f.write(",".join(cols) + "\n")
    for name, tmo in CAMPAIGN:
        if args.datasets and name not in args.datasets.split(","):
            continue
        print(f"[orch] launching {name} (timeout {tmo}s)", flush=True)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker", "--dataset", name,
               "--ngpus", args.ngpus, "--feat-dim", str(args.feat_dim), "--layers", str(args.layers)]
        if args.methods:
            cmd += ["--methods", args.methods]
        if args.wmix:
            cmd += ["--wmix"]
        if args.train:
            cmd += ["--train"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=tmo)
            outtxt = r.stdout or ""
            if r.returncode != 0:
                err = (r.stderr or "").strip().splitlines()
                print(f"[orch] {name} worker rc={r.returncode} stderr-tail: "
                      + " | ".join(err[-3:]), flush=True)
                if "ZORDJSON" not in outtxt:
                    outtxt += "\nZORDJSON " + json.dumps(dict(dataset=name, status="FAIL",
                              note=(err or [""])[-1][:140]))
        except subprocess.TimeoutExpired as e:
            outtxt = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            outtxt += "\nZORDJSON " + json.dumps(dict(dataset=name, status="TIMEOUT", note=f">{tmo}s"))
        for line in outtxt.splitlines():
            if not line.startswith("ZORDJSON "):
                continue
            rec = json.loads(line[len("ZORDJSON "):])
            with open(args.out + ".jsonl", "a") as jf:
                jf.write(json.dumps(rec) + "\n")
            with open(args.out + ".csv", "a") as cf:
                cf.write(",".join(str(rec.get(c, "")) for c in cols) + "\n")
            print({k: rec.get(k) for k in ("dataset", "ngpu", "method", "step_ms",
                                           "peak_mb_max", "same_result", "status")}, flush=True)
    print(f"[gpu-e2e] done -> {args.out}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--datasets", default="", help="restrict orchestrator")
    ap.add_argument("--methods", default="", help="restrict methods (comma list)")
    ap.add_argument("--wmix", action="store_true",
                    help="apply a per-layer FxF weight mix (real-GNN form; the fair fight for "
                         "tensor-split, which must then all-reduce)")
    ap.add_argument("--train", action="store_true",
                    help="measure the FULL training step (forward + MSE loss + backward + "
                         "all-reduced SGD) with a loss-trajectory certificate vs single device")
    ap.add_argument("--ngpus", default="1,4,8")
    ap.add_argument("--feat-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", default="results/gpu_e2e")
    args = ap.parse_args()
    worker(args) if args.worker else orchestrator(args)


if __name__ == "__main__":
    main()
