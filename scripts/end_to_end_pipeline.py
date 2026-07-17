#!/usr/bin/env python
"""END-TO-END (D33) MULTI-GPU WALL-CLOCK BENCHMARK -- the ONLY metric that matters.

zord is an end-to-end FRONT/MIDDLE/BACK pipeline; SINGLE-GPU local optimization is NOT the point.
This script measures the TOTAL multi-GPU wall-clock from data-on-disk to a trained model, split into
three stages timed SEPARATELY, for TWO arrange policies, so the central question can be answered with
real numbers: does zord's smarter (more expensive) arrange pay for itself by making the BACK stage
faster, and at WHICH epoch count does it break even?

  FRONT (load)    : load the temporal graph (zord.datasets.load) + sort_by_time + bucket into discrete
                    snapshots (TemporalGraph.to_snapshots) + materialize the global undirected edge
                    list + per-node degree. The data-on-disk -> in-memory-graph cost. Same for both
                    policies (the assignment has not been chosen yet), timed once and reused.
  MIDDLE (arrange): compute the vertex->device assignment. TWO policies compared:
                      zord     = work-balanced hetero-matched. Ranks vertices by DENSITY using the C++
                                 kernel (build/graph_algos degree/kcore ranking), measures each rank's
                                 real HBM aggregation bandwidth, and solves contiguous boundaries over
                                 the density order so the §17-CORRECTED incident-edge work model
                                 (per-rank local_edges / bw_r) is BALANCED -> the densest core lands on
                                 the strongest GPU and per-rank aggregation TIME is equalized.
                      baseline = hash/even: equal-node contiguous shards (or random) over the natural
                                 order. Near-zero arrange cost -- nothing to amortize, but a lumpy cut /
                                 straggler-heavy BACK stage.
                    This is the cost zord must AMORTIZE.
  BACK (train)    : build each rank's local CSR + feature shard UNDER the chosen assignment, then run N
                    epochs of the distributed 2-layer temporal-GNN aggregation step (full forward: 2
                    SpMM gathers over local+remote neighbors; cross-rank boundary exchange of the cut
                    feature rows via isend/irecv; a tiny backward/opt step). Per-epoch + total time.

SAME numerical result for both policies -- only the vertex->device assignment differs, which changes
per-rank edge counts -> per-rank aggregation time -> the makespan of every BACK epoch. PROCESS metric
only (TIME / feasibility), never accuracy.

KEY OUTPUTS (rank 0):
  * per-policy front_s, middle_s, back_s (per-epoch x N), TOTAL = front+middle+back, and the % breakdown.
  * does zord's TOTAL beat baseline's TOTAL at the requested --epochs?
  * the AMORTIZATION CURVE total-vs-#epochs for both, and the BREAKEVEN epoch where zord's cheaper
    per-epoch BACK has repaid its pricier MIDDLE arrange:  e* = (middle_zord - middle_base) /
    (per_epoch_base - per_epoch_zord)  (front is shared, so it cancels).

Conventions mirror scripts/multi_gpu_train.py (NCCL setup, density/bw policy, boundary exchange, the
§17-corrected incident-edge work model), scripts/loss_breakdown.py (per-phase CUDA timing), and
scripts/reorder_speedup.py (C++ graph_algos ordering call + zord.datasets.load API). Hardware-agnostic:
each rank MEASURES its own GPU's aggregation bandwidth; no NVLink/tier is hardcoded.

  srun --ntasks=P python scripts/end_to_end_pipeline.py --nodes 8000000 --edges 100000000 --feat 128 \
       --epochs 1,10,100,1000
  srun --ntasks=P python scripts/end_to_end_pipeline.py --dataset askubuntu --feat 256 --epochs 1,10,100,1000
  srun --ntasks=P python scripts/end_to_end_pipeline.py --nodes 4000000 --edges 60000000 --policy zord
"""
import argparse
import os
import struct
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist

N_GATHERS = 2                                   # 2-layer aggregation = 2 SpMM gathers
SNAPSHOTS = 32                                  # discrete-time snapshots the FRONT buckets into
# C++ graph_algos binary (degree/kcore ranking for the zord arrange); mirrors reorder_speedup.py
BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")
POLICIES = ["zord", "baseline"]                 # zord = hetero-matched; baseline = hash/even


# --------------------------------------------------------------------------- NCCL setup (mirrors multi_gpu_train.py)
def env_int(*keys, default=0):
    for k in keys:
        if k in os.environ:
            return int(os.environ[k])
    return default


def setup():
    rank = env_int("SLURM_PROCID", "RANK")
    world = env_int("SLURM_NTASKS", "WORLD_SIZE", default=1)
    local = env_int("SLURM_LOCALID", "LOCAL_RANK")
    if "MASTER_ADDR" not in os.environ:
        nl = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        if "[" in nl:
            base, rest = nl.split("[", 1)
            nl = f"{base}{rest.split('-', 1)[0].rstrip(']')}"
        os.environ["MASTER_ADDR"] = nl
    os.environ.setdefault("MASTER_PORT", "29610")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world, local


# --------------------------------------------------------------------------- timing helpers
def timed_cuda(fn, reps=20, warmup=5):
    """Mean wall-time (s) of fn over reps, after warmup, CUDA-synced. NO barrier inside the loop so
    each rank's time reflects its OWN local work (the per-rank busy time we compare across policies)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def measure_hbm_bw(dev, mb=256, reps=30):
    """Microbenchmark THIS rank's achieved memory bandwidth (a streaming copy ~ what bounds the
    memory-bound aggregation). GB/s. Lets the zord arrange react to the REAL devices in the job; no
    tier/link is assumed -- heterogeneous GPUs simply measure different bw here."""
    n = (mb * 1024 * 1024) // 4
    a = torch.empty(n, dtype=torch.float32, device=dev)
    b = torch.empty(n, dtype=torch.float32, device=dev)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        b.copy_(a)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    return (2 * n * 4) / dt / 1e9


# --------------------------------------------------------------------------- graph (deterministic, controllable locality)
def gen_graph(N, M, C, intra, seed=0):
    """Synthetic temporal graph with C communities; `intra` fraction of edges stay inside a node's
    community (so a community-aligned partition has a small cut). Returns int64 src/dst/t."""
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, C, size=N).astype(np.int64)
    order = np.argsort(comm, kind="stable")
    bounds = np.searchsorted(comm[order], np.arange(C + 1))
    m_in = int(M * intra)
    u = rng.integers(0, N, size=m_in)
    cu = comm[u]
    lo = bounds[cu].astype(np.int64)
    hi = bounds[cu + 1].astype(np.int64)
    pick = lo + (rng.random(m_in) * np.maximum(1, hi - lo)).astype(np.int64)
    v = order[np.minimum(pick, N - 1)]
    u2 = rng.integers(0, N, size=M - m_in)
    v2 = rng.integers(0, N, size=M - m_in)
    src = np.concatenate([u, u2]).astype(np.int64)
    dst = np.concatenate([v, v2]).astype(np.int64)
    t = rng.integers(0, max(1, M), size=M).astype(np.int64)   # logical timestamps -> snapshot buckets
    return src, dst, t


def node_degree(src, dst, N):
    return (np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)).astype(np.int64)


# --------------------------------------------------------------------------- C++ graph_algos ranking (zord arrange)
def write_edges(path, N, src, dst):
    """Edge list in the build/graph_algos binary format (mirrors reorder_speedup.py)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<qq", N, src.size))
        inter = np.empty(2 * src.size, dtype=np.int32)
        inter[0::2] = src.astype(np.int32)
        inter[1::2] = dst.astype(np.int32)
        inter.tofile(f)


def cpp_density_rank(src, dst, N, mode, seed):
    """Run the C++ kernel (build/graph_algos) to produce a DENSITY ranking newid[v] (0 = densest). If
    the binary is unavailable, fall back to a numpy degree ranking so the pipeline still runs end-to-end
    (the arrange COST then reflects the numpy path, which is reported as such)."""
    edges_path = f"/tmp/zord_e2e_edges_{seed}.bin"
    out_path = f"/tmp/zord_e2e_perm_{mode}_{seed}.bin"
    try:
        write_edges(edges_path, N, src, dst)
        r = subprocess.run([BIN, edges_path, mode, out_path], capture_output=True, text=True)
        if r.returncode == 0:
            with open(out_path, "rb") as f:
                n = struct.unpack("<q", f.read(8))[0]
                newid = np.fromfile(f, dtype=np.int32, count=n).astype(np.int64)
            return newid, "cpp"
    except Exception:
        pass
    # ---- numpy fallback: degree ranking (0 = densest) ----
    deg = node_degree(src, dst, N)
    order = np.argsort(-deg, kind="stable")
    newid = np.empty(N, dtype=np.int64)
    newid[order] = np.arange(N, dtype=np.int64)
    return newid, "numpy-fallback"


def solve_balanced_bounds(deg_cum, bw, N):
    """hetero-matched: contiguous boundaries over the density-sorted order so PREDICTED per-rank agg
    TIME is equalized (the §17-corrected incident-edge work model). time_r ~ local_edges_r / bw_r; give
    rank r an edge budget ~ bw_r, translate that cumulative-degree budget into a node boundary. bw is
    ordered strongest-first (== density order, so the densest core lands on the strongest GPU)."""
    D = len(bw)
    share = np.asarray(bw, dtype=np.float64) / np.sum(bw)
    target = share * deg_cum[-1]
    bounds = [0]
    acc = 0.0
    for k in range(D - 1):
        acc += target[k]
        nb = int(np.searchsorted(deg_cum, acc, side="left"))
        nb = max(bounds[-1] + 1, min(nb, N))
        bounds.append(nb)
    bounds.append(N)
    return np.array(bounds, dtype=np.int64)


# --------------------------------------------------------------------------- MIDDLE: the arrange (vertex -> device)
def arrange(policy, N, world, src, dst, deg, bw_strong_first, seed):
    """Compute the vertex->device assignment for `policy`. Returns (part[v] -> owning rank, info str).
    THIS is the MIDDLE stage being timed.
      zord     : C++ density ranking + §17-corrected balanced bounds over the density order.
      baseline : hash/even -- equal-node contiguous shards over the NATURAL node order (near-zero cost).
    """
    if policy == "zord":
        rank_of, how = cpp_density_rank(src, dst, N, "degree", seed)   # node id -> density rank (0=densest)
        deg_by_rank = np.empty(N, dtype=np.float64)
        deg_by_rank[rank_of] = deg.astype(np.float64)                  # degree indexed by density rank
        deg_cum = np.cumsum(deg_by_rank)
        bounds = solve_balanced_bounds(deg_cum, bw_strong_first, N)
        bounds = np.minimum(np.maximum.accumulate(bounds), N).astype(np.int64)
        part = (np.searchsorted(bounds, rank_of, side="right") - 1).clip(0, world - 1).astype(np.int64)
        return part, f"hetero-matched({how}, §17-balanced)"
    elif policy == "baseline":
        # hash/even: equal-node contiguous shards over the natural node id order (degree-blind).
        bounds = np.linspace(0, N, world + 1).astype(np.int64)
        part = (np.searchsorted(bounds, np.arange(N), side="right") - 1).clip(0, world - 1).astype(np.int64)
        return part, "hash/even(contiguous)"
    raise ValueError(f"unknown policy {policy!r}")


# --------------------------------------------------------------------------- BACK: build shard + one distributed epoch
def _feat_rows(global_ids, F, seed, dev):
    """Deterministic feature rows for given global ids (same on every rank that touches them), so the
    gathered remote features are consistent -- SAME numerical result regardless of policy."""
    g = global_ids.astype(np.int64)
    base = (g.astype(np.float64) * 0.0009765625) % 1.0
    cols = np.arange(F, dtype=np.float64)
    M = np.sin(base[:, None] * (cols[None, :] + 1.0) + seed)
    return torch.from_numpy(M.astype(np.float32)).to(dev)


def _exchange_counts(recv_counts, world, dev):
    """All-to-all the integer request counts so each rank learns how many rows others want FROM it."""
    recv_t = torch.from_numpy(recv_counts.astype(np.int64)).to(dev)
    send_t = torch.empty_like(recv_t)
    dist.all_to_all_single(send_t, recv_t)
    return send_t.cpu().numpy()


def build_shard(part, rank, world, dev, N, F, src, dst, seed):
    """Materialize THIS rank's local CSR + feature shard under assignment `part`, and assemble the
    per-epoch step closure (forward 2-layer aggregation + boundary exchange + tiny backward/opt).
    Returns (step_fn, build_seconds, stats) where stats describes this rank's shard."""
    t0 = time.time()
    my_nodes = np.nonzero(part == rank)[0]
    nl = int(my_nodes.size)
    g2l = np.full(N, -1, dtype=np.int64)
    g2l[my_nodes] = np.arange(nl, dtype=np.int64)

    # undirected: an edge contributes to this rank's local aggregation if its source is local.
    es = np.concatenate([src, dst])
    ed = np.concatenate([dst, src])
    mine = part[es] == rank
    e_src = es[mine]
    e_dst = ed[mine]
    rows = g2l[e_src]

    nbr_part = part[e_dst]
    is_remote = nbr_part != rank
    remote_ids = np.unique(e_dst[is_remote])
    n_remote = int(remote_ids.size)

    # feature buffer: [ local shard rows (nl) | remote fetched rows (n_remote) ]
    rid2slot = np.full(N, -1, dtype=np.int64)
    rid2slot[remote_ids] = np.arange(n_remote, dtype=np.int64) + nl
    cols = np.where(is_remote, rid2slot[e_dst], g2l[e_dst])
    nbuf = nl + n_remote

    gen = torch.Generator().manual_seed(1234 + seed)
    Xbuf = torch.empty(max(1, nbuf), F, device=dev)
    if nl:
        Xbuf[:nl] = _feat_rows(my_nodes, F, seed, dev)
    if n_remote:
        Xbuf[nl:nbuf] = _feat_rows(remote_ids, F, seed, dev)

    A = torch.sparse_coo_tensor(
        torch.stack([torch.from_numpy(rows).to(dev), torch.from_numpy(cols).to(dev)]),
        torch.ones(rows.size, device=dev),
        (max(1, nl), max(1, nbuf)),
    ).coalesce()
    W1 = (torch.randn(F, F, generator=gen, device="cpu") / F ** 0.5).to(dev).requires_grad_(True)
    W2 = (torch.randn(F, F, generator=gen, device="cpu") / F ** 0.5).to(dev).requires_grad_(True)
    opt = torch.optim.SGD([W1, W2], lr=1e-3)

    # boundary-exchange plan (ship the n_remote feature rows this rank needs = the realistic cut volume)
    remote_owner = part[remote_ids] if n_remote else np.empty(0, dtype=np.int64)
    recv_counts = np.bincount(remote_owner, minlength=world).astype(np.int64)
    send_counts = _exchange_counts(recv_counts, world, dev)
    recv_bufs = [torch.empty(max(1, int(recv_counts[p])), F, device=dev) for p in range(world)]
    send_bufs = [torch.empty(max(1, int(send_counts[p])), F, device=dev) for p in range(world)]

    def boundary():
        reqs = []
        for p in range(world):
            if p == rank:
                continue
            if recv_counts[p] > 0:
                reqs.append(dist.irecv(recv_bufs[p], src=p))
            if send_counts[p] > 0:
                reqs.append(dist.isend(send_bufs[p], dst=p))
        for r in reqs:
            r.wait()

    def step():
        """One distributed BACK epoch: boundary exchange of the cut rows, then forward 2-layer
        aggregation, then a tiny backward + opt step. Same numerical work under any assignment."""
        if world > 1:
            boundary()
        h = torch.relu(torch.sparse.mm(A, Xbuf) @ W1)          # layer 1 over local+remote neighbors
        h2buf = torch.zeros(max(1, nbuf), F, device=dev)
        h2buf[:nl] = h
        out = torch.relu(torch.sparse.mm(A, h2buf) @ W2)       # layer 2
        loss = out.float().pow(2).mean()                       # tiny scalar so backward/opt is cheap
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    build_s = time.time() - t0
    cut = int(is_remote.sum())
    local_e = int((~is_remote).sum())
    stats = (nl, local_e, cut, n_remote)
    return step, build_s, stats


# --------------------------------------------------------------------------- breakeven / amortization
def breakeven_epoch(middle_z, middle_b, per_z, per_b):
    """Epoch e* where zord's TOTAL crosses below baseline's. front is shared (cancels), so:
       middle_z + e*per_z  =  middle_b + e*per_b  ->  e* = (middle_z - middle_b) / (per_b - per_z).
    Returns (e_star or None, reason)."""
    dper = per_b - per_z                              # per-epoch BACK saving of zord (>0 if zord faster)
    dmid = middle_z - middle_b                        # extra arrange cost zord paid (>0 normally)
    if dper <= 0:
        return None, "zord BACK is not faster per epoch -> never amortizes (no breakeven)"
    if dmid <= 0:
        return 0, "zord arrange is no costlier AND faster per epoch -> wins from epoch 0"
    return dmid / dper, "amortizes after this many epochs"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=8_000_000)
    ap.add_argument("--edges", type=int, default=100_000_000)
    ap.add_argument("--comms", type=int, default=4000)
    ap.add_argument("--intra", type=float, default=0.9,
                    help="frac of edges kept inside a node's community (intra-shard locality)")
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--dataset", default="", help="real temporal graph name (else synthetic)")
    ap.add_argument("--epochs", default="1,10,100,1000",
                    help="comma list of BACK epoch counts to report the amortization curve at; the "
                         "LAST value is the count actually executed for the per-epoch measurement")
    ap.add_argument("--policy", default="all", choices=POLICIES + ["all"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rank, world, local = setup()
    dev = f"cuda:{local}"
    F = a.feat
    epochs_list = sorted({int(x) for x in a.epochs.split(",") if x.strip()})
    n_run = max(epochs_list) if epochs_list else 100         # epochs actually executed for per-epoch timing
    n_run = max(1, min(n_run, 50))                           # cap executed epochs; extrapolate the curve
    policies = POLICIES if a.policy == "all" else [a.policy]

    # ---- measure each rank's real aggregation bandwidth (hardware-agnostic) ----
    my_bw = measure_hbm_bw(dev)
    bw_dev = torch.tensor([my_bw], device=dev, dtype=torch.float64)
    bw_all = [torch.empty(1, device=dev, dtype=torch.float64) for _ in range(world)]
    dist.all_gather(bw_all, bw_dev)
    bw_by_rank = np.array([b.item() for b in bw_all], dtype=np.float64)
    bw_strong_first = bw_by_rank                             # ranks assumed launched strong->weak

    # ============================ FRONT (load) ============================
    # load/generate the temporal graph + sort_by_time + bucket into snapshots + build the global
    # undirected edge list + per-node degree. Shared by both policies (assignment not chosen yet).
    dist.barrier()
    t_front0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes
        src = np.ascontiguousarray(g.src, dtype=np.int64)
        dst = np.ascontiguousarray(g.dst, dtype=np.int64)
        M = src.size
        snaps = g.to_snapshots(num_snapshots=SNAPSHOTS)
        gname = g.name
    else:
        N, M = a.nodes, a.edges
        src, dst, t = gen_graph(N, M, a.comms, a.intra, seed=a.seed)
        g = None
        from zord.datasets import TemporalGraph
        tg = TemporalGraph(src=src, dst=dst, t=t, num_nodes=N,
                           name=f"synthetic(comms={a.comms},intra={a.intra})").sort_by_time()
        src, dst = tg.src, tg.dst                            # snapshot-sorted edge order
        snaps = tg.to_snapshots(num_snapshots=SNAPSHOTS)
        gname = tg.name
    deg = node_degree(src, dst, N)
    torch.cuda.synchronize()
    front_s = time.time() - t_front0
    # share one FRONT number (max over ranks -> the wall-clock the slowest rank imposes)
    front_t = torch.tensor([front_s], device=dev, dtype=torch.float64)
    dist.all_reduce(front_t, op=dist.ReduceOp.MAX)
    front_s = float(front_t.item())

    if rank == 0:
        print(f"E2E gpu='{torch.cuda.get_device_name(0)}' world={world} dataset={gname} "
              f"N={N} M={M} F={F} snapshots={len(snaps)} seed={a.seed}")
        print("  measured agg bw by rank: " +
              " ".join(f"r{r}={bw_by_rank[r]:.0f}GB/s" for r in range(world)) +
              ("  [heterogeneous]" if bw_by_rank.max() / max(bw_by_rank.min(), 1e-9) > 1.15
               else "  [homogeneous]"))
        print(f"  FRONT (load+snapshot+edgelist+degree) = {front_s:.3f}s  [shared by both policies]\n")

    # ============================ per-policy MIDDLE + BACK ============================
    summary = {}                                             # policy -> dict of timings
    for pol in policies:
        # ---- MIDDLE (arrange): time the vertex->device assignment ----
        dist.barrier()
        t_mid0 = time.time()
        part, info = arrange(pol, N, world, src, dst, deg, bw_strong_first, a.seed)
        middle_s = time.time() - t_mid0
        mid_t = torch.tensor([middle_s], device=dev, dtype=torch.float64)
        dist.all_reduce(mid_t, op=dist.ReduceOp.MAX)
        middle_s = float(mid_t.item())

        # ---- BACK (train): build shard under this assignment, run n_run epochs ----
        step, build_s, st = build_shard(part, rank, world, dev, N, F, src, dst, a.seed)
        per_epoch_s = timed_cuda(step, reps=n_run, warmup=5)  # mean per-epoch, this rank
        # the per-epoch makespan = the SLOWEST rank (the barrier releases when it finishes)
        pe_t = torch.tensor([per_epoch_s], device=dev, dtype=torch.float64)
        dist.all_reduce(pe_t, op=dist.ReduceOp.MAX)
        per_epoch_s = float(pe_t.item())

        # gather per-rank shard stats for the report
        st_t = torch.tensor([float(x) for x in st] + [build_s, bw_by_rank[rank]],
                            device=dev, dtype=torch.float64)
        gathered = [torch.empty_like(st_t) for _ in range(world)]
        dist.all_gather(gathered, st_t)
        dist.barrier()
        if rank == 0:
            summary[pol] = {
                "info": info, "middle_s": middle_s, "per_epoch_s": per_epoch_s,
                "rows": [t.cpu().numpy() for t in gathered],
            }

    # ============================ REPORT (rank 0) ============================
    if rank == 0:
        print(f"  BACK measured over {n_run} executed epochs (per-epoch = makespan over ranks); "
              f"curve extrapolated to {epochs_list}\n")
        for pol in policies:
            s = summary[pol]
            print(f"  [{pol}]  arrange = {s['info']}")
            for r, row in enumerate(s["rows"]):
                nl, le, cut, nr, b_s, bw = row
                print(f"      rank{r}: nodes={int(nl):>11,d} local_edges={int(le):>12,d} "
                      f"cut={int(cut):>11,d} remote_nbrs={int(nr):>9,d} bw={bw:5.0f}GB/s "
                      f"shard_build={b_s:6.3f}s")
            print(f"      MIDDLE(arrange) = {s['middle_s']:.4f}s   "
                  f"BACK per-epoch(makespan) = {s['per_epoch_s']*1e3:.2f}ms")

        # ---- per-policy TOTAL + % breakdown at the requested epoch counts ----
        print("\n  ==== STAGE BREAKDOWN + TOTAL (front shared; total = front + middle + N*back) ====")
        hdr = "    {:<9} {:>8} {:>9} {:>10} {:>10} {:>26}".format(
            "policy", "epochs", "front_s", "middle_s", "back_s", "TOTAL_s [front/mid/back %]")
        print(hdr)
        for pol in policies:
            s = summary[pol]
            for e in epochs_list:
                back = e * s["per_epoch_s"]
                total = front_s + s["middle_s"] + back
                fp, mp, bp = (100 * front_s / total, 100 * s["middle_s"] / total, 100 * back / total)
                print("    {:<9} {:>8d} {:>8.3f} {:>9.4f} {:>10.3f} {:>10.3f}  [{:4.1f}/{:4.1f}/{:4.1f}]".format(
                    pol, e, front_s, s["middle_s"], back, total, fp, mp, bp))

        # ---- KEY OUTPUT 1: does zord's TOTAL beat baseline's at each epoch count? ----
        if "zord" in summary and "baseline" in summary:
            z, b = summary["zord"], summary["baseline"]
            print("\n  ==== KEY OUTPUT (1): zord TOTAL vs baseline TOTAL ====")
            for e in epochs_list:
                tz = front_s + z["middle_s"] + e * z["per_epoch_s"]
                tb = front_s + b["middle_s"] + e * b["per_epoch_s"]
                tag = "zord WINS" if tz < tb else "baseline wins"
                print(f"    epochs={e:>6d}: zord_total={tz:9.3f}s  baseline_total={tb:9.3f}s  "
                      f"ratio(base/zord)={tb/tz:5.2f}x  -> {tag}")

            # ---- KEY OUTPUT 2: amortization curve breakeven epoch ----
            print("\n  ==== KEY OUTPUT (2): AMORTIZATION -- breakeven epoch ====")
            print(f"    per-epoch BACK: zord={z['per_epoch_s']*1e3:.2f}ms  baseline={b['per_epoch_s']*1e3:.2f}ms"
                  f"  (zord saves {(b['per_epoch_s']-z['per_epoch_s'])*1e3:+.2f}ms/epoch)")
            print(f"    arrange cost  : zord={z['middle_s']:.4f}s  baseline={b['middle_s']:.4f}s"
                  f"  (zord pays {z['middle_s']-b['middle_s']:+.4f}s extra)")
            e_star, why = breakeven_epoch(z["middle_s"], b["middle_s"],
                                          z["per_epoch_s"], b["per_epoch_s"])
            if e_star is None:
                print(f"    breakeven: NONE -- {why}")
            else:
                print(f"    breakeven epoch e* = {e_star:.1f}  ({why}); beyond e* zord's smarter "
                      f"arrange has paid for itself and every further epoch widens its lead.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
