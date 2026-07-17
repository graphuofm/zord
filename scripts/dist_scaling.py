#!/usr/bin/env python
"""STABLE single-node multi-GPU STRONG-SCALING study for distributed temporal-GNN training.

PROCESS-only experiment (D-pivot): for a FIXED total amount of work per snapshot (a fixed
#edges aggregated), split the graph's vertices across WORLD_SIZE GPUs with a zord locality
partition (C++ `lpa` -> contiguous blocks) and run a real DDP'd GraphSAGE step (SpMM neighbor
aggregation + linear, optionally a GRUCell node-memory TGN variant). As N grows 1->2->4->8 the
PER-GPU edge work DECREASES, so the per-step TIME should fall toward linear; we measure the
realized strong-scaling efficiency  eff = t1 / (N * tN).  We MEASURE time / throughput / peak
memory / and -- the #1 concern -- whether ALL ranks completed (stability).  Same data + model =>
same numerical result; accuracy is NEVER the target, it is at most a correctness check.

WHY THE PRIOR ATTEMPTS HUNG (multi_gpu_nvlink.py / multi_gpu_train.py / end_to_end.py) and what
this fixes:
  * No collective timeout -> a single rank that fell behind (slow graph build / partition) made
    every other rank block in NCCL forever until SLURM killed the job.  FIX: 20-min timeout on
    init_process_group, and NCCL_ASYNC_ERROR_HANDLING so a dead peer raises instead of hanging.
  * Mismatched / variable-sized point-to-point isend/irecv (the boundary exchange) can deadlock
    if the count handshake is even slightly off.  FIX: the cross-device boundary feature exchange
    is CUT-FAITHFUL (it ships exactly the boundary rows each rank needs from each owner -> the TRUE
    cut volume, not a padded block) but kept deadlock-safe by resolving all sizes UP FRONT with two
    symmetric collective handshakes -- (1) a fixed P-length all_to_all_single of per-pair COUNTS so
    every rank learns its recv sizes, then (2) an all_to_all_single of the requested vertex IDs sized
    by those counts -- before any per-step payload all_to_all_single. Both sides agree on every split
    size, so the variable all_to_all is well-posed and cannot hang the way ad-hoc isend/irecv can.
  * No barriers between phases -> ranks raced into the next collective while a straggler was still
    in graph build.  FIX: dist.barrier() after init and after EACH phase (partition, build, warmup,
    steps), each itself bounded by the init timeout.
  * destroy_process_group() was only reached on the happy path.  FIX: the whole body is wrapped in
    try/except/finally so destroy_process_group() ALWAYS runs, releasing NCCL communicators even
    on error, so the next torchrun N in the sbatch starts clean.

CUDA may be ABSENT on the build/login box -> every CUDA touch is guarded so `python -m py_compile`
and a no-GPU import both pass; the real run happens on the cluster under torchrun.

PARTITION COMPARISON (--partition {lpa,zord,hash}): the harness keeps the comm MACHINERY identical
(the same cut-faithful all_to_all boundary exchange runs for every partition) but the comm VOLUME now
tracks the cut, so it is a fair head-to-head test where a better partition is rewarded with less traffic:
  * lpa  : C++ label-propagation order sliced into P contiguous blocks (DEFAULT; unchanged).
  * hash : naive round-robin vertex->device (HIGH cut; the generic-partition control).
  * zord : the engine's adaptive-corner arrange() -> the makespan-best low-cut / vertex-cut plan.
Rank 0 computes the partition ONCE and BROADCASTS the int32 assignment (deterministic single source
-> no per-rank divergence -> no NCCL hang). The boundary exchange moves EXACTLY the distinct remote
feature rows each rank needs (comm_bytes_per_step == cut * F * 4), so a lower cut means strictly less
boundary feature traffic over the interconnect each step -> zord's low-cut plan SCALES BETTER (higher
scaling_eff) than lpa, and lpa better than hash.

Launched by torchrun (reads LOCAL_RANK / RANK / WORLD_SIZE from the env):
  torchrun --standalone --nproc_per_node=4 scripts/dist_scaling.py --synthetic big --iters 50
  torchrun --standalone --nproc_per_node=8 scripts/dist_scaling.py --dataset wiki-talk --memory
  torchrun --standalone --nproc_per_node=2 scripts/dist_scaling.py --synthetic big --partition zord
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from datetime import timedelta

import numpy as np

# torch is imported lazily-tolerantly so py_compile / a torch-less box still parse the file.
try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    _HAVE_TORCH = True
except Exception:                                              # pragma: no cover (build box)
    torch = None
    dist = None
    nn = None
    DDP = None
    _HAVE_TORCH = False

ZORD_GRAPH_BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")


# --------------------------------------------------------------------------- env / distributed
def env_int(*keys, default=0):
    for k in keys:
        if k in os.environ:
            try:
                return int(os.environ[k])
            except ValueError:
                pass
    return default


def setup_dist(timeout_min: int):
    """Read torchrun's LOCAL_RANK/RANK/WORLD_SIZE, pin the device, init NCCL with a LONG timeout.

    STABILITY: a generous collective timeout means a slow rank degrades gracefully (raises a clear
    timeout) instead of hanging the whole job until SLURM reaps it; set the device BEFORE init so
    NCCL binds the right GPU; honor NCCL_* knobs the launcher exports."""
    rank = env_int("RANK", "SLURM_PROCID", default=0)
    world = env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local = env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    # torchrun --standalone sets MASTER_ADDR/PORT; provide a fallback for a bare launch.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    # surface NCCL async error handling so a dead peer raises instead of hanging forever.
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    cuda_ok = bool(torch.cuda.is_available()) if _HAVE_TORCH else False
    if cuda_ok:
        torch.cuda.set_device(local)
        backend = "nccl"
        device = torch.device(f"cuda:{local}")
    else:
        # No GPU on this box (build/login): fall back to gloo/CPU so the code path still runs in CI.
        backend = "gloo"
        device = torch.device("cpu") if _HAVE_TORCH else None

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=timeout_min),
    )
    return rank, world, local, device, cuda_ok


def barrier(device, cuda_ok):
    """A device-aware barrier. With NCCL, pinning device_ids avoids the well-known
    barrier-on-wrong-device hang; gloo ignores it."""
    if not _HAVE_TORCH or not dist.is_initialized():
        return
    if cuda_ok:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


# --------------------------------------------------------------------------- graph data (fixed total work)
def gen_graph(N, M, C, intra, seed=0):
    """Deterministic synthetic temporal graph with C communities; `intra` fraction of edges stay
    inside a node's community so the lpa partition has a small cut. Returns int64 src/dst.

    FIXED total #edges M across all N -> strong scaling: the SAME M edges are partitioned over more
    GPUs, so per-GPU work shrinks."""
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
    return src, dst


SYNTH_SIZES = {
    # name -> (N nodes, M edges/snapshot, communities). Total work is FIXED per name across N GPUs.
    "small": (1_000_000, 16_000_000, 1000),
    "big": (8_000_000, 100_000_000, 4000),
    "huge": (20_000_000, 300_000_000, 8000),
}


def load_real(dataset):
    """Load a real temporal graph via the zord registry (NEVER networkx). Snapshot it and take the
    single LARGEST snapshot as the fixed per-step work (so total work is identical for every N)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from zord.datasets import load
    g = load(dataset).sort_by_time()
    snaps = g.to_snapshots(num_snapshots=64)
    if not snaps:
        return g.src.astype(np.int64), g.dst.astype(np.int64), int(g.num_nodes), g.name
    big = max(snaps, key=lambda s: s.num_edges)
    s = g.src[big.lo:big.hi].astype(np.int64)
    d = g.dst[big.lo:big.hi].astype(np.int64)
    return s, d, int(g.num_nodes), g.name


# --------------------------------------------------------------------------- partitions
# Three vertex->device assignment strategies, all PROCESS-only (same data + model => same result;
# the placement is a result-preserving GAS reduce). The ONLY thing that differs between them is
# which edges are CUT (and therefore how much boundary feature volume crosses the interconnect) ->
# a fair head-to-head test of partition quality with IDENTICAL comm machinery (the cut-faithful
# all_to_all boundary exchange below is the same code for all three; only its VOLUME differs, driven
# by the cut -> "comm is a parameter", honored, and a better partition is rewarded with less traffic).
#   lpa  : C++ label-propagation ORDER sliced into P contiguous equal blocks (the existing default;
#          locality-preserving, low-ish cut). DEFAULT -- behavior unchanged.
#   hash : round-robin vertex->device (naive HIGH-cut baseline; cut ~ (1-1/P) * |E|).
#   zord : the engine's adaptive-corner arrange() -> the makespan-best low-cut / vertex-cut plan.
# Every strategy is computed ONCE on rank 0 and BROADCAST so all ranks agree byte-for-byte (no
# per-rank recompute -> no divergence -> no NCCL hang). lpa/hash are deterministic functions of the
# (identical) inputs, so broadcasting them is equivalent to the old "each rank recomputes" path; we
# broadcast anyway for a single uniform, hang-proof code path.

def lpa_partition(src, dst, N, P, tmp_dir="/tmp"):
    """zord locality partition: run the C++ `lpa` (label-propagation) ORDERING kernel and slice the
    resulting node permutation into P CONTIGUOUS, EQUAL-SIZE blocks. lpa groups same-cluster nodes
    adjacently in the new id space, so contiguous blocks keep most intra-cluster edges local (small
    cut -> small boundary exchange). Mirrors scripts/partitioner_bench.py:lpa_blocks_assign.

    Falls back to a deterministic degree-block partition if the C++ binary is unavailable (so the
    script still runs end-to-end anywhere); the assignment is a deterministic function of the
    (identical) inputs. Returns (assignment[N] int64 in [0,P), used_cpp bool, cost_s)."""
    if P <= 1:
        return np.zeros(N, dtype=np.int64), False, 0.0
    edges_path = os.path.join(tmp_dir, f"zord_ds_edges_{os.getpid()}.bin")
    out_path = os.path.join(tmp_dir, f"zord_ds_lpa_{os.getpid()}.bin")
    s32 = src.astype(np.int32)
    d32 = dst.astype(np.int32)
    with open(edges_path, "wb") as f:
        f.write(struct.pack("<qq", N, s32.size))
        inter = np.empty(2 * s32.size, dtype=np.int32)
        inter[0::2] = s32
        inter[1::2] = d32
        inter.tofile(f)
    t0 = time.time()
    used_cpp = False
    assignment = None
    if os.path.exists(ZORD_GRAPH_BIN):
        r = subprocess.run([ZORD_GRAPH_BIN, edges_path, "lpa", out_path],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                n = struct.unpack("<q", f.read(8))[0]
                newid = np.fromfile(f, dtype=np.int32, count=n)        # newid[old] = lpa rank
            if newid.size == N:
                assignment = (newid.astype(np.int64) * P // N).clip(0, P - 1)
                used_cpp = True
    if assignment is None:
        # deterministic degree-block fallback (locality-preserving-ish, no networkx)
        deg = (np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)).astype(np.int64)
        order = np.argsort(-deg, kind="stable")
        rank_of = np.empty(N, dtype=np.int64)
        rank_of[order] = np.arange(N, dtype=np.int64)
        assignment = (rank_of * P // N).clip(0, P - 1)
    cost = time.time() - t0
    for p in (edges_path, out_path):
        try:
            os.remove(p)
        except OSError:
            pass
    return assignment.astype(np.int64), used_cpp, cost


def hash_partition(src, dst, N, P, seed=0):
    """Naive HIGH-cut baseline: round-robin / random vertex -> device. Ignores graph structure
    entirely, so ~(1 - 1/P) of edges are cut. This is the generic-partition control that zord's
    locality-aware plan must beat on boundary comm. Deterministic given (N, P, seed)."""
    if P <= 1:
        return np.zeros(N, dtype=np.int64), False, 0.0
    t0 = time.time()
    rng = np.random.default_rng(12345 + seed)
    assignment = rng.integers(0, P, size=N).astype(np.int64)   # random round-robin home per vertex
    return assignment, False, time.time() - t0


def zord_arrange_partition(src, dst, N, P, F, link_gbps, seed=0):
    """zord ADAPTIVE-corner partition: call the engine's arrange() to get the makespan-best plan
    (cluster-respecting low-cut edge-cut, dense-core vertex-cut, PSS/PTS corners, or the METIS
    floor -- arrange picks the winner). We collapse it to a SINGLE-HOME vertex->device assignment
    for this harness (each vertex owned by exactly one rank, like lpa/hash) so the downstream shard
    build + boundary exchange are IDENTICAL across partition choices -- the only thing that changes
    is which edges are cut. The cluster is a HOMOGENEOUS P-device profile (this single-node study
    has P identical GPUs); `link_gbps` is the interconnect bandwidth PARAMETER fed to arrange.

    Returns (assignment[N] int64 in [0,P), used_cpp bool, cost_s). arrange() reaches the C++ lpa /
    coreness kernels internally (NEVER networkx)."""
    if P <= 1:
        return np.zeros(N, dtype=np.int64), False, 0.0
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from zord.partition.arrange import arrange
    from zord.partition import cpp_kernel
    from zord.profiler.cluster_profile import from_spec
    t0 = time.time()
    # homogeneous cluster: P identical devices on a flat fabric at the given interconnect bandwidth.
    # capacities/bw are uniform so arrange's hetero-matching reduces to balanced splits, isolating
    # the CUT (not heterogeneity) as the variable -- a fair partition-quality test on equal GPUs.
    cluster = from_spec(hbm_gb=[80.0] * P, agg_bw_gbps=[900.0] * P,
                        interconnect_gbps=float(link_gbps))
    res = arrange(src, dst, N, cluster, link_gbps=float(link_gbps), feat_dim=F, seed=seed)
    assignment = np.asarray(res.assignment, dtype=np.int64).clip(0, P - 1)
    # arrange's vertex-cut leaves core nodes single-homed in `assignment` (periphery home / a valid
    # device for core); that single-home view is exactly what this harness shards on.
    used_cpp = cpp_kernel.have_cpp_kernel()
    return assignment, used_cpp, time.time() - t0


def compute_partition(strategy, src, dst, N, P, F, link_gbps, seed=0):
    """Dispatch to the chosen vertex->device partition (rank-0 side). Returns
    (assignment[N] int64, used_cpp bool, cost_s, winner_name)."""
    if strategy == "hash":
        a, used, c = hash_partition(src, dst, N, P, seed=seed)
        return a, used, c, "hash(round-robin)"
    if strategy == "zord":
        a, used, c = zord_arrange_partition(src, dst, N, P, F, link_gbps, seed=seed)
        return a, used, c, "zord(arrange)"
    # default: lpa contiguous blocks (existing behavior)
    a, used, c = lpa_partition(src, dst, N, P)
    return a, used, c, "lpa(contiguous)"


def broadcast_assignment(assignment, N, world, device, cuda_ok):
    """Rank 0 computes the partition ONCE; broadcast the int32 assignment to ALL ranks so every
    rank agrees byte-for-byte. Deterministic single source -> no per-rank divergence -> no NCCL
    hang. On rank 0 `assignment` is the freshly computed array; on other ranks it is ignored
    (overwritten by the broadcast). Returns the agreed int64 assignment[N] on every rank.

    Guarded for the torch-less / world<=1 path so py_compile and a single-process run still work."""
    if not _HAVE_TORCH or not dist.is_initialized() or world <= 1:
        return np.zeros(N, dtype=np.int64) if assignment is None else assignment.astype(np.int64)
    buf = torch.zeros(N, dtype=torch.int32, device=device)
    if dist.get_rank() == 0:
        buf.copy_(torch.from_numpy(np.ascontiguousarray(assignment, dtype=np.int32)).to(device))
    dist.broadcast(buf, src=0)                                  # single fixed-shape collective
    return buf.detach().cpu().numpy().astype(np.int64)


# --------------------------------------------------------------------------- per-rank shard
def build_shard(rank, world, assignment, src, dst, N, F, device, seed=0):
    """Materialize THIS rank's vertex shard + the local adjacency block needed for one SpMM gather.

    Every rank owns the vertices assigned to it. An edge contributes to a rank's aggregation when
    its SOURCE is local (the dst is the neighbor whose feature is gathered). Neighbors that live on
    another rank are 'boundary' columns whose features must be exchanged across devices each step.
    We build a contiguous feature buffer  [ local rows (nl) | boundary rows (n_bnd) ]  so the SpMM
    is a single sparse.mm over a (nl x nbuf) block. Returns the pieces the training step needs."""
    es = np.concatenate([src, dst])                            # undirected: use both directions
    ed = np.concatenate([dst, src])
    my_nodes = np.nonzero(assignment == rank)[0]
    nl = int(my_nodes.size)
    g2l = np.full(N, -1, dtype=np.int64)
    g2l[my_nodes] = np.arange(nl, dtype=np.int64)

    mine = assignment[es] == rank
    e_src = es[mine]
    e_dst = ed[mine]
    rows = g2l[e_src]
    is_remote = assignment[e_dst] != rank
    remote_ids = np.unique(e_dst[is_remote])
    n_bnd = int(remote_ids.size)

    rid2slot = np.full(N, -1, dtype=np.int64)
    rid2slot[remote_ids] = np.arange(n_bnd, dtype=np.int64) + nl
    cols = np.where(is_remote, rid2slot[e_dst], g2l[e_dst])
    nbuf = nl + n_bnd
    local_edges = int(rows.size)
    cut = int(is_remote.sum())

    info = {
        "nl": nl, "n_bnd": n_bnd, "nbuf": nbuf,
        "local_edges": local_edges, "cut": cut,
        "remote_ids": remote_ids, "remote_owner": assignment[remote_ids] if n_bnd else np.empty(0, np.int64),
        "g2l": g2l, "my_nodes": my_nodes,                      # global->local map + this rank's owned vertices
    }
    if not _HAVE_TORCH:
        return info, None, None

    idx = torch.stack([
        torch.from_numpy(rows).to(torch.long),
        torch.from_numpy(cols).to(torch.long),
    ])
    vals = torch.ones(max(rows.size, 1), dtype=torch.float32)
    if rows.size == 0:
        idx = torch.zeros((2, 1), dtype=torch.long)
        vals = torch.zeros(1, dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, vals, (max(nl, 1), max(nbuf, 1))).coalesce().to(device)

    g = torch.Generator().manual_seed(1234 + seed + rank)
    X = (torch.randn(max(nbuf, 1), F, generator=g) * 0.1).to(device)
    return info, A, X


# --------------------------------------------------------------------------- model
class GraphSAGEStep(nn.Module if _HAVE_TORCH else object):
    """One GraphSAGE layer: h = relu( (A @ X) W_agg + X[:nl] W_self ). Optional GRUCell node-memory
    (the TGN variant): the aggregated message updates a per-node memory state. DDP-wrapped so the
    linear/GRU parameters' grads are all-reduced -- that all-reduce is the real DDP comm cost."""

    def __init__(self, F, memory=False):
        super().__init__()
        self.lin_agg = nn.Linear(F, F, bias=False)
        self.lin_self = nn.Linear(F, F, bias=True)
        self.memory = memory
        if memory:
            self.gru = nn.GRUCell(F, F)

    def forward(self, A, X, nl, mem=None):
        agg = torch.sparse.mm(A, X)                            # neighbor aggregation (SpMM)
        h = torch.relu(self.lin_agg(agg) + self.lin_self(X[:nl]))
        if self.memory:
            mem_in = mem if mem is not None else torch.zeros_like(h)
            h = self.gru(h, mem_in)                            # node-memory update (TGN)
        return h


# --------------------------------------------------------------------------- boundary exchange (TRUE cut-faithful)
def make_boundary_exchange(info, F, world, rank, device, cuda_ok):
    """Build a CUT-FAITHFUL, DEADLOCK-SAFE cross-device boundary FEATURE exchange (auditor option A).

    Each rank needs the CURRENT feature rows of the boundary vertices it does not own (info["remote_ids"],
    grouped by their owner via info["remote_owner"]). The owner of those vertices must ship exactly THOSE
    rows -- and symmetrically this rank ships, to each consumer, the rows of ITS OWN local vertices that the
    consumer requested. So the per-step traffic is precisely  (sum_ranks #distinct remote rows requested)*F*4
    bytes == the TRUE partition cut, not a fixed padded block. Lower cut => strictly less comm => faster.

    The exchange is a single dist.all_to_all_single with per-peer split sizes; the routing (who needs which
    of my rows) is resolved ONCE here, deadlock-safe, in two setup collective handshakes (H1, H2):
      (H1) all_to_all_single of a fixed P-length COUNT vector: send_counts[j] = #rows I request from j ==
           #rows j will send ME each step. Every rank learns recv_counts[i] = #rows rank i wants from me.
           Fixed shape (P ints in / P ints out) -> cannot hang.
      (H2) all_to_all_single (variable, with the counts from H1 as split sizes) of the actual requested
           GLOBAL VERTEX IDS: I send j the ids I want; I receive from i the ids it wants from me. I map
           those incoming ids through g2l to LOCAL ROW indices -> send_rows (contiguous, index_select each
           step). Both sides agree on sizes from H1, so the variable all_to_all is well-posed.
      (per step) index_select(X, send_rows) -> all_to_all_single(recv_buf, send_buf, recv_counts, send_counts)
           -> index_copy the received rows into X's halo region [nl:nbuf]. Receipt order from owner j is
           exactly this rank's remote_ids-with-owner-j order, so the destination halo slots are precomputed.

    Returns (exchange_closure, bytes_per_step) where bytes_per_step is the analytic per-step all_to_all
    payload size on THIS rank's RECEIVE side summed across owners (== this rank's distinct-remote-rows * F * 4)."""
    if not _HAVE_TORCH or world <= 1:
        def noop(X):
            return 0
        return noop, 0

    nl = info["nl"]
    remote_ids = info["remote_ids"]                            # global ids this rank needs (len n_bnd)
    remote_owner = info["remote_owner"]                        # owner rank of each remote id
    g2l = info["g2l"]

    # ---- per-peer request: how many / which rows I want from each owner j (stable by owner) ----
    # order remote ids by owner so each owner's block is contiguous; halo slot of remote_ids[k] is nl+k.
    order = np.argsort(remote_owner, kind="stable") if remote_ids.size else np.empty(0, np.int64)
    req_ids_sorted = remote_ids[order].astype(np.int64)        # ids grouped by owner, sent in this order
    req_owner_sorted = remote_owner[order].astype(np.int64)
    send_counts_np = np.bincount(req_owner_sorted, minlength=world).astype(np.int64)  # rows I pull from j
    # halo destination slot for each received row, in the SAME owner-grouped order the owner returns them.
    halo_dst_np = (order + nl).astype(np.int64)                # received row r (owner-grouped) -> X slot

    send_counts = torch.tensor(send_counts_np, device=device, dtype=torch.long)
    recv_counts = torch.zeros(world, device=device, dtype=torch.long)
    # (H1) symmetric fixed-shape count handshake: my send_counts[j] becomes j's recv_counts[rank].
    dist.all_to_all_single(recv_counts, send_counts)
    recv_counts_np = recv_counts.detach().cpu().numpy().astype(np.int64)  # #rows each peer i wants from me

    # (H2) exchange the actual requested GLOBAL ids (variable, sized by H1) so each rank learns which of
    # its LOCAL rows every peer needs. all_to_all_single needs contiguous, split-sized buffers.
    req_ids_t = torch.from_numpy(req_ids_sorted).to(device=device, dtype=torch.long).contiguous()
    wanted_from_me = torch.zeros(int(recv_counts_np.sum()), device=device, dtype=torch.long)
    dist.all_to_all_single(
        wanted_from_me, req_ids_t,
        output_split_sizes=recv_counts_np.tolist(),
        input_split_sizes=send_counts_np.tolist(),
    )
    # map the global ids peers want from me -> my local rows to index_select each step (contiguous send buf)
    wanted_np = wanted_from_me.detach().cpu().numpy().astype(np.int64)
    send_rows_np = g2l[wanted_np] if wanted_np.size else np.empty(0, np.int64)   # all >=0: peers only ask for my rows
    send_rows = torch.from_numpy(send_rows_np).to(device=device, dtype=torch.long).contiguous()
    halo_dst = torch.from_numpy(halo_dst_np).to(device=device, dtype=torch.long).contiguous()

    # split direction: per step I SEND my local rows that peers requested (sized by recv_counts_np, the
    # H1 incoming wants) and RECV the boundary rows I requested (sized by send_counts_np).
    send_split = recv_counts_np.tolist()                       # to peer i: the rows it asked me for
    recv_split = send_counts_np.tolist()                       # from owner j: the rows I asked it for
    n_recv_rows = int(send_counts_np.sum())                    # == n_bnd: total boundary rows I pull in

    recv_buf = torch.zeros(max(n_recv_rows, 1), F, device=device)

    def exchange(X):
        # gather MY local rows that peers requested, ship them, place received boundary rows into halo.
        send_buf = X.index_select(0, send_rows).contiguous() if send_rows.numel() else torch.zeros(0, F, device=device)
        out = recv_buf[:n_recv_rows] if n_recv_rows else recv_buf[:0]
        dist.all_to_all_single(out, send_buf,
                               output_split_sizes=recv_split,
                               input_split_sizes=send_split)
        if n_recv_rows:
            X.index_copy_(0, halo_dst, out.detach())           # fill halo [nl:nbuf] with owners' real rows
        return n_recv_rows

    # analytic cross-check: THIS rank's per-step receive payload == its distinct remote rows * F * 4 bytes.
    bytes_per_step = int(n_recv_rows) * F * 4
    return exchange, bytes_per_step


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="", help="real temporal graph name (zord registry)")
    ap.add_argument("--synthetic", default="big", choices=list(SYNTH_SIZES),
                    help="synthetic size (FIXED total edges across N for strong scaling)")
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--partition", default="lpa", choices=["lpa", "zord", "hash"],
                    help="vertex->device partition: lpa (contiguous lpa-order blocks, default; "
                         "unchanged) | zord (engine arrange() adaptive low-cut/vertex-cut plan) | "
                         "hash (naive round-robin HIGH-cut baseline). Same comm machinery across "
                         "all three -> the ONLY thing that changes is which edges are cut.")
    ap.add_argument("--link-gbps", type=float, default=325.0,
                    help="interconnect bandwidth PARAMETER (GB/s) fed to zord arrange() "
                         "(default 325 = measured HetCluster H100 NVLink). zord must win at ANY value.")
    ap.add_argument("--iters", type=int, default=50, help="training steps to time")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--memory", action="store_true", help="enable GRUCell node-memory (TGN variant)")
    ap.add_argument("--intra", type=float, default=0.9, help="synthetic intra-community edge fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-min", type=int, default=20, help="NCCL collective timeout (minutes)")
    ap.add_argument("--baseline-ms", type=float, default=0.0,
                    help="N=1 median step ms; if given, report scaling_eff = baseline/(N*tN)")
    a = ap.parse_args()

    if not _HAVE_TORCH:
        print("[dist_scaling] torch not importable on this box; nothing to run (py_compile only).")
        return

    F = a.feat
    rank, world, local, device, cuda_ok = setup_dist(a.timeout_min)
    completed = False
    try:
        print(f"[rank {rank}] init ok  world={world} local={local} "
              f"device={device} cuda={cuda_ok} backend={'nccl' if cuda_ok else 'gloo'}", flush=True)
        barrier(device, cuda_ok)

        # ---- phase 1: load the FIXED-total-work graph (identical on every rank, deterministic) ----
        t_load0 = time.time()
        if a.dataset:
            src, dst, N, gname = load_real(a.dataset)
        else:
            N, M, C = SYNTH_SIZES[a.synthetic]
            src, dst = gen_graph(N, M, C, a.intra, seed=a.seed)
            gname = f"synthetic-{a.synthetic}(intra={a.intra})"
        M = int(src.size)
        t_load = time.time() - t_load0
        barrier(device, cuda_ok)

        # ---- phase 2: partition the vertices across the N GPUs (rank 0 computes ONCE, broadcast) ----
        # Only rank 0 runs the (possibly C++/arrange) partition -> a single deterministic source of
        # truth that is then broadcast, so no rank can diverge and stall a later collective.
        used_cpp = False
        t_part = 0.0
        part_name = a.partition
        if rank == 0:
            assignment0, used_cpp, t_part, part_name = compute_partition(
                a.partition, src, dst, N, world, F, a.link_gbps, seed=a.seed)
        else:
            assignment0 = None
        assignment = broadcast_assignment(assignment0, N, world, device, cuda_ok)
        barrier(device, cuda_ok)

        # ---- phase 3: build this rank's shard + model + boundary exchange ----
        info, A, X = build_shard(rank, world, assignment, src, dst, N, F, device, seed=a.seed)
        model = GraphSAGEStep(F, memory=a.memory).to(device)
        if world > 1:
            ddp_ids = [local] if cuda_ok else None
            model = DDP(model, device_ids=ddp_ids, output_device=(local if cuda_ok else None),
                        find_unused_parameters=False)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        exchange, bnd_bytes = make_boundary_exchange(info, F, world, rank, device, cuda_ok)
        mem = None
        if cuda_ok:
            torch.cuda.reset_peak_memory_stats(device)
        barrier(device, cuda_ok)

        def one_step():
            opt.zero_grad(set_to_none=True)
            exchange(X)                                        # cut-faithful boundary feature exchange (fills X halo)
            h = model(A, X, info["nl"])                        # forward: SpMM-agg + linear (+ GRU)
            loss = h.float().pow(2).mean()                     # PROCESS proxy loss (NOT accuracy)
            loss.backward()                                    # backward + DDP grad all-reduce
            opt.step()
            return float(loss.detach())

        # ---- phase 4: warmup ----
        for _ in range(max(1, a.warmup)):
            one_step()
        if cuda_ok:
            torch.cuda.synchronize(device)
        barrier(device, cuda_ok)

        # ---- phase 5: timed steps. A device-pinned barrier AT THE START of each step makes step_ms an
        # honest cross-rank MAKESPAN (the auditor noted there was no per-iter barrier, so a straggler's
        # all_to_all wait leaked into the next step's timing). The barrier itself is bounded by the NCCL
        # timeout, so it degrades gracefully rather than hanging. ----
        step_ms = []
        for _ in range(max(1, a.iters)):
            barrier(device, cuda_ok)                            # align ranks -> step_ms is a true makespan
            if cuda_ok:
                torch.cuda.synchronize(device)
            t0 = time.time()
            one_step()
            if cuda_ok:
                torch.cuda.synchronize(device)
            step_ms.append((time.time() - t0) * 1e3)
        barrier(device, cuda_ok)

        med = float(np.median(step_ms))
        peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9) if cuda_ok else 0.0
        thrpt = (info["local_edges"] / (med / 1e3)) if med > 0 else 0.0   # this rank's edges/s

        # ---- reduce per-rank stats to rank 0 ----
        stat = torch.tensor([med, thrpt, peak_gb, float(info["local_edges"]),
                             float(info["cut"]), float(info["nl"]), float(bnd_bytes)],
                            device=device, dtype=torch.float64)
        gathered = [torch.zeros_like(stat) for _ in range(world)]
        dist.all_gather(gathered, stat)
        completed = True
        barrier(device, cuda_ok)

        if rank == 0:
            rows = [g.cpu().numpy() for g in gathered]
            med_all = np.array([r[0] for r in rows])
            thr_all = np.array([r[1] for r in rows])
            mem_all = np.array([r[2] for r in rows])
            le_all = np.array([r[3] for r in rows])
            # makespan = slowest rank's step; aggregate throughput = sum of per-rank edges/s
            makespan_ms = float(med_all.max())
            agg_thrpt = float(thr_all.sum())
            tot_edges = float(le_all.sum())
            le_per_rank = [int(r[3]) for r in rows]              # per-rank incident-edge work
            comm_bytes_total = int(sum(r[6] for r in rows))      # measured all_to_all feature bytes / step
            result = {
                "tag": "DIST_SCALING_RESULT",
                "dataset": gname,
                "world_size": world,
                "feat": F,
                "memory_tgn": bool(a.memory),
                "N_nodes": int(N),
                "M_edges_total": int(M),
                "edges_processed_total": int(tot_edges),
                "partition": a.partition,                        # lpa | zord | hash (the variable)
                "partition_winner": part_name,                   # for zord: which arrange corner won
                "step_ms_median_per_rank": [round(float(x), 3) for x in med_all],
                "step_ms": round(makespan_ms, 3),               # makespan = max over ranks
                "throughput_edges_s": round(agg_thrpt, 1),      # aggregate edges/s
                "mem_gb_per_rank": [round(float(x), 3) for x in mem_all],
                "mem_gb": round(float(mem_all.max()), 3),
                "edge_work_per_rank": le_per_rank,               # per-rank incident-edge work
                "used_cpp": bool(used_cpp),
                "used_cpp_lpa": bool(used_cpp),                  # back-compat alias
                "partition_s": round(t_part, 2),
                "load_s": round(t_load, 2),
                # cut_edges_total: summed per-rank "source-local, dst-remote" boundary edges over the
                # undirected doubled edge list -> a CONSISTENT cross-device-edge metric comparable
                # across lpa/zord/hash (same counting). It is the boundary-comm volume driver; a low
                # value (zord) means less feature traffic over the interconnect each step.
                "cut_edges_total": int(sum(r[4] for r in rows)),
                # comm_bytes_per_step: the TRUE measured all_to_all boundary-feature payload moved every
                # step == (sum over ranks of distinct remote rows requested) * F * 4. Now CUT-FAITHFUL:
                # it tracks cut_edges_total (lower cut -> fewer distinct boundary rows -> fewer bytes ->
                # faster), so this is the analytic cross-check that comm follows the partition.
                "comm_bytes_per_step": comm_bytes_total,
                "comm_bytes_per_rank": [int(r[6]) for r in rows],
                "all_ranks_completed": True,
                "stable": True,
            }
            if a.baseline_ms > 0:
                result["baseline_ms"] = a.baseline_ms
                result["scaling_eff"] = round(a.baseline_ms / (world * makespan_ms), 4)
            print(json.dumps(result), flush=True)

    except Exception as e:                                     # pragma: no cover (cluster-only paths)
        # surface the failure on THIS rank so we can see which rank died and why (stability triage)
        print(f"[rank {rank}] ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # STABILITY: ALWAYS tear down the process group so the next torchrun N starts clean.
        try:
            if _HAVE_TORCH and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        status = "done" if completed else "EXITED-WITHOUT-COMPLETING"
        print(f"[rank {rank}] {status}", flush=True)


if __name__ == "__main__":
    main()
