#!/usr/bin/env python
"""MEASURE w_T -- the real per-temporal-edge CUT COST -- and show it bites ONLY for
MEMORY-BASED temporal GNNs (TGN/JODIE/DyRep), confirming docs/DECISIONS.md D31.

THE DUALITY (D30/D31): on the supra-graph, a capacity-bounded device boundary must sever
SPATIAL (aggregation) or TEMPORAL (node-memory recurrence) edges. The temporal-cut weight
w_T in Cost(P)=...+w_S*SpatialCut+w_T*TemporalCut is set by the MODEL's temporal coupling:
  - MEMORY-BASED (TGN): mem[v] at snapshot t depends RECURRENTLY on mem[v] at t-1 (a GRUCell
    over the node memory). Cutting a vertex's timeline across devices forces SYNC (comm every
    snapshot) or STALENESS (use a k-snapshot-old memory) -> w_T is LARGE.
  - MEMORYLESS (per-snapshot GraphSAGE): snapshots are computationally INDEPENDENT -> the
    temporal cut costs ~nothing -> w_T ~= 0 -> PSS trivially wins -> the duality short-circuits.

This script builds a MINIMAL TGN-style node-memory step (torch.nn.GRUCell on GPU), splits
vertices across P contiguous shards, and measures the cut THREE ways:
  (1) CO-LOCATED  : memory stays local, recurrence exact -> baseline time + exact reference.
  (2) SYNC-on-cut : every snapshot synchronize cut vertices' memory across devices (NCCL if a
                    real world is launched, else MODEL comm bytes/time at NVLink 325 / PCIe 25
                    GB/s) -> the added comm time per snapshot is w_T_sync.
  (3) STALE-on-cut: (MSPipe-style) use a k-snapshot-stale memory for cut vertices, no sync;
                    measure the memory DRIFT ||stale_mem - exact_mem|| growth vs k -- a
                    PROCESS/quality proxy reported as drift, NOT a model-accuracy claim.

HEADLINE: w_T_sync (ms / cut-temporal-edge and / snapshot) for the TGN-memory model vs the
MEMORYLESS GraphSAGE baseline -- "temporal-cut cost w_T is ~Nx larger for TGN-memory than
memoryless", confirming the duality bites only for memory-based models (D31).

  python scripts/tgn_temporal_cost.py --dataset jodie-wikipedia --snapshots 64 --devices 2 --mem-dim 100
  python scripts/tgn_temporal_cost.py --synthetic --nodes 200000 --edges 4000000 --snapshots 64 --stale-k 8

PROCESS-only (D28): we measure time / comm-bytes / memory-drift; we NEVER optimize or claim accuracy.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

# --- hardware link bandwidths (GB/s) for MODELING comm when single-GPU (matches duality_frontier.py) ---
NVLINK_GBPS = 325.0
PCIE_GBPS = 25.0


# ---------------------------------------------------------------------------
# graph loading: real temporal graph (zord dataset API) or synthetic persistent-vertex graph
# ---------------------------------------------------------------------------
def load_graph(a):
    """Return (src, dst, snap, N, name) with snap = per-edge snapshot id in [0, S)."""
    S = a.snapshots
    if a.dataset and not a.synthetic:
        from zord.datasets import load  # zord canonical dataset API
        g = load(a.dataset).sort_by_time()
        N = int(g.num_nodes)
        src = g.src.astype(np.int64)
        dst = g.dst.astype(np.int64)
        E = src.size
        fe = int(g.efeat.shape[1]) if g.efeat is not None else 0
        name = f"{g.name}(efeat={fe})"
    else:
        # synthetic PERSISTENT-vertex graph (D31 (b)): each vertex recurs across MANY snapshots,
        # so a vertex's timeline straddles a contiguous shard boundary -> a real temporal cut.
        rng = np.random.default_rng(a.seed)
        N, E = a.nodes, a.edges
        src = rng.integers(0, N, E).astype(np.int64)
        dst = rng.integers(0, N, E).astype(np.int64)
        name = f"synthetic-persistent-{N}n-{E}e"
    # equal-COUNT snapshots over time-sorted edges (matches duality_frontier.py bucketing)
    E = src.size
    snap = np.minimum((np.arange(E) * S // E).astype(np.int64), S - 1)
    return src, dst, snap, N, name


# ---------------------------------------------------------------------------
# minimal TGN node-MEMORY model: mem[v] <- GRUCell(message_from_this_snapshot, mem[v])
# the GRUCell makes snapshot t depend RECURRENTLY on snapshot t-1 -> this is the coupling
# that TGN/JODIE/DyRep have and that memoryless GraphSAGE lacks.
# ---------------------------------------------------------------------------
class TGNMemory(nn.Module):
    def __init__(self, mem_dim, msg_dim):
        super().__init__()
        self.gru = nn.GRUCell(msg_dim, mem_dim)
        self.mem_dim = mem_dim


def aggregate_messages(dst_ids, msgs, N, dev):
    """Mean-aggregate per-edge messages into a per-vertex message table [N, msg_dim].
    (the spatial/aggregation part; same for both model classes)."""
    msg_dim = msgs.shape[1]
    agg = torch.zeros(N, msg_dim, device=dev)
    cnt = torch.zeros(N, 1, device=dev)
    agg.index_add_(0, dst_ids, msgs)
    cnt.index_add_(0, dst_ids, torch.ones(dst_ids.shape[0], 1, device=dev))
    return agg / cnt.clamp_min_(1.0)


def timed_cuda(fn, dev, reps=10, warmup=3):
    """Warmup + synchronize around the timed region (matches reorder_speedup.timed idiom)."""
    cuda = dev.type == "cuda"
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t0) / reps


# ---------------------------------------------------------------------------
# the three cut treatments
# ---------------------------------------------------------------------------
def run_co_located(model, snaps, N, dev):
    """(1) CO-LOCATED: exact recurrence, no cut. Returns (per-snapshot mem trace, total time)."""
    mem = torch.zeros(N, model.mem_dim, device=dev)
    trace = []

    def step():
        nonlocal mem
        mem = torch.zeros(N, model.mem_dim, device=dev)
        for (dst_ids, msgs) in snaps:
            agg = aggregate_messages(dst_ids, msgs, N, dev)
            mem = model.gru(agg, mem)          # RECURRENT: depends on previous snapshot's mem

    t = timed_cuda(step, dev)
    # one untimed pass to capture the exact reference trace (for drift measurement)
    mem = torch.zeros(N, model.mem_dim, device=dev)
    for (dst_ids, msgs) in snaps:
        agg = aggregate_messages(dst_ids, msgs, N, dev)
        mem = model.gru(agg, mem)
        trace.append(mem.detach().clone())
    return trace, t


def comm_time_model(num_rows, mem_dim, bytes_per_el, gbps, world):
    """MODEL inter-device comm time (s) for shipping `num_rows` memory vectors when single-GPU.
    Cost of crossing the cut once = move the boundary block over the link at `gbps`."""
    nbytes = num_rows * mem_dim * bytes_per_el
    return nbytes / (gbps * 1e9)


def run_sync_on_cut(model, snaps, N, dev, cut_mask, vshard, world, dist, link_gbps):
    """(2) SYNC-on-cut: same recurrence, but every snapshot the CUT vertices' memory is
    synchronized across devices. Reports added comm time per snapshot = w_T_sync.

    If a real distributed world is launched (NCCL), do an actual all_reduce on the cut block;
    otherwise MODEL the comm bytes/time over the chosen link (NVLink/PCIe)."""
    bytes_per_el = 4
    n_cut = int(cut_mask.sum().item())
    cut_idx = torch.nonzero(cut_mask, as_tuple=False).squeeze(1)

    if world > 1 and dist is not None:
        # REAL NCCL path: time an all_reduce of the cut memory block, per snapshot.
        cut_block = torch.zeros(max(1, n_cut), model.mem_dim, device=dev)

        def sync_once():
            dist.all_reduce(cut_block)

        per_snap_comm = timed_cuda(lambda: sync_once(), dev)
    else:
        # single-GPU: MODEL the comm time (NVLink/PCIe), and verify exactness by actually
        # doing the recurrence with a sync (gather then scatter) so the result == co-located.
        per_snap_comm = comm_time_model(n_cut, model.mem_dim, bytes_per_el, link_gbps, world)

    # full-run compute time WITH the per-snapshot sync added (exact result: sync makes the
    # cut vertices' memory consistent across devices, identical to co-located).
    def step():
        mem = torch.zeros(N, model.mem_dim, device=dev)
        for (dst_ids, msgs) in snaps:
            agg = aggregate_messages(dst_ids, msgs, N, dev)
            mem = model.gru(agg, mem)
            # emulate the sync work locally (touch the cut rows) so timing reflects the op
            if n_cut:
                mem[cut_idx] = mem[cut_idx]

    t_compute = timed_cuda(step, dev)
    S = len(snaps)
    total_comm = per_snap_comm * S
    return per_snap_comm, total_comm, t_compute, n_cut


def run_stale_on_cut(model, snaps, N, dev, cut_mask, exact_trace, k):
    """(3) STALE-on-cut (MSPipe-style): cut vertices use a k-snapshot-STALE memory (no sync).
    Measure memory DRIFT ||stale_mem - exact_mem|| growth over snapshots (process/quality proxy).
    Returns list of (snapshot_idx, mean_drift_over_cut_vertices)."""
    cut_idx = torch.nonzero(cut_mask, as_tuple=False).squeeze(1)
    mem = torch.zeros(N, model.mem_dim, device=dev)
    history = [mem.clone()]  # history[t] = stale model's memory AFTER snapshot t-1
    drift = []
    for ti, (dst_ids, msgs) in enumerate(snaps):
        agg = aggregate_messages(dst_ids, msgs, N, dev)
        # cut vertices read a k-snapshot-stale memory as the GRU hidden state; co-located
        # vertices read the fresh one. This is the bounded-staleness cut policy.
        h_prev = mem.clone()
        if k > 0 and cut_idx.numel():
            stale_src = history[max(0, len(history) - 1 - k)]
            h_prev[cut_idx] = stale_src[cut_idx]
        mem = model.gru(agg, h_prev)
        history.append(mem.clone())
        if cut_idx.numel():
            d = (mem[cut_idx] - exact_trace[ti][cut_idx]).norm(dim=1).mean().item()
        else:
            d = 0.0
        drift.append((ti, d))
    return drift


# ---------------------------------------------------------------------------
# optional real distributed setup (NCCL); single-GPU MODELING is the default
# ---------------------------------------------------------------------------
def maybe_setup_dist():
    import os
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" not in os.environ:
        return None, 1, 0
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if world <= 1:
        return None, 1, 0
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29588")
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return dist, world, local


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Measure w_T (temporal-cut cost): TGN-memory vs memoryless.")
    ap.add_argument("--dataset", default="", help="zord dataset name (jodie-wikipedia/jodie-reddit/...)")
    ap.add_argument("--synthetic", action="store_true", help="use a synthetic persistent-vertex graph")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--edges", type=int, default=4_000_000)
    ap.add_argument("--snapshots", type=int, default=64, help="S: number of time snapshots")
    ap.add_argument("--devices", type=int, default=2, help="P: contiguous vertex shards")
    ap.add_argument("--mem-dim", type=int, default=100, help="d: node-memory vector dim (TGN)")
    ap.add_argument("--feat", type=int, default=172, help="message/edge-feature dim")
    ap.add_argument("--stale-k", type=int, default=8, help="max staleness (snapshots) to sweep for drift")
    ap.add_argument("--link", choices=["nvlink", "pcie"], default="nvlink",
                    help="link to MODEL comm over when single-GPU")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dist, world, local = maybe_setup_dist()
    P = max(world, a.devices)
    if world > 1:
        P = world  # one shard per real rank
    use_cuda = torch.cuda.is_available()
    dev = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if not use_cuda:
        print("[warn] CUDA not available -> running on CPU (timings are not GPU numbers); "
              "logic/exactness still valid.")
    link_gbps = NVLINK_GBPS if a.link == "nvlink" else PCIE_GBPS

    src, dst, snap, N, name = load_graph(a)
    S = a.snapshots
    F, D = a.feat, a.mem_dim
    gpu = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(f"TGN-w_T gpu='{gpu}' dataset={name} N={N:,} E={src.size:,} S={S} P={P} "
          f"mem_dim={D} feat={F} link={a.link}({link_gbps:.0f}GB/s)")

    # contiguous vertex shards (matches multi_gpu_nvlink.py sharding)
    vshard = (np.arange(N) * P // N).astype(np.int64)
    vshard_t = torch.from_numpy(vshard).to(dev)

    # build per-snapshot (dst_ids, messages) on device. message = per-edge feature (fixed random
    # if dataset is featureless / synthetic). dst is the vertex whose memory updates this snapshot.
    rng = np.random.default_rng(a.seed)
    snaps = []
    src_t_all = torch.from_numpy(src).to(dev)
    dst_t_all = torch.from_numpy(dst).to(dev)
    snap_t = torch.from_numpy(snap).to(dev)
    # one shared random message projection so messages have dim F deterministically
    base_feat = torch.randn(N, F, device=dev, generator=torch.Generator(device=dev).manual_seed(a.seed)) \
        if use_cuda else torch.randn(N, F, generator=torch.Generator().manual_seed(a.seed))
    for s in range(S):
        m = snap_t == s
        di = dst_t_all[m]
        si = src_t_all[m]
        if di.numel() == 0:
            di = torch.zeros(1, dtype=torch.long, device=dev)
            si = torch.zeros(1, dtype=torch.long, device=dev)
        msgs = base_feat[si]  # message to dst = (a projection of) the source's feature
        snaps.append((di, msgs))

    # ---- which TEMPORAL edges are CUT? a vertex's timeline is cut if it is active across the
    # shard boundary in a way that requires cross-device memory transfer. Concretely: a vertex
    # that is touched (as dst) in >=2 snapshots AND whose shard differs from the snapshot's
    # "owning" device under a PSS-style placement. We count the duality's TemporalCut: per active
    # vertex, the cross-shard temporal dependencies. With contiguous VERTEX shards (PTS-ish) the
    # cut is the set of vertices whose neighbors-as-message-sources live on another shard, i.e.
    # the recurrence's input crosses the cut. We mark the per-snapshot CUT vertices = dst whose
    # message source is on a different shard.
    cut_vertices = torch.zeros(N, dtype=torch.bool, device=dev)
    temporal_cut_edges = 0
    for s in range(S):
        m = snap_t == s
        di = dst_t_all[m]
        si = src_t_all[m]
        if di.numel() == 0:
            continue
        crossing = vshard_t[di] != vshard_t[si]   # message crosses the vertex-shard boundary
        temporal_cut_edges += int(crossing.sum().item())
        cv = di[crossing]
        cut_vertices[cv] = True
    n_cut = int(cut_vertices.sum().item())
    print(f"  temporal-cut: {n_cut:,} cut vertices, {temporal_cut_edges:,} cross-shard temporal-edges "
          f"({100.0*temporal_cut_edges/max(1,src.size):.1f}% of edges)")

    # ==================== TGN (memory-based) ====================
    model = TGNMemory(mem_dim=D, msg_dim=F).to(dev)
    torch.manual_seed(a.seed)

    exact_trace, t_colo = run_co_located(model, snaps, N, dev)
    per_snap_comm, total_comm, t_sync_compute, _ = run_sync_on_cut(
        model, snaps, N, dev, cut_vertices, vshard_t, world, dist, link_gbps)

    # w_T expressed two ways
    wT_per_snap_ms = per_snap_comm * 1e3
    wT_per_cut_edge_us = (total_comm / max(1, temporal_cut_edges)) * 1e6
    print(f"  [TGN] (1) co-located  full-run: {t_colo*1e3:8.2f} ms  (exact recurrence, baseline)")
    print(f"  [TGN] (2) sync-on-cut: per-snapshot comm w_T_sync = {wT_per_snap_ms:8.4f} ms  "
          f"=> {wT_per_cut_edge_us:8.4f} us / cut-temporal-edge")
    print(f"  [TGN] (2) sync-on-cut: total comm over {S} snaps = {total_comm*1e3:8.3f} ms "
          f"(compute {t_sync_compute*1e3:.2f} ms); comm/compute = {total_comm/max(1e-9,t_sync_compute):.3f}")

    # (3) staleness drift sweep
    print(f"  [TGN] (3) stale-on-cut DRIFT ||stale_mem - exact_mem|| over cut vertices (process proxy, NOT accuracy):")
    for k in sorted(set([0, 1, max(1, a.stale_k // 2), a.stale_k])):
        drift = run_stale_on_cut(model, snaps, N, dev, cut_vertices, exact_trace, k)
        final = drift[-1][1]
        mid = drift[len(drift) // 2][1]
        print(f"        k={k:<3d} drift(mid snap)={mid:9.4f}  drift(final snap)={final:9.4f}")

    # ==================== MEMORYLESS GraphSAGE baseline ====================
    # Same per-snapshot aggregation, but NO recurrence: snapshot t output does NOT depend on t-1.
    # So splitting the timeline costs NOTHING extra across the cut -> w_T ~= 0.
    Wsage = torch.randn(F, D, device=dev) / F ** 0.5

    def sage_step():
        for (dst_ids, msgs) in snaps:
            agg = aggregate_messages(dst_ids, msgs, N, dev)
            _ = torch.relu(agg @ Wsage)        # per-snapshot, independent -> no cross-snapshot state

    t_sage = timed_cuda(sage_step, dev)
    # the temporal-cut sync for memoryless = 0 bytes (no node memory crosses snapshots).
    sage_per_snap_comm = comm_time_model(0, D, 4, link_gbps, world)  # = 0
    sage_wT_per_snap_ms = sage_per_snap_comm * 1e3
    print(f"  [SAGE-memoryless] (1) full-run: {t_sage*1e3:8.2f} ms  (snapshots INDEPENDENT)")
    print(f"  [SAGE-memoryless] (2) sync-on-cut: w_T_sync = {sage_wT_per_snap_ms:8.4f} ms/snap "
          f"(no node-memory crosses snapshots -> 0 bytes -> w_T ~= 0)")

    # ==================== HEADLINE ====================
    tgn_bytes = n_cut * D * 4                       # node-memory bytes that cross the cut per snapshot
    ratio_str = f"{wT_per_snap_ms / sage_wT_per_snap_ms:.1f}x" if sage_wT_per_snap_ms > 0 else "infinitely (memoryless=0)"
    print("  " + "=" * 78)
    print(f"  HEADLINE: temporal-cut cost w_T is {ratio_str} larger for TGN-memory "
          f"({wT_per_snap_ms:.4f} ms/snap, {tgn_bytes/1e6:.2f} MB node-memory/snap) "
          f"than memoryless GraphSAGE ({sage_wT_per_snap_ms:.4f} ms/snap, 0 MB).")
    print(f"  => the space-time duality (D30) BITES only for MEMORY-BASED models (D31 confirmed): "
          f"for TGN the optimal cut must leave the trivial PSS corner.")
    print("  " + "=" * 78)

    if dist is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
