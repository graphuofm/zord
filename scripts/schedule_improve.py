#!/usr/bin/env python
"""SCHEDULE-IMPROVE (middle-layer algorithm search): is there a BETTER hardware-agnostic
allocation/scheduling algorithm than the current GREEDY hetero-matched placement for minimizing
the PREDICTED makespan of one memory-bound aggregation step on a HETEROGENEOUS cluster?

This is a PROCESS-only study (time / makespan / feasibility; same graph, same numerical result;
NEVER accuracy). It reuses the CORRECTED incident-edge cost model from scripts/hetero_matched.py:

    per-device agg WORK_k  = sum of node DEGREE for the nodes assigned to device k   (incident edges:
                             a memory-bound gather fetches EVERY incident edge's feature row, incl.
                             cross-part neighbors; a cut edge is gathered by BOTH endpoints' parts).
    predicted time_k       = WORK_k * F * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS / (hbm_bw_k * 1e9) ms
    MAKESPAN               = max_k time_k                              (lower is better)

KEY STRUCTURAL FACT this study exploits: in the incident-edge model a node's work (its degree) is
INDEPENDENT of which device it lands on, and the total work sum_k WORK_k = sum_v deg(v) = 2M is a
PARTITION-INVARIANT CONSTANT. So minimizing makespan is exactly a MIN-MAKESPAN / MULTIWAY
NUMBER-PARTITIONING problem on per-node weights (degree) onto UNRELATED-but-here-uniform machines
with speeds bw_k. The current greedy assigns CONTIGUOUS density-sorted segments and only balances the
running edge target -- contiguous segments cannot place a heavy hub on one device and a light hub on
another, so they leave a balance gap that classic bin-packing heuristics (LPT, Karmarkar-Karp) close.

Algorithms compared (all HARDWARE-AGNOSTIC -- they take an arbitrary ClusterProfile bw/mem vector):
  even            : equal NODE counts (capacity/bandwidth-blind baseline)
  bw-proportional : node counts ~ hbm_bw (bandwidth-aware, degree-blind)
  greedy hetero   : the CURRENT algorithm -- dense core -> strong device, contiguous density-sorted
                    segments whose boundaries balance the running edge target (== solve_balanced_bounds)
  LPT             : longest-processing-time-first -- sort nodes by work (degree) descending, assign
                    each to the device with the least PROJECTED FINISH TIME (work/bw), bw-scaled.
  Karmarkar-Karp  : multiway differencing on bw-weighted per-node work (greedy set-differencing).
  LPT-capacity    : LPT but RESPECTING per-device usable_mem -- a full device is skipped (overflow to
                    the next least-loaded feasible device) so the plan stays FEASIBLE.
  greedy+local    : the current greedy, then a LOCAL-SEARCH refinement that moves boundary nodes
                    between adjacent device segments to reduce makespan (shows greedy's own headroom).

HEADLINE reported at the end: does ANY algorithm beat the current greedy's makespan, and by how much
(speedup vs even AND vs greedy)? That quantifies the middle-layer algorithmic room.

  python scripts/schedule_improve.py --nodes 2000000 --edges 25000000 --comms 2000 --feat 128
  python scripts/schedule_improve.py --dataset jodie-reddit --feat 128
"""
import argparse
import os
import struct
import subprocess
import time

import numpy as np

BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")
BYTES_PER_EDGE_TRAVERSAL = 4.0   # fp32 feature word moved per edge per gather (memory-bound model)
N_GATHERS = 2                    # 2-layer aggregation does 2 SpMM gathers over the incident edges


# ---------------------------------------------------------------------------
# graph generation / IO / density ranking (mirrors hetero_matched.py exactly)
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


# ---------------------------------------------------------------------------
# cost model (CORRECTED incident-edge work) -- evaluated on a generic node->device map
# ---------------------------------------------------------------------------
def predict_times_from_work(work, bw_gbps, F):
    """Predicted per-device agg ms from incident-edge gather WORK (sum of degrees on the device)."""
    bytes_moved = np.asarray(work, dtype=np.float64) * F * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS
    return bytes_moved / (np.asarray(bw_gbps, dtype=np.float64) * 1e9) * 1e3   # ms


def eval_assignment(part, src, dst, deg, D, bw, F):
    """Evaluate ANY node->device assignment `part` (len N, values in [0,D)) under the corrected
    incident-edge model. Returns (counts, work[=incident degree], local_edges, cut, times_ms).
    Work and times here are the SAME quantity the makespan is taken over."""
    part = np.asarray(part, dtype=np.int64)
    counts = np.bincount(part, minlength=D).astype(np.int64)
    work = np.bincount(part, weights=deg.astype(np.float64), minlength=D).astype(np.int64)  # incident
    pu = part[src]; pv = part[dst]
    local_mask = pu == pv
    local_edges = np.bincount(pu[local_mask], minlength=D).astype(np.int64)
    cut = int(np.count_nonzero(~local_mask))
    times = predict_times_from_work(work, bw, F)
    return counts, work, local_edges, cut, times


def report(label, counts, work, local_edges, cut, times_ms, devs, M, extra=""):
    nnz = 2 * M
    print(f"  [{label}]{(' ' + extra) if extra else ''}")
    for k, d in enumerate(devs):
        print(f"      dev{k} {d.name:<16} nodes={counts[k]:>12,d}  incident_work={work[k]:>13,d}  "
              f"local_edges={local_edges[k]:>12,d}  hbm={d.hbm_bw_gbps:6.0f}GB/s  "
              f"pred_agg={times_ms[k]:8.3f}ms")
    makespan = float(times_ms.max())
    busy = float(times_ms.mean())
    util = busy / makespan * 100 if makespan > 0 else 100.0
    print(f"      cut={cut:>12,d} ({cut/nnz*100:5.1f}% edges)  makespan={makespan:8.3f}ms  "
          f"util={util:5.1f}%  (lower makespan is better)")
    return makespan


# ---------------------------------------------------------------------------
# BASELINE / GREEDY assignments expressed as CONTIGUOUS density-sorted segments
# (rank order; rank 0 = densest). bounds[len D+1] over rank index -> device-sorted-by-bw.
# ---------------------------------------------------------------------------
def segments_to_part(rank, bounds_sorted, order_strong, N, D):
    """Map contiguous rank-boundaries (in bw-sorted device order) back to PHYSICAL device ids,
    returning a per-node device assignment `part` (len N)."""
    seg = np.searchsorted(bounds_sorted, rank, side="right") - 1
    seg = seg.clip(0, D - 1).astype(np.int64)         # which sorted slot (0 = strongest)
    part = order_strong[seg].astype(np.int64)          # sorted slot -> physical device id
    return part


def even_bounds(N, D):
    return np.linspace(0, N, D + 1).astype(np.int64)


def bw_proportional_bounds(N, D, bw_sorted):
    frac = bw_sorted / bw_sorted.sum()
    b = np.concatenate([[0], np.cumsum((frac * N).astype(np.int64))])
    b[-1] = N
    return np.maximum.accumulate(b).astype(np.int64)


def greedy_balanced_bounds(deg_cum, bw_sorted, usable_sorted, N):
    """The CURRENT algorithm (== hetero_matched.solve_balanced_bounds): give each device an edge
    budget ~ its bw so edges_k/bw_k is ~constant, translate the running edge budget into a
    CONTIGUOUS node boundary over the density-sorted layout, with a per-device capacity clamp."""
    D = len(bw_sorted)
    share = bw_sorted / bw_sorted.sum()
    target_edges = share * deg_cum[-1]
    bounds = [0]
    acc = 0.0
    for k in range(D - 1):
        acc += target_edges[k]
        nb = int(np.searchsorted(deg_cum, acc, side="left"))
        nb = max(bounds[-1] + 1, min(nb, N))
        nb = min(nb, bounds[-1] + max(1, int(usable_sorted[k])))
        bounds.append(nb)
    bounds.append(N)
    return np.array(bounds, dtype=np.int64)


# ---------------------------------------------------------------------------
# shared closed-form water-fill: assign equal-degree BUCKETS of nodes to devices, keeping projected
# finish times level (== item-by-item LPT for equal-weight items). Used by LPT and by KK's light tail.
# ---------------------------------------------------------------------------
def _waterfill_buckets(node_order, work_sorted, bw, D, part, finish, load_nodes, cap):
    """Process `node_order` (heaviest work first) in equal-work buckets; for each bucket distribute
    its nodes across devices with a CLOSED-FORM water-fill on finish_k + n_k*work/bw_k. Writes device
    ids into `part` and updates `finish`/`load_nodes` in place. O(buckets * D log D), no per-item loop.
    A device at capacity (load_nodes[k] >= cap[k]) is skipped; residue overflows to the most-slack one."""
    bw = np.asarray(bw, dtype=np.float64)
    inv_bw = 1.0 / bw
    ns = node_order.size
    i = 0
    while i < ns:
        j = i + 1
        w = work_sorted[i]
        while j < ns and work_sorted[j] == w:
            j += 1
        cnt = j - i
        idx_block = node_order[i:j]
        room = np.where(np.isfinite(cap), np.maximum(0, cap - load_nodes), cnt).astype(np.int64)
        if w == 0.0:
            # zero-work nodes don't change finish; spread to balance node COUNTS (water-fill on load)
            ncnt = _waterfill_counts(cnt, 1.0, load_nodes.astype(np.float64), room, bw, D)
        else:
            ncnt = _waterfill_counts(cnt, w, finish, room, bw, D)
        assigned = int(ncnt.sum())
        if assigned < cnt:
            slack = np.where(np.isfinite(cap), cap - load_nodes, np.inf)
            ncnt[int(np.argmax(slack))] += cnt - assigned
        off = 0
        for k in range(D):
            c = int(ncnt[k])
            if c:
                part[idx_block[off:off + c]] = k
                off += c
        if w != 0.0:
            finish += ncnt * w * inv_bw
        load_nodes += ncnt
        i = j


def _waterfill_counts(cnt, w, finish, room, bw, D):
    """Hand `cnt` items of work `w` to D devices to greedily equalize finish_k + n_k*w/bw_k (== LPT for
    EQUAL-weight items). Closed-form continuous water-fill then integer rounding. room[k] = items device
    k may still accept; returns integer n_k with sum == min(cnt, sum(room)). O(D log D), no item loop."""
    n = np.zeros(D, dtype=np.int64)
    if cnt <= 0:
        return n
    give_cnt = min(cnt, int(room.sum()))
    if give_cnt <= 0:
        return n
    rate = bw / w                                    # items per unit waterline rise
    cap_items = room.astype(np.float64)

    def volume(L):
        x = np.clip((L - finish) * rate, 0.0, cap_items)
        return x.sum(), x

    # binary search the waterline L where volume(L) == give_cnt (volume monotone increasing in L)
    lo = float(finish.min())
    hi = float(finish.max()) + give_cnt * w / float(bw.min()) + 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        v, _ = volume(mid)
        if v < give_cnt:
            lo = mid
        else:
            hi = mid
    _, x = volume(hi)
    n = np.minimum(np.floor(x).astype(np.int64), room)
    short = give_cnt - int(n.sum())
    if short > 0:
        frac = np.where(n < room, x - np.floor(x), -1.0)
        for d in np.argsort(-frac):
            if short <= 0:
                break
            if n[d] < room[d]:
                n[d] += 1; short -= 1
    return n


# ---------------------------------------------------------------------------
# NEW algorithm 1: LPT (longest-processing-time-first) makespan minimization
# ---------------------------------------------------------------------------
def assign_lpt(deg, bw, D, N, usable_nodes=None):
    """Sort nodes by work (degree) DESCENDING; assign each to the device that would FINISH it
    earliest, i.e. minimize projected finish time (accumulated_work_k + this_work) / bw_k. This is
    the classic LPT heuristic for makespan on uniform machines (here machine speed = hbm_bw_k). It
    is bandwidth-aware AND degree-aware and is free to scatter heavy hubs across devices, which the
    contiguous greedy cannot. If `usable_nodes` is given, a device at capacity is skipped (overflow).

    Implementation: nodes are bucketed by equal degree (work) and processed heaviest-bucket first.
    For one bucket of `cnt` identical-work items added on top of the current per-device finish times,
    the LPT/least-finish-first rule, applied item by item, performs a WATER-FILL: it keeps handing the
    next item to whichever device currently finishes earliest. We compute that distribution in CLOSED
    FORM (no per-item loop): find the waterline L such that the integer items handed to each device,
    n_k = round_down((L - finish_k) * bw_k / w), sum to `cnt`, then assign that many of the bucket's
    nodes to each device. This is exact LPT for equal-weight items and is O(buckets * D log D).
    Capacity (`usable_nodes`) is respected by removing a device from the fill once it is full."""
    bw = np.asarray(bw, dtype=np.float64)
    part = np.empty(N, dtype=np.int64)
    order = np.argsort(-deg.astype(np.int64), kind="stable")     # heaviest work first
    deg_sorted = deg[order].astype(np.float64)
    finish = np.zeros(D, dtype=np.float64)          # projected finish time per device (ms-equivalent)
    load_nodes = np.zeros(D, dtype=np.int64)
    cap = (np.asarray(usable_nodes, dtype=np.float64) if usable_nodes is not None
           else np.full(D, np.inf))
    _waterfill_buckets(order, deg_sorted, bw, D, part, finish, load_nodes, cap)
    return part


# ---------------------------------------------------------------------------
# NEW algorithm 2: Karmarkar-Karp style multiway differencing on bw-weighted work
# ---------------------------------------------------------------------------
def assign_karmarkar_karp(deg, bw, D, N, max_items=20_000):
    """Karmarkar-Karp / largest-differencing multiway number partitioning, made HETEROGENEITY-aware.
    The D groups ARE the D devices (group k has speed bw[k]); each merge pairs the partition with the
    largest TIME-spread with another so their heaviest TIME-loads fall on OPPOSITE devices, driving the
    per-device TIMES (= raw work_k / bw_k) together. This is KK differencing on per-node work WEIGHTED
    by 1/bw, which is exactly what minimizes makespan. The long light tail (below the `max_items`
    heaviest nodes) is folded in by the shared closed-form water-fill on the running finish times.

    EFFICIENT label tracking: a head node keeps (label_id, local_device); each merge records only how
    the swallowed label's devices map into the survivor's device space (a length-D permutation) + a
    parent pointer. Final device per node is resolved by composing the permutations along each node's
    chain once -- O(n_head * D), no growing per-merge copies (naive member-array rewrite is quadratic).
    Because the device identities are fixed (group k == physical device k with bw[k]), the permutations
    relabel which device each partial group represents; the survivor keeps device identity 0..D-1."""
    bw = np.asarray(bw, dtype=np.float64)
    inv_bw = 1.0 / bw
    part = np.full(N, -1, dtype=np.int64)
    order = np.argsort(-deg.astype(np.int64), kind="stable")
    deg_sorted = deg[order].astype(np.float64)

    # ADAPTIVE head sizing. KK exists to place the HEAVY HUBS that a contiguous/greedy split can't
    # balance; the long remainder is handled by the exact (add-only) water-fill tail. The tail can only
    # ADD work to under-loaded devices, so the KK head must NEVER push any device above the balanced
    # finish line -- otherwise the over-committed device fixes the makespan. We therefore cap the head's
    # CUMULATIVE work at the SLOWEST device's optimal work share, total_work * min(bw)/sum(bw): below
    # that bound even the slowest device's head load stays under the lower-bound makespan, so the tail
    # water-fill can always reach balance. On near-uniform graphs this bound is hit after very few
    # nodes -> KK collapses to the optimal water-fill (== LPT); on heavy-tailed graphs it still captures
    # the genuine hubs. Capped further by max_items for the pure-Python merge loop.
    total_work = float(deg_sorted.sum())
    head_work_cap = total_work * (bw.min() / bw.sum())
    cumw = np.cumsum(deg_sorted)
    n_by_work = int(np.searchsorted(cumw, head_work_cap, side="right"))
    n_head = int(min(max_items, max(D, n_by_work)))               # at least D so every device gets a hub
    n_head = min(n_head, N)
    head_nodes = order[:n_head]
    head_w = deg_sorted[:n_head]
    import heapq
    n_labels_max = 2 * n_head
    parent = np.full(n_labels_max, -1, dtype=np.int64)            # label -> surviving merged label id
    dmap_to_parent = np.zeros((n_labels_max, D), dtype=np.int64)  # this label's slot -> parent's slot
    node_label = np.arange(n_head, dtype=np.int64)                # head node t starts in label t
    node_slot = np.zeros(n_head, dtype=np.int64)                  # local device-slot 0 at the leaf

    # A label's state is the per-slot RAW work it has accumulated; slot s currently maps (via the merge
    # chain) to some physical device, whose speed we know once resolved. To difference on TIME during
    # merges, the survivor A keeps slots aligned to physical devices 0..D-1 (so slot s == device s);
    # the swallowed B is relabeled so its heavy slots land on A's light devices.
    labels = []   # heap entries: (-time_spread, label_id, work_vec(np float64) per physical device slot)
    for t in range(n_head):
        wv = np.zeros(D, dtype=np.float64)
        wv[0] = head_w[t]                                         # singleton goes on slot 0 (device 0)
        tv = wv * inv_bw
        heapq.heappush(labels, (-(tv.max() - tv.min()), t, wv))
    next_id = n_head

    while len(labels) > 1:
        _, idA, wvA = heapq.heappop(labels)
        _, idB, wvB = heapq.heappop(labels)
        tvA = wvA * inv_bw; tvB = wvB * inv_bw
        # differencing on TIME: A's slot with the LARGEST time paired with B's slot with the SMALLEST.
        permA = np.argsort(-tvA, kind="stable")                  # A slots: largest->smallest time
        permB = np.argsort(tvB, kind="stable")                   # B slots: smallest->largest time
        mid = next_id; next_id += 1
        idA_map = np.arange(D, dtype=np.int64)                   # A slots keep their physical identity
        idB_map = np.empty(D, dtype=np.int64)
        idB_map[permB] = permA                                   # B slot permB[r] folds onto A slot permA[r]
        parent[idA] = mid; dmap_to_parent[idA] = idA_map
        parent[idB] = mid; dmap_to_parent[idB] = idB_map
        wv = wvA.copy()
        np.add.at(wv, permA, wvB[permB])
        tv = wv * inv_bw
        heapq.heappush(labels, (-(tv.max() - tv.min()), mid, wv))

    _, root, work_final = labels[0]                              # raw work per physical device after head
    cur_label = node_label.copy(); cur_slot = node_slot.copy()
    # walk leaf -> root, composing each label's slot->parent map. STOP a node BEFORE its label becomes
    # the root: the root has no parent and its dmap is the uninitialized identity-less zero row, so
    # applying it would collapse every node to slot 0 (the resolution bug we must avoid).
    active = parent[cur_label] != -1
    while active.any():
        idx = np.where(active)[0]
        labs = cur_label[idx]
        cur_slot[idx] = dmap_to_parent[labs, cur_slot[idx]]      # this label's slot -> parent's slot
        cur_label[idx] = parent[labs]                            # ascend
        active[idx] = parent[cur_label[idx]] != -1               # keep going only while a parent remains
    part[head_nodes] = cur_slot                                  # slot == physical device id
    finish = work_final * inv_bw                                 # time-equivalent finish per device

    # ---- fold in the LIGHT TAIL with the shared closed-form water-fill (no per-item loop) ----
    tail = order[n_head:]
    if tail.size:
        load_nodes = np.zeros(D, dtype=np.int64)
        np.add.at(load_nodes, cur_slot, 1)                       # head node counts per device
        _waterfill_buckets(tail, deg_sorted[n_head:], bw, D, part, finish, load_nodes,
                           np.full(D, np.inf))
    miss = np.where(part < 0)[0]
    if miss.size:
        part[miss] = int(np.argmin(finish))
    return part


# ---------------------------------------------------------------------------
# NEW algorithm 4 (optional): LOCAL-SEARCH refinement on top of the greedy segments
# ---------------------------------------------------------------------------
def local_search_refine(part, deg, bw, D, F, max_iters=20000):
    """Move single boundary/any nodes between devices to reduce the makespan. Greedy hill-climb:
    each round take the busiest device (the makespan owner) and move its CHEAPEST-to-relocate node
    (smallest degree) to the device that minimizes the new max time -- accept only if the global
    makespan strictly improves. Pure makespan objective; hardware-agnostic. Operates on the corrected
    incident-edge work, which is the SAME quantity makespan is taken over (so moves are exact)."""
    part = part.copy()
    bw = np.asarray(bw, dtype=np.float64)
    inv_bw = 1.0 / bw
    work = np.bincount(part, weights=deg.astype(np.float64), minlength=D)  # incident work per device
    unit = F * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS / 1e9 * 1e3            # work-units -> ms factor

    def times(w):
        return w * unit * inv_bw

    # bucket node ids by device for cheap "smallest-degree node on the busy device" lookup
    for _ in range(max_iters):
        t = times(work)
        hot = int(np.argmax(t))
        mk = t[hot]
        # candidate nodes on the hot device, lightest first (cheapest to move, least disturb others)
        on_hot = np.where(part == hot)[0]
        if on_hot.size <= 1:
            break
        # try the K lightest hot-device nodes (degree ascending) -- small, fast set
        cand = on_hot[np.argsort(deg[on_hot], kind="stable")[:32]]
        best_gain = 0.0; best_mv = None
        for v in cand:
            dv = deg[v]
            for tgt in range(D):
                if tgt == hot:
                    continue
                new_hot = (work[hot] - dv) * unit * inv_bw[hot]
                new_tgt = (work[tgt] + dv) * unit * inv_bw[tgt]
                # new makespan considers all devices but only hot/tgt changed
                other = t.copy(); other[hot] = new_hot; other[tgt] = new_tgt
                new_mk = other.max()
                gain = mk - new_mk
                if gain > best_gain + 1e-12:
                    best_gain = gain; best_mv = (int(v), tgt)
        if best_mv is None:
            break
        v, tgt = best_mv
        work[hot] -= deg[v]; work[tgt] += deg[v]; part[v] = tgt
    return part


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=2_000_000)
    ap.add_argument("--edges", type=int, default=25_000_000)
    ap.add_argument("--comms", type=int, default=2000)
    ap.add_argument("--intra", type=float, default=0.9)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--rank-by", default="degree", choices=["degree", "kcore"])
    ap.add_argument("--dataset", default="")
    ap.add_argument("--devices", default="",
                    help="comma device spec h100/6000/5000 counts, e.g. '1,1,1' (default 1,1,1). "
                         "Hardware-agnostic: any ClusterProfile bw/mem vector is accepted.")
    ap.add_argument("--no-local", action="store_true", help="skip the local-search refinement")
    ap.add_argument("--kk-head", type=int, default=20_000,
                    help="how many heaviest nodes Karmarkar-Karp treats individually (light tail "
                         "water-filled); the head loop is pure-Python so keep it modest")
    a = ap.parse_args()
    F = a.feat

    from zord.profiler.cluster_profile import hetcluster
    if a.devices:
        try:
            nh, n6, n5 = (int(x) for x in a.devices.split(","))
            cluster = hetcluster(num_h100=nh, num_6000ada=n6, num_5000ada=n5)
        except Exception as e:
            print(f"  bad --devices '{a.devices}' ({e}); falling back to 1,1,1"); cluster = hetcluster()
    else:
        cluster = hetcluster()
    devs = cluster.devices
    D = len(devs)
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    usable_nodes = np.array([d.usable_mem / (F * 4) for d in devs], dtype=np.float64)

    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int32); dst = g.dst.astype(np.int32); M = src.size
        print(f"SCHED dataset={g.name} N={N} M={M} F={F} D={D} rank-by={a.rank_by} bin={BIN}")
    else:
        N, M = a.nodes, a.edges
        print(f"SCHED SYNTHETIC N={N} M={M} comms={a.comms} intra={a.intra} F={F} D={D} "
              f"rank-by={a.rank_by} bin={BIN}")
        src, dst = gen_graph(N, M, a.comms, a.intra)
    print("  devices: " + " | ".join(
        f"{d.name} bw={d.hbm_bw_gbps:.0f}GB/s mem={d.usable_mem/1024**3:.0f}GB r={d.throughput:.2f}"
        for d in devs))
    print(f"  loaded/generated graph in {time.time()-t0:.1f}s")

    edges_path = "/tmp/zord_si_edges.bin"; write_edges(edges_path, N, src, dst)

    crank, cost = cpp_order(edges_path, a.rank_by, f"/tmp/zord_si_perm_{a.rank_by}.bin")
    if crank is None:
        print("  ABORT: density ranking failed."); return
    rank = crank.astype(np.int64)
    if a.rank_by == "kcore":
        rank = (N - 1) - rank
    print(f"  density ranking ({a.rank_by}) computed in {cost:.2f}s (C++)")

    deg = node_degree(src, dst, N)
    deg_by_rank = np.empty(N, dtype=np.float64)
    deg_by_rank[rank] = deg.astype(np.float64)
    deg_cum = np.cumsum(deg_by_rank)

    order_strong = np.argsort(-bw)        # physical device ids, strongest (highest bw) first
    bw_sorted = bw[order_strong]
    usable_sorted = usable_nodes[order_strong]

    total_work = float(deg.sum())         # == 2M; PARTITION-INVARIANT
    lb_ms = total_work * F * BYTES_PER_EDGE_TRAVERSAL * N_GATHERS / (bw.sum() * 1e9) * 1e3
    print(f"  total incident work (sum deg) = {int(total_work):,d} (= 2M, partition-invariant); "
          f"perfect-balance makespan LOWER BOUND = {lb_ms:.3f}ms")

    results = {}   # label -> (makespan, timing_s)

    def run(label, part, extra=""):
        t = time.time()
        counts, work, le, cut, times = eval_assignment(part, src, dst, deg, D, bw, F)
        mk = report(label, counts, work, le, cut, times, devs, M, extra=extra)
        results[label] = mk
        return mk, part

    # ---- baselines / current greedy (contiguous density segments) ----
    tg = time.time()
    eb = even_bounds(N, D)
    part_even = segments_to_part(rank, eb, order_strong, N, D)
    run("even (equal nodes)", part_even)

    bpb = bw_proportional_bounds(N, D, bw_sorted)
    part_bp = segments_to_part(rank, bpb, order_strong, N, D)
    run("bw-proportional (nodes ~ hbm_bw)", part_bp)

    t_greedy = time.time()
    gb = greedy_balanced_bounds(deg_cum, bw_sorted, usable_sorted, N)
    part_greedy = segments_to_part(rank, gb, order_strong, N, D)
    run("GREEDY hetero-matched (CURRENT: contiguous, time-balanced)", part_greedy,
        extra=f"[built in {time.time()-t_greedy:.3f}s]")

    # ---- NEW: LPT ----
    t = time.time()
    part_lpt = assign_lpt(deg, bw, D, N, usable_nodes=None)
    mk_lpt, _ = run("LPT (longest-processing-time, bw-scaled finish)", part_lpt,
                    extra=f"[built in {time.time()-t:.3f}s]")

    # ---- NEW: Karmarkar-Karp ----
    t = time.time()
    part_kk = assign_karmarkar_karp(deg, bw, D, N, max_items=a.kk_head)
    mk_kk, _ = run("Karmarkar-Karp (multiway differencing, time-weighted)", part_kk,
                   extra=f"[built in {time.time()-t:.3f}s]")

    # ---- NEW: LPT capacity-constrained (feasibility) ----
    t = time.time()
    part_lptc = assign_lpt(deg, bw, D, N, usable_nodes=usable_nodes)
    mk_lptc, _ = run("LPT-capacity (respect usable_mem, overflow)", part_lptc,
                     extra=f"[built in {time.time()-t:.3f}s]")

    # ---- NEW: greedy + local-search refinement ----
    mk_ls = None
    if not a.no_local:
        t = time.time()
        part_ls = local_search_refine(part_greedy, deg, bw, D, F)
        mk_ls, _ = run("greedy+local-search (boundary moves)", part_ls,
                       extra=f"[refined in {time.time()-t:.3f}s]")

    # ----------------------------- HEADLINE -----------------------------
    mk_even = results["even (equal nodes)"]
    mk_greedy = results["GREEDY hetero-matched (CURRENT: contiguous, time-balanced)"]
    print(f"\n  total scheduling search done in {time.time()-tg:.2f}s")
    print(f"  perfect-balance LOWER BOUND = {lb_ms:.3f}ms; CURRENT greedy = {mk_greedy:.3f}ms "
          f"(greedy is {mk_greedy/lb_ms:.2f}x the lower bound -> headroom = {(mk_greedy/lb_ms-1)*100:.1f}%)")
    print("\n  ===== RANKING (lower makespan = better) =====")
    ranked = sorted(results.items(), key=lambda kv: kv[1])
    for label, mk in ranked:
        vs_even = mk_even / mk if mk > 0 else float("inf")
        vs_greedy = mk_greedy / mk if mk > 0 else float("inf")
        flag = "  <-- CURRENT" if label.startswith("GREEDY") else ""
        print(f"    {mk:8.3f}ms  {label:<54}  {vs_even:5.2f}x vs even  {vs_greedy:5.2f}x vs greedy{flag}")

    # best NON-greedy alternative; a "real" win must clear a meaningful relative margin (>0.5%),
    # not float-rounding noise (LPT and greedy can both touch the lower bound to within microseconds).
    REL = 5e-3
    alts = [(lab, mk) for lab, mk in ranked if not lab.startswith("GREEDY")]
    best_label, best_mk = alts[0]
    beats = best_mk < mk_greedy * (1.0 - REL)
    print("\n  ===== HEADLINE =====")
    if beats:
        impr = (mk_greedy - best_mk) / mk_greedy * 100
        print(f"  YES -- there IS middle-layer algorithmic room beyond the current greedy.")
        print(f"  WINNER: '{best_label}' at {best_mk:.3f}ms BEATS greedy {mk_greedy:.3f}ms by "
              f"{impr:.1f}% ({mk_greedy/best_mk:.3f}x faster makespan), and is "
              f"{mk_even/best_mk:.2f}x vs even.")
        print(f"  Remaining gap to the perfect-balance lower bound ({lb_ms:.3f}ms): "
              f"{(best_mk/lb_ms-1)*100:.1f}%.")
    else:
        print(f"  NO meaningful room -- the current greedy ({mk_greedy:.3f}ms) TIES the best "
              f"alternative ('{best_label}' {best_mk:.3f}ms, within {REL*100:.1f}%). On this graph "
              f"contiguous density-segment balancing already reaches the {lb_ms:.3f}ms perfect-balance "
              f"lower bound (greedy is {mk_greedy/lb_ms:.3f}x LB), so the middle layer is "
              f"makespan-optimal here; no scheduling algorithm can do materially better.")
        print(f"  NOTE: greedy/LPT BOTH hit the lower bound when node degrees are well-spread. The "
              f"algorithmic room appears on SKEWED-degree graphs where heavy hubs cannot be split by "
              f"contiguous segments -- try a heavier-tailed --comms/--intra, more --devices imbalance, "
              f"or a real --dataset.")


if __name__ == "__main__":
    main()
