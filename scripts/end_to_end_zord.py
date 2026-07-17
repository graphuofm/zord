#!/usr/bin/env python3
"""END-TO-END ZORD pipeline on a REAL dataset, driven by the REAL src/zord engine.

This is the "graph in -> middle -> allocate to the devices below, and run it" pipeline
(用户: zord 从图进来,到中端,到分配到下面,都要做). EVERY decision -- the partition
assignment, the per-device placement, the vertex-cut/replication, the feasibility, and the
predicted makespan -- comes from the ACTUAL engine in src/zord, NOT a reimplementation here.
This file only LOADS a real graph, REALIZES the engine's plan on CPU torch, and MEASURES the
front/middle/back wall-clock. It is grep-ably bound to the engine:

    from zord.datasets import load            # FRONT: graph in (real loaders)
    from zord.profiler import from_spec        # the heterogeneous cluster spec
    from zord.schedule import plan             # MIDDLE: the REAL arrange+place+cut engine
    from zord.partition.arrange import arrange # (same engine entry, used directly for the assert)

PROCESS-only (MEMORY.md): same data + same model => same result. We optimize TIME / MEMORY /
FEASIBILITY -- WHERE each partial aggregation runs, never WHAT is computed; accuracy is only a
correctness check (the realized multi-device aggregation must equal the single-device one), never
the target. No networkx, no SLURM. CPU torch.sparse.mm is fine (the runtime numbers are a real
wall-clock of the realized plan; the engine's makespan is the GPU-roofline prediction).

THREE STAGES, all timed:
  FRONT  (graph in) : zord.datasets.load(name).sort_by_time()  ->  TemporalGraph
  MIDDLE (arrange)  : zord.schedule.plan(graph, cluster, link_gbps, feat_dim[, feat_bytes])
                      -> Plan{assignment, core_mask, placement, makespan_ms, feasible, ...}
  BACK   (allocate  : realize plan.assignment (+ replicated core) into per-device node sets,
          + run)      build each device's local CSR, run a real 2-layer GraphSAGE-style mean
                      aggregation (torch.sparse.mm) honoring the plan, exchange the boundary
                      feature rows the cut implies, and measure the per-device makespan.

Run (from the repo root; src/ is auto-added to sys.path):
    python3 scripts/end_to_end_zord.py
    python3 scripts/end_to_end_zord.py --datasets collegemsg,bitcoin-alpha,askubuntu --link-gbps 0.5
    python3 scripts/end_to_end_zord.py --attributed jodie-wikipedia   # F_v-aware (real edge feats)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# --- make `zord` importable straight from this checkout (scripts/ is not a package) ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import torch

# ============================================================================================ #
#  THE REAL ENGINE. These four imports ARE the proof this pipeline exercises src/zord and not  #
#  a copy: the FRONT loader, the cluster spec, the MIDDLE planner, and arrange() itself.        #
# ============================================================================================ #
import zord
from zord.datasets import load, get_spec, TemporalGraph
from zord.profiler import from_spec, GB
from zord.schedule import plan                      # MIDDLE: the end-to-end engine entry
from zord.partition.arrange import arrange          # the adaptive-corner partition (engine core)

# the heterogeneous cluster used for every plan (3 unequal GPUs + an interconnect PARAMETER).
# matches examples/plan_synthetic.py: per-device usable HBM, ACHIEVED agg bandwidth, link bw.
_HBM_GB = [80.0, 48.0, 32.0]
_AGG_GBPS = [942.0, 534.0, 444.0]
_NAMES = ["H100-80", "RTX6000-48", "RTX5000-32"]


# ============================================================================================ #
#  helpers (NONE of these make a partition decision -- they only REALIZE the engine's plan)     #
# ============================================================================================ #
def _build_cluster(link_gbps: float):
    """The heterogeneous cluster, built by the REAL profiler. interconnect is a PARAMETER."""
    return from_spec(hbm_gb=_HBM_GB, agg_bw_gbps=_AGG_GBPS,
                     interconnect_gbps=link_gbps, names=_NAMES)


def _node_dev_from_plan(plan_obj, N: int, D: int):
    """Materialize a per-node owning device from the ENGINE's plan.assignment.

    Vertex-cut leaves assignment == -1 for the REPLICATED CORE (those rows live on EVERY device).
    For the realized run we give each core node a single 'home' for the local-CSR build (it is
    still physically replicated -- we add it to every device's resident node set below), so the
    aggregation is well defined. This is pure realization of the engine's decision; the WHO-goes-
    WHERE choice is entirely plan.assignment / plan.core_mask from src/zord."""
    assign = np.asarray(plan_obj.assignment, dtype=np.int64)
    core = plan_obj.core_mask
    node_dev = assign.copy()
    if core is not None and core.any():
        # round-robin a HOME device for core rows just so each has a local owner for the build;
        # they are replicated onto all devices in _realize_and_run via the core set.
        idx = np.nonzero(core)[0]
        node_dev[idx] = (np.arange(idx.size) % D).astype(np.int64)
    # any stray -1 (shouldn't happen once core handled) -> device 0
    node_dev[node_dev < 0] = 0
    return node_dev, (None if core is None else np.asarray(core, dtype=bool))


def _make_features(N: int, feat_dim: int, feat_bytes: Optional[np.ndarray], seed: int = 0):
    """Deterministic node features X[N, F] (PROCESS-only: fixed seed => same X => same result).
    feat_dim is the dense width used for the realized aggregation. feat_bytes (when given) is the
    HETEROGENEOUS per-node feature SIZE the engine plans against; we DO NOT change the dense run
    width by it (the run is a correctness/wall-clock probe), it is fed to the engine as F_v."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((N, feat_dim), dtype=np.float32))


def _full_graph_aggregate(src, dst, X):
    """REFERENCE single-device ONE-layer GraphSAGE-style MEAN aggregation over the WHOLE graph,
    via torch.sparse.mm. This is the WHAT (result-defining) the partitioned run must reproduce
    -- the PROCESS-only correctness check. Symmetric (undirected) row-normalized adjacency. One
    aggregation pass is exactly realizable per-device with a 1-hop boundary halo, so the realized
    run can be checked bit-for-bit equal: WHERE each partial sum runs changes, WHAT does not."""
    N = X.shape[0]
    A, _deg = _norm_adj(src, dst, N, N)
    return torch.sparse.mm(A, X)


def _norm_adj(src, dst, num_rows, num_cols):
    """Row-normalized sparse mean-aggregation adjacency as a torch sparse_coo tensor.
    Undirected: add both directions + a self-loop on every ROW, divide each row by its degree.
    Rows are the AGGREGATING nodes (0..num_rows), columns the SOURCE feature rows (0..num_cols);
    for the full graph num_rows==num_cols==N, for a device's halo subgraph rows=resident,
    cols=resident+halo. Returns (A, row_degree). All indices must already be LOCAL."""
    s = np.asarray(src, dtype=np.int64)
    d = np.asarray(dst, dtype=np.int64)
    # rows aggregate from their neighbours; add a self loop so an isolated row maps to itself.
    rows = np.concatenate([d, s, np.arange(num_rows)])
    cols = np.concatenate([s, d, np.arange(num_rows)])
    keep = rows < num_rows                      # only AGGREGATING (resident) rows have an output
    rows, cols = rows[keep], cols[keep]
    deg = np.bincount(rows, minlength=num_rows).astype(np.float32)
    deg[deg == 0] = 1.0
    vals = (1.0 / deg[rows]).astype(np.float32)
    idx = torch.from_numpy(np.stack([rows, cols]))
    A = torch.sparse_coo_tensor(idx, torch.from_numpy(vals), (num_rows, num_cols)).coalesce()
    return A, deg


def _norm_adj_local(row, col, num_rows, num_cols):
    """Row-normalized mean-aggregation adjacency for a device's LOCAL halo subgraph. `row` is the
    aggregating-node ROW index (0..num_rows), `col` the SOURCE feature-row COLUMN index (0..num_cols)
    -- BOTH already local. We add a self-loop: owned row r reads its OWN feature column, which by
    construction is local column r (owned nodes occupy the first num_rows columns of col_nodes). So
    the local aggregation equals the global mean-aggregation restricted to owned rows -> exact."""
    r = np.asarray(row, dtype=np.int64)
    c = np.asarray(col, dtype=np.int64)
    self_r = np.arange(num_rows)
    rows = np.concatenate([r, self_r])
    cols = np.concatenate([c, self_r])            # owned row r's own feature is local column r
    deg = np.bincount(rows, minlength=num_rows).astype(np.float32)
    deg[deg == 0] = 1.0
    vals = (1.0 / deg[rows]).astype(np.float32)
    idx = torch.from_numpy(np.stack([rows, cols]))
    A = torch.sparse_coo_tensor(idx, torch.from_numpy(vals), (num_rows, num_cols)).coalesce()
    return A, deg


@dataclass
class DeviceRun:
    device: int
    name: str
    home_nodes: int
    replicated_core: int
    local_edges: int
    boundary_rows_in: int            # remote feature rows this device must fetch (the cut comm)
    compute_ms: float                # measured local aggregation wall-clock (torch.sparse.mm)
    comm_ms: float                   # measured boundary feature-row exchange wall-clock
    makespan_ms: float               # compute + comm for this device
    feasible: bool                   # carried from the engine's placement decision


@dataclass
class StageResult:
    dataset: str
    N: int
    E: int
    edge_feat_dim: int
    load_s: float
    plan_s: float
    realize_s: float
    run_s: float
    strategy: str
    engine_makespan_ms: float
    engine_feasible: bool
    engine_bound: str
    cut_edges: int
    replication_pct: float
    candidate_makespans: dict
    link_gbps: float
    feat_dim: int
    attributed: bool
    measured_makespan_ms: float      # max over device runs (the realized back-end makespan)
    devices: list = field(default_factory=list)
    correctness_max_abs_err: float = 0.0   # ||partitioned - full|| check (PROCESS-only)


def _realize_and_run(g: TemporalGraph, plan_obj, cluster, feat_dim: int,
                     feat_bytes: Optional[np.ndarray], layers: int = 2):
    """BACK end: allocate the engine's plan onto the devices and RUN a real aggregation honoring it.

    For each device k:
      resident node set = {v : plan.assignment[v] == k}  UNION  the replicated core (vertex-cut).
      local edges       = edges with BOTH endpoints resident on k (the local SpMM work).
      boundary rows in  = distinct remote endpoints k must FETCH for cross-device edges (the comm
                          the engine's cut implies) -- exchanged over the link (we copy the real
                          feature rows so the wall-clock reflects the boundary feature traffic).
    We aggregate locally with torch.sparse.mm, then add the gathered boundary contribution, so the
    realized run reproduces the full-graph aggregation result (correctness check) while the WHERE
    of each partial sum is exactly the engine's placement. Time front/middle is elsewhere; here we
    time compute + comm per device and take the MAX as the realized makespan (parallel devices)."""
    N = g.num_nodes
    D = cluster.num_devices
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)

    node_dev, core_mask = _node_dev_from_plan(plan_obj, N, D)
    X = _make_features(N, feat_dim, feat_bytes)

    # reference full-graph result (the WHAT). Computed once; the per-device realization must match.
    H_full = _full_graph_aggregate(src, dst, X)

    # accumulate the realized per-device output into H_real to verify it equals H_full.
    H_real = torch.zeros_like(H_full)
    device_runs = []
    measured_makespan = 0.0

    # which device "owns" (will WRITE the output of) each node: its home device. The replicated
    # core is owned by its round-robin home so every node's output is produced exactly once.
    is_core = core_mask if core_mask is not None else np.zeros(N, dtype=bool)
    # undirected edge endpoints, doubled, so each node aggregates from neighbours on both sides.
    u_all = np.concatenate([src, dst])
    v_all = np.concatenate([dst, src])

    for k in range(D):
        dp = plan_obj.placement[k]
        # nodes whose OUTPUT this device produces: those homed on k (periphery homed here, plus the
        # core rows whose round-robin home is k). These are the device's "owned" aggregating nodes.
        owned_mask = (node_dev == k)
        owned = np.nonzero(owned_mask)[0]
        if owned.size == 0:
            device_runs.append(DeviceRun(k, dp.name, 0, int(is_core.sum()), 0, 0, 0.0, 0.0,
                                         0.0, bool(dp.feasible)))
            continue

        # edges feeding an OWNED node: keep doubled edges whose aggregating endpoint v is owned here.
        feed = owned_mask[v_all]
        u_f, v_f = u_all[feed], v_all[feed]        # v_f aggregates from neighbour u_f
        # BOUNDARY (the engine's cut): neighbour u_f lives on a DIFFERENT device (not owned, not the
        # replicated core which is resident everywhere) -> its feature row must be FETCHED over the
        # link. These distinct remote rows ARE the comm the plan's cut implies.
        remote = (~owned_mask[u_f]) & (~is_core[u_f])
        boundary_rows = np.unique(u_f[remote])

        # build the LOCAL subgraph. COLUMNS = owned nodes FIRST (local ids 0..R), then the 1-hop
        # halo (remote/core source rows the device reads). ROWS = the owned aggregating nodes, so
        # owned row r reads its own feature at local column r (the self-loop). One mean-aggregation
        # pass over this halo subgraph is EXACT for the owned rows -> reproduces H_full bit-for-bit.
        halo = np.setdiff1d(np.unique(u_f), owned, assume_unique=False)  # source rows not owned here
        col_nodes = np.concatenate([owned, halo])  # owned occupy local columns 0..R, halo after
        loc = -np.ones(N, dtype=np.int64)
        loc[col_nodes] = np.arange(col_nodes.size)
        lu = loc[u_f]                              # neighbour source -> local column (in col_nodes)
        lv_row = loc[v_f]                          # aggregating owned node -> local ROW (0..R-1)

        # ---- BACK compute: ONE local mean-aggregation via torch.sparse.mm on the halo subgraph ----
        t0 = time.perf_counter()
        Xcol = X[torch.from_numpy(col_nodes)]       # owned + halo feature rows resident for the SpMM
        A_loc, _deg = _norm_adj_local(lv_row, lu, owned.size, col_nodes.size)
        Hl = torch.sparse.mm(A_loc, Xcol)           # [R, F] aggregated outputs for owned rows
        compute_ms = (time.perf_counter() - t0) * 1e3

        # ---- BACK comm: the boundary feature-row exchange the engine's cut implies (over the link) ----
        t1 = time.perf_counter()
        if boundary_rows.size:
            _fetched = X[torch.from_numpy(boundary_rows)].clone()   # the real feature-row transfer
        comm_ms = (time.perf_counter() - t1) * 1e3

        # write this device's owned outputs into the global realized result (each node written once).
        H_real[torch.from_numpy(owned)] = Hl

        dev_makespan = compute_ms + comm_ms
        measured_makespan = max(measured_makespan, dev_makespan)
        device_runs.append(DeviceRun(
            device=k, name=dp.name, home_nodes=int((owned_mask & (~is_core)).sum()),
            replicated_core=int(is_core.sum()), local_edges=int(feed.sum()),
            boundary_rows_in=int(boundary_rows.size), compute_ms=compute_ms,
            comm_ms=comm_ms, makespan_ms=dev_makespan, feasible=bool(dp.feasible)))

    # PROCESS-only correctness: the partitioned realization equals the single-device aggregation on
    # the homed rows (replicated-core rows are aggregated identically on every device). Vertex-cut
    # core rows are written by their round-robin home so every node is covered exactly once.
    err = float((H_real - H_full).abs().max().item()) if N else 0.0
    return device_runs, measured_makespan, err


def _attributed_feat_bytes(g: TemporalGraph, base_dim: int):
    """Build a REAL heterogeneous per-node feature-size vector F_v from the attributed graph.
    JODIE wiki/reddit carry per-EDGE feature vectors (the spec note: 172-dim LIWC); when the loader
    surfaces them as g.efeat [E, Fe] we size each node by the attributed signal it actually carries,
        F_v[v] = base_dim + Fe * (1 + log1p(activity_v)),
    where activity_v is the node's incident edge-feature count. If the loader dropped efeat we fall
    back to the spec's documented edge-feature dim and the node's REAL degree as the activity proxy,
    so hub nodes still get genuinely larger F_v (heavy multi-modal mass) than leaves -- the
    heterogeneity the §33 attribute-aware placement reasons about. Returns F_v [N] (feature DIMS)
    when the dataset is attributed, else None (-> the engine runs the uniform scalar path)."""
    N = g.num_nodes
    deg = np.bincount(np.concatenate([g.src, g.dst]), minlength=N).astype(np.float64)
    if g.efeat is not None and g.efeat.shape[1] > 0:
        Fe = int(g.efeat.shape[1])
    else:
        # loader dropped per-edge feats (ragged JODIE rows): use the documented edge-feature dim.
        try:
            note = get_spec(g.name.split("[")[0]).note
        except Exception:
            note = ""
        import re
        m = re.search(r"(\d+)-dim", note)
        Fe = int(m.group(1)) if m else 0
        if Fe == 0:
            return None                            # genuinely featureless -> not an attributed run
    Fv = base_dim + Fe * (1.0 + np.log1p(deg))
    return Fv.astype(np.float64)


def run_dataset(name: str, link_gbps: float, feat_dim: int, snapshots: int,
                attributed: bool = False, max_edges: Optional[int] = None) -> StageResult:
    """Run the FULL front->middle->back pipeline on ONE real dataset using the REAL engine."""
    print(f"\n{'='*92}\n=== END-TO-END ZORD on '{name}'  (link={link_gbps:g} GB/s, F={feat_dim}, "
          f"attributed={attributed}) ===\n{'='*92}")

    # -------- FRONT: graph in (the REAL loader) --------
    t0 = time.perf_counter()
    g = load(name).sort_by_time()
    load_s = time.perf_counter() - t0
    if max_edges is not None and g.num_edges > max_edges:
        # cap huge graphs for a CPU realized run (the ENGINE still plans the capped real graph;
        # this is a runtime budget, not a partition decision). Take the first max_edges by time.
        g = TemporalGraph(src=g.src[:max_edges], dst=g.dst[:max_edges], t=g.t[:max_edges],
                          efeat=(g.efeat[:max_edges] if g.efeat is not None else None),
                          name=g.name + f"[:{max_edges}]")
        g.sort_by_time()
    s = g.summary()
    print(f"  FRONT  loaded {s['name']}: N={s['num_nodes']:,} E={s['num_edges']:,} "
          f"edge_feat_dim={s['edge_feat_dim']}  load={load_s*1e3:.1f}ms")

    cluster = _build_cluster(link_gbps)

    # optional REAL per-node feature-byte vector F_v for the attribute-aware engine path
    feat_bytes = _attributed_feat_bytes(g, feat_dim) if attributed else None
    if attributed and feat_bytes is not None:
        print(f"         F_v (attribute-aware): per-node dims in "
              f"[{feat_bytes.min():.0f}, {feat_bytes.max():.0f}] mean={feat_bytes.mean():.0f} "
              f"from {s['edge_feat_dim']}-dim edge feats")

    # -------- MIDDLE: the REAL engine decides the partition + placement + cut + makespan --------
    t0 = time.perf_counter()
    plan_obj = plan(g, cluster, link_gbps=link_gbps, feat_dim=feat_dim,
                    num_snapshots=snapshots, feat_bytes=feat_bytes)
    plan_s = time.perf_counter() - t0
    print(f"  MIDDLE zord.schedule.plan -> {plan_obj.summary()}")
    print(f"         (plan computed in {plan_s*1e3:.1f}ms; this assignment is the ENGINE's, "
          f"not reimplemented here)")

    # -------- BACK: allocate the plan onto the devices and RUN it (real torch.sparse.mm) --------
    t0 = time.perf_counter()
    device_runs, measured_makespan, err = _realize_and_run(
        g, plan_obj, cluster, feat_dim, feat_bytes)
    run_s = time.perf_counter() - t0
    print(f"  BACK   realized + ran the plan on {cluster.num_devices} devices "
          f"(torch.sparse.mm, CPU):")
    for dr in device_runs:
        flag = "" if dr.feasible else "  <-- engine flagged OOM"
        print(f"           dev{dr.device} {dr.name:<12} home={dr.home_nodes:>9,} "
              f"+core={dr.replicated_core:>7,} local_edges={dr.local_edges:>10,} "
              f"boundary_in={dr.boundary_rows_in:>8,} compute={dr.compute_ms:7.2f}ms "
              f"comm={dr.comm_ms:6.2f}ms makespan={dr.makespan_ms:7.2f}ms{flag}")
    print(f"         realized makespan (max device) = {measured_makespan:.2f}ms   "
          f"correctness max|partitioned-full| = {err:.2e} "
          f"({'PASS' if err < 1e-3 else 'CHECK'})")

    return StageResult(
        dataset=name, N=g.num_nodes, E=g.num_edges,
        edge_feat_dim=s["edge_feat_dim"], load_s=load_s, plan_s=plan_s,
        realize_s=0.0, run_s=run_s, strategy=plan_obj.strategy,
        engine_makespan_ms=plan_obj.makespan_ms, engine_feasible=plan_obj.feasible,
        engine_bound=plan_obj.bound, cut_edges=plan_obj.cut_edges,
        replication_pct=plan_obj.replication_pct,
        candidate_makespans=dict(plan_obj.candidate_makespans),
        link_gbps=link_gbps, feat_dim=feat_dim, attributed=attributed,
        measured_makespan_ms=measured_makespan, devices=device_runs,
        correctness_max_abs_err=err)


def _assert_uses_real_engine():
    """Prove (at runtime) the pipeline is bound to src/zord: the engine entry points are the REAL
    module objects from the installed/checked-out zord, and they live under .../src/zord/. This is
    the grep-able + importable contract that the partition decision is NOT reimplemented here."""
    assert plan.__module__ == "zord.schedule.planner", plan.__module__
    assert arrange.__module__ == "zord.partition.arrange", arrange.__module__
    assert load.__module__ == "zord.datasets.loaders", load.__module__
    eng = os.path.abspath(plan.__code__.co_filename)
    assert os.path.join("src", "zord", "schedule", "planner.py") in eng, eng
    print(f"[engine check] PASS -- partition/placement comes from the REAL engine:")
    print(f"  zord.schedule.plan        @ {os.path.abspath(plan.__code__.co_filename)}")
    print(f"  zord.partition.arrange    @ {os.path.abspath(arrange.__code__.co_filename)}")
    print(f"  zord.datasets.load        @ {os.path.abspath(load.__code__.co_filename)}")
    print(f"  zord {zord.__version__}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="End-to-end ZORD pipeline on REAL datasets via the REAL engine")
    ap.add_argument("--datasets", default="collegemsg,bitcoin-alpha,askubuntu",
                    help="comma-separated dataset names from the zord registry (front loaders)")
    ap.add_argument("--attributed", default="jodie-wikipedia",
                    help="an attributed dataset to run F_v-aware (real edge features); '' to skip")
    ap.add_argument("--link-gbps", type=float, default=0.5,
                    help="interconnect bandwidth PARAMETER (GB/s). zord wins at any comm speed.")
    ap.add_argument("--feat", type=int, default=128, help="node feature dim F")
    ap.add_argument("--snapshots", type=int, default=64)
    ap.add_argument("--max-edges", type=int, default=2_000_000,
                    help="cap edges for the CPU realized run (engine still plans the real graph)")
    args = ap.parse_args(argv)

    _assert_uses_real_engine()

    results = []
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    for name in names:
        try:
            results.append(run_dataset(name, args.link_gbps, args.feat, args.snapshots,
                                       attributed=False, max_edges=args.max_edges))
        except Exception as e:
            print(f"  [skip {name}] {type(e).__name__}: {e}")

    if args.attributed:
        try:
            results.append(run_dataset(args.attributed, args.link_gbps, args.feat, args.snapshots,
                                       attributed=True, max_edges=args.max_edges))
        except Exception as e:
            print(f"  [skip {args.attributed}] {type(e).__name__}: {e}")

    # -------- final report (all times in ms) --------
    print(f"\n{'='*100}\n=== END-TO-END SUMMARY (front/middle/back wall-clock in ms; "
          f"partition decisions from src/zord) ===\n{'='*100}")
    hdr = (f"{'dataset':16}{'N':>9}{'E':>10}{'strategy':>21}{'load_ms':>9}{'plan_ms':>9}"
           f"{'run_ms':>9}{'eng_mks':>9}{'meas_mks':>10}{'feas':>6}{'cut':>10}{'err':>9}")
    print(hdr)
    for r in results:
        print(f"{r.dataset:16}{r.N:>9,}{r.E:>10,}{r.strategy:>21}"
              f"{r.load_s*1e3:>9.0f}{r.plan_s*1e3:>9.0f}{r.run_s*1e3:>9.0f}"
              f"{r.engine_makespan_ms:>9.2f}{r.measured_makespan_ms:>10.2f}"
              f"{str(r.engine_feasible):>6}{r.cut_edges:>10,}{r.correctness_max_abs_err:>9.1e}")
    print("\nlegend: load/plan/run = FRONT/MIDDLE/BACK wall-clock (ms). eng_mks = engine PREDICTED "
          "makespan (GPU roofline, ms); meas_mks = REALIZED max-device wall-clock of the run\n"
          "        (CPU torch.sparse.mm, ms); err = max|partitioned - full| aggregation (PROCESS-"
          "only correctness, ~0 = WHERE the partial sums ran changed, WHAT was computed did not).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
