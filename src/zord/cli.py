"""zord CLI:  zord <command>

  datasets            list the dataset registry
  info <name>         show one dataset's spec
  cluster             show the (default HetCluster) heterogeneous cluster profile
  plan <name>         load -> adaptive-corner ARRANGE + placement + vertex-cut +
                      incremental-migration -> predicted makespan + feasibility, the chosen
                      DECOMPOSITION axis (node/feature/hybrid), and the CPU<->HBM MEMORY-TIERING
                      plan (resident GB, spilled GB, predicted peak, feasibility verdict).
                      The interconnect bandwidth is a PARAMETER (--link-gbps). Heterogeneous
                      per-node feature bytes F_v via --multimodal or --feat-bytes-file.
  probe <name>        FRONT+L2: ingest the graph, run the cheap GraphStats probe, then auto-probe
                      hardware + calibrate -> the self-calibrated CostCalibration (w_S/w_T, sec_per_edge).
                      --measure microbenchmarks achieved bandwidth on a CUDA box (else spec-sheet).
  ingest <name>       FRONT-END (K1): ingest a temporal graph into a TemporalGraphInput and print the
                      GraphStats probe (N/E, degrees, clusterability, persistence, feature stats).
  schedule|run <name> L3 CONDUCTOR: ingest -> calibrate -> allocate (supra-cut + axis) -> tier +
                      bufferpool + coexec + recombine -> ONE SchedulePlan (per-epoch makespan, bound,
                      feasibility, decomposition axis). --epochs folds the end-to-end JobEstimate.
  partition <name|file> INDUSTRIAL partition entry: run zord's OWN attribute-aware C++ kernels on a
                      dataset or edge file (.npz src/dst[,fv] or binary i64 N,M + i32 src,dst).
                      --method auto picks by scale (multilevel <=20M edges, streaming above -- the
                      O(E) path that holds at 1B+). Attributes: --fv-file/--multimodal (node feature
                      dims -> vwgt bytes), --fe-dim (edge feature dim -> ewgt). Heterogeneity:
                      --ratio auto (cluster throughput shares) or explicit. LARGE-CLUSTER mode:
                      --hierarchy NODESxGPUS partitions cluster->node (level 1, feature-aware)
                      then node->GPU (level 2) -- the two-level architecture that scales to
                      10k-GPU clusters. --timeout guards every kernel call; metrics to JSON.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .datasets import DATASETS, get_spec, load
from .profiler import hetcluster, GB
from .frontend import Intent, resolve
from .schedule import plan as build_plan


def _cmd_datasets(args):
    rows = sorted(DATASETS.values(), key=lambda s: (s.tier, s.name))
    print(f"{'name':16} {'tier':7} {'edges':>13}  fmt")
    for s in rows:
        e = f"{s.edges:,}" if s.edges else "?"
        print(f"{s.name:16} {s.tier:7} {e:>13}  {s.fmt}")


def _cmd_info(args):
    s = get_spec(args.name)
    for k, v in vars(s).items():
        print(f"{k:14}: {v}")


def _cmd_cluster(args):
    c = hetcluster(num_h100=args.h100, num_6000ada=args.a6000, num_5000ada=args.a5000)
    print(f"cluster: {c.num_devices} devices, total usable mem "
          f"{c.total_usable_mem/GB:.0f} GB, total throughput {c.total_throughput:.1f}")
    for d in c.devices:
        print(f"  dev{d.id}: {d.name:18} usable {d.usable_mem/GB:5.1f} GB  r={d.throughput:.1f}  node={d.node}")


def _build_feat_bytes(args, num_nodes: int):
    """Build the per-node feature-DIMS vector F_v [N] from the CLI knobs, or None.

    Two mutually-exclusive sources (None -> the SCALAR --feat path runs UNCHANGED, bit-identical
    to §30/§32):
      --multimodal RICH_FRAC:RICH_DIM:BASE_DIM  : a RICH_FRAC fraction of nodes carry RICH_DIM dims
                                                  (the multi-modal hubs), the rest BASE_DIM (leaves).
      --feat-bytes-file PATH                    : load an [N] vector of per-node feature DIMS from a
                                                  .npy / whitespace-or-comma text file.
    PROCESS-only: F_v changes WHERE rows live / WHAT spills, never the result (same knobs => same F_v
    => same plan)."""
    if args.feat_bytes_file:
        path = args.feat_bytes_file
        if path.endswith(".npy"):
            Fv = np.load(path)
        else:
            Fv = np.loadtxt(path, delimiter="," if "," in open(path).read(64) else None)
        Fv = np.asarray(Fv, dtype=np.float64).ravel()
        if Fv.size != num_nodes:
            raise SystemExit(f"--feat-bytes-file has {Fv.size} entries but the graph has {num_nodes} nodes")
        return Fv
    if args.multimodal:
        try:
            rich_frac, rich_dim, base_dim = args.multimodal.split(":")
            rich_frac, rich_dim, base_dim = float(rich_frac), int(rich_dim), int(base_dim)
        except Exception:
            raise SystemExit("--multimodal must be RICH_FRAC:RICH_DIM:BASE_DIM, e.g. 0.10:4096:128")
        rng = np.random.default_rng(args.seed)
        Fv = np.full(num_nodes, float(base_dim), dtype=np.float64)
        n_rich = int(round(rich_frac * num_nodes))
        if n_rich > 0:
            Fv[rng.choice(num_nodes, size=n_rich, replace=False)] = float(rich_dim)
        print(f"F_v (multimodal): {n_rich:,} hubs @ {rich_dim} dims + {num_nodes - n_rich:,} "
              f"leaves @ {base_dim} dims  ->  mean_F={Fv.mean():.0f}")
        return Fv
    return None


def _cmd_plan(args):
    g = load(args.name).sort_by_time()
    print(f"loaded {g.name}: {g.summary()}")
    snaps = g.to_snapshots(num_snapshots=args.snapshots)
    print(f"snapshots: {len(snaps)}")
    c = hetcluster(num_h100=args.h100, num_6000ada=args.a6000, num_5000ada=args.a5000)
    link = args.link_gbps if args.link_gbps is not None else c.inter_node_bw
    print(f"cluster: {c.num_devices} devices, interconnect (parameter) = {link:g} GB/s")

    # (a) heterogeneous per-node feature bytes F_v (None -> scalar --feat path, BIT-IDENTICAL).
    feat_bytes = _build_feat_bytes(args, g.num_nodes)
    feat_dim = args.feat if feat_bytes is None else int(round(float(feat_bytes.mean())))

    p = build_plan(g, c, link_gbps=args.link_gbps, feat_dim=feat_dim,
                   num_snapshots=args.snapshots, mem_dim=args.mem_dim,
                   window=args.window, reuse_frac=args.reuse_frac, seed=args.seed,
                   feat_bytes=feat_bytes, decomposition=args.decomposition)
    print(p.summary())

    # (b) the chosen DECOMPOSITION axis (node / feature / hybrid). When --decomposition node
    # (default) the choice is not costed and the plan stays node-parallel + byte-identical.
    print("\n-- decomposition --")
    if p.decomposition is not None:
        print(p.decomposition.summary())
    else:
        print(f"axis=node (default node-parallel; pass --decomposition auto to also cost "
              f"feature-parallel + hybrid)")

    # (c) the CPU<->HBM MEMORY-TIERING plan: resident GB, spilled GB, predicted peak, verdict.
    if p.memory is not None:
        m = p.memory
        print("\n-- memory-tiering (CPU<->HBM) --")
        print(m.summary())
        resident_gb = sum(d.peak_hbm_bytes for d in m.per_device) / GB
        verdict = "FEASIBLE (no-OOM via tiering)" if m.all_feasible else "INFEASIBLE (needs more HBM / less F_v pressure)"
        peak_gb = max((d.peak_hbm_bytes for d in m.per_device), default=0) / GB
        print(f"  resident(peak)={resident_gb:.1f}GB  spilled={m.total_streamed_gb:.1f}GB/epoch  "
              f"predicted_peak(max dev)={peak_gb:.1f}GB  bound={m.bound}  ->  {verdict}")


# --------------------------------------------------------------------------- #
#  NEW kernel commands: probe / ingest / schedule|run  (the FRONT -> MIDDLE ->  #
#  BACK pipeline). These call the proven module entries; they do NOT touch the  #
#  scalar `plan` path above, which stays bit-identical.                          #
# --------------------------------------------------------------------------- #
def _ingest_graph(args):
    """Shared FRONT-END step: load -> ingest -> TemporalGraphInput (with F_v if requested)."""
    from .frontend import ingest_graph as _ingest
    g = load(args.name).sort_by_time()
    feat_bytes = _build_feat_bytes(args, g.num_nodes)
    mode = getattr(args, "mode", "dtdg")
    tgi = _ingest(g, num_snapshots=args.snapshots, feat_bytes=feat_bytes, mode=mode)
    return g, tgi, feat_bytes


def _cmd_ingest(args):
    g, tgi, _ = _ingest_graph(args)
    print(f"loaded {g.name}: {g.summary()}")
    print(tgi.summary())


def _cmd_probe(args):
    from .profiler import probe_and_calibrate
    g, tgi, feat_bytes = _ingest_graph(args)
    print(f"loaded {g.name}: {g.summary()}")
    c = hetcluster(num_h100=args.h100, num_6000ada=args.a6000, num_5000ada=args.a5000)
    feat_dim = args.feat if feat_bytes is None else int(round(float(feat_bytes.mean())))
    calib = probe_and_calibrate(g, c, measure=args.measure, num_snapshots=args.snapshots,
                                feat_dim=feat_dim, feat_bytes=feat_bytes, window=args.window)
    print(calib.summary())
    if calib.cluster is not None:
        # surface the probe provenance (what was measured vs spec) when --measure was requested.
        from .profiler.prober import probe_hardware
        pr = probe_hardware(c, measure=args.measure)
        for n in pr.notes:
            print(f"  [probe] {n}")


def _cmd_schedule(args):
    from .profiler import probe_and_calibrate  # noqa: F401  (kept for parity; schedule probes itself)
    from .schedule import schedule as run_schedule
    g, tgi, feat_bytes = _ingest_graph(args)
    print(f"loaded {g.name}: {g.summary()}")
    print(tgi.summary())
    c = hetcluster(num_h100=args.h100, num_6000ada=args.a6000, num_5000ada=args.a5000)
    link = args.link_gbps if args.link_gbps is not None else c.inter_node_bw
    print(f"cluster: {c.num_devices} devices, interconnect (parameter) = {link:g} GB/s")
    feat_dim = args.feat if feat_bytes is None else int(round(float(feat_bytes.mean())))
    sp = run_schedule(tgi, c, link_gbps=args.link_gbps, feat_dim=feat_dim,
                      num_snapshots=args.snapshots, window=args.window,
                      reuse_frac=args.reuse_frac, decomposition=args.decomposition,
                      num_epochs=args.epochs, seed=args.seed, measure=args.measure)
    print(sp.summary())
    est = sp.estimate_total_time(num_epochs=args.epochs)
    print(est.summary())


def _cmd_partition(a):
    """INDUSTRIAL partition entry: zord's attribute-aware kernels + auto method + hierarchy +
    timeout + JSON metrics. PROCESS-only: a partition is a result-preserving placement."""
    import json, os, signal, struct, time
    from .partition import cpp_kernel as CK

    class _Timeout(Exception):
        pass

    def _alarm(sig, frm):
        raise _Timeout()

    # ---- load edges: registry dataset | .npz(src,dst[,fv]) | raw binary (i64 N,M; i32 src,dst) ----
    t0 = time.time()
    fv_dims = None
    if os.path.exists(a.name):
        if a.name.endswith(".npz"):
            z = np.load(a.name)
            src = np.ascontiguousarray(z["src"], np.int64); dst = np.ascontiguousarray(z["dst"], np.int64)
            N = int(z["N"]) if "N" in z else int(max(src.max(), dst.max()) + 1)
            if "fv" in z:
                fv_dims = np.ascontiguousarray(z["fv"], np.int64)
        else:
            with open(a.name, "rb") as fh:
                N, M = struct.unpack("<2q", fh.read(16))
                src = np.fromfile(fh, dtype="<i4", count=M).astype(np.int64)
                dst = np.fromfile(fh, dtype="<i4", count=M).astype(np.int64)
    else:
        g = load(a.name).sort_by_time()
        src = np.ascontiguousarray(g.src, np.int64); dst = np.ascontiguousarray(g.dst, np.int64)
        N = int(g.num_nodes)
        ef = getattr(g, "efeat", None)
        if a.fe_dim == 0 and ef is not None and getattr(ef, "ndim", 0) == 2:
            a.fe_dim = int(ef.shape[1])           # use the dataset's REAL edge-feature dim
    M = int(src.size)

    # ---- attributes: per-node feature dims F_v -> vwgt bytes; per-edge F_e -> ewgt bytes ----
    if a.fv_file:
        fv_dims = (np.load(a.fv_file) if a.fv_file.endswith(".npy")
                   else np.loadtxt(a.fv_file)).astype(np.int64)
    elif a.multimodal and fv_dims is None:
        frac, rich, base = a.multimodal.split(":")
        rng = np.random.default_rng(a.seed)
        fv_dims = np.full(N, int(base), np.int64)
        fv_dims[rng.random(N) < float(frac)] = int(rich)
    if fv_dims is not None and fv_dims.size != N:
        p_err = f"fv length {fv_dims.size} != N {N}"
        print(f"[zord partition] ERROR {p_err}"); sys.exit(2)
    vwgt = (fv_dims * 4).astype(np.int64) if fv_dims is not None else None
    ewgt = np.full(M, a.fe_dim * 4, np.int64) if a.fe_dim > 0 else None

    # ---- heterogeneous capacity ratio ----
    ratio = None
    if a.ratio == "auto":
        ratio = np.asarray(hetcluster(a.h100, a.a6000, a.a5000).throughput_shares(), np.float64)
    elif a.ratio:
        ratio = np.asarray([float(x) for x in a.ratio.split(",")], np.float64)

    # ---- method selection (the portfolio; auto = scale-appropriate solution) ----
    method = a.method
    if method == "auto":
        method = "zord-mc" if M <= 20_000_000 else "zord-stream"

    def _run(s_, d_, n_, D_, vw_, ew_, rt_):
        if method == "zord-mc":
            return CK.multilevel_partition(s_, d_, n_, D_, vwgt=vw_, ewgt=ew_, ratio=rt_)
        if method == "zord-stream":
            part_, _ = CK.streaming_partition(s_, d_, n_, D_, mode="fennel", vwgt=vw_)
            return np.asarray(part_, np.int64)
        if method == "hdrf":
            ep_, _ = CK.streaming_partition(s_, d_, n_, D_, mode="hdrf")
            return np.asarray(ep_, np.int64)
        raise ValueError(f"unknown --method {method}")

    levels = [int(x) for x in a.hierarchy.split("x")] if a.hierarchy else None
    D = (levels[0] * levels[1]) if levels else a.parts
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(1, a.timeout))
    status, note = "OK", ""
    t1 = time.time()
    try:
        if levels:  # two-level cluster->node->GPU (the 10k-GPU architecture; level 1 is O(E) at scale)
            D1, D2 = levels
            p1 = _run(src, dst, N, D1, vwgt, ewgt, None)
            part = np.empty(N, np.int64)
            for d in range(D1):
                nodes = np.nonzero(p1 == d)[0]
                lid = np.full(N, -1, np.int64); lid[nodes] = np.arange(nodes.size)
                em = (p1[src] == d) & (p1[dst] == d)
                p2 = _run(lid[src[em]], lid[dst[em]], nodes.size, D2,
                          vwgt[nodes] if vwgt is not None else None,
                          ewgt[em] if ewgt is not None else None, None)
                part[nodes] = d * D2 + p2
        else:
            part = _run(src, dst, N, D, vwgt, ewgt, ratio)
    except _Timeout:
        status, note, part = "TIMEOUT", f">{a.timeout}s", None
    finally:
        signal.alarm(0)
    part_s = round(time.time() - t1, 3)

    rec = dict(input=a.name, method=method, parts=D, hierarchy=a.hierarchy or "",
               N=N, M=M, fe_dim=a.fe_dim, attribute_aware=bool(vwgt is not None or ewgt is not None),
               load_s=round(t1 - t0, 2), part_s=part_s, status=status, note=note)
    if part is not None and method != "hdrf":
        cross = part[src] != part[dst]
        rec["cut"] = int(cross.sum())
        cnt = np.bincount(part, minlength=D).astype(np.float64)
        rec["balance_count"] = round(float(cnt.max() / max(1.0, cnt.mean())), 4)
        if fv_dims is not None:
            fb = (fv_dims * 4).astype(np.float64)
            fm = np.bincount(part, weights=fb, minlength=D)
            rec["balance_featmem"] = round(float(fm.max() / max(1.0, fm.mean())), 4)
            rec["peak_featmem_gb"] = round(float(fm.max()) / 1e9, 3)
            rec["feat_comm_mb"] = round(float(fb[dst[cross]].sum()
                                              + a.fe_dim * 4.0 * int(cross.sum())) / 1e6, 2)
            if a.cap_gb > 0:
                rec["feasible"] = bool(fm.max() <= a.cap_gb * 1024**3)
        if a.out:
            np.save(a.out, part); rec["out"] = a.out
    if a.metrics:
        os.makedirs(os.path.dirname(a.metrics) or ".", exist_ok=True)
        with open(a.metrics, "w") as fh:
            json.dump(rec, fh, indent=2)
    print("[zord partition] " + json.dumps(rec))
    if status != "OK":
        sys.exit(3)


def main(argv=None):
    p = argparse.ArgumentParser(prog="zord", description=f"ZORD {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets").set_defaults(func=_cmd_datasets)
    pi = sub.add_parser("info"); pi.add_argument("name"); pi.set_defaults(func=_cmd_info)
    for name in ("cluster", "plan"):
        sp = sub.add_parser(name)
        sp.add_argument("--h100", type=int, default=1)
        sp.add_argument("--a6000", type=int, default=1)
        sp.add_argument("--a5000", type=int, default=1)
        if name == "plan":
            sp.add_argument("name")
            sp.add_argument("--snapshots", type=int, default=32)
            sp.add_argument("--feat", type=int, default=128, help="node feature dim F (scalar path)")
            sp.add_argument("--mem-dim", type=int, default=100,
                            help="TGN node-memory width m (bytes migrated per moving vertex)")
            sp.add_argument("--link-gbps", type=float, default=None,
                            help="interconnect bandwidth PARAMETER (GB/s); default = cluster link. "
                                 "zord wins on the algorithm at ANY comm speed.")
            sp.add_argument("--window", type=int, default=1,
                            help="co-resident snapshots W (the temporal batch) for memory-tiering")
            sp.add_argument("--reuse-frac", type=float, default=0.0,
                            help="rho: fraction of nodes unchanged vs the previous snapshot (reuse)")
            sp.add_argument("--multimodal", default=None, metavar="RICH_FRAC:RICH_DIM:BASE_DIM",
                            help="heterogeneous per-node feature bytes F_v: a RICH_FRAC fraction of "
                                 "nodes carry RICH_DIM dims (multi-modal hubs), the rest BASE_DIM "
                                 "(leaves). E.g. 0.10:4096:128. Omit -> scalar --feat (bit-identical).")
            sp.add_argument("--feat-bytes-file", default=None, metavar="PATH",
                            help="load per-node feature DIMS F_v [N] from a .npy or text file "
                                 "(alternative to --multimodal)")
            sp.add_argument("--decomposition", default="node",
                            choices=["node", "auto", "feature", "hybrid"],
                            help="decomposition axis: node (default, byte-identical) | auto (cost "
                                 "feature-parallel + hybrid, pick lowest feasible) | feature | hybrid")
            sp.add_argument("--seed", type=int, default=0, help="RNG seed (F_v sampling, arrange)")
            sp.add_argument("--intent", default="min_time",
                            choices=[i.value for i in Intent])
        sp.set_defaults(func=_cmd_cluster if name == "cluster" else _cmd_plan)

    # ---- NEW kernel commands: probe / ingest / schedule (alias: run). These share a common set of
    # FRONT-END + cluster knobs; they call the proven module entries and do NOT alter the `plan` path. --
    def _add_common(sp, *, with_sched_knobs: bool):
        sp.add_argument("name")
        sp.add_argument("--h100", type=int, default=1)
        sp.add_argument("--a6000", type=int, default=1)
        sp.add_argument("--a5000", type=int, default=1)
        sp.add_argument("--snapshots", type=int, default=32)
        sp.add_argument("--feat", type=int, default=128, help="node feature dim F (scalar path)")
        sp.add_argument("--window", type=int, default=1,
                        help="co-resident snapshots W (the temporal batch)")
        sp.add_argument("--multimodal", default=None, metavar="RICH_FRAC:RICH_DIM:BASE_DIM",
                        help="heterogeneous per-node feature bytes F_v (RICH_FRAC:RICH_DIM:BASE_DIM, "
                             "e.g. 0.10:4096:128). Omit -> scalar --feat.")
        sp.add_argument("--feat-bytes-file", default=None, metavar="PATH",
                        help="load per-node feature DIMS F_v [N] from a .npy or text file")
        sp.add_argument("--seed", type=int, default=0, help="RNG seed (F_v sampling, arrange)")
        sp.add_argument("--mode", default="dtdg", choices=["dtdg", "ctdg"],
                        help="dtdg snapshot model (default) | ctdg event-stream model")
        if with_sched_knobs:
            sp.add_argument("--link-gbps", type=float, default=None,
                            help="interconnect bandwidth PARAMETER (GB/s); default = cluster link.")
            sp.add_argument("--reuse-frac", type=float, default=0.0,
                            help="rho: fraction of nodes unchanged vs the previous snapshot (reuse)")
            sp.add_argument("--decomposition", default="auto",
                            choices=["node", "auto", "feature", "hybrid"],
                            help="decomposition axis (default auto: cost node/feature/hybrid)")
            sp.add_argument("--epochs", type=int, default=1,
                            help="training epochs (folds the end-to-end JobEstimate)")
            sp.add_argument("--measure", action="store_true",
                            help="microbenchmark achieved bandwidth on a CUDA box (else spec-sheet)")

    pq = sub.add_parser("partition")
    pq.add_argument("name", help="dataset name | edges .npz(src,dst[,N,fv]) | raw binary i64 N,M + i32 src,dst")
    pq.add_argument("--parts", "-D", type=int, default=8)
    pq.add_argument("--method", default="auto", choices=["auto", "zord-mc", "zord-stream", "hdrf"])
    pq.add_argument("--fv-file", default=None, help="per-node feature DIMS [N] (.npy/.txt) -> vwgt")
    pq.add_argument("--multimodal", default=None, metavar="FRAC:RICH:BASE",
                    help="modeled heterogeneous F_v (e.g. 0.2:512:128); labelled, not real data")
    pq.add_argument("--fe-dim", type=int, default=0,
                    help="edge feature dim F_e -> ewgt (auto-detected from dataset efeat)")
    pq.add_argument("--ratio", default="", help="'auto' (cluster throughput shares) or comma list")
    pq.add_argument("--hierarchy", default="", metavar="NODESxGPUS",
                    help="two-level cluster->node->GPU partition (e.g. 1250x8 on a 10k-GPU cluster)")
    pq.add_argument("--h100", type=int, default=1); pq.add_argument("--a6000", type=int, default=1)
    pq.add_argument("--a5000", type=int, default=1)
    pq.add_argument("--cap-gb", type=float, default=0, help="per-device HBM cap -> feasibility verdict")
    pq.add_argument("--timeout", type=int, default=3600)
    pq.add_argument("--seed", type=int, default=0)
    pq.add_argument("--out", default=None, help="write the partition as .npy")
    pq.add_argument("--metrics", default=None, help="write metrics JSON here")
    pq.set_defaults(func=_cmd_partition)
    pp = sub.add_parser("probe"); _add_common(pp, with_sched_knobs=True)
    pp.set_defaults(func=_cmd_probe)
    pin = sub.add_parser("ingest"); _add_common(pin, with_sched_knobs=False)
    pin.set_defaults(func=_cmd_ingest)
    for alias in ("schedule", "run"):
        ps = sub.add_parser(alias); _add_common(ps, with_sched_knobs=True)
        ps.set_defaults(func=_cmd_schedule)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
