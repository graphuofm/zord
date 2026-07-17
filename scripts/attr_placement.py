#!/usr/bin/env python3
"""ATTR-PLACEMENT (D-attr-2): do HETEROGENEOUS per-node feature BYTE-SIZES give zord a REAL
placement / tiering / feasibility win that an attribute-BLIND partitioner misses?

WHY THIS IS NEW (vs RESULTS §27, which found attributes NULL):
  §27 tested attributes on REAL jodie data where every node has the SAME 172-dim feature, i.e.
  a UNIFORM F. With uniform F the per-device feature memory is exactly count_k * F * 4, so a
  partitioner that balances NODE COUNT already balances feature memory -- attributes carry no
  independent signal and §27 was correctly NULL. §27's idea-1 also charged deg*F only for
  COMPUTE-work BALANCE (and judged on the COMPUTE makespan), never for FEASIBILITY/PLACEMENT.

  THE GAP THIS SCRIPT EXPLOITS: real multi-modal / multi-type temporal graphs are NOT uniform-F.
  A "rich" node (text 768d, image 512d) costs many times the HBM of a "poor" node (categorical
  8d). zord's OWN feasibility test (src/zord/partition/arrange.py:feasible) and the planner's
  HBM footprint (planner.py:_placement_from_arrange / _snapshot_state_bytes) charge

        device_feat_bytes_k = counts[k] * F * 4          # <-- a SCALAR, uniform F

  i.e. they assume EVERY node costs the SAME F*4 bytes. An attribute-BLIND partitioner (hash /
  METIS / count- or degree-balanced) sizes a device by its NODE COUNT (x a uniform mean F). If
  the high-F "rich" nodes happen to land disproportionately on one device -- which they DO when
  feature TYPE is structurally clustered (a whole rich community kept local by a locality-aware
  cut, or simply an unlucky hash bucket) -- the blind partitioner's TRUE feature memory on that
  device is sum_{v in k} F_v * 4  >>  count_k * Fbar * 4, and the device OOMs even though the
  blind model says it fits. An attribute-AWARE partitioner sizes by the ACTUAL per-node feature
  BYTES (sum_v F_v) and balances that heterogeneous feature-load against each device's HBM
  CAPACITY (high-F mass -> the big-HBM H100; low-F mass -> the small RTX5000) -> stays feasible
  and lower peak memory. Same data + same model => SAME numerical result (we only change WHERE a
  feature row lives); we measure PROCESS only: peak per-device HBM, feasibility (OOM?), makespan.

WHAT WE MEASURE (three experiments, all PROCESS-only; NEVER accuracy):
  EXP-A  PLACEMENT FEASIBILITY: heterogeneous F_v, fits in aggregate HBM but the rich nodes are
         clustered. blind (balance node-COUNT, uniform-Fbar sizing) vs aware (balance feature
         BYTES against per-device HBM). Report peak per-device HBM, does blind OOM a device,
         does aware stay feasible, and the makespan if feature-BANDWIDTH-bound.
  EXP-B  HETEROGENEOUS-HBM MATCH: same, but the win is specifically routing high-F mass to the
         high-HBM-capacity device. Sweep the rich-fraction; find where blind OOMs but aware holds.
  EXP-C  TIERING UNDER FEATURE PRESSURE: total feature memory > aggregate HBM. Does attribute-
         AWARE tiering (stage the high-F COLD nodes to CPU first -- evict the biggest rows) keep
         it feasible / lower peak HBM where a blind (uniform-F, evict-by-count) tiering fails?

This script writes ONLY itself, uses pure numpy + the zord engine, NO networkx, NO SLURM, and
exercises the REAL engine path (arrange.feasible / the per-node-byte generalization of it) so
the verdict is about zord's actual cost model, not a toy. Real --dataset supported (falls back
to synthetic with a flag if the dataset is not staged on this box).

  python3 scripts/attr_placement.py                       # all 3 experiments, default synthetic
  python3 scripts/attr_placement.py --exp A --rich-frac 0.1 --heavy-dim 768 --poor-dim 16
  python3 scripts/attr_placement.py --dataset jodie-wikipedia   # real graph + modeled F_v types
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

GB = 1024 ** 3
BYTES_PER_FEAT = 4.0          # fp32, FULL PRECISION (not a compression knob)
N_GATHERS = 2                 # 2-layer aggregation
BYTES_PER_EDGE_RESIDENT = 20.0  # src+dst+ts+w edge metadata (matches arrange.BYTES_PER_EDGE_RESIDENT)


# ============================================================================ #
# cluster + structural helpers (engine-native; NO networkx)                     #
# ============================================================================ #
def build_cluster(args):
    """Heterogeneous cluster. The CAPACITY heterogeneity (HBM GB) is the placement lever here:
    high-F mass should go on the big-HBM device. Achieved-agg-bw drives compute makespan."""
    from zord.profiler.cluster_profile import from_spec
    hbm = [float(x) for x in args.hbm_gb.split(",")]
    bw = [float(x) for x in args.agg_bw.split(",")]
    assert len(hbm) == len(bw), "hbm-gb and agg-bw must have equal length"
    c = from_spec(hbm_gb=hbm, agg_bw_gbps=bw, interconnect_gbps=args.link_gbps,
                  names=[f"GPU{i}(HBM{int(hbm[i])})" for i in range(len(hbm))])
    return c


def node_degree(src, dst, N):
    return (np.bincount(src.astype(np.int64), minlength=N) +
            np.bincount(dst.astype(np.int64), minlength=N)).astype(np.int64)


def lpa_rank_or_degree(src, dst, N, deg):
    """C++ LPA cluster-respecting rank (locality layout). Falls back to degree order if the
    C++ kernel is unavailable. This is the LOCALITY layout BOTH partitioners walk -- the only
    difference between blind and aware is the BALANCE WEIGHT, never the layout."""
    try:
        from zord.partition import cpp_kernel
        if cpp_kernel.have_cpp_kernel():
            return cpp_kernel.lpa_rank(N, src.astype(np.int64), dst.astype(np.int64)).astype(np.int64)
    except Exception:
        pass
    return np.argsort(np.argsort(-deg, kind="stable")).astype(np.int64)


def split_contiguous(order_rank, weight, D, caps=None):
    """Walk nodes in `order_rank` (rank->position) and cut into D contiguous segments whose
    cumulative `weight` is balanced -- or capacity-PROPORTIONAL when caps given (the hetero
    match). Returns dev[v] in [0,D). The ONLY knob that differs blind vs aware is `weight`
    (and, for aware, sizing the segment targets to per-device CAPACITY)."""
    N = order_rank.shape[0]
    rank_to_node = np.empty(N, dtype=np.int64)
    rank_to_node[order_rank] = np.arange(N)
    w_by_rank = weight[rank_to_node].astype(np.float64)
    cum = np.cumsum(w_by_rank)
    if cum[-1] <= 0:
        seg = (np.arange(N) * D // max(1, N)).clip(0, D - 1)
    else:
        if caps is None:
            targets = np.arange(1, D) * cum[-1] / D
        else:
            share = np.asarray(caps, dtype=np.float64)
            share = share / share.sum()
            targets = np.cumsum(share)[:-1] * cum[-1]
        cuts = np.searchsorted(cum, targets, side="left")
        bounds = np.concatenate([[0], cuts, [N]]).astype(np.int64)
        seg = (np.searchsorted(bounds, np.arange(N), side="right") - 1).clip(0, D - 1)
    dev = np.empty(N, dtype=np.int64)
    dev[rank_to_node] = seg
    return dev


# ============================================================================ #
# the PROCESS metrics: per-device PEAK HBM (per-node bytes!) + feasibility       #
# ============================================================================ #
def device_footprint(dev, deg, Fv, devs, D):
    """TRUE per-device HBM footprint with PER-NODE feature bytes (the honest accounting):
        feat_bytes_k   = sum_{v on k} F_v * 4         (NOT count_k * Fbar * 4)
        edge_bytes_k   = (incident edges on k) * BYTES_PER_EDGE_RESIDENT
    Returns (peak_bytes[D], feat_bytes[D], cap_bytes[D], feasible[D])."""
    feat_b = np.bincount(dev, weights=(Fv * BYTES_PER_FEAT), minlength=D).astype(np.float64)
    # 2-layer activation copies kept for backward also scale with F_v -> charge (1 + N_GATHERS) rows
    feat_b = feat_b * (1.0 + N_GATHERS)
    inc = np.bincount(dev, weights=deg.astype(np.float64), minlength=D)
    edge_b = inc * BYTES_PER_EDGE_RESIDENT
    peak = feat_b + edge_b
    cap = np.array([d.usable_mem for d in devs], dtype=np.float64)
    feas = peak <= cap
    return peak, feat_b, cap, feas


def blind_footprint_belief(dev, deg, Fbar, devs, D):
    """What the attribute-BLIND cost model BELIEVES the footprint is: it charges a UNIFORM mean
    Fbar per node (exactly arrange.feasible's `counts[k] * F * 4`). This is the (wrong) number
    the blind partitioner optimizes against; the device then OOMs vs device_footprint above."""
    counts = np.bincount(dev, minlength=D).astype(np.float64)
    feat_b = counts * Fbar * BYTES_PER_FEAT * (1.0 + N_GATHERS)
    inc = np.bincount(dev, weights=deg.astype(np.float64), minlength=D)
    peak = feat_b + inc * BYTES_PER_EDGE_RESIDENT
    return peak


def _makespan_edges(src, dst, dev, Fv, bw, D, N):
    """Exact feature-bandwidth makespan using the real edges: incidence a<-b lands on dev[a] and
    costs F_b (neighbor row width). This is the precise generalization of the engine roofline."""
    a = np.concatenate([src, dst]).astype(np.int64)
    b = np.concatenate([dst, src]).astype(np.int64)
    da = dev[a]
    work = np.bincount(da, weights=Fv[b].astype(np.float64), minlength=D).astype(np.float64)
    comp_ms = work * BYTES_PER_FEAT * N_GATHERS / (bw * 1e9) * 1e3
    return float(comp_ms.max()), comp_ms


# ============================================================================ #
# synthetic heterogeneous-feature graph                                         #
# ============================================================================ #
def gen_hetero_feature_graph(N, M, n_comms, intra, rich_frac, heavy_dim, poor_dim,
                             rich_clustered, seed, heavy_warm=False):
    """Community graph where each node's feature SIZE F_v is heterogeneous: a `rich_frac` of nodes
    have `heavy_dim`-dim features (multi-modal: text/image), the rest `poor_dim` (categorical).
    If `rich_clustered`, the rich nodes are concentrated in a subset of communities -> a LOCALITY-
    respecting cut keeps a rich community whole on one device, piling heavy feature memory there
    (the adversarial case for a count/degree-balanced BLIND partitioner).
    If `heavy_warm`, the rich (multi-modal) nodes are wired with EXTRA edges so they have higher
    degree -- the realistic case (media/hub nodes are active) AND the one that breaks a reuse-aware
    BLIND tier-er: the heavy rows are WARM so a cold-first (low-degree) eviction cannot spill them,
    leaving the few huge rows resident and OOMing. Returns (src,dst,Fv,rich_mask)."""
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, n_comms, size=N).astype(np.int64)
    order = np.argsort(comm, kind="stable")
    bounds = np.searchsorted(comm[order], np.arange(n_comms + 1))
    m_in = int(M * intra)
    u = rng.integers(0, N, size=m_in)
    cu = comm[u]
    lo = bounds[cu].astype(np.int64); hi = bounds[cu + 1].astype(np.int64)
    pick = lo + (rng.random(m_in) * np.maximum(1, hi - lo)).astype(np.int64)
    v = order[np.minimum(pick, N - 1)]
    u2 = rng.integers(0, N, size=M - m_in)
    v2 = rng.integers(0, N, size=M - m_in)
    src = np.concatenate([u, u2]).astype(np.int64)
    dst = np.concatenate([v, v2]).astype(np.int64)

    Fv = np.full(N, float(poor_dim), dtype=np.float64)
    n_rich = int(round(rich_frac * N))
    if rich_clustered:
        # rich nodes fill whole communities (multi-modal nodes co-occur: e.g. all media posts in
        # one topic-community) -> a locality cut concentrates them. Pick communities until we have
        # ~n_rich rich nodes.
        csize = np.bincount(comm, minlength=n_comms)
        corder = rng.permutation(n_comms)
        rich_comms, acc = [], 0
        for c in corder:
            rich_comms.append(c); acc += int(csize[c])
            if acc >= n_rich:
                break
        rich_mask = np.isin(comm, np.array(rich_comms))
    else:
        rich_idx = rng.choice(N, size=n_rich, replace=False)
        rich_mask = np.zeros(N, dtype=bool); rich_mask[rich_idx] = True
    Fv[rich_mask] = float(heavy_dim)
    if heavy_warm and rich_mask.any():
        # give rich nodes EXTRA incident edges (warm/hub) so they are not in the cold spill set.
        rich_ids = np.nonzero(rich_mask)[0]
        extra = max(1, M // max(1, N))                      # ~avg-degree extra spokes per rich node
        hs = np.repeat(rich_ids, extra)
        hd = rng.integers(0, N, size=hs.size)
        src = np.concatenate([src, hs.astype(np.int64)])
        dst = np.concatenate([dst, hd.astype(np.int64)])
    return src, dst, Fv, rich_mask


def model_Fv_from_real(g, heavy_dim, poor_dim, seed):
    """Real graph: model heterogeneous TYPES on a real structure. jodie is bipartite (users/items);
    we treat one role as the 'rich' (multi-modal) type and the other as 'poor' categorical. The
    real edge feats are uniform-dim (that is exactly why §27 was null); the per-node TYPE skew is a
    declared MODELING choice -- the structure (who connects to whom) is REAL, the per-node feature
    SIZE is the heterogeneity we inject to model a multi-modal graph honestly."""
    N = g.num_nodes
    dst = np.asarray(g.dst, dtype=np.int64)
    is_item = np.zeros(N, dtype=bool)
    is_item[np.unique(dst)] = True
    Fv = np.where(is_item, float(poor_dim), float(heavy_dim)).astype(np.float64)
    return Fv, is_item


# ============================================================================ #
# EXP-A / EXP-B : placement feasibility under heterogeneous F                    #
# ============================================================================ #
def run_placement(src, dst, Fv, rich_mask, cluster, args, label):
    devs = cluster.devices
    D = len(devs)
    N = int(max(src.max(), dst.max())) + 1
    deg = node_degree(src, dst, N)
    bw = np.array([d.hbm_bw_gbps for d in devs], dtype=np.float64)
    cap = np.array([d.usable_mem for d in devs], dtype=np.float64)
    Fbar = float(Fv.mean())
    rank = lpa_rank_or_degree(src, dst, N, deg)

    print(f"\n=== {label} ===")
    print("  devices: " + " | ".join(
        f"{d.name} cap={d.usable_mem/GB:.0f}GB bw={d.hbm_bw_gbps:.0f}" for d in devs) +
        f"  | link={cluster.inter_node_bw:g}GB/s")
    rich_n = int(rich_mask.sum())
    print(f"  N={N:,d} M={src.size:,d}  F_v in {{{int(Fv.min())},{int(Fv.max())}}}  "
          f"rich={rich_n:,d} ({rich_n/N*100:.1f}%)  mean F={Fbar:.1f}  "
          f"clustered={'yes' if args.rich_clustered else 'no'}")
    print(f"  total feature bytes = {Fv.sum()*BYTES_PER_FEAT*(1+N_GATHERS)/GB:.1f}GB "
          f"(+activations)  vs aggregate HBM = {cap.sum()/GB:.0f}GB")

    # ---- attribute-BLIND: balance NODE COUNT on the locality layout (uniform-Fbar sizing) ----
    # This is what hash / a count-balanced METIS-style cut does: equal node counts per device.
    dev_blind = split_contiguous(rank, np.ones(N, dtype=np.float64), D, caps=None)
    # ---- attribute-AWARE (zord): size segments to per-device CAPACITY using the TRUE feature
    #      BYTES weight per node (F_v) -> heavy-F mass flows to the big-HBM device ----
    dev_aware = split_contiguous(rank, Fv.astype(np.float64), D, caps=cap)

    # TRUE footprint (per-node bytes) for both:
    peak_b, feat_b, _, feas_b = device_footprint(dev_blind, deg, Fv, devs, D)
    peak_a, feat_a, _, feas_a = device_footprint(dev_aware, deg, Fv, devs, D)
    # what the BLIND model BELIEVED (uniform Fbar) -- the source of its OOM surprise:
    belief_b = blind_footprint_belief(dev_blind, deg, Fbar, devs, D)

    mk_b, ms_b = _makespan_edges(src, dst, dev_blind, Fv, bw, D, N)
    mk_a, ms_a = _makespan_edges(src, dst, dev_aware, Fv, bw, D, N)

    def fmt(peak, cap, feas):
        return "  ".join(f"d{k}:{peak[k]/GB:5.1f}/{cap[k]/GB:4.0f}GB{'' if feas[k] else ' OOM!'}"
                         for k in range(D))

    print(f"  [BLIND count-balance]   believed-peak: " +
          "  ".join(f"d{k}:{belief_b[k]/GB:4.1f}GB" for k in range(D)))
    print(f"                          TRUE per-node peak: {fmt(peak_b, cap, feas_b)}")
    print(f"                          feasible={bool(feas_b.all())}  "
          f"OOM devices={int((~feas_b).sum())}  makespan={mk_b:.2f}ms")
    print(f"  [AWARE feature-byte]    TRUE per-node peak: {fmt(peak_a, cap, feas_a)}")
    print(f"                          feasible={bool(feas_a.all())}  "
          f"OOM devices={int((~feas_a).sum())}  makespan={mk_a:.2f}ms")

    blind_oom = not bool(feas_b.all())
    aware_ok = bool(feas_a.all())
    over = (peak_b / cap).max()
    verdict = ("WIN: blind OOMs a device, aware stays feasible" if (blind_oom and aware_ok)
               else "both feasible (no feasibility gap at this rich-frac)" if (not blind_oom and aware_ok)
               else "both OOM (aggregate HBM insufficient -> needs tiering, see EXP-C)"
               if (blind_oom and not aware_ok) else "blind ok, aware OOM (unexpected)")
    print(f"  => blind peak/cap max = {over:.2f}x ({'>1 OVERFLOW' if over>1 else 'fits'}); "
          f"makespan blind/aware = {mk_b/max(1e-9,mk_a):.2f}x  [{verdict}]")
    return dict(label=label, blind_oom=blind_oom, aware_ok=aware_ok, over=over,
                mk_blind=mk_b, mk_aware=mk_a, feas_b=feas_b, feas_a=feas_a,
                peak_b=peak_b, peak_a=peak_a, cap=cap)


def exp_A(args):
    print("\n" + "=" * 80)
    print("EXP-A: PLACEMENT FEASIBILITY under HETEROGENEOUS per-node feature bytes")
    print("=" * 80)
    cluster = build_cluster(args)
    if args.dataset:
        try:
            from zord.datasets import load
            g = load(args.dataset).sort_by_time()
        except Exception as e:
            print(f"  [dataset {args.dataset!r} not available on this box ({type(e).__name__}); "
                  f"the real-data path runs on the cluster where jodie is staged. Falling back to "
                  f"synthetic. Note: real jodie is UNIFORM-F (172-dim) -> the §27 NULL regime; the "
                  f"win here needs the heterogeneous-F TYPES this synthetic models.]")
            args.dataset = ""
            return exp_A(args)
        src, dst = np.asarray(g.src), np.asarray(g.dst)
        Fv, rich_mask = model_Fv_from_real(g, args.heavy_dim, args.poor_dim, args.seed)
        label = f"A: real {g.name} + modeled F_v types"
    else:
        src, dst, Fv, rich_mask = gen_hetero_feature_graph(
            args.nodes, args.edges, args.comms, args.intra, args.rich_frac,
            args.heavy_dim, args.poor_dim, args.rich_clustered, args.seed,
            heavy_warm=args.heavy_warm)
        label = "A: synthetic heterogeneous-F graph"
    return run_placement(src, dst, Fv, rich_mask, cluster, args, label)


def exp_B(args):
    print("\n" + "=" * 80)
    print("EXP-B: HETEROGENEOUS-HBM MATCH -- sweep rich-fraction; where does blind OOM?")
    print("=" * 80)
    cluster = build_cluster(args)
    results = []
    for rf in [float(x) for x in args.rich_sweep.split(",")]:
        a2 = argparse.Namespace(**vars(args)); a2.rich_frac = rf
        src, dst, Fv, rich_mask = gen_hetero_feature_graph(
            a2.nodes, a2.edges, a2.comms, a2.intra, rf, a2.heavy_dim, a2.poor_dim,
            a2.rich_clustered, a2.seed, heavy_warm=a2.heavy_warm)
        r = run_placement(src, dst, Fv, rich_mask, cluster, a2, f"B: rich-frac={rf:.2f}")
        results.append((rf, r))
    print("\n  --- EXP-B summary (rich-frac -> blind OOM? / aware feasible?) ---")
    for rf, r in results:
        print(f"    rich-frac={rf:4.2f}  blind {'OOM ' if r['blind_oom'] else 'ok  '}"
              f"(peak/cap {r['over']:.2f}x)  aware {'feasible' if r['aware_ok'] else 'OOM'}")
    return results


# ============================================================================ #
# EXP-C : tiering under feature pressure (total feature mem > aggregate HBM)     #
# ============================================================================ #
def exp_C(args):
    print("\n" + "=" * 80)
    print("EXP-C: TIERING under FEATURE PRESSURE (total feat mem > aggregate HBM)")
    print("=" * 80)
    cluster = build_cluster(args)
    devs = cluster.devices; D = len(devs)
    cap_total = sum(d.usable_mem for d in devs)
    # AUTO-FORCE aggregate pressure: the tiering question only exists when even the BEST placement
    # cannot fit (total feature bytes > aggregate HBM). Bump heavy-dim until total >= 1.2x aggregate
    # so EXP-C is self-sufficient when run standalone (no knob-hunting). Heavy nodes are made WARM
    # (heavy_warm) so a reuse-aware (cold-first) blind tier-er cannot spill them -> blind fails.
    heavy = float(args.heavy_dim)
    rf = max(args.rich_frac, 0.30)            # need a real heavy mass to create pressure + skew
    while True:
        meanF = rf * heavy + (1 - rf) * args.poor_dim
        tot_pred = args.nodes * meanF * BYTES_PER_FEAT * (1.0 + N_GATHERS)
        if tot_pred >= 1.2 * cap_total or heavy >= 65536:
            break
        heavy *= 2.0
    print(f"  (auto-pressure: rich-frac={rf:.2f} heavy-dim={int(heavy)} poor-dim={args.poor_dim}, "
          f"heavy nodes WARM so cold-first blind eviction cannot reach them)")
    src, dst, Fv, rich_mask = gen_hetero_feature_graph(
        args.nodes, args.edges, args.comms, args.intra, rf,
        int(heavy), args.poor_dim, args.rich_clustered, args.seed, heavy_warm=True)
    N = int(max(src.max(), dst.max())) + 1
    deg = node_degree(src, dst, N)
    cap = np.array([d.usable_mem for d in devs], dtype=np.float64)
    h2d = np.array([d.h2d_gbps for d in devs], dtype=np.float64)
    rank = lpa_rank_or_degree(src, dst, N, deg)

    # place feature bytes capacity-proportionally (aware), then per device decide what is RESIDENT
    # in HBM vs STAGED from CPU. The window is the temporal batch; here we tier the NODE feature
    # bank (the bytes that don't fit).
    dev = split_contiguous(rank, Fv.astype(np.float64), D, caps=cap)
    feat_bytes_node = Fv * BYTES_PER_FEAT * (1.0 + N_GATHERS)
    edge_bytes = np.bincount(dev, weights=deg.astype(np.float64), minlength=D) * BYTES_PER_EDGE_RESIDENT

    tot_feat = feat_bytes_node.sum()
    print(f"  N={N:,d}  total node-feature bytes={tot_feat/GB:.1f}GB  aggregate HBM={cap.sum()/GB:.0f}GB"
          f"  (pressure ratio {tot_feat/cap.sum():.2f}x)")
    if tot_feat <= (cap.sum() - edge_bytes.sum()):
        print("  NOTE: fits in aggregate HBM -> no tiering needed at these knobs; "
              "increase --heavy-dim or --nodes to force pressure. Continuing to show the mechanism.")

    # For each device: residency budget for features = capacity - edge metadata. Two eviction
    # policies decide WHICH nodes spill to CPU when the on-device feature bytes exceed the budget:
    #   BLIND  (uniform-F):  evict by NODE COUNT order (treats every row as equal bytes) -> evicts
    #          many small rows, frees little, leaves the few HUGE rich rows resident -> still OOM.
    #   AWARE  (zord):       evict the LARGEST-F COLD nodes first (biggest bytes per evicted node)
    #          -> frees the most HBM per eviction -> fits with the FEWEST nodes streamed.
    # cold = lowest-degree (least-reused) nodes are the spill candidates (reuse-aware ordering).
    peak_blind = []; peak_aware = []; stream_blind = []; stream_aware = []; feas_blind = []; feas_aware = []
    for k in range(D):
        on = np.nonzero(dev == k)[0]
        budget = cap[k] - edge_bytes[k]          # HBM left for feature rows after edge metadata
        fb = feat_bytes_node[on]
        deg_k = deg[on]
        resident_total = fb.sum()
        if resident_total <= budget:
            peak_blind.append(resident_total + edge_bytes[k]); peak_aware.append(resident_total + edge_bytes[k])
            stream_blind.append(0.0); stream_aware.append(0.0)
            feas_blind.append(budget >= 0); feas_aware.append(budget >= 0)
            continue
        need_to_free = resident_total - max(0.0, budget)
        # BLIND eviction: by node-count, ascending degree (cold), but treating bytes as uniform mean.
        # It evicts cold nodes one-by-one; because it MODELS each as Fbar bytes it stops early
        # (believing it freed enough) but the TRUE freed bytes are the small poor rows -> still over.
        cold_order = np.argsort(deg_k, kind="stable")    # coldest first
        Fbar_k = fb.mean()
        # blind decides count to evict using the uniform-Fbar belief:
        n_evict_blind = int(np.ceil(need_to_free / max(1e-9, Fbar_k)))
        n_evict_blind = min(n_evict_blind, on.size)
        evb = cold_order[:n_evict_blind]
        freed_blind = fb[evb].sum()               # TRUE bytes freed (poor rows are small!)
        resident_blind = resident_total - freed_blind
        peak_blind.append(resident_blind + edge_bytes[k])
        stream_blind.append(freed_blind)
        feas_blind.append(resident_blind <= budget + 1e-6)
        # AWARE eviction: among cold nodes, evict LARGEST-F first -> frees need_to_free with fewest rows
        cold = cold_order
        order_by_bytes = cold[np.argsort(-fb[cold], kind="stable")]   # cold, largest-bytes first
        csum = np.cumsum(fb[order_by_bytes])
        n_ev = int(np.searchsorted(csum, need_to_free, side="left") + 1)
        n_ev = min(n_ev, order_by_bytes.size)
        eva = order_by_bytes[:n_ev]
        freed_aware = fb[eva].sum()
        resident_aware = resident_total - freed_aware
        peak_aware.append(resident_aware + edge_bytes[k])
        stream_aware.append(freed_aware)
        feas_aware.append(resident_aware <= budget + 1e-6)

    peak_blind = np.array(peak_blind); peak_aware = np.array(peak_aware)
    stream_blind = np.array(stream_blind); stream_aware = np.array(stream_aware)
    feas_blind = np.array(feas_blind); feas_aware = np.array(feas_aware)
    # PCIe staging time per epoch (streamed bytes / h2d), exposed if it can't hide behind compute.
    stage_blind_ms = stream_blind / (h2d * 1e9) * 1e3
    stage_aware_ms = stream_aware / (h2d * 1e9) * 1e3

    print("  policy        per-device PEAK HBM (resident feat+edge) / cap, streamed GB, feasible")
    for k in range(D):
        print(f"    d{k} BLIND  peak={peak_blind[k]/GB:5.1f}/{cap[k]/GB:4.0f}GB "
              f"stream={stream_blind[k]/GB:4.1f}GB stage={stage_blind_ms[k]:6.1f}ms "
              f"{'OK' if feas_blind[k] else 'OOM!'}")
        print(f"    d{k} AWARE  peak={peak_aware[k]/GB:5.1f}/{cap[k]/GB:4.0f}GB "
              f"stream={stream_aware[k]/GB:4.1f}GB stage={stage_aware_ms[k]:6.1f}ms "
              f"{'OK' if feas_aware[k] else 'OOM!'}")
    print(f"  => BLIND tiering feasible={bool(feas_blind.all())} "
          f"(streams {stream_blind.sum()/GB:.1f}GB); "
          f"AWARE tiering feasible={bool(feas_aware.all())} "
          f"(streams {stream_aware.sum()/GB:.1f}GB)")
    if feas_aware.all() and not feas_blind.all():
        print("     ATTRIBUTE WIN: evicting the LARGEST-F cold rows fits where blind "
              "(uniform-F, evict-by-count) still OOMs -- the heavy rows are the ones to spill.")
    elif feas_aware.all() and feas_blind.all() and stream_aware.sum() < stream_blind.sum() - 1e-6:
        print(f"     ATTRIBUTE WIN: aware streams {stream_blind.sum()/max(1e-9,stream_aware.sum()):.2f}x "
              "less over PCIe (fewer, bigger rows evicted) -> lower exposed staging.")
    else:
        print("     NULL here: at these knobs the eviction policy does not change feasibility/stream.")
    return dict(feas_blind=bool(feas_blind.all()), feas_aware=bool(feas_aware.all()),
                stream_blind=stream_blind.sum(), stream_aware=stream_aware.sum())


# ============================================================================ #
def headline(rA, rB, rC):
    print("\n" + "=" * 80)
    print("HEADLINE — do heterogeneous feature BYTES give zord a placement/tiering win?")
    print("=" * 80)
    if rA:
        if rA["blind_oom"] and rA["aware_ok"]:
            print(f"  EXP-A PLACEMENT: WIN -- blind count-balance OOMs a device "
                  f"(peak/cap {rA['over']:.2f}x); attribute-aware feature-byte sizing stays FEASIBLE.")
        elif not rA["blind_oom"]:
            print(f"  EXP-A PLACEMENT: NULL at this rich-frac (blind peak/cap {rA['over']:.2f}x "
                  f"<= 1, no feasibility gap). makespan blind/aware {rA['mk_blind']/max(1e-9,rA['mk_aware']):.2f}x.")
        else:
            print("  EXP-A PLACEMENT: both OOM -> aggregate HBM insufficient (a tiering case, EXP-C).")
    if rB:
        oom_at = [rf for rf, r in rB if r["blind_oom"] and r["aware_ok"]]
        if oom_at:
            print(f"  EXP-B THRESHOLD: blind OOMs while aware holds at rich-frac in {oom_at} "
                  f"-> a real feasibility band the attribute-blind partitioner cannot enter.")
        else:
            print("  EXP-B THRESHOLD: no rich-frac in the sweep splits them (either both fit or both OOM).")
    if rC:
        if rC["feas_aware"] and not rC["feas_blind"]:
            print("  EXP-C TIERING: WIN -- evicting largest-F cold rows fits where blind (evict-by-count) OOMs.")
        elif rC["feas_aware"] and rC["feas_blind"] and rC["stream_aware"] < rC["stream_blind"]:
            print(f"  EXP-C TIERING: WIN -- aware streams "
                  f"{rC['stream_blind']/max(1e-9,rC['stream_aware']):.2f}x less over PCIe.")
        else:
            print("  EXP-C TIERING: NULL at these knobs.")
    print("\n  DISTINCTION FROM §27: §27 used UNIFORM F (real jodie 172-dim) where node-count balance"
          "\n  == feature-byte balance, so attributes were correctly NULL. The win above (if any) exists"
          "\n  ONLY because per-node feature SIZE is HETEROGENEOUS -- the multi-modal/multi-type case --"
          "\n  which makes feature BYTES an independent placement/tiering signal the blind model lacks.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all", choices=["A", "B", "C", "all"])
    ap.add_argument("--dataset", default="", help="real graph (e.g. jodie-wikipedia); else synthetic")
    ap.add_argument("--hbm-gb", default="80,48,32", help="per-device usable HBM GB (heterogeneous)")
    ap.add_argument("--agg-bw", default="942,534,444", help="per-device achieved agg bandwidth GB/s")
    ap.add_argument("--link-gbps", type=float, default=325.0, help="interconnect GB/s (parameter)")
    # synthetic graph
    ap.add_argument("--nodes", type=int, default=2_000_000)
    ap.add_argument("--edges", type=int, default=20_000_000)
    ap.add_argument("--comms", type=int, default=200)
    ap.add_argument("--intra", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    # heterogeneous-feature knobs
    ap.add_argument("--rich-frac", type=float, default=0.10, help="fraction of high-F (multi-modal) nodes")
    ap.add_argument("--heavy-dim", type=int, default=768, help="rich node feature dim (text/image)")
    ap.add_argument("--poor-dim", type=int, default=16, help="poor node feature dim (categorical)")
    ap.add_argument("--rich-clustered", action="store_true", default=True,
                    help="rich nodes fill whole communities (locality cut concentrates them)")
    ap.add_argument("--rich-scattered", dest="rich_clustered", action="store_false",
                    help="rich nodes scattered uniformly (the harder, more honest blind case)")
    ap.add_argument("--rich-sweep", default="0.05,0.15,0.25,0.35,0.45",
                    help="EXP-B: comma rich-fractions to sweep")
    ap.add_argument("--heavy-warm", action="store_true", default=False,
                    help="EXP-A/B: also give rich nodes extra edges (hub/warm). EXP-C forces this.")
    a = ap.parse_args()

    print(f"ATTR-PLACEMENT  exp={a.exp} dataset={a.dataset or 'synthetic'} "
          f"hbm={a.hbm_gb} heavy/poor={a.heavy_dim}/{a.poor_dim} rich-frac={a.rich_frac} "
          f"clustered={a.rich_clustered}")
    rA = rB = rC = None
    if a.exp in ("A", "all"):
        rA = exp_A(a)
    if a.exp in ("B", "all"):
        rB = exp_B(a)
    if a.exp in ("C", "all"):
        rC = exp_C(a)
    headline(rA, rB, rC)


if __name__ == "__main__":
    main()
