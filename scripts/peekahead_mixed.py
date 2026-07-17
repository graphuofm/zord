#!/usr/bin/env python
"""Two process-only bets on the space-time cut DUALITY (cut/cost metrics, NEVER accuracy).
Mirrors scripts/duality_frontier.py: same supra-graph, same SpatialCut/TemporalCut counting,
same balanced-blocks-from-a-C++-ordering trick. CPU-only, vectorized numpy, C++ kernel for
graph orderings (degree/kcore/lpa). NEVER networkx.

BET D5  PEEK-AHEAD partitioning
  We train in BATCH, not streaming, so the partitioner is ALLOWED to look at FUTURE snapshots.
  Question: how much cut does that future knowledge actually buy? We compute a VERTEX partition
  two ways and SCORE BOTH over the FULL window:
    (a) CAUSAL / streaming-style: cluster vertices using ONLY edges in the first k% of snapshots.
    (b) PEEK-AHEAD: cluster vertices using the WHOLE window.
  We report the SpatialCut+TemporalCut each induces over the full window, the peek-ahead SAVING,
  and the AMORTIZED partition cost (one clustering pass amortized over the snapshots reused). k swept.

BET D10  MIXED-GRANULARITY (per-region duality)
  Different regions want different cut granularity. Cut the DENSE CORE by VERTEX (PTS-like: keep
  each core node's whole timeline on one device, Dt=1 there) and the SPARSE PERIPHERY by SNAPSHOT
  (PSS-like: spread the many low-degree nodes across snapshot-blocks, Dv=1 there). Core/periphery
  are classified by degree or by k-core degeneracy via the C++ tool. We count the resulting
  SpatialCut+TemporalCut of this HYBRID supra-partition and compare against pure-PSS, pure-PTS,
  and the best uniform Dv x Dt factorization (the duality_frontier corners + interior), under a
  hardware weight regime (w_S, w_T).

  python scripts/peekahead_mixed.py --dataset collegemsg --snapshots 64 --devices 64 --feat 128 --ws 1 --wt 1
  python scripts/peekahead_mixed.py --synthetic --snapshots 64 --devices 64 --feat 128 --ws 1 --wt 25
"""
import argparse, os, struct, subprocess, time
import numpy as np

# Same binary contract as duality_frontier.py. Fall back to a co-located build/graph_algos so the
# script works in this checkout (the duality_frontier default path is the HetCluster staging dir).
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_BIN = os.path.join(os.path.dirname(_HERE), "build", "graph_algos")
BIN = os.environ.get("ZORD_GRAPH_BIN", _LOCAL_BIN if os.path.exists(_LOCAL_BIN)
                     else "build/graph_algos")


# --------------------------------------------------------------------------------------------------
# C++ graph-ordering kernel (degree | kcore | lpa | ...). Returns newid[old_node] = contiguous rank.
# Identical IO contract to duality_frontier.cpp_lpa_order; generalized over the mode + edge subset.
# --------------------------------------------------------------------------------------------------
def cpp_order(src, dst, N, mode="lpa", tag="run"):
    """Run the C++ ordering kernel on (src,dst,N) and return (newid[N], wall_seconds).
    Gracefully falls back to identity order if the binary is missing/failed so the script still
    reports cut numbers (only the ordering QUALITY degrades, not the methodology)."""
    if src.size == 0:
        return np.arange(N, dtype=np.int32), 0.0
    e = np.empty(2 * src.size, dtype=np.int32); e[0::2] = src.astype(np.int32); e[1::2] = dst.astype(np.int32)
    inp = f"/tmp/zord_pam_{tag}_edges.bin"; outp = f"/tmp/zord_pam_{tag}_perm.bin"
    with open(inp, "wb") as f:
        f.write(struct.pack("<qq", N, src.size)); e.tofile(f)
    t = time.time()
    if not os.path.exists(BIN):
        print(f"  [cpp {mode}] binary not found at {BIN}; using identity order")
        return np.arange(N, dtype=np.int32), 0.0
    r = subprocess.run([BIN, inp, mode, outp], capture_output=True, text=True)
    cost = time.time() - t
    if r.returncode != 0:
        print(f"  [cpp {mode}] FAILED: {r.stderr.strip()[:160]}; using identity order")
        return np.arange(N, dtype=np.int32), cost
    with open(outp, "rb") as f:
        struct.unpack("<q", f.read(8)); newid = np.fromfile(f, dtype=np.int32, count=N)
    return newid, cost


# --------------------------------------------------------------------------------------------------
# Cut counting -- byte-for-byte the same definitions as duality_frontier.py.
#   SpatialCut(vblock)  = # spatial/aggregation edges whose endpoints land in different vertex-blocks
#   TemporalCut(Dt)     = sum over vertices of (#distinct snapshot-blocks the vertex is active in - 1)
# Both are evaluated over a chosen edge range [lo:hi) of the time-sorted graph (the "window").
# --------------------------------------------------------------------------------------------------
def spatial_cut(vblock, src, dst):
    if src.size == 0:
        return 0
    return int(np.count_nonzero(vblock[src] != vblock[dst]))


def temporal_cut(verts, vsnapblock, N, Dt):
    """verts, vsnapblock are the per-endpoint vertex id and its snapshot-block id (aligned arrays)."""
    if verts.size == 0:
        return 0
    key = verts.astype(np.int64) * Dt + vsnapblock.astype(np.int64)
    uniq = np.unique(key)
    distinct_per_v = np.bincount((uniq // Dt).astype(np.int64), minlength=N)
    return int(np.maximum(distinct_per_v - 1, 0).sum())


def balanced_blocks(newid, N, Dv):
    """Cut the C++ ordering into Dv equal-size contiguous vertex-blocks (same trick as the frontier)."""
    if Dv <= 1:
        return np.zeros(N, dtype=np.int32)
    return (newid.astype(np.int64) * Dv // N).astype(np.int32)


def factorizations(D):
    out = []; dt = 1
    while dt <= D:
        if D % dt == 0:
            out.append((D // dt, dt))
        dt *= 2
    return out


# --------------------------------------------------------------------------------------------------
# BET D5: PEEK-AHEAD vs CAUSAL vertex partitioning.
# A vertex partition is a clustering of nodes into D blocks (pure-vertex granularity, Dt=1: that is
# the regime where the *vertex clustering quality* is what matters). We build the clustering from a
# prefix of the window (causal) or the whole window (peek-ahead), then score the induced SpatialCut
# (full window) + TemporalCut (here Dt=1 so temporal cut is 0; we keep the full-duality score under
# the weight regime for honesty). The interesting signal is how much SpatialCut peek-ahead removes.
# --------------------------------------------------------------------------------------------------
def bet_peekahead(src, dst, snap, N, S, D, F, wS, wT, kpcts):
    print("\n" + "=" * 96)
    print("BET D5  PEEK-AHEAD vs CAUSAL vertex partitioning  (BATCH lets the partitioner see the future)")
    print("=" * 96)
    bytes_per = F * 4
    E = src.size
    # PEEK-AHEAD reference: cluster on the WHOLE window once.
    pa_id, pa_cost = cpp_order(src, dst, N, mode="lpa", tag="peek_full")
    pa_vblock = balanced_blocks(pa_id, N, D)
    pa_spatial = spatial_cut(pa_vblock, src, dst)        # scored over the FULL window
    print(f"  scoring window = full (E={E:,}), partition granularity = pure-vertex (Dv={D}, Dt=1)")
    print(f"  PEEK-AHEAD: cluster on 100% of snapshots once  -> FULL-window SpatialCut = {pa_spatial:,} "
          f"(clustering {pa_cost:.2f}s)")
    print()
    print(f"  {'k%':>5} {'prefixE':>12} {'causalCut(full)':>16} {'peekCut(full)':>14} "
          f"{'cut_saved':>12} {'%saved':>8} {'causal_cost':>12} {'amort/snap':>11}")
    print(f"  ('seenFrac' = fraction of nodes the causal prefix has even observed; a streaming "
          f"partitioner cannot place a node it has not seen -> it goes to a default/overflow block.)")
    rows = []
    for k in kpcts:
        # Prefix = first k% of snapshots (causal: the partitioner has seen snapshots [0, ks)).
        ks = max(1, int(round(S * k / 100.0)))
        ks = min(ks, S)
        pe = int(np.searchsorted(snap, ks, side="left"))   # first edge index with snap >= ks
        pe = max(pe, 1)
        psrc, pdst = src[:pe], dst[:pe]
        c_id, c_cost = cpp_order(psrc, pdst, N, mode="lpa", tag="causal")
        # REALISTIC streaming layout: cluster ONLY the nodes seen in the prefix into D balanced blocks
        # (by their LPA order among seen nodes); nodes never seen in the prefix CANNOT be placed by a
        # streaming partitioner, so they fall to a default block (0). This is precisely the locality
        # the partitioner forfeits by not peeking ahead.
        seen = np.zeros(N, dtype=bool)
        seen[psrc] = True; seen[pdst] = True
        seen_nodes = np.nonzero(seen)[0]
        c_vblock = np.zeros(N, dtype=np.int64)             # default/overflow block for unseen nodes
        if seen_nodes.size:
            rank = np.argsort(np.argsort(c_id[seen_nodes], kind="stable"), kind="stable")
            c_vblock[seen_nodes] = (rank.astype(np.int64) * D // seen_nodes.size)
        c_spatial = spatial_cut(c_vblock, src, dst)        # CAUSAL partition, scored over FULL window
        saved = c_spatial - pa_spatial
        pct = 100.0 * saved / c_spatial if c_spatial else 0.0
        seen_frac = seen_nodes.size / N
        # Amortized cost: one clustering pass amortized over the (S-ks) future snapshots it is reused on.
        reused = max(1, S - ks)
        amort = c_cost / reused
        rows.append((k, ks, pe, c_spatial, seen_frac, pa_spatial, saved, pct, c_cost, amort))
        print(f"  {k:>5} {pe:>12,} {c_spatial:>16,} {pa_spatial:>14,} {saved:>12,} {pct:>7.1f}% "
              f"{c_cost:>11.2f}s {amort*1e3:>9.1f}ms  seenFrac={seen_frac:5.2f}")
    # Weighted-cost view: convert the SpatialCut delta into the duality weight regime (Dt=1 -> Tcut=0).
    best_causal = min(r[3] for r in rows)
    print()
    print(f"  weight regime (w_S={wS}, w_T={wT}); pure-vertex so TemporalCut=0, cost = SpatialCut * w_S")
    pa_cost_w = pa_spatial * wS
    bc_cost_w = best_causal * wS
    gain = bc_cost_w / pa_cost_w if pa_cost_w else float("inf")
    print(f"    best CAUSAL prefix weighted-cost = {bc_cost_w:,.0f}   "
          f"PEEK-AHEAD weighted-cost = {pa_cost_w:,.0f}   peek/causal gain = {gain:.3f}x")
    full_saved = best_causal - pa_spatial
    print(f"    => peek-ahead removes {full_saved:,} crossing edges vs the best causal prefix "
          f"({100.0*full_saved/best_causal if best_causal else 0:.1f}% of SpatialCut)")
    # Translate the saved cut to transferred bytes per epoch (process metric, not accuracy).
    print(f"    transfer saved per epoch (spatial edges * {bytes_per}B): "
          f"{full_saved * bytes_per / 1024**2:,.1f} MiB")
    return rows


# --------------------------------------------------------------------------------------------------
# BET D10: MIXED-GRANULARITY supra-partition.
# Classify nodes into CORE (dense) and PERIPHERY (sparse) via degree or k-core degeneracy (C++).
#   CORE  cells are cut by VERTEX  (PTS-like): a core node's whole timeline stays on its vertex-block
#         -> contributes to SpatialCut (its crossing spatial edges) but NOT to TemporalCut.
#   PERIPH cells are cut by SNAPSHOT (PSS-like): a periphery node is placed by snapshot-block
#         -> contributes to TemporalCut (distinct snapshot-blocks - 1) but its spatial edges only
#            cut when they leave the (single) periphery snapshot-block partition.
# We assign device blocks as: core gets Dc vertex-blocks, periphery gets Dp snapshot-blocks, with
# Dc + Dp = D. We SWEEP the (Dc, Dp) split and keep the best, then count the exact
# SpatialCut + TemporalCut of this hybrid and compare to the uniform corners/interior.
# --------------------------------------------------------------------------------------------------
def mixed_cut(src, dst, snap, verts, snaps, N, S, D, core_mask, core_order, Dc, Dp):
    """Count SpatialCut + TemporalCut of the hybrid (core-by-vertex, periphery-by-snapshot) layout
    for an EXPLICIT device split (Dc vertex-blocks for the core, Dp snapshot-blocks for periphery).
    Returns (spatial, temporal, detail dict)."""
    # ---- vertex-block id per node ----
    # Core nodes: balanced into Dc vertex-blocks using the C++ core ordering (good spatial locality).
    # Periphery nodes: NOT vertex-partitioned (single logical vertex-block); they are split by snapshot.
    vblock = np.full(N, -1, dtype=np.int64)
    core_nodes = np.nonzero(core_mask)[0]
    eff_Dc = max(1, Dc)  # core is always laid out by vertex; >=1 block even if the split gave it 0
    if core_nodes.size:
        # rank core nodes among themselves by their C++ order, then cut into eff_Dc balanced blocks
        order_rank = core_order[core_nodes]
        # dense-rank the core nodes by their ordering value
        cr = np.argsort(np.argsort(order_rank, kind="stable"), kind="stable")
        cb = (cr.astype(np.int64) * eff_Dc // max(1, core_nodes.size))
        vblock[core_nodes] = cb
    # Periphery share one vertex-block id (= eff_Dc, distinct from all core blocks) -- they are not cut
    # by vertex, so all periphery-internal spatial edges are "in the same vertex region".
    peri_nodes = np.nonzero(~core_mask)[0]
    PERI_VB = eff_Dc  # a single sentinel vertex-block for the whole periphery
    if peri_nodes.size:
        vblock[peri_nodes] = PERI_VB

    # ---- SpatialCut ----
    # An edge is spatially cut iff its endpoints fall in different vertex-blocks. Core nodes have real
    # Dc blocks; periphery nodes all share PERI_VB. So: core-core edges cut across Dc blocks, core-peri
    # edges cut (different block ids), peri-peri edges never cut spatially (they live in one vertex
    # region and are instead resolved by snapshot granularity -> counted in TemporalCut below).
    sc = int(np.count_nonzero(vblock[src] != vblock[dst]))

    # ---- TemporalCut ----
    # Only PERIPHERY nodes are cut by snapshot (Dp snapshot-blocks). Core nodes keep their whole
    # timeline local (Dt=1 for the core region) -> contribute 0 temporal cut. For periphery nodes we
    # count distinct snapshot-blocks they are active in, minus 1, exactly like duality_frontier.py.
    tc = 0
    if Dp >= 1 and peri_nodes.size:
        is_peri_endpoint = ~core_mask[verts]
        pv = verts[is_peri_endpoint]
        psn = snaps[is_peri_endpoint]
        sblk = np.minimum(psn.astype(np.int64) * Dp // S, Dp - 1)
        tc = temporal_cut(pv, sblk, N, Dp)
    detail = dict(ncore=int(core_mask.sum()), nper=int((~core_mask).sum()), Dc=eff_Dc, Dp=max(0, Dp))
    return sc, tc, detail


def bet_mixed(src, dst, snap, N, S, D, F, wS, wT, core_metric, core_frac):
    print("\n" + "=" * 96)
    print("BET D10  MIXED-GRANULARITY per-region duality  (core by VERTEX / periphery by SNAPSHOT)")
    print("=" * 96)
    bytes_per = F * 4
    verts = np.concatenate([src, dst]); snaps = np.concatenate([snap, snap])

    # ---- classify CORE vs PERIPHERY using the C++ tool ----
    # degree: C++ 'degree' ranks nodes by descending degree (rank 0 = highest degree). The top
    #         core_frac fraction of ranks = dense core.
    # kcore : C++ 'kcore' returns the degeneracy (k-core peeling) order; nodes removed LAST (highest
    #         rank) sit in the densest cores. So the top core_frac by rank are the high-core nodes.
    deg = np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)
    order_id, ord_cost = cpp_order(src, dst, N, mode=core_metric, tag="classify")
    n_core = max(1, int(round(N * core_frac)))
    if core_metric == "degree":
        # rank 0..n_core-1 are highest-degree -> core
        core_mask = order_id < n_core
    elif core_metric == "kcore":
        # degeneracy order: removed first = low rank = sparse; high rank = dense core.
        core_mask = order_id >= (N - n_core)
    else:
        core_mask = deg >= np.quantile(deg, 1.0 - core_frac)
    # The spatial ordering used to lay out the CORE into vertex-blocks: reuse the C++ LPA clustering
    # (the same good-spatial-partition primitive duality_frontier.py uses), restricted by the mask.
    lpa_id, lpa_cost = cpp_order(src, dst, N, mode="lpa", tag="mixed_lpa")

    ncore = int(core_mask.sum()); nper = N - ncore
    print(f"  classify by '{core_metric}' (C++): core_frac={core_frac:.2f} -> "
          f"CORE={ncore:,} nodes ({100.0*ncore/N:.1f}%), PERIPHERY={nper:,} nodes "
          f"(classify {ord_cost:.2f}s, lpa {lpa_cost:.2f}s)")
    core_deg = deg[core_mask]; per_deg = deg[~core_mask]
    print(f"  core avg degree = {core_deg.mean() if core_deg.size else 0:.1f}  "
          f"periphery avg degree = {per_deg.mean() if per_deg.size else 0:.2f}  "
          f"(core holds {100.0*core_deg.sum()/max(1,deg.sum()):.1f}% of all edge-endpoints)")

    # ---- MIXED layout: sweep the core vertex-block count Dc and the periphery snapshot-block count
    #      Dp INDEPENDENTLY (both <= D; Dp also <= S). Core and periphery share the same pool of D
    #      physical devices (a device can host a core vertex-block AND a periphery snapshot-block), so
    #      we do not force Dc+Dp=D -- the duality currency is the CUT, and each region picks its own
    #      granularity. We keep the lowest weighted-cost layout. ----
    print(f"\n  MIXED-GRANULARITY split sweep (core=vertex/PTS, periphery=snapshot/PSS; Dc,Dp<=D share devices):")
    print(f"  {'Dc':>4} {'Dp':>4} {'SpatialCut':>14} {'TemporalCut':>14} {'weighted-cost':>16}")
    best_mixed = None
    Dc = 1
    while Dc <= D:
        Dp = 1
        while Dp <= D and Dp <= S:
            sc, tc, det = mixed_cut(src, dst, snap, verts, snaps, N, S, D, core_mask, lpa_id, Dc, Dp)
            cost = sc * wS + tc * wT
            print(f"  {det['Dc']:>4} {det['Dp']:>4} {sc:>14,} {tc:>14,} {cost:>16,.0f}")
            if best_mixed is None or cost < best_mixed[2]:
                best_mixed = (sc, tc, cost, det['Dc'], det['Dp'])
            Dp *= 2
        Dc *= 2
    m_sc, m_tc, m_cost, bDc, bDp = best_mixed
    print(f"  BEST MIXED: core->{bDc} vertex-blocks (PTS), periphery->{bDp} snapshot-blocks (PSS)")
    print(f"    SpatialCut={m_sc:,}  TemporalCut={m_tc:,}  weighted-cost(w_S={wS},w_T={wT})={m_cost:,.0f}")

    # ---- uniform baselines: pure-PTS, pure-PSS, and the best Dv x Dt factorization (frontier) ----
    print(f"\n  uniform baselines (whole graph, same supra-cut definitions as duality_frontier.py):")
    print(f"  {'Dv':>4} {'Dt':>4} {'SpatialCut':>14} {'TemporalCut':>14} {'weighted-cost':>16}  {'note':<10}")
    rows = []
    for Dv, Dt in factorizations(D):
        vblock = balanced_blocks(lpa_id, N, Dv)
        sc = spatial_cut(vblock, src, dst)
        vsblock = np.minimum(snaps.astype(np.int64) * Dt // S, Dt - 1)
        tc = temporal_cut(verts, vsblock, N, Dt)
        cost = sc * wS + tc * wT
        note = "PTS" if Dt == 1 else ("PSS" if Dv == 1 else "interior")
        rows.append((Dv, Dt, sc, tc, cost, note))
        print(f"  {Dv:>4} {Dt:>4} {sc:>14,} {tc:>14,} {cost:>16,.0f}  {note:<10}")

    c_pts = next(c for (Dv, Dt, sc, tc, c, n) in rows if Dt == 1)
    c_pss = next(c for (Dv, Dt, sc, tc, c, n) in rows if Dv == 1)
    best_uniform = min(rows, key=lambda r: r[4])
    print(f"\n  --- comparison under (w_S={wS}, w_T={wT}) ---")
    print(f"    pure-PTS  cost = {c_pts:,.0f}")
    print(f"    pure-PSS  cost = {c_pss:,.0f}")
    print(f"    best uniform Dv x Dt = Dv{best_uniform[0]}xDt{best_uniform[1]} ({best_uniform[5]}) "
          f"cost = {best_uniform[4]:,.0f}")
    print(f"    MIXED-GRANULARITY    cost = {m_cost:,.0f}")
    best_corner = min(c_pts, c_pss)
    vs_corner = best_corner / m_cost if m_cost else float("inf")
    vs_best = best_uniform[4] / m_cost if m_cost else float("inf")
    verdict = "MIXED WINS" if m_cost < best_uniform[4] else (
              "ties best uniform" if abs(m_cost - best_uniform[4]) < 1e-9 else "uniform wins")
    print(f"    mixed gain vs best CORNER (PTS/PSS) = {vs_corner:.3f}x   "
          f"vs best uniform factorization = {vs_best:.3f}x   -> {verdict}")
    # byte view
    print(f"    weighted-cost is in 'cut-edges' units; at F={F} ({bytes_per}B/cell) the MIXED layout "
          f"moves {(m_sc + m_tc) * bytes_per / 1024**2:,.1f} MiB/epoch of supra-edges")
    return dict(mixed=(m_sc, m_tc, m_cost), pts=c_pts, pss=c_pss, best=best_uniform, rows=rows)


# --------------------------------------------------------------------------------------------------
def load_graph(a):
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int32); dst = g.dst.astype(np.int32)
        return g.name, N, src, dst
    rng = np.random.default_rng(0)
    N, E = a.nodes, a.edges
    S = a.snapshots
    # Synthetic temporal graph planted to give BOTH bets real signal:
    #   * a small DENSE CORE (hubs) that is persistent across the whole window, plus
    #   * a sparse PERIPHERY whose nodes are TEMPORALLY BURSTY (each active in a short time window) and
    #     spatially scattered -> snapshot-cutting them is cheap (low temporal cut) while vertex-cutting
    #     them is expensive (this is exactly when periphery-by-snapshot helps -> BET D10), and
    #   * core community membership that DRIFTS over time so a partition built on an early prefix is
    #     stale by the end (this is what PEEK-AHEAD can fix -> BET D5).
    ncore = max(2, N // 50)
    ncomm = 8                                            # core communities
    edges_per = E
    # edge time (snapshot) ~ uniform over the window, sorted later by the caller
    et = rng.integers(0, S, edges_per)
    is_core_edge = rng.random(edges_per) < 0.7          # 70% of edges are core-core (dense)
    src = np.empty(edges_per, dtype=np.int64); dst = np.empty(edges_per, dtype=np.int64)
    # core node -> community, but community assignment DRIFTS: node u's community at snapshot s is
    # (base_comm[u] + s // (S // ncomm)) so communities rotate over time -> early-prefix partition stale.
    base_comm = rng.integers(0, ncomm, ncore)
    drift = (et * ncomm // max(1, S))                    # 0..ncomm-1 drift offset by time
    ci = np.nonzero(is_core_edge)[0]
    cu = rng.integers(0, ncore, ci.size)
    comm_u = (base_comm[cu] + drift[ci]) % ncomm
    # pick a same-(current)community core partner for cu
    cv = rng.integers(0, ncore, ci.size)
    comm_v_target = comm_u
    # cheap rejection-free trick: map cv into the target community band via modular shift
    cv = (cv % max(1, ncore // ncomm)) * ncomm + comm_v_target
    cv = np.minimum(cv, ncore - 1)
    src[ci] = cu; dst[ci] = cv
    # periphery edges: a periphery node attaches to a core hub, but the periphery node is BURSTY:
    # its id deterministically maps to a time band so it only appears in a short snapshot window.
    pi = np.nonzero(~is_core_edge)[0]
    pu = rng.integers(ncore, N, pi.size)                 # periphery node
    # make periphery node activity bursty: force its edge time near a node-specific band
    band = (pu * S // N)                                 # node -> its burst snapshot band
    et[pi] = np.minimum(band + rng.integers(0, max(1, S // 16), pi.size), S - 1)
    ph = rng.integers(0, ncore, pi.size)                # the core hub it connects to
    src[pi] = pu; dst[pi] = ph
    # reorder edges by time so equal-count snapshots line up with et (caller re-derives snap from order)
    order = np.argsort(et, kind="stable")
    src = src[order].astype(np.int32); dst = dst[order].astype(np.int32)
    return f"synthetic-{N}n-{E}e-core{ncore}-c{ncomm}-bursty", N, src, dst


def main():
    ap = argparse.ArgumentParser(description="D5 peek-ahead + D10 mixed-granularity (process-only cut metrics)")
    ap.add_argument("--dataset", default="", help="zord dataset name (else use --synthetic)")
    ap.add_argument("--synthetic", action="store_true", help="use a planted core/periphery synthetic graph")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--edges", type=int, default=2_000_000)
    ap.add_argument("--snapshots", type=int, default=64)
    ap.add_argument("--devices", type=int, default=64)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--ws", type=float, default=1.0, help="w_S: weight per SpatialCut edge (hardware regime)")
    ap.add_argument("--wt", type=float, default=1.0, help="w_T: weight per TemporalCut edge (hardware regime)")
    ap.add_argument("--core-metric", choices=["degree", "kcore"], default="degree",
                    help="C++ classifier for core/periphery (D10)")
    ap.add_argument("--core-frac", type=float, default=0.05, help="fraction of nodes treated as dense core (D10)")
    ap.add_argument("--kpcts", default="10,25,50,75,90", help="peek-ahead prefix percentages to sweep (D5)")
    a = ap.parse_args()
    if not a.dataset and not a.synthetic:
        a.synthetic = True  # default to synthetic if neither given

    S, D, F, wS, wT = a.snapshots, a.devices, a.feat, a.ws, a.wt
    t0 = time.time()
    name, N, src, dst = load_graph(a)
    E = src.size
    # equal-COUNT snapshots over the time-sorted edge stream (same convention as duality_frontier.py)
    snap = np.minimum((np.arange(E) * S // E).astype(np.int32), S - 1)
    print(f"PEEK-AHEAD + MIXED  dataset={name} N={N:,} E={E:,} S={S} D={D} F={F}  (w_S={wS}, w_T={wT})")
    print(f"  cut binary = {BIN} {'(present)' if os.path.exists(BIN) else '(MISSING -> identity fallback)'}")

    kpcts = [int(x) for x in str(a.kpcts).split(",") if x.strip()]
    bet_peekahead(src, dst, snap, N, S, D, F, wS, wT, kpcts)
    bet_mixed(src, dst, snap, N, S, D, F, wS, wT, a.core_metric, a.core_frac)
    print(f"\n  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
