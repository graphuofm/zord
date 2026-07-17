#!/usr/bin/env python
"""EMPIRICALLY test the space-time isoperimetric LOWER BOUND (THEORY.md section 7).

The claim under test (section 7.1):

    SpatialCut(P) + TemporalCut(P)  >=  L(G, P, Cap)  :=  L_spatial + L_temporal

for EVERY capacity/balance-respecting partition P. We verify it the only way an
inequality with a universal quantifier can be checked empirically: compute the
MEASURED MINIMUM of (SpatialCut + TemporalCut) over a large family of partitions
-- the PSS corner, the PTS corner, every power-of-two Dv x Dt factorization, and
many RANDOM balanced cuts -- and compare that measured-min against a CHEAP,
LOWER-BOUND-SAFE estimate of L. If the bound is real, measured-min >= L; we report
the tightness gap (measured-min / L). The L estimator is deliberately CONSERVATIVE
(it may under-estimate h(G_t) and ρ), so a PASS is meaningful and a gap > 1 is
EXPECTED -- this is a necessary check, not a tight certificate (see THEORY 7.5).

Cut counting MIRRORS scripts/duality_frontier.py / scripts/supra_solve.py exactly:
  - SpatialCut  = # within-snapshot edges whose endpoints sit on different devices.
  - TemporalCut = # same-vertex adjacent-active-snapshot pairs on different devices.

numpy only -- no networkx, no C++ binary, no cluster. Synthetic by default; a real
dataset via --dataset (uses zord.datasets.load, same as the sibling scripts).

  python scripts/lowerbound_check.py --nodes 50000 --edges 800000 --snapshots 32 --devices 16
  python scripts/lowerbound_check.py --dataset wiki-talk --snapshots 64 --devices 64 --feat 128
"""
import argparse
import time

import numpy as np


# --------------------------------------------------------------------------- #
#  Supra-cell graph construction (identical canonicalisation to supra_solve.py)
# --------------------------------------------------------------------------- #
def build_supra_cells(src, dst, snap, S):
    """Active-cell table + explicit spatial/temporal cell-pair lists.

    Cells are unique (vertex, snapshot) pairs carrying an incident edge, ordered
    by (vertex, snapshot) -- vertex-major, snapshot-minor. Returns per-cell
    coordinates and the endpoint cell-ids of every spatial / temporal supra-edge.
    """
    ks = src.astype(np.int64) * S + snap
    kd = dst.astype(np.int64) * S + snap
    keys = np.unique(np.concatenate([ks, kd]))         # sorted unique == cell ids
    C = keys.size
    cell_v = (keys // S).astype(np.int64)
    cell_t = (keys % S).astype(np.int64)
    a = np.searchsorted(keys, ks)
    b = np.searchsorted(keys, kd)
    m = a != b                                          # drop intra-cell self-loops
    sp_a, sp_b = a[m], b[m]
    # temporal pairs: consecutive cells of the SAME vertex (vertex-major order)
    same_v = cell_v[1:] == cell_v[:-1]
    idx = np.nonzero(same_v)[0]
    tp_a, tp_b = idx, idx + 1
    return cell_v, cell_t, C, sp_a, sp_b, tp_a, tp_b


def count_cuts(dev, sp_a, sp_b, tp_a, tp_b):
    """SpatialCut / TemporalCut for a per-cell device assignment (solver defn)."""
    spatial = int(np.count_nonzero(dev[sp_a] != dev[sp_b])) if sp_a.size else 0
    temporal = int(np.count_nonzero(dev[tp_a] != dev[tp_b])) if tp_a.size else 0
    return spatial, temporal


def factorizations(D):
    out, dt = [], 1
    while dt <= D:
        if D % dt == 0:
            out.append((D // dt, dt))
        dt *= 2
    return out


def factor_assignment(cell_v, cell_t, N, S, Dv, Dt):
    """device = vblock(v) * Dt + sblock(t); balanced split of vertices x snapshots.
    Dv=1,Dt=D -> PSS (whole snapshots together); Dv=D,Dt=1 -> PTS (whole timelines).
    """
    vblock = np.minimum(cell_v.astype(np.int64) * Dv // N, Dv - 1)
    sblock = np.minimum(cell_t.astype(np.int64) * Dt // S, Dt - 1)
    return (vblock * Dt + sblock).astype(np.int32)


def random_balanced_assignment(C, D, rng):
    """A uniformly RANDOM balanced D-way cut of the C cells (each device ~C/D cells)."""
    dev = (np.arange(C, dtype=np.int64) % D).astype(np.int32)
    rng.shuffle(dev)
    return dev


# --------------------------------------------------------------------------- #
#  L estimate: L_spatial (Cheeger/spectral per snapshot) + L_temporal (persistence)
# --------------------------------------------------------------------------- #
def lambda2_estimate(rows, cols, n, iters=40, rng=None):
    """Cheap LOWER-bound-safe estimate of lambda2 of the normalized Laplacian of an
    undirected graph given as symmetric coo (rows, cols), n vertices (compacted ids).

    We power-iterate on the normalized adjacency  A_norm = D^{-1/2} A D^{-1/2}
    deflated against the known top eigenvector  d^{1/2}  (eigenvalue 1). The second
    eigenvalue mu2 of A_norm gives lambda2(L_norm) = 1 - mu2. We return max(lambda2, 0).
    No scipy/networkx -- pure numpy SpMV via np.add.at / bincount.
    """
    if n <= 1 or rows.size == 0:
        return 0.0
    deg = np.bincount(rows, minlength=n).astype(np.float64)
    deg = np.maximum(deg, 1e-12)
    dinv_sqrt = 1.0 / np.sqrt(deg)
    phi0 = np.sqrt(deg)
    phi0 /= np.linalg.norm(phi0)                        # top eigvec of A_norm (eig 1)
    if rng is None:
        rng = np.random.default_rng(0)
    x = rng.standard_normal(n)

    def matvec(v):
        # A_norm v = D^-1/2 A D^-1/2 v ; SpMV via bincount (fast buffered scatter)
        y = dinv_sqrt * v
        out = np.bincount(rows, weights=y[cols], minlength=n)   # A @ (D^-1/2 v)
        return dinv_sqrt * out

    x -= (x @ phi0) * phi0
    nx = np.linalg.norm(x)
    if nx < 1e-30:
        return 0.0
    x /= nx
    mu2 = 0.0
    for _ in range(iters):
        x = matvec(x)
        x -= (x @ phi0) * phi0                          # deflate the eig-1 component
        nx = np.linalg.norm(x)
        if nx < 1e-30:
            break
        x /= nx
        mu2 = float(x @ matvec(x))                      # Rayleigh quotient -> mu2
    lam2 = max(0.0, 1.0 - mu2)
    return lam2


def compute_L(cell_v, cell_t, src, dst, snap, N, S, Cap, C, D, rng):
    """Conservative L = L_spatial + L_temporal (THEORY 7.2 + 7.3).

    L_spatial  = sum_t (h(G_t)/2) * vol_t * b_t,  using h(G_t) >= lambda2(G_t)/2.
    L_temporal = sum_v rho(v) * (k_v - 1),  with k_v the # devices v's timeline is
                 FORCED to span.

    FORCING is driven by the per-device CELL BUDGET that is hard for EVERY partition
    in the family: B = min(Cap, ceil(C/D)). Balance alone (largest part <= ~C/D) makes
    B active even with no explicit --cap, so L is non-trivial. Both factors are
    lower-bound-safe: b_t and (k_v-1) are floors any balanced/capacity-bounded P obeys.
      - b_t  = max(0, 1 - B/n_t)  : a device holds <= B cells, so >= n_t - B of a
               snapshot's vertices sit off its largest part -> that volume's boundary
               is forced (>= 0; = 0 iff the whole snapshot fits in one part = PSS).
      - k_v  = ceil(|T_v| / B)    : a timeline of |T_v| cells needs >= that many
               devices when it cannot fit one part (>= 1; > 1 only when forced).
    Returns (L, L_spatial, L_temporal, diagnostics).
    """
    B = min(Cap, int(np.ceil(C / max(D, 1)))) if Cap > 0 else int(np.ceil(C / max(D, 1)))
    B = max(B, 1)
    # ---------- L_spatial: per-snapshot spectral conductance bound ----------
    L_spatial = 0.0
    # snapshot of each edge is `snap`; build per-snapshot subgraphs cheaply.
    order = np.argsort(snap, kind="stable")
    s_sorted = snap[order]
    su = src[order]
    du = dst[order]
    bounds = np.searchsorted(s_sorted, np.arange(S + 1))
    snap_lambda = np.zeros(S)
    snap_vol = np.zeros(S)
    snap_nt = np.zeros(S, dtype=np.int64)
    snap_bt = np.zeros(S)
    for t in range(S):
        lo, hi = int(bounds[t]), int(bounds[t + 1])
        if hi <= lo:
            continue
        a = su[lo:hi]
        b = du[lo:hi]
        m = a != b                                      # drop self-loops
        a, b = a[m], b[m]
        if a.size == 0:
            continue
        # compact vertex ids for this snapshot
        verts, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
        ai = inv[:a.size]
        bi = inv[a.size:]
        nt = verts.size
        # symmetric coo (both directions) for an undirected normalized Laplacian
        rows = np.concatenate([ai, bi])
        cols = np.concatenate([bi, ai])
        vol_t = float(rows.size)                        # sum of degrees = 2*|E_t|
        lam2 = lambda2_estimate(rows, cols, nt, rng=rng)
        h_lb = lam2 / 2.0                               # easy Cheeger: h >= lambda2/2
        # forced split fraction: a single device part holds <= B cells, so a snapshot
        # with n_t > B vertices cannot sit on one part -> >= (n_t-B)/n_t volume forced off.
        b_t = max(0.0, 1.0 - B / nt) if nt > 0 else 0.0
        L_spatial += h_lb * vol_t * b_t / 2.0
        snap_lambda[t] = lam2
        snap_vol[t] = vol_t
        snap_nt[t] = nt
        snap_bt[t] = b_t

    # ---------- L_temporal: persistence x forced device-span ----------
    # |T_v| = number of distinct active snapshots per vertex (from the cell table).
    Tv = np.bincount(cell_v, minlength=N).astype(np.int64)   # cells per vertex == |T_v|
    active = Tv > 0
    rho = np.zeros(N)
    if S > 1:
        rho[active] = (Tv[active] - 1) / (S - 1)             # persistence in [0,1]
    # forced device span k_v: a timeline of |T_v| cells needs >= ceil(|T_v|/B) devices
    # when it cannot fit in one part's budget B (> 1 only when genuinely forced).
    k_v = np.ceil(np.maximum(Tv, 1) / B).astype(np.int64)
    k_v = np.maximum(k_v, 1)
    L_temporal = float(np.sum(rho * (k_v - 1)))

    diag = dict(
        budget_B=B,
        mean_lambda2=float(snap_lambda[snap_nt > 0].mean()) if np.any(snap_nt > 0) else 0.0,
        mean_bt=float(snap_bt[snap_nt > 0].mean()) if np.any(snap_nt > 0) else 0.0,
        mean_rho=float(rho[active].mean()) if np.any(active) else 0.0,
        max_Tv=int(Tv.max(initial=0)),
        forced_kv=int(np.count_nonzero(k_v > 1)),
        nonempty_snaps=int(np.count_nonzero(snap_nt > 0)),
    )
    return L_spatial + L_temporal, L_spatial, L_temporal, diag


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--nodes", type=int, default=50_000)
    ap.add_argument("--edges", type=int, default=800_000)
    ap.add_argument("--snapshots", type=int, default=16)
    ap.add_argument("--devices", type=int, default=32)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--cap", type=int, default=0,
                    help="per-device cell capacity (0 => use balance-implied cap C/D)")
    ap.add_argument("--random-cuts", type=int, default=32,
                    help="# random balanced partitions to include in the measured-min")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    S, D = a.snapshots, a.devices
    rng = np.random.default_rng(a.seed)

    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes
        src = g.src.astype(np.int64)
        dst = g.dst.astype(np.int64)
        name = g.name
    else:
        N = a.nodes
        # planted-PARTITION synthetic so snapshots have REAL conductance structure:
        # ~85% of edges intra-community (high internal density -> lambda2 > 0), the rest
        # cross-community (the bottleneck). dst for intra edges is drawn from the SAME
        # community's member list (vectorized -- no self-loop collapse), so n_t stays large.
        ncomm = max(2, D)
        comm = rng.integers(0, ncomm, N).astype(np.int64)
        order = np.argsort(comm, kind="stable")              # group members by community
        starts = np.searchsorted(comm[order], np.arange(ncomm + 1))
        sizes = np.diff(starts)
        E = a.edges
        src = rng.integers(0, N, E).astype(np.int64)
        intra = rng.random(E) < 0.85
        csrc = comm[src]
        # pick a within-community member for intra edges: start[c] + uniform(size[c])
        off = (rng.random(E) * np.maximum(sizes[csrc], 1)).astype(np.int64)
        dst_intra = order[starts[csrc] + np.minimum(off, np.maximum(sizes[csrc] - 1, 0))]
        dst_cross = rng.integers(0, N, E).astype(np.int64)
        dst = np.where(intra, dst_intra, dst_cross)
        name = f"synthetic-planted-{N}n-{E}e-{ncomm}c"
    E = src.size
    # equal-count snapshots over time-sorted edges (same bucketing as duality_frontier.py)
    snap = np.minimum((np.arange(E) * S // E).astype(np.int32), S - 1)
    print(f"LOWERBOUND dataset={name} N={N} E={E} S={S} D={D} feat={a.feat}")

    cell_v, cell_t, C, sp_a, sp_b, tp_a, tp_b = build_supra_cells(src, dst, snap, S)
    print(f"  supra cells C={C:,} spatial_pairs={sp_a.size:,} temporal_pairs={tp_a.size:,} "
          f"(build {time.time()-t0:.1f}s)")

    # capacity in cells: explicit --cap, else the balance-implied budget ceil((1.1*C)/D)
    Cap = a.cap if a.cap > 0 else int(np.ceil(1.1 * C / D))
    print(f"  per-device cell capacity Cap={Cap:,} (balance-implied)" if a.cap <= 0
          else f"  per-device cell capacity Cap={Cap:,}")

    # ---- measured min over the partition family ----------------------------
    print(f"  {'partition':>22} {'SpatialCut':>14} {'TemporalCut':>14} {'sum':>14}")
    candidates = []  # (label, S_cut, T_cut)

    for Dv, Dt in factorizations(D):
        dev = factor_assignment(cell_v, cell_t, N, S, Dv, Dt)
        sc, tc = count_cuts(dev, sp_a, sp_b, tp_a, tp_b)
        tag = "PSS" if Dv == 1 else ("PTS" if Dt == 1 else "")
        label = f"Dv{Dv}xDt{Dt}{(' '+tag) if tag else ''}"
        candidates.append((label, sc, tc))
        print(f"  {label:>22} {sc:>14,} {tc:>14,} {sc+tc:>14,}")

    rmin = None
    for i in range(max(0, a.random_cuts)):
        dev = random_balanced_assignment(C, D, rng)
        sc, tc = count_cuts(dev, sp_a, sp_b, tp_a, tp_b)
        if rmin is None or sc + tc < rmin[1] + rmin[2]:
            rmin = (f"random[{i}]", sc, tc)
    if rmin is not None:
        candidates.append(rmin)
        print(f"  {('min of %d random' % a.random_cuts):>22} {rmin[1]:>14,} {rmin[2]:>14,} "
              f"{rmin[1]+rmin[2]:>14,}")

    meas_label, ms, mt = min(candidates, key=lambda r: r[1] + r[2])
    measured_min = ms + mt

    # ---- L estimate --------------------------------------------------------
    tL = time.time()
    L, Ls, Lt, diag = compute_L(cell_v, cell_t, src, dst, snap, N, S, Cap, C, D, rng)
    print(f"\n  L estimate ({time.time()-tL:.1f}s, conservative -> lower-bound-safe; "
          f"per-part budget B={diag['budget_B']:,} cells):")
    print(f"    L_spatial  = {Ls:,.1f}  (mean lambda2={diag['mean_lambda2']:.4f}, "
          f"mean forced-split b_t={diag['mean_bt']:.3f}, nonempty_snaps={diag['nonempty_snaps']})")
    print(f"    L_temporal = {Lt:,.1f}  (mean persistence rho={diag['mean_rho']:.4f}, "
          f"max |T_v|={diag['max_Tv']}, forced-span vertices={diag['forced_kv']:,})")
    print(f"    L          = {L:,.1f}")

    # ---- verdict -----------------------------------------------------------
    print("\n  ================= BOUND CHECK =================")
    print(f"  measured-min (SpatialCut+TemporalCut) = {measured_min:,}  @ {meas_label} "
          f"(S={ms:,} T={mt:,})")
    print(f"  L (computed estimate)                 = {L:,.1f}")
    eps = 1e-6 * max(1.0, abs(L))
    ok = measured_min + eps >= L
    gap = (measured_min / L) if L > 0 else float("inf")
    print(f"  bound  measured-min >= L  -> {'PASS' if ok else 'FAIL'}")
    if L <= 0:
        print("  tightness gap = inf  (L=0: capacity/balance forces NEITHER cut here -- a corner can "
              "drive\n                its cut to 0, so the floor is the trivial >=0. The bound becomes\n"
              "                informative when BOTH cuts are forced: snapshots > per-part budget B\n"
              "                (try --snapshots < --devices) and/or timelines > B (try a small --cap).)")
    else:
        print(f"  tightness gap = measured-min / L = {gap:.3f}x  "
              f"({'tight' if gap < 2 else 'loose -- conservative L (expected, THEORY 7.5)'})")
    if not ok:
        print("  !! VIOLATION: a partition beat the lower bound -- check L estimator / assumptions")
    print(f"  total {time.time()-t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
