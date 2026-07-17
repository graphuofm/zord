#!/usr/bin/env python
"""zord CORE: drive the multilevel WEIGHTED supra-graph partitioner (src/zord/cpp/supra_solver.cpp).

Loads a temporal graph, buckets edges into S snapshots, builds the solver's binary input, runs the
compiled solver (build/supra_solver), reads the per-cell device assignment back, and reports
SpatialCut / TemporalCut / weighted-cost for zord -- COMPARED against the PSS, PTS, and Dv x Dt
factorization baselines (the cut-counting mirrors scripts/duality_frontier.py, but evaluated on the
EXPLICIT supra-cell graph so every method is scored by the SAME definition the solver optimises):
  - SpatialCut  = # spatial cell-pairs (within a snapshot, an edge endpoints) on different devices.
  - TemporalCut = # temporal cell-pairs (same vertex, adjacent ACTIVE snapshots) on different devices.

The headline it must produce:  "zord weighted cut <= min(PSS, PTS)".

  python scripts/supra_solve.py --dataset wiki-talk --snapshots 64 --devices 64 --feat 128 \
         --ws 325 --wt 25 --cap 0

w_S/w_T default to MEASURED hardware-style weights = bytes_per_cut / B_link (THEORY.md sec.2). With
--feat F the bytes are F*4; --ws/--wt are link bandwidths in GB/s (so the weight is F*4/(B*1e9)).
Pass --ws/--wt as already-scaled weights with --raw-weights to override that.
"""
import argparse
import os
import struct
import subprocess
import time

import numpy as np

SOLVER = os.environ.get("ZORD_SUPRA_BIN", "build/supra_solver")
TMP_IN = os.environ.get("ZORD_SUPRA_IN", "/tmp/zord_supra_in.bin")
TMP_OUT = os.environ.get("ZORD_SUPRA_OUT", "/tmp/zord_supra_out.bin")


def build_supra_cells(src, dst, snap, N, S):
    """Return the canonical active-cell table and the explicit supra-graph edge lists.

    Cells are unique (vertex, snapshot) pairs that carry an incident edge, ordered by
    (vertex, snapshot) -- IDENTICAL to the C++ solver's canonical cell order, so the
    device[] array it returns lines up index-for-index with `cell_v`/`cell_t` here.

    Returns:
      cell_v, cell_t : int64 [C]   per-cell coordinates (sorted vertex-major, snapshot-minor)
      sp_a, sp_b     : int64 [.]   spatial cell-pairs (cell-id endpoints of within-snapshot edges)
      tp_a, tp_b     : int64 [.]   temporal cell-pairs (same vertex, adjacent active snapshots)
    """
    # key = vertex * S + snapshot for each edge endpoint
    ks = src.astype(np.int64) * S + snap
    kd = dst.astype(np.int64) * S + snap
    all_keys = np.concatenate([ks, kd])
    keys = np.unique(all_keys)                       # sorted unique == cell ids 0..C-1
    C = keys.size
    cell_v = (keys // S).astype(np.int64)
    cell_t = (keys % S).astype(np.int64)
    # map endpoint keys -> cell ids
    a = np.searchsorted(keys, ks)
    b = np.searchsorted(keys, kd)
    # spatial pairs: drop self-loops (a==b)
    m = a != b
    sp_a, sp_b = a[m], b[m]
    # temporal pairs: consecutive cells of the SAME vertex (cells are vertex-major, time-minor)
    same_v = cell_v[1:] == cell_v[:-1]
    idx = np.nonzero(same_v)[0]
    tp_a = idx
    tp_b = idx + 1
    return cell_v, cell_t, keys, C, sp_a, sp_b, tp_a, tp_b


def count_cuts(dev, sp_a, sp_b, tp_a, tp_b):
    """SpatialCut / TemporalCut for a per-cell device assignment (same defn as the solver)."""
    spatial = int(np.count_nonzero(dev[sp_a] != dev[sp_b])) if sp_a.size else 0
    temporal = int(np.count_nonzero(dev[tp_a] != dev[tp_b])) if tp_a.size else 0
    return spatial, temporal


def factor_assignment(cell_v, cell_t, N, S, Dv, Dt, vorder=None):
    """Assign each cell to a device by the (vertex-block x snapshot-block) factorization.

    Mirrors duality_frontier.py: vertex-block = balanced split of the (optionally clustered)
    vertex order into Dv blocks; snapshot-block = balanced split of snapshots into Dt blocks;
    device = vblock * Dt + sblock  in [0, Dv*Dt).
      - Dv=1, Dt=D : PSS (whole snapshots together; temporal cut high, spatial cut 0 across blocks)
      - Dv=D, Dt=1 : PTS (whole vertex timelines together; spatial cut high, temporal cut 0)
    """
    if vorder is None:
        vrank = cell_v                                 # identity vertex order
    else:
        vrank = vorder[cell_v]                         # remapped rank per vertex
    vblock = np.minimum(vrank.astype(np.int64) * Dv // N, Dv - 1)
    sblock = np.minimum(cell_t.astype(np.int64) * Dt // S, Dt - 1)
    return (vblock * Dt + sblock).astype(np.int32)


def factorizations(D):
    out, dt = [], 1
    while dt <= D:
        if D % dt == 0:
            out.append((D // dt, dt))
        dt *= 2
    return out


def write_input(path, N, S, src, dst, snap, D, w_S, w_T, cap_cells):
    M = src.size
    trip = np.empty(3 * M, dtype=np.int32)
    trip[0::3] = src.astype(np.int32)
    trip[1::3] = dst.astype(np.int32)
    trip[2::3] = snap.astype(np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("<qqq", int(N), int(S), int(M)))
        trip.tofile(f)
        f.write(struct.pack("<iff", int(D), float(w_S), float(w_T)))
        f.write(struct.pack("<q", int(cap_cells)))


def read_output(path, C):
    with open(path, "rb") as f:
        (num,) = struct.unpack("<q", f.read(8))
        dev = np.fromfile(f, dtype=np.int32, count=num)
    if num != C:
        raise RuntimeError(f"solver returned {num} cells, expected {C}")
    return dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--edges", type=int, default=2_000_000)
    ap.add_argument("--snapshots", type=int, default=64)
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--ws", type=float, default=325.0, help="spatial link GB/s (or raw weight w/ --raw-weights)")
    ap.add_argument("--wt", type=float, default=25.0, help="temporal link GB/s (or raw weight w/ --raw-weights)")
    ap.add_argument("--raw-weights", action="store_true", help="treat --ws/--wt as already-scaled weights")
    ap.add_argument("--cap", type=int, default=0, help="per-device cap in #cells (0=unbounded)")
    a = ap.parse_args()
    S, D, F = a.snapshots, a.devices, a.feat

    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes
        src = g.src.astype(np.int32)
        dst = g.dst.astype(np.int32)
        name = g.name
    else:
        rng = np.random.default_rng(0)
        N = a.nodes
        src = rng.integers(0, N, a.edges).astype(np.int32)
        dst = rng.integers(0, N, a.edges).astype(np.int32)
        name = f"synthetic-{N}n-{a.edges}e"
    E = src.size
    # equal-count snapshots over time-sorted edges (same bucketing as duality_frontier.py)
    snap = np.minimum((np.arange(E) * S // E).astype(np.int32), S - 1)

    # hardware-style weights: bytes_per_cut / B_link  (THEORY.md sec.2). bytes = F*4.
    if a.raw_weights:
        w_S, w_T = a.ws, a.wt
    else:
        bytes_per = F * 4.0
        w_S = bytes_per / (a.ws * 1e9) if a.ws > 0 else 0.0
        w_T = bytes_per / (a.wt * 1e9) if a.wt > 0 else 0.0

    print(f"SUPRA dataset={name} N={N} E={E} S={S} D={D} F={F} w_S={w_S:.6g} w_T={w_T:.6g} cap={a.cap}")

    # ---- build the explicit supra-cell graph (shared by zord + baselines) ----------------
    cell_v, cell_t, keys, C, sp_a, sp_b, tp_a, tp_b = build_supra_cells(src, dst, snap, N, S)
    print(f"  supra cells C={C:,} spatial_pairs={sp_a.size:,} temporal_pairs={tp_a.size:,} "
          f"(build {time.time()-t0:.1f}s)")

    # ---- run zord (the C++ solver) -------------------------------------------------------
    write_input(TMP_IN, N, S, src, dst, snap, D, w_S, w_T, a.cap)
    if not os.path.exists(SOLVER):
        raise SystemExit(f"solver binary not found at {SOLVER}; build it:\n"
                         f"  g++ -O3 -std=c++17 -fopenmp src/zord/cpp/supra_solver.cpp -o {SOLVER}")
    tr = time.time()
    r = subprocess.run([SOLVER, TMP_IN, TMP_OUT], capture_output=True, text=True)
    solve_s = time.time() - tr
    if r.returncode != 0:
        raise SystemExit(f"[solver FAILED rc={r.returncode}]\n{r.stderr}")
    print("  --- solver stderr ---")
    for line in r.stderr.strip().splitlines():
        print("   ", line)
    print("  ---------------------")
    zdev = read_output(TMP_OUT, C)
    z_sc, z_tc = count_cuts(zdev, sp_a, sp_b, tp_a, tp_b)
    z_cost = w_S * z_sc + w_T * z_tc

    # ---- baselines, scored on the SAME cell graph ----------------------------------------
    # PSS = Dv1 x DtD (whole snapshots), PTS = DvD x Dt1 (whole timelines), plus all factorizations.
    print(f"  {'Dv':>4} {'Dt':>4} {'SpatialCut':>14} {'TemporalCut':>14} {'weighted-cost':>16}  who")
    pss_cost = pts_cost = float("inf")
    best_factor = None
    for Dv, Dt in factorizations(D):
        fdev = factor_assignment(cell_v, cell_t, N, S, Dv, Dt)
        sc, tc = count_cuts(fdev, sp_a, sp_b, tp_a, tp_b)
        cost = w_S * sc + w_T * tc
        tag = ""
        if Dv == 1:
            tag = "PSS"; pss_cost = cost
        elif Dt == 1:
            tag = "PTS"; pts_cost = cost
        if best_factor is None or cost < best_factor[0]:
            best_factor = (cost, Dv, Dt, sc, tc)
        print(f"  {Dv:>4} {Dt:>4} {sc:>14,} {tc:>14,} {cost:>16.6g}  {tag}")

    # ---- report ---------------------------------------------------------------------------
    print("\n  ================= zord vs baselines =================")
    print(f"  zord       SpatialCut={z_sc:,} TemporalCut={z_tc:,} weighted-cost={z_cost:.6g} "
          f"(solve {solve_s:.2f}s)")
    print(f"  PSS  (Dv1xDt{D})  weighted-cost={pss_cost:.6g}")
    print(f"  PTS  (Dv{D}xDt1)  weighted-cost={pts_cost:.6g}")
    bf = best_factor
    print(f"  best Dv x Dt factorization = Dv{bf[1]}xDt{bf[2]} weighted-cost={bf[0]:.6g}")
    corner = min(pss_cost, pts_cost)
    # headline
    eps = 1e-9 * max(1.0, abs(corner))
    ok = z_cost <= corner + eps
    print(f"\n  HEADLINE: zord weighted cut ({z_cost:.6g}) "
          f"{'<=' if ok else '>'} min(PSS,PTS) ({corner:.6g})  -> {'PASS' if ok else 'FAIL'}")
    if corner < float("inf") and corner > 0:
        print(f"            gain vs best corner = {corner / max(z_cost, 1e-30):.3f}x")
    print(f"  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
