#!/usr/bin/env python
"""THE space-time cut DUALITY, measured (THEORY.md / D30). Partition the supra-graph by factorizing
D devices = Dv (vertex-blocks) x Dt (snapshot-blocks):
  - Dv=D, Dt=1  -> pure VERTEX partition (PTS): TemporalCut=0, SpatialCut high
  - Dv=1, Dt=D  -> pure SNAPSHOT partition (PSS): SpatialCut=0, TemporalCut high
  - interior     -> trades between them.
We measure SpatialCut(Dv) (spatial/aggregation edges crossing vertex-blocks) and TemporalCut(Dt)
(node-memory edges crossing snapshot-blocks), and show the WEIGHTED total cost has an INTERIOR optimum
that beats BOTH corners -> the duality is real and the integrated cut wins (zord's claim). Vertex blocks
come from the C++ LPA clustering (good spatial partition); CPU-only counting in numpy (vectorized).
  python scripts/duality_frontier.py --dataset wiki-talk --snapshots 64 --devices 64 --feat 128
"""
import argparse, os, struct, subprocess, time
import numpy as np

BIN = os.environ.get("ZORD_GRAPH_BIN", "build/graph_algos")


def cpp_lpa_order(src, dst, N):
    e = np.empty(2 * src.size, dtype=np.int32); e[0::2] = src; e[1::2] = dst
    with open("/tmp/zord_dual_edges.bin", "wb") as f:
        f.write(struct.pack("<qq", N, src.size)); e.tofile(f)
    t = time.time()
    r = subprocess.run([BIN, "/tmp/zord_dual_edges.bin", "lpa", "/tmp/zord_dual_perm.bin"],
                       capture_output=True, text=True)
    cost = time.time() - t
    if r.returncode != 0:
        print("  [cpp lpa] FAILED:", r.stderr.strip()[:160]); return np.arange(N, dtype=np.int32), cost
    with open("/tmp/zord_dual_perm.bin", "rb") as f:
        struct.unpack("<q", f.read(8)); newid = np.fromfile(f, dtype=np.int32, count=N)
    return newid, cost


def factorizations(D):
    out = []
    dt = 1
    while dt <= D:
        if D % dt == 0:
            out.append((D // dt, dt))
        dt *= 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--nodes", type=int, default=4_000_000); ap.add_argument("--edges", type=int, default=50_000_000)
    ap.add_argument("--snapshots", type=int, default=64); ap.add_argument("--devices", type=int, default=64)
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--cap-gb", type=float, default=0.0)   # per-device HBM cap (GB); 0 = no constraint
    a = ap.parse_args()
    S, D, F = a.snapshots, a.devices, a.feat
    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset).sort_by_time()
        N = g.num_nodes; src = g.src.astype(np.int32); dst = g.dst.astype(np.int32); E = src.size
        name = g.name
    else:
        rng = np.random.default_rng(0); N, E = a.nodes, a.edges
        src = rng.integers(0, N, E).astype(np.int32); dst = rng.integers(0, N, E).astype(np.int32)
        name = f"synthetic-{N}n-{E}e"
    snap = np.minimum((np.arange(E) * S // E).astype(np.int32), S - 1)        # equal-count snapshots (time-sorted)
    print(f"DUALITY dataset={name} N={N} E={E} S={S} D={D} F={F}")

    newid, lpa_cost = cpp_lpa_order(src, dst, N)                              # vertex clustering order
    print(f"  lpa vertex clustering: {lpa_cost:.1f}s")

    # per-(vertex,snapshot) activity for temporal cut
    verts = np.concatenate([src, dst]); snaps = np.concatenate([snap, snap])

    bytes_per = F * 4
    print(f"  {'Dv':>4} {'Dt':>4} {'SpatialCut':>14} {'TemporalCut':>14}   weighted-cost (lower=better)")
    rows = []
    for Dv, Dt in factorizations(D):
        vblock = (newid.astype(np.int64) * Dv // N).astype(np.int32)          # balanced vertex-blocks from LPA order
        spatial_cut = int(np.count_nonzero(vblock[src] != vblock[dst]))       # spatial edges crossing vertex-blocks
        sblock = np.minimum(snap.astype(np.int64) * Dt // S, Dt - 1)          # snapshot-blocks (not needed per-edge)
        # temporal cut: per vertex, # distinct snapshot-blocks it is active in, minus 1 (each boundary = 1 transfer)
        vsblock = np.minimum(snaps.astype(np.int64) * Dt // S, Dt - 1)
        key = verts.astype(np.int64) * Dt + vsblock
        uniq = np.unique(key)
        distinct_per_v = np.bincount((uniq // Dt).astype(np.int64), minlength=N)
        temporal_cut = int(np.maximum(distinct_per_v - 1, 0).sum())
        base_cells = (N / Dv) * (S / Dt)                                     # resident cells per device
        foot_gb = (base_cells + spatial_cut / Dv + temporal_cut / Dt) * bytes_per / 1024 ** 3
        rows.append((Dv, Dt, spatial_cut, temporal_cut, foot_gb))
        flag = " INFEASIBLE" if (a.cap_gb > 0 and foot_gb > a.cap_gb) else ""
        print(f"  {Dv:>4} {Dt:>4} {spatial_cut:>14,} {temporal_cut:>14,}  peak={foot_gb:6.2f}GB{flag}")

    # weighted-cost frontier under a few hardware regimes (bytes / GB/s)
    regimes = {"all-NVLink (325/325)": (325.0, 325.0),
               "spatial-NVLink temporal-PCIe (325/25)": (325.0, 25.0),
               "spatial-PCIe temporal-NVLink (25/325)": (25.0, 325.0)}
    def feasible(r):
        return a.cap_gb <= 0 or r[4] <= a.cap_gb
    for rn, (bs, bt) in regimes.items():
        best = None
        for (Dv, Dt, sc, tc, fg) in rows:
            if not feasible((Dv, Dt, sc, tc, fg)):
                continue
            cost = (sc * bytes_per) / (bs * 1e9) + (tc * bytes_per) / (bt * 1e9)
            if best is None or cost < best[0]: best = (cost, Dv, Dt)
        if best is None:
            print(f"  [{rn}] NO FEASIBLE factorization under cap={a.cap_gb}GB (needs CPU tiering)"); continue
        interior = "INTERIOR" if (best[1] > 1 and best[2] > 1) else "corner"
        c_pts = c_pss = float("inf")
        for r in rows:                                                        # corner costs only if feasible
            cost = (r[2] * bytes_per) / (bs * 1e9) + (r[3] * bytes_per) / (bt * 1e9)
            if r[1] == 1 and feasible(r): c_pts = cost                        # Dt=1 (PTS)
            if r[0] == 1 and feasible(r): c_pss = cost                        # Dv=1 (PSS)
        gain = min(c_pts, c_pss) / best[0] if min(c_pts, c_pss) < float("inf") else float("inf")
        print(f"  [{rn}] BEST=Dv{best[1]}xDt{best[2]} ({interior}) cost={best[0]*1e3:.2f}ms | "
              f"feasiblePTS={c_pts*1e3:.2f}ms feasiblePSS={c_pss*1e3:.2f}ms | gain_vs_best_feasible_corner={gain:.2f}x")
    print(f"  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
