#!/usr/bin/env python
"""Incremental re-partition experiment (the Amap thesis: don't recompute nightly).

Simulate K consecutive "nights": night t = the graph grown to cumulative snapshot t.
Each night, re-partition with:
  - zord INCREMENTAL  (prior = last night's partition; pay only the delta)
  - METIS FROM SCRATCH (recompute the whole partition, like the rejected heuristic)
Measure per night: re-partition wall-time, cut quality, feasibility, and NODE
MIGRATION vs the previous night (how many nodes changed device -> nightly data
movement between servers). METIS labels are permutation-matched to the prev night
before counting migration (so we don't over-count due to arbitrary label flips).

  python scripts/incremental_exp.py askubuntu --nights 8 --feat-dim 1024 --window 8
"""
import argparse, itertools, json, time
import numpy as np

from zord.datasets import load
from zord.profiler import hetcluster
from zord.partition import ZordPartitioner, MetisPartitioner
from zord.partition.cost_model import CostParams, max_nodes_per_device
from zord.guarantees import preflight


def migration(prev, cur, P):
    if prev is None:
        return 0
    k = min(len(prev), len(cur))
    a, b = prev[:k], cur[:k]
    best = k
    for perm in itertools.permutations(range(P)):       # match arbitrary labels
        pb = np.asarray(perm)[b]
        best = min(best, int((a != pb).sum()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--nights", type=int, default=8)
    ap.add_argument("--feat-dim", type=int, default=1024)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.0,
                    help="migration budget: fraction of nodes zord may re-balance/night")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = load(a.dataset).sort_by_time()
    c = hetcluster(1, 1, 1); P = c.num_devices
    cp = CostParams(feat_dim=a.feat_dim, window=a.window)
    snaps = g.to_snapshots(num_snapshots=a.nights)
    offs = [s.hi for s in snaps]

    prevz = prevm = None
    rows, zt, mt, zmig, mmig = [], 0.0, 0.0, 0, 0
    for t, hi in enumerate(offs):
        es, ed = g.src[:hi], g.dst[:hi]
        nn = int(max(es.max(), ed.max()) + 1)
        cap = max_nodes_per_device(c, cp, avg_degree=hi / max(nn, 1))   # per-night density
        t0 = time.time(); pz = ZordPartitioner(migration_budget=a.budget).partition(es, ed, nn, c, prior=prevz, capacity=cap); tz = time.time() - t0
        t0 = time.time(); pm = MetisPartitioner().partition(es, ed, nn, c); tm = time.time() - t0
        mz = migration(prevz.assignment if prevz else None, pz.assignment, P)
        mm = migration(prevm.assignment if prevm else None, pm.assignment, P)
        zt += tz; mt += tm; zmig += mz; mmig += mm
        rows.append(dict(night=t, nodes=nn, edges=int(hi),
                         zord_time=round(tz, 3), zord_cuts=pz.total_cross_edges,
                         zord_feasible=preflight(pz, c, cp).feasible, zord_migrated=mz,
                         metis_time=round(tm, 3), metis_cuts=pm.total_cross_edges,
                         metis_feasible=preflight(pm, c, cp).feasible, metis_migrated=mm))
        print(f"night {t}: nodes={nn:>8,} | zord t={tz:6.3f}s cuts={pz.total_cross_edges:>9,} "
              f"mig={mz:>7,} | metis t={tm:6.3f}s cuts={pm.total_cross_edges:>9,} mig={mm:>7,}")
        prevz, prevm = pz, pm
    print(f"\nTOTAL re-partition time: zord {zt:.2f}s  vs  metis {mt:.2f}s  ({mt/max(zt,1e-9):.1f}x)")
    print(f"TOTAL node migration:    zord {zmig:,}  vs  metis {mmig:,}  ({mmig/max(zmig,1):.0f}x)")
    if a.out:
        json.dump({"dataset": g.name, "feat_dim": a.feat_dim, "window": a.window,
                   "zord_total_time": zt, "metis_total_time": mt,
                   "zord_total_migration": zmig, "metis_total_migration": mmig,
                   "rows": rows}, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
