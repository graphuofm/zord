#!/usr/bin/env python
"""Compare partitioners on a (staged) dataset under a realistic feature dim +
temporal window, on the measured 3-tier HetCluster profile. Reports cross-edges,
imbalance, AND the thing that actually matters -- pre-flight feasibility
(G1 no-OOM) + makespan (G2). The point: capacity-blind partitioners (hash,
fennel) minimize cuts but OOM the small card under memory pressure; zord stays
feasible by sizing to MEASURED memory.

  python scripts/compare_partitioners.py wiki-talk --feat-dim 1024 --window 16
"""
import argparse, json, time

from zord.datasets import load
from zord.profiler import hetcluster
from zord.partition import PARTITIONERS, CostParams
from zord.partition.cost_model import max_nodes_per_device
from zord.guarantees import preflight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--feat-dim", type=int, default=1024)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--h100", type=int, default=1)
    ap.add_argument("--a6000", type=int, default=1)
    ap.add_argument("--a5000", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    g = load(a.dataset).sort_by_time()
    c = hetcluster(a.h100, a.a6000, a.a5000)
    cp = CostParams(feat_dim=a.feat_dim, window=a.window)
    cap = max_nodes_per_device(c, cp, avg_degree=g.num_edges / max(g.num_nodes, 1))
    print(f"dataset={g.name} nodes={g.num_nodes:,} edges={g.num_edges:,} "
          f"devices={c.num_devices} feat_dim={a.feat_dim} window={a.window}")
    print(f"{'partitioner':9} {'cross_edges':>12} {'imbal':>6} {'feasible':>8} "
          f"{'OOM_devs':>9} {'makespan_s':>11} {'build_s':>8}")
    rows = {}
    for name, P in PARTITIONERS.items():
        if name == "random":
            continue
        kw = {"capacity": cap} if name == "zord" else {}
        t = time.time(); part = P().partition(g.src, g.dst, g.num_nodes, c, **kw); bt = time.time() - t
        pf = preflight(part, c, cp)
        rows[name] = dict(cross_edges=part.total_cross_edges, imbalance=round(part.imbalance(), 2),
                          feasible=pf.feasible, oom=pf.oom_devices,
                          makespan_s=round(pf.makespan_sec, 4), build_s=round(bt, 2))
        print(f"{name:9} {part.total_cross_edges:>12,} {part.imbalance():>6.2f} "
              f"{str(pf.feasible):>8} {str(pf.oom_devices):>9} {pf.makespan_sec:>11.4f} {bt:>8.2f}")
    # the headline: who stays feasible?
    feas = [n for n, r in rows.items() if r["feasible"]]
    print(f"\nFEASIBLE (no-OOM): {feas}")
    print(f"INFEASIBLE (would OOM): {[n for n in rows if n not in feas]}")
    if a.out:
        json.dump({"dataset": g.name, "nodes": g.num_nodes, "edges": g.num_edges,
                   "feat_dim": a.feat_dim, "window": a.window, "rows": rows},
                  open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
