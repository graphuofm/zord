#!/usr/bin/env python3
"""Runnable end-to-end ZORD example (pure CPU/numpy -- no GPU, no dataset mount).

Builds a tiny synthetic community temporal graph, then calls the real engine
(`zord.schedule.plan`) to produce a full PLAN:
  - the adaptive-corner ARRANGE partition assignment (worst-case-optimal: <= METIS),
  - per-device PLACEMENT + vertex-cut / replication decisions,
  - the INCREMENTAL-MIGRATION plan vs a prior batch (state-migration cost), and
  - the predicted MAKESPAN + feasibility.
The interconnect bandwidth is a PARAMETER (link_gbps) -- zord wins on the algorithm
at ANY comm speed, so we plan the SAME graph at a fast and a slow link to show the
adaptive corner can switch.

Run:
    python3 examples/plan_synthetic.py
    # or, from a checkout without install:
    PYTHONPATH=src python3 examples/plan_synthetic.py
"""
import os
import sys

# make `zord` importable straight from a checkout (examples/ is not a package)
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from zord.datasets import TemporalGraph
from zord.profiler import from_spec, hetcluster
from zord.schedule import plan


def make_synthetic_temporal(N=4000, M=40000, C=16, intra=0.9, seed=0):
    """Community-structured temporal graph: `intra` of edges stay inside a community,
    the rest are cross-community noise; timestamps spread across the window."""
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, C, size=N)
    order = np.argsort(comm, kind="stable")
    bounds = np.searchsorted(comm[order], np.arange(C + 1))
    m_in = int(M * intra)
    u = rng.integers(0, N, size=m_in)
    cu = comm[u]
    lo = bounds[cu]; hi = bounds[cu + 1]
    pick = lo + (rng.random(m_in) * np.maximum(1, hi - lo)).astype(np.int64)
    v = order[np.minimum(pick, N - 1)]
    u2 = rng.integers(0, N, size=M - m_in)
    v2 = rng.integers(0, N, size=M - m_in)
    src = np.concatenate([u, u2]).astype(np.int64)
    dst = np.concatenate([v, v2]).astype(np.int64)
    t = np.sort(rng.integers(0, M, size=src.size)).astype(np.int64)
    return TemporalGraph(src=src, dst=dst, t=t, num_nodes=N, name="synth-temporal")


def make_multimodal_temporal(N=1_000_000, M=4_000_000, C=200, intra=0.92,
                             rich_frac=0.6, heavy_dim=65536, poor_dim=32, seed=0):
    """Community temporal graph with HETEROGENEOUS per-node feature SIZE F_v -- a multi-modal
    graph where a `rich_frac` of nodes carry big multi-modal features (heavy_dim: text/image
    embeddings) and the rest small categorical features (poor_dim). The rich nodes fill whole
    communities, so a locality-respecting cut keeps a rich community LOCAL on one device, piling
    heavy feature memory there. Returns (TemporalGraph, F_v[N] feature-dims-per-node)."""
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, C, size=N)
    order = np.argsort(comm, kind="stable")
    bounds = np.searchsorted(comm[order], np.arange(C + 1))
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
    t = np.sort(rng.integers(0, M, size=src.size)).astype(np.int64)
    # per-node feature DIMS: fill whole communities with the heavy type until ~rich_frac reached
    Fv = np.full(N, float(poor_dim), dtype=np.float64)
    n_rich = int(round(rich_frac * N))
    csize = np.bincount(comm, minlength=C)
    rich_comms, acc = [], 0
    for c in rng.permutation(C):
        rich_comms.append(c); acc += int(csize[c])
        if acc >= n_rich:
            break
    rich_mask = np.isin(comm, np.array(rich_comms))
    Fv[rich_mask] = float(heavy_dim)
    g = TemporalGraph(src=src, dst=dst, t=t, num_nodes=N, name="multimodal-temporal")
    return g, Fv


def demo_attribute_aware(cluster):
    """The §33 ATTRIBUTE win, realized natively by `zord plan` via the per-node feat_bytes
    vector F_v. On a multi-modal temporal graph (heterogeneous per-node feature SIZE), the
    ATTRIBUTE-BLIND plan (scalar mean-F: it sizes every device by count*meanF and cannot SEE
    that heavy multi-modal rows pile onto the small-HBM devices) OOMs them; the ATTRIBUTE-AWARE
    plan (feat_bytes=F_v: feasibility + makespan use the ACTUAL per-node feature bytes) routes
    the heavy mass to the high-HBM / high-bandwidth H100 -> stays FEASIBLE and lower makespan.
    PROCESS-only: same data + same model => same result; only WHERE feature rows live changes."""
    GB = 1024 ** 3
    g, Fv = make_multimodal_temporal()
    mean_F = int(round(float(Fv.mean())))
    rich = int((Fv == Fv.max()).sum())
    print("\n" + "=" * 78)
    print("=== §33 ATTRIBUTE-AWARE placement: multi-modal heterogeneous F_v via `zord plan` ===")
    print("=" * 78)
    print(f"graph: N={g.num_nodes:,} E={g.num_edges:,}  per-node feature dims F_v in "
          f"{{{int(Fv.min())},{int(Fv.max())}}}  rich(multi-modal)={rich:,} "
          f"({rich/g.num_nodes*100:.0f}%)  mean F={mean_F}")
    print(f"  total feature bytes = {Fv.sum()*4/GB:.0f}GB across "
          f"{sum(d.usable_mem for d in cluster.devices)/GB:.0f}GB aggregate HBM\n")

    print("--- ATTRIBUTE-BLIND: scalar feat_dim = mean F (the old engine: count*meanF sizing) ---")
    p_blind = plan(g, cluster, link_gbps=325.0, feat_dim=mean_F)
    print(p_blind.summary())

    print("\n--- ATTRIBUTE-AWARE: feat_bytes = F_v (the §33 win: actual per-node feature bytes) ---")
    p_aware = plan(g, cluster, link_gbps=325.0, feat_dim=mean_F, feat_bytes=Fv)
    print(p_aware.summary())

    speedup = p_blind.makespan_ms / max(1e-9, p_aware.makespan_ms)
    print(f"\n  => BLIND feasible={p_blind.feasible} (OOM devices: "
          f"{[d.name for d in p_blind.placement if not d.feasible] or 'none'})  |  "
          f"AWARE feasible={p_aware.feasible}  |  makespan blind/aware = {speedup:.2f}x")
    if not p_blind.feasible and p_aware.feasible:
        print("     ATTRIBUTE WIN: F_v-aware zord routes heavy multi-modal mass to the high-HBM "
              "H100 -> stays feasible where the scalar-F-blind plan OOMs the small devices.")


def main():
    g = make_synthetic_temporal()
    print(f"graph: {g.summary()}\n")

    # A heterogeneous cluster passed via the GENERAL spec builder: 3 devices with
    # distinct HBM capacity + achieved aggregation bandwidth, and the interconnect
    # bandwidth as an explicit PARAMETER.
    cluster = from_spec(
        hbm_gb=[80.0, 48.0, 32.0],          # per-device usable HBM
        agg_bw_gbps=[942.0, 534.0, 444.0],  # per-device achieved aggregation bandwidth
        interconnect_gbps=50.0,             # the comm-speed PARAMETER
    )

    print("=== batch 0 (cold start) at link=50 GB/s ===")
    p0 = plan(g, cluster, link_gbps=50.0, feat_dim=128)
    print(p0.summary())

    print("\n=== same graph, SLOW link=0.5 GB/s (adaptive corner may switch) ===")
    p_slow = plan(g, cluster, link_gbps=0.5, feat_dim=128)
    print(p_slow.summary())

    # batch 1: the graph grew (a new community of vertices + edges arrives). zord reuses
    # batch 0's assignment and only re-places the changed cone under a migration budget,
    # then costs the resulting node-memory state migration over the link parameter.
    g1 = make_synthetic_temporal(N=4400, M=46000, C=16, intra=0.9, seed=1)
    g1.name = "synth-temporal-batch1"
    new_edge_lo = g.num_edges      # edges beyond batch 0's count are "new" this batch
    print("\n=== batch 1 (incremental vs batch 0, migration budget=5%) ===")
    p1 = plan(g1, cluster, link_gbps=50.0, feat_dim=128,
              prior=p0, new_edge_lo=new_edge_lo, migration_budget=0.05, mem_dim=100)
    print(p1.summary())

    # The §33 attribute win, native in the engine: pass a per-node feat_bytes vector F_v so
    # `zord plan` does attribute-aware (feature-byte) placement + feasibility. The cluster here
    # uses a FAST interconnect (325 GB/s) so the makespan is compute/placement-bound (the regime
    # the §33 feature-byte placement governs). Heterogeneous-F demo runs at N=1M (~15s).
    cluster_fast = from_spec(
        hbm_gb=[80.0, 48.0, 32.0], agg_bw_gbps=[942.0, 534.0, 444.0],
        interconnect_gbps=325.0, names=["H100-80", "RTX6000-48", "RTX5000-32"])
    demo_attribute_aware(cluster_fast)


if __name__ == "__main__":
    main()
