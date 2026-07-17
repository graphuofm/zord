#!/usr/bin/env python
"""Demo the zord global memory scheduler across memory regimes on the HetCluster cluster.
Shows it (a) degenerates to all-resident single placement when the working set fits, and
(b) switches to CPU<->HBM tiering (guaranteeing fit) under memory pressure -- predicting
epoch time and the bottleneck (compute vs PCIe-staging). Pure CPU; no GPU needed.
  PYTHONPATH=src python scripts/plan_demo.py
"""
from zord.profiler.cluster_profile import hetcluster
from zord.schedule import Workload, plan_memory

cluster = hetcluster(num_h100=1, num_6000ada=1, num_5000ada=1)

print("=" * 96)
print("REGIME A -- fits in HBM (small window): expect all-resident (~single placement, ~METIS).")
wa = Workload(num_nodes=2_000_000, num_edges=16_000_000, feat_dim=128, layers=2, window=2)
print(plan_memory(cluster, wa).summary())

print("=" * 96)
print("REGIME B -- large window exceeds HBM: expect tiering (some snapshots staged from CPU).")
wb = Workload(num_nodes=4_000_000, num_edges=40_000_000, feat_dim=256, layers=2, window=16)
print(plan_memory(cluster, wb).summary())

print("=" * 96)
print("REGIME C -- same as B with MEASURED 2-layer temporal reuse (askubuntu reuse@2hop=0.18, §11).")
wc = Workload(num_nodes=4_000_000, num_edges=40_000_000, feat_dim=256, layers=2, window=16,
              reuse_frac=0.18)
print(plan_memory(cluster, wc).summary().splitlines()[0])

print("=" * 96)
print("REGIME D -- prefetch OFF vs ON on regime B (shows PCIe staging hidden behind compute).")
print("  prefetch OFF:", plan_memory(cluster, wb, prefetch=False).summary().splitlines()[0])
print("  prefetch ON :", plan_memory(cluster, wb, prefetch=True).summary().splitlines()[0])
