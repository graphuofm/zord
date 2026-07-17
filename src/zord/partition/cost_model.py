"""Data-calibrated cost model (R1). Turns a Partition + ClusterProfile into:
  - per-device memory footprint  -> feeds G1 (no-OOM feasibility)
  - per-device compute time + comm time -> makespan -> feeds G2 (no straggler)

Footprint and time use simple, explicit models now; the profiler will replace
the constants (bytes/edge, sec/edge, GB/s) with measured values so the numbers
are real, not hand-waved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Partition
from ..profiler.cluster_profile import ClusterProfile, DeviceProfile


@dataclass
class CostParams:
    feat_dim: int = 128            # node feature dimension
    bytes_per_feat: int = 4        # float32 (1=int8, 2=fp16, 4=fp32) -- M5 precision lever
    bytes_per_edge: int = 20       # src(4)+dst(4)+ts(8)+w(4)
    window: int = 1                # snapshots co-resident in a batch
    halo_replication: float = 1.0  # avg extra copies of boundary nodes
    sec_per_edge: float = 5e-9     # compute time per local edge (per unit r_k)

    def at_precision(self, bits: int) -> "CostParams":
        """Return a copy with embedding precision set to `bits` (M5 compression).
        Measured (job 78701, askubuntu link-pred): int8 embeddings give 4x less
        feature memory AND 4x less boundary-comm bytes at ZERO accuracy loss
        (AUC 0.8433 fp32 -> 0.8436 int8). So lowering bits simultaneously raises
        max_nodes_per_device (more fits -> M1) and lowers device_comm_sec (M2):
        the SAME lever relaxes two coupled bottlenecks (memory + comm-time)."""
        from dataclasses import replace
        return replace(self, bytes_per_feat=max(1, bits // 8))


def device_footprint_bytes(p: Partition, dev_idx: int, cp: CostParams) -> int:
    nodes = int(p.nodes_per_device[dev_idx])
    edges = int(p.edges_per_device[dev_idx])
    halo = int(p.halo_per_device[dev_idx]) if p.halo_per_device is not None else 0
    feat = (nodes + int(halo * cp.halo_replication)) * cp.feat_dim * cp.bytes_per_feat
    edgemem = edges * cp.bytes_per_edge
    return cp.window * (feat + edgemem)


def device_compute_sec(p: Partition, dev_idx: int, cluster: ClusterProfile,
                       cp: CostParams) -> float:
    edges = int(p.edges_per_device[dev_idx])
    r = cluster.devices[dev_idx].throughput
    return cp.window * edges * cp.sec_per_edge / max(r, 1e-9)


def device_comm_sec(p: Partition, dev_idx: int, cluster: ClusterProfile,
                    cp: CostParams) -> float:
    """Time to ship cross-partition boundary features into dev_idx, charged at
    the (heterogeneous) link bandwidth -- cutting across slow links costs more."""
    incoming = p.cross_edges[:, dev_idx].copy()
    incoming[dev_idx] = 0
    bw = cluster.bandwidth
    total = 0.0
    bytes_per_boundary = cp.feat_dim * cp.bytes_per_feat
    for j in range(cluster.num_devices):
        if j == dev_idx or incoming[j] == 0:
            continue
        gbps = bw[j][dev_idx]
        if gbps == float("inf"):
            continue
        total += (incoming[j] * bytes_per_boundary) / (gbps * 1024 ** 3)
    return cp.window * total


def max_nodes_per_device(cluster: ClusterProfile, cp: "CostParams",
                         avg_degree: float = 0.0, margin: float = 0.85) -> np.ndarray:
    """Max #nodes each device's usable memory can hold, INVERTING the SAME
    footprint model used by device_footprint_bytes / preflight (G1). A device
    holding n nodes (with ~avg_degree local edges each) needs, per the cost model:
        footprint(n) = window * (n*feat_dim*bytes_per_feat + n*avg_degree*bytes_per_edge)
    so  max_n = usable_mem / (window * (feat_dim*bytes_per_feat + avg_degree*bytes_per_edge)).
    Pass avg_degree = num_edges/num_nodes so capacity == feasibility (no fudge factor;
    this UNIFIES the partitioner's capacity with preflight -- fixes D16)."""
    per_node = cp.window * (cp.feat_dim * cp.bytes_per_feat + avg_degree * cp.bytes_per_edge)
    per_node = max(1.0, per_node)
    # `margin` (<1) leaves slack for edge-density variation (a consolidated device
    # holds a denser-than-average core) + activations, so capacity stays <= the true
    # preflight footprint and zord is RELIABLY feasible (not infeasible-by-a-hair).
    caps = [int(margin * d.usable_mem / per_node) for d in cluster.devices]
    return np.maximum(1, np.array(caps, dtype=np.int64))


@dataclass
class CostReport:
    footprint: np.ndarray          # bytes per device
    compute_sec: np.ndarray
    comm_sec: np.ndarray
    device_sec: np.ndarray         # compute+comm per device
    makespan_sec: float            # max device_sec (the slowest finishes)
    straggler: int                 # device index that dominates


def evaluate(p: Partition, cluster: ClusterProfile, cp: CostParams = CostParams()) -> CostReport:
    n = cluster.num_devices
    fp = np.array([device_footprint_bytes(p, i, cp) for i in range(n)])
    comp = np.array([device_compute_sec(p, i, cluster, cp) for i in range(n)])
    comm = np.array([device_comm_sec(p, i, cluster, cp) for i in range(n)])
    dev = comp + comm
    return CostReport(footprint=fp, compute_sec=comp, comm_sec=comm,
                      device_sec=dev, makespan_sec=float(dev.max()),
                      straggler=int(dev.argmax()))
