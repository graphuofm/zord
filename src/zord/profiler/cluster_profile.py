"""Heterogeneous cluster description: per-device capacity/throughput and the
(unequal) interconnect. These numbers are what make zord heterogeneity-aware;
the live profiler (device_profiler.py, run on HetCluster) fills them with measured
values. Until then we ship the spec-sheet defaults for the HetCluster tiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

GB = 1024 ** 3


@dataclass
class DeviceProfile:
    id: int
    name: str
    mem_bytes: int                 # D_k  (usable VRAM)
    throughput: float              # r_k  (relative; 1.0 = RTX5000Ada baseline)
    node: int = 0                  # which physical host (for intra/inter link)
    mem_reserved: int = 2 * GB     # framework/driver reserve
    h2d_gbps: float = 12.0         # measured host<->device PCIe bandwidth (R3 swap)
    hbm_bw_gbps: float = 500.0     # ACHIEVED aggregation (SpMM) bandwidth, GB/s -- this, not
                                   # FLOPs, sets the memory-bound GNN step time (roofline §9).
                                   # NB: achieved << spec peak on the irregular gather.
    measured: bool = False         # True once device_profiler overwrites

    @property
    def usable_mem(self) -> int:
        return max(0, self.mem_bytes - self.mem_reserved)


@dataclass
class ClusterProfile:
    devices: list[DeviceProfile]
    # bandwidth[i][j] in GB/s between device i and j (intra-node >> inter-node)
    bandwidth: Optional[list[list[float]]] = None
    intra_node_bw: float = 325.0   # MEASURED HetCluster H100 node (job 78699): NVLink/NVSwitch
                                   # 325 GB/s P2P. (Ada nodes are PCIe-only, no NVLink, ~tens GB/s.)
    inter_node_bw: float = 0.12    # MEASURED HetCluster 2-node gloo (job 78677): 0.12 GB/s (~1 Gbps
                                   # Ethernet). => NVLink/Ethernet skew ~2700x -> M2 (bandwidth-
                                   # weighted cut) has huge room; METIS is blind to this.

    def __post_init__(self):
        if self.bandwidth is None:
            n = len(self.devices)
            self.bandwidth = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        self.bandwidth[i][j] = float("inf")
                    elif self.devices[i].node == self.devices[j].node:
                        self.bandwidth[i][j] = self.intra_node_bw
                    else:
                        self.bandwidth[i][j] = self.inter_node_bw

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def total_throughput(self) -> float:
        return sum(d.throughput for d in self.devices)

    @property
    def total_usable_mem(self) -> int:
        return sum(d.usable_mem for d in self.devices)

    def throughput_shares(self) -> list[float]:
        """Fraction of total work each device *should* get to balance time."""
        tot = self.total_throughput
        return [d.throughput / tot for d in self.devices]


# MEASURED on HetCluster 2026-05-30. mem/h2d from microbench (jobs 78668-71); throughput
# r_k from the REAL full-batch GraphSAGE kernel (train_smoke, jobs 78674-76): per-step
# H100 5.11ms / RTX6000 7.42ms / RTX5000 10.9ms -> r normalized to RTX5000=1.0.
# NB: the GNN kernel REVERSES the fp32-matmul microbench -- on the real workload the
# RTX6000 is FASTER than RTX5000 (1.47 vs 1.0) and the H100 gap widens (2.13 vs 1.51).
# Memory (2.5x) + H2D bandwidth (2.15x) remain the binding heterogeneity.
# hbm_bw = ACHIEVED aggregation bandwidth from the roofline (jobs 78705/78706/78708, RESULTS §9):
# H100 ~942 GB/s (only ~28% of its 3350 spec -- gather locality caps it); RTX5000 ~444 GB/s
# (~77% of 576 spec). RTX6000 measured by 78708 (placeholder ~620 until collected). NB the real
# H100:RTX5000 agg ratio is ~2.1x, matching the full-step r ratio, NOT the 5.8x bandwidth spec.
_MEASURED = {
    "H100-80GB":       dict(mem_gb=79.2, r=2.13, h2d=57.5, hbm=942.0),  # GNN step 5.11ms
    "RTX6000Ada-48GB": dict(mem_gb=47.4, r=1.47, h2d=26.7, hbm=534.0),  # GNN step 7.42ms; agg@F256 measured
    "RTX5000Ada-32GB": dict(mem_gb=31.5, r=1.00, h2d=26.6, hbm=444.0),  # GNN step 10.9ms
}


def from_spec(hbm_gb, agg_bw_gbps, interconnect_gbps: float,
              h2d_gbps=12.0, throughput=None, nodes=None,
              names=None) -> ClusterProfile:
    """Build a heterogeneous cluster from an EXPLICIT spec -- the general entry the planner
    takes so zord is not tied to HetCluster. The INTERCONNECT BANDWIDTH is a PARAMETER (zord must
    win on the algorithm at ANY comm speed; nothing about NVLink is hardcoded).

    hbm_gb            : per-device usable HBM capacity in GB (list, len == #devices).
    agg_bw_gbps       : per-device ACHIEVED aggregation (SpMM) bandwidth in GB/s (list) --
                        this, not FLOPs, sets the memory-bound GNN step time (roofline).
    interconnect_gbps : cross-device link bandwidth in GB/s (a scalar PARAMETER). Used for
                        boundary comm + state-migration; set it to your fabric's measured value.
    h2d_gbps          : per-device host<->device PCIe bandwidth (scalar or list) for CPU staging.
    throughput        : per-device relative compute throughput r_k (list); defaults to
                        agg_bw normalized to its min (bandwidth-bound proxy).
    nodes             : per-device physical-host id (list); default all on one node (so every
                        pair uses the interconnect_gbps link). Devices on the SAME node would
                        get the (fast) intra-node link; here we model a flat fabric by default.
    names             : per-device display names (list); default 'devK'.
    """
    hbm_gb = list(hbm_gb)
    agg = list(agg_bw_gbps)
    n = len(hbm_gb)
    assert len(agg) == n, "hbm_gb and agg_bw_gbps must have the same length"
    if throughput is None:
        mn = min(agg) or 1.0
        throughput = [a / mn for a in agg]
    if nodes is None:
        nodes = [0] * n                          # flat fabric: every pair uses the interconnect
    if names is None:
        names = [f"dev{i}" for i in range(n)]
    if not isinstance(h2d_gbps, (list, tuple)):
        h2d_gbps = [h2d_gbps] * n
    devs = [DeviceProfile(i, names[i], int(round(hbm_gb[i] * GB)), throughput=throughput[i],
                          node=nodes[i], h2d_gbps=h2d_gbps[i], hbm_bw_gbps=agg[i], measured=False)
            for i in range(n)]
    # intra_node_bw == inter_node_bw == the single interconnect parameter (flat fabric default);
    # if the caller used distinct `nodes`, intra-node pairs still get this same value here, so the
    # link cost is unambiguously the PARAMETER the user passed.
    return ClusterProfile(devices=devs, intra_node_bw=interconnect_gbps,
                          inter_node_bw=interconnect_gbps)


def hetcluster(num_h100=1, num_6000ada=1, num_5000ada=1, gpus_per_node=8) -> ClusterProfile:
    """Build an HetCluster heterogeneous profile from MEASURED device numbers
    (hetcluster_measured.json). One physical node per tier (intra-node NVLink vs
    inter-node Ethernet handled by ClusterProfile bandwidth)."""
    def mk(did, key, node):
        m = _MEASURED[key]
        return DeviceProfile(did, key, int(m["mem_gb"] * GB), throughput=m["r"],
                             node=node, h2d_gbps=m["h2d"], hbm_bw_gbps=m["hbm"], measured=True)
    devs, did, node = [], 0, 0
    for _ in range(num_h100):
        devs.append(mk(did, "H100-80GB", node)); did += 1
    node += 1
    for _ in range(num_6000ada):
        devs.append(mk(did, "RTX6000Ada-48GB", node)); did += 1
    node += 1
    for _ in range(num_5000ada):
        devs.append(mk(did, "RTX5000Ada-32GB", node)); did += 1
    return ClusterProfile(devices=devs)
