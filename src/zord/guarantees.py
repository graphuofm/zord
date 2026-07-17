"""Pre-flight feasibility checks (G1-G5) -- the heart of "submit and it
provably completes" instead of "submit and pray".

Given a Partition + ClusterProfile + CostParams, decide BEFORE launching:
  G1 no-OOM        : every device footprint <= its usable memory
  G2 bounded ms    : makespan estimate + straggler identification
  G5 predictability : report (fits?, est. seconds, est. peak mem) up front
(G3 no-hang and G4 convergence are runtime/training-time properties checked
elsewhere; surfaced here as TODO hooks.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .partition.base import Partition
from .partition.cost_model import CostParams, CostReport, evaluate
from .profiler.cluster_profile import ClusterProfile

GB = 1024 ** 3


@dataclass
class PreflightReport:
    feasible: bool                 # G1: fits on every device
    oom_devices: list             # device indices that would OOM
    makespan_sec: float            # G2
    straggler: int                 # G2
    headroom: list                 # usable - footprint per device (bytes)
    cost: CostReport
    warnings: list = field(default_factory=list)

    def __str__(self) -> str:
        verdict = "FEASIBLE" if self.feasible else "INFEASIBLE (would OOM)"
        lines = [f"[zord pre-flight] {verdict}",
                 f"  makespan ~= {self.makespan_sec*1e3:.2f} ms/batch  (straggler: dev {self.straggler})"]
        for i, h in enumerate(self.headroom):
            fp = self.cost.footprint[i] / GB
            flag = "  <-- OOM" if i in self.oom_devices else ""
            lines.append(f"  dev{i}: footprint {fp:6.2f} GB, headroom {h/GB:6.2f} GB{flag}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def preflight(p: Partition, cluster: ClusterProfile,
              cp: CostParams = CostParams()) -> PreflightReport:
    cost = evaluate(p, cluster, cp)
    headroom, oom = [], []
    for i, dev in enumerate(cluster.devices):
        h = dev.usable_mem - int(cost.footprint[i])
        headroom.append(h)
        if h < 0:
            oom.append(i)
    warnings = []
    if cost.makespan_sec > 0:
        ideal = cost.compute_sec.sum() / cluster.num_devices
        if cost.makespan_sec > 2 * ideal:
            warnings.append(f"load imbalance: makespan {cost.makespan_sec/ideal:.1f}x ideal "
                            f"(dev {cost.straggler} is the straggler)")
    return PreflightReport(feasible=(len(oom) == 0), oom_devices=oom,
                           makespan_sec=cost.makespan_sec, straggler=cost.straggler,
                           headroom=headroom, cost=cost, warnings=warnings)
