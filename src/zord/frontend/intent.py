"""Declarative intent -> execution plan (R4). The user states WHAT they want;
zord resolves the partition + placement knobs and reports the tradeoff it chose.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ..partition import PARTITIONERS, Partition, CostParams
from ..guarantees import preflight, PreflightReport
from ..profiler.cluster_profile import ClusterProfile


class Intent(Enum):
    MIN_TIME = "min_time"          # smallest makespan that still fits
    FIT_MEMORY = "fit_memory"      # just make it not OOM, time secondary
    MAX_THROUGHPUT = "max_throughput"
    BALANCED = "balanced"


@dataclass
class Plan:
    intent: Intent
    partitioner: str
    partition: Partition
    preflight: PreflightReport

    def __str__(self):
        return (f"[zord plan] intent={self.intent.value} partitioner={self.partitioner}\n"
                f"  cross-edges={self.partition.total_cross_edges} "
                f"imbalance={self.partition.imbalance():.2f}\n{self.preflight}")


def resolve(src: np.ndarray, dst: np.ndarray, num_nodes: int,
            cluster: ClusterProfile, intent: Intent = Intent.MIN_TIME,
            cp: Optional[CostParams] = None, prior: Optional[Partition] = None) -> Plan:
    """Resolve an intent into a concrete plan. First cut: evaluate candidate
    partitioners and pick by the intent's objective, subject to feasibility."""
    cp = cp or CostParams()
    from ..partition.cost_model import max_nodes_per_device
    avg_deg = len(src) / max(num_nodes, 1)
    cap = max_nodes_per_device(cluster, cp, avg_degree=avg_deg)   # capacity == feasibility
    candidates = ["zord", "caphash"] if intent != Intent.FIT_MEMORY else ["zord", "caphash", "hash"]
    # intent -> consolidation knob: balance (high alpha) vs consolidate (low alpha)
    alpha = 2.0 if intent in (Intent.MIN_TIME, Intent.BALANCED) else 0.5
    scored = []
    for pname in candidates:
        kw = {"prior": prior}
        if pname == "zord":
            inst = PARTITIONERS["zord"](alpha=alpha, order="auto")   # regime-adaptive (D17)
            kw["capacity"] = cap           # zord uses the real memory budget
        else:
            inst = PARTITIONERS[pname]()
        part = inst.partition(src, dst, num_nodes, cluster, **kw)
        pf = preflight(part, cluster, cp)
        scored.append((pname, part, pf))

    feasible = [s for s in scored if s[2].feasible] or scored  # fall back if none fit

    if intent in (Intent.MIN_TIME, Intent.MAX_THROUGHPUT, Intent.BALANCED):
        key = lambda s: s[2].makespan_sec
    else:  # FIT_MEMORY: maximize min headroom
        key = lambda s: -min(s[2].headroom)
    pname, part, pf = min(feasible, key=key)
    return Plan(intent=intent, partitioner=pname, partition=part, preflight=pf)
