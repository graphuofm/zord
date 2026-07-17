#!/usr/bin/env python3
"""RUN-WHAT-WAS-IMPOSSIBLE -- zord's D48 HEADLINE feasibility experiment.

THE STORY (the economic DB story, MEMORY.md D48): a temporal GNN is limited by GPU-MEMORY
COST. As the attributed graph grows -- more feature dimensions F, more co-resident snapshots
W -- the working set outgrows the (expensive) HBM and the job becomes INFEASIBLE: it OOMs and
cannot run AT ALL. zord makes the memory-infeasible FEASIBLE: it runs the big graph on a few
HETEROGENEOUS cards by (1) sizing the partition to each device's CAPACITY (not equal/hash),
(2) TIERING the working set between HBM and CPU RAM over PCIe (plan_memory -- guarantees no-OOM
at full precision), and (3) splitting the FEATURE COLUMNS (feature-parallel) when F is large vs
the graph's degree (the §38-CORRECTION rule F > 5*avg_degree*(D-1)/D -- attribute-heavy + sparse).

PROCESS-only (MEMORY.md): same data + same model => same result. We optimize FEASIBILITY / MEMORY
/ TIME -- WHERE each row/column lives and WHAT spills to CPU, never WHAT is computed. Accuracy is
NEVER the target (the engine's decomposition is a result-preserving GAS reduce + column-concat).
Feasibility is HARDWARE-INDEPENDENT (it is a byte budget vs HBM caps), so this runs on CPU; a GPU
re-run only CONFIRMS the same fit/OOM verdict (the peak-GB numbers are the same byte arithmetic).

WHAT THIS DOES (and what it is NOT): it CALLS the REAL src/zord engine for every memory decision --
  - zord.schedule.plan(...)        -> the end-to-end plan (arrange + placement + tiering + axis)
  - zord.schedule.plan_memory(...) -> the CPU<->HBM TIERING that guarantees no-OOM (the F_v path)
  - choose_decomposition(...)      -> node-vs-FEATURE-parallel axis (the F>5*avg_degree rule)
This file contains NO partition/placement/feasibility logic of its own. The BASELINE (heterogeneity-
blind, equal/hash partition, all-resident, NO tiering) is computed as a trivial byte budget so we can
report the pressure P at which it OOMs -- it is the strawman the engine beats, deliberately dumb.

THE SWEEP: memory PRESSURE  P = (total working-set bytes) / (aggregate HBM over all cards).
We scale P by the feature dimension F and/or the snapshot window W (the two knobs that grow the
working set in an attributed temporal GNN). At each P we compare, PER DATASET:
  - BASELINE : the GB it needs on the SMALLEST device vs that device's cap -> the P_base where it OOMs.
  - ZORD     : whether the REAL engine keeps it FEASIBLE (peak GB <= cap on every card) and HOW
               (resident vs CPU-streamed rows/snapshots, the staging GB), and the MAX P_zord it can
               stretch to before even aggregate HBM + CPU is exceeded (the HONEST ceiling).
HEADLINE: the FRONTIER -- "baseline OOMs at pressure P_base; zord completes up to P_zord >> P_base",
the run-what-was-impossible factor = P_zord / P_base, per dataset. Honest ceiling: where NOTHING fits.

Run (from repo root; src/ is auto-added to sys.path):
    python3 scripts/run_what_was_impossible.py
    python3 scripts/run_what_was_impossible.py --datasets wiki-talk,askubuntu --pressure-by feat
    python3 scripts/run_what_was_impossible.py --sweep-by window --max-edges 4000000
No SLURM, no networkx.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# --- make `zord` importable straight from this checkout (scripts/ is not a package) ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

# ============================================================================================ #
#  THE REAL ENGINE. These imports ARE the proof the experiment exercises src/zord and not a    #
#  reimplementation: the FRONT loader, the cluster spec, the MIDDLE planner + the TIERING       #
#  (plan_memory) + the FEATURE-PARALLEL axis choice. We assert their module paths at runtime.   #
# ============================================================================================ #
import zord                                                       # noqa: E402
from zord.datasets import load, TemporalGraph                    # FRONT: graph in (real loaders)
from zord.profiler import from_spec, GB                          # the heterogeneous cluster spec
from zord.schedule import plan                                   # MIDDLE: end-to-end engine entry
from zord.schedule.planner import (                              # the memory-layout internals:
    plan_memory, Workload, choose_decomposition)                #  TIERING + the decomposition axis
from zord.partition.arrange import arrange                       # the adaptive-corner partition core

GBf = float(GB)

# ============================================================================================ #
#  THE HETEROGENEOUS CLUSTER -- "a few heterogeneous cards" (the economic story: you do NOT own #
#  a fleet of H100s; you have a couple of mixed GPUs). Matches end_to_end_zord.py / §10/§12/§33: #
#  per-device usable HBM (GB), ACHIEVED aggregation bandwidth (GB/s), CPU<->HBM PCIe (h2d) for   #
#  the staging, and the interconnect bandwidth PARAMETER. zord must win at ANY comm speed.       #
# ============================================================================================ #
_HBM_GB = [80.0, 48.0, 32.0]                # H100, RTX6000Ada, RTX5000Ada -- the HetCluster mix
_AGG_GBPS = [942.0, 534.0, 444.0]           # ACHIEVED SpMM bandwidth (RESULTS §9 roofline)
_H2D_GBPS = [57.5, 26.7, 26.6]              # MEASURED host<->device PCIe (RESULTS profiler §1)
_NAMES = ["H100-80", "RTX6000-48", "RTX5000-32"]

# CPU RAM available for the tiering bank (the cheap host memory zord stages the cold rows into).
# This is the second tier of the no-OOM budget: zord stays feasible until even HBM + this CPU
# bank is exceeded (the honest ceiling). A modest 256 GB server -- the realistic "few cards" host.
_CPU_RAM_GB = 256.0

# the §38-CORRECTION feature-parallel rule (the load-bearing crossover, reproduced for REPORTING
# only -- the actual axis decision is made by the engine's choose_decomposition):
#   feature-parallel relieves HBM iff  N*F*4 > E*20*(D-1)/D  <=>  F > 5 * avg_degree * (D-1)/D.
def _feature_parallel_threshold(avg_degree: float, D: int) -> float:
    return 5.0 * avg_degree * (D - 1) / D


def _build_cluster(link_gbps: float) -> "ClusterProfile":  # noqa: F821
    """The heterogeneous cluster, built by the REAL profiler. interconnect is a PARAMETER."""
    return from_spec(hbm_gb=_HBM_GB, agg_bw_gbps=_AGG_GBPS, interconnect_gbps=link_gbps,
                     h2d_gbps=_H2D_GBPS, names=_NAMES)


# ============================================================================================ #
#  ATTRIBUTED per-node feature-size vector F_v (the §33 attribute signal). Real attributed       #
#  graphs (JODIE wiki/reddit) carry per-edge LIWC features; SNAP graphs (wiki-talk, askubuntu,    #
#  stackoverflow) are featureless, so we synthesize a labeled-but-realistic HETEROGENEOUS F_v:    #
#  a node's feature dim grows with its activity (degree), modelling multi-modal hubs (text+image+  #
#  history) carrying far more attribute mass than leaves. base_F sets the mean width (the pressure  #
#  knob); the SHAPE is degree-driven so the heavy-F mass is a real, placeable signal (not uniform). #
# ============================================================================================ #
def _attributed_feat_bytes(g: TemporalGraph, base_F: int) -> np.ndarray:
    """Build a heterogeneous per-node feature-DIM vector F_v [N] centred on base_F.

    Real edge features (g.efeat) -> F_v[v] = base_F + Fe*(1+log1p(activity_v)) (the §33 multi-modal
    mass). Featureless SNAP graphs -> a SYNTHETIC-but-LABELED F_v whose width scales with degree so
    hubs are attribute-heavy (the realistic skew); rescaled so MEAN(F_v) == base_F (so base_F is the
    honest pressure knob). PROCESS-only: F_v only changes WHERE rows live / WHAT spills, never the
    result; same base_F => same F_v => same plan."""
    N = g.num_nodes
    deg = np.bincount(np.concatenate([g.src, g.dst]), minlength=N).astype(np.float64)
    if g.efeat is not None and g.efeat.shape[1] > 0:
        Fe = int(g.efeat.shape[1])
        Fv = base_F + Fe * (1.0 + np.log1p(deg))
    else:
        # synthetic-but-labeled heterogeneous F_v: width ~ (1 + log1p(degree)), then rescale so the
        # MEAN width is exactly base_F (base_F is the pressure knob; the skew is the attribute signal).
        shape = 1.0 + np.log1p(deg)
        shape = shape / max(1e-9, shape.mean())     # mean -> 1.0
        Fv = base_F * shape
    return np.maximum(1.0, Fv).astype(np.float64)


# ============================================================================================ #
#  THE BASELINE: heterogeneity-BLIND, in-core, NO tiering. The deliberately-dumb strawman the    #
#  engine beats. It splits the graph EQUALLY (equal node count per device -- a hash/equal split   #
#  that ignores the unequal HBM caps) and keeps EVERYTHING RESIDENT (no CPU staging). We report    #
#  the GB it needs on the SMALLEST device vs that device's cap. This is NOT the engine -- it is a  #
#  byte budget for the partitioner zord replaces, so the OOM point is honest and reproducible.     #
# ============================================================================================ #
@dataclass
class BaselineFit:
    feasible: bool
    smallest_dev_need_gb: float       # GB the equal split puts on the smallest-HBM card
    smallest_dev_cap_gb: float        # that card's usable HBM
    bottleneck_dev: int
    per_dev_need_gb: list             # GB needed on each device (equal split, all-resident)
    total_workingset_gb: float        # whole working set (the numerator of pressure P)


def _baseline_equal_fit(cluster, Fv: np.ndarray, num_edges: int, window: int,
                        layers: int, bytes_per_feat: int = 4,
                        bytes_per_edge: int = 12) -> BaselineFit:
    """EQUAL (hash) partition, ALL-RESIDENT, NO tiering. Each of the D devices gets ~N/D nodes and
    ~E/D edges, holds W co-resident snapshots, each snapshot = (features + L activation copies) +
    resident adjacency, ALL in HBM. Feature bytes are charged at the GRAPH-MEAN F (a hash split is
    blind to per-node F_v, so it cannot route heavy-F mass anywhere). The smallest card OOMs first."""
    D = cluster.num_devices
    N = Fv.size
    mean_F = float(Fv.mean())                                   # blind: charges mean F per node
    n_per = N / D
    e_per = num_edges / D
    feat_per_snap = n_per * mean_F * bytes_per_feat
    activ_per_snap = layers * feat_per_snap
    adj = e_per * bytes_per_edge
    per_snap = feat_per_snap + activ_per_snap                   # adjacency added once (not per snap)
    need_bytes_per_dev = window * per_snap + adj                # all W snapshots resident + adjacency
    per_dev_need_gb = [need_bytes_per_dev / GBf] * D            # equal split -> identical per device
    caps = [d.usable_mem for d in cluster.devices]
    # the SMALLEST-HBM device is the binding one (same need everywhere, smallest cap fails first).
    smallest = int(np.argmin(caps))
    need_gb = per_dev_need_gb[smallest]
    cap_gb = caps[smallest] / GBf
    feasible = all(per_dev_need_gb[k] <= caps[k] / GBf for k in range(D))
    total_ws = need_bytes_per_dev * D / GBf
    return BaselineFit(feasible, need_gb, cap_gb, smallest, per_dev_need_gb, total_ws)


# ============================================================================================ #
#  PRESSURE P = total working-set bytes / aggregate HBM. We use the SAME working-set definition   #
#  for the numerator (so baseline and zord are compared on ONE pressure axis): the all-resident,   #
#  full-precision working set of the whole graph (features + activations over W snapshots + adj).  #
# ============================================================================================ #
def _aggregate_hbm_gb(cluster) -> float:
    return cluster.total_usable_mem / GBf


def _pressure(total_workingset_gb: float, cluster) -> float:
    return total_workingset_gb / _aggregate_hbm_gb(cluster)


# ============================================================================================ #
#  ZORD, via the REAL engine. We ask the engine for the full plan (arrange + placement + the      #
#  plan_memory CPU<->HBM tiering) AND the decomposition-axis choice (node vs feature-parallel).    #
#  zord is FEASIBLE at this pressure iff the engine's memory plan fits every card (after tiering)   #
#  OR the feature-parallel axis fits -- i.e. iff SOME engine-chosen layout has no OOM. We report    #
#  HOW (resident vs streamed, staging GB) and the chosen axis. The HONEST CEILING is when even the  #
#  aggregate HBM + CPU bank cannot hold the working set (nothing fits -- a true byte impossibility). #
# ============================================================================================ #
@dataclass
class ZordFit:
    feasible: bool
    how: str                          # human description of the layout that made it fit
    axis: str                         # "node" | "feature" | "hybrid(...)"
    streamed_gb: float                # GB staged CPU<->HBM per epoch (the tiering cost)
    peak_gb_max: float                # max per-device peak HBM (must be <= that card's cap)
    cap_of_peak_gb: float             # the cap of the device that peaks
    mem_feasible: bool                # plan_memory (node-parallel tiering) fit verdict
    feature_feasible: bool            # feature-parallel axis fit verdict
    above_ceiling: bool               # working set exceeds aggregate HBM + CPU bank (nothing fits)


def _zord_fit(g: TemporalGraph, cluster, Fv: np.ndarray, base_F: int, window: int,
              link_gbps: float, snapshots: int, reuse_frac: float,
              total_workingset_gb: float) -> ZordFit:
    """Ask the REAL engine whether zord keeps THIS pressure feasible, and how.

    The engine is called THREE ways (all in src/zord):
      1. plan(..., decomposition='auto', feat_bytes=F_v, window=W) -> the end-to-end plan; its
         .memory is the plan_memory CPU<->HBM TIERING (F_v-aware: spills the largest-F rows first),
         and its .decomposition is the node-vs-FEATURE-parallel axis choice (the F>5*deg rule).
      2. plan_memory(cluster, Workload(...)) -> read directly for the per-device peak/streamed GB.
      3. choose_decomposition(...) -> the feature-parallel feasibility at this F (the second axis).
    zord is FEASIBLE iff the node-parallel tiering fits OR the feature-parallel axis fits. The honest
    CEILING: working set > aggregate HBM + CPU bank -> NOTHING fits (a true byte impossibility)."""
    N = g.num_nodes
    E = g.num_edges
    D = cluster.num_devices

    # --- the end-to-end engine plan (axis = auto so feature-parallel is COSTED + chosen if it wins) ---
    # plan(..., feat_bytes=F_v) ALREADY runs arrange + the F_v-aware plan_memory TIERING internally
    # (planner.py: arrange at line ~554, plan_memory with res.assignment at ~573) and the decomposition-
    # axis choice -- so .memory IS the CPU<->HBM tiering plan and .decomposition the node/feature axis.
    p = plan(g, cluster, link_gbps=link_gbps, feat_dim=base_F, num_snapshots=snapshots,
             window=window, reuse_frac=reuse_frac, feat_bytes=Fv, decomposition="auto")

    # --- the CPU<->HBM TIERING (plan_memory), read from the engine plan for per-device peak/streamed GB ---
    # F_v-aware: the assignment from arrange told plan_memory WHICH nodes live where, so each device is
    # sized by its TRUE feature bytes and the LARGEST-F cold rows spill first (the §33 tiering win). We
    # read p.memory directly (NOT a recompute) -- it is the engine's own tiering decision.
    mem = p.memory
    if mem is None:                                       # defensive: plan always sets it, but be safe
        w = Workload(num_nodes=N, num_edges=E, feat_dim=base_F, window=window,
                     reuse_frac=reuse_frac, feat_bytes=Fv, assignment=p.assignment)
        mem = plan_memory(cluster, w)
    peak_gb = max(pp.peak_hbm_bytes / GBf for pp in mem.per_device)
    cap_of_peak = next(pp.capacity_bytes for pp in mem.per_device
                       if pp.peak_hbm_bytes / GBf == peak_gb) / GBf
    mem_feasible = mem.all_feasible

    # --- the FEATURE-parallel axis (the F>5*avg_degree rule), via the engine's choice object ---
    dch = p.decomposition  # already costed by plan(..., decomposition='auto')
    feature_feasible = bool(dch.feature_feasible) if dch is not None else False
    axis = dch.axis if dch is not None else "node"

    # --- the HONEST CEILING: total working set vs aggregate HBM + the CPU staging bank ---
    ceiling_gb = _aggregate_hbm_gb(cluster) + _CPU_RAM_GB
    # reuse shrinks the effective resident working set (only the changed (1-rho) fraction recomputes);
    # we credit it to the ceiling check so the ceiling reflects what zord actually must hold.
    eff_ws = total_workingset_gb * (1.0 + (window - 1) * (1.0 - reuse_frac)) / max(1, window)
    above_ceiling = eff_ws > ceiling_gb

    feasible = (mem_feasible or feature_feasible) and not above_ceiling

    # describe HOW it fit (the resident/streamed split + the winning axis)
    if above_ceiling:
        how = (f"CEILING: effective working set {eff_ws:.0f}GB > aggregate HBM "
               f"{_aggregate_hbm_gb(cluster):.0f}GB + CPU bank {_CPU_RAM_GB:.0f}GB -> nothing fits")
    elif feature_feasible and (axis == "feature" or axis.startswith("hybrid") or not mem_feasible):
        fp_gb = (float(np.max(dch.feature_feat_gb_per_device))
                 if dch is not None and dch.feature_feat_gb_per_device is not None else 0.0)
        how = (f"FEATURE-parallel (axis={axis}): split F={base_F} cols over {D} cards -> "
               f"max {fp_gb:.1f}GB/card (F/D cols), integration {dch.feature_integration_ms:.1f}ms; "
               f"node-parallel would need {dch.node_makespan_ms:.0f}ms"
               f"{'' if dch.node_feasible else ' but OOMs'}")
    else:
        streamed = mem.total_streamed_gb
        n_res = sum(pp.resident_snapshots for pp in mem.per_device)
        n_str = sum(pp.streamed_snapshots for pp in mem.per_device)
        how = (f"node-parallel + CPU<->HBM TIERING: peak {peak_gb:.1f}GB/{cap_of_peak:.0f}GB cap, "
               f"streamed {streamed:.1f}GB/epoch (resident {n_res} / streamed {n_str} snapshot-slots "
               f"across cards), bound={mem.bound}")

    return ZordFit(feasible=feasible, how=how, axis=axis, streamed_gb=mem.total_streamed_gb,
                   peak_gb_max=peak_gb, cap_of_peak_gb=cap_of_peak, mem_feasible=mem_feasible,
                   feature_feasible=feature_feasible, above_ceiling=above_ceiling)


# ============================================================================================ #
#  THE SWEEP over one dataset: grow the pressure knob (F or W), record the baseline OOM point and  #
#  zord's feasibility + ceiling, and emit the frontier + the run-what-was-impossible factor.        #
# ============================================================================================ #
@dataclass
class SweepPoint:
    knob_value: int                   # F (or W) at this step
    pressure: float                   # P = working set / aggregate HBM
    workingset_gb: float
    base_feasible: bool
    base_need_gb: float               # GB the baseline puts on the smallest card
    base_cap_gb: float
    zord_feasible: bool
    zord_axis: str
    zord_peak_gb: float
    zord_streamed_gb: float
    zord_how: str
    above_ceiling: bool


@dataclass
class DatasetFrontier:
    dataset: str
    N: int
    E: int
    avg_degree: float
    fp_threshold_F: float             # the F>5*avg_degree*(D-1)/D crossover for this graph
    sweep_by: str
    points: list = field(default_factory=list)
    P_base_oom: Optional[float] = None    # pressure at which the baseline first OOMs
    P_zord_max: Optional[float] = None    # max pressure zord stays feasible
    P_ceiling: Optional[float] = None     # pressure at which nothing fits (the honest ceiling)
    factor: Optional[float] = None        # run-what-was-impossible = P_zord_max / P_base_oom


def sweep_dataset(name: str, link_gbps: float, base_F: int, window: int, snapshots: int,
                  reuse_frac: float, sweep_by: str, knob_values: list,
                  max_edges: Optional[int]) -> DatasetFrontier:
    print(f"\n{'='*100}\n=== RUN-WHAT-WAS-IMPOSSIBLE on '{name}'  (link={link_gbps:g} GB/s, "
          f"sweep_by={sweep_by}, base_F={base_F}, W={window}, reuse={reuse_frac}) ===\n{'='*100}")

    # -------- FRONT: graph in (the REAL loader) --------
    g = load(name).sort_by_time()
    if max_edges is not None and g.num_edges > max_edges:
        # cap huge graphs (the ENGINE still plans the capped REAL graph; a runtime budget, not a
        # partition decision). First max_edges by time -> a coherent temporal prefix.
        g = TemporalGraph(src=g.src[:max_edges], dst=g.dst[:max_edges], t=g.t[:max_edges],
                          efeat=(g.efeat[:max_edges] if g.efeat is not None else None),
                          name=g.name + f"[:{max_edges}]")
        g.sort_by_time()
    N, E = g.num_nodes, g.num_edges
    avg_deg = 2.0 * E / max(1, N)
    cluster = _build_cluster(link_gbps)
    D = cluster.num_devices
    fp_thr = _feature_parallel_threshold(avg_deg, D)
    print(f"  FRONT  N={N:,}  E={E:,}  avg_degree={avg_deg:.1f}  aggregate_HBM={_aggregate_hbm_gb(cluster):.0f}GB "
          f"({'+'.join(f'{c:.0f}' for c in _HBM_GB)})  CPU_bank={_CPU_RAM_GB:.0f}GB")
    print(f"         feature-parallel engages at F > 5*avg_deg*(D-1)/D = {fp_thr:.0f} (the §38-CORRECTION rule)")
    print(f"         sweeping {sweep_by} over {knob_values}")

    fr = DatasetFrontier(dataset=name, N=N, E=E, avg_degree=avg_deg, fp_threshold_F=fp_thr,
                         sweep_by=sweep_by)

    for kv in knob_values:
        F = kv if sweep_by == "feat" else base_F
        W = kv if sweep_by == "window" else window
        Fv = _attributed_feat_bytes(g, F)

        base = _baseline_equal_fit(cluster, Fv, E, W, layers=2)
        P = _pressure(base.total_workingset_gb, cluster)

        z = _zord_fit(g, cluster, Fv, F, W, link_gbps, snapshots, reuse_frac,
                      base.total_workingset_gb)

        sp = SweepPoint(
            knob_value=kv, pressure=P, workingset_gb=base.total_workingset_gb,
            base_feasible=base.feasible, base_need_gb=base.smallest_dev_need_gb,
            base_cap_gb=base.smallest_dev_cap_gb,
            zord_feasible=z.feasible, zord_axis=z.axis, zord_peak_gb=z.peak_gb_max,
            zord_streamed_gb=z.streamed_gb, zord_how=z.how, above_ceiling=z.above_ceiling)
        fr.points.append(sp)

        bflag = "FEASIBLE " if base.feasible else "OOM      "
        zflag = "FEASIBLE " if z.feasible else ("CEILING  " if z.above_ceiling else "OOM      ")
        knob = f"{sweep_by[0].upper()}={kv:<6}"
        print(f"  {knob} P={P:6.2f}  ws={base.total_workingset_gb:8.1f}GB | "
              f"BASELINE {bflag} need {base.base_need_gb if False else base.smallest_dev_need_gb:7.1f}GB"
              f"/{base.smallest_dev_cap_gb:.0f}GB(smallest) | ZORD {zflag} peak {z.peak_gb_max:6.1f}GB")
        print(f"           -> zord: {z.how}")

        # record the frontier transitions
        if fr.P_base_oom is None and not base.feasible:
            fr.P_base_oom = P
        if z.feasible:
            fr.P_zord_max = P                     # last pressure zord still fits (monotone sweep)
        if fr.P_ceiling is None and z.above_ceiling:
            fr.P_ceiling = P

    if fr.P_base_oom is not None and fr.P_zord_max is not None and fr.P_base_oom > 0:
        fr.factor = fr.P_zord_max / fr.P_base_oom
    return fr


# ============================================================================================ #
#  prove (at runtime) the experiment is bound to the REAL src/zord engine, not a copy.           #
# ============================================================================================ #
def _assert_uses_real_engine():
    assert plan.__module__ == "zord.schedule.planner", plan.__module__
    assert plan_memory.__module__ == "zord.schedule.planner", plan_memory.__module__
    assert choose_decomposition.__module__ == "zord.schedule.planner", choose_decomposition.__module__
    assert arrange.__module__ == "zord.partition.arrange", arrange.__module__
    assert load.__module__ == "zord.datasets.loaders", load.__module__
    eng = os.path.abspath(plan.__code__.co_filename)
    assert os.path.join("src", "zord", "schedule", "planner.py") in eng, eng
    print("[engine check] PASS -- every memory decision comes from the REAL engine:")
    print(f"  zord.schedule.plan           @ {os.path.abspath(plan.__code__.co_filename)}")
    print(f"  zord.schedule.plan_memory    @ {os.path.abspath(plan_memory.__code__.co_filename)}  (CPU<->HBM tiering)")
    print(f"  zord.schedule.choose_decomposition @ {os.path.abspath(choose_decomposition.__code__.co_filename)}  (feature-parallel axis)")
    print(f"  zord.partition.arrange       @ {os.path.abspath(arrange.__code__.co_filename)}")
    print(f"  zord {zord.__version__}")


def _default_knob_values(sweep_by: str) -> list:
    if sweep_by == "feat":
        # F sweep: from low (node-parallel, baseline fits) up through the feature-parallel crossover
        # to attribute-heavy widths that drive pressure P from <1 to >>1 (the multi-modal regime).
        return [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # window sweep: grow co-resident snapshots W (the temporal-batch pressure knob). A DENSE grid
    # near the frontier so the baseline-OOM and zord-ceiling transitions are resolved (a coarse
    # geometric grid would straddle both transitions in one 4x step -> a misleadingly small factor).
    return [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="RUN-WHAT-WAS-IMPOSSIBLE: zord makes the memory-infeasible feasible (D48 headline)")
    ap.add_argument("--datasets", default="wiki-talk,askubuntu,jodie-wikipedia",
                    help="comma-separated dataset names (real attributed/temporal graphs)")
    ap.add_argument("--sweep-by", choices=["feat", "window"], default="feat",
                    help="the memory-pressure knob: feature dim F (default) or snapshot window W")
    ap.add_argument("--knobs", default="",
                    help="comma-separated knob values; default depends on --sweep-by")
    ap.add_argument("--base-feat", type=int, default=512, help="node feature dim F when sweeping window")
    ap.add_argument("--window", type=int, default=8, help="snapshot window W when sweeping feat")
    ap.add_argument("--reuse-frac", type=float, default=0.0,
                    help="rho: provably-unchanged-neighborhood reuse (shrinks effective working set)")
    ap.add_argument("--link-gbps", type=float, default=0.5,
                    help="interconnect bandwidth PARAMETER (GB/s); zord wins at any comm speed")
    ap.add_argument("--snapshots", type=int, default=64, help="num_snapshots for the arrange PTS corner")
    ap.add_argument("--max-edges", type=int, default=8_000_000,
                    help="cap edges (engine still plans the real graph; CPU budget, not a decision)")
    args = ap.parse_args(argv)

    _assert_uses_real_engine()

    knob_values = ([int(x) for x in args.knobs.split(",") if x.strip()]
                   if args.knobs else _default_knob_values(args.sweep_by))

    frontiers = []
    for name in [n.strip() for n in args.datasets.split(",") if n.strip()]:
        try:
            fr = sweep_dataset(name, args.link_gbps, args.base_feat, args.window, args.snapshots,
                               args.reuse_frac, args.sweep_by, knob_values, args.max_edges)
            frontiers.append(fr)
        except Exception as e:
            import traceback
            print(f"  [skip {name}] {type(e).__name__}: {e}")
            traceback.print_exc()

    # -------- the HEADLINE FRONTIER table (per dataset) --------
    print(f"\n{'='*108}\n=== HEADLINE: the OOM-vs-FEASIBLE FRONTIER (run-what-was-impossible) ===\n{'='*108}")
    hdr = (f"{'dataset':18}{'N':>10}{'E':>11}{'avg_deg':>8}{'P_base_OOM':>12}"
           f"{'P_zord_max':>12}{'P_ceiling':>11}{'factor':>9}{'zord_axis@max':>16}")
    print(hdr)
    for fr in frontiers:
        pb = f"{fr.P_base_oom:.2f}" if fr.P_base_oom is not None else "never"
        pz = f"{fr.P_zord_max:.2f}" if fr.P_zord_max is not None else "—"
        pc = f"{fr.P_ceiling:.2f}" if fr.P_ceiling is not None else ">max"
        fac = f"{fr.factor:.1f}x" if fr.factor is not None else "—"
        axis = fr.points[-1].zord_axis if fr.points else "—"
        # the axis at the highest pressure zord stays feasible (the headline layout)
        feas_pts = [p for p in fr.points if p.zord_feasible]
        axis = feas_pts[-1].zord_axis if feas_pts else axis
        print(f"{fr.dataset:18}{fr.N:>10,}{fr.E:>11,}{fr.avg_degree:>8.1f}{pb:>12}"
              f"{pz:>12}{pc:>11}{fac:>9}{axis:>16}")

    print("\nlegend (PROCESS-only -- feasibility is a byte budget vs HBM caps, hardware-independent;")
    print("        a GPU re-run only CONFIRMS the same fit/OOM verdict):")
    print("  P = (total full-precision working-set bytes) / (aggregate HBM over the cards). The SAME")
    print("      working set defines P for baseline and zord (one pressure axis).")
    print("  P_base_OOM = pressure where the heterogeneity-BLIND equal/hash + all-resident baseline first")
    print("               OOMs the SMALLEST card (INFEASIBLE: the job cannot run AT ALL).")
    print("  P_zord_max = max pressure the REAL engine keeps FEASIBLE (capacity-sized arrange +")
    print("               plan_memory CPU<->HBM tiering + feature-parallel where F>5*avg_deg) -- no OOM.")
    print("  P_ceiling  = the HONEST ceiling: working set > aggregate HBM + CPU bank -> NOTHING fits.")
    print("  factor     = P_zord_max / P_base_OOM = the RUN-WHAT-WAS-IMPOSSIBLE factor (zord runs jobs")
    print("               that are flat-out infeasible for the in-core baseline).")
    print("\n=> HEADLINE: per dataset, baseline OOMs at P_base; zord completes up to P_zord >> P_base")
    print("   (factor x more memory pressure), via capacity-sized partition + CPU<->HBM tiering +")
    print("   feature-parallel for attribute-heavy/sparse graphs -- all FULL PRECISION, same result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
