"""AUTO-PROBER (BACKLOG L2 hardware half + the calibration that closes the loop).

This module turns the *environment* into the *cost-model parameters* so that NO
weight is hand-set (closes BACKLOG L2; feeds N2). It has two responsibilities:

  1. HARDWARE PROBE (`probe_hardware`): produce a populated ``ClusterProfile``.
     - ``measure=False`` (default, fully torch-free): return the spec-sheet
       HetCluster profile (or a caller-supplied ``ClusterProfile``) unchanged, so the
       whole pipeline plans on a CPU box with no torch installed.
     - ``measure=True`` AND torch+CUDA present: microbench the ACHIEVED HBM
       aggregation bandwidth (SpMM-roofline gather), the H2D PCIe bandwidth and
       the device-to-device link bandwidth, then OVERWRITE
       ``hbm_bw_gbps`` / ``h2d_gbps`` / link with the measured numbers. torch is
       imported LAZILY inside the measure branch only -> import-safe everywhere.

  2. CALIBRATION (`calibrate`): from a ``ProbeResult`` + an optional ``GraphStats``
     derive the THEORY §2 link-set weights ``w_S, w_T`` and a ``CostParams`` whose
     ``sec_per_edge`` comes from the *measured* bottleneck HBM bandwidth. The
     result is a single ``CostCalibration`` object the scheduler consumes, so the
     cost-model parameters are self-calibrated rather than hand-set.

THEORY §2 (the formulas implemented in ``calibrate``):
    SpatialComm  = w_S * SpatialCut ,  w_S = bytes_per_halo / B_link
    TemporalComm = w_T * TemporalCut, w_T = min( bytes_per_memory / B_link,
                                                 recompute_cost * (1 - rho) )
where ``bytes_per_halo = FEATURE_ROW_BYTES * F * hops`` (a halo node ships its
feature row across the cut once per hop) and ``rho`` is the temporal persistence
(``GraphStats.persistence``, THEORY §9.4): a high-persistence graph reuses its
temporal state losslessly, so the temporal-transfer weight ``w_T`` shrinks.

PROCESS-ONLY / FULL-PRECISION: nothing here changes WHAT is computed. The
microbenchmarks move fp32 bytes (no TF32/FP16) purely to read back achieved
bandwidth; the calibration only sets cost weights. Import-safe: only the
``measure=True`` path touches torch, behind a lazy import + try/except.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from .cluster_profile import ClusterProfile, DeviceProfile, hetcluster, from_spec
from ..partition.cost_model import CostParams

if TYPE_CHECKING:  # import-safe: ingest.py may not exist / may import heavy deps
    from ..frontend.ingest import GraphStats
    from ..datasets.temporal_graph import TemporalGraph

GB: int = 1024 ** 3

# fp32 feature word -- matches arrange / cost_model (NO precision reduction).
FEATURE_ROW_BYTES: float = 4.0


# --------------------------------------------------------------------------- #
# Result dataclasses                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    """Output of the HARDWARE probe.

    cluster   : a ``ClusterProfile`` whose ``hbm_bw_gbps`` / ``h2d_gbps`` are
                either spec-sheet (measured=False) or MEASURED (measure=True).
    link_gbps : the cross-device link bandwidth (GB/s) used for cut/comm cost.
    measured  : True iff the microbenchmarks actually ran on a CUDA device.
    notes     : human-readable trace of what was measured / fell back to spec.
    """
    cluster: "ClusterProfile"
    link_gbps: float
    measured: bool
    notes: list = field(default_factory=list)


@dataclass
class CostCalibration:
    """The single self-calibrated object the scheduler consumes (closes L2).

    cluster      : the (possibly measured) hardware profile.
    link_gbps    : cross-device link bandwidth (GB/s).
    graph_stats  : the cheap graph probe (GraphStats) or None if unavailable.
    cost_params  : a ``CostParams`` whose ``sec_per_edge`` is derived from the
                   MEASURED bottleneck HBM bandwidth (no hand-set constant).
    w_S, w_T     : THEORY §2 link-set weights (spatial / temporal cut weights)
                   the corner-selection uses to pick PSS vs PTS vs blend.
    note         : free-form provenance string.
    """
    cluster: "ClusterProfile"
    link_gbps: float
    graph_stats: Optional["GraphStats"]
    cost_params: "CostParams"
    w_S: float
    w_T: float
    note: str = ""

    def summary(self) -> str:
        ndev = self.cluster.num_devices
        hbm = [round(d.hbm_bw_gbps, 1) for d in self.cluster.devices]
        regime = "spatial-bound (PSS-leaning)" if self.w_S >= self.w_T \
            else "temporal-bound (PTS-leaning)"
        lines = [
            f"CostCalibration: {ndev} dev | link={self.link_gbps:.3g} GB/s | "
            f"hbm_bw(achieved)={hbm} GB/s",
            f"  w_S={self.w_S:.4g}  w_T={self.w_T:.4g}  -> regime: {regime}",
            f"  sec_per_edge={self.cost_params.sec_per_edge:.3g}  "
            f"feat_dim={self.cost_params.feat_dim}  window={self.cost_params.window}",
        ]
        if self.graph_stats is not None:
            gs = self.graph_stats
            lines.append(
                f"  graph: N={gs.num_nodes} E={gs.num_edges} "
                f"deg_avg={gs.avg_degree:.2f} persistence={gs.persistence:.3f} "
                f"clusterability={gs.clusterability:.3f}"
            )
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Microbenchmarks (torch path, called ONLY when measure=True)                 #
# --------------------------------------------------------------------------- #
def measure_hbm_bw_gbps(device: str, bytes_moved: int = 1 << 30) -> float:
    """Achieved HBM aggregation bandwidth (GB/s) via an SpMM-style gather roofline.

    This emulates the GNN aggregation memory pattern: an irregular gather of
    feature rows followed by a reduction (a memory-bound scatter-add). We move
    ``bytes_moved`` of fp32 data so the kernel is bandwidth-bound, time it on
    CUDA events, and return ``bytes / seconds / 1e9``. The number is the
    ACHIEVED (not spec-peak) bandwidth -- exactly what sets the memory-bound GNN
    step time (THEORY §2 Compute_k = AggWork / B_hbm). FULL fp32, no TF32.

    Raises if torch/CUDA are unavailable -- only called under measure=True.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover - measure path only
        raise RuntimeError("measure_hbm_bw_gbps requires torch") from e
    if not torch.cuda.is_available():  # pragma: no cover - measure path only
        raise RuntimeError("measure_hbm_bw_gbps requires CUDA")

    dev = torch.device(device)
    # fp32, full precision -- disable TF32 so the measurement is honest.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    word = 4  # fp32
    F = 128                                    # feature width of a gathered row
    row_bytes = F * word
    # Number of gather rows so that touching src+dst moves ~bytes_moved.
    n_rows = max(1, int(bytes_moved // (2 * row_bytes)))
    n_nodes = max(2, n_rows)

    x = torch.randn(n_nodes, F, device=dev, dtype=torch.float32)
    out = torch.zeros(n_nodes, F, device=dev, dtype=torch.float32)
    # Random gather/scatter indices -> irregular access (the GNN aggregation cap).
    g = torch.randint(0, n_nodes, (n_rows,), device=dev)
    s = torch.randint(0, n_nodes, (n_rows,), device=dev)

    def _step():
        out.index_add_(0, s, x.index_select(0, g))

    # Warm up (allocations, kernel JIT).
    for _ in range(3):
        _step()
    torch.cuda.synchronize()

    reps = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        _step()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / reps
    sec = ms / 1e3
    # Each step touches: gather read (n_rows*row_bytes) + scatter read-modify-write
    # (2 * n_rows*row_bytes). Charge 3x to reflect the read+RMW traffic.
    moved = 3.0 * n_rows * row_bytes
    return float(moved / max(sec, 1e-9) / 1e9)


def measure_link_gbps(devices: list = [0, 1], bytes_moved: int = 1 << 28) -> float:
    """P2P (device->device) or H2D (host->device) copy bandwidth in GB/s.

    If two distinct CUDA devices are available, time a device-to-device copy;
    otherwise fall back to a pinned host->device copy (the staging link). fp32
    payload, CUDA-event timed. Raises if torch/CUDA absent -- measure path only.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover - measure path only
        raise RuntimeError("measure_link_gbps requires torch") from e
    if not torch.cuda.is_available():  # pragma: no cover - measure path only
        raise RuntimeError("measure_link_gbps requires CUDA")

    n_el = max(1, bytes_moved // 4)  # fp32 words
    ndev = torch.cuda.device_count()

    if ndev >= 2 and len(devices) >= 2 and devices[0] != devices[1]:
        src = torch.randn(n_el, device=torch.device(f"cuda:{devices[0]}"),
                          dtype=torch.float32)
        dst = torch.empty(n_el, device=torch.device(f"cuda:{devices[1]}"),
                          dtype=torch.float32)

        def _copy():
            dst.copy_(src)
    else:
        # Single device: measure pinned H2D as the link proxy.
        host = torch.empty(n_el, dtype=torch.float32).pin_memory()
        dst = torch.empty(n_el, device=torch.device(f"cuda:{devices[0]}"),
                          dtype=torch.float32)

        def _copy():
            dst.copy_(host, non_blocking=True)

    for _ in range(3):
        _copy()
    torch.cuda.synchronize()

    reps = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        _copy()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / reps
    sec = ms / 1e3
    moved = n_el * 4
    return float(moved / max(sec, 1e-9) / 1e9)


# --------------------------------------------------------------------------- #
# Hardware probe entry                                                        #
# --------------------------------------------------------------------------- #
def probe_hardware(cluster: Optional["ClusterProfile"] = None, *,
                   measure: bool = False,
                   device: Optional[str] = None) -> ProbeResult:
    """Produce a populated ``ClusterProfile`` (the L2 hardware half).

    measure=False (default): return the spec-sheet profile (``cluster`` if given,
        else ``hetcluster()``) unchanged. Fully torch-free -- the whole pipeline
        plans on a CPU box.
    measure=True: if torch+CUDA are present, microbench achieved HBM bandwidth
        (per device), H2D PCIe and the device link, and OVERWRITE the profile's
        ``hbm_bw_gbps`` / ``h2d_gbps`` / link fields with the measured numbers.
        If torch/CUDA are absent we degrade gracefully to the spec profile and
        record the fallback in ``notes`` (so a measure request on a CPU box does
        not crash the pipeline).

    The link bandwidth is taken as the (finite) cross-device link from the
    profile -- the smallest finite off-diagonal entry, i.e. the bottleneck fabric
    that the cut/comm cost is charged at.
    """
    base = cluster if cluster is not None else hetcluster()
    notes: list = []
    measured = False

    if measure:
        ok_torch = False
        try:
            import torch  # noqa: F401  (lazy: only here)
            ok_torch = True
        except Exception:
            ok_torch = False

        cuda_ok = False
        if ok_torch:
            try:
                import torch
                cuda_ok = bool(torch.cuda.is_available())
            except Exception:
                cuda_ok = False

        if cuda_ok:
            import torch
            ndev_phys = torch.cuda.device_count()
            for i, dev in enumerate(base.devices):
                phys = i if device is None else 0
                phys = min(phys, max(0, ndev_phys - 1))
                devstr = device if device is not None else f"cuda:{phys}"
                try:
                    bw = measure_hbm_bw_gbps(devstr)
                    dev.hbm_bw_gbps = bw
                    dev.measured = True
                    notes.append(f"dev{i} ({dev.name}): measured HBM agg "
                                 f"{bw:.1f} GB/s on {devstr}")
                except Exception as e:
                    notes.append(f"dev{i}: HBM measure failed ({e}); kept spec "
                                 f"{dev.hbm_bw_gbps:.1f} GB/s")
                try:
                    h2d = measure_link_gbps(devices=[phys], bytes_moved=1 << 26)
                    dev.h2d_gbps = h2d
                    notes.append(f"dev{i}: measured H2D {h2d:.1f} GB/s")
                except Exception as e:
                    notes.append(f"dev{i}: H2D measure failed ({e}); kept spec "
                                 f"{dev.h2d_gbps:.1f} GB/s")
            # Device link (P2P) if >= 2 physical GPUs.
            if ndev_phys >= 2:
                try:
                    link = measure_link_gbps(devices=[0, 1])
                    # Rebuild bandwidth matrix off this measured inter-device link.
                    base.intra_node_bw = link
                    base.inter_node_bw = min(base.inter_node_bw, link)
                    base.bandwidth = None
                    base.__post_init__()
                    notes.append(f"measured device link {link:.2f} GB/s")
                except Exception as e:
                    notes.append(f"link measure failed ({e}); kept spec links")
            measured = True
        else:
            notes.append("measure=True but torch/CUDA unavailable; "
                         "falling back to spec-sheet ClusterProfile")
    else:
        notes.append("spec-sheet ClusterProfile (measure=False, torch-free path)")

    link_gbps = _bottleneck_link_gbps(base)
    return ProbeResult(cluster=base, link_gbps=link_gbps,
                       measured=measured, notes=notes)


def _bottleneck_link_gbps(cluster: "ClusterProfile") -> float:
    """The finite cross-device link bandwidth (GB/s) the cut is charged at.

    Returns the smallest finite off-diagonal entry of the bandwidth matrix (the
    bottleneck fabric). For a single-device profile there is no cross-link, so we
    fall back to the device H2D (the only off-chip path)."""
    n = cluster.num_devices
    if n <= 1:
        return float(cluster.devices[0].h2d_gbps)
    finite = []
    bw = cluster.bandwidth
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = bw[i][j]
            if v != float("inf") and v > 0:
                finite.append(v)
    if not finite:
        return float(cluster.devices[0].h2d_gbps)
    return float(min(finite))


# --------------------------------------------------------------------------- #
# Calibration: probe -> cost-model weights (closes the loop)                  #
# --------------------------------------------------------------------------- #
def _bottleneck_hbm_gbps(cluster: "ClusterProfile") -> float:
    """Slowest achieved HBM aggregation bandwidth across devices.

    The makespan is set by the slowest device (the straggler bound), so the
    cost-model ``sec_per_edge`` is derived from the MINIMUM achieved bandwidth."""
    return float(min(d.hbm_bw_gbps for d in cluster.devices))


def calibrate(probe: ProbeResult, stats: Optional["GraphStats"] = None, *,
              feat_dim: int = 128, window: int = 1, layers: int = 2,
              hops: int = 2) -> CostCalibration:
    """Derive THEORY §2 link-set weights + a self-calibrated ``CostParams``.

    Weights (THEORY §2):
        bytes_per_halo  = FEATURE_ROW_BYTES * feat_dim * hops
            -- a boundary (halo) node ships its fp32 feature row across the cut
               once per hop; this is the spatial-comm byte cost per cut edge.
        w_S = bytes_per_halo / (link_gbps * 1e9)            [seconds per cut edge]

        bytes_per_memory = FEATURE_ROW_BYTES * feat_dim * window
            -- a temporal cut transfers a vertex's window of memory state.
        w_T = min( bytes_per_memory / (link_gbps * 1e9),
                   recompute_cost * (1 - rho) )
            where rho = stats.persistence (THEORY §9.4): a high-persistence graph
            reuses temporal state losslessly so the temporal-transfer weight is
            the SMALLER of "ship it" vs "recompute the non-reusable fraction".
            recompute_cost is the per-edge aggregation time (sec_per_edge*layers)
            -- recomputing temporal state instead of shipping it costs compute.

    sec_per_edge (cost_model compute weight):
        Compute_k = AggWork / B_hbm with AggWork ~ nnz * F * layers (THEORY §2).
        Per local edge that is ``feat_dim * layers * FEATURE_ROW_BYTES`` bytes of
        aggregation traffic, so
            sec_per_edge = feat_dim * layers * FEATURE_ROW_BYTES
                           / (bottleneck_hbm_gbps * 1e9)
        -- derived from MEASURED bandwidth, not the hand-set 5e-9 default.

    Returns a ``CostCalibration`` the scheduler consumes verbatim.
    """
    cluster = probe.cluster
    link_gbps = probe.link_gbps
    link_Bps = max(link_gbps, 1e-9) * 1e9          # bytes/sec
    hbm_gbps = _bottleneck_hbm_gbps(cluster)
    hbm_Bps = max(hbm_gbps, 1e-9) * 1e9

    # --- compute weight from MEASURED bottleneck bandwidth ------------------
    sec_per_edge = (feat_dim * layers * FEATURE_ROW_BYTES) / hbm_Bps

    cost_params = CostParams(
        feat_dim=feat_dim,
        bytes_per_feat=int(FEATURE_ROW_BYTES),     # fp32, full precision
        window=window,
        sec_per_edge=sec_per_edge,
    )

    # --- spatial weight w_S -------------------------------------------------
    bytes_per_halo = FEATURE_ROW_BYTES * feat_dim * max(1, hops)
    w_S = bytes_per_halo / link_Bps

    # --- temporal weight w_T (persistence-biased) ---------------------------
    rho = 0.0
    if stats is not None and getattr(stats, "persistence", None) is not None:
        rho = float(np.clip(stats.persistence, 0.0, 1.0))
    bytes_per_memory = FEATURE_ROW_BYTES * feat_dim * max(1, window)
    ship_w_T = bytes_per_memory / link_Bps
    # recompute the NON-reusable temporal fraction (1-rho) instead of shipping.
    recompute_cost = sec_per_edge * max(1, layers)
    recompute_w_T = recompute_cost * (1.0 - rho)
    w_T = float(min(ship_w_T, recompute_w_T)) if rho > 0.0 else float(ship_w_T)

    # Provenance note.
    regime = "spatial-bound" if w_S >= w_T else "temporal-bound"
    note = (f"calibrated from {'MEASURED' if probe.measured else 'spec'} hardware; "
            f"hbm_bottleneck={hbm_gbps:.1f} GB/s, link={link_gbps:.3g} GB/s, "
            f"rho={rho:.3f} -> {regime}")

    return CostCalibration(
        cluster=cluster,
        link_gbps=link_gbps,
        graph_stats=stats,
        cost_params=cost_params,
        w_S=float(w_S),
        w_T=float(w_T),
        note=note,
    )


# --------------------------------------------------------------------------- #
# C++ graph_stats binary wrapper (the GRAPH half of the auto-probe).          #
#                                                                             #
# This is the HOT-PATH bridge: at 100M-1B edges the exact COUNTS (degree,     #
# per-snapshot active nodes, |T_v|) cannot be done in numpy without           #
# allocating ~5x M int64 (OOM/30-min hang), so they move to build/graph_stats #
# (graph_stats.cpp). The wrapper mirrors cpp_kernel.graph_bin_path EXACTLY:   #
# env override first, else <repo>/build/graph_stats; little-endian binary I/O #
# via tempfile; returns None on missing-binary / nonzero-exit so the caller   #
# falls back to the pure-numpy path in frontend.ingest (planner ALWAYS runs). #
# Torch-free: nothing here imports torch.                                     #
# --------------------------------------------------------------------------- #
def graph_stats_bin_path() -> str:
    """Resolve the graph_stats binary: $ZORD_GRAPH_STATS_BIN, else <repo>/build/graph_stats.
    Repo root is four levels up from this file (src/zord/profiler/prober.py)."""
    env = os.environ.get("ZORD_GRAPH_STATS_BIN")
    if env:
        return env
    here = os.path.abspath(__file__)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    return os.path.join(repo, "build", "graph_stats")


def have_graph_stats() -> bool:
    return os.path.exists(graph_stats_bin_path())


def graph_stats_cpp(num_nodes: int, num_snapshots: int, src: np.ndarray,
                    dst: np.ndarray, snap: np.ndarray):
    """Run build/graph_stats and return ``(deg, per_snapshot_nodes, Tv)`` int64 arrays,
    or ``None`` so the caller falls back to numpy.

    Binary protocol (little-endian, SAME prefix as supra_solver/supra_build):
      IN  : int64 N, int64 S, int64 M, int32 triples[3*M] = (src, dst, snap)
      OUT : int64 N; int32 deg[N]; int64 S; int32 per_snapshot_nodes[S];
            int64 N2; int32 Tv[N]
    The three arrays are EXACT counts; Python derives avg/max/p99 degree,
    mean/max snapshot nodes and persistence = mean((Tv[active]-1)/(S-1)) from them.
    Returns None on missing binary, nonzero exit, or a header that does not match
    the requested (N, S) (defensive: a stale binary would mis-parse)."""
    binp = graph_stats_bin_path()
    if not os.path.exists(binp):
        return None
    N = int(num_nodes)
    S = int(max(1, num_snapshots))
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    snap = np.asarray(snap, dtype=np.int64)
    M = int(src.size)
    with tempfile.TemporaryDirectory(prefix="zord_kernel_") as tmp:
        inb = os.path.join(tmp, "in.bin")
        outb = os.path.join(tmp, "out.bin")
        with open(inb, "wb") as fh:
            fh.write(struct.pack("<qqq", N, S, M))
            tri = np.empty(3 * M, dtype=np.int32)
            tri[0::3] = src.astype(np.int32)
            tri[1::3] = dst.astype(np.int32)
            tri[2::3] = snap.astype(np.int32)
            tri.tofile(fh)
        r = subprocess.run([binp, inb, outb], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        try:
            with open(outb, "rb") as fh:
                n = struct.unpack("<q", fh.read(8))[0]
                deg = np.fromfile(fh, dtype=np.int32, count=n)
                s2 = struct.unpack("<q", fh.read(8))[0]
                psn = np.fromfile(fh, dtype=np.int32, count=s2)
                n2 = struct.unpack("<q", fh.read(8))[0]
                tv = np.fromfile(fh, dtype=np.int32, count=n2)
        except Exception:
            return None
    if n != N or s2 != S or n2 != N or deg.size != N or psn.size != S or tv.size != N:
        return None
    return (deg.astype(np.int64), psn.astype(np.int64), tv.astype(np.int64))


def probe_graph_stats(graph: "TemporalGraph", *, num_snapshots: int = 64,
                      feat_bytes: Optional[np.ndarray] = None,
                      sample_edges: int = 2_000_000) -> "GraphStats":
    """AUTO-PROBE the GRAPH half: a populated ``GraphStats`` whose EXACT counts route
    through build/graph_stats (the C++ hot path) when present, falling back to the
    pure-numpy path in ``frontend.ingest.graph_stats`` when the binary is absent.

    When the binary IS present we still delegate the (possibly sampled) structural
    ratios -- clusterability (LPA-modularity proxy) and feature_homophily -- to the
    SAME numpy helpers ``ingest`` uses (they route through cpp_kernel.lpa, the proven
    pattern), and only OVERRIDE the exact-count fields (degree dist, per-snapshot
    nodes, persistence) with the C++ arrays. This keeps a single source of truth for
    the GraphStats shape (ingest.GraphStats) and guarantees the C++ and numpy paths
    produce identical numbers. Torch-free.
    """
    import time
    from ..frontend import ingest as _ingest   # lazy: keeps prober import-safe / torch-free

    fb = None if feat_bytes is None else np.asarray(feat_bytes, dtype=np.float64)
    S = int(max(1, num_snapshots))

    res = None
    if have_graph_stats():
        # build_snap sorts the graph by time IN PLACE and returns snap aligned to the
        # SORTED edge order -- so it MUST run before we read src/dst (same order as
        # ingest.graph_stats, which sorts then reads).
        snap = _ingest.build_snap(graph, num_snapshots=S)
        res = graph_stats_cpp(
            int(graph.num_nodes), S,
            np.asarray(graph.src, dtype=np.int64),
            np.asarray(graph.dst, dtype=np.int64),
            snap,
        )

    if res is None:
        # numpy fallback: the exact same GraphStats the FRONT-END computes.
        return _ingest.graph_stats(graph, num_snapshots=S, feat_bytes=fb,
                                   sample_edges=sample_edges)

    # C++ exact counts in hand -> derive every count field here (no numpy hot loop),
    # then borrow the structural-ratio helpers from ingest for the cheap proxies.
    t0 = time.perf_counter()
    deg, per_snap, tv = res
    src = np.asarray(graph.src, dtype=np.int64)
    dst = np.asarray(graph.dst, dtype=np.int64)
    N = int(graph.num_nodes)
    E = int(src.size)

    if E > 0:
        max_degree = int(deg.max())
        deg_p99 = int(np.percentile(deg, 99))
        avg_degree = float(E) / float(max(1, N))
    else:
        max_degree = 0
        deg_p99 = 0
        avg_degree = 0.0
    density = float(E) / float(max(1, N * (N - 1))) if N > 1 else 0.0

    nonempty = per_snap[per_snap > 0]
    mean_snapshot_nodes = float(nonempty.mean()) if nonempty.size else 0.0
    max_snapshot_nodes = int(per_snap.max()) if per_snap.size else 0

    # persistence rho = mean over ACTIVE vertices of (|T_v|-1)/(S-1)  (THEORY 9.4),
    # IDENTICAL to ingest._persistence but computed from the C++ |T_v| array.
    if S > 1:
        active = tv > 0
        persistence = float(((tv[active] - 1.0) / float(S - 1)).mean()) if active.any() else 0.0
    else:
        persistence = 0.0

    # structural ratios via the SAME (possibly sampled) numpy helpers ingest uses.
    if E > sample_edges and sample_edges > 0:
        rng = np.random.default_rng(0)
        sel = rng.choice(E, size=sample_edges, replace=False)
        s_src, s_dst = src[sel], dst[sel]
    else:
        s_src, s_dst = src, dst
    clusterability = _ingest._clusterability_from_lpa(N, s_src, s_dst)

    feature_homophily: Optional[float] = None
    feat_dim_mean = 0.0
    feat_dim_max = 0.0
    if fb is not None:
        if fb.size:
            feat_dim_mean = float(fb.mean())
            feat_dim_max = float(fb.max())
        feature_homophily = _ingest._feature_homophily(s_src, s_dst, fb, N)
    elif getattr(graph, "efeat", None) is not None:
        fe = graph.efeat
        feat_dim_mean = float(fe.shape[1])
        feat_dim_max = float(fe.shape[1])

    return _ingest.GraphStats(
        num_nodes=N, num_edges=E, avg_degree=avg_degree, density=density,
        max_degree=max_degree, deg_p99=deg_p99, num_snapshots=S,
        mean_snapshot_nodes=mean_snapshot_nodes, max_snapshot_nodes=max_snapshot_nodes,
        clusterability=clusterability, persistence=persistence,
        feature_homophily=feature_homophily, feat_dim_mean=feat_dim_mean,
        feat_dim_max=feat_dim_max, ingest_sec=time.perf_counter() - t0,
    )


def probe_and_calibrate(graph: Optional["TemporalGraph"] = None,
                        cluster: Optional["ClusterProfile"] = None, *,
                        measure: bool = False, device: Optional[str] = None,
                        num_snapshots: int = 64, feat_dim: int = 128,
                        feat_bytes: Optional[np.ndarray] = None,
                        window: int = 1, layers: int = 2,
                        hops: int = 2) -> CostCalibration:
    """The AUTO-PROBER that CLOSES THE LOOP: probe hardware + (optionally) the graph,
    then calibrate, so NO cost-model parameter (sec_per_edge, w_S, w_T) is hand-set.

    Composition (all the pieces already in this module):
      1. ``probe_hardware(cluster, measure=measure, device=device)`` -> the (measured
         or spec) ClusterProfile + bottleneck link bandwidth.
      2. if ``graph`` is given: ``probe_graph_stats(graph, ...)`` -> a populated
         GraphStats whose EXACT counts came from the C++ ``build/graph_stats`` binary
         (numpy fallback when the binary is absent). This provides the persistence
         ``rho`` that biases ``w_T``.
      3. ``calibrate(probe, stats, feat_dim=..., window=..., layers=..., hops=...)``
         -> a single ``CostCalibration`` the scheduler consumes verbatim.

    Torch is touched ONLY when ``measure=True`` (inside ``probe_hardware``); the
    default path is fully torch-free. Returns the self-calibrated ``CostCalibration``.
    """
    probe = probe_hardware(cluster, measure=measure, device=device)
    stats = None
    if graph is not None:
        stats = probe_graph_stats(graph, num_snapshots=num_snapshots,
                                  feat_bytes=feat_bytes)
    return calibrate(probe, stats, feat_dim=feat_dim, window=window,
                     layers=layers, hops=hops)
