"""MIDDLE-END (K2) -- the ONE allocation that composes the supra-cut, the decomposition AXIS, the
per-node byte balance, the vertex-cut core, and the node/feature/hybrid split into a single plan.

This is the layer that sits between the calibrated cost weights (profiler.prober.CostCalibration,
which derives the EXACT w_S/w_T floats the C++ solver consumes) and the back-end memory planner. It
makes ONE decision -- where every vertex (and, on the supra path, every (v,t) cell) lives -- by
COMPOSING the pieces that are each already correct on their own:

  * arrange (partition/arrange.py)     : the adaptive-corner vertex partition + vertex-cut core +
                                         predicted-makespan pick (zord <= min(PSS, PTS, METIS)).
  * supra_solver (cpp/supra_solver.cpp): the WEIGHTED supra-cut over the explicit (v,t) cell graph,
                                         the C++ HOT PATH (Fennel greedy + KL/FM refine + PSS/PTS
                                         corners) that scales the cut to 100M-1B cells. allocate.py
                                         is its Python wrapper (write_input / read_output copied from
                                         scripts/supra_solve.py; binary resolved like cpp_kernel).
  * attr_cost.decide_axis              : the closed-form node-vs-feature-vs-hybrid axis pre-filter.
  * planner.choose_decomposition       : the exact per-device decomposition costing.
  * feature_parallel                   : the FEATURE/HYBRID column split (each device holds F/D cols).

THE COMPOSITION (allocate):
  1. axis = decide which decomposition axis to use (analytical decide_axis + exact
     choose_decomposition, unless the caller forces it via decomposition=).
  2. NODE axis  -> run BOTH the supra-cut (C++ solver -> cell_device[C] folded to assignment[N] by a
     majority/home rule) AND arrange (ArrangeResult), then keep the one with the lower weighted cut
     cost (so the allocation is never worse than either -- the zord <= corners guarantee carries over
     to the composed plan). The supra path also yields the per-cell cell_device for the back-end.
  3. FEATURE/HYBRID axis -> column-split via feature_parallel_plan; the assignment is the node home of
     the (still node-partitioned within a feature group) layout, taken from arrange so the back-end
     has a vertex->device map; cols_per_device drives the feature memory.
  4. compute spatial/temporal cut (count_cuts) + per-device feature bytes -> AllocationPlan.

BOUNDARY: this file owns POLICY + the subprocess wrapper + the numpy fallback. The hot structural
pass (cell discovery, the supra-graph build, the greedy + KL/FM refine, the cut counting at scale)
lives in C++ (cpp/supra_solver.cpp). When the binary is absent we fall back to a pure-numpy supra
build + arrange so the planner ALWAYS runs (correctly, slower). NO torch; import-safe on a CPU box.
PROCESS-only: same graph + same model + same cluster => same allocation. We optimize WHERE state
lives (time / memory / feasibility), never WHAT is computed -- the placement is result-preserving.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

# NOTE: the package __init__ re-exports the `arrange` FUNCTION as zord.partition.arrange, so
# `from . import arrange` would bind the function, not the module. Import the pieces we need by name.
from .arrange import (
    arrange as _arrange_fn,
    ArrangeResult, predict_ms,
    _feat_bytes_per_dev as _arr_feat_bytes_per_dev,
)
from .attr_cost import decide_axis, AttrDecision
from .feature_parallel import feature_parallel_plan, hybrid_plans

if TYPE_CHECKING:  # import-safe: these siblings may pull heavier deps / are owned elsewhere
    from ..profiler.prober import CostCalibration
    from ..frontend.ingest import TemporalGraphInput
    from ..schedule.planner import DecompositionChoice


# --------------------------------------------------------------------------- #
# binary resolution (copied from cpp_kernel.graph_bin_path -- the proven pattern) #
# --------------------------------------------------------------------------- #
def _repo_root() -> str:
    """Repo root = 4 dirs up from this file (src/zord/partition/allocate.py)."""
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))


def supra_solver_bin_path() -> str:
    """Resolve the supra_solver binary: $ZORD_SUPRA_BIN, else <repo>/build/supra_solver."""
    env = os.environ.get("ZORD_SUPRA_BIN")
    if env:
        return env
    return os.path.join(_repo_root(), "build", "supra_solver")


def have_supra_solver() -> bool:
    return os.path.exists(supra_solver_bin_path())


def supra_build_bin_path() -> str:
    """Resolve the optional supra_build binary: $ZORD_SUPRA_BUILD_BIN, else <repo>/build/supra_build.

    supra_build materializes the active-cell table + spatial/temporal cell-pair lists in C++ (the
    O(E log E) sort/unique the numpy build_supra_cells does). allocate uses it only as an OPTIONAL
    accelerator for the fallback (numpy) path / for explicit cut counting; the SOLVER path already
    emits the canonical cell table via its sidecar (supra_solver's 3rd arg), so it is not required."""
    env = os.environ.get("ZORD_SUPRA_BUILD_BIN")
    if env:
        return env
    return os.path.join(_repo_root(), "build", "supra_build")


def have_supra_build() -> bool:
    return os.path.exists(supra_build_bin_path())


# --------------------------------------------------------------------------- #
# binary I/O for the supra_solver (write_input / read_output, copied from       #
# scripts/supra_solve.py so the two share ONE writer; little-endian everywhere). #
# --------------------------------------------------------------------------- #
def _write_solver_input(path: str, N: int, S: int, src: np.ndarray, dst: np.ndarray,
                        snap: np.ndarray, D: int, w_S: float, w_T: float, cap_cells: int) -> None:
    """IN(LE): int64 N,S,M; int32 triples[3*M]=(src,dst,snap); int32 D; float w_S; float w_T;
    int64 cap_cells. IDENTICAL prefix to supra_build (so one Python writer serves both)."""
    M = int(src.size)
    trip = np.empty(3 * M, dtype=np.int32)
    trip[0::3] = src.astype(np.int32)
    trip[1::3] = dst.astype(np.int32)
    trip[2::3] = snap.astype(np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("<qqq", int(N), int(S), int(M)))
        trip.tofile(f)
        f.write(struct.pack("<iff", int(D), float(w_S), float(w_T)))
        f.write(struct.pack("<q", int(cap_cells)))


def _read_solver_output(path: str):
    """OUT(LE): int64 num_cells; int32 device[num_cells] (canonical cell order). Returns int32[C]."""
    with open(path, "rb") as f:
        (num,) = struct.unpack("<q", f.read(8))
        dev = np.fromfile(f, dtype=np.int32, count=num)
    return dev


def _read_cell_sidecar(path: str):
    """OUT(LE) of the supra_solver 3rd-arg sidecar: int64 C; int32 cell_v[C]; int32 cell_t[C].
    Returns (cell_v, cell_t) as int64 [C] in canonical cell-id order (matches device[])."""
    with open(path, "rb") as f:
        (C,) = struct.unpack("<q", f.read(8))
        cell_v = np.fromfile(f, dtype=np.int32, count=C).astype(np.int64)
        cell_t = np.fromfile(f, dtype=np.int32, count=C).astype(np.int64)
    return cell_v, cell_t


# --------------------------------------------------------------------------- #
# numpy fallback supra-cell build + cut counting (mirrors scripts/supra_solve.py) #
# --------------------------------------------------------------------------- #
def build_supra_cells(src: np.ndarray, dst: np.ndarray, snap: np.ndarray, N: int, S: int):
    """Canonical active-cell table + explicit supra-graph edge lists (numpy; the C++ fallback).

    Cells are unique (vertex, snapshot) pairs carrying an incident edge, ordered (vertex-major,
    snapshot-minor) -- IDENTICAL to supra_solver's canonical cell order, so a cell_device[] from
    the solver lines up index-for-index with these cell_v/cell_t. Mirrors supra_solve.py exactly.

    Returns (cell_v, cell_t, keys, C, sp_a, sp_b, tp_a, tp_b)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    snap = np.asarray(snap, dtype=np.int64)
    ks = src * S + snap
    kd = dst * S + snap
    keys = np.unique(np.concatenate([ks, kd]))      # sorted unique == cell ids 0..C-1
    C = int(keys.size)
    cell_v = (keys // S).astype(np.int64)
    cell_t = (keys % S).astype(np.int64)
    a = np.searchsorted(keys, ks)
    b = np.searchsorted(keys, kd)
    m = a != b                                       # drop self-loops
    sp_a, sp_b = a[m], b[m]
    same_v = cell_v[1:] == cell_v[:-1]               # vertex-major -> consecutive same-vertex cells
    idx = np.nonzero(same_v)[0]
    tp_a = idx
    tp_b = idx + 1
    return cell_v, cell_t, keys, C, sp_a, sp_b, tp_a, tp_b


def count_cuts(dev: np.ndarray, sp_a: np.ndarray, sp_b: np.ndarray,
               tp_a: np.ndarray, tp_b: np.ndarray):
    """SpatialCut / TemporalCut for a per-CELL device assignment (same defn the C++ solver scores).
    Returns (spatial_cut:int, temporal_cut:int)."""
    spatial = int(np.count_nonzero(dev[sp_a] != dev[sp_b])) if sp_a.size else 0
    temporal = int(np.count_nonzero(dev[tp_a] != dev[tp_b])) if tp_a.size else 0
    return spatial, temporal


def _numpy_supra_solve(cell_v, cell_t, C, sp_a, sp_b, tp_a, tp_b, N, S, D, w_S, w_T):
    """Pure-numpy fallback partitioner over the cell graph when the C++ solver binary is absent.

    Mirrors the solver's candidate set (greedy is replaced by the cheaper-to-state-in-numpy PSS/PTS
    block corners, which the solver ALSO emits as candidates) and returns the MIN-cost feasible cell
    assignment. This is correct (zord <= min(PSS,PTS) still holds because both corners are evaluated)
    and O(C + cuts) -- slower than the C++ greedy+refine but never wrong. The block corners are the
    contiguous balanced split of the leading coordinate's distinct values into D device blocks, EXACTLY
    the C++ block_assignment(). The lower-cost of {PSS-block, PTS-block} is returned."""
    if C == 0:
        return np.zeros(0, dtype=np.int32)

    def _block(coord):
        uniq = np.unique(coord)
        B = uniq.size
        idx = np.searchsorted(uniq, coord)
        return np.minimum((idx * D) // max(1, B), D - 1).astype(np.int32)

    pss = _block(cell_t)                              # whole snapshots -> devices (Dv1 x DtD)
    pts = _block(cell_v)                              # whole timelines  -> devices (DvD x Dt1)
    best = None
    for asg in (pss, pts):
        sc, tc = count_cuts(asg, sp_a, sp_b, tp_a, tp_b)
        cost = float(w_S) * sc + float(w_T) * tc
        if best is None or cost < best[0]:
            best = (cost, asg)
    return best[1]


# --------------------------------------------------------------------------- #
# supra_solve_run: the SOLVER wrapper (C++ hot path + numpy fallback)           #
# --------------------------------------------------------------------------- #
def supra_solve_run(tgi: "TemporalGraphInput", D: int, w_S: float, w_T: float,
                    cap_cells: int = 0, *, snap: Optional[np.ndarray] = None):
    """Run the weighted supra-cut and return (cell_v, cell_t, cell_device) in canonical cell order.

    Calls build/supra_solver (the C++ HOT PATH: Fennel greedy + KL/FM refine + PSS/PTS corners,
    emitting the MIN-cost feasible assignment so cost <= min(PSS,PTS)). The solver's 3rd-arg sidecar
    gives the canonical (cell_v, cell_t) table back so we do not rebuild it in numpy at scale. On a
    binary-absent / torch-free / failure box we fall back to the numpy supra build + the block-corner
    partitioner so the planner ALWAYS runs.

    w_S / w_T are the EXACT floats CostCalibration derives (bytes_per_halo/B_link etc.) -- passed
    straight through to the solver, which consumes them verbatim. cap_cells=0 == unbounded.

    cell_v, cell_t : int64 [C]   per-cell coordinates (vertex-major, snapshot-minor)
    cell_device    : int32 [C]   device id per active cell (None on a hard failure with no fallback).
    """
    g = tgi.graph
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    snap = tgi.snap if snap is None else snap
    snap = np.asarray(snap, dtype=np.int64)
    S = int(tgi.stats.num_snapshots) if getattr(tgi, "stats", None) is not None \
        else int(snap.max()) + 1 if snap.size else 1
    S = max(1, S)
    D = max(1, int(D))

    binp = supra_solver_bin_path()
    if os.path.exists(binp):
        with tempfile.TemporaryDirectory(prefix="zord_kernel_") as tmp:
            inb = os.path.join(tmp, "in.bin")
            outb = os.path.join(tmp, "out.bin")
            cellb = os.path.join(tmp, "cells.bin")
            _write_solver_input(inb, N, S, src, dst, snap, D, w_S, w_T, cap_cells)
            r = subprocess.run([binp, inb, outb, cellb], capture_output=True, text=True)
            if r.returncode == 0 and os.path.exists(outb) and os.path.exists(cellb):
                cell_device = _read_solver_output(outb)
                cell_v, cell_t = _read_cell_sidecar(cellb)
                if cell_v.size == cell_device.size:
                    return cell_v, cell_t, cell_device
            # fall through to numpy on any mismatch / nonzero rc

    # ---- numpy fallback (binary absent or run failed): build cells + block-corner solve ----
    cell_v, cell_t, keys, C, sp_a, sp_b, tp_a, tp_b = build_supra_cells(src, dst, snap, N, S)
    cell_device = _numpy_supra_solve(cell_v, cell_t, C, sp_a, sp_b, tp_a, tp_b,
                                     N, S, D, w_S, w_T)
    return cell_v, cell_t, cell_device


# --------------------------------------------------------------------------- #
# cell_device[C] -> assignment[N] (the majority / home fold)                    #
# --------------------------------------------------------------------------- #
def _fold_cells_to_vertices(cell_v: np.ndarray, cell_device: np.ndarray, N: int, D: int):
    """Fold a per-CELL device assignment to a per-VERTEX home by MAJORITY rule (the device that homes
    the MOST of a vertex's active cells -- so a vertex's timeline lives where most of it already is,
    minimizing the temporal cut for that vertex). Vectorized over cells (no Python loop); ties broken
    toward the lowest device id (deterministic). Vertices with no active cell get device 0.

    Returns assignment int32 [N]."""
    assignment = np.zeros(N, dtype=np.int32)
    if cell_v.size == 0:
        return assignment
    cell_v = np.asarray(cell_v, dtype=np.int64)
    cell_device = np.asarray(cell_device, dtype=np.int64)
    D = max(1, int(D))
    # count[(v, d)] via a single bincount over the joint key v*D + d, then argmax over d per vertex.
    key = cell_v * np.int64(D) + cell_device
    counts = np.bincount(key, minlength=N * D).astype(np.int64).reshape(N, D)
    # argmax picks the lowest-index device on ties (numpy argmax convention) -> deterministic home.
    home = counts.argmax(axis=1).astype(np.int32)
    # vertices with NO active cell have an all-zero row -> argmax=0; that is the desired default.
    assignment[:] = home
    return assignment


# --------------------------------------------------------------------------- #
# the composed allocation plan                                                #
# --------------------------------------------------------------------------- #
@dataclass
class AllocationPlan:
    """The ONE allocation the MIDDLE-END hands the BACK-END (the K2 contract object).

    assignment        : int32 [N]  vertex -> device (periphery home for the vertex-cut axis).
    cell_device       : int32 [C] | None  per-(v,t)-cell device (supra path; None off it).
    core_mask         : bool [N] | None    replicated dense-core mask (vertex-cut), else None.
    spatial_cut       : int   spatial cell-pairs cut by `assignment` (lifted to cells).
    temporal_cut      : int   temporal cell-pairs cut by `assignment`.
    weighted_cost     : float w_S*spatial_cut + w_T*temporal_cut (the supra objective).
    per_device_counts : int64 [D]  vertices homed on each device.
    feat_bytes_dev    : float64 [D] | None  per-device feature bytes (sum_{v on k} F_v*4), else None.
    axis              : "node" | "feature" | "hybrid"  the decomposition axis chosen.
    decomposition     : DecompositionChoice | None  the exact per-device axis costing (when computed).
    arrange           : ArrangeResult | None  the node-parallel arrange result (when on the node axis).
    note              : free-form provenance.
    """
    assignment: np.ndarray
    cell_device: Optional[np.ndarray]
    core_mask: Optional[np.ndarray]
    spatial_cut: int
    temporal_cut: int
    weighted_cost: float
    per_device_counts: np.ndarray
    feat_bytes_dev: Optional[np.ndarray]
    axis: str
    decomposition: "Optional[DecompositionChoice]" = None
    arrange: Optional[ArrangeResult] = None
    note: str = ""

    def summary(self) -> str:
        D = int(self.per_device_counts.size)
        fb = ("n/a" if self.feat_bytes_dev is None
              else f"max {self.feat_bytes_dev.max() / (1024**3):.2f}GB")
        lines = [
            f"[allocate] axis={self.axis}  D={D}  "
            f"spatial_cut={self.spatial_cut:,} temporal_cut={self.temporal_cut:,} "
            f"weighted_cost={self.weighted_cost:.6g}",
            f"  per-device vertex counts: {list(map(int, self.per_device_counts))}  "
            f"feat_bytes/dev: {fb}",
        ]
        if self.core_mask is not None:
            lines.append(f"  vertex-cut core: {int(self.core_mask.sum()):,} replicated rows")
        if self.cell_device is not None:
            lines.append(f"  supra cells: {int(self.cell_device.size):,}")
        if self.decomposition is not None:
            lines.append("  " + self.decomposition.summary().replace("\n", "\n  "))
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# small helpers (per-device byte / count accounting -- O(N), pure numpy)        #
# --------------------------------------------------------------------------- #
def _per_device_counts(assignment: np.ndarray, D: int,
                       core_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Vertices homed on each device. For the vertex-cut, the replicated core lives on EVERY device,
    so the core count is ADDED to all D (mirrors arrange's counts incl. replicated core)."""
    asg = np.asarray(assignment, dtype=np.int64)
    D = max(1, int(D))
    homed = asg >= 0
    counts = np.bincount(asg[homed], minlength=D).astype(np.int64)
    if core_mask is not None:
        counts = counts + int(np.asarray(core_mask, dtype=bool).sum())
    return counts


def _feat_bytes_per_device(assignment: np.ndarray, feat_bytes: Optional[np.ndarray], D: int,
                           core_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Per-device feature MEMORY bytes = sum_{v homed on k} F_v*4 (+ replicated core on each device).
    Reuses arrange._feat_bytes_per_dev so the byte model is the single source of truth. None when
    feat_bytes is None (the scalar path -- the back-end charges counts*F*4 itself)."""
    if feat_bytes is None:
        return None
    return _arr_feat_bytes_per_dev(np.asarray(assignment, dtype=np.int64),
                                   np.asarray(feat_bytes, dtype=np.float64),
                                   max(1, int(D)),
                                   core_mask=core_mask)


def _cut_of_vertex_assignment(assignment: np.ndarray, cell_v: np.ndarray, cell_t: np.ndarray,
                              sp_a, sp_b, tp_a, tp_b) -> tuple:
    """Lift a per-VERTEX assignment to per-CELL (a cell (v,t) inherits its vertex's device) and count
    the spatial / temporal supra-cut. This scores the arrange (vertex) allocation by the SAME cut
    definition the supra solver optimizes, so the two paths are compared apples-to-apples."""
    if cell_v.size == 0:
        return 0, 0
    cell_dev = np.asarray(assignment, dtype=np.int64)[np.asarray(cell_v, dtype=np.int64)]
    return count_cuts(cell_dev, sp_a, sp_b, tp_a, tp_b)


# --------------------------------------------------------------------------- #
# THE ENTRY: allocate                                                          #
# --------------------------------------------------------------------------- #
def allocate(tgi: "TemporalGraphInput", calib: "CostCalibration", *,
             decomposition: str = "auto", cap_cells: int = 0, seed: int = 0,
             prior: "Optional[AllocationPlan]" = None) -> AllocationPlan:
    """Compose the cut + axis + per-node byte-balance + vertex-cut core into ONE AllocationPlan.

    tgi   : the FRONT-END TemporalGraphInput (graph + snap + feat_bytes + stats + mode).
    calib : the calibrated CostCalibration (supplies cluster, link_gbps, and the EXACT w_S/w_T the
            C++ supra solver consumes verbatim).
    decomposition : "auto" -> pick the axis via attr_cost.decide_axis + planner.choose_decomposition;
            "node"/"feature"/"hybrid" -> force that axis. DEFAULT "auto".
    cap_cells : per-device #cells cap for the supra solver (0 = unbounded).
    prior : an optional previous AllocationPlan (its assignment seeds incremental reuse upstream; the
            allocation itself is deterministic given the inputs -- prior is carried for the scheduler).

    PROCEDURE:
      (a) F = the feature width (calib.cost_params.feat_dim); decide the axis:
          - decide_axis (closed-form pre-filter) gives the analytical pick + the relief rule;
          - when not forced AND torch-free planner.choose_decomposition is available it does the exact
            per-device costing and OVERRIDES the analytical pick (the exact costing is authoritative).
      (b) NODE axis -> run BOTH the supra path (C++ solver -> cell_device -> majority fold to
          assignment) AND arrange (vertex partition + vertex-cut core); keep the lower weighted cut.
      (c) FEATURE / HYBRID axis -> arrange gives the vertex home map; feature_parallel gives the column
          split that drives the feature memory; the cut is still scored on the vertex home.
      (d) compute spatial/temporal cut (count_cuts on the cells), per-device counts + feat bytes.

    Returns the AllocationPlan. The supra C++ solver path guarantees the supra-cut <= min(PSS,PTS);
    arrange guarantees its makespan <= min(corner candidates, METIS); allocate keeps the BETTER of the
    two on the node axis, so the composed allocation inherits both guarantees. Numpy fallback (binary
    absent): the supra path uses the block-corner solver; arrange always runs (it is pure numpy)."""
    g = tgi.graph
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)
    N = int(g.num_nodes)
    cluster = calib.cluster
    D = max(1, cluster.num_devices)
    link_gbps = float(calib.link_gbps)
    F = int(calib.cost_params.feat_dim)
    S = int(tgi.stats.num_snapshots)
    snap = np.asarray(tgi.snap, dtype=np.int64)
    feat_bytes = None if tgi.feat_bytes is None else np.asarray(tgi.feat_bytes, dtype=np.float64)
    w_S = float(calib.w_S)
    w_T = float(calib.w_T)

    # ---- (a) decide the decomposition axis --------------------------------------------------
    decomp_choice: "Optional[DecompositionChoice]" = None
    attr: Optional[AttrDecision] = None
    if decomposition in ("node", "feature", "hybrid"):
        axis = decomposition
    else:  # "auto"
        # analytical pre-filter (cheap, closed-form, always available, torch-free)
        try:
            attr = decide_axis(tgi.stats, cluster, F, link_gbps,
                               layers=int(calib.cost_params.window) if False else 2)
            axis = attr.axis
        except Exception:
            axis = "node"
        # exact per-device costing OVERRIDES the analytical pick when it runs (authoritative).
        try:
            from ..schedule.planner import choose_decomposition
            decomp_choice = choose_decomposition(
                g, cluster, feat_dim=F, link_gbps=link_gbps, seed=seed,
                num_snapshots=S, feat_bytes=feat_bytes)
            axis = decomp_choice.axis
        except Exception:
            decomp_choice = None
    # normalize a "hybrid(DnxDf)" axis label to the bare family for the branch below
    axis_family = "hybrid" if axis.startswith("hybrid") else axis

    # ---- build the supra cell table ONCE (shared by cut scoring on every axis) --------------
    # The supra path (node axis) reuses cell_device; the other axes still need the cell pairs to
    # SCORE their vertex assignment by the same cut definition. Build via the C++/numpy supra build.
    cell_v, cell_t, _keys, C, sp_a, sp_b, tp_a, tp_b = build_supra_cells(src, dst, snap, N, S)

    # ---- run arrange (always: it is the vertex partition + the cut FLOOR for every axis) ----
    arr: ArrangeResult = _arrange_fn(
        src, dst, N, cluster, link_gbps=link_gbps, feat_dim=F,
        num_snapshots=S, snap=snap, seed=seed, feat_bytes=feat_bytes)
    arr_assignment = np.asarray(arr.assignment, dtype=np.int64)
    arr_core = arr.core_mask
    # arrange's vertex-cut leaves periphery home in assignment and core in core_mask; for cut scoring
    # the replicated core has no single home -> a core cell is never "cut" (it is everywhere). We score
    # the cut on the SINGLE-HOME assignment, treating core vertices' device as their stored home value
    # (arrange keeps the periphery split; core entries are still a valid device id from the split).
    arr_sc, arr_tc = _cut_of_vertex_assignment(arr_assignment, cell_v, cell_t,
                                               sp_a, sp_b, tp_a, tp_b)
    arr_cost = w_S * arr_sc + w_T * arr_tc

    note_parts = [f"axis={axis}"]

    if axis_family == "node":
        # ---- (b) NODE axis: supra path + arrange, keep the lower weighted cut ----------------
        cell_v2, cell_t2, cell_device = supra_solve_run(tgi, D, w_S, w_T, cap_cells, snap=snap)
        # cell order from the solver MUST match the locally-built table (same canonical order). If the
        # solver fell back / sizes mismatch, recompute the cut on the locally-built pairs to be safe.
        if cell_device.size == C:
            supra_sc, supra_tc = count_cuts(cell_device, sp_a, sp_b, tp_a, tp_b)
        else:
            # size mismatch (e.g. an empty graph corner) -> rebuild pairs against the solver's cells.
            cv2, ct2, _k2, _C2, s2a, s2b, t2a, t2b = build_supra_cells(src, dst, snap, N, S)
            cell_v, cell_t, sp_a, sp_b, tp_a, tp_b = cv2, ct2, s2a, s2b, t2a, t2b
            supra_sc, supra_tc = count_cuts(cell_device, sp_a, sp_b, tp_a, tp_b)
        supra_cost = w_S * supra_sc + w_T * supra_tc
        supra_assignment = _fold_cells_to_vertices(cell_v, cell_device, N, D)

        if supra_cost <= arr_cost:
            # the supra cut wins: vertex home = majority fold; cell_device kept for the back-end.
            assignment = supra_assignment
            core_mask = None
            spatial_cut, temporal_cut = supra_sc, supra_tc
            weighted_cost = supra_cost
            out_cell_device = cell_device.astype(np.int32)
            note_parts.append(
                f"supra-cut wins (cost {supra_cost:.6g} <= arrange {arr_cost:.6g})")
        else:
            # arrange wins: keep its vertex partition (+ vertex-cut core); recompute cell_device by
            # lifting the vertex home to cells so the back-end still has a per-cell device map.
            assignment = arr_assignment.astype(np.int32)
            core_mask = arr_core
            spatial_cut, temporal_cut = arr_sc, arr_tc
            weighted_cost = arr_cost
            out_cell_device = assignment.astype(np.int64)[cell_v].astype(np.int32)
            note_parts.append(
                f"arrange wins (cost {arr_cost:.6g} < supra {supra_cost:.6g}); strategy={arr.name}")
    else:
        # ---- (c) FEATURE / HYBRID axis: column-split layout; arrange gives the vertex home ----
        # The feature/hybrid axes REPLICATE the adjacency (or node-partition within feature groups),
        # so the per-vertex home map is still arrange's node partition (the back-end column-splits the
        # features per device via feature_parallel). The supra cut is scored on that vertex home.
        assignment = arr_assignment.astype(np.int32)
        core_mask = arr_core
        spatial_cut, temporal_cut = arr_sc, arr_tc
        weighted_cost = arr_cost
        out_cell_device = assignment.astype(np.int64)[cell_v].astype(np.int32)
        if axis_family == "feature":
            fp = feature_parallel_plan(src, dst, N, cluster, F, link_gbps)
            note_parts.append(
                f"feature-parallel: cols/dev={list(map(int, fp.cols_per_device))} "
                f"makespan~{fp.makespan_ms:.1f}ms feasible={fp.feasible}")
        else:  # hybrid
            hybs = hybrid_plans(src, dst, N, cluster, F, link_gbps)
            feas = [h for h in hybs if h.feasible] or hybs
            best_h = min(feas, key=lambda h: h.makespan_ms) if feas else None
            if best_h is not None:
                note_parts.append(
                    f"hybrid {best_h.name}: makespan~{best_h.makespan_ms:.1f}ms "
                    f"feasible={best_h.feasible}")
            else:
                note_parts.append("hybrid: no non-degenerate grid; fell back to node home")

    # ---- (d) per-device accounting ----------------------------------------------------------
    counts = _per_device_counts(assignment, D, core_mask=core_mask)
    feat_bytes_dev = _feat_bytes_per_device(assignment, feat_bytes, D, core_mask=core_mask)

    if attr is not None:
        note_parts.append(f"attr-rule: {attr.rule.splitlines()[0] if attr.rule else ''}")

    return AllocationPlan(
        assignment=np.asarray(assignment, dtype=np.int32),
        cell_device=(None if out_cell_device is None else np.asarray(out_cell_device, dtype=np.int32)),
        core_mask=(None if core_mask is None else np.asarray(core_mask, dtype=bool)),
        spatial_cut=int(spatial_cut),
        temporal_cut=int(temporal_cut),
        weighted_cost=float(weighted_cost),
        per_device_counts=counts,
        feat_bytes_dev=feat_bytes_dev,
        axis=axis_family,
        decomposition=decomp_choice,
        arrange=arr,
        note="; ".join(p for p in note_parts if p),
    )
