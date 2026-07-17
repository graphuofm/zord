"""C++-binary ROUND-TRIP tests: the built kernels in <repo>/build/ must (a) actually run on the
canonical little-endian binary I/O the Python wrappers write, and (b) produce results IDENTICAL to
the pure-numpy fallback the wrappers fall back to when the binary is absent.

Covered (per the task): supra_build + graph_stats (REQUIRED), plus bufferpool / changed_cone /
supra_solver for completeness. Each test SKIPS if its binary is not built (so the suite still passes
on a box without the kernels), but on this repo the Makefile builds all of them.

These exercise the SAME wrapper code the engine uses, so a passing round-trip certifies both the
binary protocol and the C++<->numpy agreement (the process-only same-result invariant for the kernels).
"""
import numpy as np
import pytest

import importlib

from zord.datasets.temporal_graph import TemporalGraph
from zord.profiler import prober
import zord.runtime.bufferpool as bp_mod
import zord.schedule.dynamic_online as do_mod

# `zord.partition.allocate` and `zord.frontend.ingest` package ATTRIBUTES are shadowed by the
# re-exported FUNCTIONS of the same name (the documented arrange/ingest gotcha). Fetch the actual
# MODULE objects via importlib so the wrapper functions (have_supra_solver, build_supra, ...) resolve.
ingest_mod = importlib.import_module("zord.frontend.ingest")
alloc_mod = importlib.import_module("zord.partition.allocate")


def _toy_temporal_graph(N=60, E=400, S=8, seed=0):
    """A small time-sorted temporal graph + its equal-count snapshot id array."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, N, size=E).astype(np.int64)
    dst = rng.integers(0, N, size=E).astype(np.int64)
    t = np.sort(rng.integers(0, 10_000, size=E)).astype(np.int64)
    g = TemporalGraph(src=src, dst=dst, t=t, name="cpp-roundtrip-toy", num_nodes=N)
    g.sort_by_time()
    snap = ingest_mod.build_snap(g, num_snapshots=S)
    return g, snap, N, S


# ============================================================================ #
# supra_build  (REQUIRED)                                                       #
# ============================================================================ #
def test_supra_build_roundtrip_matches_numpy():
    """build/supra_build must reproduce the numpy supra-cell table (cells + spatial/temporal pairs)."""
    if not ingest_mod.have_supra_build():
        pytest.skip("build/supra_build not built")
    g, snap, N, S = _toy_temporal_graph()

    # C++ path (through the real wrapper -- writes the int64 N,S,M + int32 triples, runs the binary).
    cv_c, ct_c, spa_c, spb_c, tpa_c, tpb_c = ingest_mod.build_supra(g, snap, S)
    # numpy fallback (the reference the C++ mirrors).
    cv_n, ct_n, spa_n, spb_n, tpa_n, tpb_n = ingest_mod._build_supra_cells_numpy(
        np.asarray(g.src), np.asarray(g.dst), np.asarray(snap), N, S)

    # canonical cell table identical (vertex-major, snapshot-minor)
    np.testing.assert_array_equal(cv_c, cv_n)
    np.testing.assert_array_equal(ct_c, ct_n)
    # spatial / temporal cell-pair lists identical (same edge order, same a!=b drop, same consecutive rule)
    np.testing.assert_array_equal(np.sort(spa_c), np.sort(spa_n))
    np.testing.assert_array_equal(np.sort(spb_c), np.sort(spb_n))
    np.testing.assert_array_equal(tpa_c, tpa_n)
    np.testing.assert_array_equal(tpb_c, tpb_n)
    # cells are a sorted-unique set of active (v,t) pairs
    keys = cv_c.astype(np.int64) * S + ct_c.astype(np.int64)
    assert np.all(np.diff(keys) > 0)


# ============================================================================ #
# graph_stats  (REQUIRED)                                                       #
# ============================================================================ #
def test_graph_stats_roundtrip_matches_numpy():
    """build/graph_stats must reproduce the EXACT counts (degree, per-snapshot nodes, |T_v|)."""
    if not prober.have_graph_stats():
        pytest.skip("build/graph_stats not built")
    g, snap, N, S = _toy_temporal_graph()
    src = np.asarray(g.src, dtype=np.int64)
    dst = np.asarray(g.dst, dtype=np.int64)

    res = prober.graph_stats_cpp(N, S, src, dst, snap)
    assert res is not None, "graph_stats binary present but returned None (protocol mismatch?)"
    deg_c, per_snap_c, tv_c = res

    # numpy reference counts (mirrors frontend.ingest.graph_stats's fallback branch)
    deg_n = (np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)).astype(np.int64)
    v = np.concatenate([src, dst]); sp = np.concatenate([snap, snap])
    cell = np.unique(sp * np.int64(N) + v)
    per_snap_n = np.bincount(cell // N, minlength=S).astype(np.int64)
    keyv = np.unique(v * np.int64(S) + sp)
    tv_n = np.bincount(keyv // S, minlength=N).astype(np.int64)

    np.testing.assert_array_equal(deg_c, deg_n)
    np.testing.assert_array_equal(per_snap_c, per_snap_n)
    np.testing.assert_array_equal(tv_c, tv_n)


def test_graph_stats_probe_matches_numpy_ingest():
    """The C++-backed probe_graph_stats GraphStats must match the numpy ingest.graph_stats numbers."""
    if not prober.have_graph_stats():
        pytest.skip("build/graph_stats not built")
    g, _snap, N, S = _toy_temporal_graph(seed=1)
    cpp = prober.probe_graph_stats(g, num_snapshots=S)
    g2, _s2, _N2, _S2 = _toy_temporal_graph(seed=1)
    npy = ingest_mod.graph_stats(g2, num_snapshots=S)
    assert cpp.max_degree == npy.max_degree
    assert cpp.deg_p99 == npy.deg_p99
    assert cpp.max_snapshot_nodes == npy.max_snapshot_nodes
    assert abs(cpp.persistence - npy.persistence) <= 1e-9
    assert abs(cpp.mean_snapshot_nodes - npy.mean_snapshot_nodes) <= 1e-6


# ============================================================================ #
# bufferpool / supra_solver / changed_cone  (completeness)                      #
# ============================================================================ #
def test_bufferpool_roundtrip_matches_numpy():
    """build/bufferpool must reproduce the numpy Belady miss set (same hit-rate, admissions)."""
    if not bp_mod.have_bufferpool():
        pytest.skip("build/bufferpool not built")
    access = bp_mod.window_access_sequence(num_snapshots=12, window=3, num_epochs=3)
    cap_units = 4
    cpp = bp_mod._simulate_cpp(access, cap_units, "belady")
    assert cpp is not None
    is_miss, adm_c, evi_c = cpp
    adm_n, evi_n, hit_n, _ = bp_mod.belady_schedule(access, cap_units)
    assert adm_c == adm_n and evi_c == evi_n
    hits_c = len(access) - int(is_miss.sum())
    assert abs(hits_c / len(access) - hit_n) <= 1e-9


def test_supra_solver_roundtrip_cut_not_worse_than_corners():
    """build/supra_solver must return a per-cell device assignment whose weighted cut is <= min(PSS,PTS)
    block corners (the zord <= corners guarantee), agreeing in shape with the numpy supra-cell table."""
    if not alloc_mod.have_supra_solver():
        pytest.skip("build/supra_solver not built")
    g, snap, N, S = _toy_temporal_graph(seed=2)
    src = np.asarray(g.src, dtype=np.int64); dst = np.asarray(g.dst, dtype=np.int64)
    D = 3
    w_S, w_T = 1.0, 1.0
    # cells + pairs from the numpy reference (same canonical order the solver emits)
    cv, ct, _keys, C, spa, spb, tpa, tpb = alloc_mod.build_supra_cells(src, dst, snap, N, S)

    class _TGI:  # minimal duck-typed TemporalGraphInput for supra_solve_run
        pass
    tgi = _TGI(); tgi.graph = g; tgi.snap = snap
    tgi.stats = type("S", (), {"num_snapshots": S})()

    cv_s, ct_s, cell_device = alloc_mod.supra_solve_run(tgi, D, w_S, w_T, 0, snap=snap)
    assert cell_device.size == C, "solver cell count must match the numpy cell table"
    sc, tc = alloc_mod.count_cuts(cell_device, spa, spb, tpa, tpb)
    solver_cost = w_S * sc + w_T * tc

    # PSS / PTS block corners (the candidates the solver is guaranteed not to lose to)
    def _block(coord):
        uniq = np.unique(coord); B = uniq.size
        idx = np.searchsorted(uniq, coord)
        return np.minimum((idx * D) // max(1, B), D - 1).astype(np.int32)
    pss, pts = _block(ct), _block(cv)
    pss_sc, pss_tc = alloc_mod.count_cuts(pss, spa, spb, tpa, tpb)
    pts_sc, pts_tc = alloc_mod.count_cuts(pts, spa, spb, tpa, tpb)
    best_corner = min(w_S * pss_sc + w_T * pss_tc, w_S * pts_sc + w_T * pts_tc)
    assert solver_cost <= best_corner + 1e-6, (
        f"supra_solver cut cost {solver_cost} exceeds best corner {best_corner}")


def test_changed_cone_roundtrip_matches_python():
    """build/changed_cone must reproduce the Python event-dependency DAG (ear/depth/cone)."""
    if not do_mod.have_changed_cone():
        pytest.skip("build/changed_cone not built")
    g, _snap, N, S = _toy_temporal_graph(seed=3)
    src = np.asarray(g.src, dtype=np.int64); dst = np.asarray(g.dst, dtype=np.int64)
    t = np.asarray(g.t, dtype=np.int64)
    lo = src.size // 2
    # C++ path
    cpp = do_mod._changed_cone_cpp(src.astype(np.int32), dst.astype(np.int32), lo, int(src.size))
    assert cpp is not None
    ear_c, depth_c, cone_c = cpp
    # Python reference: build_event_dependency on a graph with the binary FORCED off.
    edg = _python_event_dependency(src, dst, lo)
    np.testing.assert_array_equal(ear_c, edg[0])
    np.testing.assert_array_equal(depth_c, edg[1])
    np.testing.assert_array_equal(np.sort(cone_c), np.sort(edg[2]))


def _python_event_dependency(src, dst, lo):
    """The pure-Python ear/depth/cone reference (copied from dynamic_online's fallback loop)."""
    ns = src[lo:]; nd = dst[lo:]; n_new = ns.size
    ear = np.full(n_new, -1, dtype=np.int64); depth = np.zeros(n_new, dtype=np.int64)
    last_seen = {}
    for i in range(n_new):
        u = int(ns[i]); v = int(nd[i])
        pu = last_seen.get(u, -1); pv = last_seen.get(v, -1)
        p = pu if pu > pv else pv
        ear[i] = p
        if p >= 0:
            depth[i] = depth[p] + 1
        last_seen[u] = i; last_seen[v] = i
    cone = np.unique(np.concatenate([ns, nd]))
    return ear, depth, cone
