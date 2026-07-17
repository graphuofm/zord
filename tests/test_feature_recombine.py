"""MUST-DO #1 / H2 acceptance: the feature-split FULL-LAYER recombination is RESULT-PRESERVING.

The engine's feature-parallel axis splits the F feature COLUMNS across D devices. One GNN layer is
aggregate -> W-mix -> nonlinearity; only the aggregation is column-separable, so the W-mix needs a
Megatron-style all-reduce of full-width partials and the nonlinearity is applied ONCE after the reduce.
These tests certify that the feature-split + recombine path reproduces the single-device L-layer GNN
END-TO-END within the fp32 same-result tolerance (max-abs-err <= 1e-4), for L >= 2 layers, with the
W-mix present (the trap case) and for the proven aggregation-only case.

PROCESS-only / FULL PRECISION: same data + model => same result; the split changes only WHERE columns
live and HOW partials are reduced. Pure numpy fp64 reference; torch-free.
"""
import numpy as np

from zord.runtime.feature_recombine import (
    verify_recombine, reference_llayer, sharded_llayer,
    recombine_aggregation, recombine_full_layer, split_weight_rows, plan_recombine,
)

TOL = 1e-4  # the MUST-DO #1 fp32 acceptance bound


def _toy_graph(N=40, E=180, F=16, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, N, size=E).astype(np.int64)
    dst = rng.integers(0, N, size=E).astype(np.int64)
    X = rng.standard_normal((N, F)).astype(np.float64)
    return src, dst, X, N, F


def test_full_layer_recombine_same_result_L2():
    """L=2 layers WITH the W-mix (the trap): feature-split + Megatron recombine == single device,
    max-abs-err <= 1e-4. This is the MUST-DO #1 end-to-end acceptance (aggregate->W-mix->nonlinearity)."""
    src, dst, X, N, F = _toy_graph()
    rng = np.random.default_rng(7)
    W = rng.standard_normal((F, F)).astype(np.float64)        # square W-mix (mixes ALL columns)
    splits = [7, 5, 4]                                         # 3-device column shard summing to F=16
    assert sum(splits) == F
    err, ok = verify_recombine(src, dst, X, splits, layers=2, W=W, nonlinearity="relu", tol=TOL)
    assert ok, f"full-layer recombine err {err:.2e} exceeds tol {TOL}"
    assert err <= TOL


def test_full_layer_recombine_same_result_L3_uneven_shards():
    """L=3, uneven/edge-case shards (incl. a 1-wide and a 0-wide shard) still match end-to-end."""
    src, dst, X, N, F = _toy_graph(N=55, E=260, F=12, seed=3)
    rng = np.random.default_rng(11)
    W = rng.standard_normal((F, F)).astype(np.float64)
    splits = [1, 11, 0, 0]                                     # degenerate shards must still recombine
    assert sum(splits) == F
    err, ok = verify_recombine(src, dst, X, splits, layers=3, W=W, nonlinearity="tanh", tol=TOL)
    assert ok, f"L=3 uneven-shard recombine err {err:.2e} exceeds tol {TOL}"


def test_aggregation_only_recombine_is_bit_identical():
    """W is None -> the proven aggregation-concat case: column-shard + concat is EXACT (err 0)."""
    src, dst, X, N, F = _toy_graph(seed=5)
    splits = [8, 8]
    err, ok = verify_recombine(src, dst, X, splits, layers=2, W=None, nonlinearity="relu", tol=TOL)
    assert ok and err == 0.0, f"aggregation-only concat should be exact, got err {err:.2e}"


def test_recombine_matches_sharded_vs_reference_directly():
    """Directly compare the two L-layer code paths (reference vs sharded) tensor-for-tensor."""
    src, dst, X, N, F = _toy_graph(N=30, E=120, F=10, seed=2)
    rng = np.random.default_rng(1)
    W = rng.standard_normal((F, F)).astype(np.float64)
    splits = [4, 3, 3]
    ref = reference_llayer(src, dst, X, W, layers=2, nonlinearity="relu")
    got = sharded_llayer(src, dst, X, splits, W, layers=2, nonlinearity="relu")
    assert ref.shape == got.shape == (N, F)
    assert float(np.abs(ref - got).max()) <= TOL


def test_full_layer_partial_form_A_all_reduce():
    """recombine_full_layer form (A): summing the already-full-width per-device partials + phi ONCE
    equals applying phi to the single-device h @ W. The all-reduce is order-independent up to fp add."""
    src, dst, X, N, F = _toy_graph(N=24, E=90, F=9, seed=4)
    rng = np.random.default_rng(9)
    W = rng.standard_normal((F, F)).astype(np.float64)
    splits = [3, 3, 3]
    # single-device: h = A@X ; z = h@W ; y = relu(z)
    from zord.runtime.feature_recombine import _agg_one_layer, _apply_nonlinearity
    h = _agg_one_layer(src, dst, X)
    y_ref = _apply_nonlinearity(h @ W, "relu")
    # sharded: per device partial p_d = (A @ X[:,cols_d]) @ W[cols_d,:], all-reduce, relu once
    Wsh = split_weight_rows(W, splits)
    offs = np.cumsum([0] + list(splits))
    partials = [_agg_one_layer(src, dst, X[:, offs[d]:offs[d + 1]]) @ Wsh[d] for d in range(len(splits))]
    y = recombine_full_layer(partials, splits, W_shards=None, nonlinearity="relu")
    assert float(np.abs(y - y_ref).max()) <= TOL


def test_plan_recombine_cost_is_well_formed():
    """plan_recombine costs the column-shard integration: full-layer moves more bytes than agg-only,
    and the bytes scale linearly with L. PROCESS cost only (no result change)."""
    cols = np.array([6, 5, 5], dtype=np.int64)               # F=16, D=3
    F, N, link = 16, 100_000, 50.0
    agg = plan_recombine(cols, F, N, layers=2, link_gbps=link, full_layer=False)
    full = plan_recombine(cols, F, N, layers=2, link_gbps=link, full_layer=True)
    assert full.recombine_bytes > agg.recombine_bytes      # the W-mix all-reduce adds bytes
    full_4 = plan_recombine(cols, F, N, layers=4, link_gbps=link, full_layer=True)
    assert abs(full_4.recombine_bytes - 2 * full.recombine_bytes) <= 4   # linear in L
    assert full.recombine_ms > 0.0
