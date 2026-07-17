"""FEATURE-SPLIT FULL-LAYER recombination (Megatron-style) -- the MUST-DO #1 closure (H2).

WHAT THIS CLOSES (docs/MUST_DO.md #1 + BACKLOG H2):
  zord's feature-parallel axis (zord.partition.feature_parallel) splits the F feature COLUMNS
  across D devices. fp_aggregate_consistency() ALREADY proves that the AGGREGATION step
  h_v[c] = sum_{u in N(v)} x_u[c] is column-separable -> column-shard + CONCAT == single device,
  bit-identical (max abs diff = 0). BUT one GNN layer is THREE operations, and only the first is
  column-separable:

    1. AGGREGATE  h = A @ X         -- per-column independent -> CONCAT recombine is exact.        OK
    2. W-MIX      z = h @ W         -- output column j depends on ALL input columns (W mixes them)
                                       -> NOT column-separable. Needs a PARTIAL-SUM / all-reduce.  TRAP
    3. NONLINEAR  y = phi(z)        -- elementwise, but must see the FULL mixed row z, so it can
                                       only be applied AFTER the recombine.                         TRAP

  THE MEGATRON-STYLE PATTERN (column/row-parallel linear) that keeps the WHOLE layer result-preserving:

    * X is column-sharded:    X = [ X^(0) | X^(1) | ... | X^(D-1) ]  with X^(d) = X[:, cols_d].
    * Aggregation is local:   h^(d) = A @ X^(d)  (each device runs the FULL-graph SpMM on its slice;
                              the adjacency is replicated, no partial sum crosses a device DURING agg).
    * W is ROW-sharded to match the column shards:  W = [ W^(0) ; W^(1) ; ... ]  (W^(d) = W[cols_d, :]),
                              because the matmul  h @ W = sum_d  h^(d) @ W^(d)  contracts over the input
                              feature axis -- exactly the axis that is sharded. So each device computes a
                              FULL-WIDTH partial   p^(d) = h^(d) @ W^(d)   over its own columns, and the
                              layer output is the SUM (all-reduce) of the partials:
                                   z = sum_d p^(d)
    * The NONLINEARITY is applied ONCE, AFTER the reduce:  y = phi(z).  y is then RE-SHARDED by columns
      (split y's columns the SAME way) to feed the NEXT layer's aggregation -> repeat for L layers.

  WHERE a column lives / WHEN a partial is reduced never changes WHAT is computed:
       sum_d ( (A @ X[:, cols_d]) @ W[cols_d, :] )  ==  (A @ X) @ W   (associativity + the contraction
  being over the sharded axis). The ONLY numerical difference vs a single device is fp ADD ORDER in the
  cross-device reduce, which is <= 1e-4 in fp32 and 0 in fp64 -- this is the same-result certificate.

PROCESS-only / FULL PRECISION (SACRED): no FP16 / BF16 / TF32 / any precision reduction. The numpy
reference path is fp64 (exact); the optional torch path uses fp32 with TF32 disabled via
zord.runtime.memtier.enforce_full_precision(). This module changes only WHERE feature columns live and
HOW the per-device partials are reduced -- never the algorithm's result.

IMPORT-SAFETY: torch is OPTIONAL and imported lazily. `import zord.runtime.feature_recombine` and the
whole numpy CPU-sim + verify_recombine() certificate run on a CPU box with NO torch / NO CUDA. The torch
path (recombine_full_layer_torch / verify_recombine on a torch device) is used only when torch is present.

NEVER networkx; the adjacency is consumed as edge lists (src/dst) exactly like the rest of the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Reuse the EXISTING proven aggregation primitive and the numpy reference aggregator.
from ..partition.feature_parallel import fp_aggregate_consistency
from .memtier import Aggregator

# fp32 word; matches arrange.FEATURE_ROW_BYTES / feature_parallel so recombine cost is apples-to-apples.
FEATURE_ROW_BYTES: float = 4.0
N_GATHERS: int = 2                # 2 SpMM gathers per aggregation layer (shared roofline constant)

# fp32 same-result tolerance (the MUST-DO #1 acceptance bound); fp64 is exact (0).
_FP32_TOL: float = 1e-4

# ---- torch is OPTIONAL. Imported LAZILY inside the torch path so the module imports on a CPU box. ----
try:                                              # pragma: no cover - exercised only where torch lives
    import torch as _torch                        # noqa: F401
    _HAS_TORCH = True
except Exception:                                 # torch absent -> numpy CPU-sim path only
    _torch = None
    _HAS_TORCH = False


def torch_available() -> bool:
    """True iff `import torch` succeeded (the torch recombine path is usable)."""
    return _HAS_TORCH


# ============================================================================ #
# Column-shard layout helpers (the split boundaries of the feature axis).       #
# ============================================================================ #
def _splits_to_offsets(splits: Sequence[int]) -> List[Tuple[int, int]]:
    """Turn per-device column COUNTS into contiguous [c0, c1) column ranges in ORIGINAL order."""
    offs = []
    c0 = 0
    for w in splits:
        c1 = c0 + int(w)
        offs.append((c0, c1))
        c0 = c1
    return offs


def split_columns(X: np.ndarray, splits: Sequence[int]) -> List[np.ndarray]:
    """Slice X[:, :] into D contiguous COLUMN shards X^(d) = X[:, c0:c1] (original column order)."""
    X = np.asarray(X)
    F = X.shape[1]
    assert int(sum(splits)) == F, f"column splits {list(splits)} must sum to F={F}"
    return [X[:, c0:c1] for (c0, c1) in _splits_to_offsets(splits)]


def split_weight_rows(W: np.ndarray, splits: Sequence[int]) -> List[np.ndarray]:
    """ROW-shard the weight matrix W[F, Fout] to MATCH the column shards: W^(d) = W[c0:c1, :].

    This is the Megatron row-parallel-linear partition: because the matmul h @ W contracts over the
    INPUT feature axis (W's rows), sharding W's rows by the SAME boundaries as h's columns makes
    h @ W = sum_d  h^(d) @ W^(d)  a clean partial-sum across devices."""
    W = np.asarray(W)
    F = W.shape[0]
    assert int(sum(splits)) == F, f"weight row splits {list(splits)} must sum to F={F}"
    return [W[c0:c1, :] for (c0, c1) in _splits_to_offsets(splits)]


# ============================================================================ #
# Cost + layout of the column-shard integration (the RecombineSpec).            #
# ============================================================================ #
@dataclass
class RecombineSpec:
    """Cost + layout of the feature-split full-layer recombination on D devices.

    Fields:
      num_devices    : D, the number of column shards.
      F              : total feature width (sum of cols_per_device).
      cols_per_device: int[D], the F_d columns each device owns (the column shard sizes).
      layers         : L, the number of GNN layers (each layer recombines once).
      full_layer     : True  -> the cost INCLUDES the Megatron W-mix all-reduce per layer (H2: the
                                WHOLE layer is recombined, not just aggregation).
                       False -> aggregation-only concat (the already-proven §38 case).
      recombine_bytes: total bytes moved over the link by the recombination across all L layers.
      recombine_ms   : the recombine wall-clock contribution (feeds JobEstimate.per_epoch_sec)."""
    num_devices: int
    F: int
    cols_per_device: np.ndarray
    layers: int
    full_layer: bool = True
    recombine_bytes: int = 0
    recombine_ms: float = 0.0

    def summary(self) -> str:
        cols = np.asarray(self.cols_per_device).tolist()
        kind = "full-layer (agg-concat + W-mix all-reduce)" if self.full_layer else "aggregation-concat only"
        return (f"RecombineSpec[D={self.num_devices} F={self.F} cols={cols} L={self.layers}] "
                f"{kind}: {self.recombine_bytes/1e6:.2f} MB, {self.recombine_ms:.3f} ms over the link")


def plan_recombine(cols_per_device, F: int, num_nodes: int, layers: int, link_gbps: float,
                   *, full_layer: bool = True) -> RecombineSpec:
    """Cost + layout the column-shard integration over the slow link, for L layers.

    Per layer the recombination moves, over the link:
      * AGGREGATION concat (always): each device's F_d-wide aggregated rows are gathered so the row is
        whole again -> sum_d F_d * N = N * F feature words.  (This is the proven concat case.)
      * W-MIX all-reduce (full_layer=True, H2): each device produces a FULL-WIDTH partial output
        p^(d) in R^{N x Fout}; an all-reduce sums the D partials. We model Fout == F (square mix, the
        usual GNN hidden width) and an all-reduce that moves ~2 * N * F words (ring all-reduce moves
        ~2*(D-1)/D * payload; we use the simple 2*payload upper estimate, consistent with arrange's
        conservative roofline). This is the term that closes the aggregation-only H2 gap on the COST side.

    Returns a RecombineSpec with recombine_bytes / recombine_ms (the back-end per-epoch contribution).
    Pure numpy + dataclass; torch-free."""
    cols = np.asarray(cols_per_device, dtype=np.int64)
    F = int(F)
    N = int(num_nodes)
    L = max(0, int(layers))
    link = max(float(link_gbps), 1e-9)
    assert int(cols.sum()) == F, f"cols_per_device {cols.tolist()} must sum to F={F}"

    # Aggregation concat: gather the full N x F width back together -> N*F words per layer.
    agg_words = float(N) * float(F)
    # W-mix all-reduce of D full-width partials: ~2 * N * F words per layer (the Megatron all-reduce).
    mix_words = (2.0 * float(N) * float(F)) if full_layer else 0.0
    per_layer_words = agg_words + mix_words
    total_bytes = int(round(per_layer_words * FEATURE_ROW_BYTES * L))
    recombine_ms = total_bytes / (link * 1e9) * 1e3
    return RecombineSpec(
        num_devices=int(cols.size), F=F, cols_per_device=cols, layers=L, full_layer=bool(full_layer),
        recombine_bytes=total_bytes, recombine_ms=float(recombine_ms))


# ============================================================================ #
# AGGREGATION-layer recombine (the proven concat case).                         #
# ============================================================================ #
def recombine_aggregation(parts: Sequence[np.ndarray], splits: Sequence[int]) -> np.ndarray:
    """Concatenate the per-device aggregated column-shards back into the [N, F] matrix, in ORIGINAL
    column order. `parts[d]` is device d's aggregated slice (shape [N, splits[d]]). Because the
    aggregation is column-independent, the concatenation is BIT-IDENTICAL to the single-device A @ X
    (the proven §38 / fp_aggregate_consistency case). Pure numpy, full precision."""
    parts = [np.asarray(p) for p in parts]
    assert len(parts) == len(splits), "one part per column shard"
    for d, (p, w) in enumerate(zip(parts, splits)):
        assert p.shape[1] == int(w), f"part {d} has {p.shape[1]} cols, expected {int(w)}"
    return np.concatenate(parts, axis=1)


# ============================================================================ #
# FULL-LAYER recombine (the Megatron W-mix all-reduce -- the H2 closure).       #
# ============================================================================ #
def recombine_full_layer(partials: Sequence[np.ndarray], splits: Sequence[int],
                         W_shards: Optional[Sequence[np.ndarray]] = None,
                         *, nonlinearity: str = "relu") -> np.ndarray:
    """Recombine a feature-split layer END-TO-END (Megatron row/col-parallel), result-preserving.

    Two call shapes (both are the same math, just where the W-mix happens):

      (A) `partials` are the FULL-WIDTH per-device partial OUTPUTS p^(d) = h^(d) @ W^(d) already, and
          `W_shards` is None. We SUM them (the all-reduce):  z = sum_d p^(d), then apply the
          nonlinearity ONCE:  y = phi(z).  This is the canonical Megatron pattern -- each device has
          done its own h^(d) @ W^(d) over its column shard; the recombine is just the reduce + phi.

      (B) `partials` are the per-device AGGREGATED column-shards h^(d) = A @ X[:, cols_d] (each [N, F_d])
          and `W_shards` is the ROW-sharded weight [W^(d)] (each [F_d, Fout]). We form each device's
          partial p^(d) = h^(d) @ W^(d), SUM them, then apply phi ONCE. This is the form the executor
          uses when the W-mix is fused into the recombine step.

    Either way the result equals the single-device  phi( (A @ X) @ W )  up to fp add-order, because
    the matmul contracts over the SHARDED input-feature axis:  (A@X) @ W = sum_d (A@X)[:,cols_d] @ W[cols_d,:].
    The nonlinearity is applied AFTER the reduce (it must see the full mixed row). full precision (fp64)."""
    partials = [np.asarray(p, dtype=np.float64) for p in partials]
    assert len(partials) == len(splits), "one partial/shard per column shard"

    if W_shards is None:
        # (A) partials are already full-width p^(d); just all-reduce.
        z = partials[0].copy()
        for p in partials[1:]:
            z = z + p
    else:
        # (B) partials are aggregated shards h^(d); apply each device's W^(d) then all-reduce.
        W_shards = [np.asarray(Wd, dtype=np.float64) for Wd in W_shards]
        assert len(W_shards) == len(partials), "one W-shard per column shard"
        z = None
        for d, (h_d, W_d) in enumerate(zip(partials, W_shards)):
            assert h_d.shape[1] == int(splits[d]), (
                f"shard {d} has {h_d.shape[1]} cols, expected {int(splits[d])}")
            assert W_d.shape[0] == int(splits[d]), (
                f"W-shard {d} has {W_d.shape[0]} rows, expected {int(splits[d])}")
            p_d = h_d @ W_d                      # FULL-WIDTH partial over device d's columns
            z = p_d if z is None else (z + p_d)
    return _apply_nonlinearity(z, nonlinearity)


def _apply_nonlinearity(z: np.ndarray, nonlinearity: str) -> np.ndarray:
    """Elementwise nonlinearity applied ONCE after the cross-device reduce. full precision."""
    nl = (nonlinearity or "identity").lower()
    if nl in ("relu",):
        return np.maximum(z, 0.0)
    if nl in ("identity", "none", "linear"):
        return z
    if nl in ("tanh",):
        return np.tanh(z)
    if nl in ("sigmoid",):
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"unknown nonlinearity {nonlinearity!r}")


# ============================================================================ #
# Reference L-layer GNN (single device) and the sharded L-layer execution.      #
# ============================================================================ #
def _agg_one_layer(src: np.ndarray, dst: np.ndarray, X: np.ndarray) -> np.ndarray:
    """One undirected sum-aggregation h_v = sum_{u in N(v)} X_u (the SpMM A @ X), full precision fp64.
    Matches feature_parallel.fp_aggregate_consistency's _agg (both directions aggregated)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    h = np.zeros((N, X.shape[1]), dtype=np.float64)
    np.add.at(h, dst, X[src])
    np.add.at(h, src, X[dst])
    return h


def reference_llayer(src: np.ndarray, dst: np.ndarray, X: np.ndarray,
                     W: Optional[np.ndarray] = None, *, layers: int = 2,
                     nonlinearity: str = "relu") -> np.ndarray:
    """The SINGLE-DEVICE L-layer GNN reference: for each layer, h = A @ H ; z = h @ W ; H = phi(z).
    When W is None, the layer is aggregation-only (z = h) so the reference reduces to the proven
    aggregation case. Square W (Fout == F) so the width is stable across layers. fp64 (exact)."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    H = np.asarray(X, dtype=np.float64)
    L = max(0, int(layers))
    for _ in range(L):
        h = _agg_one_layer(src, dst, H)          # aggregate
        if W is not None:
            z = h @ np.asarray(W, dtype=np.float64)   # W-mix (mixes all columns)
        else:
            z = h
        H = _apply_nonlinearity(z, nonlinearity)      # nonlinearity AFTER the mix
    return H


def sharded_llayer(src: np.ndarray, dst: np.ndarray, X: np.ndarray, splits: Sequence[int],
                   W: Optional[np.ndarray] = None, *, layers: int = 2,
                   nonlinearity: str = "relu") -> np.ndarray:
    """The FEATURE-SPLIT L-layer execution WITH recombination -- the path zord's executor follows.

    Per layer, across the D column shards (splits):
      1. AGGREGATE per shard:   h^(d) = A @ H[:, cols_d]  (full-graph SpMM on the replicated adjacency).
      2. RECOMBINE before the W-mix:
           * aggregation-only (W is None): CONCAT the shards back to [N, F]  (recombine_aggregation),
             then apply phi -> H_next.  (the proven §38 path)
           * full layer (W given): ROW-shard W to match the column shards, form each device's
             partial p^(d) = h^(d) @ W^(d), ALL-REDUCE (sum) them, apply phi ONCE  (recombine_full_layer
             form (B)) -> z then H_next = phi(z).  (the Megatron H2 path)
      3. RE-SHARD H_next's columns the SAME way to feed the next layer.

    This is bit-identical (up to fp add order) to reference_llayer; verify_recombine certifies it.
    full precision fp64."""
    H = np.asarray(X, dtype=np.float64)
    L = max(0, int(layers))
    offs = _splits_to_offsets(splits)
    W_shards = split_weight_rows(W, splits) if W is not None else None
    for _ in range(L):
        # 1. per-shard aggregation on each device (the full-graph SpMM over its F_d columns)
        h_shards = [_agg_one_layer(src, dst, H[:, c0:c1]) for (c0, c1) in offs]
        # 2. recombine before the mix + nonlinearity
        if W is None:
            h_full = recombine_aggregation(h_shards, splits)   # concat -> [N, F]
            H = _apply_nonlinearity(h_full, nonlinearity)
        else:
            # full-layer: per-device partial h^(d) @ W^(d), all-reduce, phi ONCE (after recombine)
            H = recombine_full_layer(h_shards, splits, W_shards, nonlinearity=nonlinearity)
        # 3. H is full-width [N, F] again; the next layer re-shards its columns (offs unchanged).
    return H


# ============================================================================ #
# Same-result certificate (the H2 / MUST-DO #1 acceptance test).                #
# ============================================================================ #
def verify_recombine(src: np.ndarray, dst: np.ndarray, X: np.ndarray, splits: Sequence[int],
                     *, layers: int = 2, W: Optional[np.ndarray] = None,
                     nonlinearity: str = "relu", tol: float = _FP32_TOL) -> Tuple[float, bool]:
    """END-TO-END same-result certificate for the feature split across the WHOLE L-layer pipeline.

    Runs (a) the single-device L-layer reference (reference_llayer) and (b) the feature-split path WITH
    recombination (sharded_llayer), and returns (max_abs_err, ok) where ok = (max_abs_err <= tol).

    This is the MUST-DO #1 acceptance: aggregate -> W-mix -> nonlinearity over L >= 2 layers, single
    device vs feature-split+recombine, asserting max-abs-error within fp tolerance END-TO-END across all
    layers. With W given (the full layer) it exercises the Megatron W-mix all-reduce; with W None it
    exercises the proven aggregation-concat case. Computed in fp64 (the reference is exact; the split
    differs only by fp add order, so the error is ~machine-epsilon and well under tol=1e-4). torch-free.

    Args:
      src, dst : int64 [E] undirected edge endpoints.
      X        : [N, F] node features.
      splits   : per-device column counts summing to F (the feature shard sizes).
      layers   : L (>= 1; use >= 2 for the MUST-DO acceptance).
      W        : optional [F, F] square weight matrix for the W-mix (None -> aggregation-only).
      tol      : same-result tolerance (1e-4 for fp32; fp64 reference is ~exact).
    Returns: (max_abs_err: float, ok: bool)."""
    X = np.asarray(X, dtype=np.float64)
    F = X.shape[1]
    assert int(sum(splits)) == F, f"column splits {list(splits)} must sum to F={F}"
    if W is not None:
        W = np.asarray(W, dtype=np.float64)
        assert W.shape[0] == F, f"W must have F={F} rows (the input feature axis), got {W.shape}"

    ref = reference_llayer(src, dst, X, W, layers=layers, nonlinearity=nonlinearity)
    got = sharded_llayer(src, dst, X, splits, W, layers=layers, nonlinearity=nonlinearity)
    max_abs_err = float(np.abs(ref - got).max()) if ref.size else 0.0
    return max_abs_err, bool(max_abs_err <= tol)


# ============================================================================ #
# OPTIONAL torch path (full precision fp32, TF32 disabled). Lazy + guarded.      #
# ============================================================================ #
def recombine_full_layer_torch(partials: Sequence["object"], splits: Sequence[int],
                               W_shards: Optional[Sequence["object"]] = None,
                               *, nonlinearity: str = "relu"):
    """torch counterpart of recombine_full_layer: all-reduce the per-device partials (or form
    h^(d) @ W^(d) first when W_shards given), then apply the nonlinearity ONCE. FULL PRECISION fp32 --
    TF32 is disabled via memtier.enforce_full_precision() so the matmul does not silently round.
    Raises if torch is absent (use the numpy path on a CPU box)."""
    if not _HAS_TORCH:
        raise RuntimeError("recombine_full_layer_torch needs torch; use recombine_full_layer() (numpy).")
    from .memtier import enforce_full_precision
    enforce_full_precision()                      # SACRED: keep fp32 fp32 (no TF32 rounding)
    ps = [p if _is_torch_tensor(p) else _torch.as_tensor(np.asarray(p), dtype=_torch.float32)
          for p in partials]
    if W_shards is None:
        z = ps[0].clone()
        for p in ps[1:]:
            z = z + p
    else:
        Ws = [w if _is_torch_tensor(w) else _torch.as_tensor(np.asarray(w), dtype=_torch.float32)
              for w in W_shards]
        z = None
        for d, (h_d, W_d) in enumerate(zip(ps, Ws)):
            p_d = h_d @ W_d
            z = p_d if z is None else (z + p_d)
    return _apply_nonlinearity_torch(z, nonlinearity)


def _is_torch_tensor(x) -> bool:
    return _HAS_TORCH and isinstance(x, _torch.Tensor)


def _apply_nonlinearity_torch(z, nonlinearity: str):
    nl = (nonlinearity or "identity").lower()
    if nl == "relu":
        return _torch.clamp(z, min=0.0)
    if nl in ("identity", "none", "linear"):
        return z
    if nl == "tanh":
        return _torch.tanh(z)
    if nl == "sigmoid":
        return _torch.sigmoid(z)
    raise ValueError(f"unknown nonlinearity {nonlinearity!r}")


__all__ = [
    "FEATURE_ROW_BYTES", "N_GATHERS",
    "RecombineSpec", "plan_recombine",
    "split_columns", "split_weight_rows",
    "recombine_aggregation", "recombine_full_layer",
    "reference_llayer", "sharded_llayer",
    "verify_recombine",
    "recombine_full_layer_torch", "torch_available",
]
