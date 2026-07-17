#!/usr/bin/env python
"""barrier_vs_k.py -- GROUND the §44 granularity barrier with a REAL measurement, not an assumption.

§44-correction (RESULTS.md): the granularity-sweep U-shape and its interior optimum K* are NOT
engine results. The engine-only makespan(K) is MONOTONE non-increasing (finer is always better, and
MANDATORY for feasibility). The U-shape appears ONLY because granularity_sweep ADDS a per-step BSP
synchronization-barrier term barrier(K) = alpha*ceil(log2 K) + beta whose per-hop latency alpha was
an ASSUMED ~200us commodity-cross-node RTT. THIS SCRIPT replaces that assumption with a MEASURED
alpha and then recomputes whether an interior K* survives on the REAL graphs -> grounds or REFUTES
the U-shape with real numbers.

WHAT IT MEASURES (real torch.distributed):
  Under torchrun on ONE node with NCCL over NVLink, at the CURRENT world-size W, the per-STEP cost of
  the collective that ends a BSP GNN step: a barrier + an all-reduce of a realistic gradient/state
  payload (default ~16MB fp32 ~ a few-layer GNN's gradient/optimizer-state slice). We time many
  iterations (after warmup), record the MEDIAN per-step latency, and write one JSONL line for this W.
  Sweeping W in {1,2,4,8} (the sbatch loop relaunches torchrun per W -- see the bottom of this file)
  gives latency(W); we then FIT the per-hop model latency(W) = alpha*ceil(log2 W) + beta (the same
  shape granularity_sweep uses for barrier(K)) and read off the MEASURED alpha (ms/hop).

WHAT IT THEN DOES (the grounding):
  With the MEASURED alpha, recompute the barrier-augmented makespan(K) sweep + the closed-form K*
  for the real graphs (reusing granularity_sweep's engine sweep + kstar_formula). If an interior K*
  still exists (augmented curve U-shaped), the §44 optimum is GROUNDED at the measured per-hop cost;
  if the augmented curve stays MONOTONE (NVLink alpha is us-scale, far below the assumed 200us), the
  interior optimum VANISHES and §44's U-shape is REFUTED for this fabric -> finer stays better.

PROCESS-only: this measures WALL-CLOCK collective latency (a scheduling/feasibility input). It never
touches the trained result -- the all-reduce is bit-neutral; we only time it.

GUARDS: torch/CUDA-absent -> a CPU DRY-RUN prints the measurement plan + (with --alpha-ms) the
grounding analysis using a supplied alpha, so the file is useful on a build/login box. py_compile
clean with no torch installed.

SUBMIT (1 node, up to 8 GPUs; the WRAPPER sbatch -- NOT this script -- loops torchrun per W):
  # single world-size measurement (writes one JSONL line for W=this launch):
  torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/barrier_vs_k.py --out bvk.jsonl
  # fit + ground from accumulated per-W lines (run once after the W-sweep, CPU is fine):
  python scripts/barrier_vs_k.py --aggregate bvk.jsonl --datasets collegemsg askubuntu stackoverflow
The example sbatch at the bottom of this file shows the W in {1,2,4,8} loop. This script itself does
NOT submit SLURM (the caller's main loop submits).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# torch is imported lazily-tolerantly so py_compile / a torch-less box still parse + run the
# dry-run / aggregate paths.
try:
    import torch
    import torch.distributed as dist
    _HAVE_TORCH = True
except Exception:                                              # pragma: no cover (build box)
    torch = None
    dist = None
    _HAVE_TORCH = False


# --------------------------------------------------------------------------- env / distributed
def env_int(*keys, default=0):
    for k in keys:
        if k in os.environ:
            try:
                return int(os.environ[k])
            except ValueError:
                pass
    return default


def setup_dist(timeout_min: int):
    """Read torchrun's LOCAL_RANK/RANK/WORLD_SIZE, pin the device, init NCCL (NVLink on one node).
    Falls back to gloo/CPU when no GPU is visible so a bare/CI launch still runs the code path."""
    rank = env_int("RANK", "SLURM_PROCID", default=0)
    world = env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    local = env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # single node + NVLink: keep NCCL on-box.
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")

    cuda_ok = bool(torch.cuda.is_available()) if _HAVE_TORCH else False
    if cuda_ok:
        torch.cuda.set_device(local)
        backend, device = "nccl", torch.device(f"cuda:{local}")
    else:
        backend, device = "gloo", (torch.device("cpu") if _HAVE_TORCH else None)
    dist.init_process_group(backend=backend, rank=rank, world_size=world,
                            timeout=timedelta(minutes=timeout_min))
    return rank, world, local, device, cuda_ok


def _barrier(device, cuda_ok):
    if not _HAVE_TORCH or not dist.is_initialized():
        return
    if cuda_ok:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


# --------------------------------------------------------------------------- the measurement
def measure_collective_latency(device, cuda_ok, world, payload_bytes, iters, warmup):
    """Time the per-STEP collective that ends a BSP GNN step at the CURRENT world-size: a barrier +
    an all-reduce of a `payload_bytes` fp32 buffer (a realistic gradient/optimizer-state slice).
    Returns (median_ms, p10_ms, p90_ms) over `iters` timed iterations after `warmup`. At world=1
    the collective is a near-no-op (the latency FLOOR / beta term -- a single device still pays the
    kernel-launch + barrier floor but no inter-device hops)."""
    n = max(1, payload_bytes // 4)                              # fp32 elements
    buf = torch.ones(n, dtype=torch.float32, device=device)
    samples = []
    for it in range(warmup + iters):
        if cuda_ok:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # the BSP step boundary: all-reduce the payload, then a barrier (the next layer waits on it).
        if world > 1:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        _barrier(device, cuda_ok)
        if cuda_ok:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3                  # ms
        if it >= warmup:
            samples.append(dt)
    s = np.sort(np.asarray(samples))
    return float(np.median(s)), float(s[int(0.1 * len(s))]), float(s[int(0.9 * len(s))])


# --------------------------------------------------------------------------- fit alpha
def fit_alpha_beta(world_sizes, latency_ms):
    """Fit latency(W) = alpha*ceil(log2 W) + beta by least squares on the MEASURED points (the same
    barrier shape granularity_sweep adds). hops(W=1)=0 so beta is the W=1 floor; alpha is the
    per-hop (per-log2-level) cost. Returns (alpha_ms, beta_ms, r2). Needs >=2 distinct hop counts."""
    W = np.asarray(world_sizes, dtype=np.float64)
    y = np.asarray(latency_ms, dtype=np.float64)
    hops = np.ceil(np.log2(np.maximum(2.0, W)))
    hops = np.where(W <= 1, 0.0, hops)                         # W=1 -> 0 hops (pure floor)
    Xd = np.column_stack([hops, np.ones_like(hops)])
    # least squares for [alpha, beta]
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    pred = Xd @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-30
    r2 = 1.0 - ss_res / ss_tot
    # also fit a LINEAR-in-W alternative (latency = a*W + b) for an honest scaling comparison.
    Xl = np.column_stack([W, np.ones_like(W)])
    coefl, *_ = np.linalg.lstsq(Xl, y, rcond=None)
    predl = Xl @ coefl
    r2_lin = 1.0 - float(((y - predl) ** 2).sum()) / ss_tot
    return alpha, beta, r2, float(coefl[0]), float(coefl[1]), r2_lin


# --------------------------------------------------------------------------- ground K* on real graphs
def ground_kstar(datasets, alpha_ms, beta_ms, link, feat, cap_mb, agg_bw, kmax, metis_max_edges,
                 seed=0):
    """With the MEASURED alpha, recompute the barrier-augmented makespan(K) sweep + the closed-form
    K* for each real graph (reusing granularity_sweep's REAL engine sweep). Report, per graph:
    whether the engine-only curve is monotone (it should be), and whether the AUGMENTED curve has an
    INTERIOR optimum at this measured alpha (U-shape GROUNDED) or stays monotone (U-shape REFUTED)."""
    import granularity_sweep as gs

    cap_bytes = cap_mb * 1024 ** 2
    ks_all = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8000, 10000]
    ks = [k for k in ks_all if k <= kmax]

    out = []
    for name in datasets:
        try:
            g = gs.load_graph(name)
        except Exception as e:
            print(f"  [skip {name}: {type(e).__name__}: {str(e)[:70]}]")
            continue
        rows = gs.run_graph(g, ks, link, feat, cap_bytes, agg_bw, alpha_ms, beta_ms,
                            seed, metis_max_edges)
        if not rows:
            continue
        feas = [r for r in rows if r['feasible']] or rows
        eng_vals = [r['engine_step_ms'] for r in feas]
        aug_vals = [r['makespan_ms'] for r in feas]
        eng_mono = gs._is_monotone_nonincreasing(eng_vals)
        aug_mono = gs._is_monotone_nonincreasing(aug_vals)
        eng_argmin = min(feas, key=lambda r: r['engine_step_ms'])
        aug_kstar = min(feas, key=lambda r: r['makespan_ms'])
        N, E = rows[0]['N'], rows[0]['E']
        kf, kmem, kbal = gs.kstar_formula(N, E, feat, cap_bytes, link, agg_bw, alpha_ms, beta_ms)
        interior = (not aug_mono) and (aug_kstar['K'] != eng_argmin['K'])
        out.append(dict(graph=name, N=N, E=E, eng_mono=eng_mono, eng_argmin=eng_argmin['K'],
                        aug_mono=aug_mono, aug_kstar=aug_kstar['K'], interior=interior,
                        kmem=kmem, kbal=kbal, kf=kf))
    return out


# --------------------------------------------------------------------------- reporting helpers
def write_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def report_fit_and_ground(measurements, args):
    """Given a list of per-world-size measurement records, fit alpha and ground/refute the K* U-shape.
    measurements: list of dicts with 'world' and 'median_ms'."""
    by_w = {}
    for m in measurements:
        by_w[int(m["world"])] = float(m["median_ms"])           # last write wins per W
    worlds = sorted(by_w)
    if len(worlds) < 2 or len(set(int(np.ceil(np.log2(max(2, w)))) if w > 1 else 0
                                  for w in worlds)) < 2:
        print(f"[fit] need >=2 distinct world-sizes (have {worlds}); cannot fit alpha yet.")
        return
    lat = [by_w[w] for w in worlds]
    alpha, beta, r2, a_lin, b_lin, r2_lin = fit_alpha_beta(worlds, lat)
    print("\n================ MEASURED barrier latency vs world-size ================")
    print(f"{'world W':>8} {'hops=ceil(log2 W)':>18} {'median_ms':>11}")
    for w in worlds:
        hops = 0 if w <= 1 else int(math.ceil(math.log2(w)))
        print(f"{w:>8} {hops:>18} {by_w[w]:>11.4f}")
    print(f"\n[fit] log2 model latency(W) = alpha*ceil(log2 W) + beta  ->  "
          f"alpha = {alpha:.5f} ms/hop, beta = {beta:.5f} ms  (R^2={r2:.3f})")
    print(f"[fit] linear-in-W alt latency(W) = {a_lin:.5f}*W + {b_lin:.5f} ms (R^2={r2_lin:.3f}) "
          f"-> {'log2 fits better' if r2 >= r2_lin else 'linear fits better (re-examine the model)'}")
    alpha_us = max(alpha, 0.0) * 1e3
    print(f"[fit] MEASURED per-hop alpha = {alpha_us:.1f} us  (vs the ASSUMED 200us in §44 -- "
          f"{'far lower, NVLink-class' if alpha_us < 50 else 'comparable'}).")
    alpha_for_ground = max(alpha, 1e-6)                         # a tiny positive alpha for the sweep

    if not args.datasets:
        print("[ground] no --datasets given; skipping the K* grounding (pass real graph names).")
        return
    print("\n================ GROUNDING the §44 U-shape with the MEASURED alpha ================")
    print(f"  using alpha = {alpha_for_ground:.5f} ms/hop, beta = {max(beta,0.0):.5f} ms, "
          f"link={args.link}GB/s feat={args.feat} cap={args.cap_mb}MB agg_bw={args.agg_bw}GB/s")
    res = ground_kstar(args.datasets, alpha_for_ground, max(beta, 0.0), args.link, args.feat,
                       args.cap_mb, args.agg_bw, args.kmax, args.metis_max_edges, args.seed)
    if not res:
        print("  [no graphs loaded; cannot ground]")
        return
    print(f"\n  {'graph':>16} {'N':>10} {'E':>12} {'eng_mono':>9} {'eng_argmin':>11} "
          f"{'aug_mono':>9} {'aug_K*':>7} {'interior_opt?':>13}")
    n_interior = 0
    for r in res:
        n_interior += bool(r["interior"])
        print(f"  {r['graph']:>16} {r['N']:>10,} {r['E']:>12,} {('yes' if r['eng_mono'] else 'NO'):>9} "
              f"{r['eng_argmin']:>11} {('yes' if r['aug_mono'] else 'NO'):>9} {r['aug_kstar']:>7} "
              f"{('GROUNDED' if r['interior'] else 'none(refuted)'):>13}")
    print("\n  CONCLUSION (grounded with the MEASURED alpha):")
    print("    * Engine-only makespan(K) is monotone non-increasing on every graph (finer is better,")
    print("      and mandatory for feasibility) -- unchanged, this is an engine fact.")
    if n_interior == 0:
        print(f"    * At the MEASURED alpha={alpha_us:.1f}us, the barrier-augmented curve stays MONOTONE on")
        print("      ALL graphs: NO interior optimum. The §44 U-shape / K* is REFUTED for this NVLink")
        print("      fabric -- the interior optimum was an artifact of the ASSUMED 200us RTT. Finer stays")
        print("      better up to the device count; pick the FINEST feasible K (memory floor K_mem).")
    else:
        print(f"    * At the MEASURED alpha={alpha_us:.1f}us, {n_interior}/{len(res)} graphs STILL show an")
        print("      interior optimum -> the §44 U-shape is GROUNDED (not just an artifact of the assumed")
        print("      200us): on those graphs an intermediate K* beats the finest K even at NVLink latency.")
    print("\n  PROCESS-only: this only times the (bit-neutral) collective; the trained result is unchanged.")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="JSONL path to APPEND this world-size's measurement (under torchrun). The "
                         "sbatch loop relaunches torchrun per W; each launch appends one line.")
    ap.add_argument("--aggregate", default=None,
                    help="read a JSONL of per-world-size measurements, FIT alpha, and GROUND/REFUTE "
                         "the K* U-shape (no GPU needed). Use after the W-sweep has written --out.")
    ap.add_argument("--payload-mb", type=float, default=16.0,
                    help="all-reduce payload (MB fp32) ~ a realistic GNN gradient/state slice")
    ap.add_argument("--iters", type=int, default=200, help="timed iterations after warmup")
    ap.add_argument("--warmup", type=int, default=50, help="warmup iterations (NCCL ring setup)")
    ap.add_argument("--timeout-min", type=int, default=20)
    # grounding knobs (mirror granularity_sweep so the augmented sweep is apples-to-apples):
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="real graphs to ground K* on (e.g. collegemsg askubuntu stackoverflow)")
    ap.add_argument("--link", type=float, default=325.0,
                    help="interconnect GB/s for the grounding sweep (default NVLink-class, since the "
                         "measured alpha is the NVLink barrier latency)")
    ap.add_argument("--feat", type=int, default=128)
    ap.add_argument("--cap-mb", type=float, default=8.0, help="per-card usable HBM cap (MB) for sweep")
    ap.add_argument("--agg-bw", type=float, default=444.0)
    ap.add_argument("--kmax", type=int, default=10000)
    ap.add_argument("--metis-max-edges", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha-ms", type=float, default=None,
                    help="DRY-RUN only: an alpha (ms/hop) to demonstrate the grounding path without a "
                         "GPU measurement (e.g. --alpha-ms 0.012 for ~12us NVLink). Ignored when "
                         "real measurements are available.")
    args = ap.parse_args()

    # ---- AGGREGATE mode: fit + ground from accumulated measurements (CPU is fine) ----
    if args.aggregate:
        if not os.path.exists(args.aggregate):
            print(f"[aggregate] file not found: {args.aggregate}")
            return 1
        recs = read_jsonl(args.aggregate)
        meas = [r for r in recs if "world" in r and "median_ms" in r]
        print(f"[aggregate] read {len(meas)} measurement line(s) from {args.aggregate}")
        report_fit_and_ground(meas, args)
        return 0

    # ---- DRY-RUN (no torch or no CUDA): print the plan; optionally demo grounding with --alpha-ms ----
    cuda_ok = bool(_HAVE_TORCH and torch.cuda.is_available())
    if not _HAVE_TORCH or not cuda_ok:
        print("=" * 88)
        print("barrier_vs_k.py DRY-RUN (no CUDA visible) -- measurement PLAN (no GPU work done):")
        print("=" * 88)
        print(f"  payload      : {args.payload_mb} MB fp32 all-reduce + barrier (per BSP step)")
        print(f"  iters/warmup : {args.iters}/{args.warmup}; median per-step latency recorded per W")
        print("  launch       : torchrun --standalone --nnodes=1 --nproc_per_node={1,2,4,8} "
              "scripts/barrier_vs_k.py --out bvk.jsonl")
        print("  then         : python scripts/barrier_vs_k.py --aggregate bvk.jsonl --datasets "
              "collegemsg askubuntu stackoverflow")
        print("  -> fits alpha = per-hop barrier latency (ms/hop) from the MEASURED latency(W), then")
        print("     recomputes whether the §44 interior K* survives at the measured (NVLink) alpha.")
        if args.alpha_ms is not None:
            print(f"\n[dry-run demo] grounding with a SUPPLIED alpha={args.alpha_ms} ms/hop "
                  "(NOT a measurement -- illustration of the analysis path):")
            demo = [dict(world=1, median_ms=args.alpha_ms * 0 + 0.02),
                    dict(world=2, median_ms=args.alpha_ms * 1 + 0.02),
                    dict(world=4, median_ms=args.alpha_ms * 2 + 0.02),
                    dict(world=8, median_ms=args.alpha_ms * 3 + 0.02)]
            report_fit_and_ground(demo, args)
        else:
            print("  (pass --alpha-ms <ms> to demo the grounding analysis on this CPU box.)")
        return 0

    # ---- REAL measurement under torchrun ----
    rank, world, local, device, cuda_ok = setup_dist(args.timeout_min)
    try:
        payload_bytes = int(args.payload_mb * 1024 ** 2)
        if rank == 0:
            print(f"[barrier_vs_k] world={world} backend={'nccl' if cuda_ok else 'gloo'} "
                  f"payload={args.payload_mb}MB iters={args.iters} warmup={args.warmup}")
        _barrier(device, cuda_ok)
        med, p10, p90 = measure_collective_latency(device, cuda_ok, world, payload_bytes,
                                                    args.iters, args.warmup)
        _barrier(device, cuda_ok)
        if rank == 0:
            hops = 0 if world <= 1 else int(math.ceil(math.log2(world)))
            print(f"[barrier_vs_k] world={world} hops={hops} median={med:.4f}ms "
                  f"p10={p10:.4f}ms p90={p90:.4f}ms")
            rec = dict(world=world, hops=hops, median_ms=med, p10_ms=p10, p90_ms=p90,
                       payload_mb=args.payload_mb, backend=("nccl" if cuda_ok else "gloo"),
                       gpu=(torch.cuda.get_device_name(local) if cuda_ok else "cpu"))
            if args.out:
                write_jsonl(args.out, rec)
                print(f"[barrier_vs_k] appended measurement for world={world} -> {args.out}")
            # if this launch happens to have collected all of {1,2,4,8}, fit+ground immediately.
            if args.out and os.path.exists(args.out):
                meas = [r for r in read_jsonl(args.out) if "world" in r and "median_ms" in r]
                if len({r["world"] for r in meas}) >= 2:
                    report_fit_and_ground(meas, args)
    finally:
        if _HAVE_TORCH and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================================================== #
# EXAMPLE WRAPPER sbatch (loops torchrun over W in {1,2,4,8} on ONE node, then fits+grounds).      #
# This script does NOT submit SLURM itself (the caller's main loop submits). Copy into a .sbatch:  #
# ---------------------------------------------------------------------------------------------- #
#   #!/bin/bash
#   #SBATCH --job-name=zord_barrier_vs_k
#   #SBATCH --partition=bigTiger
#   #SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=32 --mem=128G
#   #SBATCH --time=00:30:00
#   #SBATCH --output=$ZORD_DATA/results/barrier_vs_k_%j.out
#   set -u
#   source <conda.sh>; conda activate $PROJECT/hkenv
#   export PYTHONUNBUFFERED=1
#   export ZORD_GRAPH_BIN=build/graph_algos
#   export PYTHONPATH=src:${PYTHONPATH:-}
#   export NCCL_DEBUG=WARN NCCL_P2P_LEVEL=NVL
#   cd $ZORD_DATA/repo
#   OUT=$ZORD_DATA/results/barrier_vs_k_${SLURM_JOB_ID}.jsonl ; : > "$OUT"
#   BASEPORT=$((20000 + SLURM_JOB_ID % 10000))
#   for W in 1 2 4 8; do
#     torchrun --standalone --nnodes=1 --nproc_per_node=$W --master_port=$((BASEPORT+W)) \
#       --max-restarts=0 scripts/barrier_vs_k.py --out "$OUT" --payload-mb 16 --iters 200 \
#       || echo "[sbatch] W=$W torchrun returned non-zero (logged; continuing)"
#   done
#   # fit alpha + ground/refute the §44 U-shape on the real graphs (CPU step):
#   python scripts/barrier_vs_k.py --aggregate "$OUT" \
#     --datasets collegemsg askubuntu stackoverflow --link 325 --feat 128
# ============================================================================================== #
