#!/usr/bin/env python
"""Two counter-intuitive PROCESS bets for boundary nodes in a distributed temporal-GNN
partition (TIME / MEMORY / COMM only -- SAME result, NEVER accuracy):

  D3  REPLICATE hot boundary nodes.  Instead of fetching a high-degree boundary node's
      features across the link EVERY step, keep a replicated copy on each consuming
      device.  We model the MEMORY cost of replicating the top-X% highest-degree boundary
      nodes vs the COMM saved (the cross-device fetches we no longer issue).  Sweep X;
      report comm-bytes-saved vs extra-memory and the breakeven point.

  D4  RECOMPUTE instead of communicate.  For a boundary node, recompute its 1-hop
      embedding LOCALLY from (already-replicated) inputs instead of fetching the computed
      embedding across the link.  We compare COMPUTE cost (a local SpMM over those nodes)
      vs COMM cost (transferring the embeddings) under three link bandwidths --
      NVLink 325 / PCIe 25 / Ethernet 0.12 GB/s -- and find when recompute wins.

Setup: N nodes contiguously sharded across P devices (shard p owns [p*n, (p+1)*n)).
An `--intra` knob controls the fraction of edges kept inside a shard (a good locality
partition -> high intra -> few/cheap boundary; bad partition -> low intra -> many).
A boundary node is one OWNED by shard p but NEEDED (as a neighbor) by some other shard.

The SpMM in D4 is measured on a real GPU when one is available (torch+CUDA); otherwise we
fall back to an explicit, clearly-labelled roofline cost model so the regimes still print.

  python scripts/replicate_vs_recompute.py --devices 4 --feat 256 --intra 0.8
  python scripts/replicate_vs_recompute.py --dataset wikipedia --devices 4 --feat 128
"""
import argparse
import time

import numpy as np

# torch is optional at import time; we only need it for the GPU SpMM measurement in D4.
try:
    import torch
    _HAVE_TORCH = True
except Exception:  # pragma: no cover - environment without torch
    torch = None
    _HAVE_TORCH = False

BYTES = 4  # float32 feature element

# Link bandwidths (GB/s, 1 GB = 1e9 bytes) -- the regimes we sweep in D4 and price D3 in.
LINKS = {"NVLink": 325.0, "PCIe": 25.0, "Ethernet": 0.12}


# --------------------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------------------
def gen_synthetic(N, M, P, intra, seed=0):
    """Contiguous-shard synthetic graph. With prob `intra` an edge stays inside the
    source node's shard (good partition); otherwise the destination is anywhere.
    Returns (src, dst) as int64 endpoint arrays (undirected: we add both directions
    downstream when counting consumption)."""
    rng = np.random.default_rng(seed)
    n = N // P
    src = rng.integers(0, N, size=M)
    shard = (src // n).clip(0, P - 1)
    lo = (shard * n)
    hi = np.where(shard == P - 1, N, lo + n)
    dst_in = lo + (rng.random(M) * np.maximum(1, hi - lo)).astype(np.int64)
    dst_any = rng.integers(0, N, size=M)
    keep = rng.random(M) < intra
    dst = np.where(keep, dst_in, dst_any).clip(0, N - 1)
    return src.astype(np.int64), dst.astype(np.int64)


def shard_of(nodes, N, P):
    n = N // P
    return np.minimum(nodes // n, P - 1)


# --------------------------------------------------------------------------------------
# boundary / consumption accounting (the "cut")
# --------------------------------------------------------------------------------------
def boundary_accounting(src, dst, N, P):
    """Treat the graph as undirected: an edge (u,v) means each endpoint needs the other's
    feature. A node `o` owned by shard p is a BOUNDARY node if some consumer node in a
    DIFFERENT shard q has `o` as a neighbor; that consumer must fetch o's feature once
    per step (de-duplicated per (owner, consuming-shard) pair, as a real fetch would be).

    Returns:
      bnodes        int64[B]   distinct boundary owner-node ids
      bdeg          int64[B]   how many DISTINCT remote shards need each boundary node
                               (= replicas required if we replicate it everywhere it's used,
                                and = number of fetch-rows saved per step by replicating it)
      total_fetch_rows int     baseline cross-device feature rows fetched per step
                               (sum over boundary nodes of #distinct remote consuming shards)
    """
    # Make undirected endpoint pairs: (owner, consumer)
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    su = shard_of(u, N, P)
    sv = shard_of(v, N, P)
    cross = su != sv                      # u owned here, v consumes from another shard
    owner = u[cross]
    csh = sv[cross]                       # consuming shard
    if owner.size == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int64), 0)
    # de-duplicate (owner, consuming-shard): a shard fetches a given remote node once/step
    key = owner.astype(np.int64) * P + csh.astype(np.int64)
    key = np.unique(key)
    bn = key // P
    bnodes, bdeg = np.unique(bn, return_counts=True)   # #distinct remote shards per owner
    total_fetch_rows = int(bdeg.sum())
    return bnodes, bdeg.astype(np.int64), total_fetch_rows


def degree_of(src, dst, N):
    return (np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)).astype(np.int64)


# --------------------------------------------------------------------------------------
# D3 -- replicate hot boundary nodes: memory cost vs comm saved
# --------------------------------------------------------------------------------------
def run_d3(bnodes, bdeg, deg, F, total_fetch_rows, steps_per_epoch, link_name):
    """Sweep X = top-X% highest-degree boundary nodes to REPLICATE.

    extra memory  = (#replica copies created) * F * BYTES
                    a node needed by k remote shards costs k replica copies if replicated
                    everywhere it is consumed (we replicate exactly where it would be fetched).
    comm saved/step = (#fetch rows eliminated) * F * BYTES
                    replicating a node removes exactly its k remote fetches each step.
    So per node the *row* count is identical (bdeg) for both memory cost and comm-per-step
    saved -- the bet is that the ONE-TIME memory buy-in is repaid by per-step comm savings.
    """
    B = bnodes.size
    print("\n" + "=" * 96)
    print(f"D3  REPLICATE hot boundary nodes (F={F}, {steps_per_epoch} steps/epoch, "
          f"link={link_name} {LINKS[link_name]:g} GB/s)")
    print("=" * 96)
    if B == 0:
        print("  no boundary nodes (intra=1.0 / single device) -> nothing to replicate.")
        return
    row_bytes = F * BYTES
    # order boundary nodes by their *true* graph degree (hot first)
    order = np.argsort(-deg[bnodes], kind="stable")
    bdeg_sorted = bdeg[order]
    cum_rows = np.cumsum(bdeg_sorted)                  # replica rows if we take top-k nodes
    total_rows = int(cum_rows[-1])                     # == total_fetch_rows
    print(f"  boundary nodes B={B}  total replica/fetch rows={total_rows}  "
          f"row={row_bytes} B  baseline fetch/step={total_fetch_rows*row_bytes/1e6:.2f} MB")
    print(f"  {'X%':>5} {'#nodes':>8} {'extra_mem':>11} {'comm_saved/step':>16} "
          f"{'comm_saved/epoch':>17} {'steps_to_breakeven':>19}")
    for X in (0.5, 1, 2, 5, 10, 25, 50, 100):
        k = max(1, int(round(B * X / 100.0)))
        k = min(k, B)
        rows = int(cum_rows[k - 1])
        extra_mem = rows * row_bytes                   # bytes resident on devices
        saved_step = rows * row_bytes                  # bytes not fetched each step
        saved_epoch = saved_step * steps_per_epoch
        # breakeven: replicating costs `extra_mem` bytes of memory once; it saves
        # `saved_step` bytes of comm per step. In MEMORY==COMM-byte terms breakeven is
        # 1 step (rows are equal). The meaningful breakeven is in *time*: extra memory is
        # a capacity cost, comm saved is a recurring time cost -> see the link-time line.
        steps_be = extra_mem / saved_step if saved_step else float("inf")
        print(f"  {X:>5g} {k:>8d} {extra_mem/1e6:>9.2f}MB {saved_step/1e6:>14.2f}MB "
              f"{saved_epoch/1e9:>15.3f}GB {steps_be:>19.2f}")
    # Time framing: how long does the saved comm take on this link (the recurring win)?
    full_saved_s = total_rows * row_bytes / (LINKS[link_name] * 1e9)
    print(f"  => replicating ALL boundary nodes adds {total_rows*row_bytes/1e6:.2f} MB of "
          f"device memory and removes {full_saved_s*1e3:.3f} ms of {link_name} fetch per step")
    print(f"     ({full_saved_s*steps_per_epoch*1e3:.1f} ms/epoch). The bet wins whenever that "
          f"recurring comm-time matters and the one-time memory fits.")


# --------------------------------------------------------------------------------------
# D4 -- recompute vs communicate
# --------------------------------------------------------------------------------------
def measure_spmm_gpu(n_rows, avg_deg, F, dev):
    """Measure a real 1-hop SpMM (gather+transform) for `n_rows` boundary nodes on GPU.
    Returns seconds/step. Builds a random sparse block (n_rows x C) with avg_deg nnz/row."""
    C = max(n_rows, 1024)
    e = int(n_rows * max(1, avg_deg))
    g = torch.Generator().manual_seed(0)
    rows = torch.randint(0, n_rows, (e,), generator=g)
    cols = torch.randint(0, C, (e,), generator=g)
    vals = torch.ones(e, device=dev)
    A = torch.sparse_coo_tensor(torch.stack([rows.to(dev), cols.to(dev)]), vals,
                                (n_rows, C)).coalesce()
    X = torch.randn(C, F, device=dev)
    W = torch.randn(F, F, device=dev) / F ** 0.5

    def step():
        return torch.relu(torch.sparse.mm(A, X) @ W)

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    reps = 20
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


def spmm_cost_model(n_rows, avg_deg, F, flops=1.5e13, mem_bw=1.5e12):
    """Roofline fallback when no GPU: a 1-hop SpMM gathers (n_rows*avg_deg) feature rows
    and does an FxF transform. Memory-bound term usually dominates aggregation.
      bytes moved ~ nnz*F*BYTES (gather) + n_rows*F*BYTES (write) ; reused via transform.
      flops       ~ 2 * nnz * F (gather-add) + 2 * n_rows * F * F (transform)
    Returns seconds/step = max(mem_time, compute_time). Defaults ~ a mid-range datacentre GPU."""
    nnz = n_rows * max(1, avg_deg)
    bytes_moved = (nnz + n_rows) * F * BYTES + F * F * BYTES
    fl = 2.0 * nnz * F + 2.0 * n_rows * F * F
    return max(bytes_moved / mem_bw, fl / flops)


def run_d4(bnodes, deg, F, steps_per_epoch, dev):
    """Compare, for the set of boundary nodes, RECOMPUTE (local 1-hop SpMM) vs
    COMMUNICATE (fetch the computed embeddings) across the three link bandwidths."""
    print("\n" + "=" * 96)
    print(f"D4  RECOMPUTE vs COMMUNICATE for boundary nodes (F={F})")
    print("=" * 96)
    B = bnodes.size
    if B == 0:
        print("  no boundary nodes -> nothing to recompute/communicate.")
        return
    avg_deg = float(deg[bnodes].mean())
    comm_bytes = B * F * BYTES                         # transfer one embedding row per node
    # recompute time
    if _HAVE_TORCH and dev is not None and dev.startswith("cuda"):
        t_recompute = measure_spmm_gpu(B, avg_deg, F, dev)
        src_label = f"MEASURED on {torch.cuda.get_device_name(0)}"
    else:
        t_recompute = spmm_cost_model(B, avg_deg, F)
        src_label = "COST-MODEL (no CUDA GPU available)"
    print(f"  boundary nodes B={B}  avg_deg={avg_deg:.1f}  embedding transfer={comm_bytes/1e6:.2f} MB")
    print(f"  recompute (local 1-hop SpMM) = {t_recompute*1e3:.3f} ms/step   [{src_label}]")
    print(f"  {'link':>9} {'GB/s':>8} {'comm_time/step':>15} {'recompute/step':>15} "
          f"{'winner':>12} {'speedup':>9}")
    for name, gbs in LINKS.items():
        t_comm = comm_bytes / (gbs * 1e9)
        if t_recompute < t_comm:
            winner, sp = "RECOMPUTE", t_comm / max(t_recompute, 1e-12)
        else:
            winner, sp = "COMMUNICATE", t_recompute / max(t_comm, 1e-12)
        print(f"  {name:>9} {gbs:>8g} {t_comm*1e3:>13.3f}ms {t_recompute*1e3:>13.3f}ms "
              f"{winner:>12} {sp:>8.2f}x")
    # Breakeven bandwidth: recompute wins when comm_time > recompute_time
    #   comm_bytes / (BW*1e9) > t_recompute  ->  BW < comm_bytes / (t_recompute*1e9)
    be_bw = comm_bytes / (t_recompute * 1e9) if t_recompute else float("inf")
    print(f"  => breakeven link bandwidth = {be_bw:.2f} GB/s: recompute is cheaper than "
          f"communicate on ANY link SLOWER than this.")
    regime = []
    for name, gbs in LINKS.items():
        regime.append(f"{name}({'recompute' if gbs < be_bw else 'communicate'})")
    print("     regimes: " + ", ".join(regime))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="", help="zord dataset name (else synthetic)")
    ap.add_argument("--devices", type=int, default=4, help="number of partitions/devices P")
    ap.add_argument("--feat", type=int, default=256, help="feature width F")
    ap.add_argument("--intra", type=float, default=0.8,
                    help="synthetic: fraction of edges kept inside a shard (locality knob)")
    ap.add_argument("--nodes", type=int, default=2_000_000, help="synthetic node count")
    ap.add_argument("--edges", type=int, default=20_000_000, help="synthetic edge count")
    ap.add_argument("--steps", type=int, default=200,
                    help="steps/epoch for amortizing the D3 memory buy-in")
    ap.add_argument("--d3-link", default="PCIe", choices=list(LINKS),
                    help="link used to price the D3 comm savings in time")
    a = ap.parse_args()

    P, F = a.devices, a.feat
    t0 = time.time()
    if a.dataset:
        from zord.datasets import load
        g = load(a.dataset)
        N = int(g.num_nodes)
        src = np.asarray(g.src, dtype=np.int64)
        dst = np.asarray(g.dst, dtype=np.int64)
        M = src.size
        print(f"REPL-vs-RECOMP dataset={g.name} N={N} M={M} P={P} F={F} "
              f"(contiguous shards; real node-id locality)")
    else:
        N, M = a.nodes, a.edges
        N = (N // P) * P  # make shards even
        src, dst = gen_synthetic(N, M, P, a.intra)
        print(f"REPL-vs-RECOMP SYNTHETIC N={N} M={M} P={P} F={F} intra={a.intra} "
              f"(contiguous shards)")
    print(f"  built graph in {time.time()-t0:.1f}s")

    dev = None
    if _HAVE_TORCH and torch.cuda.is_available():
        dev = "cuda:0"
        print(f"  GPU: {torch.cuda.get_device_name(0)} (D4 SpMM will be MEASURED)")
    else:
        print("  no CUDA GPU -> D4 SpMM uses an explicit roofline COST-MODEL")

    deg = degree_of(src, dst, N)
    bnodes, bdeg, total_fetch_rows = boundary_accounting(src, dst, N, P)
    frac = 100.0 * bnodes.size / max(1, N)
    print(f"  partition cut: {bnodes.size} boundary nodes ({frac:.2f}% of N), "
          f"{total_fetch_rows} cross-device fetch rows/step "
          f"({total_fetch_rows*F*BYTES/1e6:.2f} MB/step)")

    run_d3(bnodes, bdeg, deg, F, total_fetch_rows, a.steps, a.d3_link)
    run_d4(bnodes, deg, F, a.steps, dev)

    print("\n" + "-" * 96)
    print("SUMMARY")
    print("  D3 wins when: per-step cross-device FETCH time (hot boundary nodes) is a")
    print("     recurring cost and the one-time replica MEMORY fits -- replicate the few")
    print("     highest-degree nodes first (they carry most fetch rows for least memory).")
    print("  D4 wins when: the link is SLOWER than the recompute breakeven bandwidth --")
    print("     fast NVLink favours COMMUNICATE; PCIe/Ethernet favour RECOMPUTE.")
    print("-" * 96)


if __name__ == "__main__":
    main()
