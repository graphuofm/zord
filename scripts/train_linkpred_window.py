#!/usr/bin/env python
"""C2 contradiction, MEASURED: temporal link-prediction accuracy vs history WINDOW vs memory.
A windowed temporal GraphSAGE predicts future edges from the last W snapshots (recency-decayed).
Longer window -> more temporal context (downstream AUC) BUT activation memory grows ~linearly with
W (all W per-snapshot propagations kept for backprop) -> OOMs at large W. This is the inherent
tension zord's memory scheduler resolves (stream/distribute older snapshots -> reach the higher-W,
higher-AUC model the in-core baseline cannot). Here we MEASURE the curve + the OOM cliff.
  python scripts/train_linkpred_window.py superuser --periods 48 --windows 1,2,4,8,16,24,32,48
"""
import argparse, time
import numpy as np
import torch

from zord.datasets import load


def norm_adj(src, dst, n, dev):
    if src.size == 0:
        return torch.sparse_coo_tensor(torch.zeros(2, 1, dtype=torch.long, device=dev),
                                       torch.zeros(1, device=dev), (n, n)).coalesce()
    i = torch.tensor(np.concatenate([src, dst]), dtype=torch.long)
    j = torch.tensor(np.concatenate([dst, src]), dtype=torch.long)
    A = torch.sparse_coo_tensor(torch.stack([i, j]), torch.ones(i.shape[0]), (n, n)).coalesce().to(dev)
    deg = torch.sparse.sum(A, 1).to_dense().clamp(min=1.0)
    return torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]], (n, n)).coalesce()


def auc_rank(pos, neg):
    s = torch.cat([pos, neg]); y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = torch.argsort(s); ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=s.dtype)
    p, n = pos.numel(), neg.numel()
    return ((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--periods", type=int, default=48)
    ap.add_argument("--windows", default="1,2,4,8,16,24,32,48"); ap.add_argument("--gamma", type=float, default=0.85)
    a = ap.parse_args()
    dev = "cuda:0"
    g = load(a.dataset).sort_by_time()
    N, E, F = g.num_nodes, g.num_edges, a.feat_dim
    split = int(0.7 * E)
    # training message edges split into `periods` chronological snapshots
    P = a.periods
    bnd = np.linspace(0, split, P + 1).astype(int)
    train_deg = np.bincount(np.concatenate([g.src[:split], g.dst[:split]]), minlength=N)
    warm = train_deg > 0
    tm = warm[g.src[split:]] & warm[g.dst[split:]]
    te_u = torch.tensor(g.src[split:][tm], device=dev); te_v = torch.tensor(g.dst[split:][tm], device=dev)
    name = torch.cuda.get_device_name(0)
    print(f"WINDOW dataset={g.name} gpu='{name}' N={N} E={E} periods={P} warm_test={int(tm.sum())}")

    for W in [int(x) for x in a.windows.split(",")]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        try:
            # the last W training periods (most recent history)
            use = list(range(P - W, P))
            As = [norm_adj(g.src[bnd[k]:bnd[k + 1]], g.dst[bnd[k]:bnd[k + 1]], N, dev) for k in use]
            emb = torch.nn.Parameter(torch.randn(N, F, device=dev) * 0.1)
            W1 = torch.nn.Parameter(torch.randn(F, F, device=dev) * (1 / F ** 0.5))
            W2 = torch.nn.Parameter(torch.randn(F, F, device=dev) * (1 / F ** 0.5))
            opt = torch.optim.Adam([emb, W1, W2], lr=0.01)
            bce = torch.nn.BCEWithLogitsLoss()
            tp_u = torch.tensor(g.src[:split], device=dev); tp_v = torch.tensor(g.dst[:split], device=dev)
            K = min(50_000, split)

            def fwd():
                acc = torch.zeros(N, F, device=dev)
                for wi, Aw in enumerate(As):                # all W kept in autograd graph -> mem ~ W
                    acc = acc + (a.gamma ** (W - 1 - wi)) * torch.sparse.mm(Aw, emb)
                h = torch.relu(acc @ W1)
                return torch.sparse.mm(As[-1], h) @ W2

            t0 = time.time()
            for ep in range(a.epochs):
                opt.zero_grad(); H = fwd()
                idx = torch.randint(0, split, (K,), device=dev)
                pu, pv = tp_u[idx], tp_v[idx]
                nu = torch.randint(0, N, (K,), device=dev); nv = torch.randint(0, N, (K,), device=dev)
                ps = (H[pu] * H[pv]).sum(-1); ns = (H[nu] * H[nv]).sum(-1)
                loss = bce(ps, torch.ones_like(ps)) + bce(ns, torch.zeros_like(ns))
                loss.backward(); opt.step()
            torch.cuda.synchronize(); tr = time.time() - t0
            with torch.no_grad():
                H = fwd()
                m = min(50_000, te_u.shape[0]); sel = torch.randint(0, te_u.shape[0], (m,), device=dev)
                nu = torch.randint(0, N, (m,), device=dev); nv = torch.randint(0, N, (m,), device=dev)
                auc = auc_rank((H[te_u[sel]] * H[te_v[sel]]).sum(-1), (H[nu] * H[nv]).sum(-1))
            peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 3
            print(f"  W={W:<3} test_AUC={auc:.4f} peak_HBM={peak:5.2f}GB train_s={tr:5.1f}")
            del As, emb, W1, W2, opt
        except RuntimeError as e:
            msg = "OOM" if "out of memory" in str(e).lower() else type(e).__name__
            print(f"  W={W:<3} FAILED ({msg}) -- in-core cannot hold this window (zord would stream/distribute it)")
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
