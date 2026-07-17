#!/usr/bin/env python
"""Minimal full-batch GraphSAGE LINK PREDICTION on a temporal graph -> a REAL learning
metric (test AUC) + wall time. Shows zord's training path yields learning, not just step
time. NOTE: single-GPU, structural (learnable node embeddings), temporal 70/30 split;
full distributed-to-convergence + accuracy-vs-baselines remains (HONEST_GAPS #1).
  python scripts/train_linkpred.py askubuntu --feat-dim 128 --epochs 50
"""
import argparse, time
import numpy as np
import torch

from zord.datasets import load


def norm_adj(src, dst, n, dev):
    i = torch.tensor(np.concatenate([src, dst]), dtype=torch.long)
    j = torch.tensor(np.concatenate([dst, src]), dtype=torch.long)
    A = torch.sparse_coo_tensor(torch.stack([i, j]), torch.ones(i.shape[0]), (n, n)).coalesce().to(dev)
    deg = torch.sparse.sum(A, 1).to_dense().clamp(min=1.0)
    return torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]], (n, n)).coalesce()


def auc_rank(pos, neg):                       # Mann-Whitney AUC, O((p+n)log(p+n))
    s = torch.cat([pos, neg]); y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = torch.argsort(s)
    ranks = torch.empty_like(s); ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=s.dtype)
    p, n = pos.numel(), neg.numel()
    return ((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--lr", type=float, default=0.01)
    a = ap.parse_args()
    dev = "cuda:0"
    g = load(a.dataset).sort_by_time()
    N, E, F = g.num_nodes, g.num_edges, a.feat_dim
    split = int(0.7 * E)
    A = norm_adj(g.src[:split], g.dst[:split], N, dev)          # message graph = past 70%
    tp_u = torch.tensor(g.src[:split], device=dev); tp_v = torch.tensor(g.dst[:split], device=dev)
    # WARM test edges only: both endpoints seen in the train graph (transductive; new
    # nodes have random embeddings = noise, so cold-start edges are excluded for a fair AUC).
    train_deg = np.bincount(np.concatenate([g.src[:split], g.dst[:split]]), minlength=N)
    warm = train_deg > 0
    tmask = warm[g.src[split:]] & warm[g.dst[split:]]
    te_u = torch.tensor(g.src[split:][tmask], device=dev); te_v = torch.tensor(g.dst[split:][tmask], device=dev)
    print(f"test edges total={int((g.num_edges-split))} warm={int(tmask.sum())}")

    emb = torch.nn.Parameter(torch.randn(N, F, device=dev) * 0.1)
    W1 = torch.nn.Parameter(torch.randn(F, F, device=dev) * (1 / F ** 0.5))
    W2 = torch.nn.Parameter(torch.randn(F, F, device=dev) * (1 / F ** 0.5))
    opt = torch.optim.Adam([emb, W1, W2], lr=a.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    def fwd():
        h = torch.relu(torch.sparse.mm(A, emb) @ W1)
        return torch.sparse.mm(A, h) @ W2

    K = min(100_000, tp_u.shape[0])
    t0 = time.time()
    for ep in range(a.epochs):
        opt.zero_grad(); H = fwd()
        idx = torch.randint(0, tp_u.shape[0], (K,), device=dev)
        pu, pv = tp_u[idx], tp_v[idx]
        nu = torch.randint(0, N, (K,), device=dev); nv = torch.randint(0, N, (K,), device=dev)
        ps = (H[pu] * H[pv]).sum(-1); ns = (H[nu] * H[nv]).sum(-1)
        loss = bce(ps, torch.ones_like(ps)) + bce(ns, torch.zeros_like(ns))
        loss.backward(); opt.step()
    torch.cuda.synchronize(); train_s = time.time() - t0

    def quantize(H, bits):                     # storage/comm precision of embeddings
        if bits >= 32:
            return H
        if bits == 16:
            return H.half().float()
        s = H.abs().amax() / 127.0 + 1e-12     # int8 per-tensor symmetric fake-quant
        return (H / s).round().clamp(-127, 127) * s

    with torch.no_grad():
        H = fwd()
        m = min(50_000, te_u.shape[0]); sel = torch.randint(0, te_u.shape[0], (m,), device=dev)
        nu = torch.randint(0, N, (m,), device=dev); nv = torch.randint(0, N, (m,), device=dev)
        for bits in (32, 16, 8):
            Hq = quantize(H, bits)
            pos = (Hq[te_u[sel]] * Hq[te_v[sel]]).sum(-1)
            neg = (Hq[nu] * Hq[nv]).sum(-1)
            print(f"LINKPRED dataset={g.name} feat={F} epochs={a.epochs} bits={bits} "
                  f"bytes/elem={bits//8} test_AUC={auc_rank(pos, neg):.4f} train_s={train_s:.1f}")


if __name__ == "__main__":
    main()
