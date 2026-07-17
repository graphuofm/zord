#!/usr/bin/env python
"""Dynamic EDGE-SIGN classification on a temporal SIGNED graph (bitcoin-otc/alpha):
predict whether an interaction is TRUST (+) or DISTRUST (-). The distrust class is RARE
(~11%) -> a TAIL-SENSITIVE task. Contrast with link-pred (a ranking task that tolerated
int8 at zero loss): here we measure how the precision/discard budget (M5) must SHRINK when
the metric is the minority class. AP_minority is the metric that matters; AUC is overall.
  python scripts/train_edgesign.py bitcoin-otc --feat-dim 128 --epochs 300
"""
import argparse, time
import numpy as np
import torch

from zord.datasets import load


def norm_adj(src, dst, n, dev):                      # symmetric, connectivity only (sign is the LABEL)
    i = torch.tensor(np.concatenate([src, dst]), dtype=torch.long)
    j = torch.tensor(np.concatenate([dst, src]), dtype=torch.long)
    A = torch.sparse_coo_tensor(torch.stack([i, j]), torch.ones(i.shape[0]), (n, n)).coalesce().to(dev)
    deg = torch.sparse.sum(A, 1).to_dense().clamp(min=1.0)
    return torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]], (n, n)).coalesce()


def auc_rank(pos, neg):                               # Mann-Whitney AUC
    s = torch.cat([pos, neg]); y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = torch.argsort(s); ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=s.dtype)
    p, n = pos.numel(), neg.numel()
    return ((ranks[y == 1].sum() - p * (p + 1) / 2) / (p * n)).item()


def average_precision(score, y):                      # AP of the positive (=minority/distrust) class
    order = torch.argsort(score, descending=True)
    y = y[order].float()
    tp = torch.cumsum(y, 0)
    prec = tp / torch.arange(1, y.numel() + 1, device=y.device)
    rec_gain = y / y.sum().clamp(min=1)
    return (prec * rec_gain).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--lr", type=float, default=0.01)
    a = ap.parse_args()
    dev = "cuda:0"
    g = load(a.dataset).sort_by_time()
    N, E, F = g.num_nodes, g.num_edges, a.feat_dim
    y_all = (g.w < 0).astype(np.float32)              # POSITIVE class = DISTRUST (the rare one)
    split = int(0.7 * E)
    A = norm_adj(g.src[:split], g.dst[:split], N, dev)
    def edge_t(lo, hi):
        return (torch.tensor(g.src[lo:hi], device=dev), torch.tensor(g.dst[lo:hi], device=dev),
                torch.tensor(y_all[lo:hi], device=dev))
    tr_u, tr_v, tr_y = edge_t(0, split)
    te_u, te_v, te_y = edge_t(split, E)
    print(f"edge-sign {g.name}: E={E} train_distrust={tr_y.mean():.3f} test_distrust={te_y.mean():.3f} "
          f"test_n={te_y.numel()}")

    emb = torch.nn.Parameter(torch.randn(N, F, device=dev) * 0.1)
    W1 = torch.nn.Parameter(torch.randn(F, F, device=dev) * (1 / F ** 0.5))
    W2 = torch.nn.Parameter(torch.randn(F, F, device=dev) * (1 / F ** 0.5))
    theta = torch.nn.Parameter(torch.zeros(F, device=dev)); b = torch.nn.Parameter(torch.zeros(1, device=dev))
    opt = torch.optim.Adam([emb, W1, W2, theta, b], lr=a.lr)
    pos_w = ((tr_y == 0).sum() / (tr_y == 1).sum().clamp(min=1)).clamp(max=50)   # reweight rare class
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def fwd():
        h = torch.relu(torch.sparse.mm(A, emb) @ W1)
        return torch.sparse.mm(A, h) @ W2

    def score(H, u, v):
        return ((H[u] * H[v]) * theta).sum(-1) + b

    t0 = time.time()
    for ep in range(a.epochs):
        opt.zero_grad(); H = fwd()
        loss = bce(score(H, tr_u, tr_v), tr_y)
        loss.backward(); opt.step()
    torch.cuda.synchronize(); train_s = time.time() - t0

    def quantize(H, bits):
        if bits >= 32: return H
        if bits == 16: return H.half().float()
        s = H.abs().amax() / 127.0 + 1e-12
        return (H / s).round().clamp(-127, 127) * s

    with torch.no_grad():
        H = fwd()
        for bits in (32, 16, 8):
            Hq = quantize(H, bits)
            sc = score(Hq, te_u, te_v)
            auc = auc_rank(sc[te_y == 1], sc[te_y == 0])
            ap = average_precision(sc, te_y)
            print(f"EDGESIGN dataset={g.name} feat={F} epochs={a.epochs} bits={bits} "
                  f"bytes/elem={bits//8} test_AUC={auc:.4f} AP_distrust={ap:.4f} train_s={train_s:.1f}")


if __name__ == "__main__":
    main()
