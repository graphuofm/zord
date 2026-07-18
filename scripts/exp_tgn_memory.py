#!/usr/bin/env python
"""E2E training of a MEMORY-BASED temporal GNN (TGN-style) on D GPUs.

Model (a deterministic snapshot-batched TGN variant):
  - per-vertex memory M[v] in R^{Fm}, initialised to zeros each epoch;
  - chronological history split into B equal edge batches; at batch t the
    memory is first updated IN-GRAPH from batch t-1's interactions
    (identity messages [M_src || X_src], per-vertex mean aggregation, a
    shared GRUCell), exactly TGN's "memory updated with messages from
    previous batches" scheme, so the GRU receives gradients through the
    batch loss; memory is detached between batches;
  - embedding: z = relu([X || M] W_in + A_{t-1} [X || M] W_nb), where
    A_{t-1} is the mean-normalised adjacency of the previous batch (the
    recency neighbourhood); dot-product decoder, BCE with one seeded
    negative per positive, synchronous SGD after every batch;
  - evaluation: the last 30% of edges are scored streaming (no gradient),
    memory keeps updating, rank-based AUC.

Deliberate simplifications vs. the full TGN (disclosed wherever used):
identity message function, recency neighbourhood instead of sampled
temporal neighbours (sampling would break run-to-run equivalence), no
time encoding. The memory mechanism itself (GRU state evolving over the
stream, shipped across devices when edges cross the cut) is exact.

Determinism/distribution: every per-vertex reduction (neighbour mean,
message mean) is a CSR sparse.mm on the vertex's owner device over a
canonically ordered edge list, so the floating-point accumulation order
is independent of the partition; remote [X || M] rows are fetched with
autograd-safe index_select (gradients flow back to the owner device).
Certificate: per-batch loss trajectory and final AUC against a
single-device run with identical seeds.
"""
import os, sys, csv, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_gpu_e2e as base
from exp_tgn_linkpred import auc_rank


def run_tgnmem(src, dst, N, part, D, F, Fm, seed, epochs, lr, nbatch, dev_kind="cuda"):
    import torch
    devs = [torch.device(f"cuda:{i}" if dev_kind == "cuda" else "cpu") for i in range(D)]
    cuda = dev_kind == "cuda"
    part = np.asarray(part, np.int64)
    M = src.size
    ncut = int(M * 0.7)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, F)).astype(np.float32)
    Hdim = Fm  # embedding width = memory width

    locals_g, lidx = [], np.full(N, -1, np.int64)
    for d in range(D):
        g = np.nonzero(part == d)[0]; locals_g.append(g); lidx[g] = np.arange(g.size)
    Xd = [torch.from_numpy(X[locals_g[d]]).to(devs[d]) for d in range(D)]

    # replicated parameters (identical init on every device)
    g0 = torch.Generator().manual_seed(seed + 77)
    W_in0 = (torch.randn(F + Fm, Hdim, generator=g0) / (F + Fm) ** 0.5)
    W_nb0 = (torch.randn(F + Fm, Hdim, generator=g0) / (F + Fm) ** 0.5)
    gru0 = torch.nn.GRUCell(F + Fm, Fm)
    with torch.no_grad():
        for p in gru0.parameters():
            p.copy_(torch.randn(p.shape, generator=g0) * 0.1)
    params = []
    for d in range(D):
        W_in = W_in0.clone().to(devs[d]).requires_grad_(True)
        W_nb = W_nb0.clone().to(devs[d]).requires_grad_(True)
        gru = torch.nn.GRUCell(F + Fm, Fm).to(devs[d])
        with torch.no_grad():
            for p, p0 in zip(gru.parameters(), gru0.parameters()):
                p.copy_(p0.to(devs[d]))
        for p in gru.parameters():
            p.requires_grad_(True)
        params.append([W_in, W_nb] + list(gru.parameters()))
    grus = [torch.nn.GRUCell(F + Fm, Fm).to(devs[d]) for d in range(D)]
    for d in range(D):  # rebind gru modules to the leaf parameter tensors
        grus[d].weight_ih = torch.nn.Parameter(params[d][2]); grus[d].weight_hh = torch.nn.Parameter(params[d][3])
        grus[d].bias_ih = torch.nn.Parameter(params[d][4]); grus[d].bias_hh = torch.nn.Parameter(params[d][5])
        params[d] = [params[d][0], params[d][1]] + list(grus[d].parameters())

    def gather_rows(tables, verts, to_dev):
        """Fetch rows for global vertex ids from per-device tables (autograd-safe)."""
        pieces, orders = [], []
        for o_ in range(D):
            sel = np.nonzero(part[verts] == o_)[0]
            if sel.size:
                idx = torch.from_numpy(lidx[verts[sel]]).to(devs[o_])
                pieces.append(tables[o_].index_select(0, idx).to(to_dev))
                orders.append(sel)
        inv = np.argsort(np.concatenate(orders), kind="stable")
        return torch.cat(pieces, 0).index_select(0, torch.from_numpy(inv).to(to_dev))

    def mean_csr(tgt, srcv, tables, d):
        """Deterministic per-vertex mean over (tgt <- srcv) pairs owned by device d.
        Returns (local_target_index_tensor, mean_rows). Pairs are pre-sorted globally
        by (tgt, position), so accumulation order is partition-independent."""
        m = part[tgt] == d
        gt, gs = tgt[m], srcv[m]
        if gt.size == 0:
            return None, None
        ut, tinv = np.unique(gt, return_inverse=True)
        rows = gather_rows(tables, gs, devs[d])
        deg = np.bincount(tinv, minlength=ut.size).astype(np.float32)
        o = np.argsort(tinv, kind="stable")
        ei = tinv[o]
        crow = np.zeros(ut.size + 1, np.int64); np.add.at(crow, ei + 1, 1); np.cumsum(crow, out=crow)
        vals = (1.0 / deg)[ei].astype(np.float32)
        A = torch.sparse_csr_tensor(torch.from_numpy(crow).to(devs[d]),
                                    torch.from_numpy(o).to(devs[d]),
                                    torch.from_numpy(vals).to(devs[d]),
                                    size=(ut.size, gt.size))
        return torch.from_numpy(lidx[ut]).to(devs[d]), torch.sparse.mm(A, rows)

    P_eval = M - ncut
    bnd = np.linspace(0, ncut, nbatch + 1).astype(np.int64)
    ebnd = np.linspace(ncut, M, max(2, nbatch // 3) + 1).astype(np.int64)
    negs = {}
    for e in range(epochs):
        for b in range(nbatch):
            n_b = bnd[b + 1] - bnd[b]
            negs[(e, b)] = np.random.default_rng(seed + 5000 + e * 1000 + b).integers(0, N, n_b)
    for b in range(len(ebnd) - 1):
        negs[("ev", b)] = np.random.default_rng(seed + 900000 + b).integers(0, N, ebnd[b + 1] - ebnd[b])

    def encode(Mem, pu_, pv_):
        """Embed with recency adjacency (pu_, pv_ = previous batch edges)."""
        tabs = [torch.cat([Xd[d], Mem[d]], 1) for d in range(D)]
        Z = []
        for d in range(D):
            z = tabs[d] @ params[d][0]
            if pu_ is not None and pu_.size:
                t2, mrows = mean_csr(np.concatenate([pu_, pv_]), np.concatenate([pv_, pu_]), tabs, d)
                if t2 is not None:
                    nb = torch.zeros(locals_g[d].size, Hdim, device=devs[d])
                    nb = nb.index_copy(0, t2, mrows @ params[d][1])
                    z = z + nb
            Z.append(torch.relu(z))
        return Z

    def mem_update(Mem, pu_, pv_):
        """In-graph GRU update of memory from edges (pu_, pv_); returns new memory."""
        tabs = [torch.cat([Xd[d], Mem[d]], 1) for d in range(D)]
        out = []
        for d in range(D):
            tgt = np.concatenate([pu_, pv_]); sv = np.concatenate([pv_, pu_])
            t2, msg = mean_csr(tgt, sv, tabs, d)
            if t2 is None:
                out.append(Mem[d])
            else:
                upd = grus[d](msg, Mem[d].index_select(0, t2))
                out.append(Mem[d].index_copy(0, t2, upd))
        return out

    def score_edges(Z, eu, ev, nv):
        sp_all, sn_all, loss = [], [], 0.0
        n_tot = 2.0 * eu.size
        for d in range(D):
            m = part[eu] == d
            if not m.any():
                continue
            zu = Z[d].index_select(0, torch.from_numpy(lidx[eu[m]]).to(devs[d]))
            zp = gather_rows(Z, ev[m], devs[d])
            zn = gather_rows(Z, nv[m], devs[d])
            sp = (zu * zp).sum(-1); sn = (zu * zn).sum(-1)
            ld = -(torch.nn.functional.logsigmoid(sp).sum()
                   + torch.nn.functional.logsigmoid(-sn).sum()) / n_tot
            yield d, ld, sp, sn

    losses, times = [], []
    if cuda:
        for d in range(D):
            torch.cuda.reset_peak_memory_stats(devs[d])
    for e in range(epochs):
        Mem = [torch.zeros(locals_g[d].size, Fm, device=devs[d]) for d in range(D)]
        prev = None
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        for b in range(nbatch):
            eu, ev = src[bnd[b]:bnd[b + 1]], dst[bnd[b]:bnd[b + 1]]
            for d in range(D):
                for p in params[d]:
                    if p.grad is not None: p.grad = None
            Mn = mem_update(Mem, prev[0], prev[1]) if prev is not None else Mem
            Z = encode(Mn, prev[0] if prev else None, prev[1] if prev else None)
            tot = 0.0
            for d, ld, sp, sn in score_edges(Z, eu, ev, negs[(e, b)]):
                ld.to(devs[0]).backward(retain_graph=True)
                tot += float(ld.detach().cpu())
            with torch.no_grad():
                for pi in range(len(params[0])):
                    gs = sum((params[d][pi].grad.to(devs[0]) if params[d][pi].grad is not None
                              else torch.zeros_like(params[0][pi], device=devs[0])) for d in range(D))
                    for d in range(D):
                        params[d][pi] -= lr * gs.to(devs[d])
            Mem = [mn.detach() for mn in Mn]
            prev = (eu, ev)
            losses.append(tot)
        if cuda: torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    # streaming evaluation on the last 30%
    sc_p, sc_n = [], []
    with torch.no_grad():
        for b in range(len(ebnd) - 1):
            eu, ev = src[ebnd[b]:ebnd[b + 1]], dst[ebnd[b]:ebnd[b + 1]]
            Mn = mem_update(Mem, prev[0], prev[1]) if prev is not None else Mem
            Z = encode(Mn, prev[0] if prev else None, prev[1] if prev else None)
            ordbuf = {}
            for d, ld, sp, sn in score_edges(Z, eu, ev, negs[("ev", b)]):
                m = np.nonzero(part[eu] == d)[0]
                ordbuf[d] = (m, sp.detach().cpu().numpy(), sn.detach().cpu().numpy())
            om = np.concatenate([v[0] for v in ordbuf.values()])
            inv = np.argsort(om, kind="stable")
            sc_p.append(np.concatenate([v[1] for v in ordbuf.values()])[inv])
            sc_n.append(np.concatenate([v[2] for v in ordbuf.values()])[inv])
            Mem = [mn.detach() for mn in Mn]
            prev = (eu, ev)
    peak = [round(torch.cuda.max_memory_allocated(devs[d]) / 2**20, 1) for d in range(D)] if cuda else [0.0]
    auc = auc_rank(np.concatenate(sc_p), np.concatenate(sc_n))
    return dict(epoch_ms=round(float(np.median(times)), 2), losses=losses, auc=auc, peak_mb=peak)


def run_cell(name, method, D, F, Fm, seed, epochs, lr, nbatch, dev_kind="cuda", ref_cache=None):
    src, dst, N = base.load_cached(name)
    fv = np.full(N, F, np.int64)
    t0 = time.perf_counter()
    part = base.make_partition(method, src, dst, N, D, fv) if D > 1 else np.zeros(N, np.int64)
    part_s = time.perf_counter() - t0
    r = run_tgnmem(src, dst, N, part, D, F, Fm, seed, epochs, lr, nbatch, dev_kind)
    key = (name, F, Fm, epochs, nbatch)
    if ref_cache is not None and key in ref_cache:
        ref = ref_cache[key]
    else:
        ref = run_tgnmem(src, dst, N, np.zeros(N, np.int64), 1, F, Fm, seed, epochs, lr, nbatch, dev_kind)
        if ref_cache is not None: ref_cache[key] = ref
    dev = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(r["losses"], ref["losses"]))
    return dict(dataset=name, method=method, D=D, F=F, Fm=Fm, nbatch=nbatch,
                partition_s=round(part_s, 2), epoch_ms=r["epoch_ms"],
                ref_epoch_ms=ref["epoch_ms"], peak_mb_max=max(r["peak_mb"]),
                auc=round(r["auc"], 6), ref_auc=round(ref["auc"], 6),
                auc_diff=round(abs(r["auc"] - ref["auc"]), 8),
                loss_dev=f"{dev:.3e}", same=bool(dev <= 1e-3),
                final_loss=round(r["losses"][-1], 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="jodie-wikipedia,jodie-mooc,mathoverflow,askubuntu,superuser,wiki-talk,stackoverflow")
    ap.add_argument("--methods", default="hash,metis-aware,zord-polish")
    ap.add_argument("--dlist", default="8")
    ap.add_argument("--feat-dim", type=int, default=64)
    ap.add_argument("--mem-dim", type=int, default=64)
    ap.add_argument("--nbatch", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev_kind = "cpu" if a.cpu else "cuda"
    rows, ref_cache = [], {}
    for name in a.datasets.split(","):
        for D in [int(x) for x in a.dlist.split(",")]:
            for method in a.methods.split(","):
                try:
                    r = run_cell(name, method, D, a.feat_dim, a.mem_dim, a.seed,
                                 a.epochs, a.lr, a.nbatch, dev_kind, ref_cache=ref_cache)
                    r["status"] = "OK"
                except Exception as e:
                    r = dict(dataset=name, method=method, D=D, status=f"FAIL:{type(e).__name__}:{str(e)[:90]}")
                print("[cell]", json.dumps(r), flush=True)
                rows.append(r)
    if a.out and rows:
        keys = sorted({k for r in rows for k in r})
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)
        print("[csv]", a.out, flush=True)


if __name__ == "__main__":
    main()
