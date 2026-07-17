#!/usr/bin/env python
"""Measure TEMPORAL REUSE opportunity across MANY datasets (small->ultra, different types) -- the
scheduler's reuse_frac lever. Vectorized so it scales; 2-hop uses scipy sparse matvec (guarded by
size). Different temporal-graph TYPES churn differently (bursty social vs steady QA vs trust), so
the reuse distribution itself is a finding. CPU-only.

  reuse@1hop = fraction of active nodes with NO incident edge this snapshot (1-hop nbr set unchanged)
  reuse@2hop = fraction with no edge change within 2 hops (full 2-layer GraphSAGE embedding reusable)
  python scripts/reuse_probe.py --datasets collegemsg,email-eu,bitcoin-otc,mathoverflow,askubuntu,superuser,wiki-talk --snapshots 8
"""
import argparse
import numpy as np

from zord.datasets import load

TWO_HOP_MAX_E = 8_000_000          # skip exact 2-hop above this cumulative-edge count (too heavy)


def reuse_for(name, S):
    try:
        g = load(name).sort_by_time()
    except Exception as e:
        print(f"REUSE dataset={name} SKIP ({type(e).__name__}: {str(e)[:50]})")
        return
    N, E = g.num_nodes, g.num_edges
    src, dst = g.src, g.dst
    bnd = np.linspace(0, E, S + 1).astype(int)
    active = np.zeros(N, dtype=bool)
    r1, r2 = [], []
    have_scipy = True
    try:
        import scipy.sparse as sp
    except Exception:
        have_scipy = False
    print(f"REUSE dataset={name} N={N} E={E} snapshots={S} 2hop={'on' if have_scipy else 'no-scipy'}")
    for s in range(S):
        lo, hi = bnd[s], bnd[s + 1]
        su, sv = src[lo:hi], dst[lo:hi]
        active_before = active.copy()
        nb = int(active_before.sum())
        touched = np.zeros(N, dtype=bool)
        touched[su] = True; touched[sv] = True
        if s > 0 and nb > 0:
            reuse1 = 1.0 - (touched & active_before).sum() / nb
            r1.append(reuse1)
            reuse2 = float("nan")
            if have_scipy and hi <= TWO_HOP_MAX_E:
                # cumulative symmetric adjacency up to hi; one-hop spread of touched-ness
                a = np.concatenate([src[:hi], dst[:hi]]); b = np.concatenate([dst[:hi], src[:hi]])
                A = sp.csr_matrix((np.ones(a.size, dtype=np.float32), (a, b)), shape=(N, N))
                nbr_touched = (A.dot(touched.astype(np.float32)) > 0)
                affected = touched | nbr_touched
                reuse2 = 1.0 - (affected & active_before).sum() / nb
                r2.append(reuse2)
            print(f"  snap {s}: active_before={nb:>9} new_edges={hi-lo:>9} "
                  f"reuse@1hop={reuse1:.3f} reuse@2hop={reuse2:.3f}")
        active[su] = True; active[sv] = True
    if r1:
        m2 = f"{np.mean(r2):.3f}" if r2 else "n/a"
        print(f"  >>> {name}: MEAN reuse@1hop={np.mean(r1):.3f} reuse@2hop={m2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="askubuntu")
    ap.add_argument("--snapshots", type=int, default=8)
    a = ap.parse_args()
    for name in a.datasets.split(","):
        reuse_for(name.strip(), a.snapshots)
        print()


if __name__ == "__main__":
    main()
