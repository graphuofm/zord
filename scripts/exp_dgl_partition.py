#!/usr/bin/env python
"""DGL (DistDGL's shipped partitioning pipeline) vs. zord kernels, same inputs.

DistDGL's partition step runs on CPU via dgl.distributed.partition_graph
(METIS assignment + halo construction + per-part serialization). This is the
layer zord competes at, so the comparison is direct and runs with the
CPU build of DGL. For each graph we time and score:
  - dgl-metis-assign: dgl.metis_partition_assignment(g, D) only
    (balance_edges default), cut/balance from the returned assignment;
  - dgl-partition-graph: the full shipped pipeline
    dgl.distributed.partition_graph(...) wall-clock (writes to node-local
    scratch, deleted afterwards);
  - our methods (metis-aware, zord-mc-aware, zord-stream-mem) re-measured in
    the same process for a same-campaign comparison.
Cut = number of edges with endpoints in different parts (one count per
undirected edge, mirroring exp_kmp_baseline.metrics); balance under uniform
F so count balance equals byte balance. SIGALRM guards each call.
"""
import os, sys, csv, json, time, shutil, signal, argparse, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_gpu_e2e as base
from exp_kmp_baseline import metrics, Timeout, _alarm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="collegemsg,bitcoin-otc,jodie-mooc,jodie-wikipedia,jodie-reddit,mathoverflow,askubuntu,superuser,wiki-talk,stackoverflow")
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--scratch", default="/tmp/zord_dglpart")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import dgl
    import torch
    signal.signal(signal.SIGALRM, _alarm)
    rows = []
    for name in a.datasets.split(","):
        src, dst, N = base.load_cached(name)
        fv = np.full(N, a.feat_dim, np.int64)
        fvb = fv * 4
        # undirected simple graph, the same view our kernels and Metis cut
        g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=N)
        g = dgl.to_bidirected(dgl.to_simple(g))
        cells = []
        t0 = time.perf_counter()
        try:
            signal.alarm(a.timeout)
            assign = dgl.metis_partition_assignment(g, a.D)
            signal.alarm(0)
            part = assign.numpy().astype(np.int64)
            cells.append(dict(method="dgl-metis-assign",
                              part_s=round(time.perf_counter() - t0, 2), status="OK",
                              **metrics(part, src, dst, N, fvb)))
        except Timeout:
            cells.append(dict(method="dgl-metis-assign", status=f"TIMEOUT>{a.timeout}s"))
        except Exception as e:
            signal.alarm(0)
            cells.append(dict(method="dgl-metis-assign", status=f"FAIL:{type(e).__name__}:{str(e)[:80]}"))
        out_path = os.path.join(a.scratch, name)
        shutil.rmtree(out_path, ignore_errors=True)
        t0 = time.perf_counter()
        try:
            signal.alarm(a.timeout)
            dgl.distributed.partition_graph(g, name, a.D, out_path)
            signal.alarm(0)
            cells.append(dict(method="dgl-partition-graph",
                              part_s=round(time.perf_counter() - t0, 2), status="OK"))
        except Timeout:
            cells.append(dict(method="dgl-partition-graph", status=f"TIMEOUT>{a.timeout}s"))
        except Exception as e:
            signal.alarm(0)
            cells.append(dict(method="dgl-partition-graph", status=f"FAIL:{type(e).__name__}:{str(e)[:80]}"))
        finally:
            shutil.rmtree(out_path, ignore_errors=True)
        for method in ("metis-aware", "zord-mc-aware", "zord-stream-mem"):
            t0 = time.perf_counter()
            try:
                signal.alarm(a.timeout)
                part = np.asarray(base.make_partition(method, src, dst, N, a.D, fv), np.int64)
                signal.alarm(0)
                cells.append(dict(method=method, part_s=round(time.perf_counter() - t0, 2),
                                  status="OK", **metrics(part, src, dst, N, fvb)))
            except Timeout:
                cells.append(dict(method=method, status=f"TIMEOUT>{a.timeout}s"))
            except Exception as e:
                signal.alarm(0)
                cells.append(dict(method=method, status=f"FAIL:{type(e).__name__}:{str(e)[:80]}"))
        for c in cells:
            row = dict(dataset=name, D=a.D, N=N, M=src.size, **c)
            print("[cell]", json.dumps(row), flush=True)
            rows.append(row)
    if a.out and rows:
        keys = sorted({k for r in rows for k in r})
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)
        print("[csv]", a.out, flush=True)


if __name__ == "__main__":
    main()
