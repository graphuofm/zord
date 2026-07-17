> Anonymous artifact for a SIGMOD 2027 submission (double-anonymous review).
> Code (C++17 kernels + Python optimizer/harnesses) and the raw measurement
> CSVs behind every table in the paper (`results/`). Paths and cluster names
> are genericized; set `ZORD_DATA`/`ZORD_GRAPH_BIN` for your environment.

# zord

**Attribute-aware partitioning and placement for temporal GNN training on GPU clusters.**

The bytes that dominate GNN training are the *attributes* — node and edge feature
tensors outweigh the graph structure by one to two orders of magnitude — yet
standard graph partitioners balance vertex counts and cut edge counts, not bytes.
On real data this mismatch leaves one GPU holding 20–31% more feature memory than
its peers, and at large feature widths it is the difference between a job that
runs and a job that OOMs.

zord puts bytes into the objective itself:

- **Byte-weighted partitioning kernels** (C++17/OpenMP, no external partitioner
  dependency): multilevel min-cut and single-pass streaming, both accepting
  per-vertex weights (feature bytes `F_v`) and per-edge weights (edge-feature
  bytes `F_e`), plus heterogeneous per-device capacity ratios. The streaming
  kernel partitions a **10⁹-edge graph in ~2 minutes** on one node where serial
  METIS fails outright.
- **A warm-start polish mode** that refines *any* existing partition (including
  METIS's) with boundary FM under the byte objective.
- **Axis selection**: a measured-cost model that chooses among node-parallel,
  feature-parallel (tensor/column), hybrid, and temporal decompositions per input
  and per cluster, with feasibility as a hard gate.
- **Process-only contract**: for fixed data and model, execution results equal
  single-device results (certified per run, fp32 summation-order tolerance).
  zord optimizes time, memory, and feasibility — never accuracy.

## Quick start

```bash
# build the native kernels (g++ >= 9, OpenMP)
make -C src/zord/cpp

# partition an edge list (binary: i64 N, i64 M, i32 src[M], i32 dst[M])
PYTHONPATH=src python3 -m zord.cli partition graph.bin -D 8 \
    --method auto --fv-file fv.npy --fe-dim 172 --cap-gb 32 \
    --out part.npy --metrics metrics.json

# two-level partitioning for large clusters (nodes x GPUs-per-node)
PYTHONPATH=src python3 -m zord.cli partition graph.bin --hierarchy 128x8 \
    --method auto --out part.npy

# full planning pipeline on a registered dataset
PYTHONPATH=src python3 -m zord.cli plan wiki-talk --feat 256 --window 8
```

`--method auto` picks the scale-appropriate kernel (multilevel min-cut up to
~2·10⁷ edges, O(E) streaming above). All methods accept attribute weights; the
defaults reproduce classical structural partitioning bit-for-bit.

## Layout

| path | contents |
|---|---|
| `src/zord/cpp/` | native kernels: `multilevel`, `streaming`, `spectral`, `arrange`, `graph_stats`, `rmat`, … |
| `src/zord/partition/` | partitioning + placement engine, cost models, kernel bindings |
| `src/zord/schedule/` | planner, axis/K selection, distributed coordinator, online/dynamic scheduling |
| `src/zord/runtime/` | tiering executor, memory control loop, buffer pool, feature recombination |
| `scripts/` | experiment harnesses (partition campaigns, multi-GPU step benchmarks) |
| `tests/` | pytest suite (`PYTHONPATH=src python3 -m pytest tests/ -q`) |
| `docs/ARCHITECTURE.md` | module-by-module architecture map |
| `docs/THEORY.md` | the space–time cut duality and cost model |

## Status

Research prototype under active development. The partitioning kernels, CLI,
planner, and single-node multi-GPU execution paths are tested on a real
heterogeneous SLURM cluster (H100 / RTX 6000 Ada / RTX 5000 Ada); multi-node
execution and framework adapters (PyG/DGL) are in progress. Interfaces may
change.

## License

MIT — see [LICENSE](LICENSE).
