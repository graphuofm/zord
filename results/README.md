# Raw measurement CSVs

Every number in the paper's tables comes from one of these files, produced by
the harnesses in `scripts/` on the evaluation cluster (three GPU tiers; see
paper Sec. 6). Selected mapping:

- `attr_partition_79134.csv` -> partitioning at scale (Table: E11)
- `e12_mag_151416.csv` -> byte skew on ogbn-mag (Table: E12)
- `cluster_archive/gpu_e2e_f1024.csv` -> feature-width-1024 frontier (E13)
- `cluster_archive/gpu_e2e_exp3.csv`, `gpu_e2e_wmix.csv`, `gpu_e2e_prof.csv` -> axis regimes (Table: E14)
- `gpu_e2e_polish.csv`, `gpu_e2e_pinned.csv` -> warm-start polish (Table: E15)
- `tgn_linkpred_151415.csv` -> end-to-end link-prediction training (E17)
- `cluster_archive/gpu_e2e_rtx_6000.csv` etc. -> multi-GPU step measurements (E5, E16)
