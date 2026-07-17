#!/bin/bash
# Ultra-scale partitioner comparison (stackoverflow ~2.6M nodes, 63.5M edges).
# CPU-only work but request a small GPU so it schedules on bigTiger. Submit:
#   sbatch --gres=gpu:rtx_5000:1 scripts/compare.sbatch
#SBATCH --partition=bigTiger
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=192G
#SBATCH --time=01:30:00
#SBATCH --output=$ZORD_DATA/results/compare_so_%j.out
set -e
source <conda.sh>
conda activate $PROJECT/hkenv
export PYTHONUNBUFFERED=1
cd $ZORD_DATA/repo
for W in 8 16; do
  echo "########## stackoverflow window=$W ##########"
  python scripts/compare_partitioners.py stackoverflow --feat-dim 1024 --window $W \
     --out $ZORD_DATA/results/cmp_so_w$W.json
done
echo "########## DONE ##########"
