#!/bin/bash
# Live GPU-OOM probe on the 32GB RTX5000 (logical device 2). Submit:
#   sbatch --gres=gpu:rtx_5000:1 scripts/oom_probe.sbatch
# Expect: hash (equal split, ~N/3 nodes) -> CUDA_OOM ; zord (memory-aware) -> OK.
#SBATCH --partition=bigTiger
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=$ZORD_DATA/results/oom_probe_%j.out
set -e
source <conda.sh>
conda activate $PROJECT/hkenv
export PYTHONUNBUFFERED=1
cd $ZORD_DATA/repo
echo "### wiki-talk window=24 feat=1024 on the 32GB card (logical dev 2) ###"
for S in hash caphash fennel zord; do
  python scripts/oom_probe.py wiki-talk --scheme $S --device 2 --feat-dim 1024 --window 24 || true
done
echo "### DONE ###"
