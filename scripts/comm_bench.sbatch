#!/bin/bash
# 2-node inter-node gloo bandwidth microbench. Submit:
#   sbatch --gres=gpu:rtx_5000:1 scripts/comm_bench.sbatch
#SBATCH --partition=bigTiger
#SBATCH --nodes=2 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=$ZORD_DATA/results/comm_bench_%j.out
source <conda.sh>
conda activate $PROJECT/hkenv
export PYTHONUNBUFFERED=1
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -1)
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID} % 10000))
cd $ZORD_DATA/repo
srun python scripts/comm_bench.py
