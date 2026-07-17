#!/bin/bash
# End-to-end 2-node iteration-time benchmark (zord vs hash placement). Submit:
#   sbatch --gres=gpu:rtx_5000:1 scripts/end_to_end.sbatch
#SBATCH --partition=bigTiger
#SBATCH --nodes=2 --ntasks-per-node=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=$ZORD_DATA/results/e2e_%j.out
source <conda.sh>
conda activate $PROJECT/hkenv
export PYTHONUNBUFFERED=1
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_NODELIST} | head -1)
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID} % 10000))
cd $ZORD_DATA/repo
for S in hash zord; do
  echo "########## scheme=$S ##########"
  srun python scripts/end_to_end.py askubuntu --scheme $S --feat-dim 256 --steps 20
done
echo "########## E2E DONE ##########"
