#!/bin/bash
# STABLE single-node multi-GPU STRONG-SCALING study for distributed temporal-GNN training.
# ONE node, EIGHT GPUs (NVLink), held for the whole job; torchrun runs N=1,2,4,8 SEQUENTIALLY in
# the SAME allocation so one submission produces the full strong-scaling curve (and racks up GPU
# hours). Fixed total work per N => per-GPU work shrinks => step time should fall toward linear.
#
# SUBMIT (pick a GPU tier via GPUTYPE; works for h100 or rtx_6000):
#   GPUTYPE=h100     sbatch --gres=gpu:h100:8     scripts/dist_scaling.sbatch
#   GPUTYPE=rtx_6000 sbatch --gres=gpu:rtx_6000:8 scripts/dist_scaling.sbatch
# (the --gres on the command line wins; GPUTYPE only labels the output. If you don't pass --gres,
#  the #SBATCH default below requests 8 generic GPUs.)
#
#SBATCH --job-name=zord_dist_scaling
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=$ZORD_DATA/results/dist_scaling_%j.out

set -u

# ---- environment ----------------------------------------------------------
source <conda.sh>
conda activate $PROJECT/hkenv
export PYTHONUNBUFFERED=1
export ZORD_GRAPH_BIN=build/graph_algos
export PYTHONPATH=src:${PYTHONPATH:-}
cd $ZORD_DATA/repo

# ---- experiment knobs (override via env at submit time) -------------------
GPUTYPE=${GPUTYPE:-unknown}          # label only; h100 / rtx_6000 / rtx_5000
DATASET=${DATASET:-}                 # e.g. DATASET=wiki-talk for a real graph; empty => synthetic
SYNTH=${SYNTH:-big}                  # synthetic size: small | big | huge
FEAT=${FEAT:-128}
ITERS=${ITERS:-50}
MEMORY=${MEMORY:-0}                  # 1 => enable the GRUCell node-memory (TGN) variant
PART=${PART:-lpa}                    # vertex->device partition: lpa (default) | zord | hash
EXTRA=""
[ -n "$DATASET" ] && EXTRA="--dataset $DATASET" || EXTRA="--synthetic $SYNTH"
[ "$MEMORY" = "1" ] && EXTRA="$EXTRA --memory"
EXTRA="$EXTRA --partition $PART"     # pass the chosen partition through to every torchrun N

# ---- STABILITY: NCCL knobs -----------------------------------------------
# WARN (not INFO) keeps the log readable but still surfaces NCCL problems; async error handling so a
# dead/slow peer RAISES instead of hanging the whole job until SLURM reaps it.
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# single node + NVLink: keep NCCL on-box (no IB / no cross-node), reduces init-time surprises.
export NCCL_P2P_LEVEL=NVL
# unique rendezvous port per job so back-to-back submissions never collide.
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 10000))

echo "==================================================================="
echo "ZORD DIST SCALING  job=${SLURM_JOB_ID:-NA}  node=$(hostname)  GPUTYPE=$GPUTYPE"
echo "  dataset='${DATASET:-synthetic:$SYNTH}'  feat=$FEAT  iters=$ITERS  memory=$MEMORY  partition=$PART"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo "==================================================================="

RESULTS_DIR=$ZORD_DATA/results
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/dist_scaling_${SLURM_JOB_ID:-NA}_summary.jsonl"
: > "$SUMMARY"

BASELINE_MS=""        # N=1 median step ms; captured from the N=1 run, fed to N>=2 for scaling_eff

for N in 1 2 4 8; do
  echo ""
  echo "########################## N=$N GPU(s) ##########################"
  # bump the rendezvous port per N so a torn-down run never clashes with the next.
  PORT=$((MASTER_PORT + N))
  BASE_ARG=""
  [ -n "$BASELINE_MS" ] && BASE_ARG="--baseline-ms $BASELINE_MS"

  RUN_LOG="$RESULTS_DIR/dist_scaling_${SLURM_JOB_ID:-NA}_N${N}.log"
  # --standalone => single-node rendezvous on localhost; --max-restarts 0 so a crashed run fails
  # fast instead of silently retrying (we WANT to see instability, not mask it).
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$N \
    --master_port=$PORT \
    --max-restarts=0 \
    scripts/dist_scaling.py \
    $EXTRA --feat $FEAT --iters $ITERS --timeout-min 20 $BASE_ARG \
    2>&1 | tee "$RUN_LOG" || echo "[sbatch] N=$N torchrun returned non-zero (logged; continuing)"

  # pull the rank-0 JSON result line for this N (the strong-scaling row) into the summary.
  RLINE=$(grep '"tag": "DIST_SCALING_RESULT"' "$RUN_LOG" | tail -1)
  if [ -n "$RLINE" ]; then
    echo "$RLINE" >> "$SUMMARY"
    echo "[sbatch] N=$N result: $RLINE"
    # capture the N=1 makespan as the baseline for scaling efficiency of the larger N.
    if [ "$N" = "1" ]; then
      BASELINE_MS=$(python -c "import json,sys; print(json.loads(sys.argv[1])['step_ms'])" "$RLINE" 2>/dev/null || echo "")
      [ -n "$BASELINE_MS" ] && echo "[sbatch] baseline (N=1) step_ms=$BASELINE_MS captured for scaling_eff"
    fi
  else
    echo "[sbatch] N=$N produced NO result line (run failed or hung; see $RUN_LOG)"
  fi

  # confirm every rank for this N printed 'done' (the stability check we care about most).
  NDONE=$(grep -c "] done" "$RUN_LOG" 2>/dev/null || echo 0)
  echo "[sbatch] N=$N : $NDONE/$N ranks reported done"
  sleep 3        # let NCCL fully release before the next torchrun grabs the GPUs
done

echo ""
echo "==================================================================="
echo "STRONG-SCALING SUMMARY (one JSON line per N; step_ms is the makespan):"
cat "$SUMMARY"
echo "==================================================================="
echo "########## DIST SCALING DONE ##########"
