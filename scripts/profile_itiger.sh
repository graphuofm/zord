#!/bin/bash
# Measure real per-GPU throughput r_k and H2D/D2H bandwidth on an HetCluster node,
# for one GPU tier at a time. Submit once per tier:
#   sbatch --gres=gpu:h100:1     scripts/profile_hetcluster.sh
#   sbatch --gres=gpu:rtx_6000:1 scripts/profile_hetcluster.sh
#   sbatch --gres=gpu:rtx_5000:1 scripts/profile_hetcluster.sh
# Output -> $ZORD_DATA/profile/<gpu>_<jobid>.json  (feeds DeviceProfile)
#SBATCH --partition=bigTiger
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=$ZORD_DATA/profile/profile_%j.out
set -e
source <conda.sh>
conda activate $PROJECT/hkenv
export PYTHONUNBUFFERED=1
mkdir -p $ZORD_DATA/profile
python - <<'PY'
import json, os, time, torch
dev = torch.device("cuda:0")
name = torch.cuda.get_device_name(0)
total_mem = torch.cuda.get_device_properties(0).total_memory
def bench_mm(n=8192, iters=30):
    a = torch.randn(n, n, device=dev); b = torch.randn(n, n, device=dev)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): c = a @ b
    torch.cuda.synchronize(); return (time.time()-t0)/iters
def bench_copy(nbytes=256*1024*1024, iters=30, to_gpu=True):
    host = torch.empty(nbytes//4, dtype=torch.float32, pin_memory=True)
    dvt = torch.empty(nbytes//4, dtype=torch.float32, device=dev)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(iters):
        (dvt.copy_(host, non_blocking=True) if to_gpu else host.copy_(dvt, non_blocking=True))
    torch.cuda.synchronize(); dt=(time.time()-t0)/iters
    return nbytes/dt/1e9  # GB/s
mm = bench_mm()
out = {"gpu": name, "total_mem_gb": round(total_mem/1024**3,1),
       "mm_8192_sec": round(mm,5), "throughput_gflops": round(2*8192**3/mm/1e9,1),
       "h2d_gbps": round(bench_copy(to_gpu=True),1),
       "d2h_gbps": round(bench_copy(to_gpu=False),1),
       "jobid": os.environ.get("SLURM_JOB_ID","local")}
p=f"$ZORD_DATA/profile/{name.replace(' ','_')}_{out['jobid']}.json"
json.dump(out, open(p,"w"), indent=2); print("WROTE", p); print(json.dumps(out,indent=2))
PY
