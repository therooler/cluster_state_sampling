#!/bin/bash
#SBATCH -C rome
#SBATCH --job-name=angle                                       # create a short name for your job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G                                            # memory per cpu-core (4G is default)
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=./outputs/angle.%j.%N.out
#SBATCH --error=./errors/angle.%j.%N.err

## Usage/notes:
module load julia

source "$HOME/.bashrc"   # ensure jup is available (or source a dedicated env file)
set -euo pipefail

# Resolve repo root and sysimage path (use absolute path so srun/julia find it reliably)
REPO_ROOT="$(pwd)"
SYSIMAGE="${SYSIMAGE:-${REPO_ROOT}/sys_bp_cluster_state.so}"
if [ ! -f "$SYSIMAGE" ]; then
  echo "[WARN] sysimage not found at $SYSIMAGE — the runs will use the default Julia startup and may trigger precompilation."
else
  echo "[INFO] Using sysimage: $SYSIMAGE"
fi
seed=${1:-}
if [ -z "$seed" ]; then
  echo "Usage: sbatch run_experiments.sh <seed>" >&2
  echo "Please provide a seed as the first argument (e.g. 100)" >&2
  exit 2
fi
echo "[INFO] Seed $seed"
### Define the list of N values (system sizes). Modify as needed.
Ls=(4 8 12 16 20 24 28 32)
Ds=(2 4)
# Certification R values (adjust as needed)
Rs=(2 4 8 16)

# We will select jobs from the Cartesian product Ls x Ds x Rs.
# Each SLURM array task will launch up to 16 srun jobs. The selection index for
# the k-th job of this array element is: global_idx = SLURM_ARRAY_TASK_ID * 16 + k
# Map global_idx to (iL, iD, iR) lexicographically with N as the outermost loop.
array_id=${SLURM_ARRAY_TASK_ID:-0}
echo "[INFO] SLURM_ARRAY_TASK_ID=${array_id} -> launching up to 16 jobs for this array element"
echo "[INFO] Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}"

# make sure outputs dir exists
mkdir -p outputs
mkdir -p errors

# number of cpus to give each julia process
CPUS_PER_RUN=8

echo "[INFO] Preparing to launch up to 16 jobs per array element using ${CPUS_PER_RUN} cpus each"

# Export thread/env vars for Julia and common numeric libraries so they use the CPUs assigned
export JULIA_NUM_THREADS=${CPUS_PER_RUN}

PIDS=()
nn=${#Ls[@]}
nd=${#Ds[@]}
nr=${#Rs[@]}
total=$(( nn * nd * nr ))

start=$(( array_id * 16 ))
for k in $(seq 0 15); do
  global_idx=$(( start + k ))
  if (( global_idx >= total )); then
    echo "[INFO] global_idx ${global_idx} >= total ${total}; skipping"
    break
  fi
  # map global_idx -> iL, iD, iR with lexicographic order: for L in Ls; for D in Ds; for R in Rs
  block=$(( nd * nr ))
  iL=$(( global_idx / block ))
  rem=$(( global_idx % block ))
  iD=$(( rem / nr ))
  iR=$(( rem % nr ))
  L=${Ls[$iL]}
  D=${Ds[$iD]}
  R=${Rs[$iR]}

  OUTF="outputs/angle.S${seed}_L${L}_D${D}_R${R}.out"
  ERRF="errors/angle.S${seed}L${L}_D${D}_R${R}.err"
  mkdir -p "$(dirname "$OUTF")" "$(dirname "$ERRF")"

  echo "[INFO] Starting job idx=${global_idx} -> L=${L}, D=${D}, R=${R} -> logging to ${OUTF} ${ERRF}"
  # srun will launch the process under the current allocation and honor --cpus-per-task
  # We call batch_experiments.jl with single-element lists for L and D and a single R.
  srun --exclusive --cpus-per-task=${CPUS_PER_RUN} --cpu_bind=cores \
       --output=${OUTF} --error=${ERRF} \
       env JULIA_NUM_THREADS=${JULIA_NUM_THREADS} \
       julia --project=. --sysimage=./sys_bp_cluster_state.so \
           batch_experiments.jl "${L}" "${D}" certify 1000 ${seed} data "${R}" &
  PIDS+=($!)
done

# Wait for all backgrounded srun tasks to finish
for pid in "${PIDS[@]}"; do
  wait ${pid}
done

echo "[INFO] All launched jobs finished (array element ${array_id})"
