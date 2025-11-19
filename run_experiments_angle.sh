#!/bin/bash
#SBATCH -C rome
#SBATCH --job-name=angle                                       # create a short name for your job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G                                            # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00
#SBATCH --array=0-6
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
Ns=(2 4 6 8 10 12 14 16 18 20 22 24 26 28)

# Use SLURM_ARRAY_TASK_ID to index into Ns. If not set, default to 0 (useful for direct testing).
idx=${SLURM_ARRAY_TASK_ID:-0}

if (( idx < 0 || idx >= ${#Ns[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID ($idx) is out of range for Ns (0..$((${#Ns[@]}-1)))." >&2
  exit 1
fi

N=${Ns[$idx]}

echo "[INFO] Running array task ${idx} -> N=${N}"
echo "[INFO] Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}"

# make sure outputs dir exists
mkdir -p outputs
mkdir -p errors

# Launch one job per bond-dimension D, each using 8 CPUs from this allocation.
# We use srun --exclusive so Slurm will dedicate the requested CPUs to each task.
Ds=(2 4 8 16 32 64 128 256)

# number of cpus to give each julia process
CPUS_PER_RUN=8

echo "[INFO] Launching ${#Ds[@]} runs (D list: ${Ds[*]}) with ${CPUS_PER_RUN} cpus each"

# Export thread/env vars for Julia and common numeric libraries so they use the CPUs assigned
export JULIA_NUM_THREADS=${CPUS_PER_RUN}

PIDS=()
for D in "${Ds[@]}"; do
  OUTF="outputs/angle.S${seed}_N${N}_D${D}.out"
  ERRF="errors/angle.S${seed}N${N}_D${D}.err"
  mkdir -p "$(dirname "$OUTF")" "$(dirname "$ERRF")"

  echo "[INFO] Starting D=${D} -> logging to ${OUTF} ${ERRF}"
  # srun will launch the process under the current allocation and honor --cpus-per-task
  srun --exclusive --cpus-per-task=${CPUS_PER_RUN} --cpu_bind=cores \
       --output=${OUTF} --error=${ERRF} \
       env JULIA_NUM_THREADS=${JULIA_NUM_THREADS} \
       julia --project=. --sysimage=./sys_bp_cluster_state.so batch_experiments.jl ${N} ${D} 1000 data $seed &
  PIDS+=($!)
done

# Wait for all backgrounded srun tasks to finish
for pid in "${PIDS[@]}"; do
  wait ${pid}
done

echo "[INFO] All runs finished for N=${N}"
