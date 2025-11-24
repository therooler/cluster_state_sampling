#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=angle_gpu                                       # create a short name for your job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                                           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G                                            # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=./outputs/gpu_angle.%j.%N.out
#SBATCH --error=./errors/gpu_angle.%j.%N.err
#SBATCH --gpus-per-task=4

## Usage/notes:
module load julia

source "$HOME/.bashrc"   # ensure jup is available (or source a dedicated env file)
set -euo pipefail

# Resolve repo root (use absolute path so srun/julia find it reliably)
REPO_ROOT="$(pwd)"

seed=${1:-}
if [ -z "$seed" ]; then
  echo "Usage: sbatch run_experiments.sh <seed>" >&2
  echo "Please provide a seed as the first argument (e.g. 100)" >&2
  exit 2
fi
echo "[INFO] Seed $seed"
### Define the list of N values (system sizes). Modify as needed.
Ns=(4 8 12 16 20 24 28 32)

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

# Launch one job per bond-dimension D. Each job will request 1 GPU and up to CPUS_PER_RUN CPUs.
# We limit the number of concurrent backgrounded tasks to GPUS_PER_NODE so each running task gets a GPU.
Ds=(2 4 8 16)

# number of cpus to give each julia process
CPUS_PER_RUN=8
GPUS_PER_NODE=4   # maximum GPUs we requested in the SBATCH header

echo "[INFO] Launching ${#Ds[@]} runs (D list: ${Ds[*]}) with ${CPUS_PER_RUN} cpus each and up to ${GPUS_PER_NODE} concurrent GPU tasks"

# Export thread/env vars for Julia and common numeric libraries so they use the CPUs assigned
export JULIA_NUM_THREADS=${CPUS_PER_RUN}

PIDS=()
for D in "${Ds[@]}"; do
  # If we've reached the max concurrent GPU tasks, wait for one to finish before starting another
  while ((${#PIDS[@]} >= GPUS_PER_NODE)); do
    echo "[INFO] Reached ${GPUS_PER_NODE} concurrent GPU jobs — waiting for a slot..."
    # Wait for any background job to finish and remove it from PIDS
    wait -n
    # rebuild PIDS to only include live jobs
    newp=()
    for pid in "${PIDS[@]}"; do
      if kill -0 ${pid} 2>/dev/null; then
        newp+=("${pid}")
      fi
    done
    PIDS=("${newp[@]}")
  done

  OUTF="outputs/angle.S${seed}_N${N}_D${D}.out"
  ERRF="errors/angle.S${seed}N${N}_D${D}.err"
  mkdir -p "$(dirname "$OUTF")" "$(dirname "$ERRF")"

  echo "[INFO] Starting D=${D} -> logging to ${OUTF} ${ERRF}"
  # srun will launch the process under the current allocation and request one GPU for the task.
  # Use both --gres=gpu:1 (widely supported) and --gpus-per-task=1 (modern Slurm) for compatibility.
  srun --exclusive --cpus-per-task=${CPUS_PER_RUN} --cpu_bind=cores \
       --gpus-per-task=1 --output=${OUTF} --error=${ERRF} --gres=gpu:1\
       env JULIA_NUM_THREADS=${JULIA_NUM_THREADS} \
       julia --project=. batch_experiments.jl ${N} ${D} 1000 data $seed &
  PIDS+=($!)
  # small sleep to avoid a tight loop starting processes too fast
  sleep 0.2
done

# Wait for all backgrounded srun tasks to finish
for pid in "${PIDS[@]}"; do
  wait ${pid}
done

echo "[INFO] All runs finished for N=${N}"
