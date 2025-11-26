module purge


### Define the list of N values (system sizes). Modify as needed.
Ls=(4 8 12 16 20 24 28 32)
Ds=(2 4 8 16)
Rs=(2 4 8 16)
SEEDS=(100 200 300 400 500)

# make sure outputs dir exists
mkdir -p outputs
mkdir -p disbatchfiles

# Tasks file: collect commands to run later (one per line)
TASKS="TASKS"
>"${TASKS}"

echo "[INFO] Preparing to generate task list"
for seed in "${SEEDS[@]}"; do
  for L in "${Ls[@]}"; do
    for D in "${Ds[@]}"; do
      for R in "${Rs[@]}"; do
        OUTF="outputs/certify.S${seed}_L${L}_D${D}_R${R}.out"
        echo "[INFO] Adding task -> seed=${seed}, L=${L}, D=${D}, R=${R} -> logging to ${OUTF}"
        cmd="sh run_julia.sh batch_experiments.jl "${L}" "${D}" certify 1000 ${seed} data "${R}" &> ${OUTF}"
        echo "$cmd" >> "${TASKS}"
      done
    done
  done
done
echo "[INFO] Wrote tasks to ${TASKS} (one command per line)."
module load disBatch/beta
## Submit disBatch job and write stdout/stderr into the repo's outputs/ and errors/ folders
sbatch --output=outputs/out.out -N 2 -n 32 -c 8 -C rome --time=48:00:00 disBatch TASKS -p disbatchfiles/
echo "[INFO] Done launching jobs"
