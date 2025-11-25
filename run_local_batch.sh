#!/usr/bin/env zsh
# Run many experiments in a single Julia process to minimize per-run precompile/JIT overhead.
# Configure the comma-separated lists below or override via environment variables.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# Ensure output dir exists
mkdir -p data

# Comma-separated lists (override by exporting L_LIST and D_LIST)
L_LIST=$1
D_LIST=$2
NSAMPLES=$3
SEED=$4
OUTDIR=$5

echo "[INFO] Batch run: L_LIST=$L_LIST D_LIST=$D_LIST NSAMPLES=$NSAMPLES SEED=$SEED OUTDIR=$OUTDIR"

julia --project=. batch_experiments.jl "$L_LIST" "$D_LIST" "certify" "$NSAMPLES" "$SEED" "$OUTDIR" 4
# julia --project=. batch_experiments.jl "$L_LIST" "$D_LIST" "sample" "$NSAMPLES" "$SEED" "$OUTDIR"

echo "[INFO] Done. CSVs in $OUTDIR"