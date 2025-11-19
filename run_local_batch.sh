#!/usr/bin/env zsh
# Run many experiments in a single Julia process to minimize per-run precompile/JIT overhead.
# Configure the comma-separated lists below or override via environment variables.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# Ensure output dir exists
mkdir -p data

# Comma-separated lists (override by exporting L_LIST and D_LIST)
L_LIST=${L_LIST:-"2,3"}
D_LIST=${D_LIST:-"4,8"}
NSAMPLES=${NSAMPLES:-100}
OUTDIR=${OUTDIR:-data}

echo "[INFO] Batch run: L_LIST=$L_LIST D_LIST=$D_LIST NSAMPLES=$NSAMPLES OUTDIR=$OUTDIR"

julia --project=. batch_experiments.jl "$L_LIST" "$D_LIST" "$NSAMPLES" "$OUTDIR"

echo "[INFO] Done. CSVs in $OUTDIR"