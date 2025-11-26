#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=1

module purge

julia --project=. $*

echo "[INFO] All launched jobs finished (array element ${array_id})"
