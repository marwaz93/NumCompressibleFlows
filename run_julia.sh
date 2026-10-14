#!/bin/bash
# Description: runs one job of the study, all arguments are passed to scripts/run.jl
# Requires: chmod +x run_julia.sh

echo "Starting job on $(hostname)"
echo "Date: $(date)"
echo "Arguments: $@"

## one thread per job (request_cpus = 1)
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

julia --project=. ./scripts/run.jl "$@"
