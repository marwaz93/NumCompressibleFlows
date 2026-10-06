#!/bin/bash
# Description: calls the Julia code to multiply two random matrices
# Requires: chmod +x run_julia.sh

echo "Starting job on $(hostname)"
echo "Date: $(date)"
echo "Arguments: $@"

julia ./scripts/run.jl "$@"                                                                             