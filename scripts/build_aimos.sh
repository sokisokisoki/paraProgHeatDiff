#!/usr/bin/env bash
# Build on AiMOS front-end after scp'ing this directory.
# Adjust module names if `module avail` differs from the course PDF.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Loading modules (edit if needed): module load xl r spectrum-mpi cuda"
# shellcheck disable=SC1091
if command -v module >/dev/null 2>&1; then
  module load xl r spectrum-mpi cuda 2>/dev/null || {
    echo "Default module line failed; try manually, e.g.:"
    echo "  module avail xl spectrum-mpi cuda"
    exit 1
  }
else
  echo "No 'module' in PATH — assuming MPI/CUDA already configured."
fi

export MPICC="${MPICC:-mpicc}"
export MPICXX="${MPICXX:-mpicxx}"
export NVCC="${NVCC:-nvcc}"
# V100 (AiMOS): compute_70 / sm_70
export CUDA_GENCODE="${CUDA_GENCODE:-arch=compute_70,code=sm_70}"

make clean
make all

echo "Built:"
ls -la build/
