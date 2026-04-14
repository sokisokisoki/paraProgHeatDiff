#!/usr/bin/env bash
# Example: on an allocated GPU node (after ssh from salloc), from this repo:
#   export OMP_NUM_THREADS=1
#   ./scripts/run_single_node.sh mpi_cuda 4 512 512 2000
#
# Usage: run_single_node.sh <serial|mpi_cpu|mpi_cuda> [nranks] [nx] [ny] [niters] [extra args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:?mode: serial | mpi_cpu | mpi_cuda}"
NRANKS="${2:-1}"
NX="${3:-256}"
NY="${4:-256}"
NIT="${5:-500}"
EXTRA=("${@:6}")

BIN=""
case "$MODE" in
  serial) BIN="$ROOT/build/heat_serial" ; NRANKS=1 ;;
  mpi_cpu) BIN="$ROOT/build/heat_mpi_cpu" ;;
  mpi_cuda) BIN="$ROOT/build/heat_mpi_cuda" ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

if [[ ! -x "$BIN" ]]; then
  echo "Missing binary: $BIN — run scripts/build_aimos.sh first."
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ "$MODE" == "serial" ]]; then
  exec "$BIN" -x "$NX" -y "$NY" -n "$NIT" "${EXTRA[@]}"
fi

exec mpirun -np "$NRANKS" "$BIN" -x "$NX" -y "$NY" -n "$NIT" "${EXTRA[@]}"
