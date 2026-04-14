#!/usr/bin/env bash
# Sweep problem sizes / ranks and append CSV lines to a log file for plotting.
# Usage:
#   ./scripts/run_strong_weak.sh strong mpi_cpu /tmp/out.csv
#   ./scripts/run_strong_weak.sh weak mpi_cuda /tmp/out.csv
#
# strong: fixed nx=ny=1024, vary NR in WORLD_SIZE or second arg list
# weak:   nx=ny=256*NRANKS (rows grow with ranks), 1D row split
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KIND="${1:?strong|weak}"
MODE="${2:?mpi_cpu|mpi_cuda}"
OUT="${3:-scaling_log.csv}"

BIN="$ROOT/build/heat_mpi_cpu"
[[ "$MODE" == "mpi_cuda" ]] && BIN="$ROOT/build/heat_mpi_cuda"
if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN"
  exit 1
fi

: "${OMP_NUM_THREADS:=1}"
export OMP_NUM_THREADS

NR_LIST="${NR_LIST:-1 2 4 8}"
NIT="${NIT:-400}"

echo "# kind=$KIND mode=$MODE niters=$NIT date=$(date -Iseconds)" >>"$OUT"

if [[ "$KIND" == "strong" ]]; then
  NX="${NX_STRONG:-1024}"
  NY="${NY_STRONG:-1024}"
  for NR in $NR_LIST; do
    echo "== strong NR=$NR nx=$NX ny=$NY" | tee -a "$OUT"
    mpirun -np "$NR" "$BIN" -x "$NX" -y "$NY" -n "$NIT" --csv | tee -a "$OUT" | tail -n1
  done
elif [[ "$KIND" == "weak" ]]; then
  BASE="${WEAK_BASE:-256}"
  for NR in $NR_LIST; do
    NY=$((BASE * NR))
    NX="$BASE"
    echo "== weak NR=$NR nx=$NX ny=$NY (rows ~$BASE per rank)" | tee -a "$OUT"
    mpirun -np "$NR" "$BIN" -x "$NX" -y "$NY" -n "$NIT" --csv | tee -a "$OUT" | tail -n1
  done
else
  echo "Unknown kind: $KIND"
  exit 1
fi

echo "Wrote $OUT"
