# Heat diffusion (MPI + CUDA)

2D explicit heat equation (Jacobi stencil) for an HPC course project: serial CPU baseline, MPI+CPU, and MPI+CUDA with row-wise domain decomposition, halo exchange, optional MPI-IO checkpoints, and timing hooks for scaling studies.

## Build

- **Makefile** — builds `build/heat_serial`, `build/heat_mpi_cpu`, and `build/heat_mpi_cuda` (set `MPICC`, `MPICXX`, `NVCC`, and `CUDA_GENCODE` if your system differs).
- **`scripts/build_aimos.sh`** — loads the AiMOS module stack from the course notes, then runs `make clean && make all` for use on CCI/AiMOS after you copy the tree over.

## Other scripts

- **`scripts/run_single_node.sh`** — convenience wrapper around the built binaries and `mpirun` for quick runs on an allocated node.
- **`scripts/run_strong_weak.sh`** — sweeps ranks / problem sizes and appends CSV-style lines for strong or weak scaling plots.
- **`scripts/plot_results.py`** — reads those CSV logs and writes PNGs (wall time, speedup, phase fractions, comm vs compute). Install deps with `pip install -r requirements-plot.txt`, then e.g. `python3 scripts/plot_results.py scaling_log.csv -o plots/`.
