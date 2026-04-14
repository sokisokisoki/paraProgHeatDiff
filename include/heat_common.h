#ifndef HEAT_COMMON_H
#define HEAT_COMMON_H

#include <stddef.h>
#include <stdint.h>

#define HEAT_MAGIC_CHECKPOINT 0x48454154u /* "HEAT" */

typedef struct heat_params {
    int nx;       /* interior columns */
    int ny;       /* interior rows (global for MPI) */
    int niters;
    int checkpoint_every; /* 0 = disabled */
    int csv;
    int quiet;
    const char *checkpoint_path;
    double r; /* explicit step coeff, must be < 0.25 for stability */
} heat_params;

typedef struct __attribute__((packed)) heat_checkpoint_hdr {
    uint32_t magic;
    int32_t nx;
    int32_t ny;
    int32_t iter;
    int32_t nprocs;
    int32_t pad0;
    int32_t pad1;
} heat_checkpoint_hdr;

static inline int idx2d(int i, int j, int pitch) { return i * pitch + j; }

/* One Jacobi step for explicit heat: u_new = u + r * Laplacian(u) */
static inline void jacobi_step_cpu(const double *u, double *unew, int local_ny, int nx, double r)
{
    const int p = nx + 2;
    for (int i = 1; i <= local_ny; i++) {
        for (int j = 1; j <= nx; j++) {
            const int k = i * p + j;
            unew[k] = u[k] + r * (u[k - p] + u[k + p] + u[k - 1] + u[k + 1] - 4.0 * u[k]);
        }
    }
}

static inline double grid_checksum(const double *u, int local_ny, int nx)
{
    const int p = nx + 2;
    double s = 0.0;
    for (int i = 1; i <= local_ny; i++) {
        for (int j = 1; j <= nx; j++) {
            s += u[i * p + j];
        }
    }
    return s;
}

static inline void init_grid(double *u, int local_ny, int nx, int global_y0)
{
    const int p = nx + 2;
    for (int i = 0; i < local_ny + 2; i++) {
        for (int j = 0; j < nx + 2; j++) {
            u[i * p + j] = 0.0;
        }
    }
    /* Interior: 1.0; global top row (global_y == 1) kept hot for visible diffusion */
    for (int i = 1; i <= local_ny; i++) {
        int gy = global_y0 + (i - 1);
        for (int j = 1; j <= nx; j++) {
            if (gy == 1) {
                u[i * p + j] = 1.0;
            } else {
                u[i * p + j] = 0.0;
            }
        }
    }
}

#endif /* HEAT_COMMON_H */
