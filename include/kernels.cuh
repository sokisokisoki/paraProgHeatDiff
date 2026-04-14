#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_jacobi(const double *d_u, double *d_unew, int nx, int local_ny, double r, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* KERNELS_CUH */
