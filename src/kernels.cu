#include <cuda_runtime.h>

#include "kernels.cuh"

__global__ void jacobi_kernel(const double *__restrict__ u, double *__restrict__ unew, int nx, int local_ny,
                              double r)
{
    const int pitch = nx + 2;
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= local_ny && j <= nx) {
        const int k = i * pitch + j;
        unew[k] = u[k] + r * (u[k - pitch] + u[k + pitch] + u[k - 1] + u[k + 1] - 4.0 * u[k]);
    }
}

void launch_jacobi(const double *d_u, double *d_unew, int nx, int local_ny, double r, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((nx + block.x - 1) / block.x, (local_ny + block.y - 1) / block.y);
    jacobi_kernel<<<grid, block, 0, stream>>>(d_u, d_unew, nx, local_ny, r);
}
