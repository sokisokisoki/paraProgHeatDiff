/*
 * Hybrid MPI + CUDA 2D heat (Jacobi), row decomposition, MPI-IO checkpoint, timers.
 */
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "clockcycle.h"
#include "heat_common.h"
#include "kernels.cuh"

#define CUDACHK(stmt)                                                                                                  \
    do {                                                                                                               \
        cudaError_t _e = (stmt);                                                                                       \
        if (_e != cudaSuccess) {                                                                                       \
            fprintf(stderr, "%s:%d CUDA error %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                            \
        }                                                                                                              \
    } while (0)

static void usage(const char *p)
{
    fprintf(stderr,
            "Usage: %s [-x NX] [-y NY] [-n NITERS] [-r R] [--checkpoint-every K] [--checkpoint-path P] "
            "[--csv] [-q]\n",
            p);
}

static int parse_args(int argc, char **argv, heat_params *hp)
{
    hp->nx = 256;
    hp->ny = 256;
    hp->niters = 500;
    hp->checkpoint_every = 0;
    hp->csv = 0;
    hp->quiet = 0;
    hp->checkpoint_path = "heat_chk_mpi_cuda.bin";
    hp->r = 0.2;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return -1;
        }
        if (!strcmp(argv[i], "-x") && i + 1 < argc) {
            hp->nx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-y") && i + 1 < argc) {
            hp->ny = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            hp->niters = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            hp->r = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--checkpoint-every") && i + 1 < argc) {
            hp->checkpoint_every = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--checkpoint-path") && i + 1 < argc) {
            hp->checkpoint_path = argv[++i];
        } else if (!strcmp(argv[i], "--csv")) {
            hp->csv = 1;
        } else if (!strcmp(argv[i], "-q")) {
            hp->quiet = 1;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return -1;
        }
    }
    if (hp->nx < 1 || hp->ny < 1 || hp->niters < 1) {
        fprintf(stderr, "Invalid nx/ny/niters\n");
        return -1;
    }
    return 0;
}

static void decompose_rows(int ny, int nprocs, int rank, int *row_start, int *local_ny)
{
    const int rem = ny % nprocs;
    const int base = ny / nprocs;
    *local_ny = base + (rank < rem ? 1 : 0);
    *row_start = rank * base + (rank < rem ? rank : rem);
}

static void exchange_halos_host(double *h_up_send, double *h_dn_send, double *h_up_recv, double *h_dn_recv, int nx,
                                int rank, int nprocs, MPI_Comm comm)
{
    MPI_Request reqs[4];
    int nr = 0;
    if (rank > 0) {
        MPI_Irecv(h_up_recv, nx, MPI_DOUBLE, rank - 1, 100, comm, &reqs[nr++]);
    }
    if (rank < nprocs - 1) {
        MPI_Irecv(h_dn_recv, nx, MPI_DOUBLE, rank + 1, 101, comm, &reqs[nr++]);
    }
    if (rank > 0) {
        MPI_Isend(h_up_send, nx, MPI_DOUBLE, rank - 1, 101, comm, &reqs[nr++]);
    }
    if (rank < nprocs - 1) {
        MPI_Isend(h_dn_send, nx, MPI_DOUBLE, rank + 1, 100, comm, &reqs[nr++]);
    }
    if (nr > 0) {
        MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);
    }
}

static double checkpoint_write_mpiio(const char *path, const double *h_u, int local_ny, int nx, int row_start,
                                     int global_ny, int iter, int rank, int nprocs, MPI_Comm comm, double *io_wall_out,
                                     uint64_t *io_cycles_out)
{
    MPI_File fh;
    double t0 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
    uint64_t c0 = clock_now();
#else
    uint64_t c0 = 0;
#endif
    if (MPI_File_open(comm, (char *)path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank == 0) {
            fprintf(stderr, "MPI_File_open failed: %s\n", path);
        }
        *io_wall_out = 0;
        *io_cycles_out = 0;
        return 0;
    }

    heat_checkpoint_hdr hdr;
    if (rank == 0) {
        hdr.magic = HEAT_MAGIC_CHECKPOINT;
        hdr.nx = nx;
        hdr.ny = global_ny;
        hdr.iter = iter;
        hdr.nprocs = nprocs;
        hdr.pad0 = hdr.pad1 = 0;
        MPI_File_write_at(fh, 0, &hdr, sizeof(hdr), MPI_BYTE, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(comm);

    const MPI_Offset data0 = (MPI_Offset)sizeof(heat_checkpoint_hdr);
    const MPI_Offset my_off = data0 + (MPI_Offset)row_start * (MPI_Offset)nx * sizeof(double);

    const int pitch = nx + 2;
    const size_t chunk = (size_t)local_ny * (size_t)nx;
    double *buf = (double *)malloc(chunk * sizeof(double));
    if (!buf) {
        MPI_File_close(&fh);
        *io_wall_out = MPI_Wtime() - t0;
        *io_cycles_out = 0;
        return 0;
    }
    for (int i = 1; i <= local_ny; i++) {
        memcpy(buf + (size_t)(i - 1) * (size_t)nx, &h_u[i * pitch + 1], (size_t)nx * sizeof(double));
    }

    MPI_File_write_at(fh, my_off, buf, (int)chunk, MPI_DOUBLE, MPI_STATUS_IGNORE);
    free(buf);

    MPI_File_close(&fh);
    double t1 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
    uint64_t c1 = clock_now();
    *io_cycles_out = (c1 >= c0) ? (c1 - c0) : 0;
#else
    *io_cycles_out = 0;
#endif
    *io_wall_out = t1 - t0;
    return t1 - t0;
}

static void copy_interior_to_host(const double *d_u, double *h_u, int local_ny, int nx)
{
    const int pitch = nx + 2;
    const size_t row_bytes = (size_t)nx * sizeof(double);
    for (int i = 1; i <= local_ny; i++) {
        CUDACHK(cudaMemcpy(h_u + (size_t)i * (size_t)pitch + 1, d_u + (size_t)i * (size_t)pitch + 1, row_bytes,
                           cudaMemcpyDeviceToHost));
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = 0, node_size = 1;
    if (node_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_size(node_comm, &node_size);
    }

    int dev_count = 0;
    CUDACHK(cudaGetDeviceCount(&dev_count));
    if (dev_count < 1) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const int dev_id = node_rank % dev_count;
    CUDACHK(cudaSetDevice(dev_id));

    heat_params hp;
    if (parse_args(argc, argv, &hp) != 0) {
        MPI_Finalize();
        return 1;
    }

    const int nx = hp.nx;
    const int ny = hp.ny;
    int row_start = 0, local_ny = 0;
    decompose_rows(ny, nprocs, rank, &row_start, &local_ny);
    if (local_ny == 0) {
        if (rank == 0) {
            fprintf(stderr, "Empty subdomain\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int pitch = nx + 2;
    const size_t n = (size_t)(local_ny + 2) * (size_t)pitch;
    double *h_u = (double *)malloc(n * sizeof(double));
    double *h_ut = (double *)malloc(n * sizeof(double));
    double *h_up_send = (double *)malloc((size_t)nx * sizeof(double));
    double *h_dn_send = (double *)malloc((size_t)nx * sizeof(double));
    double *h_up_recv = (double *)malloc((size_t)nx * sizeof(double));
    double *h_dn_recv = (double *)malloc((size_t)nx * sizeof(double));
    if (!h_u || !h_ut || !h_up_send || !h_dn_send || !h_up_recv || !h_dn_recv) {
        perror("malloc");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int global_y0 = row_start + 1;
    init_grid(h_u, local_ny, nx, global_y0);
    memcpy(h_ut, h_u, n * sizeof(double));

    double *d_a = nullptr;
    double *d_b = nullptr;
    CUDACHK(cudaMalloc(&d_a, n * sizeof(double)));
    CUDACHK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDACHK(cudaMemcpy(d_a, h_u, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_b, h_ut, n * sizeof(double), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CUDACHK(cudaStreamCreate(&stream));

    double t_comm = 0.0, t_comp = 0.0, t_halo_d2h = 0.0, t_halo_h2d = 0.0, t_io = 0.0;
    uint64_t cyc_comm = 0, cyc_comp = 0, cyc_io = 0, cyc_halo_d2h = 0, cyc_halo_h2d = 0;

    double *d_cur = d_a;
    double *d_nxt = d_b;
    const double t_wall0 = MPI_Wtime();

    for (int it = 0; it < hp.niters; it++) {
        double w0 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        uint64_t k0 = clock_now();
#endif
        /* Pack edge rows on host for MPI */
        CUDACHK(cudaMemcpy(h_up_send, d_cur + (size_t)1 * (size_t)pitch + 1, (size_t)nx * sizeof(double),
                           cudaMemcpyDeviceToHost));
        CUDACHK(cudaMemcpy(h_dn_send, d_cur + (size_t)local_ny * (size_t)pitch + 1, (size_t)nx * sizeof(double),
                           cudaMemcpyDeviceToHost));
        double w1 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        uint64_t k1 = clock_now();
        cyc_halo_d2h += (k1 >= k0) ? (k1 - k0) : 0;
#endif
        t_halo_d2h += (w1 - w0);

        w0 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        k0 = clock_now();
#endif
        exchange_halos_host(h_up_send, h_dn_send, h_up_recv, h_dn_recv, nx, rank, nprocs, MPI_COMM_WORLD);
        w1 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        k1 = clock_now();
        cyc_comm += (k1 >= k0) ? (k1 - k0) : 0;
#endif
        t_comm += (w1 - w0);

        w0 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        k0 = clock_now();
#endif
        if (rank > 0) {
            CUDACHK(cudaMemcpy(d_cur + (size_t)0 * (size_t)pitch + 1, h_up_recv, (size_t)nx * sizeof(double),
                               cudaMemcpyHostToDevice));
        }
        if (rank < nprocs - 1) {
            CUDACHK(cudaMemcpy(d_cur + (size_t)(local_ny + 1) * (size_t)pitch + 1, h_dn_recv,
                               (size_t)nx * sizeof(double), cudaMemcpyHostToDevice));
        }
        w1 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        k1 = clock_now();
        cyc_halo_h2d += (k1 >= k0) ? (k1 - k0) : 0;
#endif
        t_halo_h2d += (w1 - w0);

        w0 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        k0 = clock_now();
#endif
        launch_jacobi(d_cur, d_nxt, nx, local_ny, hp.r, stream);
        CUDACHK(cudaGetLastError());
        CUDACHK(cudaStreamSynchronize(stream));
        w1 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        k1 = clock_now();
        cyc_comp += (k1 >= k0) ? (k1 - k0) : 0;
#endif
        t_comp += (w1 - w0);

        double *tmp = d_cur;
        d_cur = d_nxt;
        d_nxt = tmp;

        if (hp.checkpoint_every > 0 && (it + 1) % hp.checkpoint_every == 0) {
            copy_interior_to_host(d_cur, h_u, local_ny, nx);
            double iow = 0;
            uint64_t ioc = 0;
            double tic = checkpoint_write_mpiio(hp.checkpoint_path, h_u, local_ny, nx, row_start, ny, it + 1, rank,
                                                nprocs, MPI_COMM_WORLD, &iow, &ioc);
            (void)tic;
            t_io += iow;
            cyc_io += ioc;
        }
    }

    const double t_wall1 = MPI_Wtime();

    copy_interior_to_host(d_cur, h_u, local_ny, nx);
    double sum_local = grid_checksum(h_u, local_ny, nx);
    double chk = 0.0;
    MPI_Reduce(&sum_local, &chk, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_comm, max_comp, max_hd, max_hu, max_io;
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_halo_d2h, &max_hd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_halo_h2d, &max_hu, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_io, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unsigned long long max_c_comm = 0, max_c_comp = 0, max_c_io = 0, max_c_hd = 0, max_c_hu = 0;
    unsigned long long c_comm_ll = (unsigned long long)cyc_comm;
    unsigned long long c_comp_ll = (unsigned long long)cyc_comp;
    unsigned long long c_io_ll = (unsigned long long)cyc_io;
    unsigned long long c_hd_ll = (unsigned long long)cyc_halo_d2h;
    unsigned long long c_hu_ll = (unsigned long long)cyc_halo_h2d;
    MPI_Reduce(&c_comm_ll, &max_c_comm, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_comp_ll, &max_c_comp, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_io_ll, &max_c_io, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_hd_ll, &max_c_hd, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_hu_ll, &max_c_hu, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0 && !hp.quiet) {
        printf("mpi_cuda nx=%d ny=%d niters=%d nprocs=%d checksum=%.12e wall_s=%.6f t_comm=%.6f t_comp=%.6f "
               "t_halo_d2h=%.6f t_halo_h2d=%.6f t_io=%.6f\n",
               nx, ny, hp.niters, nprocs, chk, t_wall1 - t_wall0, max_comm, max_comp, max_hd, max_hu, max_io);
#if HEAT_HAVE_CLOCK_NOW
        printf("  cycles(max rank): comm=%llu comp=%llu halo_d2h=%llu halo_h2d=%llu io=%llu\n",
               (unsigned long long)max_c_comm, (unsigned long long)max_c_comp, (unsigned long long)max_c_hd,
               (unsigned long long)max_c_hu, (unsigned long long)max_c_io);
#endif
    }
    if (rank == 0 && hp.csv) {
        printf("mode,nx,ny,niters,nprocs,checksum,wall_s,t_comm,t_comp,t_halo_d2h,t_halo_h2d,t_io,cyc_comm,cyc_comp,"
               "cyc_halo_d2h,cyc_halo_h2d,cyc_io\n");
        printf("mpi_cuda,%d,%d,%d,%d,%.17e,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%llu,%llu,%llu,%llu,%llu\n", nx, ny,
               hp.niters, nprocs, chk, t_wall1 - t_wall0, max_comm, max_comp, max_hd, max_hu, max_io,
               (unsigned long long)max_c_comm, (unsigned long long)max_c_comp, (unsigned long long)max_c_hd,
               (unsigned long long)max_c_hu, (unsigned long long)max_c_io);
    }

    CUDACHK(cudaStreamDestroy(stream));
    CUDACHK(cudaFree(d_a));
    CUDACHK(cudaFree(d_b));
    free(h_u);
    free(h_ut);
    free(h_up_send);
    free(h_dn_send);
    free(h_up_recv);
    free(h_dn_recv);
    if (node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&node_comm);
    }
    MPI_Finalize();
    return 0;
}
