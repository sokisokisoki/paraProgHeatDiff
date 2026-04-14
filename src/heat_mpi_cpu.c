/*
 * MPI+CPU 2D heat (row decomposition), halo exchange, timers, optional MPI-IO checkpoint.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "clockcycle.h"
#include "heat_common.h"

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
    hp->checkpoint_path = "heat_chk_mpi_cpu.bin";
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

static void exchange_halos(double *u, int local_ny, int nx, int rank, int nprocs, MPI_Comm comm)
{
    const int p = nx + 2;
    MPI_Request reqs[4];
    int nr = 0;

    /* Post all receives before sends to avoid neighbor cycles. */
    if (rank > 0) {
        MPI_Irecv(&u[0 * p + 1], nx, MPI_DOUBLE, rank - 1, 100, comm, &reqs[nr++]);
    }
    if (rank < nprocs - 1) {
        MPI_Irecv(&u[(local_ny + 1) * p + 1], nx, MPI_DOUBLE, rank + 1, 101, comm, &reqs[nr++]);
    }
    if (rank > 0) {
        MPI_Isend(&u[1 * p + 1], nx, MPI_DOUBLE, rank - 1, 101, comm, &reqs[nr++]);
    }
    if (rank < nprocs - 1) {
        MPI_Isend(&u[local_ny * p + 1], nx, MPI_DOUBLE, rank + 1, 100, comm, &reqs[nr++]);
    }
    if (nr > 0) {
        MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);
    }
}

static double checkpoint_write_mpiio(const char *path, const double *u, int local_ny, int nx, int row_start,
                                     int global_ny, int iter, int rank, int nprocs, MPI_Comm comm,
                                     double *io_wall_out, uint64_t *io_cycles_out)
{
    MPI_File fh;
    double t0 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
    uint64_t c0 = clock_now();
#else
    uint64_t c0 = 0;
#endif
    if (MPI_File_open(comm, (char *)path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) !=
        MPI_SUCCESS) {
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

    /* Pack interior rows without halos into contiguous buffer */
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
        memcpy(buf + (size_t)(i - 1) * (size_t)nx, &u[i * pitch + 1], (size_t)nx * sizeof(double));
    }

    MPI_Status st;
    MPI_File_write_at(fh, my_off, buf, (int)chunk, MPI_DOUBLE, &st);
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

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    heat_params hp;
    if (parse_args(argc, argv, &hp) != 0) {
        MPI_Finalize();
        return 1;
    }

    const int nx = hp.nx;
    const int ny = hp.ny;
    if (ny < nprocs && rank == 0) {
        fprintf(stderr, "Warning: ny < nprocs; some ranks idle (unsupported)\n");
    }

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
    double *u = (double *)malloc(n * sizeof(double));
    double *ut = (double *)malloc(n * sizeof(double));
    if (!u || !ut) {
        perror("malloc");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int global_y0 = row_start + 1;
    init_grid(u, local_ny, nx, global_y0);
    memcpy(ut, u, n * sizeof(double));

    double t_comm = 0.0, t_comp = 0.0, t_io = 0.0;
    uint64_t cyc_comm = 0, cyc_comp = 0, cyc_io = 0;

    double *a = u;
    double *b = ut;
    const double t_wall0 = MPI_Wtime();

    for (int it = 0; it < hp.niters; it++) {
        double w0 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        uint64_t k0 = clock_now();
#endif
        exchange_halos(a, local_ny, nx, rank, nprocs, MPI_COMM_WORLD);
        double w1 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        uint64_t k1 = clock_now();
        cyc_comm += (k1 >= k0) ? (k1 - k0) : 0;
#endif
        t_comm += (w1 - w0);

        w0 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        k0 = clock_now();
#endif
        jacobi_step_cpu(a, b, local_ny, nx, hp.r);
        w1 = MPI_Wtime();
#if HEAT_HAVE_CLOCK_NOW
        k1 = clock_now();
        cyc_comp += (k1 >= k0) ? (k1 - k0) : 0;
#endif
        t_comp += (w1 - w0);

        double *tmp = a;
        a = b;
        b = tmp;

        if (hp.checkpoint_every > 0 && (it + 1) % hp.checkpoint_every == 0) {
            double iow = 0;
            uint64_t ioc = 0;
            double tic = checkpoint_write_mpiio(hp.checkpoint_path, a, local_ny, nx, row_start, ny, it + 1,
                                                rank, nprocs, MPI_COMM_WORLD, &iow, &ioc);
            (void)tic;
            t_io += iow;
            cyc_io += ioc;
        }
    }

    const double t_wall1 = MPI_Wtime();

    double sum_local = grid_checksum(a, local_ny, nx);
    double chk = 0.0;
    MPI_Reduce(&sum_local, &chk, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double max_comm, max_comp, max_io;
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_io, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unsigned long long max_c_comm = 0, max_c_comp = 0, max_c_io = 0;
    unsigned long long c_comm_ll = (unsigned long long)cyc_comm;
    unsigned long long c_comp_ll = (unsigned long long)cyc_comp;
    unsigned long long c_io_ll = (unsigned long long)cyc_io;
    MPI_Reduce(&c_comm_ll, &max_c_comm, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_comp_ll, &max_c_comp, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&c_io_ll, &max_c_io, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0 && !hp.quiet) {
        printf("mpi_cpu nx=%d ny=%d niters=%d nprocs=%d checksum=%.12e wall_s=%.6f t_comm=%.6f t_comp=%.6f "
               "t_io=%.6f\n",
               nx, ny, hp.niters, nprocs, chk, t_wall1 - t_wall0, max_comm, max_comp, max_io);
#if HEAT_HAVE_CLOCK_NOW
        printf("  cycles(max rank): comm=%llu comp=%llu io=%llu\n", (unsigned long long)max_c_comm,
               (unsigned long long)max_c_comp, (unsigned long long)max_c_io);
#endif
    }
    if (rank == 0 && hp.csv) {
        printf("mode,nx,ny,niters,nprocs,checksum,wall_s,t_comm,t_comp,t_io,cyc_comm,cyc_comp,cyc_io\n");
        printf("mpi_cpu,%d,%d,%d,%d,%.17e,%.9f,%.9f,%.9f,%.9f,%llu,%llu,%llu\n", nx, ny, hp.niters, nprocs,
               chk, t_wall1 - t_wall0, max_comm, max_comp, max_io, (unsigned long long)max_c_comm,
               (unsigned long long)max_c_comp, (unsigned long long)max_c_io);
    }

    free(u);
    free(ut);
    MPI_Finalize();
    return 0;
}
