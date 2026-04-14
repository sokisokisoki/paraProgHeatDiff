/*
 * Serial 2D explicit heat (Jacobi) — baseline for scaling comparisons.
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "heat_common.h"

static void usage(const char *p)
{
    fprintf(stderr,
            "Usage: %s [-x NX] [-y NY] [-n NITERS] [-r R] [--csv] [-q]\n",
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
    hp->checkpoint_path = NULL;
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

int main(int argc, char **argv)
{
    heat_params hp;
    if (parse_args(argc, argv, &hp) != 0) {
        return 1;
    }

    const int nx = hp.nx;
    const int ny = hp.ny;
    const int p = nx + 2;
    const size_t n = (size_t)(ny + 2) * (size_t)p;
    double *u = (double *)malloc(n * sizeof(double));
    double *ut = (double *)malloc(n * sizeof(double));
    if (!u || !ut) {
        perror("malloc");
        return 1;
    }

    init_grid(u, ny, nx, 1);
    memcpy(ut, u, n * sizeof(double));

    double t0 = 0.0; /* wall clock optional */
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = ts.tv_sec + 1e-9 * ts.tv_nsec;
#endif

    double *a = u;
    double *b = ut;
    for (int it = 0; it < hp.niters; it++) {
        jacobi_step_cpu(a, b, ny, nx, hp.r);
        double *tmp = a;
        a = b;
        b = tmp;
    }

    double t1 = 0.0;
#ifdef CLOCK_MONOTONIC
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t1 = ts.tv_sec + 1e-9 * ts.tv_nsec;
#endif

    const double chk = grid_checksum(a, ny, nx);
    if (!hp.quiet) {
        printf("serial nx=%d ny=%d niters=%d checksum=%.12e time_s=%.6f\n", nx, ny, hp.niters, chk,
               t1 - t0);
    }
    if (hp.csv) {
        printf("mode,nx,ny,niters,nprocs,checksum,time_s\n");
        printf("serial,%d,%d,%d,1,%.17e,%.9f\n", nx, ny, hp.niters, chk, t1 - t0);
    }

    free(u);
    free(ut);
    return 0;
}
