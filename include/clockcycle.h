/* POWER9/AiMOS time-base counter — from course materials; do not use on other arches. */
#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H

#include <stdint.h>

#if defined(__powerpc__) || defined(__PPC__) || defined(__ppc64__)

static inline uint64_t clock_now(void)
{
    unsigned int tbl, tbu0, tbu1;
    do {
        __asm__ __volatile__("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__("mftb %0" : "=r"(tbl));
        __asm__ __volatile__("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);
    return (((uint64_t)tbu0) << 32) | tbl;
}

#define HEAT_HAVE_CLOCK_NOW 1

#else

static inline uint64_t clock_now(void) { return 0; }

#define HEAT_HAVE_CLOCK_NOW 0

#endif

#endif /* CLOCKCYCLE_H */
