/* How expensive is a getenv() call when it misses?
 *
 * The engine reads sixty of these on the qwen35 path and several are reached per
 * matvec call. getenv is a linear scan of the environment block with a string
 * compare per entry, so its cost depends on how many variables the process has -
 * which for this deployment is a few hundred, since the launcher injects the
 * whole profile. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>

static double now(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e3 / (double)f.QuadPart;
}

int main(void) {
    int env_count = 0;
    for (char **e = _environ; *e; e++) env_count++;
    printf("environment entries: %d\n", env_count);

    const int N = 200000;
    /* a name that is present, and one that never is - both patterns occur */
    const char *present = "PATH";
    const char *absent  = "DS4_QWEN35_BYTE_DBG";

    double t0 = now();
    volatile int sink = 0;
    for (int i = 0; i < N; i++) sink += getenv(present) != NULL;
    double t1 = now();
    for (int i = 0; i < N; i++) sink += getenv(absent) != NULL;
    double t2 = now();

    printf("getenv present : %.3f us/call\n", (t1 - t0) * 1e3 / N);
    printf("getenv absent  : %.3f us/call\n", (t2 - t1) * 1e3 / N);
    printf("(sink %d)\n", (int)sink);
    return 0;
}
