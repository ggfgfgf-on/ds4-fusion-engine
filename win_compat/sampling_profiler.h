/* Sampling profiler for the CPU forward pass.
 *
 * Not wired in yet. It exists because six separate attempts to find the
 * remaining prefill overhead by reading code and adding timers all failed: nine
 * timers, eight candidate explanations, every one of them measured and rejected.
 * What is left - roughly 2000 ms of a 4600 ms prefill at the start of the
 * session, and a good part of what remains at 15.3 t/s - does not show up
 * unless something looks at where the threads actually are.
 *
 * -finstrument-functions was tried and cannot finish: millions of calls, each
 * paying a callback.
 *
 * This samples instead. One thread every millisecond, Toolhelp32Snapshot for the
 * process's threads, SuspendThread, GetThreadContext for RIP, ResumeThread, and a
 * histogram. At exit it prints the hottest addresses; resolve them with
 * llvm-symbolizer against the exe's PDB.
 *
 * Cost when enabled: about one suspension per thread per millisecond, which is
 * far below the noise floor of the measurement. Cost when disabled: nothing.
 *
 * Wiring, once this is in the build:
 *
 *   qwen35_prof_start();      once, before the forward pass
 *   ... run ...
 *   qwen35_prof_stop();       prints the top forty addresses
 *
 * Symbols:
 *   llvm-symbolizer --obj=ds4.exe 0xADDRESS ...
 */
#pragma once

#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define QWEN35_PROF_SLOTS 65536u
#define QWEN35_PROF_TOP     40u

typedef struct {
    uintptr_t pc;
    long      hits;
} qwen35_prof_hit;

static volatile LONG      g_prof_run;
static volatile LONG      g_prof_samples;
static CRITICAL_SECTION   g_prof_cs;
static qwen35_prof_hit    g_prof_tab[QWEN35_PROF_SLOTS];
static uintptr_t          g_prof_base;    /* preferred load address */
static HANDLE             g_prof_thread;

static void qwen35_prof_record(uintptr_t pc) {
    /* open addressing on the low bits; collisions just merge, which is fine for
     * a hotspot list */
    uint32_t s = (uint32_t)((pc >> 4) * 2654435761u) & (QWEN35_PROF_SLOTS - 1u);
    for (uint32_t i = 0; i < 64u; i++) {
        const uint32_t k = (s + i) & (QWEN35_PROF_SLOTS - 1u);
        EnterCriticalSection(&g_prof_cs);
        if (g_prof_tab[k].pc == 0) {
            g_prof_tab[k].pc = pc;
            g_prof_tab[k].hits = 1;
            LeaveCriticalSection(&g_prof_cs);
            return;
        }
        if (g_prof_tab[k].pc == pc) {
            g_prof_tab[k].hits++;
            LeaveCriticalSection(&g_prof_cs);
            return;
        }
        LeaveCriticalSection(&g_prof_cs);
    }
}

static DWORD WINAPI qwen35_prof_main(LPVOID unused) {
    (void)unused;
    const DWORD self = GetCurrentProcessId();
    while (InterlockedCompareExchange(&g_prof_run, 1, 1) == 1) {
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snap != INVALID_HANDLE_VALUE) {
            THREADENTRY32 te;
            te.dwSize = sizeof te;
            if (Thread32First(snap, &te)) {
                do {
                    if (te.th32OwnerProcessID != self) continue;
                    if (te.th32ThreadID == GetCurrentThreadId()) continue;
                    HANDLE th = OpenThread(THREAD_SUSPEND_RESUME |
                                           THREAD_GET_CONTEXT |
                                           THREAD_QUERY_INFORMATION,
                                           FALSE, te.th32ThreadID);
                    if (!th) continue;
                    if (SuspendThread(th) != (DWORD)-1) {
                        CONTEXT ctx;
                        memset(&ctx, 0, sizeof ctx);
                        ctx.ContextFlags = CONTEXT_CONTROL;
                        if (GetThreadContext(th, &ctx))
                            qwen35_prof_record((uintptr_t)ctx.Rip);
                        InterlockedIncrement(&g_prof_samples);
                        ResumeThread(th);
                    }
                    CloseHandle(th);
                } while (Thread32Next(snap, &te));
            }
            CloseHandle(snap);
        }
        Sleep(1);
    }
    return 0;
}

static void qwen35_prof_start(void) {
    InitializeCriticalSection(&g_prof_cs);
    {
        HMODULE hm = GetModuleHandleA(NULL);
        MODULEINFO mi;
        if (GetModuleInformation(GetCurrentProcess(), hm, &mi, sizeof mi))
            g_prof_base = (uintptr_t)mi.lpBaseOfDll;
        else
            g_prof_base = (uintptr_t)hm;
    }
    InterlockedExchange(&g_prof_run, 1);
    g_prof_thread = CreateThread(NULL, 0, qwen35_prof_main, NULL, 0, NULL);
    if (g_prof_thread) SetThreadPriority(g_prof_thread, THREAD_PRIORITY_TIME_CRITICAL);
}

static int qwen35_prof_cmp(const void *a, const void *b) {
    const qwen35_prof_hit *x = a, *y = b;
    return (x->hits < y->hits) - (x->hits > y->hits);
}

static void qwen35_prof_stop(void) {
    InterlockedExchange(&g_prof_run, 0);
    if (g_prof_thread) {
        WaitForSingleObject(g_prof_thread, 2000);
        CloseHandle(g_prof_thread);
        g_prof_thread = NULL;
    }
    /* collect and sort */
    qwen35_prof_hit *hits = malloc(sizeof(qwen35_prof_hit) * QWEN35_PROF_SLOTS);
    size_t n = 0;
    for (uint32_t i = 0; i < QWEN35_PROF_SLOTS; i++)
        if (g_prof_tab[i].pc) hits[n++] = g_prof_tab[i];
    qsort(hits, n, sizeof *hits, qwen35_prof_cmp);
    fprintf(stderr, "[prof] %ld samples, %zu distinct addresses\n",
            (long)g_prof_samples, n);
    const size_t top = n < QWEN35_PROF_TOP ? n : QWEN35_PROF_TOP;
    for (size_t i = 0; i < top; i++)
        fprintf(stderr, "[prof] %5.1f%%  +0x%llx\n",
                100.0 * (double)hits[i].hits / (double)(g_prof_samples ? g_prof_samples : 1),
                (unsigned long long)(hits[i].pc - g_prof_base));
    fprintf(stderr, "[prof] resolve with: llvm-symbolizer --obj=<exe> 0x<addr>\n");
    free(hits);
}
