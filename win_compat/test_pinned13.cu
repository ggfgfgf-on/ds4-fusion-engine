// test_pinned13.cu — does VRAM occupancy shrink the pinned (locked-page)
// pool? Allocate 3.5GB VRAM first, then measure how much cudaHostAlloc
// can still lock.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <windows.h>
#pragma comment(lib, "advapi32.lib")

static void enable_lock_memory(void) {
    HANDLE tok = NULL;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return;
    TOKEN_PRIVILEGES tp;
    LUID luid;
    if (!LookupPrivilegeValueA(NULL, "SeLockMemoryPrivilege", &luid)) { CloseHandle(tok); return; }
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    (void)AdjustTokenPrivileges(tok, FALSE, &tp, sizeof(tp), NULL, NULL);
    CloseHandle(tok);
}

int main() {
    enable_lock_memory();
    // occupy VRAM
    char *vram[7];
    int vram_blocks = 0;
    for (int i = 0; i < 7; i++) {
        vram[i] = NULL;
        if (cudaMalloc((void **)&vram[i], 512ull * 1024ull * 1024ull) != cudaSuccess) break;
        vram_blocks++;
    }
    printf("VRAM occupied: %.1f GB\n", (double)vram_blocks * 0.5);
    // now pinned
    char *pin[16];
    int pin_blocks = 0;
    for (int i = 0; i < 16; i++) {
        pin[i] = NULL;
        cudaError_t err = cudaHostAlloc((void **)&pin[i],
                                        512ull * 1024ull * 1024ull,
                                        cudaHostAllocMapped);
        if (err != cudaSuccess) {
            printf("pinned block %d failed at %.1f GB: %s\n",
                   i, (double)pin_blocks * 0.5, cudaGetErrorString(err));
            break;
        }
        pin_blocks++;
    }
    printf("pinned total with %.1f GB VRAM: %.1f GB\n",
           (double)vram_blocks * 0.5, (double)pin_blocks * 0.5);
    for (int i = 0; i < pin_blocks; i++) cudaFreeHost(pin[i]);
    for (int i = 0; i < vram_blocks; i++) cudaFree(vram[i]);
    return 0;
}
