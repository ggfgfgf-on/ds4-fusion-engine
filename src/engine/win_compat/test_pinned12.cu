// test_pinned12.cu — enable SeLockMemoryPrivilege on the current token
// (AdjustTokenPrivileges) then test how much pinned memory we can lock:
// cudaHostAlloc blocks growing 512MB at a time up to 6GB.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <windows.h>
#pragma comment(lib, "advapi32.lib")

static int enable_lock_memory(void) {
    HANDLE tok = NULL;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        printf("OpenProcessToken failed: %lu\n", GetLastError());
        return 0;
    }
    TOKEN_PRIVILEGES tp;
    LUID luid;
    if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)) {
        printf("LookupPrivilegeValue failed: %lu\n", GetLastError());
        CloseHandle(tok);
        return 0;
    }
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    if (!AdjustTokenPrivileges(tok, FALSE, &tp, sizeof(tp), NULL, NULL)) {
        printf("AdjustTokenPrivileges failed: %lu\n", GetLastError());
        CloseHandle(tok);
        return 0;
    }
    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED) {
        printf("privilege NOT assigned to this token\n");
        CloseHandle(tok);
        return 0;
    }
    CloseHandle(tok);
    return 1;
}

int main() {
    if (!enable_lock_memory()) {
        printf("SeLockMemoryPrivilege not available in this session\n");
    } else {
        printf("SeLockMemoryPrivilege enabled on token\n");
    }
    // now try growing pinned allocations
    char *blocks[12];
    int ok_blocks = 0;
    for (int i = 0; i < 12; i++) {
        blocks[i] = NULL;
        cudaError_t err = cudaHostAlloc((void **)&blocks[i],
                                        512ull * 1024ull * 1024ull,
                                        cudaHostAllocMapped);
        if (err != cudaSuccess) {
            printf("block %d (512MB) failed at total %.1f GB: %s\n",
                   i, (double)ok_blocks * 0.5, cudaGetErrorString(err));
            break;
        }
        ok_blocks++;
    }
    printf("locked total: %.1f GB (%d x 512MB)\n", (double)ok_blocks * 0.5, ok_blocks);
    for (int i = 0; i < ok_blocks; i++) cudaFreeHost(blocks[i]);
    return 0;
}
