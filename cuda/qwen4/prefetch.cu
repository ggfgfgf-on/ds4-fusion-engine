/* A prefetch queue: copies on their own thread, issued ahead of the products that use them.
 *
 * The point is depth. A copy out of pageable memory blocks the thread that issues it until
 * the driver has staged it, so issuing a copy and then its product from one thread
 * serialises them however many streams exist - measured, two streams gave 106.95 against
 * 106.57. On a second thread the copy blocks nobody that matters, and a window of several
 * items lets the first product start while the rest are still arriving.
 *
 *   pf_start();                        once, idempotent
 *   pf_issue(src, dst, bytes);         the producer is the compute thread
 *   pf_consume();                      one completion, in order
 *
 * A caller that issues three items a pass and consumes three is balanced; one that returns
 * early leaves the ring out of step, so callers must consume what they issue.
 */
#include <windows.h>
#include <cuda_runtime.h>
#include <stddef.h>

#define PF_SLOTS 4

typedef struct {
    const void *src;
    unsigned char *dst;
    size_t bytes;
    cudaEvent_t done;
} pf_item;

static struct {
    HANDLE work, ack;
    int stop;
    cudaStream_t stream;    /* the copies live here, the products on the default stream */
} g_pf;
static HANDLE g_pf_thread = NULL;
static pf_item g_pf_item[PF_SLOTS];
static volatile LONG g_pf_head = 0, g_pf_tail = 0;
static volatile LONG g_pf_errors = 0;
static volatile LONG g_pf_confirmed = 0;

static DWORD WINAPI pf_worker(LPVOID param) {
    (void)param;
    for (;;) {
        WaitForSingleObject(g_pf.work, INFINITE);
        if (g_pf.stop) break;
        const int slot = (int)(InterlockedIncrement(&g_pf_tail) - 1) & (PF_SLOTS - 1);
        cudaMemcpyAsync(g_pf_item[slot].dst, g_pf_item[slot].src, g_pf_item[slot].bytes,
                        cudaMemcpyHostToDevice, g_pf.stream);
        if (cudaGetLastError() != cudaSuccess) InterlockedIncrement(&g_pf_errors);
        cudaEventRecord(g_pf_item[slot].done, g_pf.stream);
        ReleaseSemaphore(g_pf.ack, 1, NULL);
    }
    return 0;
}

extern "C" int ds4_pf_start(void) {
    if (g_pf_thread) return 1;
    if (cudaStreamCreateWithFlags(&g_pf.stream, cudaStreamNonBlocking) != cudaSuccess)
        return 0;
    for (int i = 0; i < PF_SLOTS; i++)
        if (cudaEventCreateWithFlags(&g_pf_item[i].done, cudaEventDisableTiming) !=
            cudaSuccess)
            return 0;
    g_pf.work = CreateSemaphoreA(NULL, 0, PF_SLOTS, NULL);
    g_pf.ack = CreateSemaphoreA(NULL, 0, PF_SLOTS, NULL);
    if (!g_pf.work || !g_pf.ack) return 0;
    g_pf_thread = CreateThread(NULL, 0, pf_worker, NULL, 0, NULL);
    return g_pf_thread != NULL;
}

extern "C" void ds4_pf_issue(const void *src, unsigned char *dst, size_t bytes) {
    const int slot = (int)(InterlockedIncrement(&g_pf_head) - 1) & (PF_SLOTS - 1);
    g_pf_item[slot].src = src;
    g_pf_item[slot].dst = dst;
    g_pf_item[slot].bytes = bytes;
    if (!ReleaseSemaphore(g_pf.work, 1, NULL)) InterlockedIncrement(&g_pf_errors);
}

extern "C" int ds4_pf_consume(void) {
    if (WaitForSingleObject(g_pf.ack, 10000) != WAIT_OBJECT_0) return 0;
    /* the copy was issued on the prefetch stream; the product that follows runs on the
     * default one, so it has to wait for that event rather than for a thread */
    const int slot = (int)(InterlockedIncrement(&g_pf_confirmed) - 1) & (PF_SLOTS - 1);
    return cudaEventSynchronize(g_pf_item[slot].done) == cudaSuccess;
}

extern "C" long ds4_pf_error_count(void) { return g_pf_errors; }
