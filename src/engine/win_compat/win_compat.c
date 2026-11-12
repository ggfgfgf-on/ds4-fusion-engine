/* win_compat.c - POSIX compatibility implementations for Windows builds.
 * See win_compat.h for the mapping table. Compile this file and link it
 * into every ds4 binary on Windows.
 */
#include "win_compat.h"
#include "termios.h"
#include "sys/ioctl.h"
#include "dirent.h"
#include "fnmatch.h"

#ifdef _WIN32

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdarg.h>

/* Make sure the implementation below calls the real Winsock functions
 * instead of the ds4_* wrappers declared in win_compat.h. */
#undef socket
#undef setsockopt
#undef getsockopt

#ifndef ETIMEDOUT
#define ETIMEDOUT 138
#endif
#ifndef ESRCH
#define ESRCH 3
#endif
#ifndef ECHILD
#define ECHILD 10
#endif

/* ================================================================== */
/* console / process init                                              */
/* ================================================================== */
void ds4_win_init(void) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
}

/* Enable SeLockMemoryPrivilege on the current token. The "Lock pages in
 * memory" policy may already grant it (secpol.msc -> User Rights), but it
 * arrives DISABLED; CUDA does not enable it, and without it the WDDM
 * locked-page pool caps pinned allocations at ~1 GiB (cudaHostAlloc /
 * cudaHostRegister then fail with "out of memory" and lock pressure drags
 * the whole system down). Enabling it lifts pinned capacity to RAM scale
 * (verified: 6 GiB cudaHostAllocMapped). No-op when not granted. */
void ds4_win_enable_lock_memory(void) {
    HANDLE tok = NULL;
    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        return;
    }
    TOKEN_PRIVILEGES tp;
    LUID luid;
    if (!LookupPrivilegeValueA(NULL, "SeLockMemoryPrivilege", &luid)) {
        CloseHandle(tok);
        return;
    }
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Luid = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    (void)AdjustTokenPrivileges(tok, FALSE, &tp, sizeof(tp), NULL, NULL);
    CloseHandle(tok);
}

/* Force binary mode: the CRT text-mode default treats 0x1A as EOF and
 * truncates every read at the first Ctrl-Z byte (GGUF data is full of
 * them). */
int ds4_open(const char *path, int oflag, ...) {
    int pmode = 0;
    if (oflag & _O_CREAT) {
        va_list ap;
        va_start(ap, oflag);
        pmode = va_arg(ap, int);
        va_end(ap);
    }
    return _open(path, oflag | _O_BINARY, pmode);
}

/* ================================================================== */
/* hybrid fd/socket helpers                                            */
/* ================================================================== */
static int is_socket_fd(intptr_t fd) {
    int type = 0;
    int tlen = sizeof(type);
    if (fd < 0) return 0;
    return getsockopt((SOCKET)fd, SOL_SOCKET, SO_TYPE, (char *)&type, &tlen) == 0;
}

static int wsa_started = 0;

int ds4_socket(int af, int type, int protocol) {
    if (!wsa_started) {
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return -1;
        wsa_started = 1;
    }
    SOCKET s = socket(af, type, protocol);
    if (s == INVALID_SOCKET) return -1;
    return (int)s;
}

long ds4_close(long fd) {
    if (is_socket_fd(fd)) {
        /* Graceful close: Winsock closesocket() discards queued unsent data
         * and sends RST when the peer requested Connection: close, killing
         * the response before the client can read it.  Linger + SD_SEND
         * flushes pending sends before the FIN. */
        SOCKET s = (SOCKET)fd;
        struct linger lg;
        lg.l_onoff = 1;
        lg.l_linger = 5; /* seconds */
        setsockopt(s, SOL_SOCKET, SO_LINGER, (const char *)&lg, sizeof(lg));
        shutdown(s, SD_SEND);
        return closesocket(s);
    }
    return _close((int)fd);
}

ssize_t ds4_read(long fd, void *buf, size_t len) {
    if (is_socket_fd(fd)) {
        int r = recv((SOCKET)fd, (char *)buf, (int)len, 0);
        return (r == SOCKET_ERROR) ? -1 : (ssize_t)r;
    }
    return (ssize_t)_read((int)fd, buf, (unsigned)len);
}

ssize_t ds4_write(long fd, const void *buf, size_t len) {
    if (is_socket_fd(fd)) {
        int r = send((SOCKET)fd, (const char *)buf, (int)len, 0);
        return (r == SOCKET_ERROR) ? -1 : (ssize_t)r;
    }
    return (ssize_t)_write((int)fd, buf, (unsigned)len);
}

int ds4_setsockopt(long fd, int level, int optname, const void *optval,
                   int optlen) {
    return setsockopt((SOCKET)fd, level, optname, (const char *)optval,
                      optlen);
}

int ds4_getsockopt(long fd, int level, int optname, void *optval,
                   int *optlen) {
    return getsockopt((SOCKET)fd, level, optname, (char *)optval, optlen);
}

static int set_nonblocking(int fd, u_long on) {
    return ioctlsocket((SOCKET)fd, FIONBIO, &on) == 0 ? 0 : -1;
}

int ds4_recv_dontwait(long fd, void *buf, size_t len) {
    if (set_nonblocking((int)fd, 1) != 0) return -1;
    int r = recv((SOCKET)fd, (char *)buf, (int)len, 0);
    int err = WSAGetLastError();
    set_nonblocking((int)fd, 0);
    if (r == SOCKET_ERROR) {
        errno = (err == WSAEWOULDBLOCK) ? EAGAIN : EIO;
        return -1;
    }
    return r;
}

int ds4_send_dontwait(long fd, const void *buf, size_t len) {
    if (set_nonblocking((int)fd, 1) != 0) return -1;
    int r = send((SOCKET)fd, (const char *)buf, (int)len, 0);
    int err = WSAGetLastError();
    set_nonblocking((int)fd, 0);
    if (r == SOCKET_ERROR) {
        errno = (err == WSAEWOULDBLOCK) ? EAGAIN : EIO;
        return -1;
    }
    return r;
}

/* ================================================================== */
/* mmap / munmap / mlock                                               */
/* ================================================================== */
#define MAP_REG_MAX 8192
typedef struct {
    void *addr;
    HANDLE mapping;
    size_t length;
} map_entry_t;

static map_entry_t *map_reg;
static size_t map_reg_n;

static void map_reg_init(void) {
    if (map_reg) return;
    map_reg = (map_entry_t *)calloc(MAP_REG_MAX, sizeof(map_entry_t));
    map_reg_n = 0;
}

void *ds4_mmap(void *addr, size_t length, int prot, int flags,
               long fd, off_t offset) {
    (void)addr;
    if (length == 0) {
        errno = EINVAL;
        return MAP_FAILED;
    }
    if (flags & MAP_ANONYMOUS) {
        DWORD flProtect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
        void *p = VirtualAlloc(NULL, length, MEM_RESERVE | MEM_COMMIT, flProtect);
        if (!p) {
            errno = ENOMEM;
            return MAP_FAILED;
        }
        return p;
    }
    HANDLE fh = (HANDLE)_get_osfhandle((int)fd);
    if (fh == INVALID_HANDLE_VALUE || fh == NULL) {
        errno = EBADF;
        return MAP_FAILED;
    }
    DWORD flProtect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
    HANDLE mapping = CreateFileMappingA(fh, NULL, flProtect, 0, 0, NULL);
    if (!mapping) {
        errno = ENOMEM;
        return MAP_FAILED;
    }
    DWORD access = (prot & PROT_WRITE) ? FILE_MAP_WRITE : FILE_MAP_READ;
    void *view = MapViewOfFile(mapping, access,
                               (DWORD)((unsigned __int64)offset >> 32),
                               (DWORD)((unsigned __int64)offset & 0xffffffff),
                               length);
    if (!view) {
        CloseHandle(mapping);
        errno = ENOMEM;
        return MAP_FAILED;
    }
    map_reg_init();
    if (map_reg && map_reg_n < MAP_REG_MAX) {
        map_reg[map_reg_n].addr = view;
        map_reg[map_reg_n].mapping = mapping;
        map_reg[map_reg_n].length = length;
        map_reg_n++;
    }
    return view;
}

static map_entry_t *map_reg_find(void *addr) {
    size_t i;
    if (!map_reg) return NULL;
    for (i = 0; i < map_reg_n; i++) {
        if (map_reg[i].addr == addr) return &map_reg[i];
    }
    return NULL;
}

int ds4_munmap(void *addr, size_t length) {
    (void)length;
    map_entry_t *e = map_reg_find(addr);
    if (e) {
        UnmapViewOfFile(e->addr);
        CloseHandle(e->mapping);
        size_t idx = (size_t)(e - map_reg);
        for (size_t i = idx; i + 1 < map_reg_n; i++)
            map_reg[i] = map_reg[i + 1];
        map_reg_n--;
        return 0;
    }
    if (VirtualFree(addr, 0, MEM_RELEASE)) return 0;
    errno = EINVAL;
    return -1;
}

int ds4_madvise(void *addr, size_t length, int advice) {
    (void)addr; (void)length; (void)advice;
    return 0;
}

int ds4_mlock(const void *addr, size_t length) {
    if (VirtualLock((LPVOID)addr, length)) return 0;
    errno = ENOMEM;
    return -1;
}

int ds4_munlock(const void *addr, size_t length) {
    if (VirtualUnlock((LPVOID)addr, length)) return 0;
    errno = ENOMEM;
    return -1;
}

int ds4_msync(void *addr, size_t length, int flags) {
    (void)flags;
    map_entry_t *e = map_reg_find(addr);
    if (e) {
        FlushViewOfFile(addr, length);
        return 0;
    }
    errno = EINVAL;
    return -1;
}

/* ================================================================== */
/* clocks                                                              */
/* ================================================================== */
static unsigned __int64 epoch_offset_100ns(void) {
    return 116444736000000000ULL;
}

static void filetime_to_timespec(const FILETIME *ft, struct timespec *ts) {
    unsigned __int64 t = ((unsigned __int64)ft->dwHighDateTime << 32) |
                         (unsigned __int64)ft->dwLowDateTime;
    if (t < epoch_offset_100ns()) t = epoch_offset_100ns();
    t -= epoch_offset_100ns();
    ts->tv_sec = (__int64)(t / 10000000ULL);
    ts->tv_nsec = (__int64)((t % 10000000ULL) * 100ULL);
}

int ds4_clock_gettime(int clk_id, struct timespec *ts) {
    if (clk_id == CLOCK_REALTIME || clk_id == CLOCK_REALTIME_COARSE) {
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        filetime_to_timespec(&ft, ts);
        return 0;
    }
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    __int64 secs = now.QuadPart / freq.QuadPart;
    __int64 rem = now.QuadPart % freq.QuadPart;
    ts->tv_sec = secs;
    ts->tv_nsec = (__int64)((rem * 1000000000ULL) / freq.QuadPart);
    return 0;
}

int ds4_gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    unsigned __int64 t = ((unsigned __int64)ft.dwHighDateTime << 32) |
                         (unsigned __int64)ft.dwLowDateTime;
    if (t < epoch_offset_100ns()) t = epoch_offset_100ns();
    t -= epoch_offset_100ns();
    tv->tv_sec = (long)(t / 10000000ULL);
    tv->tv_usec = (long)((t % 10000000ULL) / 10ULL);
    return 0;
}

/* ================================================================== */
/* sysconf / pread / sleep                                             */
/* ================================================================== */
long ds4_sysconf(int name) {
    if (name == _SC_PAGESIZE) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwPageSize;
    } else if (name == _SC_NPROCESSORS_ONLN) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwNumberOfProcessors;
    } else if (name == _SC_PHYS_PAGES || name == _SC_AVPHYS_PAGES) {
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (!GlobalMemoryStatusEx(&ms)) return -1;
        unsigned __int64 bytes =
            (name == _SC_PHYS_PAGES) ? ms.ullTotalPhys : ms.ullAvailPhys;
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)(bytes / si.dwPageSize);
    }
    errno = EINVAL;
    return -1;
}

/* ds4_pread: true pread via OVERLAPPED ReadFile at an absolute offset.
 * Unlike the CRT _read (shared file position), this needs no lock, so the
 * layer-major prepare/pagein workers and the weight staging pipeline can
 * read the model file in parallel (the earlier slim-lock serialization
 * capped multi-threaded expert pagein at single-thread bandwidth). */
ssize_t ds4_pread(long fd, void *buf, size_t count, off_t offset) {
    HANDLE h = (HANDLE)_get_osfhandle((int)fd);
    if (h == INVALID_HANDLE_VALUE || h == NULL) {
        errno = EBADF;
        return -1;
    }
    uint8_t *p = (uint8_t *)buf;
    size_t done = 0;
    while (done < count) {
        size_t n = count - done;
        if (n > 0x40000000u) n = 0x40000000u; /* 1 GiB per read */
        OVERLAPPED ov;
        memset(&ov, 0, sizeof(ov));
        ov.Offset = (DWORD)(((uint64_t)(offset + (off_t)done)) & 0xFFFFFFFFull);
        ov.OffsetHigh =
            (DWORD)((((uint64_t)(offset + (off_t)done)) >> 32) & 0xFFFFFFFFull);
        DWORD got = 0;
        if (ReadFile(h, p + done, (DWORD)n, &got, &ov)) {
            done += got;
            if (got == 0) break; /* EOF */
            continue;
        }
        const DWORD err = GetLastError();
        if (err == ERROR_HANDLE_EOF) break;
        if (err == ERROR_IO_PENDING) {
            if (!GetOverlappedResult(h, &ov, &got, TRUE)) {
                errno = EIO;
                return -1;
            }
            done += got;
            if (got == 0) break;
            continue;
        }
        errno = EIO;
        return -1;
    }
    return (ssize_t)done;
}

void ds4_usleep(unsigned long usec) {
    if (usec == 0) {
        SwitchToThread();
        return;
    }
    DWORD ms = (DWORD)(usec / 1000);
    if (ms == 0) ms = 1;
    Sleep(ms);
}

unsigned int ds4_sleep(unsigned int seconds) {
    Sleep((DWORD)seconds * 1000);
    return 0;
}

int ds4_nanosleep(const struct timespec *req, struct timespec *rem) {
    (void)rem;
    unsigned __int64 ns = (unsigned __int64)req->tv_sec * 1000000000ULL +
                          (unsigned __int64)req->tv_nsec;
    DWORD ms = (DWORD)(ns / 1000000ULL);
    if (ms == 0 && ns > 0) ms = 1;
    Sleep(ms);
    return 0;
}

int ds4_fcntl(long fd, int cmd, ...) {
    (void)fd;
    if (cmd == F_SETFD || cmd == F_SETFL) return 0; /* no-op */
    if (cmd == F_GETFD || cmd == F_GETFL) return 0;
    errno = EINVAL;
    return -1;
}

int ds4_dprintf(long fd, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return -1;
    if ((size_t)n >= sizeof(buf)) n = (int)sizeof(buf) - 1;
    ssize_t w = ds4_write(fd, buf, (size_t)n);
    return (int)w;
}

struct tm *ds4_localtime_r(const time_t *t, struct tm *tm) {
    if (!tm) return NULL;
    if (localtime_s(tm, t) != 0) return NULL;
    return tm;
}

int ds4_mkstemp(char *tmpl) {
    /* Windows has _mktemp_s (unique name) but no atomic create; loop with
     * O_CREAT|O_EXCL to get mkstemp semantics. */
    size_t len = strlen(tmpl);
    for (int attempt = 0; attempt < 100; attempt++) {
        if (_mktemp_s(tmpl, len + 1) != 0) {
            errno = EINVAL;
            return -1;
        }
        int fd = _open(tmpl, _O_CREAT | _O_EXCL | _O_RDWR | _O_BINARY,
                       _S_IREAD | _S_IWRITE);
        if (fd >= 0) return fd;
        if (errno != EEXIST) return -1;
    }
    errno = EEXIST;
    return -1;
}

/* ================================================================== */
/* sigaction (minimal) + flock                                         */
/* ================================================================== */
int ds4_sigaction(int signum, const struct sigaction *act,
                  struct sigaction *oldact) {
    void (*prev)(int) = signal(signum, act ? act->sa_handler : SIG_DFL);
    if (prev == SIG_ERR) return -1;
    if (oldact) {
        oldact->sa_handler = prev;
        oldact->sa_mask = 0;
        oldact->sa_flags = 0;
    }
    return 0;
}

int ds4_flock(long fd, int operation) {
    HANDLE h = (HANDLE)_get_osfhandle((int)fd);
    if (h == INVALID_HANDLE_VALUE || h == NULL) return -1;
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    BOOL ok;
    if (operation & LOCK_UN) {
        ok = UnlockFileEx(h, 0, 1, 0, &ov);
    } else {
        DWORD flags = 0;
        if (operation & LOCK_NB) flags |= LOCKFILE_FAIL_IMMEDIATELY;
        ok = LockFileEx(h, flags, 0, 1, 0, &ov);
    }
    return ok ? 0 : -1;
}

/* ================================================================== */
/* readv / writev                                                      */
/* ================================================================== */
int ds4_readv(long fd, const struct iovec *iov, int iovcnt) {
    ssize_t total = 0;
    for (int i = 0; i < iovcnt; i++) {
        ssize_t n = ds4_read(fd, iov[i].iov_base, iov[i].iov_len);
        if (n < 0) return (total == 0) ? -1 : (int)total;
        total += n;
        if ((size_t)n < iov[i].iov_len) break;
    }
    return (int)total;
}

int ds4_writev(long fd, const struct iovec *iov, int iovcnt) {
    ssize_t total = 0;
    for (int i = 0; i < iovcnt; i++) {
        ssize_t n = ds4_write(fd, iov[i].iov_base, iov[i].iov_len);
        if (n < 0) return (total == 0) ? -1 : (int)total;
        total += n;
        if ((size_t)n < iov[i].iov_len) break;
    }
    return (int)total;
}

/* ================================================================== */
/* poll: mixed socket + CRT fd implementation                           */
/* ================================================================== */
int ds4_poll(struct pollfd *fds, unsigned long nfds, int timeout_ms) {
    if (nfds == 0) {
        if (timeout_ms > 0) Sleep((DWORD)timeout_ms);
        return 0;
    }
    DWORD deadline = 0;
    if (timeout_ms >= 0) deadline = GetTickCount() + (DWORD)timeout_ms;

    WSAPOLLFD *w = (WSAPOLLFD *)malloc(nfds * sizeof(WSAPOLLFD));
    if (!w) {
        errno = ENOMEM;
        return -1;
    }

    for (;;) {
        int ready = 0;

        /* 1) CRT fds: console stdin, pipes, files. */
        for (unsigned long i = 0; i < nfds; i++) {
            fds[i].revents = 0;
            intptr_t fd = fds[i].fd;
            if (is_socket_fd(fd)) continue;
            if (fd == STDIN_FILENO) {
                HANDLE ih = GetStdHandle(STD_INPUT_HANDLE);
                DWORD nevents = 0;
                if (GetNumberOfConsoleInputEvents(ih, &nevents) &&
                    nevents > 0) {
                    if (fds[i].events & POLLIN) {
                        fds[i].revents |= POLLIN;
                        ready++;
                    }
                } else {
                    DWORD avail = 0;
                    if (PeekNamedPipe(ih, NULL, 0, NULL, &avail, NULL) &&
                        avail > 0) {
                        if (fds[i].events & POLLIN) {
                            fds[i].revents |= POLLIN;
                            ready++;
                        }
                    } else if (GetLastError() == ERROR_BROKEN_PIPE) {
                        if (fds[i].events & POLLIN) {
                            fds[i].revents |= (POLLIN | POLLHUP);
                            ready++;
                        }
                    }
                }
            } else if (fd >= 0) {
                HANDLE h = (HANDLE)_get_osfhandle((int)fd);
                if (h && h != INVALID_HANDLE_VALUE) {
                    DWORD avail = 0;
                    if (PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL) &&
                        avail > 0) {
                        if (fds[i].events & POLLIN) {
                            fds[i].revents |= POLLIN;
                            ready++;
                        }
                    } else if (GetLastError() == ERROR_BROKEN_PIPE) {
                        if (fds[i].events & POLLIN) {
                            fds[i].revents |= (POLLIN | POLLHUP);
                            ready++;
                        }
                    }
                }
            }
        }

        /* 2) sockets via WSAPoll (zero wait: instant probe). */
        unsigned long nsock = 0;
        for (unsigned long i = 0; i < nfds; i++) {
            if (!is_socket_fd((intptr_t)fds[i].fd)) continue;
            w[nsock].fd = (SOCKET)fds[i].fd;
            w[nsock].events = 0;
            if (fds[i].events & POLLIN) w[nsock].events |= POLLRDNORM;
            if (fds[i].events & POLLOUT) w[nsock].events |= POLLWRNORM;
            w[nsock].revents = 0;
            nsock++;
        }
        if (nsock > 0) {
            int rc = WSAPoll(w, (ULONG)nsock, 0);
            if (rc == SOCKET_ERROR) {
                free(w);
                errno = EIO;
                return -1;
            }
            if (rc > 0) {
                unsigned long si = 0;
                for (unsigned long i = 0; i < nfds; i++) {
                    if (!is_socket_fd((intptr_t)fds[i].fd)) continue;
                    short rev = 0;
                    if (w[si].revents & (POLLRDNORM | POLLRDBAND))
                        rev |= POLLIN;
                    if (w[si].revents & (POLLWRNORM | POLLWRBAND))
                        rev |= POLLOUT;
                    if (w[si].revents & POLLERR) rev |= POLLERR;
                    if (w[si].revents & POLLHUP) rev |= POLLHUP;
                    if (w[si].revents & POLLNVAL) rev |= POLLNVAL;
                    fds[i].revents |= rev;
                    if (rev) ready++;
                    si++;
                }
            }
        }

        if (ready) {
            free(w);
            return ready;
        }
        if (timeout_ms >= 0 && (int)(GetTickCount() - deadline) >= 0) {
            free(w);
            return 0;
        }
        Sleep(5);
    }
}

/* ================================================================== */
/* process control (agent bash tool)                                   */
/* ================================================================== */
int ds4_pipe(int fds[2]) {
    return _pipe(fds, 4096, _O_BINARY);
}

pid_t ds4_waitpid(pid_t pid, int *status, int options) {
    if (pid <= 0) {
        errno = EINVAL;
        return -1;
    }
    HANDLE h = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                           FALSE, (DWORD)pid);
    if (!h) {
        errno = ECHILD;
        return -1;
    }
    DWORD code = STILL_ACTIVE;
    for (;;) {
        if (!GetExitCodeProcess(h, &code)) {
            CloseHandle(h);
            errno = ECHILD;
            return -1;
        }
        if (code != STILL_ACTIVE) {
            CloseHandle(h);
            if (status) *status = (int)code;
            return pid;
        }
        if (options & WNOHANG) {
            CloseHandle(h);
            return 0;
        }
        Sleep(10);
    }
}

int ds4_kill(pid_t pid, int sig) {
    (void)sig;
    if (pid < 0) pid = -pid; /* process-group kill: only the direct child
                                exists on Windows; kill it. */
    if (pid <= 0) {
        errno = EINVAL;
        return -1;
    }
    HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)pid);
    if (!h) {
        errno = ESRCH;
        return -1;
    }
    BOOL ok = TerminateProcess(h, 1);
    CloseHandle(h);
    return ok ? 0 : -1;
}

/* ================================================================== */
/* pthread subset                                                      */
/* ================================================================== */
int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *attr) {
    (void)attr;
    if (!m) return EINVAL;
    if (!m->initialized) {
        InitializeCriticalSection(&m->cs);
        m->initialized = 1;
    }
    return 0;
}

int pthread_mutex_destroy(pthread_mutex_t *m) {
    if (!m) return EINVAL;
    if (m->initialized) {
        DeleteCriticalSection(&m->cs);
        m->initialized = 0;
    }
    return 0;
}

int pthread_mutex_lock(pthread_mutex_t *m) {
    if (!m) return EINVAL;
    if (!m->initialized) pthread_mutex_init(m, NULL);
    EnterCriticalSection(&m->cs);
    return 0;
}

int pthread_mutex_trylock(pthread_mutex_t *m) {
    if (!m) return EINVAL;
    if (!m->initialized) pthread_mutex_init(m, NULL);
    return TryEnterCriticalSection(&m->cs) ? 0 : EBUSY;
}

int pthread_mutex_unlock(pthread_mutex_t *m) {
    if (!m || !m->initialized) return EINVAL;
    LeaveCriticalSection(&m->cs);
    return 0;
}

int pthread_mutexattr_init(pthread_mutexattr_t *attr) {
    attr->type = PTHREAD_MUTEX_NORMAL;
    return 0;
}

int pthread_mutexattr_destroy(pthread_mutexattr_t *attr) {
    (void)attr;
    return 0;
}

int pthread_mutexattr_settype(pthread_mutexattr_t *attr, int type) {
    attr->type = type;
    return 0;
}

int pthread_cond_init(pthread_cond_t *c, const void *attr) {
    (void)attr;
    if (!c) return EINVAL;
    if (!c->initialized) {
        InitializeConditionVariable(&c->cv);
        c->initialized = 1;
    }
    return 0;
}

int pthread_cond_destroy(pthread_cond_t *c) {
    (void)c;
    return 0;
}

int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    if (!m->initialized) pthread_mutex_init(m, NULL);
    if (!c->initialized) pthread_cond_init(c, NULL);
    SleepConditionVariableCS(&c->cv, &m->cs, INFINITE);
    return 0;
}

int pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m,
                           const struct timespec *abstime) {
    if (!m->initialized) pthread_mutex_init(m, NULL);
    if (!c->initialized) pthread_cond_init(c, NULL);
    struct timespec now;
    ds4_clock_gettime(CLOCK_REALTIME, &now);
    __int64 dsec = abstime->tv_sec - now.tv_sec;
    __int64 dnsec = abstime->tv_nsec - now.tv_nsec;
    if (dnsec < 0) {
        dsec--;
        dnsec += 1000000000LL;
    }
    if (dsec < 0) {
        LeaveCriticalSection(&m->cs);
        EnterCriticalSection(&m->cs);
        return ETIMEDOUT;
    }
    DWORD ms = (DWORD)dsec * 1000 + (DWORD)(dnsec / 1000000);
    if (ms == 0) ms = 1;
    if (SleepConditionVariableCS(&c->cv, &m->cs, ms)) return 0;
    return (GetLastError() == ERROR_TIMEOUT) ? ETIMEDOUT : EINVAL;
}

int pthread_cond_signal(pthread_cond_t *c) {
    if (!c->initialized) pthread_cond_init(c, NULL);
    WakeConditionVariable(&c->cv);
    return 0;
}

int pthread_cond_broadcast(pthread_cond_t *c) {
    if (!c->initialized) pthread_cond_init(c, NULL);
    WakeAllConditionVariable(&c->cv);
    return 0;
}

int pthread_once(pthread_once_t *once_control, void (*init_routine)(void)) {
    BOOL ok = InitOnceExecuteOnce(&once_control->once,
                                  (PINIT_ONCE_FN)init_routine, NULL, NULL);
    return ok ? 0 : EINVAL;
}

typedef struct {
    void *(*start_routine)(void *);
    void *arg;
} ds4_thread_start_t;

static unsigned __stdcall ds4_thread_wrapper(void *p) {
    ds4_thread_start_t *s = (ds4_thread_start_t *)p;
    void *(*fn)(void *) = s->start_routine;
    void *arg = s->arg;
    free(s);
    if (fn) fn(arg);
    return 0;
}

int pthread_create(pthread_t *thread, const void *attr,
                   void *(*start_routine)(void *), void *arg) {
    (void)attr;
    ds4_thread_start_t *s =
        (ds4_thread_start_t *)malloc(sizeof(ds4_thread_start_t));
    if (!s) return ENOMEM;
    s->start_routine = start_routine;
    s->arg = arg;
    uintptr_t h = _beginthreadex(NULL, 0, ds4_thread_wrapper, s, 0, NULL);
    if (h == 0) {
        free(s);
        return EAGAIN;
    }
    *thread = (HANDLE)h;
    return 0;
}

int pthread_join(pthread_t thread, void **value_ptr) {
    (void)value_ptr;
    if (!thread) return EINVAL;
    WaitForSingleObject((HANDLE)thread, INFINITE);
    CloseHandle((HANDLE)thread);
    return 0;
}

int pthread_detach(pthread_t thread) {
    if (!thread) return EINVAL;
    CloseHandle((HANDLE)thread);
    return 0;
}

int pthread_equal(pthread_t a, pthread_t b) {
    return GetThreadId((HANDLE)a) == GetThreadId((HANDLE)b);
}

pthread_t pthread_self(void) {
    return (pthread_t)GetCurrentThread();
}

/* ================================================================== */
/* termios (console-mode approximation) + ioctl(TIOCGWINSZ)             */
/* ================================================================== */
static HANDLE termios_handle_for(int fd) {
    if (fd == 0) return GetStdHandle(STD_INPUT_HANDLE);
    if (fd == 1) return GetStdHandle(STD_OUTPUT_HANDLE);
    return GetStdHandle(STD_ERROR_HANDLE);
}

int ds4_tcgetattr(int fd, struct termios *t) {
    HANDLE h = termios_handle_for(fd);
    DWORD mode = 0;
    if (!GetConsoleMode(h, &mode)) {
        errno = ENOTTY;
        return -1;
    }
    memset(t, 0, sizeof(*t));
    if (mode & ENABLE_LINE_INPUT) t->c_lflag |= ICANON;
    if (mode & ENABLE_ECHO_INPUT) t->c_lflag |= ECHO;
    if (mode & ENABLE_PROCESSED_INPUT) t->c_lflag |= (IEXTEN | ISIG);
    t->c_cc[VMIN] = 1;
    t->c_cc[VTIME] = 0;
    return 0;
}

int ds4_tcsetattr(int fd, int action, const struct termios *t) {
    (void)action;
    HANDLE h = termios_handle_for(fd);
    DWORD mode = 0;
    if (!GetConsoleMode(h, &mode)) {
        errno = ENOTTY;
        return -1;
    }
    mode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT);
    if (t->c_lflag & ICANON) mode |= ENABLE_LINE_INPUT;
    if (t->c_lflag & ECHO) mode |= ENABLE_ECHO_INPUT;
    if (t->c_lflag & (IEXTEN | ISIG)) mode |= ENABLE_PROCESSED_INPUT;
    DWORD mode_vt = mode | ENABLE_VIRTUAL_TERMINAL_INPUT;
    if (!SetConsoleMode(h, mode_vt) && !SetConsoleMode(h, mode)) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}

int ds4_ioctl(int fd, unsigned long request, void *arg) {
    if (request == TIOCGWINSZ) {
        HANDLE h = termios_handle_for(fd == 1 ? 1 : 0);
        CONSOLE_SCREEN_BUFFER_INFO info;
        struct winsize *ws = (struct winsize *)arg;
        if (!ws) {
            errno = EINVAL;
            return -1;
        }
        if (!GetConsoleScreenBufferInfo(h, &info)) {
            errno = ENOTTY;
            return -1;
        }
        ws->ws_row = (unsigned short)info.dwSize.Y;
        ws->ws_col = (unsigned short)info.dwSize.X;
        ws->ws_xpixel = 0;
        ws->ws_ypixel = 0;
        return 0;
    }
    errno = ENOTTY;
    return -1;
}

/* ================================================================== */
/* dirent (FindFirstFile/FindNextFile)                                  */
/* ================================================================== */
DIR *ds4_opendir(const char *path) {
    size_t len = strlen(path);
    char pattern[2048];
    if (len + 3 >= sizeof(pattern)) {
        errno = ENAMETOOLONG;
        return NULL;
    }
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    DIR *d = (DIR *)calloc(1, sizeof(DIR));
    if (!d) {
        errno = ENOMEM;
        return NULL;
    }
    d->handle = FindFirstFileA(pattern, &d->data);
    if (d->handle == INVALID_HANDLE_VALUE) {
        free(d);
        errno = ENOENT;
        return NULL;
    }
    d->first = 1;
    return d;
}

struct dirent *ds4_readdir(DIR *dir) {
    if (!dir || dir->handle == INVALID_HANDLE_VALUE) return NULL;
    if (!dir->first) {
        if (!FindNextFileA(dir->handle, &dir->data)) return NULL;
    }
    dir->first = 0;
    strncpy(dir->ent.d_name, dir->data.cFileName, sizeof(dir->ent.d_name) - 1);
    dir->ent.d_name[sizeof(dir->ent.d_name) - 1] = '\0';
    dir->ent.d_type =
        (dir->data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? DT_DIR : DT_REG;
    return &dir->ent;
}

int ds4_closedir(DIR *dir) {
    if (!dir) {
        errno = EINVAL;
        return -1;
    }
    if (dir->handle != INVALID_HANDLE_VALUE) FindClose(dir->handle);
    free(dir);
    return 0;
}

/* ================================================================== */
/* fnmatch (minimal glob: '*', '?')                                     */
/* ================================================================== */
static int ds4_fnmatch_impl(const char *p, const char *s) {
    while (*p) {
        if (*p == '*') {
            while (*p == '*') p++;
            if (!*p) return 0;
            for (const char *t = s;; t++) {
                if (ds4_fnmatch_impl(p, t) == 0) return 0;
                if (!*t) break;
            }
            return FNM_NOMATCH;
        } else if (*p == '?') {
            if (!*s) return FNM_NOMATCH;
            p++;
            s++;
        } else {
            if (tolower((unsigned char)*p) != tolower((unsigned char)*s))
                return FNM_NOMATCH;
            p++;
            s++;
        }
    }
    return (*s == '\0') ? 0 : FNM_NOMATCH;
}

int ds4_fnmatch(const char *pattern, const char *string, int flags) {
    (void)flags;
    if (!pattern || !string) return FNM_NOMATCH;
    return ds4_fnmatch_impl(pattern, string);
}

/* ================================================================== */
/* strings                                                             */
/* ================================================================== */
char *ds4_strcasestr(const char *haystack, const char *needle) {
    if (!*needle) return (char *)haystack;
    for (; *haystack; haystack++) {
        const char *h = haystack, *n = needle;
        while (*h && *n && tolower((unsigned char)*h) ==
                              tolower((unsigned char)*n)) {
            h++;
            n++;
        }
        if (!*n) return (char *)haystack;
    }
    return NULL;
}

char *ds4_strsep(char **stringp, const char *delim) {
    char *s = *stringp;
    if (!s) return NULL;
    char *p = s + strcspn(s, delim);
    if (*p) {
        *p = '\0';
        *stringp = p + 1;
    } else {
        *stringp = NULL;
    }
    return s;
}

#endif /* _WIN32 */
