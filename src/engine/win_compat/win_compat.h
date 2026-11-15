/* win_compat.h - POSIX compatibility layer for building DwarfStar (ds4)
 * natively on Windows with MSVC/clang-cl.
 *
 * The upstream codebase is written for POSIX (Linux/macOS). This header is
 * force-included (/FI) on Windows builds and maps the POSIX surface used by
 * ds4 onto the Windows CRT and Win32 APIs. Implementations live in
 * win_compat.c, which must be compiled and linked into every binary.
 *
 * Note: this file is written in C99 but must also be safe for C++ TUs
 * (the .cu files are compiled as C++ by nvcc).
 */
#ifndef DS4_WIN_COMPAT_H
#define DS4_WIN_COMPAT_H

#ifdef _WIN32

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#ifndef _WINSOCK_DEPRECATED_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NTDDI_VERSION
#define NTDDI_VERSION 0x0A00000A
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <process.h>
#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utime.h>
#include <malloc.h>
#include <intrin.h>
#include <time.h>
#include <signal.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* 64-bit file offsets. MSVC's plain stat has a 32-bit st_size, which  */
/* cannot represent the ~81 GB GGUF files. Force the _stat64 family.   */
/* off_t is macro-shadowed AFTER the system includes so every use in    */
/* the ds4 sources resolves to __int64 without a typedef conflict.      */
/* ------------------------------------------------------------------ */
#define off_t __int64
typedef __int64 ssize_t;
#define SSIZE_MAX 0x7fffffffffffffffLL
typedef long suseconds_t;
typedef long useconds_t;
typedef int pid_t;
typedef int mode_t;

#ifndef S_IRUSR
#define S_IRUSR 0x100
#define S_IWUSR 0x80
#define S_IXUSR 0x40
#define S_IRWXG 0x38
#define S_IRWXO 0x7
#endif
/* Windows files have no POSIX mode bits; chmod on an open handle is a
 * safe no-op for the ds4 use (linenoise history file). */
#define fchmod(fd, mode) 0
#define stat _stat64
#define fstat _fstat64
#define lstat _stat64
#define lseek _lseeki64
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISLNK
#define S_ISLNK(m) 0
#endif
#ifndef S_IFMT
#define S_IFMT _S_IFMT
#define S_IFREG _S_IFREG
#define S_IFDIR _S_IFDIR
#endif
typedef unsigned long nfds_t;

/* ------------------------------------------------------------------ */
/* unistd / file API mapping                                            */
/* ------------------------------------------------------------------ */
#define F_OK 0
#define R_OK 4
#define W_OK 2
#define X_OK 1
#define PATH_MAX 4096
#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#endif

#define O_RDONLY _O_RDONLY
#define O_WRONLY _O_WRONLY
#define O_RDWR _O_RDWR
#define O_CREAT _O_CREAT
#define O_TRUNC _O_TRUNC
#define O_APPEND _O_APPEND
#define O_BINARY _O_BINARY
#define O_EXCL _O_EXCL
#ifndef O_SEQUENTIAL
/* Sequential-scan cache hint (MSVCRT _O_SEQUENTIAL): the cache manager
 * reuses a small window instead of pinning the whole sweep (see
 * FUSION_REPORT, stage-read LRU). */
#define O_SEQUENTIAL _O_SEQUENTIAL
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

/* Always open in binary mode: the CRT default text mode truncates reads at
 * Ctrl-Z (0x1A) bytes, which appear everywhere in GGUF tensor data. */
int ds4_open(const char *path, int oflag, ...);
#define open ds4_open
#define unlink _unlink
#define ftruncate _chsize_s
#define fsync _commit
#define fdatasync _commit
#define mkdir(path, mode) _mkdir(path)
#define access _access
#define isatty _isatty
#define fileno _fileno
#define getpid _getpid
#define chdir _chdir
#define getcwd _getcwd
#define fdopen _fdopen
#define strdup _strdup
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#define umask(x) 0

/* Advisory file locking (used for the /tmp/ds4.lock singleton guard). */
#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8
int ds4_flock(long fd, int operation);
#define flock ds4_flock

/* Hybrid fd close/read/write: the same call sites handle both CRT file
 * descriptors and socket handles. Implementation probes with getsockopt. */
#define close ds4_close
#define read ds4_read
#define write ds4_write
#define socket ds4_socket

long ds4_close(long fd);
ssize_t ds4_read(long fd, void *buf, size_t len);
ssize_t ds4_write(long fd, const void *buf, size_t len);
int ds4_socket(int af, int type, int protocol);

/* ------------------------------------------------------------------ */
/* memory mapping                                                       */
/* ------------------------------------------------------------------ */
#define MAP_SHARED 0x0001
#define MAP_PRIVATE 0x0002
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS 0x0020
#endif
#ifndef MAP_ANON
#define MAP_ANON MAP_ANONYMOUS
#endif
#define MAP_NORESERVE 0x0040
#define MAP_FAILED ((void *)-1)
#define PROT_NONE 0x0
#define PROT_READ 0x1
#define PROT_WRITE 0x2

#define MADV_RANDOM 0
#define MADV_SEQUENTIAL 1
#define MADV_WILLNEED 2
#define MADV_DONTNEED 3
#define MADV_HUGEPAGE 4
#define MADV_NORMAL 5

void *ds4_mmap(void *addr, size_t length, int prot, int flags, long fd, off_t offset);
int ds4_munmap(void *addr, size_t length);
int ds4_madvise(void *addr, size_t length, int advice);
int ds4_mlock(const void *addr, size_t length);
int ds4_munlock(const void *addr, size_t length);
int ds4_msync(void *addr, size_t length, int flags);

#define mmap ds4_mmap
#define munmap ds4_munmap
#define madvise ds4_madvise
#define mlock ds4_mlock
#define munlock ds4_munlock
#define msync ds4_msync
#define MS_SYNC 1
#define MS_ASYNC 2

/* ------------------------------------------------------------------ */
/* clocks (struct timeval comes from winsock2.h; struct timespec is    */
/* provided by the Windows 10+ SDK <time.h>)                            */
/* ------------------------------------------------------------------ */
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#define CLOCK_UPTIME_RAW 6
#define CLOCK_REALTIME_COARSE 7
#define CLOCK_MONOTONIC_COARSE 8
#define TIMER_ABSTIME 1

int ds4_clock_gettime(int clk_id, struct timespec *ts);
int ds4_gettimeofday(struct timeval *tv, void *tz);

#define clock_gettime ds4_clock_gettime
#define gettimeofday ds4_gettimeofday

/* ------------------------------------------------------------------ */
/* sysconf / pread / misc                                               */
/* ------------------------------------------------------------------ */
#define _SC_PAGESIZE 30
#define _SC_NPROCESSORS_ONLN 84
#define _SC_PHYS_PAGES 85
#define _SC_AVPHYS_PAGES 86

long ds4_sysconf(int name);
ssize_t ds4_pread(long fd, void *buf, size_t count, off_t offset);

#define sysconf ds4_sysconf
#define pread ds4_pread
#define getpagesize() ((int)ds4_sysconf(_SC_PAGESIZE))

void ds4_usleep(unsigned long usec);
unsigned int ds4_sleep(unsigned int seconds);
#define usleep ds4_usleep
#define sleep ds4_sleep

int ds4_nanosleep(const struct timespec *req, struct timespec *rem);
#define nanosleep ds4_nanosleep

/* fcntl: only F_SETFD/F_GETFD(FD_CLOEXEC) appear in ds4; no-op on Windows. */
#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4
#define FD_CLOEXEC 1
#define O_NONBLOCK 0x1000
int ds4_fcntl(long fd, int cmd, ...);
#define fcntl ds4_fcntl

int ds4_dprintf(long fd, const char *fmt, ...);
#define dprintf ds4_dprintf

#define ftello _ftelli64
#define fseeko _fseeki64

int ds4_mkstemp(char *tmpl);
#define mkstemp ds4_mkstemp

/* Windows console uses a legacy codepage; force UTF-8 so model output
 * and prompts render correctly. Called from each binary's main(). */
void ds4_win_init(void);

/* Enable SeLockMemoryPrivilege on the current token so pinned allocations
 * (cudaHostAlloc/cudaHostRegister) are not capped at ~1 GiB by the WDDM
 * locked-page pool. No-op when the policy does not grant the privilege.
 * Called from the CUDA backend init. */
void ds4_win_enable_lock_memory(void);

/* ------------------------------------------------------------------ */
/* poll(): mixed socket + CRT-fd implementation (WSAPoll for sockets,  */
/* console/pipe readiness probes for stdin and pipes). Always routed    */
/* through ds4_poll so the agent UI can multiplex stdin + wake pipes.   */
/* ------------------------------------------------------------------ */
#ifdef POLLIN
/* Recent SDK: struct pollfd and POLL* come from winsock2.h. */
int ds4_poll(struct pollfd *fds, unsigned long nfds, int timeout_ms);
#define poll ds4_poll
#else
struct pollfd {
    int fd;
    short events;
    short revents;
};
#define POLLIN 0x0001
#define POLLOUT 0x0004
#define POLLERR 0x0008
#define POLLHUP 0x0010
#define POLLNVAL 0x0020

int ds4_poll(struct pollfd *fds, unsigned long nfds, int timeout_ms);
#define poll ds4_poll
#endif

/* ------------------------------------------------------------------ */
/* process control (used by the ds4-agent bash tool)                    */
/* ------------------------------------------------------------------ */
#ifndef SIGKILL
#define SIGKILL 9
#endif
#define WNOHANG 1
#define WIFEXITED(s) 1
#define WEXITSTATUS(s) ((s) & 0xff)
#define WIFSIGNALED(s) 0
#define WTERMSIG(s) 0

pid_t ds4_waitpid(pid_t pid, int *status, int options);
int ds4_kill(pid_t pid, int sig);
#define waitpid ds4_waitpid
#define kill ds4_kill
#define setpgid(pid, pgid) 0
#define getpgid(pid) 0
#define dup2 _dup2
#define pipe ds4_pipe
int ds4_pipe(int fds[2]);

/* ------------------------------------------------------------------ */
/* pthread subset: mutex (recursive-capable), condition variable with  */
/* timed wait, once, and thread create/join/detach. No rwlocks needed. */
/* ------------------------------------------------------------------ */
typedef struct {
    int initialized;
    CRITICAL_SECTION cs;
} pthread_mutex_t;
#define PTHREAD_MUTEX_INITIALIZER {0, {0}}

typedef struct {
    int initialized;
    CONDITION_VARIABLE cv;
} pthread_cond_t;
#define PTHREAD_COND_INITIALIZER {0, {0}}

typedef struct {
    int type;
} pthread_mutexattr_t;
#define PTHREAD_MUTEX_NORMAL 0
#define PTHREAD_MUTEX_RECURSIVE 1

typedef struct {
    int initialized;
    INIT_ONCE once;
} pthread_once_t;
#define PTHREAD_ONCE_INIT {0, {0}}

/* Pointer-typed pthread_t keeps `if (thread)` truthiness tests valid. */
typedef void *pthread_t;

int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *attr);
int pthread_mutex_destroy(pthread_mutex_t *m);
int pthread_mutex_lock(pthread_mutex_t *m);
int pthread_mutex_trylock(pthread_mutex_t *m);
int pthread_mutex_unlock(pthread_mutex_t *m);
int pthread_mutexattr_init(pthread_mutexattr_t *attr);
int pthread_mutexattr_destroy(pthread_mutexattr_t *attr);
int pthread_mutexattr_settype(pthread_mutexattr_t *attr, int type);

int pthread_cond_init(pthread_cond_t *c, const void *attr);
int pthread_cond_destroy(pthread_cond_t *c);
int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m);
int pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m,
                           const struct timespec *abstime);
int pthread_cond_signal(pthread_cond_t *c);
int pthread_cond_broadcast(pthread_cond_t *c);

int pthread_once(pthread_once_t *once_control, void (*init_routine)(void));

int pthread_create(pthread_t *thread, const void *attr,
                   void *(*start_routine)(void *), void *arg);
int pthread_join(pthread_t thread, void **value_ptr);
int pthread_detach(pthread_t thread);
int pthread_equal(pthread_t a, pthread_t b);
pthread_t pthread_self(void);

/* ------------------------------------------------------------------ */
/* misc GNU-isms                                                        */
/* ------------------------------------------------------------------ */
#ifndef __thread
#define __thread __declspec(thread)
#endif

char *ds4_strcasestr(const char *haystack, const char *needle);
char *ds4_strsep(char **stringp, const char *delim);
#define strcasestr ds4_strcasestr
#define strsep ds4_strsep

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#ifndef socklen_t
typedef int socklen_t;
#endif
/* SO_NOSIGPIPE is intentionally NOT defined: the upstream code guards its
 * use with #ifdef and skips it entirely on platforms without it. */

/* Winsock recv/send with MSG_DONTWAIT semantics (non-blocking attempt). */
int ds4_recv_dontwait(long fd, void *buf, size_t len);
int ds4_send_dontwait(long fd, const void *buf, size_t len);

/* setsockopt/getsockopt: the ds4 call sites pass int* optval; Winsock
 * declares a char pointer there, which clang rejects as incompatible. */
int ds4_setsockopt(long fd, int level, int optname, const void *optval,
                   int optlen);
#define setsockopt(fd, level, optname, optval, optlen) \
    ds4_setsockopt((long)(fd), (level), (optname), (optval), (optlen))
int ds4_getsockopt(long fd, int level, int optname, void *optval,
                   int *optlen);
#define getsockopt(fd, level, optname, optval, optlen) \
    ds4_getsockopt((long)(fd), (level), (optname), (optval), (optlen))

#define strtok_r(s, d, p) strtok_s((s), (d), (p))

struct tm *ds4_localtime_r(const time_t *t, struct tm *tm);
#define localtime_r(t, tm) ds4_localtime_r((t), (tm))

#ifndef SHUT_RD
#define SHUT_RD 0
#define SHUT_WR 1
#define SHUT_RDWR 2
#endif

/* ------------------------------------------------------------------ */
/* errno constants MSVC may not define                                  */
/* ------------------------------------------------------------------ */
#ifndef ETIMEDOUT
#define ETIMEDOUT 138
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK 140
#endif
#ifndef EAGAIN
#define EAGAIN 11
#endif

/* ------------------------------------------------------------------ */
/* minimal sigaction for SIGINT/SIGTERM handlers                        */
/* ------------------------------------------------------------------ */
typedef int sigset_t;
struct sigaction {
    void (*sa_handler)(int);
    sigset_t sa_mask;
    int sa_flags;
};
int ds4_sigaction(int signum, const struct sigaction *act,
                  struct sigaction *oldact);
/* Function-like macro: expands only at call sites, so the `struct
 * sigaction` tag above keeps its name. */
#define sigaction(signum, act, oldact) \
    ds4_sigaction((signum), (act), (oldact))
#define sigemptyset(s) ((void)(*(s) = 0))
#define sigfillset(s) ((void)(*(s) = ~0))

/* ------------------------------------------------------------------ */
/* iovec (readv/writev for file descriptors)                            */
/* ------------------------------------------------------------------ */
#ifndef _DS4_IOVEC_DEFINED
#define _DS4_IOVEC_DEFINED
struct iovec {
    void *iov_base;
    size_t iov_len;
};
#endif
int ds4_readv(long fd, const struct iovec *iov, int iovcnt);
int ds4_writev(long fd, const struct iovec *iov, int iovcnt);
#define readv ds4_readv
#define writev ds4_writev

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _WIN32 */
#endif /* DS4_WIN_COMPAT_H */
