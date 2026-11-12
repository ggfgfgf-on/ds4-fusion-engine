#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include "win_compat.h"

int main(void) {
    const char *path = "D:\\DSHwork\\models\\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";
    int fd = open(path, O_RDONLY);
    printf("open fd=%d errno=%d (%s)\n", fd, errno, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) { printf("fstat fail errno=%d\n", errno); return 1; }
    printf("size=%lld\n", (long long)st.st_size);
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    printf("map=%p\n", map);
    void *map2 = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    printf("map2=%p\n", map2);
    char buf[4096];
    errno = 0;
    ssize_t n = pread(fd, buf, sizeof(buf), 0);
    printf("pread1 n=%zd errno=%d (%s)\n", n, errno, strerror(errno));
    if (n > 0) printf("magic=%02x %02x %02x %02x\n", (unsigned char)buf[0], (unsigned char)buf[1], (unsigned char)buf[2], (unsigned char)buf[3]);
    errno = 0;
    n = pread(fd, buf, sizeof(buf), 4096);
    printf("pread2 n=%zd errno=%d (%s)\n", n, errno, strerror(errno));
    errno = 0;
    n = pread(fd, buf, 1024, 1048576);
    printf("pread3 n=%zd errno=%d (%s)\n", n, errno, strerror(errno));
    return 0;
}
