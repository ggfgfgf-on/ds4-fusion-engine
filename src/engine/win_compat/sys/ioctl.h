/* Windows shim for <sys/ioctl.h>: TIOCGWINSZ via console screen buffer. */
#pragma once
#include "win_compat.h"

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif

struct winsize {
    unsigned short ws_row;
    unsigned short ws_col;
    unsigned short ws_xpixel;
    unsigned short ws_ypixel;
};

#define TIOCGWINSZ 0x5413

int ds4_ioctl(int fd, unsigned long request, void *arg);
#define ioctl(fd, request, arg) ds4_ioctl((fd), (request), (arg))

#ifdef __cplusplus
}
#endif
#endif /* _WIN32 */
