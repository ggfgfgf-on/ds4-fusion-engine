/* Windows shim for <termios.h>: console-mode based approximation of the
 * POSIX terminal attributes used by ds4 (linenoise raw mode + eval TUI). */
#pragma once
#include "win_compat.h"

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long tcflag_t;
typedef unsigned char cc_t;

struct termios {
    tcflag_t c_iflag;
    tcflag_t c_oflag;
    tcflag_t c_cflag;
    tcflag_t c_lflag;
    cc_t c_cc[16];
};

/* c_iflag */
#define ICRNL 0x0001
#define BRKINT 0x0002
#define INPCK 0x0004
#define ISTRIP 0x0008
#define IXON 0x0010
/* c_oflag */
#define OPOST 0x0100
/* c_cflag */
#define CS8 0x0200
/* c_lflag */
#define ICANON 0x0400
#define ECHO 0x0800
#define IEXTEN 0x1000
#define ISIG 0x2000

/* c_cc indices */
#define VMIN 5
#define VTIME 6

/* actions */
#define TCSANOW 0
#define TCSADRAIN 1
#define TCSAFLUSH 2

int ds4_tcgetattr(int fd, struct termios *t);
int ds4_tcsetattr(int fd, int action, const struct termios *t);

#define tcgetattr ds4_tcgetattr
#define tcsetattr ds4_tcsetattr

#ifdef __cplusplus
}
#endif
#endif /* _WIN32 */
