/* Windows shim for <sys/wait.h>: the waitpid/fork sites in ds4_web.c are
 * guarded by #ifdef _WIN32 and replaced with Win32 process APIs; this
 * header only satisfies the unconditional #include. */
#pragma once
#include "win_compat.h"
