/* Windows shim for <fnmatch.h>: minimal glob-style matcher supporting
 * '*', '?' and character classes-free literal text (enough for the
 * ds4-agent file filter). */
#pragma once
#include "win_compat.h"

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif

#define FNM_NOMATCH 1
#define FNM_PATHNAME (1 << 0)
#define FNM_PERIOD (1 << 1)

int ds4_fnmatch(const char *pattern, const char *string, int flags);
#define fnmatch ds4_fnmatch

#ifdef __cplusplus
}
#endif
#endif /* _WIN32 */
