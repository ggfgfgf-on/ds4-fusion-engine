/* Windows shim for <regex.h>: minimal POSIX regex over std::regex
 * (implemented in win_compat_regex.cpp). Supports REG_EXTENDED,
 * REG_NOSUB, REG_ICASE and regexec without pmatch - exactly what
 * ds4_agent.c needs. */
#pragma once
#include "win_compat.h"

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif

#define REG_EXTENDED 1
#define REG_NOSUB 2
#define REG_ICASE 4
#define REG_NOMATCH 1

typedef struct {
    void *impl; /* std::regex * */
    int flags;
} regex_t;

int ds4_regcomp(regex_t *preg, const char *pattern, int cflags);
int ds4_regexec(const regex_t *preg, const char *string, size_t nmatch,
                void *pmatch, int eflags);
size_t ds4_regerror(int errcode, const regex_t *preg, char *errbuf,
                    size_t errbuf_size);
void ds4_regfree(regex_t *preg);

#define regcomp ds4_regcomp
#define regexec ds4_regexec
#define regerror ds4_regerror
#define regfree ds4_regfree

#ifdef __cplusplus
}
#endif
#endif /* _WIN32 */
