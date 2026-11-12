// win_compat_regex.cpp - POSIX regex surface for Windows builds,
// implemented with std::regex in POSIX-extended mode.
#include "win_compat.h"
#include "regex.h"

#ifdef _WIN32

#include <regex>
#include <string>

int ds4_regcomp(regex_t *preg, const char *pattern, int cflags) {
    if (!preg || !pattern) return 1;
    try {
        auto flags = std::regex_constants::extended;
        if (cflags & REG_ICASE) flags |= std::regex_constants::icase;
        std::regex *re = new std::regex(pattern, flags);
        if (preg->impl) delete static_cast<std::regex *>(preg->impl);
        preg->impl = re;
        preg->flags = cflags;
        return 0;
    } catch (const std::regex_error &) {
        return 2; /* REG_BADPAT */
    } catch (...) {
        return 2;
    }
}

int ds4_regexec(const regex_t *preg, const char *string, size_t nmatch,
                void *pmatch, int eflags) {
    (void)nmatch;  /* REG_NOSUB is always set by the caller */
    (void)pmatch;
    (void)eflags;
    if (!preg || !preg->impl || !string) return REG_NOMATCH;
    try {
        auto *re = static_cast<std::regex *>(preg->impl);
        return std::regex_search(std::string(string), *re) ? 0 : REG_NOMATCH;
    } catch (...) {
        return REG_NOMATCH;
    }
}

size_t ds4_regerror(int errcode, const regex_t *preg, char *errbuf,
                    size_t errbuf_size) {
    (void)preg;
    const char *msg = (errcode == REG_NOMATCH)
                          ? "no match"
                          : "invalid regular expression";
    if (errbuf && errbuf_size) {
        strncpy(errbuf, msg, errbuf_size - 1);
        errbuf[errbuf_size - 1] = '\0';
    }
    return strlen(msg);
}

void ds4_regfree(regex_t *preg) {
    if (!preg) return;
    if (preg->impl) {
        delete static_cast<std::regex *>(preg->impl);
        preg->impl = nullptr;
    }
}

#endif /* _WIN32 */
