/* Windows shim for <net/if.h>: if_nametoindex from iphlpapi; IFF_* flags
 * already come from ws2ipdef.h via win_compat.h. */
#pragma once
#include "win_compat.h"

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
#ifndef IF_NAMESIZE
#define IF_NAMESIZE 16
#endif
#ifdef __cplusplus
}
#endif
#endif
