/* Windows shim for <netdb.h>: struct addrinfo, getaddrinfo/freeaddrinfo/
 * getnameinfo and the AI_* / NI_* constants are provided by ws2tcpip.h,
 * which win_compat.h includes. */
#pragma once
#include "win_compat.h"
