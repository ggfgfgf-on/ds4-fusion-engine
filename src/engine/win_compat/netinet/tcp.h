/* Windows shim for <netinet/tcp.h>; TCP_NODELAY and friends live in
 * winsock2.h (already included via win_compat.h). */
#pragma once
#include "win_compat.h"
