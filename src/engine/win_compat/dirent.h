/* Windows shim for <dirent.h>: minimal opendir/readdir/closedir over
 * FindFirstFile/FindNextFile, enough for ds4_kvstore directory scans. */
#pragma once
#include "win_compat.h"

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif

#define DT_UNKNOWN 0
#define DT_REG 1
#define DT_DIR 2

struct dirent {
    char d_name[512];
    unsigned char d_type;
};

typedef struct ds4_DIR {
    HANDLE handle;
    WIN32_FIND_DATAA data;
    int first;
    struct dirent ent;
} DIR;

DIR *ds4_opendir(const char *path);
struct dirent *ds4_readdir(DIR *dir);
int ds4_closedir(DIR *dir);

#define opendir ds4_opendir
#define readdir ds4_readdir
#define closedir ds4_closedir

#ifdef __cplusplus
}
#endif
#endif /* _WIN32 */
