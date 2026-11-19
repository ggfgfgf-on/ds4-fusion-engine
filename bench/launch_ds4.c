/* Launch the engine with its arguments baked in, so uProf can profile it.
 *
 * uProfCLI parses the command line of the program it launches, and the engine's
 * own options are spelled --cpu, --ctx and so on, which uProf takes for its own.
 * There is no separate-arguments separator that works, so the arguments live
 * here instead and uProf sees a program it can start with none. */
#include <windows.h>
#include <stdio.h>

int main(void)
{
    char cmd[8192];
    snprintf(cmd, sizeof cmd,
        "\"%s\" --cpu -m \"%s\" --ctx 2048 -p \"%s\" -n 1",
        "D:\\DSHwork\\ds4-upstream\\ds4.exe",
        "D:\\DSHwork\\models\\Qwen3.8-27B-Uncensored-Q4_K_M.gguf",
        "Explain in detail how a CPU cache hierarchy works, covering L1, L2, L3, "
        "the role of associativity, replacement policies such as LRU and "
        "pseudo-LRU, write-back versus write-through behaviour, and how "
        "prefetching interacts with all of these mechanisms in modern processors.");

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof si);
    si.cb = sizeof si;
    if (!CreateProcessA(NULL, cmd, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        printf("CreateProcess failed: %lu\n", GetLastError());
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return 0;
}
