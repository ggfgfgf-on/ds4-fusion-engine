# ds4 (DwarfStar) Windows 原生移植说明

状态：**CPU 后端全部构建成功并冒烟通过**；CUDA 后端待工具链组件下载完成后构建。

## 构建产物（D:\DSHwork\src\ds4\）

| 二进制 | 说明 |
|--------|------|
| `ds4-cpu.exe` | 交互式 CLI（对话） |
| `ds4-server-cpu.exe` | OpenAI/Responses/Anthropic 兼容 HTTP 服务 |
| `ds4-bench-cpu.exe` | 上下文前沿吞吐基准 |
| `ds4-eval-cpu.exe` | 能力评测（自检已通过：`--self-test-extractors` exit 0） |
| `ds4-agent-cpu.exe` | 原生编码 agent |

## 工具链

- Visual Studio 2022 Build Tools 17.14（MSVC 头文件/库 + Windows SDK 26100）
- LLVM 22.1.8 clang：`clang --target=x86_64-pc-windows-msvc`（GNU 风格驱动，
  支持 C99 VLA/复合字面量/__attribute__，产出 MSVC ABI 的 COFF 对象）
- lld-link 链接；ws2_32/iphlpapi 用 SDK `um\x64` 完整路径
- CUDA 13.3（nvcc，sm_120a，MXFP4）：组件 nvcc/cudart/cublas 需从
  `https://developer.download.nvidia.com/compute/cuda/13.3.0/network_installers/windows/x86_64/{nvcc,cudart,cublas}.exe`
  经代理下载后 `-s` 静默安装（NVIDIA 网络安装器直连超时，不可用）

## 构建命令

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DSHwork\src\ds4\build_win.ps1          # CPU
powershell -NoProfile -ExecutionPolicy Bypass -File D:\DSHwork\src\ds4\build_win.ps1 -Cuda    # CUDA
```

脚本自动定位 vswhere → vcvars64 → clang → nvcc。

## win_compat 兼容层（win_compat\ 目录）

强制包含（`-include win_compat/win_compat.h`）+ `-Iwin_compat` 提供 shim 头。
实现集中在 `win_compat.c` + `win_compat_regex.cpp`（std::regex 提供 POSIX regex）。

覆盖范围：
- **文件系统**：stat/fstat/lstat→_stat64（64 位 st_size）、off_t/ssize_t 64 位、
  open/read/write/close（混合 fd/socket 探测）、pread、ftruncate/fsync/fdopen、
  mkstemp（_mktemp_s+O_EXCL）、mkdir/unlink/rename/access、dirent（FindFirstFile）
- **mmap**：CreateFileMappingA+MapViewOfFile（81GB 大文件映射），munmap 注册表、
  mlock→VirtualLock、madvise no-op
- **线程**：pthread 全套（mutex/cond/timedwait/once/create/join/detach），
  pthread_t 为指针型保持 `if (t)` 语义
- **网络**：winsock2 封装（socket/setsockopt/getsockopt 签名适配、混合 close/read/write）、
  poll→混合实现（socket 走 WSAPoll、stdin 走 GetNumberOfConsoleInputEvents、
  管道走 PeekNamedPipe，支持 agent UI 的 stdin+wake 复用）、MSG_NOSIGNAL=0、
  SHUT_*、netdb/net/if/netinet/arpa 头 shim（iphlpapi 的 if_nametoindex）
- **进程**（agent bash 工具）：pipe→_pipe、waitpid→GetExitCodeProcess、
  kill→TerminateProcess、setpgid no-op、dup2→_dup2、WIFEXITED 宏
- **终端**：termios→控制台模式映射（tcgetattr/tcsetattr，ENABLE_VIRTUAL_TERMINAL_INPUT
  使方向键以 ESC 序列经 read() 呈现，linenoise 零改动）、ioctl(TIOCGWINSZ)→
  GetConsoleScreenBufferInfo
- **杂项**：clock_gettime/gettimeofday（QPC/FILETIME）、usleep/nanosleep/sleep、
  fcntl no-op、dprintf、flock（LockFileEx）、sigaction（signal 封装）、
  strdup/strcasecmp/strcasestr/strsep/strtok_r、localtime_r、fseeko/ftello 64 位、
  fnmatch（* ? 通配）、getopt 未使用（CLI 自解析）
- **UTF-8**：ds4_win_init() 设 CP_UTF8 + stdout/stderr 二进制模式（各 main 已调用）

## 上游源码直接修改（保持 POSIX 路径不变，#ifdef _WIN32 分支）

| 文件 | 修改 |
|------|------|
| ds4.c | 锁文件/暂存路径走 %TEMP%；fmemopen→tmpfile 回写/读出 |
| ds4_bench.c | Windows 下 cudaProfilerStart/Stop 为普通 extern（非 weak） |
| ds4_server.c | main 加 ds4_win_init()；SIGPIPE 守卫 |
| ds4_distributed.c | SIGPIPE 守卫 |
| ds4_web.c | Chrome 探测加 Windows 路径（Edge 优先）；fork/exec→CreateProcess（CDP 浏览器） |
| ds4_agent.c | bash 任务子系统 Windows 分支（CreateProcess+管道+PeekNamedPipe）；输出暂存走 %TEMP% |

## 模型

`ds4f-q2`：`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`（~81GB）
- 来源：https://hf-mirror.com/antirez/deepseek-v4-gguf（直连 HF 超时，aria2 -x16 ~5.7MB/s）
- 位置：`D:\DSHwork\models\`

## 运行（模型下载完成后）

```powershell
# CPU 冒烟（预期慢）
.\ds4-server-cpu.exe -m D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf --cpu --ctx 4096

# CUDA（8GB 显存：专家缓存预算约 4-5GB）
.\ds4-server.exe --cuda -m <模型> --ssd-streaming --ssd-streaming-cache-experts 4GB --ctx 32768
```

## 已知限制

- 无 fork/exec：agent 的 shell 工具用 cmd.exe 子进程（输出经管道+临时文件）
- kill(-pid) 仅终止直接子进程（无进程组；孙进程可能残留）
- fcntl 非阻塞为 no-op；所有非阻塞读路径改用 PeekNamedPipe/混合 poll
- fchmod no-op、S_ISLNK 恒 0（Windows 无 POSIX 模式位）
- 分布式/TP/RDMA 路径编译通过但未实测（本机单卡）

## 实测记录（2026-09-05，RTX 5070 Laptop 8GB / sm_120）

- 推理验证通过：`ds4-server.exe --cuda --ssd-streaming --ssd-streaming-cache-experts 4GB --ctx 8192`
- 显存占用：**7706 / 8151 MiB**（94%，模型 2.81GiB 常驻 + 专家缓存 + KV 0.46GiB）
- 生成吞吐：**0.73 tok/s**（IQ2XXS 284B MoE，SSD 流式换入专家权重，瓶颈在磁盘带宽）
- 内存规划：KV 0.46 + buffers 0.06 + 常驻模型 2.81 + 专家缓存 0.62 + prefill 预留 3.38 = 7.33 GiB

### 排障记录（五个关键修复）

1. **CRT open 默认文本模式**：`_open` 默认 text mode，遇 GGUF 数据中的 0x1A（Ctrl-Z）
   字节即截断/EOF（文件头 0x1A 位于偏移 395/1818/1902，张量数据中更密集），
   导致 CUDA 直读全部失败。修复：win_compat 新增 `ds4_open()` 垫片强制 `_O_BINARY`。
2. **HTTP `Expect: 100-continue` 死锁**：.NET HttpWebRequest 默认发送该头并等待 100 响应，
   服务器未处理导致请求体永远不发送（"bad HTTP request"）。修复：read_http_request
   检测到该头时立即回 `HTTP/1.1 100 Continue`；客户端也可 `ServicePointManager.Expect100Continue=$false`。
3. **MTP 推测解码 × SSD 流式**（融合引擎里程碑 1）：上游限制 `--ssd-streaming 与 --mtp-model
   不兼容` 已解除（ds4.c 补丁）——MTP 支持模型（3.55 GiB）走 mmap + 权重范围缓存
   宿主映射回退，主模型保留 fd 流式槽位（`ds4_gpu_set_model_fd_for_map` 顺序保证）。
4. **层主序预填充崩溃**：>18 token 提示词（解码式流式预填充上限 wide?64:18）落入
   layer-major 路径 → `CUDA streaming expert id -1` 崩溃。修复：
   `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=1024`（固化进 run_ds4_server.cmd）。
5. **Winsock 关闭 RST**：客户端带 `Connection: close`（Python urllib 默认）时，
   服务器 closesocket 丢弃未发送队列发 RST，响应丢失。修复：ds4_close 对 socket
   先 SO_LINGER{1,5} + shutdown(SD_SEND) 再 closesocket。

### 融合引擎（fusion_engine）

`D:\DSHwork\fusion_engine\`：plan.md 设想的融合推理引擎的真实落地（详细映射与
14 项目核实结论见 `FUSION_REPORT.md`）。推荐配置：
**5GB 专家缓存 + MTP draft=2 + DS4_METAL_STREAMING_DECODE_PREFILL_MAX=1024**，
实测热态 1.44 tok/s、长生成稳态 1.03 tok/s（基线 0.73）。

### 下载完整性（无 aria2 控制文件时）

- 权威校验：HF API `lfs.oid` 可能陈旧（Xet 迁移后内容更新）；以 `/resolve/` 实际服务内容为准，
  用随机区间采样对比（31/31 全中）+ 零区扫描（NTFS 预分配未写区为全零）补齐。
- 工具（`D:\DSHwork\models\`）：`scan_zero.exe`（零区扫描）、`fill_ranges.exe`（并行 Range 补拉，断点续传）、
  `fill_loop.ps1`（签名 URL 自动刷新 + hf-mirror/CDN 通道自动切换）。
