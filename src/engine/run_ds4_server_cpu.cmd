@echo off
rem run_ds4_server_cpu.cmd - Unified CPU server: fastest decode on this laptop
rem (2.0-2.3 tok/s with MTP vs 1.44 on the GPU path) + OpenAI-compatible API
rem + KV disk checkpoints.  Recommended default for interactive/agent use.
rem NOTE: the CUDA runtime DLLs must be on PATH even in --cpu mode (the MTP
rem support-model path touches CUDA-linked symbols).
setlocal
set MODEL=D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
set MTP_MODEL=D:\DSHwork\models\DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
set CUDA_BIN=D:\DSHwork\mf\Lib\site-packages\nvidia\cu13
set PATH=%CUDA_BIN%\bin;%CUDA_BIN%\bin\x86_64;%PATH%

cd /d D:\DSHwork\src\ds4
if not exist "%MODEL%" (
    echo model not found: %MODEL%
    exit /b 1
)
if not exist "%MTP_MODEL%" (
    echo MTP support model not found: %MTP_MODEL%
    exit /b 1
)
ds4-server.exe --cpu -m "%MODEL%" ^
  --ctx 8192 ^
  --mtp-model "%MTP_MODEL%" --mtp-draft 2 ^
  --kv-disk-dir .cache\server-kv-cpu --kv-disk-space-mb 4096 --kv-cache-min-tokens 8 ^
  --host 127.0.0.1 --port 8000
