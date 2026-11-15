@echo off
rem run_ds4_server.cmd - Windows native DeepSeek V4 Flash (ds4/DwarfStar) server
rem Tuned for RTX 5070 Laptop 8GB (sm_120):
rem   5GB expert cache (1.62 GiB dynamic, ~95% hit rate) + MTP speculative
rem   decoding (depth 2) + fixed layer-major batch prefill (small chunks;
rem   >18-token prompts prefill ~4x faster than the per-token path).
setlocal
set MODEL=D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
set MTP_MODEL=D:\DSHwork\models\DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
set CUDA_BIN=D:\DSHwork\mf\Lib\site-packages\nvidia\cu13
set PATH=%CUDA_BIN%\bin;%CUDA_BIN%\bin\x86_64;%PATH%
rem Weight-arena headroom cap stays at its 8 GiB default: the prefill-safe
rem LRU scheduler (DS4_CUDA_WEIGHT_ARENA_PREFILL_LRU, default on) recycles
rem completed-layer ranges at the cap instead of freezing the arena.

cd /d D:\DSHwork\src\ds4
if not exist "%MODEL%" (
    echo model not found: %MODEL%
    exit /b 1
)
if not exist "%MTP_MODEL%" (
    echo MTP support model not found: %MTP_MODEL%
    exit /b 1
)
ds4-server.exe --cuda -m "%MODEL%" ^
  --ssd-streaming --ssd-streaming-cache-experts 5GB ^
  --prefill-chunk 8 ^
  --ctx 8192 ^
  --mtp-model "%MTP_MODEL%" --mtp-draft 2 --mtp-exact-sampling ^
  --kv-disk-dir .cache\server-kv --kv-disk-space-mb 4096 ^
  --host 127.0.0.1 --port 8000
