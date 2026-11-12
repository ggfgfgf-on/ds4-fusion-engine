@echo off
rem run_ds4_server_hybrid.cmd - Unified hybrid server:
rem   GPU (SSD-streamed, chunk 16) prefills the prompt,
rem   CPU decodes (warm 4.1-5.6 t/s; cold first turn ~2.2 t/s).
rem   MTP draft 2 measured -36% on CPU decode (2.93 vs 4.59 t/s, 09-21):
rem   opt in with --mtp-model <path> --mtp-draft 2 only if desired.
rem The handoff uses the KV disk checkpoint (backend-neutral payload).
rem NOTE: CUDA runtime DLLs must be on PATH (the GPU prefill engine needs them).
setlocal
set MODEL=D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
set CUDA_BIN=D:\DSHwork\mf\Lib\site-packages\nvidia\cu13
set PATH=%CUDA_BIN%\bin;%CUDA_BIN%\bin\x86_64;%PATH%
set DS4_THREADS=28
set DS4_POOL_SPIN_US=500
set DS4_POOL_PIN=1
set DS4_HYBRID_PREFILL_CHUNK=0
rem Weight-arena headroom cap stays at its 8 GiB default: the prefill-safe
rem LRU scheduler (DS4_CUDA_WEIGHT_ARENA_PREFILL_LRU, default on) recycles
rem completed-layer weight ranges at the cap instead of freezing, so the
rem long-prompt streaming prefill keeps ~8-9 t/s under shared-pool pressure
rem while the DWM reserve stays intact.

cd /d D:\DSHwork\src\ds4
if not exist "%MODEL%" (
    echo model not found: %MODEL%
    exit /b 1
)
ds4-server.exe --hybrid --cuda -m "%MODEL%" ^
  --ssd-streaming-cache-experts 16 ^
  --ctx 8192 ^
  --host 127.0.0.1 --port 8000
