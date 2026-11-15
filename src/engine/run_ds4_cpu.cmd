@echo off
rem run_ds4_cpu.cmd - CPU decode mode: the fastest generation path on this
rem laptop (2.29 t/s with MTP vs 1.44 t/s on the 8GB GPU path).
rem Recommended for interactive decode; use run_ds4_server.cmd for batched
rem prefill-heavy workloads (GPU prefill: 3.88 t/s).
setlocal
set MODEL=D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
set MTP_MODEL=D:\DSHwork\models\DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf

cd /d D:\DSHwork\src\ds4
if not exist "%MODEL%" (
    echo model not found: %MODEL%
    exit /b 1
)
ds4-cpu.exe -m "%MODEL%" --cpu ^
  --mtp-model "%MTP_MODEL%" --mtp-draft 2 ^
  %*
