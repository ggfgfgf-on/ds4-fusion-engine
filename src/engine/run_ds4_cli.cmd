@echo off
rem run_ds4_cli.cmd - one-shot or interactive CLI with CUDA SSD streaming.
setlocal
set MODEL=D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
set CUDA_BIN=D:\DSHwork\mf\Lib\site-packages\nvidia\cu13
set PATH=%CUDA_BIN%\bin;%CUDA_BIN%\bin\x86_64;%PATH%

cd /d D:\DSHwork\src\ds4
if "%1"=="" (
    ds4.exe --cuda -m "%MODEL%" --ssd-streaming --ssd-streaming-cache-experts 4GB --ctx 16384 --nothink
) else (
    ds4.exe --cuda -m "%MODEL%" --ssd-streaming --ssd-streaming-cache-experts 4GB --ctx 16384 --nothink -p "%*"
)
