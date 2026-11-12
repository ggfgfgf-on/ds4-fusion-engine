# build_win.ps1 - Build DwarfStar (ds4) natively on Windows.
# Uses clang (LLVM) with the MSVC ABI target for all C code and nvcc
# (with cl.exe host) for the CUDA TUs. Output objects are COFF/MSVC ABI.
#
#   .\build_win.ps1          -> CPU smoke build: ds4-server-cpu.exe,
#                               ds4-bench-cpu.exe (validates win_compat)
#   .\build_win.ps1 -Cuda    -> CUDA build: ds4.exe, ds4-server.exe, ...
param(
    [switch]$Cuda,
    [switch]$Clean
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# --- locate Visual Studio Build Tools and load its environment ---
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found - install VS 2022 Build Tools (C++ workload)" }
$vs = & $vswhere -latest -products * -property installationPath
if (-not $vs) { throw "Visual Studio Build Tools not found" }
"VS: $vs"
$envOut = cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set"
foreach ($line in $envOut) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

$clang = Get-Command clang.exe -ErrorAction SilentlyContinue
if (-not $clang) {
    if (Test-Path 'C:\Program Files\LLVM\bin\clang.exe') {
        $clang = 'C:\Program Files\LLVM\bin\clang.exe'
    } else {
        throw "clang not found - install LLVM (winget install LLVM.LLVM)"
    }
}
"clang: $clang"

$CudaToolkit = $env:CUDA_PATH
if (-not $CudaToolkit -and (Test-Path 'D:\DSHwork\mf\Lib\site-packages\nvidia\cu13\bin\nvcc.exe')) {
    $CudaToolkit = 'D:\DSHwork\mf\Lib\site-packages\nvidia\cu13'
}
if (-not $CudaToolkit -and (Test-Path 'D:\DSHwork\mf\Library\bin\nvcc.exe')) {
    $CudaToolkit = 'D:\DSHwork\mf\Library'
}
if (-not $CudaToolkit) { $CudaToolkit = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
if ($Cuda -and -not (Test-Path "$CudaToolkit\bin\nvcc.exe")) {
    throw "nvcc not found at $CudaToolkit - install the CUDA 13.3 toolkit (pip nvidia-cuda-nvcc or NVIDIA installer)"
}
# Make the toolkit's DLLs (cicc, nvvm, cudart, cublas, ...) visible at runtime.
$env:PATH = "$CudaToolkit\bin;$CudaToolkit\bin\x86_64;$env:PATH"
"CudaToolkit: $CudaToolkit"

$ObjDir = Join-Path $Root 'build_win'
New-Item -ItemType Directory -Force -Path $ObjDir | Out-Null
if ($Clean) { Remove-Item "$ObjDir\*" -Force -ErrorAction SilentlyContinue }

# --- compiler invocations -------------------------------------------------
$CFlagsCommon = @(
    '--target=x86_64-pc-windows-msvc',
    '-std=c99', '-O2', '-ffast-math', '-fno-finite-math-only', '-g',
    '-march=native',
    '-Wall', '-Wextra', '-Wno-unused-function', '-Wno-unused-parameter',
    '-Wno-unused-variable', '-Wno-missing-braces', '-Wno-deprecated-declarations',
    '-Wno-unknown-attributes', '-Wno-ignored-attributes',
    '-D_GNU_SOURCE',
    '-include', 'win_compat/win_compat.h',
    '-I.', '-Iwin_compat', '-Ithird_party/zstd'
)

function Compile-C($src, $outName, [string[]]$extraFlags = @()) {
    $out = Join-Path $ObjDir $outName
    & $clang @CFlagsCommon @extraFlags -c $src -o $out 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "clang failed on $src" }
    "  [ok] $outName"
}

function Compile-Regex {
    $out = Join-Path $ObjDir 'win_compat_regex.obj'
    & $clang '--target=x86_64-pc-windows-msvc' '-std=c++17' '-O2' '-g' '-I.' '-Iwin_compat' '-c' 'win_compat\win_compat_regex.cpp' '-o' $out 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "clang++ failed on win_compat_regex.cpp" }
    "  [ok] win_compat_regex.obj"
}

# Windows SDK library dir for the UM libraries (ws2_32, iphlpapi, ...).
$UmLib = ($env:LIB -split ';' | Where-Object { $_ -match 'um\\x64$' } | Select-Object -First 1)
if (-not $UmLib) { throw "could not locate the Windows SDK um\x64 lib dir in LIB" }
"SysLibs: $UmLib"

function Link-Exe($exeName, $appObjs, $coreObjs) {
    $objs = @()
    foreach ($o in ($appObjs + $coreObjs)) { $objs += Join-Path $ObjDir $o }
    $out = Join-Path $Root $exeName
    $sdkLibs = @(
        (Join-Path $UmLib 'ws2_32.lib'),
        (Join-Path $UmLib 'iphlpapi.lib'),
        (Join-Path $UmLib 'advapi32.lib')
    )
    & $clang @CFlagsCommon $objs -o $out -fuse-ld=lld $sdkLibs 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "link failed for $exeName" }
    "  [link ok] $exeName"
}

$CoreObjects = @(
    'ds4_cpu.obj', 'ds4_image.obj', 'ds4_distributed.obj',
    'ds4_tp.obj', 'ds4_ssd.obj', 'ds4_layer_pack.obj', 'zstd.obj'
)

function Build-Cpu {
    Write-Host "== Stage 1: CPU build =="
    Compile-C 'win_compat\win_compat.c' 'win_compat.obj'
    Compile-Regex
    Compile-C 'ds4.c' 'ds4_cpu.obj' @('-DDS4_NO_GPU')
    Compile-C 'ds4_image.c' 'ds4_image.obj'
    Compile-C 'ds4_distributed.c' 'ds4_distributed.obj'
    Compile-C 'ds4_tp.c' 'ds4_tp.obj'
    Compile-C 'ds4_ssd.c' 'ds4_ssd.obj'
    Compile-C 'ds4_layer_pack.c' 'ds4_layer_pack.obj'
    Compile-C 'ds4_kvstore.c' 'ds4_kvstore.obj'
    Compile-C 'third_party\zstd\src\zstd.c' 'zstd.obj'
    Compile-C 'rax.c' 'rax.obj'
    Compile-C 'ds4_help.c' 'ds4_help.obj'
    Compile-C 'ds4_prompt_prefix.c' 'ds4_prompt_prefix.obj'
    Compile-C 'ds4_web.c' 'ds4_web.obj'
    Compile-C 'ds4_gpu_args.c' 'ds4_gpu_args_cpu.obj' @('-DDS4_NO_GPU')

    $apps = @(
        @{src='ds4_server.c'; flags=@('-DDS4_NO_GPU')},
        @{src='ds4_bench.c'; flags=@('-DDS4_NO_GPU')},
        @{src='ds4_cli.c'; flags=@('-DDS4_NO_GPU')},
        @{src='ds4_eval.c'; flags=@('-DDS4_NO_GPU')},
        @{src='ds4_agent.c'; flags=@('-DDS4_NO_GPU')}
    )
    foreach ($a in $apps) {
        $base = [IO.Path]::GetFileNameWithoutExtension($a.src)
        Compile-C $a.src "$($base)_cpu.obj" $a.flags
    }
    Compile-C 'linenoise.c' 'linenoise.obj'

    $cores = $CoreObjects + @('win_compat.obj', 'win_compat_regex.obj')
    # ds4-server-cpu
    Link-Exe 'ds4-server-cpu.exe' @('ds4_server_cpu.obj','ds4_help.obj','ds4_kvstore.obj','rax.obj','ds4_gpu_args_cpu.obj') $cores
    # ds4-bench-cpu
    Link-Exe 'ds4-bench-cpu.exe' @('ds4_bench_cpu.obj','ds4_help.obj','ds4_gpu_args_cpu.obj') $cores
    # ds4-cpu (CLI)
    Link-Exe 'ds4-cpu.exe' @('ds4_cli_cpu.obj','ds4_help.obj','ds4_prompt_prefix.obj','linenoise.obj','ds4_gpu_args_cpu.obj') $cores
    # ds4-eval-cpu
    Link-Exe 'ds4-eval-cpu.exe' @('ds4_eval_cpu.obj','ds4_help.obj') $cores
    # ds4-agent-cpu
    Link-Exe 'ds4-agent-cpu.exe' @('ds4_agent_cpu.obj','ds4_help.obj','ds4_prompt_prefix.obj','ds4_web.obj','ds4_kvstore.obj','linenoise.obj','ds4_gpu_args_cpu.obj') $cores
    Write-Host "== CPU build OK =="
}

if ($Clean -and -not $Cuda) {
    Write-Host "clean done"
    exit 0
}

if ($Cuda) {
    Write-Host "== Stage 2: CUDA build (sm_120a) =="
    Compile-C 'win_compat\win_compat.c' 'win_compat.obj'
    Compile-Regex
    Compile-C 'ds4.c' 'ds4.obj' @("-I$CudaToolkit\include")
    Compile-C 'ds4_image.c' 'ds4_image.obj'
    Compile-C 'ds4_distributed.c' 'ds4_distributed.obj'
    Compile-C 'ds4_tp.c' 'ds4_tp.obj'
    Compile-C 'ds4_ssd.c' 'ds4_ssd.obj'
    Compile-C 'ds4_layer_pack.c' 'ds4_layer_pack.obj'
    Compile-C 'ds4_kvstore.c' 'ds4_kvstore.obj'
    Compile-C 'third_party\zstd\src\zstd.c' 'zstd.obj'
    Compile-C 'rax.c' 'rax.obj'
    Compile-C 'ds4_help.c' 'ds4_help.obj'
    Compile-C 'ds4_prompt_prefix.c' 'ds4_prompt_prefix.obj'
    Compile-C 'ds4_web.c' 'ds4_web.obj'
    Compile-C 'ds4_gpu_args.c' 'ds4_gpu_args.obj'
    Compile-C 'ds4_server.c' 'ds4_server.obj'
    Compile-C 'ds4_bench.c' 'ds4_bench.obj'
    Compile-C 'ds4_cli.c' 'ds4_cli.obj'
    Compile-C 'ds4_eval.c' 'ds4_eval.obj'
    Compile-C 'ds4_agent.c' 'ds4_agent.obj'
    Compile-C 'ds4_router_train.c' 'ds4_router_train.obj'
    Compile-C 'linenoise.c' 'linenoise.obj'

    $nvcc = "$CudaToolkit\bin\nvcc.exe"
    $NvFlags = @(
        '-O3', '-g', '-lineinfo', '--use_fast_math',
        '-gencode', 'arch=compute_120a,code=sm_120a', '-DDS4_CUDA_HAVE_MXF4=1',
        '--std', 'c++17',
        '-Xcompiler', '/O2', '-Xcompiler', '/EHsc',
        '-Xcompiler', '/Zc:preprocessor',
        '-I.', '-Iwin_compat', '-Icuda/mmq', "-I$CudaToolkit\include"
    )
    function Compile-Cu($src, $outName) {
        $out = Join-Path $ObjDir $outName
        & $nvcc @NvFlags -c $src -o $out 2>&1 | ForEach-Object { "$_" }
        if ($LASTEXITCODE -ne 0) { throw "nvcc failed on $src" }
        "  [ok] $outName"
    }
    Compile-Cu 'ds4_cuda.cu' 'ds4_cuda.obj'
    $mmqFiles = @('ds4_ggml_stubs.cu','ds4_mmq.cu','ds4_mmq_d2r.cu','quantize.cu','mmid.cu','mmvq.cu','ds4_repack.cu')
    $mmqObjs = @()
    foreach ($f in $mmqFiles) {
        $name = [IO.Path]::GetFileNameWithoutExtension($f) + '.obj'
        Compile-Cu "cuda\mmq\$f" $name
        $mmqObjs += $name
    }

    $cores = @('ds4.obj','ds4_image.obj','ds4_distributed.obj','ds4_tp.obj','ds4_ssd.obj','ds4_cuda.obj','ds4_layer_pack.obj','zstd.obj') + $mmqObjs + @('win_compat.obj', 'win_compat_regex.obj')

    $cudart = "$CudaToolkit\lib\x64\cudart.lib"
    $cublas = "$CudaToolkit\lib\x64\cublas.lib"
    if (-not (Test-Path $cudart) -and (Test-Path "$CudaToolkit\lib\cudart.lib")) {
        # conda layout: libs directly under lib\
        $cudart = "$CudaToolkit\lib\cudart.lib"
        $cublas = "$CudaToolkit\lib\cublas.lib"
    }
    function Link-Cuda($exeName, $appObjs) {
        $objs = @()
        foreach ($o in ($appObjs + $cores)) { $objs += Join-Path $ObjDir $o }
        $out = Join-Path $Root $exeName
        $sdkLibs = @(
            (Join-Path $UmLib 'ws2_32.lib'),
            (Join-Path $UmLib 'iphlpapi.lib'),
            (Join-Path $UmLib 'advapi32.lib'),
            (Join-Path $UmLib 'dxgi.lib'),
            (Join-Path $UmLib 'pdh.lib')
        )
        & $nvcc @NvFlags $objs -o $out $cudart $cublas $sdkLibs 2>&1 | ForEach-Object { "$_" }
        if ($LASTEXITCODE -ne 0) { throw "nvcc link failed for $exeName" }
        "  [link ok] $exeName"
    }
    Link-Cuda 'ds4.exe' @('ds4_cli.obj','ds4_help.obj','ds4_prompt_prefix.obj','linenoise.obj','ds4_gpu_args.obj')
    Link-Cuda 'ds4-server.exe' @('ds4_server.obj','ds4_help.obj','ds4_kvstore.obj','rax.obj','ds4_gpu_args.obj')
    Link-Cuda 'ds4-bench.exe' @('ds4_bench.obj','ds4_help.obj','ds4_gpu_args.obj')
    Link-Cuda 'ds4-eval.exe' @('ds4_eval.obj','ds4_help.obj')
    Link-Cuda 'ds4-agent.exe' @('ds4_agent.obj','ds4_help.obj','ds4_prompt_prefix.obj','ds4_web.obj','ds4_kvstore.obj','linenoise.obj','ds4_gpu_args.obj')
    Link-Cuda 'ds4-router-train.exe' @('ds4_router_train.obj')
    Write-Host "== CUDA build OK =="
} else {
    Build-Cpu
}
