# build_quantize_win.ps1 - Windows build for gguf-tools/deepseek4-quantize.
# Standalone C plus the engine's win_compat pthread shim. Produces
# deepseek4-quantize.exe next to this script.
# Requires: VS2022 Build Tools, LLVM clang (C:\Program Files\LLVM\bin or PATH).
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Engine = Join-Path (Split-Path -Parent $Root) 'engine'
if (-not (Test-Path (Join-Path $Engine 'win_compat\win_compat.c'))) {
    throw "engine win_compat not found at $Engine"
}

# --- VS Build Tools environment (vcvars) ---
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found - install VS 2022 Build Tools" }
$vs = & $vswhere -latest -products * -property installationPath
if (-not $vs) { throw "Visual Studio Build Tools not found" }
$envOut = cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set"
foreach ($line in $envOut) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

$clang = Get-Command clang.exe -ErrorAction SilentlyContinue
if (-not $clang) {
    if (Test-Path 'C:\Program Files\LLVM\bin\clang.exe') { $clang = 'C:\Program Files\LLVM\bin\clang.exe' }
    else { throw 'clang not found - install LLVM (winget install LLVM.LLVM)' }
}
$UmLib = ($env:LIB -split ';' | Where-Object { $_ -match 'um\\x64$' } | Select-Object -First 1)
if (-not $UmLib) { throw "could not locate the Windows SDK um\x64 lib dir" }

$CCommon = @('--target=x86_64-pc-windows-msvc', '-std=c11', '-O2', '-g', '-D_GNU_SOURCE',
             '-Wall', '-Wextra', '-Wno-deprecated-declarations', '-Wno-unused-function')

# win_compat objects (same shape as build_win.ps1)
& $clang @CCommon '-c' (Join-Path $Engine 'win_compat\win_compat.c') '-o' (Join-Path $Root 'win_compat.obj')
if ($LASTEXITCODE -ne 0) { throw "clang failed on win_compat.c" }
& $clang '--target=x86_64-pc-windows-msvc' '-std=c++17' '-O2' '-g' '-I.' '-Iwin_compat' '-c' `
    (Join-Path $Engine 'win_compat\win_compat_regex.cpp') '-o' (Join-Path $Root 'win_compat_regex.obj')
if ($LASTEXITCODE -ne 0) { throw "clang++ failed on win_compat_regex.cpp" }

& $clang @CCommon `
    "-I$Engine" "-I$(Join-Path $Engine 'win_compat')" `
    '-include' (Join-Path $Engine 'win_compat\win_compat.h') `
    '-o' (Join-Path $Root 'deepseek4-quantize.exe') `
    (Join-Path $Root 'deepseek4-quantize.c') (Join-Path $Root 'quants.c') `
    (Join-Path $Root 'win_compat.obj') (Join-Path $Root 'win_compat_regex.obj') `
    (Join-Path $UmLib 'ws2_32.lib') (Join-Path $UmLib 'iphlpapi.lib') (Join-Path $UmLib 'advapi32.lib')
if ($LASTEXITCODE -ne 0) { throw "link failed" }
Remove-Item (Join-Path $Root 'win_compat.obj'), (Join-Path $Root 'win_compat_regex.obj') -ErrorAction SilentlyContinue
"built deepseek4-quantize.exe"
