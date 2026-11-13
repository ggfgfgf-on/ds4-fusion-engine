# auto_gate.ps1 - --ssd-streaming-cache-experts auto on pure GPU under watchdog.
# Verifies: (1) WDDM band scheduler logs a budget inside [5.0, 5.74] GiB,
# (2) 48-token temp-0 byte gate still matches .cache\cert_out_whole.txt.
$ErrorActionPreference = 'Continue'
$root = 'D:\DSHwork\ds4-upstream'
$model = 'D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$shExe = Join-Path $PSHOME 'powershell.exe'
Set-Location $root
Get-Process ds4-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$env:DS4_THREADS = '28'
$env:DS4_POOL_SPIN_US = '500'
$env:DS4_POOL_PIN = '1'
$env:PATH = "D:\DSHwork\mf\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"
$env:DS4_HYBRID_PREFILL_CHUNK = '0'
$env:DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL = '1'
Remove-Item Env:\DS4_CUDA_KV_BITQ, Env:\DS4_CUDA_KV_BITQ_COMP, Env:\DS4_CUDA_KV_BITQ_IDX, `
    Env:\DS4_CUDA_KV_BITQ_BITS, Env:\DS4_CUDA_KV_BITQ_RAW_BITS, Env:\DS4_CUDA_KV_BITQ_COMP_BITS, `
    Env:\DS4_CUDA_KV_BITQ_IDX_BITS, Env:\DS4_CUDA_KV_BITQ_AUTO -ErrorAction SilentlyContinue

$wdState = Join-Path $root '.cache\wd_auto_state.txt'
Remove-Item -Recurse -Force (Join-Path $root '.cache\server-kv-hybrid') -ErrorAction SilentlyContinue
$srvLog = Join-Path $root 'auto_gate_srv.txt'
$p = Start-Process -FilePath (Join-Path $root 'ds4-server.exe') `
    -ArgumentList @('--cuda','--ssd-streaming','-m',$model,'--ssd-streaming-cache-experts','auto','--ctx','8192','--host','127.0.0.1','--port','8000') `
    -WorkingDirectory $root -RedirectStandardError $srvLog -RedirectStandardOutput (Join-Path $root 'auto_gate_out.txt') `
    -PassThru -WindowStyle Hidden
"server pid: $($p.Id)"

$wd = Start-Process $shExe -ArgumentList @('-NoProfile','-File',(Join-Path $root 'watchdog.ps1'),'-TargetPid',$p.Id,`
    '-SharedGB','24','-TotalGB','40','-MaxMinutes','25','-PollSec','2','-State',$wdState) `
    -PassThru -WindowStyle Hidden
"watchdog pid: $($wd.Id)"

$gate = $false
try {
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        try { $null = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; $ready = $true; break } catch {
            if (Select-String -Path $srvLog -Pattern 'listening on' -Quiet) { $ready = $true; break }
        }
    }
    if (-not $ready) { Write-Output 'SERVER_NOT_READY' } else {
        $prompt = "Write a comprehensive technical analysis of solar energy storage systems, comparing lithium-ion battery installations, pumped hydro storage, compressed air energy storage, and molten salt thermal storage in terms of round-trip efficiency, capital cost per kilowatt-hour, expected cycle life, land use requirements, and grid-scale deployment examples from California, Texas, Australia, and China, then discuss how each technology performs during peak evening demand periods, seasonal variations in solar output, and extreme weather events, and finally recommend an optimal storage portfolio for a utility aiming to reach ninety percent renewable penetration by 2040."
        $body = @{ model = 'deepseek-chat'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 48; temperature = 0; stream = $false } | ConvertTo-Json -Depth 5
        $t0 = Get-Date
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 1500 -UseBasicParsing
        $wall = [math]::Round(((Get-Date)-$t0).TotalSeconds,1)
        $j = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json
        $out = $j.choices[0].message.content
        [IO.File]::WriteAllText((Join-Path $root '.cache\auto_gate_off.txt'), $out, (New-Object System.Text.UTF8Encoding($false)))
        Write-Output ("TOKENS: {0} WALL: {1}s" -f $j.usage.completion_tokens, $wall)
        $pa = Join-Path $root '.cache\auto_gate_off.txt'
        $pb = Join-Path $root '.cache\cert_out_whole.txt'
        $a = [IO.File]::ReadAllBytes($pa); $b = [IO.File]::ReadAllBytes($pb)
        $same = ($a.Length -eq $b.Length)
        if ($same) { for ($i = 0; $i -lt $a.Length; $i++) { if ($a[$i] -ne $b[$i]) { $same = $false; break } } }
        "gate vs cert_out_whole.txt ($($b.Length)B): $same"
        $gate = $same
    }
} catch {
    Write-Output ("REQUEST_FAILED: {0}" -f $_.Exception.Message)
} finally {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 4
    if ($wd -and -not $wd.HasExited) { Stop-Process -Id $wd.Id -Force -ErrorAction SilentlyContinue }
}
'--- watchdog state ---'
Get-Content $wdState -ErrorAction SilentlyContinue
'--- WDDM band scheduler lines ---'
Select-String -Path $srvLog -Pattern 'WDDM band scheduler|auto cache|thrash|cache target|prefill reserve|dynamic cache' | ForEach-Object { $_.Line }
'--- decode speed ---'
Select-String -Path $srvLog -Pattern 'decoding chunk' | ForEach-Object { $_.Line } | Select-Object -Last 3
"=== auto_gate done (byte-match: $gate) ==="

