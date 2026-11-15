# rel_gate_0926.ps1 - release gate for ds4-release-0926.
# Two byte gates on the release-tree ds4-server.exe:
#   mode '16'   - --ssd-streaming-cache-experts 16 (matches the 9/21 release gate)
#   mode 'auto' - --ssd-streaming-cache-experts auto (WDDM band scheduler)
# Each: 48-token temp-0 output must match cert_out_whole.txt byte for byte.
param([string]$Mode = 'auto')
$ErrorActionPreference = 'Continue'
$root = 'D:\DSHwork\ds4-release-0926'
$model = 'D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$shExe = Join-Path $PSHOME 'powershell.exe'
Set-Location $root
Get-Process ds4-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
$env:DS4_THREADS = '28'; $env:DS4_POOL_SPIN_US = '500'; $env:DS4_POOL_PIN = '1'
$env:PATH = "D:\DSHwork\mf\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"
$env:DS4_HYBRID_PREFILL_CHUNK = '0'; $env:DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL = '1'
Remove-Item Env:\DS4_CUDA_KV_BITQ, Env:\DS4_CUDA_KV_BITQ_AUTO, Env:\DS4_CUDA_WEIGHT_ARENA_LRU -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $root '.cache') | Out-Null
$wdState = Join-Path $root '.cache\wd_rel_state.txt'
$srvLog = Join-Path $root ".rel_gate_${Mode}_srv.txt"
$cacheArg = if ($Mode -eq '16') { '16' } else { 'auto' }
$p = Start-Process -FilePath (Join-Path $root 'bin\cuda\ds4-server.exe') `
    -ArgumentList @('--cuda','--ssd-streaming','-m',$model,'--ssd-streaming-cache-experts',$cacheArg,'--ctx','8192','--host','127.0.0.1','--port','8000') `
    -WorkingDirectory $root -RedirectStandardError $srvLog -RedirectStandardOutput (Join-Path $root ".rel_gate_${Mode}_out.txt") `
    -PassThru -WindowStyle Hidden
"server pid: $($p.Id)"
$wd = Start-Process $shExe -ArgumentList @('-NoProfile','-File',(Join-Path $root 'scripts\watchdog.ps1'),'-TargetPid',$p.Id,`
    '-SharedGB','38','-TotalGB','50','-MaxMinutes','30','-PollSec','2','-State',$wdState) `
    -PassThru -WindowStyle Hidden
$gate = $false
try {
    $ready = $false
    for ($i = 0; $i -lt 150; $i++) {
        Start-Sleep -Seconds 2
        try { $null = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; $ready = $true; break } catch {
            if (Select-String -Path $srvLog -Pattern 'listening on' -Quiet) { $ready = $true; break }
        }
    }
    if (-not $ready) { 'SERVER_NOT_READY' } else {
        $prompt = "Write a comprehensive technical analysis of solar energy storage systems, comparing lithium-ion battery installations, pumped hydro storage, compressed air energy storage, and molten salt thermal storage in terms of round-trip efficiency, capital cost per kilowatt-hour, expected cycle life, land use requirements, and grid-scale deployment examples from California, Texas, Australia, and China, then discuss how each technology performs during peak evening demand periods, seasonal variations in solar output, and extreme weather events, and finally recommend an optimal storage portfolio for a utility aiming to reach ninety percent renewable penetration by 2040."
        $body = @{ model = 'deepseek-chat'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 48; temperature = 0; stream = $false } | ConvertTo-Json -Depth 5
        $t0 = Get-Date
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 1500 -UseBasicParsing
        $j = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json
        [IO.File]::WriteAllText((Join-Path $root ".cache\rel_gate_${Mode}.txt"), $j.choices[0].message.content, (New-Object System.Text.UTF8Encoding($false)))
        $a = [IO.File]::ReadAllBytes((Join-Path $root ".cache\rel_gate_${Mode}.txt"))
        $b = [IO.File]::ReadAllBytes('D:\DSHwork\src\ds4\.cache\cert_out_whole.txt')
        $same = ($a.Length -eq $b.Length)
        if ($same) { for ($i = 0; $i -lt $a.Length; $i++) { if ($a[$i] -ne $b[$i]) { $same = $false; break } } }
        "release gate[$Mode] vs cert_out_whole.txt ($($b.Length)B): $same (wall $([math]::Round(((Get-Date)-$t0).TotalSeconds,1))s)"
        $gate = $same
    }
} catch { "REQUEST_FAILED: $($_.Exception.Message)" } finally {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 4
    if ($wd -and -not $wd.HasExited) { Stop-Process -Id $wd.Id -Force -ErrorAction SilentlyContinue }
}
'--- watchdog ---'
Get-Content $wdState -ErrorAction SilentlyContinue
if ($Mode -eq 'auto') {
    '--- WDDM scheduler line ---'
    Select-String -Path $srvLog -Pattern 'WDDM band scheduler|cache target|memory: KV' | ForEach-Object { $_.Line }
}
"=== rel_gate_0926[$Mode] done (byte-match: $gate) ==="
