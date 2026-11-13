# upstream_lru_stress6.ps1 - cluster 2b certification: long-prompt prefill
# pressure must exercise the arena LRU evictor without illegal accesses.
$ErrorActionPreference = 'Continue'
$root = 'D:\DSHwork\ds4-upstream'
$model = 'D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$shExe = Join-Path $PSHOME 'powershell.exe'
Set-Location $root
Get-Process ds4-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
$env:DS4_THREADS = '28'; $env:DS4_POOL_SPIN_US = '500'; $env:DS4_POOL_PIN = '1'
$env:PATH = "D:\DSHwork\mf\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"
$env:DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL = '1'
$env:DS4_CUDA_WEIGHT_CACHE_VERBOSE = '1'
$env:DS4_CUDA_WEIGHT_CACHE_LIMIT_GB = '6'
foreach ($v in 'http_proxy','https_proxy','no_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','NO_PROXY','ALL_PROXY') { Remove-Item Env:\$v -ErrorAction SilentlyContinue }
$srvLog = Join-Path $root '.upstream_lru_stress6_srv.txt'
$p = Start-Process -FilePath (Join-Path $root 'ds4-server.exe') `
    -ArgumentList @('--cuda','--ssd-streaming','-m',$model,'--ssd-streaming-cache-experts','auto','--ctx','8192','--host','127.0.0.1','--port','8000') `
    -WorkingDirectory $root -RedirectStandardError $srvLog -RedirectStandardOutput (Join-Path $root '.upstream_lru_stress6_out.txt') `
    -PassThru -WindowStyle Hidden
$wd = Start-Process $shExe -ArgumentList @('-NoProfile','-File','D:\DSHwork\src\ds4\watchdog.ps1','-TargetPid',$p.Id,`
    '-SharedGB','24','-TotalGB','40','-MaxMinutes','25','-PollSec','2','-State',(Join-Path $root '.wd_stress.txt')) `
    -PassThru -WindowStyle Hidden -ErrorAction SilentlyContinue
try {
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        try { $null = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; $ready = $true; break } catch {
            if (Select-String -Path $srvLog -Pattern 'listening on' -Quiet) { $ready = $true; break }
        }
    }
    if (-not $ready) { 'SERVER_NOT_READY' } else {
        $base = "Write a comprehensive technical analysis of solar energy storage systems, comparing lithium-ion battery installations, pumped hydro storage, compressed air energy storage, and molten salt thermal storage in terms of round-trip efficiency, capital cost per kilowatt-hour, expected cycle life, land use requirements, and grid-scale deployment examples from California, Texas, Australia, and China, then discuss how each technology performs during peak evening demand periods, seasonal variations in solar output, and extreme weather events, and finally recommend an optimal storage portfolio for a utility aiming to reach ninety percent renewable penetration by 2040."
        $prompt = (1..25 | ForEach-Object { "Section $($_): $base" }) -join "`n"
        $body = @{ model = 'deepseek-chat'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 48; temperature = 0; stream = $false } | ConvertTo-Json -Depth 5
        $t0 = Get-Date
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 1800 -UseBasicParsing
        $j = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json
        "STRESS tokens=$($j.usage.prompt_tokens) prompt, $($j.usage.completion_tokens) gen, wall=$([math]::Round(((Get-Date)-$t0).TotalSeconds,1))s"
        $j.choices[0].message.content.Substring(0, [math]::Min(120, $j.choices[0].message.content.Length))
    }
} catch { "STRESS REQUEST_FAILED: $($_.Exception.Message)" } finally {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 4
    if ($wd -and -not $wd.HasExited) { Stop-Process -Id $wd.Id -Force -ErrorAction SilentlyContinue }
}
'--- evictor activity ---'
$ev = Select-String -Path $srvLog -Pattern 'arena LRU evicted' | Measure-Object
"LRU evictions: $($ev.Count)"
Select-String -Path $srvLog -Pattern 'arena LRU evicted' | Select-Object -First 3 | ForEach-Object { $_.Line }
'--- crash markers ---'
$crash = Select-String -Path $srvLog -Pattern 'illegal memory access|an illegal|CUDA synchronize failed' | Measure-Object
"crash markers: $($crash.Count)"
'--- prefill ---'
Select-String -Path $srvLog -Pattern 'prompt done|decoding chunk' | ForEach-Object { $_.Line }

