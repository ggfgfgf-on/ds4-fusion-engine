# upstream_hybrid_smoke.ps1 - hybrid dual-engine smoke: GPU prefill -> CPU decode.
$ErrorActionPreference = 'Continue'
$root = 'D:\DSHwork\ds4-upstream'
$model = 'D:\DSHwork\models\Qwen3-8B-Instruct-Q4_K_M.gguf'
Set-Location $root
foreach ($v in 'http_proxy','https_proxy','no_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','NO_PROXY','ALL_PROXY') { Remove-Item Env:\$v -ErrorAction SilentlyContinue }
$env:DS4_THREADS = '28'; $env:DS4_POOL_SPIN_US = '500'; $env:DS4_POOL_PIN = '1'
$env:DS4_HYBRID_PREFILL_CHUNK = '0'
$env:PATH = "D:\DSHwork\mf\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"
Remove-Item Env:\DS4_CUDA_KV_BITQ, Env:\DS4_CUDA_KV_BITQ_AUTO -ErrorAction SilentlyContinue
Get-Process ds4-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Remove-Item -Recurse -Force (Join-Path $root '.cache\server-kv-hybrid') -ErrorAction SilentlyContinue
$srvLog = Join-Path $root '.hybrid_smoke_srv.txt'
$p = Start-Process -FilePath (Join-Path $root 'ds4-server.exe') `
    -ArgumentList @('--hybrid','--cuda','-m',$model,'--ssd-streaming-cache-experts','16','--ctx','8192','--host','127.0.0.1','--port','8000') `
    -WorkingDirectory $root -RedirectStandardError $srvLog -RedirectStandardOutput (Join-Path $root '.hybrid_smoke_out.txt') `
    -PassThru -WindowStyle Hidden
$wd = Start-Process (Join-Path $PSHOME 'powershell.exe') -ArgumentList @('-NoProfile','-File',(Join-Path $root 'watchdog.ps1'),'-TargetPid',$p.Id,'-SharedGB','24','-TotalGB','40','-MaxMinutes','25','-PollSec','2','-State',(Join-Path $root '.wd_hybrid.txt')) -PassThru -WindowStyle Hidden
try {
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Seconds 2
        if (Select-String -Path $srvLog -Pattern 'listening on' -Quiet) { $ready = $true; break }
    }
    if (-not $ready) { 'HYBRID SERVER_NOT_READY' } else {
        $prompt = "Write a comprehensive technical analysis of solar energy storage systems, comparing lithium-ion battery installations, pumped hydro storage, compressed air energy storage, and molten salt thermal storage in terms of round-trip efficiency, capital cost per kilowatt-hour, expected cycle life, land use requirements, and grid-scale deployment examples from California, Texas, Australia, and China, then discuss how each technology performs during peak evening demand periods, seasonal variations in solar output, and extreme weather events, and finally recommend an optimal storage portfolio for a utility aiming to reach ninety percent renewable penetration by 2040."
        $body = @{ model = 'deepseek-chat'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 32; temperature = 0; stream = $false } | ConvertTo-Json -Depth 5
        $t0 = Get-Date
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 1800 -UseBasicParsing
        $wall = ((Get-Date) - $t0).TotalSeconds
        $j = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json
        $o = $j.choices[0].message.content
        "HYBRID tokens=$($j.usage.completion_tokens) bytes=$($o.Length) wall=$([math]::Round($wall,1))s"
        $o.Substring(0, [math]::Min(100, $o.Length))
    }
} catch { "HYBRID FAILED: $($_.Exception.Message)" } finally {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    if ($wd -and -not $wd.HasExited) { Stop-Process -Id $wd.Id -Force -ErrorAction SilentlyContinue }
}
'--- hybrid handoff lines ---'
Select-String -Path $srvLog -Pattern 'hybrid|prefill done' | ForEach-Object { $_.Line } | Select-Object -First 4
