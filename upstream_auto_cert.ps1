# upstream_auto_cert.ps1 - WDDM cluster certification on the upstream base:
# 16-expert run vs auto run must produce byte-identical 48-token output
# (the scheduler changes only the budget number, never the token path).
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
Remove-Item Env:\DS4_CUDA_KV_BITQ, Env:\DS4_CUDA_KV_BITQ_AUTO, Env:\DS4_CUDA_WEIGHT_ARENA_LRU -ErrorAction SilentlyContinue
foreach ($v in 'http_proxy','https_proxy','no_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','NO_PROXY','ALL_PROXY') { Remove-Item Env:\$v -ErrorAction SilentlyContinue }
$prompt = "Write a comprehensive technical analysis of solar energy storage systems, comparing lithium-ion battery installations, pumped hydro storage, compressed air energy storage, and molten salt thermal storage in terms of round-trip efficiency, capital cost per kilowatt-hour, expected cycle life, land use requirements, and grid-scale deployment examples from California, Texas, Australia, and China, then discuss how each technology performs during peak evening demand periods, seasonal variations in solar output, and extreme weather events, and finally recommend an optimal storage portfolio for a utility aiming to reach ninety percent renewable penetration by 2040."
$outs = @{}
foreach ($mode in '16','auto') {
    Remove-Item -Recurse -Force (Join-Path $root '.cache\server-kv-hybrid') -ErrorAction SilentlyContinue
    $srvLog = Join-Path $root ".upstream_${mode}_srv.txt"
    $p = Start-Process -FilePath (Join-Path $root 'ds4-server.exe') `
        -ArgumentList @('--cuda','--ssd-streaming','-m',$model,'--ssd-streaming-cache-experts',$mode,'--ctx','8192','--host','127.0.0.1','--port','8000') `
        -WorkingDirectory $root -RedirectStandardError $srvLog -RedirectStandardOutput (Join-Path $root ".upstream_${mode}_out.txt") `
        -PassThru -WindowStyle Hidden
    $wd = Start-Process $shExe -ArgumentList @('-NoProfile','-File','D:\DSHwork\src\ds4\watchdog.ps1','-TargetPid',$p.Id,`
        '-SharedGB','24','-TotalGB','40','-MaxMinutes','20','-PollSec','2','-State',(Join-Path $root ".wd_${mode}.txt")) `
        -PassThru -WindowStyle Hidden -ErrorAction SilentlyContinue
    try {
        $ready = $false
        for ($i = 0; $i -lt 90; $i++) {
            Start-Sleep -Seconds 2
            try { $null = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; $ready = $true; break } catch {
                if (Select-String -Path $srvLog -Pattern 'listening on' -Quiet) { $ready = $true; break }
            }
        }
        if (-not $ready) { "MODE $mode SERVER_NOT_READY" } else {
            $body = @{ model = 'deepseek-chat'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 48; temperature = 0; stream = $false } | ConvertTo-Json -Depth 5
            $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 1500 -UseBasicParsing
            $j = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json
            $outs[$mode] = $j.choices[0].message.content
            "MODE $mode tokens=$($j.usage.completion_tokens)"
        }
    } catch { "MODE $mode REQUEST_FAILED: $($_.Exception.Message)" } finally {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 4
        if ($wd -and -not $wd.HasExited) { Stop-Process -Id $wd.Id -Force -ErrorAction SilentlyContinue }
    }
}
if ($outs['16'] -and $outs['auto']) {
    $same = ($outs['16'].Length -eq $outs['auto'].Length)
    if ($same) { for ($i = 0; $i -lt $outs['16'].Length; $i++) { if ($outs['16'][$i] -ne $outs['auto'][$i]) { $same = $false; break } } }
    "WDDM cluster cert (16 vs auto, upstream base): $same ($($outs['16'].Length)B)"
} else { 'WDDM cluster cert: INCOMPLETE' }
'--- WDDM scheduler line ---'
Select-String -Path (Join-Path $root '.upstream_auto_srv.txt') -Pattern 'WDDM band scheduler|cache target' | ForEach-Object { $_.Line }
