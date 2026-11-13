# upstream_bitq_cert.ps1 - bitq certification: OFF regression must stay
# byte-identical; B/A profiles must boot, decode 48 tokens, zero crashes.
$ErrorActionPreference = 'Continue'
$root = 'D:\DSHwork\ds4-upstream'
$model = 'D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$shExe = Join-Path $PSHOME 'powershell.exe'
Set-Location $root
$prompt = "Write a comprehensive technical analysis of solar energy storage systems, comparing lithium-ion battery installations, pumped hydro storage, compressed air energy storage, and molten salt thermal storage in terms of round-trip efficiency, capital cost per kilowatt-hour, expected cycle life, land use requirements, and grid-scale deployment examples from California, Texas, Australia, and China, then discuss how each technology performs during peak evening demand periods, seasonal variations in solar output, and extreme weather events, and finally recommend an optimal storage portfolio for a utility aiming to reach ninety percent renewable penetration by 2040."
foreach ($v in 'http_proxy','https_proxy','no_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','NO_PROXY','ALL_PROXY') { Remove-Item Env:\$v -ErrorAction SilentlyContinue }
$env:DS4_THREADS = '28'; $env:DS4_POOL_SPIN_US = '500'; $env:DS4_POOL_PIN = '1'
$env:PATH = "D:\DSHwork\mf\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"
$env:DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL = '1'
Remove-Item Env:\DS4_CUDA_KV_BITQ, Env:\DS4_CUDA_KV_BITQ_BITS, Env:\DS4_CUDA_KV_BITQ_RAW_BITS, Env:\DS4_CUDA_KV_BITQ_COMP_BITS, Env:\DS4_CUDA_KV_BITQ_IDX_BITS -ErrorAction SilentlyContinue

function Run-One($tag, $extraEnv) {
    Get-Process ds4-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    foreach ($k in $extraEnv.Keys) { Set-Item "Env:\$k" $extraEnv[$k] }
    $srvLog = Join-Path $root ".bitq_${tag}_srv.txt"
    $p = Start-Process -FilePath (Join-Path $root 'ds4-server.exe') `
        -ArgumentList @('--cuda','--ssd-streaming','-m',$model,'--ssd-streaming-cache-experts','16','--ctx','8192','--host','127.0.0.1','--port','8000') `
        -WorkingDirectory $root -RedirectStandardError $srvLog -RedirectStandardOutput (Join-Path $root ".bitq_${tag}_out.txt") `
        -PassThru -WindowStyle Hidden
    $wd = Start-Process $shExe -ArgumentList @('-NoProfile','-File','D:\DSHwork\ds4-upstream\watchdog.ps1','-TargetPid',$p.Id,`
        '-SharedGB','24','-TotalGB','40','-MaxMinutes','20','-PollSec','2','-State',(Join-Path $root ".wd_bitq_${tag}.txt")) `
        -PassThru -WindowStyle Hidden -ErrorAction SilentlyContinue
    try {
        $ready = $false
        for ($i = 0; $i -lt 90; $i++) {
            Start-Sleep -Seconds 2
            try { $null = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop; $ready = $true; break } catch {
                if (Select-String -Path $srvLog -Pattern 'listening on' -Quiet) { $ready = $true; break }
            }
        }
        if (-not $ready) { "BITQ $tag SERVER_NOT_READY" } else {
            $body = @{ model = 'deepseek-chat'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = 48; temperature = 0; stream = $false } | ConvertTo-Json -Depth 5
            $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $body -TimeoutSec 1500 -UseBasicParsing
            $j = [System.Text.Encoding]::UTF8.GetString($r.RawContentStream.ToArray()) | ConvertFrom-Json
            $out = $j.choices[0].message.content
            Set-Content -Path (Join-Path $root ".bitq_content_${tag}.txt") -Value $out -Encoding utf8
            "BITQ $tag tokens=$($j.usage.completion_tokens) bytes=$($out.Length)"
            $out.Substring(0, [math]::Min(90, $out.Length))
        }
    } catch { "BITQ $tag REQUEST_FAILED: $($_.Exception.Message)" } finally {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 4
        if ($wd -and -not $wd.HasExited) { Stop-Process -Id $wd.Id -Force -ErrorAction SilentlyContinue }
    }
    foreach ($k in $extraEnv.Keys) { Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
    '--- memory plan ---'
    Select-String -Path $srvLog -Pattern 'memory: KV|planned' | ForEach-Object { $_.Line }
    '--- crash markers ---'
    $crash = Select-String -Path $srvLog -Pattern 'illegal memory access|an illegal' | Measure-Object
    "crash markers: $($crash.Count)"
}

'=== [B profile] raw6/comp4/idx4 ==='
Run-One 'componly' @{ DS4_CUDA_KV_BITQ_COMP = '1'; DS4_CUDA_KV_BITQ_COMP_BITS = '4' }
Run-One 'B' @{ DS4_CUDA_KV_BITQ = '1'; DS4_CUDA_KV_BITQ_RAW_BITS = '6'; DS4_CUDA_KV_BITQ_COMP = '1'; DS4_CUDA_KV_BITQ_COMP_BITS = '4' }
'=== [A profile] raw4/comp4/idx4 ==='
Run-One 'A' @{ DS4_CUDA_KV_BITQ = '1'; DS4_CUDA_KV_BITQ_BITS = '4'; DS4_CUDA_KV_BITQ_COMP = '1';  }
'=== bitq cert chain done ==='



