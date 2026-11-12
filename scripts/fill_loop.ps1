# fill_loop.ps1 — resumable model-file range filler with channel auto-switch.
# Prefers hf-mirror (fast, no expiry); falls back to signed CDN URL via proxy (refresh every 40min).
param([string]$Manifest = 'D:\DSHwork\models\zero_ranges.json',
      [string]$OutFile = 'D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf',
      [string]$RepoFile = 'DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf')
$manifest = $Manifest
$file     = $OutFile
$mirrorUrl = "https://hf-mirror.com/antirez/deepseek-v4-gguf/resolve/main/$RepoFile"
$hfUrl     = "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/$RepoFile"
$exe       = 'D:\DSHwork\models\fill_ranges.exe'
$log       = 'D:\DSHwork\models\fill_loop.log'
function Log($m){ $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $line | Tee-Object -FilePath $log -Append }
Log "fill_loop start"

while ($true) {
    # prefer hf-mirror if it is back
    $code = curl.exe -s -o NUL -w "%{http_code}" --max-time 8 -r 0-1023 $mirrorUrl 2>$null
    if ($code -eq '206') {
        $url = $mirrorUrl; $window = 1500; Log "channel=hf-mirror (recovered!)"
    } else {
        $loc = (curl.exe -sI -x http://127.0.0.1:7890 --max-time 25 $hfUrl 2>$null | Select-String -Pattern '^location:').ToString()
        if (-not $loc -or $loc.Length -lt 20) { Log "cannot refresh signed URL, sleep 300"; Start-Sleep 300; continue }
        $url = $loc.Substring(10).Trim(); $window = 2400; Log "channel=cdn-signed"
    }
    $p = Start-Process -FilePath $exe -ArgumentList @($manifest, $file, $url, 'direct', 16, 2097152) -NoNewWindow -PassThru
    Log "filler pid=$($p.Id) window=${window}s"
    while (-not $p.HasExited -and ((Get-Date) - $p.StartTime).TotalSeconds -lt $window) { Start-Sleep 15 }
    if (-not $p.HasExited) {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        Log "window elapsed, killed filler for URL refresh"
    }
    $p.WaitForExit()
    $code = $p.ExitCode
    Log "filler exited code=$code"
    if ($code -eq 0) { Log "ALL RANGES FILLED"; break }
    Log "partial exit, restarting with refreshed channel"
    Start-Sleep 5
}
Log "fill_loop done"
