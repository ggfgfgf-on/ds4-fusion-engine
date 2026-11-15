# watch_model.ps1 - keeps the model download alive until complete.
# If aria2 dies for any reason, this restarts it with the same URL and
# resume state (the .aria2 control file preserves segmented progress).
$model = 'D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$dir = 'D:\DSHwork\models'
$url = 'https://hf-mirror.com/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$name = Split-Path $model -Leaf

function Test-Complete {
    if (-not (Test-Path $model)) { return $false }
    return ((Get-Item $model).Length -ge (81GB * 0.999))
}

while (-not (Test-Complete)) {
    $proc = Get-Process aria2c -ErrorAction SilentlyContinue
    if (-not $proc) {
        "watchdog: aria2 not running, starting resume at $(Get-Date -Format HH:mm:ss)"
        Start-Process aria2c -ArgumentList @(
            '--console-log-level=warn', '--summary-interval=120',
            '-x', '8', '-s', '8', '-k', '4M',
            '--file-allocation=none', '--continue=true',
            '--max-tries=0', '--retry-wait=15', '--timeout=30', '--connect-timeout=15',
            '-d', $dir, '-o', $name, $url
        ) -WindowStyle Hidden
    }
    Start-Sleep -Seconds 60
}
"watchdog: model download complete at $(Get-Date -Format HH:mm:ss)"
