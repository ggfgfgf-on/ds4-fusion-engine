# watch_dspark.ps1 - download the DSpark drafter GGUF (Q8_0, 0731 series)
# via hf-mirror with aria2 resume, same pattern as watch_model.ps1.
$dir = 'D:\DSHwork\models'
$name = 'dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf'
$model = Join-Path $dir $name
$url = 'https://hf-mirror.com/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF/resolve/main/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf'

function Test-Complete {
    # completion = aria2 gone AND a control file no longer present AND
    # the file exists with non-trivial size (exact size unknown up front)
    if (-not (Test-Path $model)) { return $false }
    if (Test-Path "$model.aria2") { return $false }
    $proc = Get-Process aria2c -ErrorAction SilentlyContinue
    if ($proc) { return $false }
    return ((Get-Item $model).Length -gt 100MB)
}

while (-not (Test-Complete)) {
    $proc = Get-Process aria2c -ErrorAction SilentlyContinue
    if (-not $proc) {
        "watchdog: aria2 not running, starting resume at $(Get-Date -Format HH:mm:ss)" | Tee-Object -FilePath (Join-Path $dir 'dspark_download.log') -Append
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
"watchdog: DSpark drafter download complete at $(Get-Date -Format HH:mm:ss), size $([math]::Round((Get-Item $model).Length/1MB,1)) MiB"
