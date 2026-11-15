# dl_resume_loop.ps1 - resume-until-complete loop for the DSpark drafter.
# curl -C - per iteration (300s cap), repeats until the file reaches the
# Content-Length advertised by the mirror. Survives throttling by simply
# retrying; exits when done.
$file = 'D:\DSHwork\models\dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf'
$url  = 'https://hf-mirror.com/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF/resolve/main/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf'
$log  = 'D:\DSHwork\models\dspark_resume.log'

$expected = $null
for ($i = 0; $i -lt 10; $i++) {
    try {
        $r = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 20 -MaximumRedirection 5
        $expected = [long]$r.Headers['Content-Length']
        break
    } catch { Start-Sleep -Seconds 5 }
}
if (-not $expected) { "no Content-Length; abort" | Out-File $log -Append; exit 1 }
"expected bytes: $expected" | Out-File $log

$iter = 0
while ($true) {
    $sz = if (Test-Path $file) { (Get-Item $file).Length } else { 0 }
    if ($sz -ge $expected) { "DONE at $([math]::Round($sz/1MB,2)) MiB" | Out-File $log -Append; break }
    $iter++
    $t0 = Get-Date
    curl.exe -L -s -C - -o $file $url --max-time 300 2>$null | Out-Null
    $sz2 = (Get-Item $file).Length
    $dt = ((Get-Date) - $t0).TotalSeconds
    $rate = if ($dt -gt 0) { ($sz2 - $sz) / $dt / 1MB } else { 0 }
    "[{0}] iter {1}: {2:N2} MiB (+{3:N2} MiB @ {4:N2} MiB/s, remain {5:N2} MiB)" -f (Get-Date -Format HH:mm:ss), $iter, ($sz2/1MB), (($sz2-$sz)/1MB), $rate, (($expected-$sz2)/1MB) | Out-File $log -Append
    if ($sz2 -ge $expected) { "DONE" | Out-File $log -Append; break }
    Start-Sleep -Seconds 2
}
