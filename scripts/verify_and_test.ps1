# verify_and_test.ps1 — post-fill verification chain:
# fine zero-scan -> (fill if needed) -> SHA256 vs authoritative -> inference test.
$ErrorActionPreference = 'Continue'
$Model = 'D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
$ExpectedSha = 'ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0'
$ScanExe = 'D:\DSHwork\models\scan_zero.exe'
$Fine = 'D:\DSHwork\models\zero_ranges_fine.json'

"[{0}] fine scan (4KB blocks)..." -f (Get-Date -Format 'HH:mm:ss')
& $ScanExe $Model $Fine 4096
if ($LASTEXITCODE -ne 0) { "scan failed"; exit 1 }

# parse remaining zero bytes from scan output
$scanOut = (& $ScanExe $Model 'D:\DSHwork\models\zero_ranges_fine2.json' 4096) -join "`n"
$rem = 0
if ($scanOut -match 'zeroBytes=(\d+)') { $rem = [long]$Matches[1] }

if ($rem -gt 0) {
    "remaining zero bytes: {0:N1} GiB - running fill loop again" -f ($rem/1GB)
    & powershell -NoProfile -ExecutionPolicy Bypass -File D:\DSHwork\models\fill_loop.ps1 -Manifest $Fine
    if ($LASTEXITCODE -ne 0) { "fill loop incomplete"; exit 2 }
    # rescan after fill
    & $ScanExe $Model $Fine 4096
}

"[{0}] SHA256..." -f (Get-Date -Format 'HH:mm:ss')
$h = Get-FileHash -Algorithm SHA256 -Path $Model
"local sha256:  $($h.Hash)"
"expect sha256: $ExpectedSha"
if ($h.Hash -ne $ExpectedSha) {
    "HASH MISMATCH - manual intervention needed"
    exit 3
}
"HASH MATCH - model file is complete and correct."

"[{0}] running inference test..." -f (Get-Date -Format 'HH:mm:ss')
& D:\DSHwork\src\ds4\test_ds4.ps1
exit $LASTEXITCODE
