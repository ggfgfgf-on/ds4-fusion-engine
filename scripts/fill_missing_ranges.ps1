# fill_missing_ranges.ps1 - Fallback when hf-mirror is down for a long time.
# Parses the aria2 control file bitfield, then downloads each missing piece
# range through the FlClash proxy directly into the model file.
param(
    [string]$Model = 'D:\DSHwork\models\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf',
    [string]$Url = 'https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf',
    [int]$Concurrency = 8
)
$ErrorActionPreference = 'Stop'
$ctl = "$Model.aria2"
if (-not (Test-Path $ctl)) { throw "control file missing: $ctl" }

# aria2 control file layout (all multi-byte fields network/big-endian):
# u16 version(1) | u16 extension | u32 infoHashLength | infoHash |
# u32 pieceLength | u64 totalLength | u64 uploadLength | u8 bitfieldLength |
# bitfield (bits: 1 = complete piece)
$bytes = [IO.File]::ReadAllBytes($ctl)
function BE16([byte[]]$b, [int]$o) { $v = [BitConverter]::ToUInt16($b, $o); return ((($v -band 0xFF) -shl 8) -bor ($v -shr 8)) }
function BE32([byte[]]$b, [int]$o) { $t = $b[$o..($o+3)]; [Array]::Reverse($t); return [BitConverter]::ToUInt32($t, 0) }
function BE64([byte[]]$b, [int]$o) { $t = $b[$o..($o+7)]; [Array]::Reverse($t); return [BitConverter]::ToUInt64($t, 0) }
$ver = BE16 $bytes 0
if ($ver -ne 1) { throw "unsupported control file version $ver" }
$ihl = BE32 $bytes 4
$off = 8 + $ihl
$pieceLen = BE32 $bytes $off; $off += 4
$totalLen = BE64 $bytes $off; $off += 8
$uploadLen = BE64 $bytes $off; $off += 8
$bitLen = $bytes[$off]; $off += 1
$pieces = [int]($totalLen / $pieceLen)
if ($totalLen % $pieceLen -ne 0) { $pieces++ }
"total=$totalLen pieceLen=$pieceLen pieces=$pieces bitfieldBytes=$bitLen"

# find missing pieces -> coalesce into ranges
$ranges = New-Object System.Collections.Generic.List[string]
$start = -1
for ($i = 0; $i -lt $pieces; $i++) {
    $byteIdx = [int]($i / 8)
    $bit = 1 -shl ($i % 8)
    $done = ($byteIdx -lt $bitLen) -and (($bytes[$off + $byteIdx] -band $bit) -ne 0)
    if (-not $done -and $start -lt 0) { $start = $i }
    if ($done -and $start -ge 0) {
        $ranges.Add("$start-$($i-1)")
        $start = -1
    }
}
if ($start -ge 0) { $ranges.Add("$start-$($pieces-1)") }
"missing piece groups: $($ranges.Count)"

# download each missing piece through the proxy (curl per piece, concurrency N)
$jobs = @()
foreach ($g in $ranges) {
    $sp = [int]($g.Split('-')[0]); $ep = [int]($g.Split('-')[1])
    $byteStart = [int64]$sp * $pieceLen
    $byteEnd = [Math]::Min([int64]($ep + 1) * $pieceLen, $totalLen) - 1
    $tmp = "$Model.rng$sp-$ep.part"
    $cmd = "curl.exe -sL -x http://127.0.0.1:7890 --retry 10 --retry-delay 5 -r $byteStart-$byteEnd -o `"$tmp`" `"$Url`""
    $jobs += [pscustomobject]@{ Sp=$sp; Ep=$ep; Start=$byteStart; End=$byteEnd; Tmp=$tmp; Pid=0 }
}
"spawning $($jobs.Count) curl workers (concurrency $Concurrency)..."
$running = New-Object System.Collections.Generic.List[object]
foreach ($j in $jobs) {
    while ($running.Count -ge $Concurrency) {
        $running = $running | Where-Object { -not $_.HasExited }
        if ($running.Count -ge $Concurrency) { Start-Sleep -Seconds 2 }
    }
    $p = Start-Process curl.exe -ArgumentList @('-sL','-x','http://127.0.0.1:7890','--retry','10','--retry-delay','5','-r',"$($j.Start)-$($j.End)",'-o',$j.Tmp,$Url) -PassThru -WindowStyle Hidden
    $running.Add($p)
}
$running | ForEach-Object { $_.WaitForExit() }

# verify + splice into the model file at the correct offsets
$fs = [IO.File]::Open($Model, [IO.FileMode]::Open, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
try {
    foreach ($j in $jobs) {
        $len = (Get-Item $j.Tmp).Length
        $expect = $j.End - $j.Start + 1
        if ($len -ne $expect) { "range $($j.Sp)-$($j.Ep): size mismatch $len vs $expect"; continue }
        $data = [IO.File]::ReadAllBytes($j.Tmp)
        $fs.Seek($j.Start, [IO.SeekOrigin]::Begin) | Out-Null
        $fs.Write($data, 0, $data.Length)
        Remove-Item $j.Tmp -ErrorAction SilentlyContinue
        "range $($j.Sp)-$($j.Ep): OK"
    }
} finally { $fs.Close() }
"done. model size now: $([math]::Round((Get-Item $Model).Length/1GB,2)) GB"
