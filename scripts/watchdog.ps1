# watchdog.ps1 - kill switch for ds4-server GPU-shared-pool flood experiments.
#
# WDDM lets cudaMalloc spill into the shared GPU pool; exhausting it freezes
# DWM and the whole desktop.  This script force-kills the watched process the
# moment its shared-pool usage crosses the threshold, BEFORE the desktop can
# starve.  It is independent of the app's own code (perf-counter based), so it
# works even when ds4 hangs or floods silently.
#
# Usage (run as a background job from a test script):
#   powershell -File watchdog.ps1 -TargetPid <pid> [-SharedGB 24] [-TotalGB 40]
#        [-MaxMinutes 20] [-PollSec 2] [-State watchdog_state.txt]
#
# Exit codes: 0 = watched process exited on its own (see state file)
#             42 = KILLED by the watchdog (state file says why)
param(
    [Parameter(Mandatory = $true)][int]$TargetPid,
    [double]$SharedGB = 24,
    [double]$TotalGB = 40,
    [int]$MaxMinutes = 20,
    [int]$PollSec = 2,
    [string]$State = 'watchdog_state.txt'
)

$ErrorActionPreference = 'Continue'
Set-Location 'D:\DSHwork\src\ds4'
$sharedBytes = [long]($SharedGB * 1073741824.0)
$totalBytes  = [long]($TotalGB  * 1073741824.0)
$deadline = (Get-Date).AddMinutes($MaxMinutes)
$inst = 'pid_{0}*' -f $TargetPid

function Get-GpuUsage {
    $shared = 0L; $dedicated = 0L
    $sharedSamples = (Get-Counter ("\GPU Process Memory({0})\Shared Usage" -f $inst) -ErrorAction SilentlyContinue).CounterSamples
    if ($sharedSamples) { $shared = [long](($sharedSamples | Measure-Object CookedValue -Sum).Sum) }
    $dedicatedSamples = (Get-Counter ("\GPU Process Memory({0})\Dedicated Usage" -f $inst) -ErrorAction SilentlyContinue).CounterSamples
    if ($dedicatedSamples) { $dedicated = [long](($dedicatedSamples | Measure-Object CookedValue -Sum).Sum) }
    return @($shared, $dedicated)
}

function Write-State([string]$status, [string]$detail) {
    "[{0}] {1} {2}" -f (Get-Date -Format 'HH:mm:ss'), $status, $detail | Out-File -FilePath $State -Encoding ascii
}

"watchdog armed: pid=$TargetPid sharedThreshold=$SharedGB GiB totalThreshold=$TotalGB GiB maxMinutes=$MaxMinutes" | Out-File -FilePath $State -Encoding ascii

while ($true) {
    Start-Sleep -Seconds $PollSec
    $proc = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if (-not $proc) {
        $u = Get-GpuUsage
        Write-State 'ENDED' "process exited; last shared=$([math]::Round($u[0]/1MB,1)) MiB dedicated=$([math]::Round($u[1]/1MB,1)) MiB"
        exit 0
    }
    $u = Get-GpuUsage
    $sharedMiB = [math]::Round($u[0] / 1MB, 1)
    $dedMiB    = [math]::Round($u[1] / 1MB, 1)
    if ($u[0] -gt $sharedBytes) {
        Write-State 'KILLED' "shared ${sharedMiB} MiB > $SharedGB GiB (dedicated ${dedMiB} MiB)"
        Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
        exit 42
    }
    if (($u[0] + $u[1]) -gt $totalBytes) {
        Write-State 'KILLED' "dedicated+shared $([math]::Round(($u[0]+$u[1])/1MB,1)) MiB > $TotalGB GiB"
        Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
        exit 42
    }
    if ((Get-Date) -gt $deadline) {
        Write-State 'KILLED' "max-runtime $MaxMinutes min exceeded"
        Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
        exit 42
    }
}
