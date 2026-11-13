# mid_batch_cert.ps1 - certification chain for the migrated state so far
# (clusters 1-3 + Q8 reserve + prefetch). One background run, one report.
$ErrorActionPreference = 'Continue'
$root = 'D:\DSHwork\ds4-upstream'
Set-Location $root
Get-Process ds4-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
'=== [1/3] 16-vs-auto byte gate ==='
.\upstream_auto_cert.ps1 2>&1 | Select-String -Pattern 'MODE|WDDM cluster cert|WDDM band scheduler' | ForEach-Object { $_.Line }
'=== [2/3] long-prompt stress (headroom cap active) ==='
.\upstream_lru_stress.ps1 2>&1 | Select-String -Pattern 'STRESS|LRU evictions|crash markers' | ForEach-Object { $_.Line }
'=== [3/3] prefetch smoke (PREDICT=1 + LOG) ==='
$env:DS4_CUDA_PREFETCH_PREDICT = '1'
$env:DS4_CUDA_PREFETCH_PREDICT_LOG = '1'
.\upstream_lru_stress.ps1 2>&1 | Select-String -Pattern 'STRESS|LRU evictions|crash markers|prefetch stats' | ForEach-Object { $_.Line }
Remove-Item Env:\DS4_CUDA_PREFETCH_PREDICT, Env:\DS4_CUDA_PREFETCH_PREDICT_LOG -ErrorAction SilentlyContinue
'=== mid-batch chain done ==='
