param([string]$Python = "$PSScriptRoot\venv\python.exe", [switch]$Resume, [ValidateRange(0,5000)][int]$ValidationLimit = 500, [ValidateRange(0,1000000)][int]$BenchmarkRequests = 500, [switch]$QuickBenchmark)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot 'runs\ultralytics'
function Invoke-Suite([string[]]$Arguments) {
    & $Python -m vonnx_tools @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Suite stopped: $($Arguments -join ' ')" }
}
Invoke-Suite @('prepare','--datasets','full')
$validateArgs = @('validate','--full','--output','runs\validation','--timeout','21600')
if ($ValidationLimit -gt 0) { $validateArgs += @('--limit', "$ValidationLimit") }
if ($Resume) { $validateArgs += '--resume' }
Invoke-Suite $validateArgs
$benchArgs = @('benchmark','--output','runs\benchmark','--validation','runs\validation')
if ($BenchmarkRequests -gt 0) { $benchArgs += @('--max-requests', "$BenchmarkRequests") }
if ($QuickBenchmark) {
    $benchArgs = @('benchmark','--output','runs\benchmark_quick','--validation','runs\validation','--sizes','640','--streams','1','16','--repeats','1','--max-requests','100')
    Write-Host 'Quick benchmark: 640, streams 1/16, one repeat, up to 100 measured requests per model/case.'
}
if ($ValidationLimit -gt 0) { $benchArgs += '--allow-sampled-validation' }
if ($Resume) { $benchArgs += '--resume' }
Invoke-Suite $benchArgs
Write-Host 'Complete. Results: runs\benchmark or runs\benchmark_quick\REPORT.md and runs\validation\summary.json'
