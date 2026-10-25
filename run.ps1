# Start the app on Windows (PowerShell equivalent of run.sh).
# Usage:  powershell -ExecutionPolicy Bypass -File .\run.ps1 [-Reload]
# Starts Ollama only if it is not already running, and stops it on exit if we started it.
param([switch]$Reload)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "."
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No .venv found. Run setup.ps1 first." }

function Test-Ollama {
    try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}

$ollamaProc = $null
if (Test-Ollama) {
    Write-Host "Ollama already running. It will stay up after you stop the app."
} else {
    Write-Host "Starting Ollama for this app session (log: .ollama-runtime.log)..."
    $ollamaProc = Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru `
        -RedirectStandardError ".ollama-runtime.log"
    $ready = $false
    for ($i = 0; $i -lt 45; $i++) {
        if (Test-Ollama) { $ready = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { Stop-Process -Id $ollamaProc.Id -ErrorAction SilentlyContinue; throw "Ollama did not start." }
    Write-Host "Ollama is ready."
}

try {
    $hostAddr = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
    $port = if ($env:PORT) { $env:PORT } else { "8000" }
    $uvicornArgs = @("-m", "uvicorn", "backend.app.main:app", "--host", $hostAddr, "--port", $port)
    if ($Reload) { $uvicornArgs += "--reload" }
    & $py @uvicornArgs
} finally {
    if ($ollamaProc) {
        Write-Host "Stopping Ollama (started by this app)..."
        Stop-Process -Id $ollamaProc.Id -ErrorAction SilentlyContinue
    }
}
