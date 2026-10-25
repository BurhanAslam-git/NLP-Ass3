# One-time setup on Windows (PowerShell equivalent of setup.sh).
# Usage:  powershell -ExecutionPolicy Bypass -File .\setup.ps1 [-SystemSitePackages]
#   -SystemSitePackages  reuse packages already installed globally (e.g. torch) to save disk space.
param([switch]$SystemSitePackages)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1. Python virtual environment
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment .venv ..."
    if ($SystemSitePackages) { python -m venv --system-site-packages .venv } else { python -m venv .venv }
}
$py = ".\.venv\Scripts\python.exe"

# 2. Python dependencies (no pip cache, to save disk space)
Write-Host "Installing Python dependencies..."
& $py -m pip install --no-cache-dir --timeout 120 --retries 8 -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 3. Local config
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

# 4. Ollama + LLM model
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama not found. Install it from https://ollama.com/download/windows and re-run."
}
$model = "qwen2.5:1.5b-instruct"
$line = Select-String -Path ".env" -Pattern "^OLLAMA_MODEL=(.+)$" | Select-Object -First 1
if ($line) { $model = $line.Matches[0].Groups[1].Value.Trim() }
if (-not (ollama list | Select-String -SimpleMatch $model)) {
    Write-Host "Pulling $model (one-time download, ~2 GB)..."
    ollama pull $model
} else {
    Write-Host "Model already present: $model"
}

# 5. Local voice models (Silero VAD, faster-whisper, Piper voice) into models/
Write-Host "Downloading voice models (one time)..."
$env:PYTHONPATH = "."
& $py scripts\download_models.py
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }

# 6. Data folders + RAG index
New-Item -ItemType Directory -Force "data\documents", "data\chroma_db", "logs" | Out-Null
Write-Host "Building the RAG index..."
$env:PYTHONPATH = "."
& $py -m backend.app.rag.indexer

Write-Host ""
Write-Host "Setup complete. Start the app with:  powershell -ExecutionPolicy Bypass -File .\run.ps1"
Write-Host "Then open http://127.0.0.1:8000"
