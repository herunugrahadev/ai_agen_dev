<#
.SYNOPSIS
    OpenMausBot + 9Router - Native Windows PowerShell Runner (Tanpa Docker)
.DESCRIPTION
    Menjalankan 9Router, Redis/Worker (opsional), dan OpenMausBot API secara lokal di Windows.
#>

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$BackgroundProcs = @()

try {
    # 1. Pastikan .env tersedia
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host ">>> File .env dibuat dari .env.example (isi API key LLM Anda di .env)." -ForegroundColor Cyan
    }

    # 2. Override environment agar berjalan di localhost (SQLite default jika tidak pakai PostgreSQL)
    if (-not $env:NINE_ROUTER_BASE_URL) { $env:NINE_ROUTER_BASE_URL = "http://localhost:20128/v1" }
    if (-not $env:REDIS_URL) { $env:REDIS_URL = "redis://localhost:6379/0" }
    if (-not $env:DATABASE_URL) { $env:DATABASE_URL = "sqlite+aiosqlite:///./openmausbot.db" }

    # 3. Siapkan Python virtual environment (.venv)
    if (-not (Test-Path ".venv")) {
        Write-Host ">>> Membuat Python virtual environment (.venv)..." -ForegroundColor Cyan
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            uv venv .venv
        } else {
            python -m venv .venv
        }
    }

    $VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
    $VenvUvicorn = Join-Path $RootDir ".venv\Scripts\uvicorn.exe"

    # Install dependensi jika belum terpasang
    & $VenvPython -c "import fastapi, openai, aiosqlite" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ">>> Menginstall dependensi Python..." -ForegroundColor Cyan
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            uv pip install -e ".[dev]"
        } else {
            & $VenvPython -m pip install -e ".[dev]"
        }
    }

    # 4. Cek Redis lokal; jika tidak ada, gunakan mode sinkron (REDIS_ENABLED=false)
    $RedisCmd = Get-Command redis-server -ErrorAction SilentlyContinue
    if ($RedisCmd) {
        Write-Host ">>> Menjalankan redis-server di localhost:6379..." -ForegroundColor Cyan
        $RedisProc = Start-Process -FilePath $RedisCmd.Source -ArgumentList "--port 6379" -PassThru -WindowStyle Hidden
        $BackgroundProcs += $RedisProc
        Start-Sleep -Seconds 1
        $env:REDIS_ENABLED = "true"
    } else {
        Write-Host ">>> [INFO] redis-server tidak ditemukan. Berjalan dalam mode sinkron (REDIS_ENABLED=false)." -ForegroundColor Yellow
        $env:REDIS_ENABLED = "false"
    }

    # 5. Jalankan 9Router Gateway di port 20128
    $NpxCmd = Get-Command npx -ErrorAction SilentlyContinue
    if ($NpxCmd) {
        Write-Host ">>> Menjalankan 9Router di http://localhost:20128/v1 ..." -ForegroundColor Cyan
        $RouterProc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c npx --yes 9router" -PassThru -WindowStyle Hidden
        $BackgroundProcs += $RouterProc
        Start-Sleep -Seconds 2
    } else {
        Write-Host ">>> [PERINGATAN] Node.js / npx tidak ditemukan. Pastikan 9Router sudah berjalan di $env:NINE_ROUTER_BASE_URL." -ForegroundColor Yellow
    }

    # 6. Jalankan OpenMausBot Worker jika Redis aktif
    if ($env:REDIS_ENABLED -eq "true") {
        Write-Host ">>> Menjalankan OpenMausBot Worker..." -ForegroundColor Cyan
        $WorkerProc = Start-Process -FilePath $VenvPython -ArgumentList "-m worker.main" -PassThru -NoNewWindow
        $BackgroundProcs += $WorkerProc
    }

    # 7. Jalankan OpenMausBot API & Webhook (foreground)
    Write-Host ">>> Menjalankan OpenMausBot API di http://localhost:8000 ..." -ForegroundColor Green
    Write-Host ">>> Tekan Ctrl+C untuk menghentikan semua layanan." -ForegroundColor Green
    & $VenvUvicorn api.main:app --host 0.0.0.0 --port 8000
}
finally {
    Write-Host ""
    Write-Host ">>> Menghentikan seluruh layanan lokal..." -ForegroundColor Cyan
    foreach ($proc in $BackgroundProcs) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host ">>> Selesai." -ForegroundColor Cyan
}
