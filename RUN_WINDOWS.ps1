$ErrorActionPreference = "Stop"

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:KQ_DISABLE_TABPFN = "1"
$env:KQ_AUTO_OPEN_BROWSER = "0"

$SrcPath = Join-Path $Root "src"
if (Test-Path -LiteralPath $SrcPath) {
    if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
        $env:PYTHONPATH = $SrcPath
    } else {
        $env:PYTHONPATH = "$SrcPath;$env:PYTHONPATH"
    }
}

function Test-PythonCommand {
    param(
        [string]$Command,
        [string[]]$Args = @()
    )

    $exists = $false
    if (Test-Path -LiteralPath $Command) {
        $exists = $true
    } elseif (Get-Command $Command -ErrorAction SilentlyContinue) {
        $exists = $true
    }

    if (-not $exists) {
        return $null
    }

    try {
        $version = & $Command @Args --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$version" -match "Python") {
            return [pscustomobject]@{ Command = $Command; Args = $Args; Version = "$version" }
        }
    } catch {
        return $null
    }
    return $null
}

function Find-Python {
    $candidates = @(
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:USERPROFILE\anaconda3\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "py"; Args = @("-3.13") },
        [pscustomobject]@{ Command = "py"; Args = @("-3.12") },
        [pscustomobject]@{ Command = "py"; Args = @("-3.11") },
        [pscustomobject]@{ Command = "py"; Args = @("-3.10") },
        [pscustomobject]@{ Command = "py"; Args = @("-3") },
        [pscustomobject]@{ Command = "python"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        $found = Test-PythonCommand -Command $candidate.Command -Args $candidate.Args
        if ($found) {
            return $found
        }
    }
    return $null
}

function Open-BrowserLater {
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 8
        $url = "http://127.0.0.1:8888/"
        $paths = @(
            "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
            "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
            "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
        )
        $chrome = $paths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if ($chrome) {
            Start-Process -FilePath $chrome -ArgumentList $url
        } else {
            Start-Process $url
        }
    } | Out-Null
}

Write-Host "============================================================"
Write-Host "KQ Quant Tool - Windows launcher"
Write-Host "============================================================"
Write-Host "Project folder: $Root"
Write-Host ""

if (-not (Test-Path -LiteralPath (Join-Path $Root "server.py"))) {
    Write-Host "server.py was not found. Please run this file inside the project folder." -ForegroundColor Red
    exit 1
}

$python = Find-Python
if (-not $python) {
    Write-Host "Python was not found on this computer." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python 3.10 or newer, then run RUN_KQ_TOOL.bat again."
    Write-Host "Download: https://www.python.org/downloads/"
    Write-Host "Important: check 'Add python.exe to PATH' during installation."
    Write-Host ""
    Write-Host "If Anaconda is already installed, open Anaconda Prompt and run:"
    Write-Host "  cd /d `"$Root`""
    Write-Host "  set PYTHONPATH=%CD%\src;%PYTHONPATH%"
    Write-Host "  python -m kq_tool"
    exit 1
}

Write-Host "Python found: $($python.Version)"
if ("$($python.Version)" -match "Python 3\.14") {
    Write-Host "Note: Python 3.14 skips optional hmmlearn because Windows wheels are not available yet." -ForegroundColor Yellow
    Write-Host "      The app will still run with the built-in regime transition fallback."
}
Write-Host ""

if (Test-Path -LiteralPath (Join-Path $Root "requirements.txt")) {
    Write-Host "Checking/installing Python packages..."
    & $python.Command @($python.Args) -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Package installation failed. Please check the error above." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host ""
}

Write-Host "Starting KQ Quant Tool server..."
Write-Host "URL: http://127.0.0.1:8888/"
Write-Host "Keep this window open while using the tool."
Write-Host "Press Ctrl+C in this window to stop the server."
Write-Host ""

Open-BrowserLater
& $python.Command @($python.Args) -m kq_tool
