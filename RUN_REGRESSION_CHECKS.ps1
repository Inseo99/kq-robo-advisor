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
        [pscustomobject]@{ Command = "python"; Args = @() },
        [pscustomobject]@{ Command = "py"; Args = @("-3") },
        [pscustomobject]@{ Command = "$env:USERPROFILE\anaconda3\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"; Args = @() },
        [pscustomobject]@{ Command = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        $found = Test-PythonCommand -Command $candidate.Command -Args $candidate.Args
        if ($found) {
            return $found
        }
    }
    return $null
}

function Test-PythonModule {
    param(
        [object]$Python,
        [string]$Module
    )

    & $Python.Command @($Python.Args) -c "import $Module" 2>$null
    return $LASTEXITCODE -eq 0
}

function Ensure-Pytest {
    param(
        [object]$Python
    )

    if (Test-PythonModule -Python $Python -Module "pytest") {
        return
    }

    Write-Host ""
    Write-Host "pytest is not installed for this Python. Installing pytest..."
    & $Python.Command @($Python.Args) -m pip install pytest
    if ($LASTEXITCODE -ne 0) {
        throw "pytest installation failed. Run: python -m pip install pytest"
    }

    if (-not (Test-PythonModule -Python $Python -Module "pytest")) {
        throw "pytest is still unavailable after installation."
    }
}

function Invoke-Check {
    param(
        [string]$Name,
        [scriptblock]$Block
    )

    Write-Host ""
    Write-Host "---- $Name ----"
    & $Block
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Host "PASS: $Name" -ForegroundColor Green
}

Write-Host "============================================================"
Write-Host "KQ Quant Tool - Regression checks"
Write-Host "============================================================"
Write-Host "Project folder: $Root"

if (-not (Test-Path -LiteralPath (Join-Path $Root "server.py"))) {
    Write-Host "server.py was not found. Please run this file inside the project folder." -ForegroundColor Red
    exit 1
}

$python = Find-Python
if (-not $python) {
    Write-Host "Python was not found on this computer." -ForegroundColor Red
    Write-Host "Install Python 3.10 or newer, then run this file again."
    Write-Host "Download: https://www.python.org/downloads/"
    Write-Host "Important: check 'Add python.exe to PATH' during installation."
    exit 1
}

Write-Host "Python found: $($python.Version)"
Ensure-Pytest -Python $python

Invoke-Check "Python syntax" {
    & $python.Command @($python.Args) -m py_compile `
        server.py `
        tests\validation_signal_quality_alpha_decay.py `
        tests\smoke_api.py `
        tests\verify_test_count_docs.py
}

Invoke-Check "Package compile" {
    & $python.Command @($python.Args) -m compileall -q src\kq_tool
}

Invoke-Check "Unit tests" {
    & $python.Command @($python.Args) -m pytest tests\unit -q
}

Invoke-Check "Documented test count" {
    & $python.Command @($python.Args) tests\verify_test_count_docs.py
}

Write-Host ""
Write-Host "All regression checks passed." -ForegroundColor Green
