[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$PdfPath = "J:\Imaging Edge Mobile\Imaging Edge Mobile_paddleocr_uvdoc.pdf",
    [string]$OutputRoot = "",
    [int]$SampleSize = 1,
    [int]$Dpi = 160,
    [string[]]$Engines = @(),
    [string]$BootstrapPython = "py",
    [string]$BootstrapVersion = "3.11",
    [switch]$Recreate,
    [switch]$SkipEditableInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    if (-not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
        $scriptDir = Split-Path -Parent $PSCommandPath
        $RepoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
    }
    else {
        $RepoRoot = (Get-Location).Path
    }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "artifacts\ocr_latest_matrix"
}

New-Item -ItemType Directory -Force $OutputRoot | Out-Null

if (!(Test-Path $PdfPath)) {
    throw "PDF fixture not found: $PdfPath"
}

$basePath = $env:Path
$toolPath = "J:\PC\AI\Tesseract;J:\PC\python\Scripts;$basePath"

$repoPaddleCache = Join-Path $RepoRoot ".paddlex_cache"
$repoHfCache = Join-Path $RepoRoot ".hf_cache"
$repoModelscopeCache = Join-Path $RepoRoot ".modelscope_cache"
$repoYoloCfg = Join-Path $RepoRoot ".ultralytics"

New-Item -ItemType Directory -Force $repoPaddleCache, $repoHfCache, $repoModelscopeCache, $repoYoloCfg | Out-Null

$engineMatrix = @(
    @{
        name = "pytesseract"
        deps = @("pytesseract", "pypdf", "pymupdf")
    },
    @{
        name = "ocrmypdf"
        deps = @("ocrmypdf", "pypdf", "img2pdf", "pymupdf")
    },
    @{
        name = "pymupdf"
        deps = @("pymupdf", "pypdf", "pytesseract")
    },
    @{
        name = "paddleocr"
        deps = @("paddleocr", "paddlex", "paddlepaddle")
    },
    @{
        name = "surya"
        deps = @("surya-ocr")
    },
    @{
        name = "mineru"
        deps = @("mineru", "doclayout-yolo", "ultralytics", "ftfy", "dill", "omegaconf")
    }
)

if ($Engines.Count -gt 0) {
    $normalized = $Engines | ForEach-Object { $_.Trim().ToLowerInvariant() }
    $engineMatrix = @($engineMatrix | Where-Object { $normalized -contains $_.name })
    if ($engineMatrix.Count -eq 0) {
        throw "No matching engines found for filter: $($Engines -join ', ')"
    }
}

function Invoke-Logged {
    param(
        [string]$Exe,
        [string[]]$ArgList,
        [string]$LogPath,
        [string]$StepName,
        [switch]$AllowFailure
    )

    "`n=== $StepName ===" | Tee-Object -FilePath $LogPath -Append | Out-Null
    if ($null -eq $ArgList) {
        $ArgList = @()
    }
    "$Exe $($ArgList -join ' ')" | Tee-Object -FilePath $LogPath -Append | Out-Null

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @ArgList 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if (($exitCode -ne 0) -and (-not $AllowFailure)) {
        throw "Step '$StepName' failed with exit code $exitCode."
    }

    return $exitCode
}

$results = @()
$pdfStem = [System.IO.Path]::GetFileNameWithoutExtension($PdfPath)

foreach ($engine in $engineMatrix) {
    $engineName = $engine.name
    $engineDeps = $engine.deps
    $venvPath = Join-Path $RepoRoot ".venv_latest_$engineName"
    $venvPython = Join-Path $venvPath "Scripts\python.exe"

    $engineOutput = Join-Path $OutputRoot $engineName
    New-Item -ItemType Directory -Force $engineOutput | Out-Null
    $logPath = Join-Path $engineOutput "run.log"
    if (Test-Path $logPath) {
        Remove-Item -Force $logPath
    }

    $entry = [ordered]@{
        engine = $engineName
        venv_path = $venvPath
        status = "not_run"
        install_exit_code = $null
        benchmark_exit_code = $null
        elapsed_seconds = $null
        text_chars = $null
        memory_delta_mb = $null
        artifact_path = $null
        report_path = $null
        error = $null
        log_path = $logPath
    }

    try {
        if ($Recreate -and (Test-Path $venvPath)) {
            Remove-Item -Recurse -Force $venvPath
        }

        if (!(Test-Path $venvPython)) {
            if ($BootstrapPython -eq "py" -and -not [string]::IsNullOrWhiteSpace($BootstrapVersion)) {
                $null = Invoke-Logged -Exe $BootstrapPython -ArgList @("-$BootstrapVersion", "-m", "venv", $venvPath) -LogPath $logPath -StepName "Create venv"
            }
            else {
                $null = Invoke-Logged -Exe $BootstrapPython -ArgList @("-m", "venv", $venvPath) -LogPath $logPath -StepName "Create venv"
            }
        }

        if (!(Test-Path $venvPython)) {
            throw "Python interpreter not found in venv: $venvPython"
        }

        $null = Invoke-Logged -Exe $venvPython -ArgList @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -LogPath $logPath -StepName "Install base tooling"

        if (-not $SkipEditableInstall) {
            $null = Invoke-Logged -Exe $venvPython -ArgList @("-m", "pip", "install", "--upgrade", "-e", $RepoRoot) -LogPath $logPath -StepName "Install project editable"
        }

        $null = Invoke-Logged -Exe $venvPython -ArgList (@("-m", "pip", "install", "--upgrade") + $engineDeps) -LogPath $logPath -StepName "Install engine deps"
        $entry.install_exit_code = 0

        $versionPkgs = @(
            "uniscan",
            "pytesseract",
            "ocrmypdf",
            "pymupdf",
            "paddleocr",
            "paddlex",
            "paddlepaddle",
            "surya-ocr",
            "mineru",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "ftfy",
            "dill",
            "omegaconf"
        )
        $versionsPath = Join-Path $engineOutput "versions.txt"
        $freezePath = Join-Path $engineOutput "requirements_freeze.txt"
        $null = Invoke-Logged -Exe $venvPython -ArgList (@("-m", "pip", "show") + $versionPkgs) -LogPath $versionsPath -StepName "Version snapshot" -AllowFailure
        $null = Invoke-Logged -Exe $venvPython -ArgList @("-m", "pip", "freeze") -LogPath $freezePath -StepName "Freeze snapshot" -AllowFailure

        $env:Path = $toolPath
        $env:PADDLE_PDX_CACHE_HOME = $repoPaddleCache
        $env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
        $env:HF_HOME = $repoHfCache
        $env:MODELSCOPE_CACHE = $repoModelscopeCache
        $env:YOLO_CONFIG_DIR = $repoYoloCfg
        $env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"

        $benchArgs = @(
            "-m", "uniscan",
            "benchmark-ocr",
            "--pdf", $PdfPath,
            "--output", $engineOutput,
            "--sample-size", "$SampleSize",
            "--dpi", "$Dpi",
            "--engines", $engineName,
            "--strict"
        )
        $benchExit = Invoke-Logged -Exe $venvPython -ArgList $benchArgs -LogPath $logPath -StepName "Run benchmark" -AllowFailure
        $entry.benchmark_exit_code = $benchExit

        $reportPath = Join-Path $engineOutput "${pdfStem}_ocr_benchmark.json"
        $entry.report_path = $reportPath
        if (Test-Path $reportPath) {
            $report = Get-Content $reportPath -Raw | ConvertFrom-Json
            $engineResult = $report.results | Where-Object { $_.engine -eq $engineName } | Select-Object -First 1
            if ($null -ne $engineResult) {
                $entry.status = $engineResult.status
                $entry.elapsed_seconds = $engineResult.elapsed_seconds
                $entry.text_chars = $engineResult.text_chars
                $entry.memory_delta_mb = $engineResult.memory_delta_mb
                $entry.artifact_path = $engineResult.artifact_path
                $entry.error = $engineResult.error
            }
            else {
                $entry.status = "error"
                $entry.error = "No engine result entry found in report."
            }
        }
        else {
            $entry.status = "error"
            $entry.error = "Benchmark report was not created."
        }
    }
    catch {
        $entry.status = "error"
        $entry.error = $_.Exception.Message
        if ($null -eq $entry.install_exit_code) {
            $entry.install_exit_code = 1
        }
        if ($null -eq $entry.benchmark_exit_code) {
            $entry.benchmark_exit_code = 1
        }
        $_ | Out-String | Tee-Object -FilePath $logPath -Append | Out-Null
    }
    finally {
        $results += [pscustomobject]$entry
        $env:Path = $basePath
    }
}

$summaryJson = Join-Path $OutputRoot "summary.json"
$summaryCsv = Join-Path $OutputRoot "summary.csv"

$results | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryJson -Encoding UTF8
$results | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8

$results |
    Select-Object engine, status, elapsed_seconds, text_chars, memory_delta_mb, benchmark_exit_code |
    Format-Table -AutoSize

Write-Host ""
Write-Host "Summary JSON: $summaryJson"
Write-Host "Summary CSV : $summaryCsv"
