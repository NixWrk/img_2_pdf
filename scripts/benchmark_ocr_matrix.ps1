[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$PdfPath = "",
    [string]$OutputRoot = "",
    [int]$SampleSize = 1,
    [string]$Pages = "",
    [int]$Dpi = 160,
    [string]$Lang = "",
    [string[]]$Engines = @(),
    [string]$BootstrapPython = "py",
    [string]$BootstrapVersion = "3.11",
    [string]$TesseractPath = "",
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

# Locate a test PDF if none provided
if ([string]::IsNullOrWhiteSpace($PdfPath)) {
    # Try common fixture locations
    $candidates = @(
        (Join-Path $RepoRoot "tests\fixtures\sample.pdf"),
        (Join-Path $RepoRoot "fixtures\sample.pdf")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $PdfPath = $c; break }
    }
    if ([string]::IsNullOrWhiteSpace($PdfPath)) {
        throw "No -PdfPath provided and no fixture PDF found. Pass -PdfPath <path-to-pdf>."
    }
}

if (!(Test-Path $PdfPath)) {
    throw "PDF fixture not found: $PdfPath"
}

# Build tool PATH: auto-detect Tesseract, keep system PATH
$basePath = $env:Path
$extraPaths = @()

# Auto-detect Tesseract
if (-not [string]::IsNullOrWhiteSpace($TesseractPath)) {
    $extraPaths += $TesseractPath
}
else {
    $tesseractExe = Get-Command tesseract -ErrorAction SilentlyContinue
    if ($null -ne $tesseractExe) {
        $extraPaths += (Split-Path $tesseractExe.Source)
    }
    else {
        # Common Tesseract locations on Windows
        $defaultPaths = @(
            "C:\Program Files\Tesseract-OCR",
            "C:\Program Files (x86)\Tesseract-OCR",
            "J:\PC\AI\Tesseract"
        )
        foreach ($dp in $defaultPaths) {
            if (Test-Path (Join-Path $dp "tesseract.exe")) {
                $extraPaths += $dp
                break
            }
        }
    }
}
$toolPath = (($extraPaths + @($basePath)) -join ";")

# ---------------------------------------------------------------------------
# GPU detection: query nvidia-smi for compute capability
# ---------------------------------------------------------------------------
$gpuComputeCap = 0.0
$gpuName = "none"
try {
    $nvOut = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($nvOut)) {
        $parts = $nvOut.Split(",")
        $gpuName = $parts[0].Trim()
        $gpuComputeCap = [double]($parts[1].Trim())
    }
}
catch { }

$hasCuda = $gpuComputeCap -gt 0
$paddleGpuOk = $gpuComputeCap -ge 7.5   # PaddlePaddle requires > 7.5
$torchGpuOk  = $gpuComputeCap -ge 3.5   # PyTorch cu121 supports 3.5+

Write-Host ""
Write-Host "GPU: $gpuName (compute $gpuComputeCap)"
Write-Host "  torch GPU  : $(if ($torchGpuOk) {'YES'} else {'NO (need >= 3.5)'})"
Write-Host "  paddle GPU : $(if ($paddleGpuOk) {'YES'} else {'NO (need >= 7.5)'})"
Write-Host ""

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
        deps = @(
            "paddleocr==3.4.0",
            "paddlex==3.4.2",
            "requests"
        )
        # PaddlePaddle GPU requires compute capability > 7.5 (GTX 1070 = 6.1, won't work).
        # Install GPU via paddlepaddle.org.cn index; falls back to CPU if GPU fails.
        gpu_paddle = $true
    },
    @{
        name = "surya"
        deps = @(
            "surya-ocr",
            "requests",
            "transformers==4.57.1",
            "tokenizers==0.22.1",
            "huggingface-hub==0.34.4"
        )
        # torch GPU installed separately via --index-url
        gpu_torch = $true
    },
    @{
        name = "mineru"
        deps = @(
            "mineru",
            "doclayout-yolo",
            "ultralytics",
            "ftfy",
            "dill",
            "omegaconf",
            "shapely",
            "pyclipper",
            "requests",
            "transformers==4.57.1",
            "tokenizers==0.22.1",
            "huggingface-hub==0.34.4"
        )
        # torch GPU installed separately via --index-url
        gpu_torch = $true
    },
    @{
        name = "chandra"
        deps = @(
            "chandra-ocr[hf]",
            "requests"
        )
        # torch GPU installed separately via --index-url
        gpu_torch = $true
    },
    @{
        name = "olmocr"
        deps = @(
            "olmocr",
            "requests"
        )
    }
)

if ($Engines.Count -gt 0) {
    $normalized = @()
    foreach ($engineArg in $Engines) {
        if ([string]::IsNullOrWhiteSpace($engineArg)) {
            continue
        }
        foreach ($part in ($engineArg -split ",")) {
            $name = $part.Trim().ToLowerInvariant()
            if (-not [string]::IsNullOrWhiteSpace($name)) {
                $normalized += $name
            }
        }
    }
    $normalized = @($normalized | Select-Object -Unique)
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

    $stderrPath = Join-Path $env:TEMP ("uniscan_ocr_matrix_stderr_{0}.log" -f ([guid]::NewGuid().ToString("N")))
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $hasNativeEap = $null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue)
    if ($hasNativeEap) {
        $previousNativeEap = $global:PSNativeCommandUseErrorActionPreference
        $global:PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        # Keep native argument semantics (paths with spaces remain intact),
        # while redirecting stderr away from PowerShell error records.
        & $Exe @ArgList 2> $stderrPath | Tee-Object -FilePath $LogPath -Append | Out-Host
        if (Test-Path $stderrPath) {
            Get-Content $stderrPath | Tee-Object -FilePath $LogPath -Append | Out-Host
        }
        $exitCode = [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativeEap) {
            $global:PSNativeCommandUseErrorActionPreference = $previousNativeEap
        }
        if (Test-Path $stderrPath) {
            Remove-Item -Force $stderrPath
        }
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
        try {
            Remove-Item -Force $logPath -ErrorAction Stop
        }
        catch {
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $logPath = Join-Path $engineOutput ("run_{0}.log" -f $timestamp)
        }
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
            try {
                Remove-Item -Recurse -Force $venvPath -ErrorAction Stop
            }
            catch {
                $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
                $venvPath = Join-Path $RepoRoot (".venv_latest_{0}_{1}" -f $engineName, $timestamp)
                $venvPython = Join-Path $venvPath "Scripts\python.exe"
                $entry.venv_path = $venvPath
                ("[warn] Unable to remove existing venv. Using fallback venv path: {0}" -f $venvPath) | Tee-Object -FilePath $logPath -Append | Out-Host
            }
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

        $null = Invoke-Logged -Exe $venvPython -ArgList @("-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel") -LogPath $logPath -StepName "Install base tooling"

        if (-not $SkipEditableInstall) {
            $null = Invoke-Logged -Exe $venvPython -ArgList @("-m", "pip", "install", "--upgrade", "-e", $RepoRoot) -LogPath $logPath -StepName "Install project editable"
        }

        # Install PyTorch with CUDA support for GPU-enabled engines.
        # Surya is handled separately because current surya-ocr pins may upgrade
        # torch and break preinstalled torchvision/torchaudio combos.
        if ($engine["gpu_torch"] -eq $true) {
            if ($engineName -ne "surya") {
                if ($torchGpuOk) {
                    $null = Invoke-Logged -Exe $venvPython -ArgList @(
                        "-m", "pip", "install", "--upgrade",
                        "torch", "torchvision", "torchaudio",
                        "--index-url", "https://download.pytorch.org/whl/cu121"
                    ) -LogPath $logPath -StepName "Install PyTorch+CUDA (cu121)"
                }
                else {
                    "GPU compute $gpuComputeCap < 3.5; installing PyTorch CPU." | Tee-Object -FilePath $logPath -Append | Out-Host
                    $null = Invoke-Logged -Exe $venvPython -ArgList @(
                        "-m", "pip", "install", "--upgrade",
                        "torch", "torchvision", "torchaudio"
                    ) -LogPath $logPath -StepName "Install PyTorch CPU (fallback)"
                }
            }
            else {
                "Skipping preinstall torch stack for surya; engine deps will resolve torch first." | Tee-Object -FilePath $logPath -Append | Out-Host
            }
        }

        # Install PaddlePaddle: GPU from official CN index if supported, else CPU
        if ($engine["gpu_paddle"] -eq $true) {
            $paddleInstalled = $false
            if ($paddleGpuOk) {
                try {
                    $null = Invoke-Logged -Exe $venvPython -ArgList @(
                        "-m", "pip", "install",
                        "paddlepaddle-gpu==3.2.2",
                        "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
                    ) -LogPath $logPath -StepName "Install PaddlePaddle-GPU (cu118)"
                    $paddleInstalled = $true
                }
                catch {
                    "PaddlePaddle-GPU install failed, falling back to CPU." | Tee-Object -FilePath $logPath -Append | Out-Host
                }
            }
            else {
                "GPU compute $gpuComputeCap < 7.5; skipping PaddlePaddle-GPU." | Tee-Object -FilePath $logPath -Append | Out-Host
            }
            if (-not $paddleInstalled) {
                $null = Invoke-Logged -Exe $venvPython -ArgList @(
                    "-m", "pip", "install", "--upgrade",
                    "paddlepaddle==3.1.1"
                ) -LogPath $logPath -StepName "Install PaddlePaddle-CPU (fallback)"
            }
        }

        $null = Invoke-Logged -Exe $venvPython -ArgList (@("-m", "pip", "install", "--upgrade") + $engineDeps) -LogPath $logPath -StepName "Install engine deps"

        if ($engineName -eq "surya") {
            # Ensure torchvision/torchaudio ABI matches the torch version
            # resolved by surya-ocr dependency constraints.
            $null = Invoke-Logged -Exe $venvPython -ArgList @(
                "-m", "pip", "install", "--upgrade",
                "torchvision", "torchaudio"
            ) -LogPath $logPath -StepName "Align torchvision/torchaudio with torch"
        }
        $entry.install_exit_code = 0

        $versionPkgs = @("uniscan", "pymupdf")
        switch ($engineName) {
            "pytesseract" {
                $versionPkgs += @("pytesseract", "pypdf")
                break
            }
            "ocrmypdf" {
                $versionPkgs += @("ocrmypdf", "pypdf", "img2pdf")
                break
            }
            "pymupdf" {
                $versionPkgs += @("pypdf", "pytesseract")
                break
            }
            "paddleocr" {
                $versionPkgs += @("paddleocr", "paddlex", "paddlepaddle", "requests")
                break
            }
            "surya" {
                $versionPkgs += @("surya-ocr", "requests", "transformers", "tokenizers", "huggingface-hub")
                break
            }
            "mineru" {
                $versionPkgs += @(
                    "mineru",
                    "doclayout-yolo",
                    "ultralytics",
                    "ftfy",
                    "dill",
                    "omegaconf",
                    "shapely",
                    "pyclipper",
                    "requests",
                    "transformers",
                    "tokenizers",
                    "huggingface-hub"
                )
                break
            }
            "chandra" {
                $versionPkgs += @("chandra-ocr", "requests")
                break
            }
            "olmocr" {
                $versionPkgs += @("olmocr", "requests")
                break
            }
        }
        $versionPkgs = @($versionPkgs | Select-Object -Unique)
        $versionsPath = Join-Path $engineOutput "versions.txt"
        $freezePath = Join-Path $engineOutput "requirements_freeze.txt"
        $null = Invoke-Logged -Exe $venvPython -ArgList (@("-m", "pip", "show") + $versionPkgs) -LogPath $versionsPath -StepName "Version snapshot" -AllowFailure
        $null = Invoke-Logged -Exe $venvPython -ArgList @("-m", "pip", "freeze") -LogPath $freezePath -StepName "Freeze snapshot" -AllowFailure

        # Prepend the venv Scripts dir so CLI tools (chandra, surya_ocr, …)
        # installed into the venv are discoverable via shutil.which().
        $venvScripts = Join-Path $venvPath "Scripts"
        $env:Path = "$venvScripts;$toolPath"
        $env:PADDLE_PDX_CACHE_HOME = $repoPaddleCache
        $env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
        $env:HF_HOME = $repoHfCache
        $env:MODELSCOPE_CACHE = $repoModelscopeCache
        $env:YOLO_CONFIG_DIR = $repoYoloCfg
        $env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
        if ($engineName -eq "paddleocr") {
            # Reduce Paddle CPU backend incompatibilities in latest stacks.
            $env:FLAGS_enable_pir_api = "0"
            $env:FLAGS_use_mkldnn = "0"
            $env:PADDLE_PDX_USE_PIR_TRT = "false"
            $env:FLAGS_enable_new_ir_in_executor = "0"
            $env:FLAGS_enable_pir_in_executor = "0"
        }
        if ($engineName -eq "olmocr") {
            # On Windows, local vLLM startup is fragile; use dockerized backend.
            $env:UNISCAN_OLMOCR_BACKEND = "docker"
            $env:UNISCAN_OLMOCR_DOCKER_IMAGE = "chatdoc/ocrflux:latest"
            $env:UNISCAN_OLMOCR_DOCKER_WORKERS = "1"
            $env:UNISCAN_OLMOCR_DOCKER_CACHE = (Join-Path $RepoRoot ".hf_cache_ocrflux")
        }

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
        if (-not [string]::IsNullOrWhiteSpace($Pages)) {
            $benchArgs += @("--pages", $Pages)
        }
        if (-not [string]::IsNullOrWhiteSpace($Lang)) {
            $benchArgs += @("--lang", $Lang)
        }
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
        foreach ($name in @(
            "UNISCAN_OLMOCR_BACKEND",
            "UNISCAN_OLMOCR_DOCKER_IMAGE",
            "UNISCAN_OLMOCR_DOCKER_WORKERS",
            "UNISCAN_OLMOCR_DOCKER_CACHE",
            "UNISCAN_OLMOCR_DOCKER_GPU",
            "UNISCAN_OLMOCR_DOCKER_MODEL",
            "UNISCAN_OLMOCR_DOCKER_GPU_MEM_UTIL"
        )) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        }
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
