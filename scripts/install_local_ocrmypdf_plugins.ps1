[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [switch]$RefreshNixCopies
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

$pythonPath = $PythonExe
if (-not [System.IO.Path]::IsPathRooted($pythonPath)) {
    $pythonPath = Join-Path $RepoRoot $pythonPath
}
if (!(Test-Path $pythonPath)) {
    throw "Python executable not found: $pythonPath"
}

$pluginRoot = Join-Path $RepoRoot "OCRmypdf_plugins"
if (!(Test-Path $pluginRoot)) {
    throw "Plugin folder not found: $pluginRoot"
}

$nestedPluginRoot = Join-Path $pluginRoot "OCRmypdf_plugins"
$sourceRoots = @($pluginRoot)
if (Test-Path $nestedPluginRoot) {
    # Some archives unpack as OCRmypdf_plugins\OCRmypdf_plugins\<repo>.
    $sourceRoots = @($nestedPluginRoot, $pluginRoot)
}

Write-Host "Plugin source roots:"
$sourceRoots | ForEach-Object { Write-Host " - $_" }

$pluginSpecs = @(
    @{
        name = "ocrmypdf-surya-main"
    },
    @{
        name = "OCRmyPDF-PaddleOCR-main"
    },
    @{
        name = "ocrmypdf-paddleocr-master"
    },
    @{
        name = "OCRmyPDF-EasyOCR-main"
    },
    @{
        name = "ocrmypdf-doctr-master"
    },
    @{
        name = "OCRmyPDF-AppleOCR-main"
    },
    @{
        # Bridge sketch from local experiments; can be copied for reference but
        # has no package metadata, so editable pip install is skipped.
        name = "Ocrmypdf+surya"
    }
)

function Resolve-PluginSourcePath {
    param(
        [string]$Name,
        [string[]]$Roots
    )
    foreach ($root in $Roots) {
        $candidate = Join-Path $root $Name
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

$installed = @()
$skipped = @()
foreach ($spec in $pluginSpecs) {
    $name = [string]$spec.name
    $sourcePath = Resolve-PluginSourcePath -Name $name -Roots $sourceRoots
    if ([string]::IsNullOrWhiteSpace($sourcePath)) {
        continue
    }

    $nixPath = Join-Path $pluginRoot ("{0}_NIX" -f $name)
    if ($RefreshNixCopies -and (Test-Path $nixPath)) {
        Write-Host "Refreshing NIX copy: $nixPath"
        Remove-Item -Recurse -Force $nixPath
    }
    if (!(Test-Path $nixPath)) {
        Write-Host "Creating NIX copy: $sourcePath -> $nixPath"
        Copy-Item -Path $sourcePath -Destination $nixPath -Recurse -Force
    }
    else {
        Write-Host "Using existing NIX copy: $nixPath"
    }

    $hasPyProject = Test-Path (Join-Path $nixPath "pyproject.toml")
    $hasSetupPy = Test-Path (Join-Path $nixPath "setup.py")
    if (-not ($hasPyProject -or $hasSetupPy)) {
        $reason = "no pyproject.toml/setup.py (reference sketch only)"
        Write-Host "Skipping pip install for $name`: $reason" -ForegroundColor Yellow
        $skipped += ("{0}_NIX ({1})" -f $name, $reason)
        continue
    }

    Write-Host "Installing plugin from NIX copy $nixPath"
    & $pythonPath -m pip install -e $nixPath
    if ($LASTEXITCODE -ne 0) {
        $reason = "pip install failed"
        Write-Host "Skipping $name`: $reason" -ForegroundColor Yellow
        $skipped += ("{0}_NIX ({1})" -f $name, $reason)
        continue
    }
    $installed += ("{0}_NIX" -f $name)
}

if (($installed.Count -eq 0) -and ($skipped.Count -eq 0)) {
    Write-Host "No known plugin repos found under $pluginRoot"
    exit 0
}

Write-Host ""
if ($installed.Count -gt 0) {
    Write-Host "Installed plugin repos:"
    $installed | ForEach-Object { Write-Host " - $_" }
}
if ($skipped.Count -gt 0) {
    Write-Host "Copied but not installed:"
    $skipped | ForEach-Object { Write-Host " - $_" }
}
