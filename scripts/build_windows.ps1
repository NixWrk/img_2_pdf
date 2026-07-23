[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") {
    throw "The release artifact must be built on Windows."
}
if (-not [Environment]::Is64BitProcess) {
    throw "The Windows release must be built by 64-bit Python/PowerShell."
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv. Create it with: py -3.13-64 -m venv .venv"
}

Push-Location $Root
try {
    & $Python scripts\verify_windows_runtime.py
    if ($LASTEXITCODE -ne 0) { throw "Python is not a Windows x64 build." }
    & $Python scripts\verify_release_version.py
    if ($LASTEXITCODE -ne 0) { throw "Release metadata is inconsistent." }

    if (-not $SkipInstall) {
        & $Python -m pip install -e ".[dev,release]"
        if ($LASTEXITCODE -ne 0) { throw "Release dependency installation failed." }
    }

    # DocShadow is deliberately not a Git blob. Fetch the immutable v1.0.0
    # release asset and publish it only after its pinned size and SHA-256 match.
    & $Python scripts\download_model_assets.py --asset docshadow_sd7k --target src\uniscan\models
    if ($LASTEXITCODE -ne 0) { throw "Pinned DocShadow asset download failed." }

    if (-not $SkipTests) {
        & $Python -m pip check
        if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }
        & $Python -m ruff check .
        if ($LASTEXITCODE -ne 0) { throw "Ruff failed." }
        & $Python -m ruff format --check .
        if ($LASTEXITCODE -ne 0) { throw "Formatting check failed." }
        & $Python -m pytest --cov=uniscan --cov-report=term --cov-fail-under=60 -q
        if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
        & $Python -m coverage report --omit=src/uniscan/ui/app.py --fail-under=80
        if ($LASTEXITCODE -ne 0) { throw "Non-GUI coverage check failed." }
        $QualityReport = Join-Path $env:TEMP "uniscan-quality-$PID.json"
        & $Python -m uniscan benchmark-quality --input benchmarks\corpus_v1 --output $QualityReport --baseline benchmarks\corpus_v1\baseline.json
        if ($LASTEXITCODE -ne 0) { throw "Document detection quality baseline failed." }
        Remove-Item -LiteralPath $QualityReport -ErrorAction SilentlyContinue
        $GeometryReport = Join-Path $env:TEMP "uniscan-geometry-$PID.json"
        & $Python -m uniscan benchmark-geometry --input benchmarks\geometry_v1 --output $GeometryReport --baseline benchmarks\geometry_v1\baseline.json
        if ($LASTEXITCODE -ne 0) { throw "Geometry quality baseline failed." }
        Remove-Item -LiteralPath $GeometryReport -ErrorAction SilentlyContinue
    }

    & $Python -m PyInstaller --noconfirm --clean uniscan.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

    Copy-Item -LiteralPath "LICENSE" -Destination "dist\uniscan\LICENSE.txt" -Force
    Copy-Item -LiteralPath "docs\portable_readme.md" -Destination "dist\uniscan\README.md" -Force
    Copy-Item -LiteralPath "CHANGELOG.md" -Destination "dist\uniscan\CHANGELOG.md" -Force
    New-Item -ItemType Directory -Path "dist\uniscan\docs" -Force | Out-Null
    Copy-Item -LiteralPath "docs\windows_release.md" -Destination "dist\uniscan\docs\windows_release.md" -Force
    Copy-Item -LiteralPath "docs\manual_smoke_checklist.md" -Destination "dist\uniscan\docs\manual_smoke_checklist.md" -Force
    Copy-Item -LiteralPath "docs\document_geometry.md" -Destination "dist\uniscan\docs\document_geometry.md" -Force
    & $Python scripts\collect_third_party_licenses.py `
        dist\uniscan\THIRD_PARTY_LICENSES `
        --portable-root dist\uniscan `
        --pyinstaller-toc build\uniscan\PYZ-00.toc `
        --pyinstaller-toc build\uniscan\COLLECT-00.toc
    if ($LASTEXITCODE -ne 0) { throw "Third-party license compatibility audit failed." }
    & $Python scripts\audit_portable_contents.py dist\uniscan
    if ($LASTEXITCODE -ne 0) { throw "Portable content audit failed." }

    & $Python scripts\smoke_windows_artifact.py dist\uniscan
    if ($LASTEXITCODE -ne 0) { throw "Packaged artifact smoke test failed." }

    $Version = (& $Python -c "from uniscan import __version__; print(__version__)").Trim()
    $ArtifactVersion = $Version
    if ($env:GITHUB_REF_TYPE -ne "tag") {
        $Commit = (& git -c core.excludesFile= rev-parse --short=12 HEAD 2>$null)
        if ($LASTEXITCODE -ne 0 -or -not $Commit) {
            $Commit = Get-Date -Format "yyyyMMddHHmmss"
        }
        $ArtifactVersion = "$Version-dev-$($Commit.Trim())"
        $Dirty = (& git -c core.excludesFile= status --porcelain 2>$null)
        if ($LASTEXITCODE -eq 0 -and $Dirty) {
            $ArtifactVersion = "$ArtifactVersion-dirty"
        }
    }
    $Artifacts = Join-Path $Root "artifacts"
    New-Item -ItemType Directory -Path $Artifacts -Force | Out-Null
    $Archive = Join-Path $Artifacts "uniscan-$ArtifactVersion-windows-x64.zip"
    $Checksum = "$Archive.sha256"
    if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive }
    if (Test-Path -LiteralPath $Checksum) { Remove-Item -LiteralPath $Checksum }
    Compress-Archive -Path "dist\uniscan" -DestinationPath $Archive -CompressionLevel Optimal
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
    "$Hash  $(Split-Path -Leaf $Archive)" | Set-Content -LiteralPath $Checksum -Encoding ascii
    Write-Host "Built $Archive"
    Write-Host "SHA256 $Hash"
}
finally {
    Pop-Location
}
