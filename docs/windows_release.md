# Windows release and clean-machine installation

UniScan ships as a portable, versioned x64 ZIP. It does not require a development Python
installation and does not register services or an uninstaller.

## Build

From a 64-bit Windows PowerShell with the development `.venv` present:

```powershell
.\scripts\build_windows.ps1
```

The script verifies that `.venv` is a `win-amd64` interpreter, installs the development and release
extras, runs lint/tests/quality baselines, builds `dist\uniscan`, generates the third-party license
compatibility/frozen-payload inventory with Python/Tcl/Tk notices, and runs frozen
CLI/PDF/GUI-runtime smoke checks before writing:

- tagged CI: `artifacts\uniscan-<version>-windows-x64.zip` plus `.sha256`;
- local/manual build: `artifacts\uniscan-<version>-dev-<commit>[-dirty]-windows-x64.zip` plus
  `.sha256`, so an unreleased build cannot overwrite a versioned release artifact.

The package contains only Windows x64 Tk drag-and-drop binaries; optional Office Lens ONNX weights
are not redistributed. The ZIP also carries the UniScan license, a portable-specific readme,
changelog, required dependency and bundled-runtime notices, and Windows/manual-smoke documentation.

A Git tag must exactly equal `v<source-version>`, and the matching changelog section must be cut
before CI accepts it. A tag creates a **draft prerelease**, not a public release. Signing, clean
machine/manual-camera evidence, checksum verification, and artifact review must be completed before
a maintainer publishes that draft.

## Install and run on a clean machine

1. Verify the ZIP against the adjacent SHA-256 file.
2. Extract it to a user-writable directory such as `%LOCALAPPDATA%\Programs\UniScan`.
3. Run `uniscan\uniscan.exe doctor`.
4. Run `uniscan\uniscan.exe` for the GUI or use `uniscan\uniscan.exe convert ...` for CLI work.

Windows may show a SmartScreen warning because a locally built preview artifact is unsigned. Treat
it as internal until every gate in the release checklist is complete. CI deliberately cannot
publish a tagged build automatically.

## Uninstall

Close UniScan and delete the extracted `uniscan` directory. To remove recoverable GUI sessions
as well, delete `%LOCALAPPDATA%\UniScan`. No other system state is installed.

## Signing decision

Internal/test artifacts may remain unsigned. Authenticode signing is required before public
distribution because an unsigned camera application produces avoidable SmartScreen friction and
offers no publisher identity. Obtain an organization-backed code-signing certificate, protect the
key in an approved CI signing service, sign `uniscan.exe`, and verify its chain and timestamp before
publishing the release draft. The release job must not receive an exportable private key.
