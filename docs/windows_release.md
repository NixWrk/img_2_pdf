# Windows release and clean-machine installation

UniScan ships as a portable, versioned x64 ZIP. It does not require a development Python
installation and does not register services or an uninstaller.

## Build

From a 64-bit Windows PowerShell with the development `.venv` present:

```powershell
.\scripts\build_windows.ps1
```

The script installs the release extra, runs lint/tests, builds `dist\uniscan`, runs frozen
`--version` and `doctor --json` smoke checks, then writes:

- `artifacts\uniscan-<version>-windows-x64.zip`
- `artifacts\uniscan-<version>-windows-x64.zip.sha256`

The package contains the ONNX models and Tk drag-and-drop runtime. Git tag `v*` builds the same
artifact in GitHub Actions and publishes it with its checksum. The ZIP also carries the UniScan
license, readme, changelog, and Windows/manual-smoke documentation.

## Install and run on a clean machine

1. Verify the ZIP against the adjacent SHA-256 file.
2. Extract it to a user-writable directory such as `%LOCALAPPDATA%\Programs\UniScan`.
3. Run `uniscan\uniscan.exe doctor`.
4. Run `uniscan\uniscan.exe` for the GUI or use `uniscan\uniscan.exe convert ...` for CLI work.

Windows may show a SmartScreen warning because the current preview artifact is unsigned.
Treat this artifact as an internal preview until the signing and dependency-license gates in the
release checklist are completed.

## Uninstall

Close UniScan and delete the extracted `uniscan` directory. To remove recoverable GUI sessions
as well, delete `%LOCALAPPDATA%\UniScan`. No other system state is installed.

## Signing decision

Current decision: internal/test artifacts remain unsigned. Authenticode signing is required
before public distribution because an unsigned camera application produces avoidable SmartScreen
friction and offers no publisher identity. Before the first public release, obtain an
organization-backed code-signing certificate, protect the signing key in an approved CI signing
service, sign `uniscan.exe`, and verify the signature before creating the ZIP. The release job
must not receive an exportable private key.
