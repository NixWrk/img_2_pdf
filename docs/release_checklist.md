# Release checklist

- [ ] Update `src/uniscan/__init__.py`, `pyproject.toml`, and `CHANGELOG.md` to the same version.
- [ ] Run `ruff check .` and `ruff format --check .`.
- [ ] Run tests with total coverage >=60% and non-GUI coverage >=80%.
- [ ] Run the Office Lens quality baseline with no regressions.
- [ ] Build the Windows artifact and verify its frozen diagnostics.
- [ ] Complete `docs/manual_smoke_checklist.md` on a clean Windows x64 machine and real camera.
- [ ] Verify ZIP checksum and inspect archive contents for source caches, secrets, or test data.
- [ ] Generate/review the third-party dependency license inventory and include required notices.
- [ ] For a public release, sign `uniscan.exe` and verify its Authenticode chain/timestamp.
- [ ] Push annotated tag `v<version>` and verify the GitHub release contains ZIP plus SHA-256.
- [ ] Download the published artifact, repeat install/CLI/GUI/uninstall smoke, and record evidence.
