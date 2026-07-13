# Release checklist

- [ ] Update the single version source in `src/uniscan/__init__.py` and cut the matching dated
  `CHANGELOG.md` section; run `scripts/verify_release_version.py --tag v<version>`.
- [ ] Run `ruff check .` and `ruff format --check .`.
- [ ] Run tests with total coverage >=60% and non-GUI coverage >=80%.
- [ ] Run the default OpenCV detector and geometry quality baselines with no regressions.
- [ ] Build the Windows artifact and verify its frozen diagnostics.
- [ ] Complete `docs/manual_smoke_checklist.md` on a clean Windows x64 machine and real camera.
- [ ] Verify ZIP checksum and inspect archive contents for source caches, secrets, or test data.
- [ ] Review the generated compatibility/frozen-payload inventory and all dependency, asset, and
  Python/Tcl/Tk notices under `THIRD_PARTY_LICENSES`.
- [ ] For a public release, sign `uniscan.exe` and verify its Authenticode chain/timestamp.
- [ ] Push annotated tag `v<version>` and verify CI creates a non-public draft with ZIP plus SHA-256.
- [ ] Verify no Office Lens model weights or unrelated platform-native payloads are in the ZIP.
- [ ] Before claiming the public source repository is model-free, complete the separately approved
  Git history purge described in `office_lens_onnx.md` and audit every retained tag/branch.
- [ ] Publish the draft only after all preceding signing/manual/license gates have recorded evidence.
- [ ] Download the published artifact, repeat install/CLI/GUI/uninstall smoke, and record evidence.
