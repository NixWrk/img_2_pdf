import multiprocessing
import os
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest

from uniscan.export import (
    export_image_paths_as_files,
    export_image_paths_as_pdf,
    export_pages_as_files,
    export_pages_as_pdf,
)


def _hold_pdf_export_lock(output: str, ready, release) -> None:
    from uniscan.export import exporters

    with exporters._pdf_export_lock(Path(output)):
        ready.set()
        if not release.wait(20):
            raise TimeoutError("Timed out waiting to release PDF export lock.")


def _pages() -> list[np.ndarray]:
    a = np.zeros((30, 50, 3), dtype=np.uint8)
    b = np.full((30, 50, 3), 220, dtype=np.uint8)
    return [a, b]


def _pdf_page_size(path) -> tuple[float, float]:
    document = pdfium.PdfDocument(str(path))
    try:
        page = document[0]
        try:
            return page.get_size()
        finally:
            page.close()
    finally:
        document.close()


def test_export_pages_as_files(tmp_path) -> None:
    out = export_pages_as_files(_pages(), output_dir=tmp_path, ext="png", base_name="p")
    assert len(out) == 2
    assert out[0].exists()
    assert out[1].exists()


def test_export_pages_as_pdf(tmp_path) -> None:
    out_pdf = tmp_path / "out.pdf"
    result = export_pages_as_pdf(_pages(), out_pdf=out_pdf, dpi=200)
    assert result.exists()
    assert result.suffix.lower() == ".pdf"
    assert result.stat().st_size > 0


def test_export_image_paths_variants(tmp_path) -> None:
    source_dir = tmp_path / "src"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = export_pages_as_files(_pages(), output_dir=source_dir, ext="png", base_name="src")

    out_pdf = export_image_paths_as_pdf(source, out_pdf=tmp_path / "paths.pdf", dpi=180)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0

    out_files = export_image_paths_as_files(
        source, output_dir=tmp_path / "jpgs", ext="jpg", base_name="e"
    )
    assert len(out_files) == 2
    assert out_files[0].suffix.lower() == ".jpg"


@pytest.mark.parametrize(
    ("dpi", "expected_width_pt", "expected_height_pt"),
    ((72, 300.0, 600.0), (300, 72.0, 144.0)),
)
def test_pdf_export_uses_requested_physical_dpi(
    tmp_path, dpi: int, expected_width_pt: float, expected_height_pt: float
) -> None:
    page = np.full((600, 300, 3), 255, dtype=np.uint8)
    output = export_pages_as_pdf([page], out_pdf=tmp_path / f"dpi-{dpi}.pdf", dpi=dpi)

    width, height = _pdf_page_size(output)
    assert width == pytest.approx(expected_width_pt, abs=0.1)
    assert height == pytest.approx(expected_height_pt, abs=0.1)


def test_pdf_export_cancellation_before_publish_preserves_existing_file(tmp_path) -> None:
    output = tmp_path / "output.pdf"
    output.write_bytes(b"previous")
    checks = 0

    def cancel_after_conversion() -> bool:
        nonlocal checks
        checks += 1
        return checks == 4

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        export_pages_as_pdf(
            _pages()[:1],
            out_pdf=output,
            dpi=300,
            cancel_cb=cancel_after_conversion,
        )

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".output.pdf.stage-*"))


def test_pdf_export_fails_closed_while_another_process_holds_lock(tmp_path) -> None:
    output = tmp_path / "shared.pdf"
    output.write_bytes(b"previous")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_pdf_export_lock,
        args=(str(output), ready, release),
    )
    process.start()
    try:
        assert ready.wait(10), "Child process did not acquire the PDF export lock."
        with pytest.raises(RuntimeError, match="Another UniScan process"):
            export_pages_as_pdf(_pages()[:1], out_pdf=output)
    finally:
        release.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert process.exitcode == 0
    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".shared.pdf.stage-*"))


def test_pdf_export_rejects_hardlinked_lock_without_touching_referent(tmp_path) -> None:
    from uniscan.export import exporters

    output = tmp_path / "shared.pdf"
    output.write_bytes(b"previous-pdf")
    referent = tmp_path / "personal.bin"
    referent.write_bytes(b"personal-lock-referent")
    lock_path = exporters._file_export_lock_path(output)
    try:
        os.link(referent, lock_path)
    except OSError as exc:
        pytest.skip(f"Hard links are not supported on this filesystem: {exc}")

    with pytest.raises(ValueError, match="Invalid file export lock path"):
        export_pages_as_pdf(_pages()[:1], out_pdf=output)

    assert referent.read_bytes() == b"personal-lock-referent"
    assert lock_path.read_bytes() == b"personal-lock-referent"
    assert output.read_bytes() == b"previous-pdf"


def test_pdf_export_rejects_symlink_target_without_touching_referent(tmp_path) -> None:
    referent = tmp_path / "personal.pdf"
    referent.write_bytes(b"personal")
    output = tmp_path / "output.pdf"
    try:
        output.symlink_to(referent)
    except OSError as exc:  # Windows may require Developer Mode for symlinks.
        pytest.skip(f"Cannot create a test symlink: {exc}")

    with pytest.raises(ValueError, match="link or junction"):
        export_pages_as_pdf(_pages()[:1], out_pdf=output)

    assert output.is_symlink()
    assert referent.read_bytes() == b"personal"


def test_pdf_export_checks_cancellation_immediately_after_page_encoding(
    tmp_path, monkeypatch
) -> None:
    from uniscan.export import exporters

    output = tmp_path / "output.pdf"
    output.write_bytes(b"previous")
    encoded = False
    real_write = exporters.imwrite_unicode

    def write_then_cancel(path, image) -> bool:
        nonlocal encoded
        result = real_write(path, image)
        encoded = True
        return result

    monkeypatch.setattr(exporters, "imwrite_unicode", write_then_cancel)

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        export_pages_as_pdf(
            _pages()[:1],
            out_pdf=output,
            cancel_cb=lambda: encoded,
        )

    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".output.pdf.stage-*"))


def test_file_export_replaces_entire_directory_without_stale_pages(tmp_path) -> None:
    output_dir = tmp_path / "pages"
    export_pages_as_files(_pages(), output_dir=output_dir, ext="png", base_name="page")
    (output_dir / "unrelated.txt").write_text("stale", encoding="utf-8")

    paths = export_pages_as_files(_pages()[:1], output_dir=output_dir, ext="png", base_name="page")

    assert paths == [output_dir / "page_00001.png"]
    assert sorted(path.name for path in output_dir.iterdir()) == [
        ".uniscan-export-manifest.json",
        "page_00001.png",
        "unrelated.txt",
    ]
    assert (output_dir / "unrelated.txt").read_text(encoding="utf-8") == "stale"


def test_file_export_preserves_personal_page_like_neighbours_and_fails_on_collision(
    tmp_path,
) -> None:
    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    personal_png = output_dir / "page_00002.png"
    personal_pdf = output_dir / "page_00002.pdf"
    personal_png.write_bytes(b"personal-png")
    personal_pdf.write_bytes(b"personal-pdf")

    outputs = export_pages_as_files(
        _pages()[:1], output_dir=output_dir, ext="png", base_name="page"
    )

    assert outputs[0].is_file()
    assert personal_png.read_bytes() == b"personal-png"
    assert personal_pdf.read_bytes() == b"personal-pdf"

    with pytest.raises(ValueError, match="unowned image-export collision"):
        export_pages_as_files(_pages(), output_dir=output_dir, ext="png", base_name="page")

    assert personal_png.read_bytes() == b"personal-png"
    assert personal_pdf.read_bytes() == b"personal-pdf"


def test_file_export_rejects_invalid_ownership_manifest(tmp_path) -> None:
    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    manifest = output_dir / ".uniscan-export-manifest.json"
    manifest.write_text('{"schemaVersion": 999, "files": ["notes.txt"]}', encoding="utf-8")
    notes = output_dir / "notes.txt"
    notes.write_text("personal", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid UniScan image-export manifest"):
        export_pages_as_files(_pages()[:1], output_dir=output_dir)

    assert notes.read_text(encoding="utf-8") == "personal"
    assert manifest.exists()


def test_file_export_failure_preserves_existing_directory(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    keep = output_dir / "keep.txt"
    keep.write_text("previous", encoding="utf-8")
    calls = 0

    def fail_second_write(path, image) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            return False
        path.write_bytes(b"first")
        return True

    monkeypatch.setattr("uniscan.export.exporters.imwrite_unicode", fail_second_write)

    with pytest.raises(RuntimeError, match="Failed to write page image"):
        export_pages_as_files(_pages(), output_dir=output_dir)

    assert keep.read_text(encoding="utf-8") == "previous"
    assert [path.name for path in output_dir.iterdir()] == ["keep.txt"]
    assert not list(tmp_path.glob(".pages.stage-*"))


def test_file_export_failed_refresh_preserves_owned_pages_byte_for_byte(
    tmp_path, monkeypatch
) -> None:
    output_dir = tmp_path / "pages"
    previous_paths = export_pages_as_files(_pages(), output_dir=output_dir)
    previous_bytes = {path.name: path.read_bytes() for path in previous_paths}
    calls = 0

    def fail_second_write(path, image) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            return False
        path.write_bytes(b"new-staged-page")
        return True

    monkeypatch.setattr("uniscan.export.exporters.imwrite_unicode", fail_second_write)

    with pytest.raises(RuntimeError, match="Failed to write page image"):
        export_pages_as_files(_pages(), output_dir=output_dir)

    assert {path.name: path.read_bytes() for path in previous_paths} == previous_bytes
    assert not list(tmp_path.glob(".pages.stage-*"))


def test_file_export_hardlinks_unrelated_files_instead_of_copying_bytes(
    tmp_path, monkeypatch
) -> None:
    from uniscan.export import exporters

    output_dir = tmp_path / "pages"
    export_pages_as_files(_pages()[:1], output_dir=output_dir)
    unrelated = output_dir / "archive.bin"
    unrelated.write_bytes(os.urandom(2 * 1024 * 1024))
    inode_before = unrelated.stat().st_ino

    probe = tmp_path / "hardlink-probe"
    try:
        os.link(unrelated, probe)
    except OSError as exc:
        pytest.skip(f"Hard links are not supported on this filesystem: {exc}")
    else:
        probe.unlink()

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("Regular unrelated files should be hard-linked, not copied.")

    monkeypatch.setattr(exporters.shutil, "copy2", unexpected_copy)
    export_pages_as_files(_pages()[:1], output_dir=output_dir)

    assert unrelated.stat().st_ino == inode_before
    assert unrelated.stat().st_size == 2 * 1024 * 1024


def test_file_export_falls_back_to_copy_when_hardlinks_are_unavailable(
    tmp_path, monkeypatch
) -> None:
    from uniscan.export import exporters

    output_dir = tmp_path / "pages"
    export_pages_as_files(_pages()[:1], output_dir=output_dir)
    unrelated = output_dir / "keep.txt"
    unrelated.write_text("personal", encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(exporters.os, "link", unavailable)
    export_pages_as_files(_pages()[:1], output_dir=output_dir)

    assert unrelated.read_text(encoding="utf-8") == "personal"


def test_file_export_rejects_symlink_output_without_touching_referent(tmp_path) -> None:
    referent = tmp_path / "personal-pages"
    referent.mkdir()
    (referent / "keep.txt").write_text("personal", encoding="utf-8")
    output_dir = tmp_path / "pages"
    try:
        output_dir.symlink_to(referent, target_is_directory=True)
    except OSError as exc:  # Windows may require Developer Mode for symlinks.
        pytest.skip(f"Cannot create a test symlink: {exc}")

    with pytest.raises(ValueError, match="link or junction"):
        export_pages_as_files(_pages()[:1], output_dir=output_dir)

    assert output_dir.is_symlink()
    assert (referent / "keep.txt").read_text(encoding="utf-8") == "personal"


def test_file_export_cancellation_preserves_existing_directory(tmp_path) -> None:
    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    keep = output_dir / "keep.txt"
    keep.write_text("previous", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        export_pages_as_files(_pages(), output_dir=output_dir, cancel_cb=lambda: True)

    assert keep.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".pages.stage-*"))


def test_file_export_cancellation_during_publish_rolls_back(tmp_path) -> None:
    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    keep = output_dir / "keep.txt"
    keep.write_text("previous", encoding="utf-8")
    checks = 0

    def cancel_after_backup() -> bool:
        nonlocal checks
        checks += 1
        return checks == 4

    with pytest.raises(RuntimeError, match="Cancelled by user"):
        export_pages_as_files(
            _pages(),
            output_dir=output_dir,
            cancel_cb=cancel_after_backup,
        )

    assert keep.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".pages.stage-*"))
    assert not list(tmp_path.glob(".pages.backup-*"))


def test_file_export_backup_cleanup_failure_is_post_commit(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("previous", encoding="utf-8")

    from uniscan.export import exporters

    real_remove = exporters._remove_path

    def locked_backup(path) -> None:
        if ".backup-" in path.name:
            raise PermissionError("locked by scanner")
        real_remove(path)

    monkeypatch.setattr(exporters, "_remove_path", locked_backup)

    outputs = export_pages_as_files(_pages()[:1], output_dir=output_dir)

    assert outputs == [output_dir / "page_00001.png"]
    assert outputs[0].is_file()
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "previous"
    assert len(list(tmp_path.glob(".pages.backup-*"))) == 1
    assert (tmp_path / ".pages.uniscan-directory-transaction.json").is_file()


def test_file_export_recovers_exact_journaled_backup_left_before_publish(
    tmp_path, monkeypatch
) -> None:
    from uniscan.export import exporters

    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("previous", encoding="utf-8")
    real_replace = exporters.os.replace

    def crash_before_directory_publish(source_path, target_path) -> None:
        source = Path(source_path)
        target = Path(target_path)
        if target == output_dir and source.name.startswith(".pages.stage-"):
            raise KeyboardInterrupt("simulated termination")
        real_replace(source_path, target_path)

    monkeypatch.setattr(exporters.os, "replace", crash_before_directory_publish)
    with pytest.raises(KeyboardInterrupt, match="simulated termination"):
        export_pages_as_files(_pages()[:1], output_dir=output_dir)
    monkeypatch.setattr(exporters.os, "replace", real_replace)

    journal = tmp_path / ".pages.uniscan-directory-transaction.json"
    assert journal.is_file()
    assert not output_dir.exists()
    assert len(list(tmp_path.glob(".pages.backup-*"))) == 1

    outputs = export_pages_as_files(_pages()[:1], output_dir=output_dir)

    assert outputs[0].is_file()
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "previous"
    assert not journal.exists()
    assert not list(tmp_path.glob(".pages.backup-*"))


def test_file_export_ignores_foreign_backup_like_siblings(tmp_path) -> None:
    output_dir = tmp_path / "pages"
    foreign_directory = tmp_path / ".pages.backup-personal"
    foreign_file = tmp_path / f".pages.backup-{'a' * 32}"
    foreign_directory.mkdir()
    (foreign_directory / "photos.txt").write_text("personal-dir", encoding="utf-8")
    foreign_file.write_text("personal-file", encoding="utf-8")

    outputs = export_pages_as_files(_pages()[:1], output_dir=output_dir)

    assert outputs[0].is_file()
    assert (foreign_directory / "photos.txt").read_text(encoding="utf-8") == "personal-dir"
    assert foreign_file.read_text(encoding="utf-8") == "personal-file"


def test_file_export_fails_safely_while_same_output_lock_is_active(tmp_path) -> None:
    from uniscan.export import exporters

    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    keep = output_dir / "keep.txt"
    keep.write_text("personal", encoding="utf-8")

    with exporters._directory_export_lock(output_dir):
        with pytest.raises(RuntimeError, match="Another UniScan process"):
            export_pages_as_files(_pages()[:1], output_dir=output_dir)

    assert keep.read_text(encoding="utf-8") == "personal"
    assert not list(tmp_path.glob(".pages.stage-*"))
    assert not (tmp_path / ".pages.uniscan-directory-transaction.json").exists()


def test_image_path_export_failure_preserves_existing_directory(tmp_path) -> None:
    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    keep = output_dir / "keep.txt"
    keep.write_text("previous", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        export_image_paths_as_files(
            [tmp_path / "missing.png"],
            output_dir=output_dir,
            ext="png",
        )

    assert keep.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".pages.stage-*"))


def test_image_path_export_can_atomically_refresh_sources_in_same_directory(tmp_path) -> None:
    output_dir = tmp_path / "pages"
    source_paths = export_pages_as_files(
        _pages()[:1],
        output_dir=output_dir,
        ext="png",
        base_name="page",
    )
    original = source_paths[0].read_bytes()

    refreshed = export_image_paths_as_files(
        source_paths,
        output_dir=output_dir,
        ext="png",
        base_name="page",
    )

    assert refreshed == [output_dir / "page_00001.png"]
    assert refreshed[0].read_bytes() == original


@pytest.mark.parametrize(
    ("source_ext", "output_ext"),
    (("jpeg", "jpg"), ("tiff", "tif")),
)
def test_image_path_export_preserves_bytes_for_equivalent_extensions(
    tmp_path, source_ext: str, output_ext: str
) -> None:
    source_dir = tmp_path / "source"
    source_paths = export_pages_as_files(
        _pages()[:1],
        output_dir=source_dir,
        ext=source_ext,
        base_name="source",
    )
    original = source_paths[0].read_bytes()

    outputs = export_image_paths_as_files(
        source_paths,
        output_dir=tmp_path / "converted",
        ext=output_ext,
        base_name="page",
    )

    assert outputs[0].suffix == f".{output_ext}"
    assert outputs[0].read_bytes() == original


def test_image_path_export_rejects_corrupt_same_format_before_publish(tmp_path) -> None:
    output_dir = tmp_path / "pages"
    output_dir.mkdir()
    keep = output_dir / "keep.txt"
    keep.write_text("previous", encoding="utf-8")
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    with pytest.raises(RuntimeError, match="Cannot read source image"):
        export_image_paths_as_files(
            [corrupt],
            output_dir=output_dir,
            ext="png",
        )

    assert keep.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".pages.stage-*"))
