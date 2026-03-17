from pathlib import Path

from uniscan.ocr.engine import (
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_PYMUPDF,
    OcrDependencyStatus,
    OcrEngineStatus,
    detect_ocr_dependencies,
    detect_ocr_engine_status,
    image_paths_to_searchable_pdf,
)


def test_detect_ocr_dependencies_missing() -> None:
    def _importer(_name: str):
        raise ImportError("missing")

    status = detect_ocr_dependencies(import_module=_importer, which_fn=lambda _cmd: None)
    assert not status.ready
    assert status.missing == ["pytesseract", "pypdf", "tesseract"]


def test_detect_ocr_dependencies_ready() -> None:
    def _importer(_name: str):
        return object()

    status = detect_ocr_dependencies(import_module=_importer, which_fn=lambda _cmd: "ok")
    assert status.ready
    assert status.missing == []


def test_detect_ocr_engine_status_ocrmypdf_missing_cmd() -> None:
    status = detect_ocr_engine_status(
        OCR_ENGINE_OCRMYPDF,
        import_module=lambda _name: object(),
        which_fn=lambda _cmd: None,
    )
    assert not status.ready
    assert "ocrmypdf" in status.missing
    assert status.searchable_pdf


def test_detect_ocr_engine_status_paddleocr_ready_but_no_searchable_pdf() -> None:
    status = detect_ocr_engine_status(
        OCR_ENGINE_PADDLEOCR,
        import_module=lambda _name: object(),
        which_fn=lambda _cmd: None,
    )
    assert status.ready
    assert not status.searchable_pdf


def test_image_paths_to_searchable_pdf_pytesseract_merges_pages(tmp_path: Path) -> None:
    class FakePytesseract:
        @staticmethod
        def image_to_pdf_or_hocr(src: str, extension: str, lang: str) -> bytes:
            assert extension == "pdf"
            assert lang == "eng"
            return f"PDF:{Path(src).name}".encode("utf-8")

    class FakeMerger:
        def __init__(self) -> None:
            self.pages: list[bytes] = []

        def append(self, stream) -> None:
            self.pages.append(stream.read())
            stream.seek(0)

        def write(self, fh) -> None:
            fh.write(b"MERGED|" + b"|".join(self.pages))

        def close(self) -> None:
            return None

    class FakePypdf:
        PdfMerger = FakeMerger

    def _importer(name: str):
        if name == "pytesseract":
            return FakePytesseract
        if name == "pypdf":
            return FakePypdf
        raise ImportError(name)

    out_pdf = tmp_path / "ocr.pdf"
    image_paths = [tmp_path / "a.png", tmp_path / "b.png"]
    out = image_paths_to_searchable_pdf(
        image_paths,
        out_pdf=out_pdf,
        lang="eng",
        dependency_status=OcrDependencyStatus(True, True, True),
        import_module=_importer,
    )

    assert out == out_pdf
    assert out.exists()
    data = out.read_bytes()
    assert b"MERGED|" in data
    assert b"PDF:a.png" in data
    assert b"PDF:b.png" in data


def test_image_paths_to_searchable_pdf_ocrmypdf_calls_cli(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, capture_output, text):
        assert capture_output
        assert text
        calls.append([str(x) for x in cmd])
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"%PDF-FAKE")
        return FakeProc()

    def _build_pdf(_paths, out_pdf, dpi):
        assert int(dpi) == 300
        Path(out_pdf).write_bytes(b"%PDF-INPUT")

    out = image_paths_to_searchable_pdf(
        [tmp_path / "1.png"],
        out_pdf=tmp_path / "ocrmypdf_out.pdf",
        engine_name=OCR_ENGINE_OCRMYPDF,
        lang="eng",
        engine_status=OcrEngineStatus(
            engine_name=OCR_ENGINE_OCRMYPDF,
            ready=True,
            missing=[],
            searchable_pdf=True,
        ),
        which_fn=lambda _cmd: "ocrmypdf",
        run_cmd=_run,
        build_pdf_fn=_build_pdf,
    )
    assert out.exists()
    assert calls
    assert calls[0][0] == "ocrmypdf"
    assert "--language" in calls[0]


def test_image_paths_to_searchable_pdf_pymupdf_no_ocr_support_raises(tmp_path: Path) -> None:
    class FakePixmap:
        def __init__(self, _path: str) -> None:
            pass

    class FakeFitz:
        Pixmap = FakePixmap

    class FakeMerger:
        def append(self, _stream) -> None:
            return None

        def write(self, _fh) -> None:
            return None

        def close(self) -> None:
            return None

    class FakePypdf:
        PdfMerger = FakeMerger

    def _importer(name: str):
        if name == "fitz":
            return FakeFitz
        if name == "pypdf":
            return FakePypdf
        raise ImportError(name)

    try:
        image_paths_to_searchable_pdf(
            [tmp_path / "x.png"],
            out_pdf=tmp_path / "out.pdf",
            engine_name=OCR_ENGINE_PYMUPDF,
            engine_status=OcrEngineStatus(
                engine_name=OCR_ENGINE_PYMUPDF,
                ready=True,
                missing=[],
                searchable_pdf=True,
            ),
            import_module=_importer,
        )
    except RuntimeError as exc:
        assert "no OCR support" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for PyMuPDF without OCR support")


def test_image_paths_to_searchable_pdf_empty_raises(tmp_path: Path) -> None:
    try:
        image_paths_to_searchable_pdf([], out_pdf=tmp_path / "empty.pdf")
    except ValueError as exc:
        assert "No image paths to OCR" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty image paths")
