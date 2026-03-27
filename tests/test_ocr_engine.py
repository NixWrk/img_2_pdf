from pathlib import Path

import pytest

from uniscan.ocr.engine import (
    OCR_ENGINE_LABELS,
    OCR_ENGINE_CHANDRA,
    OCR_ENGINE_OLMOCR,
    OCR_ENGINE_MINERU,
    OCR_ENGINE_OCRMYPDF,
    OCR_ENGINE_PADDLEOCR,
    OCR_ENGINE_PYMUPDF,
    OCR_ENGINE_PYTESSERACT,
    OCR_ENGINE_SURYA,
    OCR_ENGINE_VALUES,
    SEARCHABLE_PDF_ENGINES,
    OcrDependencyStatus,
    OcrEngineStatus,
    detect_ocr_dependencies,
    detect_ocr_engine_status,
    image_paths_to_searchable_pdf,
)


def _importer_factory(available_modules: set[str]):
    def _importer(name: str):
        if name in available_modules:
            return object()
        raise ImportError(name)

    return _importer


def _which_factory(available_commands: set[str]):
    def _which(name: str):
        return name if name in available_commands else None

    return _which


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


def test_ocr_engine_registry_is_stable() -> None:
    assert OCR_ENGINE_VALUES == (
        OCR_ENGINE_PYTESSERACT,
        OCR_ENGINE_OCRMYPDF,
        OCR_ENGINE_PADDLEOCR,
        OCR_ENGINE_PYMUPDF,
        OCR_ENGINE_SURYA,
        OCR_ENGINE_MINERU,
        OCR_ENGINE_CHANDRA,
        OCR_ENGINE_OLMOCR,
    )
    assert SEARCHABLE_PDF_ENGINES == (
        OCR_ENGINE_PYTESSERACT,
        OCR_ENGINE_OCRMYPDF,
        OCR_ENGINE_PYMUPDF,
    )
    assert all(engine in OCR_ENGINE_LABELS for engine in OCR_ENGINE_VALUES)


@pytest.mark.parametrize(
    ("engine_name", "expected_searchable_pdf"),
    [
        (OCR_ENGINE_PYTESSERACT, True),
        (OCR_ENGINE_OCRMYPDF, True),
        (OCR_ENGINE_PADDLEOCR, False),
        (OCR_ENGINE_PYMUPDF, True),
        (OCR_ENGINE_SURYA, False),
        (OCR_ENGINE_MINERU, False),
        (OCR_ENGINE_CHANDRA, False),
        (OCR_ENGINE_OLMOCR, False),
    ],
)
def test_detect_ocr_engine_status_ready_matrix(engine_name: str, expected_searchable_pdf: bool) -> None:
    status = detect_ocr_engine_status(
        engine_name,
        import_module=_importer_factory(
            {
                "pytesseract",
                "pypdf",
                "img2pdf",
                "fitz",
                "paddleocr",
                "surya",
                "marker",
                "mineru",
                "magic_pdf",
                "ftfy",
                "dill",
                "omegaconf",
                "chandra_ocr",
                "olmocr",
            }
        ),
        which_fn=_which_factory({"tesseract", "ocrmypdf", "chandra"}),
    )
    assert status.ready
    assert status.searchable_pdf is expected_searchable_pdf
    assert status.label == OCR_ENGINE_LABELS[engine_name]


@pytest.mark.parametrize(
    ("engine_name", "imported_modules", "available_commands", "expected_missing"),
    [
        (OCR_ENGINE_PYTESSERACT, set(), set(), ["pytesseract", "pypdf", "tesseract"]),
        (OCR_ENGINE_OCRMYPDF, {"img2pdf"}, set(), ["ocrmypdf"]),
        (OCR_ENGINE_PADDLEOCR, set(), set(), ["paddleocr"]),
        (OCR_ENGINE_PYMUPDF, set(), set(), ["pymupdf(fitz)", "pypdf", "tesseract"]),
        (OCR_ENGINE_SURYA, set(), set(), ["surya/marker"]),
        (OCR_ENGINE_MINERU, set(), set(), ["mineru(magic_pdf)"]),
        (OCR_ENGINE_CHANDRA, set(), set(), ["chandra-ocr(chandra)"]),
        (OCR_ENGINE_OLMOCR, set(), set(), ["olmocr"]),
    ],
)
def test_detect_ocr_engine_status_missing_matrix(
    engine_name: str,
    imported_modules: set[str],
    available_commands: set[str],
    expected_missing: list[str],
) -> None:
    status = detect_ocr_engine_status(
        engine_name,
        import_module=_importer_factory(imported_modules),
        which_fn=_which_factory(available_commands),
    )
    assert not status.ready
    assert status.missing == expected_missing
    assert status.searchable_pdf is (engine_name in SEARCHABLE_PDF_ENGINES)


def test_detect_ocr_engine_status_ocrmypdf_missing_cmd() -> None:
    status = detect_ocr_engine_status(
        OCR_ENGINE_OCRMYPDF,
        import_module=_importer_factory({"img2pdf"}),
        which_fn=lambda _cmd: None,
    )
    assert not status.ready
    assert "ocrmypdf" in status.missing
    assert status.searchable_pdf


def test_detect_ocr_engine_status_paddleocr_ready_but_no_searchable_pdf() -> None:
    status = detect_ocr_engine_status(
        OCR_ENGINE_PADDLEOCR,
        import_module=_importer_factory({"paddleocr"}),
        which_fn=_which_factory(set()),
    )
    assert status.ready
    assert not status.searchable_pdf


def test_detect_ocr_engine_status_mineru_requires_runtime_modules() -> None:
    status = detect_ocr_engine_status(
        OCR_ENGINE_MINERU,
        import_module=_importer_factory({"mineru"}),
        which_fn=_which_factory(set()),
    )
    assert not status.ready
    assert set(status.missing) == {"ftfy", "dill", "omegaconf"}


def test_detect_ocr_engine_status_paddleocr_searchable_via_ocrmypdf_plugin() -> None:
    status = detect_ocr_engine_status(
        OCR_ENGINE_PADDLEOCR,
        import_module=_importer_factory({"paddleocr", "img2pdf", "ocrmypdf_paddleocr"}),
        which_fn=_which_factory({"ocrmypdf"}),
    )
    assert status.ready
    assert status.searchable_pdf


def test_detect_ocr_engine_status_chandra_ready_from_cli_only() -> None:
    status = detect_ocr_engine_status(
        OCR_ENGINE_CHANDRA,
        import_module=_importer_factory(set()),
        which_fn=_which_factory({"chandra"}),
    )
    assert status.ready
    assert not status.searchable_pdf


def test_image_paths_to_searchable_pdf_rejects_unwired_ocr_engines(tmp_path: Path) -> None:
    for engine_name in (
        OCR_ENGINE_PADDLEOCR,
        OCR_ENGINE_SURYA,
        OCR_ENGINE_MINERU,
        OCR_ENGINE_CHANDRA,
        OCR_ENGINE_OLMOCR,
    ):
        try:
            image_paths_to_searchable_pdf(
                [tmp_path / "page.png"],
                out_pdf=tmp_path / f"{engine_name}.pdf",
                engine_name=engine_name,
                engine_status=OcrEngineStatus(
                    engine_name=engine_name,
                    ready=True,
                    missing=[],
                    searchable_pdf=False,
                ),
                import_module=_importer_factory(
                    {"paddleocr", "surya", "marker", "mineru", "magic_pdf", "chandra_ocr", "olmocr"}
                ),
                which_fn=_which_factory({"tesseract", "ocrmypdf"}),
            )
        except NotImplementedError as exc:
            assert OCR_ENGINE_LABELS[engine_name] in str(exc)
        else:
            raise AssertionError(f"Expected NotImplementedError for {engine_name}")


def test_image_paths_to_searchable_pdf_pytesseract_merges_pages(tmp_path: Path) -> None:
    class FakePytesseract:
        @staticmethod
        def image_to_pdf_or_hocr(src: str, extension: str, lang: str) -> bytes:
            assert extension == "pdf"
            assert lang == "eng"
            return f"PDF:{Path(src).name}".encode("utf-8")

    class FakeReader:
        def __init__(self, stream, strict=False) -> None:
            assert strict is False
            self.pages = [stream.read()]
            stream.seek(0)

    class FakeWriter:
        def __init__(self) -> None:
            self.pages: list[bytes] = []

        def add_page(self, page) -> None:
            self.pages.append(page)

        def write(self, fh) -> None:
            fh.write(b"MERGED|" + b"|".join(self.pages))

    class FakePypdf:
        PdfReader = FakeReader
        PdfWriter = FakeWriter

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


def test_image_paths_to_searchable_pdf_paddleocr_via_ocrmypdf_plugin_calls_cli(tmp_path: Path) -> None:
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
        out_path.write_bytes(b"%PDF-PLUGIN")
        return FakeProc()

    def _build_pdf(_paths, out_pdf, dpi):
        assert int(dpi) == 300
        Path(out_pdf).write_bytes(b"%PDF-INPUT")

    out = image_paths_to_searchable_pdf(
        [tmp_path / "1.png"],
        out_pdf=tmp_path / "paddle_plugin_out.pdf",
        engine_name=OCR_ENGINE_PADDLEOCR,
        lang="eng",
        engine_status=OcrEngineStatus(
            engine_name=OCR_ENGINE_PADDLEOCR,
            ready=True,
            missing=[],
            searchable_pdf=True,
        ),
        import_module=_importer_factory({"img2pdf", "ocrmypdf_paddleocr"}),
        which_fn=_which_factory({"ocrmypdf"}),
        run_cmd=_run,
        build_pdf_fn=_build_pdf,
    )
    assert out.exists()
    assert calls
    assert calls[0][0] == "ocrmypdf"
    assert "--plugin" in calls[0]
    plugin_index = calls[0].index("--plugin")
    assert calls[0][plugin_index + 1] == "ocrmypdf_paddleocr"


def test_image_paths_to_searchable_pdf_empty_raises(tmp_path: Path) -> None:
    try:
        image_paths_to_searchable_pdf([], out_pdf=tmp_path / "empty.pdf")
    except ValueError as exc:
        assert "No image paths to OCR" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty image paths")
