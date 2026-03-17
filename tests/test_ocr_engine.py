from pathlib import Path

from uniscan.ocr.engine import OcrDependencyStatus, detect_ocr_dependencies, image_paths_to_searchable_pdf


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


def test_image_paths_to_searchable_pdf_merges_pages(tmp_path: Path) -> None:
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


def test_image_paths_to_searchable_pdf_empty_raises(tmp_path: Path) -> None:
    try:
        image_paths_to_searchable_pdf([], out_pdf=tmp_path / "empty.pdf")
    except ValueError as exc:
        assert "No image paths to OCR" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty image paths")
