from __future__ import annotations

from types import SimpleNamespace

from uniscan.ui import pipeline_strip
from uniscan.ui.review_pipeline import build_pipeline_cards


class _FakeWidget:
    def __init__(self, parent=None, **options) -> None:
        self.parent = parent
        self.options = options
        self.children: list[_FakeWidget] = []
        if parent is not None:
            parent.children.append(self)

    def pack(self, **_options) -> None:
        return None

    def pack_propagate(self, _enabled: bool) -> None:
        return None

    def winfo_children(self) -> list[_FakeWidget]:
        return list(self.children)

    def destroy(self) -> None:
        if self.parent is not None:
            self.parent.children.remove(self)

    def cget(self, name: str):
        return self.options.get(name)


def _texts(widget: _FakeWidget) -> list[str]:
    texts: list[str] = []
    for child in widget.winfo_children():
        text = child.cget("text")
        if text:
            texts.append(str(text))
        texts.extend(_texts(child))
    return texts


def _entry(*, committed=None, revision=4):
    return SimpleNamespace(
        revision=revision,
        crop_state="applied" if committed else "none",
        detected_contour=object() if committed else None,
        detected_backend="automatic" if committed else None,
        review_reasons=(),
        committed_processing=committed,
    )


def _recipe():
    return SimpleNamespace(
        dewarp_method="auto",
        deskew_method="hybrid",
        shadow_method="auto",
        page_layout="a4",
    )


def test_pipeline_strip_renders_initial_committed_and_pending_states(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_strip.ctk, "CTkFrame", _FakeWidget)
    monkeypatch.setattr(pipeline_strip.ctk, "CTkLabel", _FakeWidget)
    monkeypatch.setattr(pipeline_strip.ctk, "CTkFont", lambda **options: options)
    frame = _FakeWidget()

    pipeline_strip.render_pipeline_strip(frame, ())
    assert "Select one page" in " ".join(_texts(frame))

    committed = SimpleNamespace(recipe=_recipe(), diagnostics={})
    committed_cards = build_pipeline_cards(_entry(committed=committed), pending_request=None)
    pipeline_strip.render_pipeline_strip(frame, committed_cards)
    committed_texts = _texts(frame)
    assert sum(card.title in committed_texts for card in committed_cards) == 7
    assert any("Applied" in text for text in committed_texts)

    pending_cards = build_pipeline_cards(_entry(revision=8), pending_request=_recipe())
    pipeline_strip.render_pipeline_strip(frame, pending_cards)
    pending_texts = _texts(frame)
    assert sum(card.title in pending_texts for card in pending_cards) == 7
    assert pending_cards[-1].status_label == "Running"
