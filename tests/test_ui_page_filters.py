from types import SimpleNamespace

from uniscan.session import CROP_STATE_APPLIED, CROP_STATE_NONE, CROP_STATE_PROPOSED
from uniscan.ui.app import (
    PAGE_FILTER_EDITED,
    PAGE_FILTER_ERRORS,
    PAGE_FILTER_NEEDS_REVIEW,
    page_matches_filter,
    page_status,
    visible_session_indices,
)
from uniscan.ui.app import UnifiedScanApp


def _page(entry_id: str, **kwargs):
    values = dict(
        entry_id=entry_id,
        name=entry_id,
        crop_state=CROP_STATE_APPLIED,
        detected_contour=None,
        needs_review=False,
        revision=0,
        committed_processing=None,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_page_status_uses_existing_readiness_and_processing_facts() -> None:
    review = _page("review", crop_state=CROP_STATE_NONE)
    edited = _page("edited", revision=2)
    failed = _page("failed", processing_error="preview failed")

    assert page_status(review) == ("?", "Needs review")
    assert page_status(edited) == ("~", "Edited")
    assert page_status(failed) == ("!", "Error")
    assert page_status(_page("candidate"), candidate=True) == ("~", "Candidate")
    assert page_status(_page("ready")) == ("✓", "Ready")


def test_crop_proposal_is_always_needs_review_status() -> None:
    proposal = _page("proposal", crop_state=CROP_STATE_PROPOSED, detected_contour=object())

    assert page_status(proposal) == ("?", "Needs review")
    assert page_matches_filter(proposal, PAGE_FILTER_NEEDS_REVIEW)


def test_page_filter_mapping_preserves_session_indexes() -> None:
    pages = (
        _page("ready"),
        _page("review", crop_state=CROP_STATE_NONE),
        _page("edited", revision=1),
        _page("error", processing_error="failed"),
    )

    assert visible_session_indices(pages, "All") == (0, 1, 2, 3)
    assert visible_session_indices(pages, PAGE_FILTER_NEEDS_REVIEW) == (1,)
    assert visible_session_indices(pages, PAGE_FILTER_EDITED) == (2,)
    assert visible_session_indices(pages, PAGE_FILTER_ERRORS) == (3,)
    assert page_matches_filter(pages[2], PAGE_FILTER_EDITED)
    assert not page_matches_filter(pages[0], PAGE_FILTER_EDITED)


def test_candidate_and_error_overrides_are_entry_scoped() -> None:
    pages = (_page("first"), _page("second"), _page("third"))

    assert visible_session_indices(pages, PAGE_FILTER_EDITED, candidate_entry_ids={"second"}) == (1,)
    assert visible_session_indices(pages, PAGE_FILTER_ERRORS, error_entry_ids={"third"}) == (2,)


def test_filter_refresh_smoke_routes_visible_selection_to_session_page() -> None:
    class Listbox:
        def __init__(self):
            self.items = []
            self.selected = set()

        def delete(self, *_args):
            self.items.clear()
            self.selected.clear()

        def insert(self, _position, value):
            self.items.append(value)

        def curselection(self):
            return tuple(sorted(self.selected))

        def selection_set(self, index, *_args):
            if index == 0 and _args:
                self.selected.update(range(len(self.items)))
            else:
                self.selected.add(index)

        def selection_clear(self, *_args):
            self.selected.clear()

    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    app = object.__new__(UnifiedScanApp)
    app.session = SimpleNamespace(
        entries=[
            _page("ready"),
            _page("edited", revision=1),
            _page("review", crop_state=CROP_STATE_NONE),
        ]
    )
    app.page_listbox = Listbox()
    app.page_filter_var = Var("All")
    app.page_count_var = Var("")
    app.crop_warning_var = Var("")
    app.page_filter_buttons = {}
    app._page_filter_counts = {}
    app.toolbar_export_pdf_button = None
    app.toolbar_export_options_button = None
    app._update_crop_warning = lambda: None
    app._update_export_readiness = lambda: None
    app._update_page_action_states = lambda: None
    app._sync_controls_from_single_committed_page = lambda: None
    app.update_page_preview = lambda: None
    app._set_status = lambda _text: None

    app.refresh_page_list()
    app.page_listbox.selection_set(1)
    app._sync_page_selection_to_session()
    app._set_page_filter(PAGE_FILTER_EDITED)

    assert app._visible_session_indices == (1,)
    assert app._selected_entry_indices() == [1]
    assert "edited" in app.page_listbox.items[0]


def test_status_only_refresh_does_not_touch_preview_generation_or_render() -> None:
    app = object.__new__(UnifiedScanApp)
    app.review_preview_generation = 19
    calls = []
    app.refresh_page_list = lambda **kwargs: calls.append(kwargs)
    app.update_page_preview = lambda: (_ for _ in ()).throw(AssertionError("preview rerun"))

    app.refresh_page_rows(keep_entry_ids=("page",))

    assert calls == [{"keep_entry_ids": ("page",), "update_preview": False}]
    assert app.review_preview_generation == 19
