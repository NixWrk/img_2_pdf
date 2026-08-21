from types import SimpleNamespace

from uniscan.ui.export_preflight import build_export_preflight


def test_export_preflight_summarizes_ready_warning_and_blocked_pages() -> None:
    pages = build_export_preflight(
        [
            SimpleNamespace(
                name="ready", entry_id="ready", crop_state="applied", committed_processing=object()
            ),
            SimpleNamespace(
                name="raw", entry_id="raw", crop_state="applied", committed_processing=None
            ),
            SimpleNamespace(
                name="review", entry_id="review", crop_state="none", committed_processing=None
            ),
            SimpleNamespace(
                name="candidate",
                entry_id="candidate",
                crop_state="applied",
                committed_processing=object(),
            ),
        ],
        candidate_entry_ids={"candidate"},
    )

    assert pages.ready_count == 1
    assert pages.warning_count == 1
    assert pages.blocked_count == 2
    assert pages.can_export is False
    summary = pages.summary()
    assert "1 ready" in summary
    assert "1 warning" in summary
    assert "2 blocked" in summary
    assert "page 4 · candidate" in summary


def test_export_preflight_allows_plain_committed_pages() -> None:
    preflight = build_export_preflight(
        [
            SimpleNamespace(
                name="page", entry_id="page", crop_state="applied", committed_processing=object()
            )
        ]
    )
    assert preflight.can_export is True
    assert preflight.ready_count == 1
