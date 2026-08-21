"""Pure export-readiness checks used by the GUI before publishing output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from uniscan.session import CROP_STATE_NONE, CROP_STATE_PROPOSED


@dataclass(frozen=True, slots=True)
class ExportPreflightPage:
    page_number: int
    name: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return bool(self.blockers)


@dataclass(frozen=True, slots=True)
class ExportPreflight:
    pages: tuple[ExportPreflightPage, ...]

    @property
    def ready_count(self) -> int:
        return sum(not page.blocked and not page.warnings for page in self.pages)

    @property
    def warning_count(self) -> int:
        return sum(bool(page.warnings) for page in self.pages)

    @property
    def blocked_count(self) -> int:
        return sum(page.blocked for page in self.pages)

    @property
    def can_export(self) -> bool:
        return self.blocked_count == 0

    def summary(self, *, limit: int = 8) -> str:
        total = len(self.pages)
        page_label = "page" if total == 1 else "pages"
        warning_label = "warning" if self.warning_count == 1 else "warnings"
        lines = [
            f"Export readiness: {self.ready_count} ready · "
            f"{self.warning_count} {warning_label} · {self.blocked_count} blocked "
            f"— {total} {page_label}."
        ]
        details = [
            ("Blocked", page.page_number, page.name, page.blockers)
            for page in self.pages
            if page.blockers
        ]
        details.extend(
            ("Warning", page.page_number, page.name, page.warnings)
            for page in self.pages
            if page.warnings
        )
        for label, page_number, name, reasons in details[:limit]:
            lines.append(f"{label}: page {page_number} · {name} — {'; '.join(reasons)}")
        if len(details) > limit:
            lines.append(f"… and {len(details) - limit} more page(s).")
        return "\n".join(lines)


def _has_crop_proposal(entry) -> bool:
    return (
        getattr(entry, "crop_state", None) == CROP_STATE_PROPOSED
        and getattr(entry, "detected_contour", None) is not None
    )


def _needs_crop_review(entry) -> bool:
    return (
        bool(getattr(entry, "needs_review", False))
        or getattr(entry, "crop_state", None) == CROP_STATE_NONE
    )


def build_export_preflight(
    entries: Iterable[object],
    *,
    candidate_entry_ids: Iterable[str] = (),
) -> ExportPreflight:
    """Assess only page facts that make export unsafe or worth confirming."""
    candidates = frozenset(candidate_entry_ids)
    pages: list[ExportPreflightPage] = []
    for page_number, entry in enumerate(entries, start=1):
        blockers: list[str] = []
        warnings: list[str] = []
        if _has_crop_proposal(entry):
            blockers.append("Crop proposal is not applied")
        elif _needs_crop_review(entry):
            blockers.append("Automatic crop needs review")
        if getattr(entry, "entry_id", None) in candidates:
            blockers.append("Preview candidate is not committed")
        if getattr(entry, "committed_processing", None) is None:
            if blockers:
                # The crop/candidate blocker already explains why Apply is needed.
                pass
            else:
                warnings.append(
                    "No processing recipe is committed; current stored pixels will be exported"
                )
        pages.append(
            ExportPreflightPage(
                page_number=page_number,
                name=str(getattr(entry, "name", "Untitled page")).strip() or "Untitled page",
                blockers=tuple(blockers),
                warnings=tuple(warnings),
            )
        )
    return ExportPreflight(tuple(pages))
