# Historical follow-up plan after 0.1.0

> Superseded by the canonical [`roadmap.md`](roadmap.md). This file is retained to explain earlier
> release and architecture decisions.

This plan keeps OCR out of scope and focuses on turning the verified preview into a maintainable
Windows scanning application.

## P0 — release evidence

1. Run the manual checklist on a clean Windows 11 x64 machine with at least one USB camera and
   attach logs, screenshots, artifact SHA-256, and camera model to the release issue.
2. Exercise a 500-page mixed image/PDF conversion while recording peak memory, elapsed time, and
   output rollback under cancellation.
3. Select an organization-backed Authenticode provider, add keyless/managed signing to the tag
   workflow, and verify signature plus timestamp before ZIP creation.

Exit: the portable artifact passes clean-machine GUI, camera, CLI, removal, and signature checks.

## P1 — one processing controller

1. Move GUI import/review orchestration out of `ui/app.py` into GUI-independent controllers.
2. Make GUI PDF import consume `iter_input_items()` directly so large PDFs are page-streamed in
   the desktop flow as well as the headless flow.
3. Reuse the same detector policy, strict/fallback semantics, preprocessing options, progress
   events, and report schema in GUI and CLI.
4. Add controller-level cancellation/rollback tests and reduce widget tests to rendering/wiring.

Exit: GUI and CLI produce byte-equivalent pages for the same options, with no duplicated
pipeline policy.

## P1 — session lifecycle

1. Add explicit “New session”, “Save session as”, “Open session”, and “Discard recovery” actions.
2. Version the session manifest migration path and quarantine corrupt/incomplete sessions instead
   of repeatedly retrying them.
3. Add bounded retention and cleanup for abandoned session directories.
4. Persist per-page processing settings and export defaults, not only page assets/order.

Exit: users can intentionally manage sessions and storage cannot grow without a visible policy.

## P2 — acquisition and quality

1. Add camera capability negotiation (formats, frame rates, focus/exposure support) and surface
   actionable diagnostics rather than backend property numbers.
2. Extend the quality corpus with consented real-camera captures under a documented license;
   retain synthetic cases for deterministic CI.
3. Split lighting correction into shadow normalization and glare detection, report clipped areas,
   and avoid claiming recovery where highlights contain no source detail.
4. Track quality metrics by category and compare both Office Lens and CV hybrid baselines.

Exit: quality changes are category-specific, measurable, and validated on synthetic plus licensed
real captures.

## P2 — Windows productization

1. Decide whether to keep the portable console executable or ship separate GUI (windowed) and CLI
   entrypoints with shared runtime files.
2. Add application icon/version resources and an optional per-user installer only if portable ZIP
   support becomes insufficient.
3. Generate an SBOM, dependency license inventory, vulnerability scan, and reproducible release
   provenance alongside every artifact.
4. Add upgrade compatibility tests for autosaved sessions between released versions.

Exit: releases have publisher identity, provenance, dependency evidence, and a tested upgrade path.
