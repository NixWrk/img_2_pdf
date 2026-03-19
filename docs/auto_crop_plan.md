# Auto Crop Upgrade Plan

## Goal

Reduce manual corner editing for long document batches by improving automatic page boundary detection.

## Candidate Approaches

1. `camscan` backend
Current external scanner logic already integrated in the project. Keep it as one candidate, but stop relying on it as the only path.

2. OpenCV quad detector
Classical CV pipeline: grayscale, blur, threshold/edges, morphology, contour ranking, quadrilateral fitting, perspective warp. This is the lowest-friction offline fallback and should be the default baseline.

3. PaddleOCR UVDoc
Model-based image rectification is available in PaddleOCR as `TextImageUnwarping (UVDoc)`. This is promising for hard perspective cases, but adds model download/runtime weight and should be an optional test branch, not the first-line dependency.

## Commit Plan

1. `docs(plan): add auto-crop detector roadmap`
Write the detector strategy, evaluation variants, and staged rollout plan.

2. `feat(cv): add robust OpenCV document detector fallback`
Implement a local multi-stage contour detector and perspective warp fallback behind the existing scanner adapter.

3. `test(cv): add synthetic detector regression coverage`
Add deterministic tests for contour detection and warp output on perspective-distorted synthetic pages.

4. `feat(ui): wire auto-crop editor to detector chain`
Use the new detector chain inside the review auto-crop flow and surface backend/debug status if useful.

5. `perf(cv): batch auto-crop scoring and background preview pass`
Precompute candidate corners for long batches and avoid making the user step through each page manually.

6. `spike(cv): evaluate optional UVDoc branch`
Prototype PaddleOCR UVDoc as an optional engine for difficult cases and compare speed/quality against the OpenCV fallback.

## Acceptance Criteria

1. Auto-crop works without vendored `camscan`.
2. Long imported document batches can be auto-cropped with a single pass and spot-check edits.
3. Detection failures degrade gracefully to manual edit, without blocking batch workflow.
4. Regression tests cover basic page-like perspective cases.
