# Editable automatic pipeline plan

Status: planned future work. This plan extends Phase 8 of the canonical
[`roadmap.md`](roadmap.md); it does not claim that the described behavior is implemented.

## Goal and product contract

Keep the normal document workflow automatic while making every material correction stage visible,
bypassable, and editable on one screen. Expensive models run only when cheap local evidence says
they may help. OCR remains outside this pipeline.

The workspace presents one non-destructive left-to-right chain:

```text
source -> perspective -> page waves -> lighting -> result
```

Each stage has an `Auto`/`Off` control, an `Edit` action, an input/output preview, and one explicit
status: `not needed`, `applied`, `rejected`, or `edited`. A manual edit refines or replaces only
that stage. Changing an earlier stage invalidates and recomputes downstream previews. The immutable
source is never overwritten, and Apply/export renders the saved recipe at full resolution.

## Work packages

### 1. Persist editable stage decisions

- Retain the detected and user-adjusted perspective quadrilateral against the raw page instead of
  baking away its editability.
- Retain the selected dewarp backend plus user residual curves.
- Retain the lighting method, correction strength/blend, and any protected region.
- Version and replay the complete per-page recipe in both GUI and CLI.
- Preserve accepted work in unrelated stages; invalidate only downstream results after an edit.

Acceptance: restart and CLI replay reproduce equivalent full-resolution output from the immutable
source and saved recipe.

### 2. Build one horizontal stage workspace

- Show editable boundary handles over the source image and the rectified result beside it.
- Feed the rectified result into automatic dewarp and expose the existing top/middle/bottom curve
  editor inline.
- Feed the dewarped result into lighting and expose manual correction strength and protected-region
  controls.
- Keep page selection, zoom, pan, cancellation, reset, and before/after comparison available while
  editing; do not replace the whole workspace with separate stage dialogs.
- Display `Auto`/`Off`, `Edit`, and the current `not needed`/`applied`/`rejected`/`edited` status on
  every stage.

Acceptance: an operator can inspect, bypass, edit, reset, and apply every stage without leaving the
workspace, and earlier edits refresh the downstream previews.

### 3. Skip unnecessary model inference without OCR

- Leave an already rectangular page unchanged when perspective correction has no measurable
  benefit.
- Before UVDoc, measure text-line curvature, table-line residuals, and projective evidence with
  deterministic OpenCV/NumPy analysis. Do not initialize the model when all evidenced defects are
  below calibrated thresholds.
- Keep the existing lighting no-op when no measurable shadow is present.
- Retain an explicit operator choice that forces a backend when automatic policy says no-op.
- Report why a stage was skipped, applied, rejected, or forced.

Acceptance: flat/no-op fixtures prove through call spies that optional models were not invoked;
explicit model selection still invokes the requested backend.

### 4. Gate content clipping, edge loss, and table regressions

Extend both the production candidate gate and the geometry benchmark with the same non-OCR
measurements:

- detect newly clipped text-like connected components separately on the left, right, top, and
  bottom sides;
- compare meaningful edge content per side instead of relying on one aggregate ink ratio;
- measure long horizontal/vertical table-line angle, residual curvature, continuity, and retained
  intersections;
- make the table rule a no-op when there is insufficient table evidence;
- reject a candidate that loses content or worsens an evidenced table even if another geometry
  metric improves.

Acceptance: deterministic synthetic edge-text and table fixtures cover clipping, retained borders,
line straightness, intersections, and insufficient-evidence behavior without recognizing text.
The difficult real PDF remains a manual visual regression case, with particular attention to edge
loss on cropped page halves.

### 5. Keep automatic and manual correction composable

- Four-corner perspective edits operate on the raw source.
- UVDoc or another selected backend produces the automatic page-wave candidate.
- User curves refine the accepted automatic candidate instead of silently disabling its backend.
- Apply the composed geometry with the Phase 8 single-resample design where possible.
- Defer arbitrary two-dimensional manual mesh construction until real examples prove that the
  curve editor cannot express the required residual correction.

Acceptance: model plus user refinement has the same preview and Apply semantics, preserves model
identity in diagnostics, and does not add an avoidable interpolation pass.

### 6. Add workflow and replay tests

- Cover `Auto`, `Off`, edit, reset, forced backend, and status presentation.
- Cover downstream invalidation: perspective invalidates waves and lighting; waves invalidate
  lighting; lighting changes no geometry stage.
- Prove that cancellation or a rejected candidate leaves source pixels and the previous committed
  recipe unchanged.
- Prove that saved manual recipes replay after restart and through the CLI.
- Keep benchmark and production quality calculations on one implementation.

Acceptance: controller tests establish state and pixel contracts; GUI smoke tests cover only the
visible controls and wiring.

### 7. Record opt-in correction feedback

Store a local training record only when the user explicitly enables it. Each record contains:

- immutable input identity and stage input coordinates;
- exact implementation, model version, and weight hash;
- automatic proposal and diagnostics;
- final user parameters;
- accept, reject, forced, edited, or off decision;
- before/after non-OCR quality evidence.

Do not store recognized text, upload records, or use them for training implicitly. Provide an
inspectable export and deletion path.

Acceptance: a correction record is reproducible against its exact input/model identities and can be
removed without affecting the scanning session.

### 8. Learn the decision policy before image-model fine-tuning

- First calibrate or train the gate that decides whether a model should run and which accepted
  candidate should win. Manual `Off`, forced, accept, and edit decisions are direct supervision for
  this problem.
- Later use corrected quadrilaterals as boundary/perspective supervision.
- Use user wave curves as residual-dewarp supervision only after their coordinate transforms and
  model provenance are reproducible.
- Treat lighting edits initially as method/strength preferences rather than ground truth for a new
  image-generation model.
- Never update production weights online after each page. Curate, validate, and version offline
  datasets and model releases to prevent label noise, overfitting, and catastrophic forgetting.

Acceptance: training is a separate, opt-in offline workflow with held-out evaluation and a model
rollback path. Collecting correction data does not change runtime output.

## Suggested implementation commits

1. `feat: persist editable perspective and stage decisions`
2. `feat: add horizontal automatic stage workspace`
3. `feat: skip unnecessary geometry model inference`
4. `feat: gate clipping edge loss and table regressions`
5. `test: cover editable stage workflow and recipe replay`
6. `feat: record opt-in manual correction feedback`

## Completion criteria

A newly imported page receives automatic perspective, wave, and lighting decisions without
unnecessary model inference. An operator can bypass or edit every stage without leaving the
workspace. Downstream previews update without destroying the source or unrelated accepted work.
Content clipping and table regressions are rejected in production and covered by the same benchmark
metrics. Correction data, when explicitly enabled, is reproducible and suitable for later offline
gate or model training.
