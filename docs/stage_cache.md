# Processing stage cache

GUI preview/apply and optional headless conversion use the same bounded lossless disk cache after
page-boundary detection.

Cached stages are:

```text
source pixels -> orientation -> deskew -> dewarp -> cleanup -> layout
```

Each key is SHA-256 over the upstream key, stage name/version, and normalized settings. Therefore:

- changing page pixels invalidates every stage;
- changing dewarp keeps orientation/deskew but invalidates dewarp and everything after it;
- changing binarization or despeckle keeps geometry but invalidates cleanup/layout;
- changing margins or alignment invalidates layout only.

Disabled identity stages participate in downstream fingerprints but do not duplicate their image
on disk. Images are stored as lossless PNG with typed diagnostics in JSON. A new entry is published
through temporary files; incomplete or corrupt pairs are treated as misses and removed. Cache I/O
failure does not fail page processing.

The GUI cache lives under the UniScan state directory, is limited to 512 MiB and 256 entries, and
can be cleared from Processing → Advanced → Clear cache. Headless conversion enables persistence
only when `--stage-cache-dir` is provided; `--stage-cache-max-mb` controls its size. JSON reports
include per-page hit stages/timings and aggregate hit/miss/write/eviction counts.

Boundary detection is not cached yet because it can emit zero, one, or two pages and belongs to the
acquisition pipeline. It is the next cache extension after per-page processing settings are moved
into the session model.
