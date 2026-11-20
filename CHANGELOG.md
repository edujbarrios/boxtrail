# Changelog

## 0.3.0 — Bounding-box formats and normalized coordinates

- Added `input_format` and `output_format` options: `xyxy`, `xywh`, and `cxcywh`.
- Added `input_normalized` and `output_normalized` options, relative to the appropriate image/canvas dimensions.
- Extended batch conversions to support the same format and normalization options.
- Added regression tests for padding-aware YOLO-style coordinates, round trips, clipping, formats, and validation.
- Kept every existing API default unchanged and retained zero runtime dependencies.

## 0.2.0 — Recorded geometry and batch conversions

- Added `from_geometry()` for constructing transforms from the measured resized dimensions and padding.
- Added `to_source_many()` and `to_target_many()`, preserving input order and identifying invalid box indices.
- Added optional NumPy input compatibility for sizes, bounding boxes and batches, without runtime dependencies.
- Added a multi-Python-version CI matrix and optional NumPy interoperability tests.
- Retained AGPL-3.0-only copyright and author attribution.

## 0.1.0 — Initial release

- Implemented letterbox geometry and reversible `xyxy` transforms.
- Added validation, clipping behavior, introspection and packaging.
- Published under AGPL-3.0-only with author notices.
