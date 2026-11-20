# Changelog

## 0.2.0 — Current release

- Added `from_geometry()` for constructing transforms from the measured resized dimensions and padding.
- Added `to_source_many()` and `to_target_many()`, preserving input order and identifying invalid box indices.
- Added optional NumPy input compatibility for sizes, bounding boxes and batches, without runtime dependencies.
- Added a multi-Python-version CI matrix and optional NumPy interoperability tests.
- Rewrote the README with an emphasis on geometry accuracy, explicit conventions and limitations.
- Retained AGPL-3.0-only copyright and author attribution.

## 0.1.0 — Initial release

- Implemented letterbox geometry and reversible `xyxy` transforms.
- Added validation, clipping behavior, introspection and packaging.
- Published under AGPL-3.0-only with author notices.
