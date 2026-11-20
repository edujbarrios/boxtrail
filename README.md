# BoxTrail

**Move bounding boxes between original and letterboxed image coordinates — without guessing the padding.**

[![License: AGPL-3.0-only](https://img.shields.io/badge/license-AGPL--3.0--only-blue)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Runtime dependencies: 0](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen)

BoxTrail is a tiny, **pure-Python, zero-runtime-dependency** library for reconstructing bounding-box coordinates after a resize-and-pad (**letterbox**) operation. It exposes the actual integer resize geometry, the padding and the per-axis scales so results are inspectable and reproducible.

> **Status: v0.1.0 — source ready; not published on PyPI.** No release or upload workflow is configured.

## Why it exists

Computer-vision models often take square inputs. A 1920 × 1080 image may be resized to 640 × 360 and padded to 640 × 640. Predictions in the 640 × 640 canvas must be mapped back to the original image. Getting padding order, rounding, or width/height conventions wrong can shift boxes. BoxTrail handles this one task and deliberately does **not** resize images, run models, or provide annotation tooling.

## Install from source (until a release is approved)

```bash
git clone https://github.com/edujbarrios/boxtrail.git
cd boxtrail
python -m pip install -e .
```

## Quick start

```python
from boxtrail import letterbox

transform = letterbox(source=(1920, 1080), target=(640, 640))

# A detection in the 640 x 640 model input (xyxy pixel-edge coordinates)
box = (128, 208, 400, 352)
print(transform.to_source(box))
# (384.0, 204.0, 1200.0, 636.0)

# Move a source-space box into the model input canvas
print(transform.to_target((384, 204, 1200, 636)))
# (128.0, 208.0, 400.0, 352.0)
```

## Inspect the transform

```python
print(transform.resized_size)  # (640, 360)
print(transform.padding)       # (0, 140, 0, 140), left/top/right/bottom
print(transform.scale)         # (1/3, 1/3) approximately
print(transform.content_box)   # (0.0, 140.0, 640.0, 500.0)
print(transform.as_dict())     # JSON-serializable geometry
```

### When detections overlap padding

By default, boxes outside the valid image/content region raise `ValueError`. Set `clip=True` to intersect them with the valid region. A box entirely inside padding has no meaningful source coordinates and raises even with clipping.

```python
print(transform.to_source((20, 100, 100, 200), clip=True))
# (60.0, 0.0, 300.0, 180.0)
```

### Resize behavior

```python
# Keep a small image at its original size, pad to the requested canvas.
t = letterbox((20, 10), (100, 100), scale_up=False)
print(t.padding)   # (40, 45, 40, 45)

# Or anchor it to the upper-left corner instead of centering it.
t = letterbox((20, 10), (100, 100), scale_up=False, center=False)
print(t.padding)   # (0, 0, 80, 90)
```

## Conventions and guarantees

- **Sizes:** `(width, height)`; **boxes:** `(x1, y1, x2, y2)` with continuous pixel-edge coordinates and strictly positive width/height.
- **Padding:** `(left, top, right, bottom)` in integer pixels.
- **Resize:** preserve aspect ratio; calculate `ratio = min(target_w/source_w, target_h/source_h)`, capped at `1` when `scale_up=False`. Each resized dimension uses Python's `round()` (ties to even). Any odd padding remainder goes to right/bottom when centered.
- **Actual scale:** computed separately for x and y from the *rounded* resized dimensions; this matters when integer rounding slightly changes the aspect ratio.
- **Bounds:** strict by default with a tiny floating-point tolerance; use `clip=True` for an explicit intersection. Non-finite, reversed, zero-area or non-intersecting boxes raise `ValueError`.
- **Round-trip:** coordinate mapping is invertible within the unpadded image content up to floating-point precision; lost image pixels or image interpolation cannot be reversed.

**Important:** BoxTrail does not operate on the image itself. The actual model preprocessing **must use exactly the same resize dimensions, rounding and padding convention**. If your pipeline uses stride alignment, a different rounding rule, cropping, stretched resize, or automatic padding, the returned geometry does not describe that pipeline. In v0.1.0 only the `xyxy` format and `letterbox()` are supported.

## Run tests

```bash
python -m unittest discover -s tests -v
```

There are no third-party runtime or test dependencies. To build distributables **locally**, see [docs/RELEASING.md](docs/RELEASING.md); publication is intentionally not configured.

## License, copyleft and attribution

**GNU Affero General Public License v3.0 only (`AGPL-3.0-only`).** Copyright (C) 2026 **Eduardo J. Barrios** ([@edujbarrios](https://github.com/edujbarrios)). See [LICENSE](LICENSE) and [NOTICE](NOTICE). Preserve applicable copyright, licensing and attribution notices when conveying covered code. Modified versions and certain network-service uses carry source-code obligations under the AGPL. Commercial use is **not** prohibited. You should review the license before integrating the library into proprietary projects.

Contributions are welcome under the same license; see [CONTRIBUTING.md](CONTRIBUTING.md).
