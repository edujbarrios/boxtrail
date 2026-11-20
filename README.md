# BoxTrail

**A small, auditable geometry library for mapping bounding boxes between original images and letterboxed model inputs.**

[![PyPI](https://img.shields.io/pypi/v/boxtrail?label=PyPI&logo=pypi)](https://pypi.org/project/boxtrail/)
[![Supported Python](https://img.shields.io/pypi/pyversions/boxtrail?label=Python)](https://pypi.org/project/boxtrail/)
[![CI](https://github.com/edujbarrios/boxtrail/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/boxtrail/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/license-AGPL--3.0--only-blue)](LICENSE)

BoxTrail handles **coordinate geometry only**. It does not resize image pixels, draw boxes, run inference, or depend on NumPy, PyTorch, or OpenCV. It is designed for applications that already have a preprocessing pipeline but need reliable, inspectable mapping between coordinate spaces.

## Installation

```bash
pip install boxtrail
```

Requires Python 3.10+. There are **no required runtime dependencies**. NumPy arrays are accepted as inputs when NumPy is installed, but BoxTrail never imports NumPy to operate.

## Quick start

```python
from boxtrail import letterbox

transform = letterbox(source=(1920, 1080), target=(640, 640))

# A model prediction in the 640 x 640 coordinate space:
prediction = (128, 208, 400, 352)

print(transform.to_source(prediction))
# (384.0, 204.0, 1200.0, 636.0)

# Map an original-image box into the model's coordinate space:
print(transform.to_target((384, 204, 1200, 636)))
# (128.0, 208.0, 400.0, 352.0)
```

The operation above describes a 1920 × 1080 image resized to 640 × 360 and padded vertically with 140 pixels on each side. BoxTrail calculates **actual per-axis scale from the integer resized dimensions**, rather than assuming the rounded resize has exactly uniform scale.

## Use actual preprocessing geometry

**Recommended for production pipelines:** record the resize and padding values that were really applied, then construct the transformation explicitly.

```python
from boxtrail import from_geometry

transform = from_geometry(
    source=(1920, 1080),     # Original width, height
    resized=(640, 360),      # Actual resized image, before padding
    padding=(0, 140, 0, 140) # Left, top, right, bottom
)

print(transform.target_size)     # (640, 640)
print(transform.to_source((128, 208, 400, 352)))
# (384.0, 204.0, 1200.0, 636.0)
```

This avoids assumptions about which library computed the resize, how it rounded dimensions, or how it divided padding. It is the caller's responsibility to provide correct measurements; BoxTrail does not inspect image pixels.

### Transform multiple detections

```python
boxes = [
    (128, 208, 400, 352),
    (0, 140, 640, 500),
]

original_boxes = transform.to_source_many(boxes)
# [(384.0, 204.0, 1200.0, 636.0), (0.0, 0.0, 1920.0, 1080.0)]

# The opposite direction is also supported.
model_boxes = transform.to_target_many(original_boxes)
```

Both batch methods preserve input order and return a list of `xyxy` tuples. An invalid box raises `ValueError` with the offending index, rather than silently dropping a prediction.

### NumPy interoperability (optional)

```python
import numpy as np
from boxtrail import letterbox

t = letterbox(np.array([1920, 1080]), np.array([640, 640]))
boxes = np.array([[128, 208, 400, 352]], dtype=np.float32)

result = t.to_source_many(boxes)
# [(384.0, 204.0, 1200.0, 636.0)]
```

NumPy is not a dependency of BoxTrail and is not needed for regular Python lists and tuples. Batch outputs are Python lists of tuples (not `ndarray` objects); use `np.asarray(result)` if an array is required.

## Rules and error behavior

- **Image sizes:** `(width, height)` in positive integer pixels.
- **Boxes:** `(x1, y1, x2, y2)` in continuous pixel-edge coordinates. A box must have positive width and height.
- **Padding:** `(left, top, right, bottom)` in nonnegative integer pixels.
- **Strict bounds by default:** boxes crossing outside the image or padded content raise `ValueError`.
- **Optional clipping:** pass `clip=True` to intersect a box with the valid image region. A box entirely outside the valid region still raises.
- **Round-trip:** coordinates can be recovered up to floating-point precision when no clipping occurs. Image pixels or image interpolation are never recovered.
- **Inspection:** `transform.resized_size`, `transform.padding`, `transform.scale`, `transform.content_box`, and `transform.as_dict()` expose the geometry.

The convenience `letterbox()` function assumes an aspect-ratio-preserving contain resize and Python's `round()` on each resized dimension (ties to even). Centered padding puts any odd-pixel remainder on the right/bottom. Set `scale_up=False` to avoid enlarging small images or `center=False` to anchor the image at the top left.

**Compatibility warning:** some preprocessors use stride-aligned padding, alternative rounding rules, automatic crop or stretch-to-fit behavior. `letterbox()` cannot infer those operations from the source and target sizes alone. For real-world inference pipelines, prefer `from_geometry()` using the actual recorded values. For this release, BoxTrail does not support rotated boxes, polygons, `xywh`, normalized coordinates or arbitrary image transformations.

## Development and tests

```bash
git clone https://github.com/edujbarrios/boxtrail.git
cd boxtrail
python -m pip install -e .
python -m unittest discover -s tests -v
```

CI covers Python 3.10–3.14 and tests optional NumPy compatibility on Python 3.11. No binary extensions or model downloads are required.

## License and attribution

Copyright © 2026 **Eduardo J. Barrios** ([@edujbarrios](https://github.com/edujbarrios)).

BoxTrail is free and open-source software under the **GNU Affero General Public License, version 3 only (AGPL-3.0-only)**. Copies and covered modifications must preserve the required copyright and license notices; the AGPL also imposes source-availability requirements for certain network-based uses. Commercial use is allowed under the terms of the license. Read [LICENSE](LICENSE) and [NOTICE](NOTICE) before redistributing or integrating the software.

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). Release process: [docs/RELEASING.md](docs/RELEASING.md).
