# BoxTrail

**Lightweight, auditable bounding-box coordinate transformations for letterboxed images.**

[![PyPI](https://img.shields.io/pypi/v/boxtrail?label=PyPI&logo=pypi)](https://pypi.org/project/boxtrail/)
[![Python](https://img.shields.io/pypi/pyversions/boxtrail)](https://pypi.org/project/boxtrail/)
[![CI](https://github.com/edujbarrios/boxtrail/actions/workflows/ci.yml/badge.svg)](https://github.com/edujbarrios/boxtrail/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/license-AGPL--3.0--only-blue)](LICENSE)

BoxTrail maps bounding boxes between original-image and model-input coordinates, using explicit resize and padding geometry. It supports `xyxy`, `xywh`, and `cxcywh` representations, absolute or normalized coordinates, and lists of detections. **No runtime dependencies** are required.

BoxTrail **does not resize image pixels**, run inference, or draw annotations. It works alongside your existing preprocessing pipeline.

## Install

```bash
pip install boxtrail
```

Python 3.10 or newer is required. NumPy arrays can be passed as inputs if NumPy is installed, but NumPy is **not** a dependency.

## Basic usage

```python
from boxtrail import letterbox

t = letterbox(source=(1920, 1080), target=(640, 640))

# Absolute xyxy detection in model coordinates:
print(t.to_source((128, 208, 400, 352)))
# (384.0, 204.0, 1200.0, 636.0)

# Convert from original image coordinates to model coordinates:
print(t.to_target((384, 204, 1200, 636)))
# (128.0, 208.0, 400.0, 352.0)
```

For this example, a 1920 × 1080 image is resized to 640 × 360, then padded 140 pixels on the top and bottom.

## Work with YOLO-style normalized boxes

A YOLO-style detection often uses normalized `cxcywh`: center x, center y, width and height, each relative to the **whole model input canvas, including padding**.

```python
from boxtrail import letterbox

t = letterbox((1920, 1080), (640, 640))

# Center, width and height, normalized to 640 x 640.
yolo_box = (0.25, 0.4375, 0.425, 0.225)

# Convert into pixel-edge xyxy in the original 1920 x 1080 image:
original = t.to_source(
    yolo_box,
    input_format="cxcywh",
    input_normalized=True,
)
print(original)
# (72.0, 216.0, 888.0, 648.0)
```

These conversions are explicit: by default the output remains absolute `xyxy`. If you need a normalized output too, specify `output_format` and `output_normalized`.

## Use actual preprocessing geometry

For production pipelines, prefer `from_geometry()` with the dimensions and padding recorded **after actual preprocessing**, rather than predicting them from the image sizes.

```python
from boxtrail import from_geometry

t = from_geometry(
    source=(100, 50),
    resized=(200, 100),
    padding=(10, 20, 30, 40),  # left, top, right, bottom
)

# Absolute top-left x,y,width,height -> original-image xyxy:
print(t.to_source((30, 30, 60, 40), input_format="xywh"))
# (10.0, 5.0, 40.0, 25.0)

# Original-image xyxy -> normalized YOLO-style center coordinates:
print(t.to_target(
    (10, 5, 40, 25),
    output_format="cxcywh",
    output_normalized=True,
))
# (0.25, 0.3125, 0.25, 0.25)
```

Using `from_geometry()` avoids assumptions about rounding rules, padding distribution, and nonuniform integer scaling. You must supply measurements consistent with your actual preprocessing.

## Multiple boxes

```python
boxes = [
    (30, 30, 60, 40),
    (10, 20, 200, 100),
]

mapped = t.to_source_many(
    boxes,
    input_format="xywh",
    clip=True,
)

# Convert back to model coordinates using normalized cxcywh:
normalized = t.to_target_many(
    mapped,
    output_format="cxcywh",
    output_normalized=True,
)
```

The `_many` methods preserve input order and return `list[tuple[float, float, float, float]]`. If an input is invalid, `ValueError` identifies the first invalid index. The methods accept iterables and optional NumPy arrays shaped `(N, 4)`.

## Coordinate conventions

| Option | Input or output values |
|---|---|
| `xyxy` (default) | `(x1, y1, x2, y2)`: two corners |
| `xywh` | `(x, y, width, height)`: top-left corner and dimensions |
| `cxcywh` | `(center_x, center_y, width, height)` |
| `input_normalized=True` | Input values measured in fractions of the **input coordinate space** |
| `output_normalized=True` | Output values measured in fractions of the **output coordinate space** |

Normalization uses **source image size** for source-space boxes and **full target canvas size, including padding** for target-space boxes. This applies separately to both directions.

Dimensions are always `(width, height)`. Boxes have positive width and height; coordinates are continuous pixel **edges**, not indices. Input boxes crossing image bounds or padding raise `ValueError` by default. Pass `clip=True` to intersect the box with valid image content; a completely disjoint box still raises an error.

### Full API: optional keyword arguments

All `to_source()`, `to_target()`, `to_source_many()`, and `to_target_many()` methods accept:

```python
clip=False
input_format="xyxy"
output_format="xyxy"
input_normalized=False
output_normalized=False
```

The defaults preserve the API and output conventions of earlier BoxTrail versions. Output coordinates are floating-point numbers.

## Geometry, scope and limitations

- `letterbox()` describes aspect-ratio-preserving contain resizing with Python's `round()` for integer dimensions and centered padding (any odd remainder goes right/bottom). Use `scale_up=False` to avoid enlargement and `center=False` to anchor top-left.
- `from_geometry()` records exact integer `resized` dimensions and `(left, top, right, bottom)` padding, including asymmetric padding.
- Geometry inspection: `t.source_size`, `t.target_size`, `t.resized_size`, `t.padding`, `t.scale`, `t.content_box` and `t.as_dict()`.
- Coordinate mapping round-trips within floating-point precision when no clipping occurs. Clipped coordinates and image pixels cannot be reconstructed.
- **Not supported:** rotated boxes, polygons, perspective transforms, cropping, stride padding inferred automatically from dimensions, or conversion of the image pixels themselves.
- If your preprocessing uses different rounding/padding, `letterbox()` alone is not sufficient: use **measured** `from_geometry()`.

## Development and testing

```bash
git clone https://github.com/edujbarrios/boxtrail.git
cd boxtrail
python -m pip install -e .
python -m unittest discover -s tests -v
```

CI tests Python 3.10–3.14 and optional NumPy input interoperability. Automated PyPI publishing occurs only when both package version references are updated to a new version and merged into `main`; see [the release guide](docs/RELEASING.md).

## License and attribution

Copyright © 2026 **Eduardo J. Barrios** ([@edujbarrios](https://github.com/edujbarrios)).

BoxTrail is distributed under **GNU Affero General Public License v3.0 only (AGPL-3.0-only)**. Covered redistribution requires preservation of applicable copyright and license notices and compliance with AGPL copyleft requirements. Commercial use is allowed under the license's conditions. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions are welcome under the same license; see [CONTRIBUTING.md](CONTRIBUTING.md).
