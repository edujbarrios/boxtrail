# Copyright (C) 2026 Eduardo J. Barrios
# SPDX-License-Identifier: AGPL-3.0-only
"""Invertible bounding-box geometry for letterboxed images.

BoxTrail transforms *coordinates*, not image pixels. It deliberately avoids
image-processing and machine-learning dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from operator import index as integer_index
from typing import Iterable, Sequence

Box = tuple[float, float, float, float]
Size = tuple[int, int]
Padding = tuple[int, int, int, int]


def _items(value: object, count: int, name: str) -> tuple:
    """Read a fixed-length indexable input, including an optional NumPy array.

    No NumPy import is required. A one-dimensional ndarray exposes len() and
    integer indexing but is not a collections.abc.Sequence.
    """
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {count} values")
    try:
        if len(value) != count:
            raise ValueError(f"{name} must contain {count} values")
        return tuple(value[i] for i in range(count))
    except (TypeError, IndexError, KeyError) as exc:
        raise ValueError(f"{name} must contain {count} indexable values") from exc


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = integer_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _size(value: Sequence[int], name: str) -> Size:
    width, height = _items(value, 2, name)
    return (_integer(width, f"{name} width", minimum=1),
            _integer(height, f"{name} height", minimum=1))


def _padding(value: Sequence[int]) -> Padding:
    left, top, right, bottom = _items(value, 4, "padding")
    return tuple(_integer(v, "padding coordinate") for v in (left, top, right, bottom))


def _box(value: Sequence[Real]) -> Box:
    parts = _items(value, 4, "box")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in parts):
        raise ValueError("box coordinates must be finite real numbers")
    coords = tuple(float(v) for v in parts)
    if not all(isfinite(v) for v in coords):
        raise ValueError("box coordinates must be finite real numbers")
    x1, y1, x2, y2 = coords
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box must have positive width and height (xyxy)")
    return (x1, y1, x2, y2)


def _bounded(box: Box, bounds: Box, *, clip: bool, label: str) -> Box:
    x1, y1, x2, y2 = box
    left, top, right, bottom = bounds
    if clip:
        x1, y1 = max(x1, left), max(y1, top)
        x2, y2 = min(x2, right), min(y2, bottom)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"box does not intersect {label}")
    else:
        # Round-trip float operations can introduce sub-pixel boundary noise.
        eps = 1e-9 * max(1.0, right - left, bottom - top)
        if x1 < left - eps or y1 < top - eps or x2 > right + eps or y2 > bottom + eps:
            raise ValueError(f"box extends outside {label}; use clip=True to clip")
        x1, y1 = max(x1, left), max(y1, top)
        x2, y2 = min(x2, right), min(y2, bottom)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"box has no positive area inside {label}")
    return (x1, y1, x2, y2)


_FORMATS = frozenset({"xyxy", "xywh", "cxcywh"})


def _validate_format(box_format: str, name: str) -> None:
    if not isinstance(box_format, str) or box_format not in _FORMATS:
        raise ValueError(f"{name} must be 'xyxy', 'xywh', or 'cxcywh'")


def _validate_normalized(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _decode_box(
    box: Sequence[Real], *, box_format: str, normalized: bool, size: Size
) -> Box:
    """Convert a supported box into absolute xyxy pixel-edge coordinates."""
    _validate_format(box_format, "input_format")
    _validate_normalized(normalized, "input_normalized")
    parts = _items(box, 4, "box")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in parts):
        raise ValueError("box coordinates must be finite real numbers")
    values = tuple(float(v) for v in parts)
    if not all(isfinite(v) for v in values):
        raise ValueError("box coordinates must be finite real numbers")
    a, b, c, d = values
    if normalized:
        width, height = size
        a, b, c, d = a * width, b * height, c * width, d * height
    if box_format == "xyxy":
        xyxy = (a, b, c, d)
    elif box_format == "xywh":
        xyxy = (a, b, a + c, b + d)
    else:
        xyxy = (a - c / 2, b - d / 2, a + c / 2, b + d / 2)
    return _box(xyxy)


def _encode_box(
    box: Box, *, box_format: str, normalized: bool, size: Size
) -> Box:
    """Convert an absolute xyxy box into a requested output format."""
    _validate_format(box_format, "output_format")
    _validate_normalized(normalized, "output_normalized")
    x1, y1, x2, y2 = box
    if box_format == "xyxy":
        result = (x1, y1, x2, y2)
    elif box_format == "xywh":
        result = (x1, y1, x2 - x1, y2 - y1)
    else:
        result = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    if normalized:
        width, height = size
        result = (result[0] / width, result[1] / height,
                  result[2] / width, result[3] / height)
    return result



@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Geometry of a resize-and-pad operation.

    Dimensions use (width, height); padding is (left, top, right, bottom).
    Box coordinates are continuous pixel edges in xyxy order, not indices.
    """

    source_size: Size
    target_size: Size
    resized_size: Size
    padding: Padding

    def __post_init__(self) -> None:
        source = _size(self.source_size, "source_size")
        target = _size(self.target_size, "target_size")
        resized = _size(self.resized_size, "resized_size")
        p = _padding(self.padding)
        if resized[0] + p[0] + p[2] != target[0] or resized[1] + p[1] + p[3] != target[1]:
            raise ValueError("resized_size and padding must exactly fill target_size")
        if resized[0] > target[0] or resized[1] > target[1]:
            raise ValueError("resized_size cannot exceed target_size")
        # Defensive normalization for callers who instantiate this class directly.
        object.__setattr__(self, "source_size", source)
        object.__setattr__(self, "target_size", target)
        object.__setattr__(self, "resized_size", resized)
        object.__setattr__(self, "padding", p)

    @property
    def scale(self) -> tuple[float, float]:
        """Actual per-axis scale after integer resize rounding (x, y)."""
        return (
            self.resized_size[0] / self.source_size[0],
            self.resized_size[1] / self.source_size[1],
        )

    @property
    def content_box(self) -> Box:
        """Non-padding region in target pixel-edge coordinates."""
        left, top, _, _ = self.padding
        return (float(left), float(top), float(left + self.resized_size[0]), float(top + self.resized_size[1]))

    def to_target(
        self,
        box: Sequence[Real],
        *,
        clip: bool = False,
        input_format: str = "xyxy",
        output_format: str = "xyxy",
        input_normalized: bool = False,
        output_normalized: bool = False,
    ) -> Box:
        """Map a box from the source image to the letterboxed target canvas.

        Input and output formats may be 'xyxy' (corner edges), 'xywh'
        (top-left, width, height) or 'cxcywh' (center, width, height).
        Normalized values are relative to their OWN coordinate space:
        source dimensions on input, complete target canvas on output.
        By default invalid out-of-bounds boxes raise ValueError; clip=True
        intersects the input with the valid source image.
        """
        source_width, source_height = self.source_size
        pixel_box = _decode_box(
            box, box_format=input_format, normalized=input_normalized,
            size=self.source_size,
        )
        x1, y1, x2, y2 = _bounded(
            pixel_box, (0.0, 0.0, float(source_width), float(source_height)),
            clip=clip, label="source image",
        )
        sx, sy = self.scale
        left, top, _, _ = self.padding
        mapped = (x1 * sx + left, y1 * sy + top,
                  x2 * sx + left, y2 * sy + top)
        return _encode_box(
            mapped, box_format=output_format, normalized=output_normalized,
            size=self.target_size,
        )

    def to_source(
        self,
        box: Sequence[Real],
        *,
        clip: bool = False,
        input_format: str = "xyxy",
        output_format: str = "xyxy",
        input_normalized: bool = False,
        output_normalized: bool = False,
    ) -> Box:
        """Map a box from the target canvas back to the original source.

        Normalized inputs use FULL target canvas dimensions, including
        padding; normalized outputs use original source dimensions.
        Boxes that overlap padding raise unless clip=True. A box fully in
        padding cannot be mapped and always raises ValueError.
        """
        pixel_box = _decode_box(
            box, box_format=input_format, normalized=input_normalized,
            size=self.target_size,
        )
        x1, y1, x2, y2 = _bounded(
            pixel_box, self.content_box, clip=clip,
            label="target image content",
        )
        sx, sy = self.scale
        left, top, _, _ = self.padding
        source_width, source_height = self.source_size
        mapped = (
            min(float(source_width), max(0.0, (x1 - left) / sx)),
            min(float(source_height), max(0.0, (y1 - top) / sy)),
            min(float(source_width), max(0.0, (x2 - left) / sx)),
            min(float(source_height), max(0.0, (y2 - top) / sy)),
        )
        return _encode_box(
            mapped, box_format=output_format, normalized=output_normalized,
            size=self.source_size,
        )

    def to_target_many(
        self,
        boxes: Iterable[Sequence[Real]],
        *,
        clip: bool = False,
        input_format: str = "xyxy",
        output_format: str = "xyxy",
        input_normalized: bool = False,
        output_normalized: bool = False,
    ) -> list[Box]:
        """Map multiple source-space boxes and preserve input order."""
        return self._map_many(
            boxes, source=False, clip=clip,
            input_format=input_format, output_format=output_format,
            input_normalized=input_normalized,
            output_normalized=output_normalized,
        )

    def to_source_many(
        self,
        boxes: Iterable[Sequence[Real]],
        *,
        clip: bool = False,
        input_format: str = "xyxy",
        output_format: str = "xyxy",
        input_normalized: bool = False,
        output_normalized: bool = False,
    ) -> list[Box]:
        """Map multiple target-space boxes and preserve input order."""
        return self._map_many(
            boxes, source=True, clip=clip,
            input_format=input_format, output_format=output_format,
            input_normalized=input_normalized,
            output_normalized=output_normalized,
        )

    def _map_many(
        self,
        boxes: Iterable[Sequence[Real]],
        *,
        source: bool,
        clip: bool,
        input_format: str,
        output_format: str,
        input_normalized: bool,
        output_normalized: bool,
    ) -> list[Box]:
        _validate_format(input_format, "input_format")
        _validate_format(output_format, "output_format")
        _validate_normalized(input_normalized, "input_normalized")
        _validate_normalized(output_normalized, "output_normalized")
        try:
            iterator = iter(boxes)
        except TypeError as exc:
            raise ValueError("boxes must be an iterable of boxes") from exc
        convert = self.to_source if source else self.to_target
        converted: list[Box] = []
        for idx, box in enumerate(iterator):
            try:
                converted.append(
                    convert(
                        box, clip=clip,
                        input_format=input_format, output_format=output_format,
                        input_normalized=input_normalized,
                        output_normalized=output_normalized,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"box at index {idx}: {exc}") from exc
        return converted

    def as_dict(self) -> dict[str, object]:
        """Return the inspectable transform as JSON-serializable geometry."""
        return {
            "source_size": self.source_size,
            "target_size": self.target_size,
            "resized_size": self.resized_size,
            "padding": self.padding,
            "scale": self.scale,
            "content_box": self.content_box,
        }


def letterbox(
    source: Sequence[int],
    target: Sequence[int],
    *,
    scale_up: bool = True,
    center: bool = True,
) -> LetterboxTransform:
    """Describe a standard contain-resize and pad operation.

    Args:
        source: Original image size (width, height).
        target: Canvas size after resize/padding (width, height).
        scale_up: When false, do not enlarge images that already fit.
        center: When true, split extra pixels symmetrically with remainder
            on the right/bottom; otherwise align content to the top-left.

    The resized integer dimensions use Python's ``round`` (ties to even).
    Use this only when the actual preprocessing uses the same resize and
    padding policy; BoxTrail does not manipulate pixels.
    """
    src_w, src_h = _size(source, "source")
    tgt_w, tgt_h = _size(target, "target")
    if not isinstance(scale_up, bool) or not isinstance(center, bool):
        raise ValueError("scale_up and center must be booleans")
    ratio = min(tgt_w / src_w, tgt_h / src_h)
    if not scale_up:
        ratio = min(ratio, 1.0)
    new_w = max(1, min(tgt_w, round(src_w * ratio)))
    new_h = max(1, min(tgt_h, round(src_h * ratio)))
    extra_w, extra_h = tgt_w - new_w, tgt_h - new_h
    left = extra_w // 2 if center else 0
    top = extra_h // 2 if center else 0
    padding = (left, top, extra_w - left, extra_h - top)
    return LetterboxTransform((src_w, src_h), (tgt_w, tgt_h), (new_w, new_h), padding)


def from_geometry(
    *,
    source: Sequence[int],
    resized: Sequence[int],
    padding: Sequence[int],
) -> LetterboxTransform:
    """Construct geometry from the *actual* preprocessing operation.

    Args:
        source: Original image (width, height).
        resized: Integer image size after resizing, before padding.
        padding: Measured (left, top, right, bottom) integer padding.

    Unlike letterbox(), this does not guess the resize rounding or padding
    policy. The target canvas is derived from resized size plus padding.
    Images must already have been preprocessed by the caller.
    """
    src = _size(source, "source")
    resized_dims = _size(resized, "resized")
    pad = _padding(padding)
    target = (resized_dims[0] + pad[0] + pad[2],
              resized_dims[1] + pad[1] + pad[3])
    return LetterboxTransform(src, target, resized_dims, pad)
