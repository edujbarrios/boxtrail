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
from typing import Sequence

Box = tuple[float, float, float, float]
Size = tuple[int, int]
Padding = tuple[int, int, int, int]


def _size(value: Sequence[int], name: str) -> Size:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must be a (width, height) pair of positive integers")
    width, height = value
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ValueError(f"{name} must be a (width, height) pair of positive integers")
    return (width, height)


def _box(value: Sequence[Real]) -> Box:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("box must contain four xyxy coordinates")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in value):
        raise ValueError("box coordinates must be finite real numbers")
    coords = tuple(float(v) for v in value)
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
        p = self.padding
        if (not isinstance(p, tuple) or len(p) != 4 or
                any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in p)):
            raise ValueError("padding must be four non-negative integers")
        if resized[0] + p[0] + p[2] != target[0] or resized[1] + p[1] + p[3] != target[1]:
            raise ValueError("resized_size and padding must exactly fill target_size")
        if resized[0] > target[0] or resized[1] > target[1]:
            raise ValueError("resized_size cannot exceed target_size")
        # Defensive normalization for callers who instantiate this class directly.
        object.__setattr__(self, "source_size", source)
        object.__setattr__(self, "target_size", target)
        object.__setattr__(self, "resized_size", resized)

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

    def to_target(self, box: Sequence[Real], *, clip: bool = False) -> Box:
        """Map a source xyxy box to the letterboxed target coordinate system.

        By default, an out-of-bounds box raises ValueError. Set ``clip=True``
        to intersect it with the source image before the conversion.
        """
        source_width, source_height = self.source_size
        x1, y1, x2, y2 = _bounded(
            _box(box), (0.0, 0.0, float(source_width), float(source_height)),
            clip=clip, label="source image",
        )
        sx, sy = self.scale
        left, top, _, _ = self.padding
        return (x1 * sx + left, y1 * sy + top, x2 * sx + left, y2 * sy + top)

    def to_source(self, box: Sequence[Real], *, clip: bool = False) -> Box:
        """Map a target xyxy box back to the original source image.

        The valid target region excludes padding. For a detection overlapping
        padding, ``clip=True`` clips it to image content before conversion.
        A box entirely inside padding has no source box and raises ValueError.
        """
        x1, y1, x2, y2 = _bounded(
            _box(box), self.content_box, clip=clip, label="target image content",
        )
        sx, sy = self.scale
        left, top, _, _ = self.padding
        source_width, source_height = self.source_size
        # Clamp only tiny floating-point noise at the boundaries.
        return (
            min(float(source_width), max(0.0, (x1 - left) / sx)),
            min(float(source_height), max(0.0, (y1 - top) / sy)),
            min(float(source_width), max(0.0, (x2 - left) / sx)),
            min(float(source_height), max(0.0, (y2 - top) / sy)),
        )

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
