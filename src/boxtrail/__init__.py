# Copyright (C) 2026 Eduardo J. Barrios
# SPDX-License-Identifier: AGPL-3.0-only
"""BoxTrail: inspectable bounding-box coordinates through letterboxing."""

from ._core import Box, LetterboxTransform, letterbox

__version__ = "0.1.0"
__all__ = ["Box", "LetterboxTransform", "letterbox"]
