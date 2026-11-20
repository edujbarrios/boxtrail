# Copyright (C) 2026 Eduardo J. Barrios
# SPDX-License-Identifier: AGPL-3.0-only
"""Standard-library tests for BoxTrail geometry."""

import json
import math
import unittest
from dataclasses import FrozenInstanceError

from boxtrail import LetterboxTransform, letterbox


class LetterboxTests(unittest.TestCase):
    def test_example_from_readme(self):
        t = letterbox((1920, 1080), (640, 640))
        self.assertEqual(t.resized_size, (640, 360))
        self.assertEqual(t.padding, (0, 140, 0, 140))
        self.assertEqual(t.to_source((128, 208, 400, 352)), (384.0, 204.0, 1200.0, 636.0))

    def test_forward_mapping(self):
        t = letterbox((1920, 1080), (640, 640))
        self.assertEqual(t.to_target((384, 204, 1200, 636)), (128.0, 208.0, 400.0, 352.0))

    def test_content_box(self):
        self.assertEqual(letterbox((1920, 1080), (640, 640)).content_box, (0.0, 140.0, 640.0, 500.0))

    def test_non_square_resize_rounding(self):
        t = letterbox((100, 101), (640, 640))
        self.assertNotEqual(t.scale[0], t.scale[1])
        box = (10.5, 20.25, 80.75, 90.5)
        got = t.to_source(t.to_target(box))
        for actual, expected in zip(got, box):
            self.assertAlmostEqual(actual, expected, places=10)

    def test_round_trip_full_frame(self):
        for source, target in [((1, 1), (1, 1)), ((9, 17), (640, 640)), ((1920, 1080), (641, 641)), ((1920, 1080), (256, 128))]:
            with self.subTest(source=source, target=target):
                t = letterbox(source, target)
                result = t.to_source(t.to_target((0, 0, source[0], source[1])))
                self.assertAlmostEqual(result[0], 0)
                self.assertAlmostEqual(result[1], 0)
                self.assertAlmostEqual(result[2], source[0])
                self.assertAlmostEqual(result[3], source[1])

    def test_odd_padding_remainder_goes_to_bottom(self):
        t = letterbox((6, 4), (10, 10))
        self.assertEqual(t.resized_size, (10, 7))
        self.assertEqual(t.padding, (0, 1, 0, 2))

    def test_no_scale_up(self):
        t = letterbox((20, 10), (100, 100), scale_up=False)
        self.assertEqual(t.resized_size, (20, 10))
        self.assertEqual(t.padding, (40, 45, 40, 45))

    def test_top_left_alignment(self):
        t = letterbox((20, 10), (100, 100), scale_up=False, center=False)
        self.assertEqual(t.padding, (0, 0, 80, 90))

    def test_source_out_of_bounds_default_raises(self):
        with self.assertRaisesRegex(ValueError, "source image"):
            letterbox((10, 10), (20, 20)).to_target((-1, 0, 5, 5))

    def test_source_out_of_bounds_can_clip(self):
        t = letterbox((10, 10), (20, 20))
        self.assertEqual(t.to_target((-1, 0, 5, 5), clip=True), (0.0, 0.0, 10.0, 10.0))

    def test_target_padding_out_of_bounds_raises(self):
        t = letterbox((1920, 1080), (640, 640))
        with self.assertRaisesRegex(ValueError, "target image content"):
            t.to_source((20, 100, 100, 200))

    def test_target_padding_can_clip(self):
        t = letterbox((1920, 1080), (640, 640))
        self.assertEqual(t.to_source((20, 100, 100, 200), clip=True), (60.0, 0.0, 300.0, 180.0))

    def test_padding_only_box_cannot_map(self):
        t = letterbox((1920, 1080), (640, 640))
        with self.assertRaisesRegex(ValueError, "does not intersect"):
            t.to_source((20, 20, 100, 100), clip=True)

    def test_invalid_sizes(self):
        for invalid in [(0, 10), (10, -1), (True, 5), (20.5, 5), (10,), "10x10"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                letterbox(invalid, (10, 10))
        with self.assertRaises(ValueError):
            letterbox((10, 10), (0, 10))

    def test_invalid_boxes(self):
        t = letterbox((10, 10), (10, 10))
        for invalid in [(3, 3, 3, 7), (4, 2, 3, 8), (0, 0, math.nan, 3), (0, 0, math.inf, 4), (True, 0, 5, 5), (0, 0, 1), "0000"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                t.to_source(invalid)

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            letterbox((10, 10), (20, 20), center="yes")
        with self.assertRaises(ValueError):
            letterbox((10, 10), (20, 20), scale_up=1)

    def test_as_dict_json_serializable(self):
        result = json.loads(json.dumps(letterbox((6, 4), (10, 10)).as_dict()))
        self.assertEqual(result["padding"], [0, 1, 0, 2])
        self.assertEqual(result["source_size"], [6, 4])

    def test_dataclass_is_immutable(self):
        t = letterbox((10, 10), (20, 20))
        with self.assertRaises(FrozenInstanceError):
            t.padding = (1, 1, 1, 1)

    def test_invalid_direct_construction(self):
        with self.assertRaises(ValueError):
            LetterboxTransform((10, 10), (20, 20), (10, 10), (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
