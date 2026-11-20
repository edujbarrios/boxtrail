# Copyright (C) 2026 Eduardo J. Barrios
# SPDX-License-Identifier: AGPL-3.0-only
"""Regression tests for bounding-box formats and normalized coordinates."""

import math
import random
import unittest

from boxtrail import from_geometry, letterbox


class FormatTests(unittest.TestCase):
    def setUp(self):
        # Width, height: source 100x50; target 240x160, with asymmetric padding.
        self.t = from_geometry(
            source=(100, 50), resized=(200, 100),
            padding=(10, 20, 30, 40),
        )
        self.source_xyxy = (10.0, 5.0, 40.0, 25.0)
        self.target_xyxy = (30.0, 30.0, 90.0, 70.0)

    def assertBoxClose(self, observed, expected):
        self.assertEqual(len(observed), 4)
        for got, want in zip(observed, expected):
            self.assertAlmostEqual(got, want, places=8)

    def test_original_xyxy_api_is_unchanged(self):
        self.assertEqual(self.t.to_target(self.source_xyxy), self.target_xyxy)
        self.assertEqual(self.t.to_source(self.target_xyxy), self.source_xyxy)

    def test_xywh_target_input(self):
        self.assertBoxClose(
            self.t.to_source((30, 30, 60, 40), input_format="xywh"),
            self.source_xyxy,
        )

    def test_cxcywh_target_input(self):
        self.assertBoxClose(
            self.t.to_source((60, 50, 60, 40), input_format="cxcywh"),
            self.source_xyxy,
        )

    def test_xywh_and_cxcywh_output(self):
        self.assertBoxClose(
            self.t.to_target(self.source_xyxy, output_format="xywh"),
            (30, 30, 60, 40),
        )
        self.assertBoxClose(
            self.t.to_target(self.source_xyxy, output_format="cxcywh"),
            (60, 50, 60, 40),
        )

    def test_normalized_target_input_includes_padding_in_canvas(self):
        self.assertBoxClose(
            self.t.to_source(
                (60 / 240, 50 / 160, 60 / 240, 40 / 160),
                input_format="cxcywh", input_normalized=True,
            ),
            self.source_xyxy,
        )

    def test_normalized_target_output(self):
        self.assertBoxClose(
            self.t.to_target(
                self.source_xyxy, output_format="cxcywh",
                output_normalized=True,
            ),
            (60 / 240, 50 / 160, 60 / 240, 40 / 160),
        )

    def test_normalized_source_input(self):
        self.assertBoxClose(
            self.t.to_target(
                (0.1, 0.1, 0.4, 0.5),
                input_format="xyxy", input_normalized=True,
            ),
            self.target_xyxy,
        )

    def test_normalized_source_output(self):
        self.assertBoxClose(
            self.t.to_source(
                self.target_xyxy,
                output_format="xywh", output_normalized=True,
            ),
            (0.1, 0.1, 0.3, 0.4),
        )

    def test_normalized_xywh_target_round_trip(self):
        got = self.t.to_source(
            (30 / 240, 30 / 160, 60 / 240, 40 / 160),
            input_format="xywh", input_normalized=True,
            output_format="cxcywh", output_normalized=True,
        )
        self.assertBoxClose(got, (0.25, 0.3, 0.3, 0.4))

    def test_batch_with_format_options(self):
        observed = self.t.to_source_many(
            [(60 / 240, 50 / 160, 60 / 240, 40 / 160)],
            input_format="cxcywh", input_normalized=True,
            output_format="xywh",
        )
        self.assertEqual(len(observed), 1)
        self.assertBoxClose(observed[0], (10, 5, 30, 20))
        again = self.t.to_target_many(
            observed, input_format="xywh",
            output_format="cxcywh", output_normalized=True,
        )
        self.assertBoxClose(
            again[0], (60 / 240, 50 / 160, 60 / 240, 40 / 160)
        )

    def test_validation_of_invalid_formats_and_flags(self):
        for key, value in [
            ("input_format", "coco"), ("output_format", "YOLO"),
            ("input_format", None), ("output_format", 0),
            ("input_normalized", 1), ("output_normalized", "yes"),
        ]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    self.t.to_source(self.target_xyxy, **{key: value})
                with self.assertRaises(ValueError):
                    self.t.to_source_many([], **{key: value})

    def test_invalid_box_in_xywh_and_center_formats(self):
        for box, fmt in [
            ((10, 10, 0, 10), "xywh"),
            ((10, 10, -1, 10), "xywh"),
            ((10, 10, 10, 0), "cxcywh"),
            ((10, 10, 10, -5), "cxcywh"),
            ((math.nan, 5, 5, 5), "cxcywh"),
            ((1, 1, math.inf, 1), "xywh"),
        ]:
            with self.subTest(box=box, fmt=fmt), self.assertRaises(ValueError):
                self.t.to_source(box, input_format=fmt)

    def test_clip_when_xywh_crosses_padding(self):
        with self.assertRaisesRegex(ValueError, "content"):
            self.t.to_source((0, 10, 60, 40), input_format="xywh")
        self.assertBoxClose(
            self.t.to_source(
                (0, 10, 60, 40), input_format="xywh", clip=True,
            ),
            (0.0, 0.0, 25.0, 15.0),
        )

    def test_padding_only_normalized_yolo_box_is_rejected(self):
        # Center y=0.05 is in top padding; no part intersects actual image.
        with self.assertRaises(ValueError):
            self.t.to_source(
                (0.5, 0.05, 0.1, 0.04),
                input_format="cxcywh", input_normalized=True,
                clip=True,
            )

    def test_round_trip_random_boxes_all_formats(self):
        rng = random.Random(42)
        for source, target in [
            ((100, 101), (640, 640)),
            ((1920, 1080), (641, 641)),
            ((29, 47), (171, 99)),
        ]:
            transform = letterbox(source, target)
            sw, sh = source
            for _ in range(15):
                x1 = rng.random() * sw * 0.6
                y1 = rng.random() * sh * 0.6
                x2 = x1 + (sw - x1) * (0.1 + 0.9 * rng.random())
                y2 = y1 + (sh - y1) * (0.1 + 0.9 * rng.random())
                box = (x1, y1, x2, y2)
                for fmt in ("xyxy", "xywh", "cxcywh"):
                    with self.subTest(source=source, fmt=fmt):
                        target_box = transform.to_target(
                            box, output_format=fmt, output_normalized=True,
                        )
                        restored = transform.to_source(
                            target_box, input_format=fmt,
                            input_normalized=True,
                        )
                        for got, want in zip(restored, box):
                            self.assertAlmostEqual(
                                got, want, delta=1e-8 * max(sw, sh)
                            )

    def test_normalized_source_clip(self):
        with self.assertRaises(ValueError):
            self.t.to_target(
                (-0.1, 0.2, 0.2, 0.5),
                input_normalized=True,
            )
        self.assertBoxClose(
            self.t.to_target(
                (-0.1, 0.2, 0.2, 0.5),
                input_normalized=True, clip=True,
                output_normalized=True,
            ),
            (10 / 240, 40 / 160, 50 / 240, 70 / 160),
        )


if __name__ == "__main__":
    unittest.main()
