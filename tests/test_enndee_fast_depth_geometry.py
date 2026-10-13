"""Geometry invariants of `enndee_meridian_fast_depth` that must not regress.

Two fixes landed for "the subject reads too shallow", and both are easy to break
silently:

1. `aspect_x_scale` - the x factor that makes the identity frame reproject exactly when
   the cloud aspect and the render canvas aspect disagree (1.4884 vs 1.3529 measured on
   the Hanks still -> a 1.100x horizontal overscan before the fix).
2. `apply_depth_relief` - depth relief exaggeration about the median. It must leave the
   median (which is `zm`, the gauge the camera rig lives in) untouched, or the whole
   rig slides.

Run directly:  python tests\\test_enndee_fast_depth_geometry.py
"""
import math
import pathlib
import sys
import unittest

PACK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(PACK / "nodes"))

import enndee_meridian_fast_depth as fd                      # noqa: E402


class AspectXScaleTests(unittest.TestCase):
    def test_identity_when_the_aspects_already_match(self):
        self.assertAlmostEqual(fd.aspect_x_scale(1920, 1290, 1920, 1290), 1.0, places=12)
        self.assertAlmostEqual(fd.aspect_x_scale(800, 600, 1600, 1200), 1.0, places=12)

    def test_the_hanks_case_removes_the_ten_percent_overscan(self):
        # working still 1920x1290 (aspect 1.4884), canvas 736x544 (aspect 1.3529)
        scale = fd.aspect_x_scale(1920, 1290, 736, 544)
        self.assertAlmostEqual(scale, (736 / 544) / (1920 / 1290), places=12)
        self.assertAlmostEqual(scale, 0.9090, places=4)
        # the overscan it removes is the reciprocal
        self.assertAlmostEqual(1.0 / scale, 1.1001, places=4)

    def test_the_identity_frame_reprojects_exactly(self):
        """Project the unprojection back and demand u_out == u.

        This is the invariant the fix exists for. It is checked here with the real
        pinhole the renderer uses (`f = 0.5 * H / tan(vfov/2)` on both sides), so a
        regression in either the cloud or the canvas side fails loudly.
        """
        cloud_w, cloud_h = 1920, 1290
        out_w, out_h = 736, 544
        vfov = 38.043
        f_cloud = 0.5 * cloud_h / math.tan(math.radians(vfov) / 2.0)
        f_canvas = 0.5 * out_h / math.tan(math.radians(vfov) / 2.0)
        x_scale = fd.aspect_x_scale(cloud_w, cloud_h, out_w, out_h)
        z = 1.37                                     # any depth; it must cancel
        for u in (0.0, 0.17, 0.5, 0.83, 1.0):
            x = (u - 0.5) * cloud_w / f_cloud * z * x_scale
            u_out = 0.5 + f_canvas * x / (z * out_w)
            self.assertAlmostEqual(u_out, u, places=12,
                                   msg=f"u={u} reprojected to {u_out}")

    def test_without_the_scale_the_overscan_is_exactly_the_aspect_ratio(self):
        cloud_w, cloud_h, out_w, out_h = 1920, 1290, 736, 544
        vfov = 38.043
        f_cloud = 0.5 * cloud_h / math.tan(math.radians(vfov) / 2.0)
        f_canvas = 0.5 * out_h / math.tan(math.radians(vfov) / 2.0)
        z = 2.0
        u = 0.9
        x = (u - 0.5) * cloud_w / f_cloud * z        # NO correction
        u_out = 0.5 + f_canvas * x / (z * out_w)
        expected = 0.5 + (u - 0.5) * ((cloud_w / cloud_h) / (out_w / out_h))
        self.assertAlmostEqual(u_out, expected, places=12)
        self.assertGreater(u_out, u)                 # it overflows past the source edge


class DepthReliefTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.z = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2], dtype=torch.float64)

    def test_one_is_the_identity(self):
        out = fd.apply_depth_relief(self.z, 1.0)
        self.assertTrue(self.torch.equal(out, self.z))

    def test_none_is_the_identity(self):
        self.assertTrue(self.torch.equal(fd.apply_depth_relief(self.z, None), self.z))

    def test_the_median_is_untouched(self):
        """`zm` is the median - if it moves, the whole camera rig slides."""
        for relief in (0.5, 1.0, 1.4, 2.0, 3.5):
            out = fd.apply_depth_relief(self.z, relief)
            self.assertAlmostEqual(float(out.median()), float(self.z.median()),
                                   places=12, msg=f"relief={relief}")

    def test_the_spread_scales_by_the_relief(self):
        for relief in (0.5, 1.2, 1.4, 2.0):
            out = fd.apply_depth_relief(self.z, relief)
            before = float(self.z.max() - self.z.min())
            after = float(out.max() - out.min())
            self.assertAlmostEqual(after, before * relief, places=10,
                                   msg=f"relief={relief}")

    def test_depth_relief_validation_is_finite_and_in_range(self):
        for value in (0.25, 1.0, 1.2, 4.0):
            self.assertEqual(fd.resolve_depth_relief(value), float(value))
        for value in (0.249, 4.001, float("nan"), float("inf"), "invalid"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "depth_relief"):
                fd.resolve_depth_relief(value)

    def test_it_never_returns_a_nonpositive_depth(self):
        torch = self.torch
        nasty = torch.tensor([0.01, 0.02, 5.0], dtype=torch.float64)
        out = fd.apply_depth_relief(nasty, 0.1)     # compress hard toward the median
        self.assertGreater(float(out.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
