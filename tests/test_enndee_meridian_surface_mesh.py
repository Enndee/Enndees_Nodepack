"""Tests for the Meridian surface renderer: the measured FOV and the triangulated mesh.

Both are pure geometry, so everything here runs on the CPU with a faked depth model - no
checkpoint, no GPU. The mesh tests check *coverage* and *occlusion* rather than pixels, because
"no holes" and "the near surface wins" are the two properties the point renderer could only
approximate with `point_size` and `back_face_cull`.
"""

import contextlib
import io
import json
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PACK_DIR = Path(__file__).resolve().parents[1]
NODES_DIR = PACK_DIR / "nodes"
sys.path.insert(0, str(NODES_DIR))

import enndee_meridian_fast_depth as fast_depth  # noqa: E402


def _still(height=64, width=112):
    """[1, H, W, 3] in [0, 1] with a left-to-right ramp, so colours identify their column."""
    ramp = torch.linspace(0.0, 1.0, width).view(1, 1, width).expand(1, height, width)
    return torch.stack([ramp, ramp.flip(-1), torch.full_like(ramp, 0.5)], dim=-1)


def _slanting_depth(width, height):
    """A plane: near on the left, far on the right (larger = farther, the backend's contract)."""
    return torch.linspace(1.0, 4.0, width).view(1, width).expand(height, width).contiguous()


def _fake_da3(model_name, first, device, process_res=fast_depth.DA3_RES):
    return _slanting_depth(int(first.shape[2]), int(first.shape[1]))


def _hole_mask(render):
    """Boolean mask of the pixels nothing was drawn into."""
    return (render - fast_depth.HOLE_COLOR / 255.0).abs().max(-1).values < 1e-6


class FovFromIntrinsicsTests(unittest.TestCase):
    """MoGe's `intrinsics` -> the source camera's vertical FOV, or None to keep the fallback."""

    BEACH = torch.tensor([[0.8564221858978271, 0.0, 0.5],
                          [0.0, 1.2846332788467407, 0.5],
                          [0.0, 0.0, 1.0]])

    def test_beach_intrinsics_give_the_measured_42_53_degrees(self):
        self.assertAlmostEqual(fast_depth._fov_y_degrees_from_intrinsics(self.BEACH), 42.5336,
                               places=3)

    def test_a_batched_intrinsics_tensor_is_accepted(self):
        self.assertAlmostEqual(fast_depth._fov_y_degrees_from_intrinsics(self.BEACH[None]),
                               42.5336, places=3)

    def test_a_square_pinhole_of_focal_one_is_the_documented_53_degrees(self):
        # fy = 1 means "the focal is exactly one image height", i.e. 2*atan(0.5) = 53.13 deg.
        self.assertAlmostEqual(fast_depth._fov_y_degrees_from_intrinsics(torch.eye(3)),
                               53.1301, places=3)

    def test_missing_or_unusable_values_keep_the_fallback(self):
        for value in (None, torch.eye(1), torch.zeros(3, 3), "nonsense",
                      torch.tensor([[1.0, 0, 0], [0, float("nan"), 0], [0, 0, 1.0]]),
                      torch.tensor([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])):
            with self.subTest(value=value):
                self.assertIsNone(fast_depth._fov_y_degrees_from_intrinsics(value))

    def test_an_implausible_fov_is_rejected_rather_than_rendered(self):
        # fy = 1e9 -> essentially a 0 degree lens; fy tiny -> a fisheye. Both are refused.
        absurd_narrow = torch.tensor([[1.0, 0, 0], [0, 1e9, 0], [0, 0, 1.0]])
        absurd_wide = torch.tensor([[1.0, 0, 0], [0, 0.0001, 0], [0, 0, 1.0]])
        self.assertIsNone(fast_depth._fov_y_degrees_from_intrinsics(absurd_narrow))
        self.assertIsNone(fast_depth._fov_y_degrees_from_intrinsics(absurd_wide))


class QuadTriangleTests(unittest.TestCase):
    """The depth grid -> triangles: trusted corners AND a quad that does not straddle a step."""

    FLAT = torch.ones(3, 4)

    def test_a_fully_trusted_grid_gives_two_triangles_per_quad(self):
        tris = fast_depth._quad_triangles(torch.ones(3, 4, dtype=torch.bool), self.FLAT, 4)
        self.assertEqual(tuple(tris.shape), (2 * 2 * 3, 3))       # 2x3 quads x 2 triangles

    def test_an_untrusted_corner_removes_every_quad_touching_it(self):
        keep = torch.ones(3, 4, dtype=torch.bool)
        keep[1, 1] = False
        # the corner belongs to 4 quads, so 6 - 4 = 2 quads survive
        self.assertEqual(fast_depth._quad_triangles(keep, self.FLAT, 4).shape[0], 2 * 2)

    def test_an_empty_mask_gives_no_triangles(self):
        tris = fast_depth._quad_triangles(torch.zeros(3, 4, dtype=torch.bool), self.FLAT, 4)
        self.assertEqual(tris.numel(), 0)

    def test_every_index_is_inside_the_grid(self):
        tris = fast_depth._quad_triangles(torch.ones(5, 7, dtype=torch.bool), torch.ones(5, 7), 7)
        self.assertGreaterEqual(int(tris.min()), 0)
        self.assertLess(int(tris.max()), 5 * 7)

    def test_the_two_triangles_of_a_quad_share_their_diagonal(self):
        tris = fast_depth._quad_triangles(torch.ones(2, 2, dtype=torch.bool), torch.ones(2, 2), 2)
        first, second = set(tris[0].tolist()), set(tris[1].tolist())
        self.assertEqual(len(first & second), 2)                  # the shared edge is the diagonal

    def test_a_quad_that_straddles_a_depth_step_is_cut(self):
        # 1.0 next to 10.0 is a 900 % relative spread: drawn as a surface it becomes two long
        # triangles reaching from the near plane to the far one. A point cannot stretch, so the
        # point renderer never needed this test - the mesh does.
        depth = torch.tensor([[1.0, 1.0], [1.0, 10.0]])
        tris = fast_depth._quad_triangles(torch.ones(2, 2, dtype=torch.bool), depth, 2)
        self.assertEqual(tris.numel(), 0)

    def test_a_gentle_slope_is_not_cut(self):
        depth = torch.tensor([[1.00, 1.05], [1.05, 1.10]])       # a 10 % spread, under the 30 %
        tris = fast_depth._quad_triangles(torch.ones(2, 2, dtype=torch.bool), depth, 2)
        self.assertEqual(tuple(tris.shape), (2, 3))

    def test_the_tolerance_is_the_widget_value(self):
        depth = torch.tensor([[1.0, 1.5], [1.0, 1.5]])           # a 50 % spread
        keep = torch.ones(2, 2, dtype=torch.bool)
        self.assertEqual(fast_depth._quad_triangles(keep, depth, 2, 0.30).numel(), 0)
        self.assertEqual(fast_depth._quad_triangles(keep, depth, 2, 0.60).shape[0], 2)


class NearPlaneClipTests(unittest.TestCase):
    """`_clip_triangles`: a triangle crossing the camera plane is split, not dropped whole.

    The implementation rests on one claim - that the crossing's *screen* position is a
    perspective-correct blend of the endpoints' screen positions, with the focal length and the
    principal point cancelling out. So these check it against an independent pinhole projection of
    the true 3D crossing point, not against the same formula twice.
    """

    F, CX, CY, NEAR = 400.0, 320.0, 240.0, 0.5

    def _project(self, points):
        """[N, 3] camera space -> [N, 2] canvas pixels, a plain pinhole."""
        return torch.stack([self.CX + self.F * points[:, 0] / points[:, 2],
                            self.CY + self.F * points[:, 1] / points[:, 2]], dim=1)

    def _crossing(self, first, second):
        """The 3D point where the segment first -> second meets z = near (first in front)."""
        t = (self.NEAR - first[2]) / (second[2] - first[2])
        return (first + t * (second - first))[None]

    def _clip(self, points, colors=None):
        uv = self._project(points)
        cols = torch.zeros(points.shape[0], 3) if colors is None else colors
        return fast_depth._clip_triangles(uv[:, 0][None], uv[:, 1][None], points[:, 2][None],
                                          cols[None], self.NEAR)

    def _on_plane(self, tu, tv, tz):
        """Every emitted vertex sitting exactly on the near plane, as (u, v)."""
        rows, cols = torch.nonzero(tz == self.NEAR, as_tuple=True)
        self.assertGreater(int(rows.numel()), 0, "nothing was placed on the near plane")
        return [(float(tu[r, c]), float(tv[r, c])) for r, c in zip(rows.tolist(), cols.tolist())]

    def _nearest(self, emitted, truth):
        return min(emitted, key=lambda point: math.hypot(point[0] - float(truth[0]),
                                                         point[1] - float(truth[1])))

    def test_a_crossing_lands_where_the_3d_point_projects(self):
        points = torch.tensor([[0.2, 0.1, 2.0], [0.5, -0.3, -1.0], [1.0, 0.4, 3.0]])
        tu, tv, tz, _tc = self._clip(points)
        truth = self._project(self._crossing(points[0], points[1]))[0]
        got = self._nearest(self._on_plane(tu, tv, tz), truth)
        self.assertAlmostEqual(got[0], float(truth[0]), delta=1e-3)
        self.assertAlmostEqual(got[1], float(truth[1]), delta=1e-3)

    def test_both_crossings_of_a_two_in_front_clip_land_correctly(self):
        points = torch.tensor([[0.2, 0.1, 2.0], [0.5, -0.3, 1.5], [1.0, 0.4, -0.5]])
        tu, tv, tz, _tc = self._clip(points)
        self.assertEqual(int(tu.shape[0]), 2)                    # a quad -> two triangles
        # the shared crossing sits in both triangles, so the distinct positions are what matter
        emitted = sorted(set(self._on_plane(tu, tv, tz)))
        self.assertEqual(len(emitted), 2)
        for edge in ((1, 2), (2, 0)):
            truth = self._project(self._crossing(points[edge[0]], points[edge[1]]))[0]
            got = self._nearest(emitted, truth)
            self.assertLess(math.hypot(got[0] - float(truth[0]), got[1] - float(truth[1])), 1e-3)

    def test_the_crossing_colour_is_the_linear_blend_not_the_projective_one(self):
        """Depth and colour are linear along the edge in 3D; only the screen position is projective."""
        points = torch.tensor([[0.2, 0.1, 2.0], [0.5, -0.3, -1.0], [1.0, 0.4, 3.0]])
        colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        _tu, _tv, tz, tc = self._clip(points, colors)
        t = (self.NEAR - 2.0) / (-1.0 - 2.0)
        rows, cols = torch.nonzero(tz == self.NEAR, as_tuple=True)
        self.assertTrue(torch.allclose(tc[rows[0], cols[0]],
                                       colors[0] + t * (colors[1] - colors[0]), atol=1e-5))

    def test_an_all_in_front_triangle_is_returned_untouched(self):
        points = torch.tensor([[0.2, 0.1, 2.0], [0.5, -0.3, 1.5], [1.0, 0.4, 3.0]])
        tu, _tv, _tz, _tc = self._clip(points)
        self.assertEqual(int(tu.shape[0]), 1)
        self.assertTrue(torch.equal(tu[0], self._project(points)[:, 0]))

    def test_an_all_behind_triangle_is_dropped(self):
        points = torch.tensor([[0.2, 0.1, -2.0], [0.5, -0.3, -1.5], [1.0, 0.4, -3.0]])
        tu, _tv, _tz, _tc = self._clip(points)
        self.assertEqual(int(tu.shape[0]), 0)

    def test_every_emitted_vertex_stays_in_front_of_the_plane(self):
        points = torch.tensor([[0.2, 0.1, 2.0], [0.5, -0.3, -1.0], [1.0, 0.4, -4.0]])
        _tu, _tv, tz, _tc = self._clip(points)
        self.assertTrue(bool((tz >= self.NEAR).all()))
        self.assertTrue(bool((tz > 0).all()))                    # what the `1 / z` depth needs

    def test_a_guard_band_drops_the_crossing_instead_of_drawing_it(self):
        """`RASTER_GUARD`: a crossing further out than the bound means the piece is a smear."""
        points = torch.tensor([[0.2, 0.1, 2.0], [0.5, -0.3, -1.0], [1.0, 0.4, 3.0]])
        uv = self._project(points)
        cols = torch.zeros(3, 3)
        tu, _tv, _tz, _tc = self._clip(points)
        self.assertEqual(int(tu.shape[0]), 2)                    # two in front -> two triangles
        wide = fast_depth._clip_triangles(uv[:, 0][None], uv[:, 1][None], points[:, 2][None],
                                          cols[None], self.NEAR, guard=1.0)
        self.assertEqual(int(wide[0].shape[0]), 0)               # the crossing is way past 1 px
        narrow = fast_depth._clip_triangles(uv[:, 0][None], uv[:, 1][None], points[:, 2][None],
                                            cols[None], self.NEAR, guard=1e9)
        self.assertEqual(int(narrow[0].shape[0]), 2)


class RasterMeshTests(unittest.TestCase):
    """The rasteriser: coverage, the shared diagonal, and draw-order-independent occlusion."""

    QUAD = torch.tensor([[0.5, 0.5], [3.5, 0.5], [3.5, 3.5], [0.5, 3.5]])
    QUAD_TRIS = torch.tensor([[0, 1, 2], [0, 2, 3]])

    def _raster(self, verts, z, cols, tris, w=4, h=4):
        return fast_depth._rasterize_mesh(verts, z, cols, tris, w, h, fast_depth.HOLE_COLOR)

    def test_a_single_triangle_covers_exactly_its_interior_lattice_points(self):
        verts = torch.tensor([[0.5, 0.5], [3.5, 0.5], [0.5, 3.5]])
        cols = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=torch.uint8)
        img = self._raster(verts, torch.tensor([1.0, 1.0, 1.0]), cols, torch.tensor([[0, 1, 2]]))
        # x + y <= 4 with both coordinates >= 1: (1,1) (1,2) (1,3) (2,1) (2,2) (3,1)
        self.assertEqual(int((img.sum(-1) != 128 * 3).sum()), 6)

    def test_the_shared_diagonal_leaves_no_crack(self):
        # Both triangles of the quad are drawn, so every lattice point inside the square is hit -
        # if the two triangles used opposite winding rules the diagonal would be missed.
        img = self._raster(self.QUAD, torch.ones(4), torch.full((4, 3), 200, dtype=torch.uint8),
                           self.QUAD_TRIS)
        self.assertEqual(int((img.sum(-1) != 128 * 3).sum()), 9)   # x, y in {1, 2, 3}

    def test_occlusion_does_not_depend_on_draw_order(self):
        red = torch.full((4, 3), 255, dtype=torch.uint8)
        red[:, 1:] = 0                                             # near
        blue = torch.full((4, 3), 255, dtype=torch.uint8)
        blue[:, :2] = 0                                            # far
        verts = torch.cat([self.QUAD, self.QUAD])
        cols = torch.cat([red, blue])
        z = torch.tensor([1.0] * 4 + [5.0] * 4)                    # red is the near quad
        near_first = torch.cat([self.QUAD_TRIS, self.QUAD_TRIS + 4])
        far_first = torch.cat([self.QUAD_TRIS + 4, self.QUAD_TRIS])
        for label, tris in (("near first", near_first), ("far first", far_first)):
            with self.subTest(order=label):
                img = self._raster(verts, z, cols, tris)
                covered = img.sum(-1) != 128 * 3
                self.assertTrue(bool(covered.any()))
                # the near (red) quad wins whichever order the two were submitted in
                self.assertTrue(bool((img[covered][:, 0] == 255).all()))
                self.assertTrue(bool((img[covered][:, 2] == 0).all()))

    def test_an_empty_mesh_is_all_hole(self):
        img = self._raster(torch.zeros(0, 2), torch.zeros(0),
                           torch.zeros(0, 3, dtype=torch.uint8), torch.zeros(0, 3, dtype=torch.long))
        self.assertTrue(bool((img == fast_depth.HOLE_COLOR).all()))

    def test_a_degenerate_triangle_is_skipped_not_crashed(self):
        verts = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])   # collinear: zero area
        img = self._raster(verts, torch.ones(3), torch.full((3, 3), 200, dtype=torch.uint8),
                           torch.tensor([[0, 1, 2]]))
        self.assertTrue(bool((img == fast_depth.HOLE_COLOR).all()))

    def test_a_crossing_far_outside_the_canvas_is_dropped_not_smeared(self):
        """A crossing ~100x the vertex positions out is a full-frame smear at ~`near` depth.

        Measured on the auto-camera spiral: drawing these took the subject from 3529 px to 0 px
        from frame 36 on - 39 of 73 frames without the subject, against 1 once `RASTER_GUARD`
        dropped them. z = 1 against a near plane of 0.01 is that case.
        """
        img = self._raster(self.QUAD, torch.tensor([-1.0, 1.0, 1.0, 1.0]),
                           torch.full((4, 3), 200, dtype=torch.uint8), self.QUAD_TRIS)
        self.assertTrue(bool((img == fast_depth.HOLE_COLOR).all()))

    def test_a_crossing_near_the_canvas_keeps_the_part_in_front_of_the_plane(self):
        """What the clip is *for*: the surface dipping through the camera plane keeps its near half.

        The crossing is magnified by `z / near`, so this stays on-canvas only when the geometry is
        within a small factor of the plane - which is exactly the case the clip is meant to save.
        """
        verts = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0]])
        img = self._raster(verts, torch.tensor([0.02, 0.02, -0.01]),
                           torch.full((3, 3), 200, dtype=torch.uint8),
                           torch.tensor([[0, 1, 2]]), w=8, h=8)
        covered = img.sum(-1) != 128 * 3
        self.assertGreater(int(covered.sum()), 3, "the visible part of the triangle was dropped")

    def test_geometry_off_canvas_is_dropped(self):
        img = self._raster(self.QUAD + 100.0, torch.ones(4),
                           torch.full((4, 3), 200, dtype=torch.uint8), self.QUAD_TRIS)
        self.assertTrue(bool((img == fast_depth.HOLE_COLOR).all()))


class SurfaceModeTests(unittest.TestCase):
    """`surface_mode` picks the renderer; the mesh needs no `point_size` and no back-face cull."""

    OPTIONS = dict(model_size="Depth-Anything-3-Small", frames=3, canvas_mode="custom",
                   cloud_scale=1, edge_cull=True, edge_threshold=0.30,
                   camera=None, custom_camera=None, depth_res=112)

    def _render(self, **overrides):
        options = dict(self.OPTIONS)
        options.update(overrides)
        with mock.patch.object(fast_depth, "_predict_da3_depth", _fake_da3):
            return fast_depth.render_depth_aligned(_still(), torch.device("cpu"), **options)

    def test_the_mesh_fills_what_a_bare_point_splat_leaves_holey(self):
        # canvas 224x128 against a 112x64 cloud: the points land on every second pixel, so a 1x1
        # footprint can only ever cover a quarter of it.
        #
        # The outer 1 px ring is measured apart from the interior, because it is a boundary effect
        # rather than a coverage one: the mesh stops at the last lattice point, so which side of the
        # outermost pixel centre the frame edge falls on is a sub-pixel question the gauge can move.
        # Measured on this fixture, the proportional map moved 381 px into the ring while leaving
        # the interior byte-identical (378 px of 28672 both before and after), so the interior is
        # what is bounded tightly here and the ring only loosely.
        common = dict(custom_width=224, custom_height=128)
        _s, mesh, _w, _h, _n = self._render(surface_mode="mesh", point_size=0, **common)
        _s, points, _w, _h, _n = self._render(surface_mode="points", point_size=0, **common)
        inner_mesh = float(_hole_mask(mesh)[:, 1:-1, 1:-1].float().mean())
        inner_points = float(_hole_mask(points)[:, 1:-1, 1:-1].float().mean())
        self.assertLess(inner_mesh, 0.02)
        self.assertGreater(inner_points, 0.5)
        self.assertGreater(inner_points, 10 * inner_mesh)
        self.assertLess(float(_hole_mask(mesh).float().mean()), 0.025)

    def test_mesh_mode_ignores_point_size_and_the_back_face_cull(self):
        base = self._render(surface_mode="mesh", custom_width=112, custom_height=64,
                            point_size=0, back_face_cull=False)[1]
        for overrides in (dict(point_size=3), dict(back_face_cull=True)):
            other = self._render(surface_mode="mesh", custom_width=112, custom_height=64,
                                 **overrides)[1]
            self.assertTrue(torch.equal(base, other), overrides)

    def test_the_point_mode_still_honours_point_size(self):
        thin = self._render(surface_mode="points", custom_width=224, custom_height=128,
                            point_size=0)[1]
        fat = self._render(surface_mode="points", custom_width=224, custom_height=128,
                           point_size=2)[1]
        self.assertLess(float(_hole_mask(fat).float().mean()),
                        float(_hole_mask(thin).float().mean()))

    def test_an_unknown_surface_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "surface_mode"):
            self._render(surface_mode="wireframe", custom_width=112, custom_height=64)

    def test_both_modes_produce_finite_frames(self):
        for mode in fast_depth.SURFACE_MODES:
            with self.subTest(mode=mode):
                _s, render, _w, _h, _n = self._render(surface_mode=mode, custom_width=112,
                                                      custom_height=64)
                self.assertTrue(bool(torch.isfinite(render).all()))


class MeasuredFovTests(unittest.TestCase):
    """The measured FOV has to reach BOTH `f_cloud` and `f_canvas`, or the render distorts."""

    OPTIONS = dict(model_size="MoGe-3-ViT-L", frames=3, canvas_mode="custom", custom_width=112,
                   custom_height=64, cloud_scale=1, edge_cull=True, edge_threshold=0.30,
                   camera=None, custom_camera=None, depth_res=112, surface_mode="mesh")

    def _render(self, fov, **overrides):
        depth = _slanting_depth(112, 64)
        options = dict(self.OPTIONS)
        options.update(overrides)
        with mock.patch.object(fast_depth, "_predict_moge_depth_and_fov",
                               lambda *a, **k: (depth, fov)):
            return fast_depth.render_depth_aligned(_still(), torch.device("cpu"), **options)

    def test_the_identity_frame_reproduces_the_source_whatever_the_fov(self):
        # At the opening pose the cloud is built and drawn through the same camera, so the frame
        # must come back as the still. That only holds if `f_cloud` and `f_canvas` used the SAME
        # number - mixing the measured value with the 55 degree assumption stretches it visibly.
        for fov in (None, 42.5336):
            with self.subTest(fov=fov):
                source, render, _w, _h, _n = self._render(fov)
                self.assertLess(float((render[0] - source[0]).abs().mean()), 0.02)
                self.assertLess(float(_hole_mask(render[0]).float().mean()), 0.02)

    def test_a_moved_frame_changes_when_the_fov_is_measured(self):
        # The parametric default camera is a *still* (motion lives in the `args` string), so use an
        # explicit two-key path - otherwise every frame is the identity and nothing can differ.
        signal = json.dumps({"frames": 90, "path": [
            {"t": 0, "src": 0, "pos": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]},
            {"t": 89, "src": 89, "pos": [0.6, 0.1, 0.6], "look": [0.0, 0.0, 1.0]},
        ]})
        motion = dict(frames=90, custom_camera=signal)
        _s, assumed, _w, _h, _n = self._render(None, **motion)        # the 55 degree fallback
        _s, measured, _w, _h, _n = self._render(42.5336, **motion)
        self.assertFalse(torch.equal(assumed[0], assumed[-1]))        # the camera really moved
        self.assertFalse(torch.equal(assumed[-1], measured[-1]))      # and the FOV matters

    def test_the_measured_fov_is_reported(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self._render(42.5336)
        self.assertIn("42.53", buffer.getvalue())

    def test_the_fallback_is_silent(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self._render(None)
        self.assertNotIn("measured", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
