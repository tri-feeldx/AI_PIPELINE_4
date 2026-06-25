import unittest
from unittest.mock import patch

import fitz
from shapely.affinity import rotate
from shapely.geometry import Polygon

from src.slab_v2.config import SlabV2Config
from src.slab_v2.floor_system_resolver import (
    build_floor_system_candidates,
    resolve_floor_systems,
)
from src.slab_v2.models import (
    ElementFootprint,
    FloorSystemSemanticProfile,
    VectorPath,
)


class FloorSystemCandidateTests(unittest.TestCase):
    def setUp(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=1000, height=600)
        self.page.insert_text((400, 580), "POST TENSIONED SLAB")
        self.page.insert_text((850, 40), "FLOOR STRUCTURE")
        self.gross = Polygon([(100, 100), (900, 100), (900, 500),
                              (100, 500), (100, 100)])
        inner = Polygon([(180, 150), (820, 150), (820, 400),
                         (180, 400), (180, 150)])
        self.paths = [VectorPath(
            id=7, style_id=3,
            segments=list(zip(list(inner.exterior.coords),
                              list(inner.exterior.coords)[1:])),
            is_closed=True, is_filled=True, seqno=1,
            fill_polygon=inner)]
        left = Polygon([(100, 250), (178, 250), (178, 400),
                        (100, 400), (100, 250)])
        right = Polygon([(822, 250), (900, 250), (900, 400),
                         (822, 400), (822, 250)])
        self.openings = [
            ElementFootprint("STAIR", left, "STAIR L", left.bounds, left.area),
            ElementFootprint("STAIR", right, "STAIR R", right.bounds, right.area),
        ]
        self.profile = FloorSystemSemanticProfile(
            concrete_slab_terms=["POST TENSIONED SLAB"],
            floor_extent_terms=["FLOOR STRUCTURE"],
            steel_floor_terms=["STEELWORK"], confidence=0.9)
        self.cfg = SlabV2Config()

    def tearDown(self):
        self.doc.close()

    def test_nested_extent_plus_stairs_creates_two_other_floor_strips(self):
        candidates = build_floor_system_candidates(
            self.page, self.paths, self.gross, [], self.profile, self.cfg,
            context_objects=self.openings)
        other = [c for c in candidates if c.id.startswith("floor_other")]
        self.assertEqual(len(other), 2)
        self.assertTrue(all("external_stair_interface" in
                            c.negative_pt_evidence for c in other))
        self.assertTrue(all(abs(c.polygon.bounds[3] - 400) < 0.1
                            for c in other))
        self.assertTrue(all(c.rejected_extension_area_pt2 > 0
                            for c in other))
        main = next(c for c in candidates if c.id == "floor_pt_001")
        self.assertAlmostEqual(main.polygon.area, self.gross.area)

    def test_nested_extent_without_verified_stair_does_not_split(self):
        candidates = build_floor_system_candidates(
            self.page, self.paths, self.gross, [], self.profile, self.cfg)
        self.assertFalse(any(c.id.startswith("floor_other") for c in candidates))
        main = next(c for c in candidates if c.id == "floor_pt_001")
        self.assertAlmostEqual(main.polygon.area, self.gross.area)

    def test_terminal_mismatch_is_review_and_keeps_gross_pt(self):
        short_stairs = []
        for opening in self.openings:
            minx, miny, maxx, _maxy = opening.polygon.bounds
            poly = Polygon([(minx, miny), (maxx, miny), (maxx, 350),
                            (minx, 350), (minx, miny)])
            short_stairs.append(ElementFootprint(
                "STAIR", poly, opening.label, poly.bounds, poly.area))
        candidates = build_floor_system_candidates(
            self.page, self.paths, self.gross, [], self.profile, self.cfg,
            scale=100, context_objects=short_stairs)
        self.assertFalse(any(c.id.startswith("floor_other") for c in candidates))
        self.assertTrue(any(c.id.startswith("floor_review") for c in candidates))
        main = next(c for c in candidates if c.id == "floor_pt_001")
        self.assertAlmostEqual(main.polygon.area, self.gross.area)

    def test_rotated_plan_uses_local_separator_axis(self):
        angle = 27
        origin = (500, 300)
        gross = rotate(self.gross, angle, origin=origin)
        inner = rotate(self.paths[0].fill_polygon, angle, origin=origin)
        path = VectorPath(
            id=8, style_id=3,
            segments=list(zip(list(inner.exterior.coords),
                              list(inner.exterior.coords)[1:])),
            is_closed=True, is_filled=True, seqno=1,
            fill_polygon=inner)
        openings = []
        for opening in self.openings:
            poly = rotate(opening.polygon, angle, origin=origin)
            openings.append(ElementFootprint(
                "STAIR", poly, opening.label, poly.bounds, poly.area))
        candidates = build_floor_system_candidates(
            self.page, [path], gross, [], self.profile, self.cfg,
            scale=100, context_objects=openings)
        other = [c for c in candidates if c.id.startswith("floor_other")]
        self.assertEqual(len(other), 2)
        self.assertTrue(all(c.cut_status == "bounded_verified" for c in other))

    def test_unknown_terminal_is_preserved_in_pt_geometry(self):
        short_stairs = []
        for opening in self.openings:
            minx, miny, maxx, _maxy = opening.polygon.bounds
            poly = Polygon([(minx, miny), (maxx, miny), (maxx, 350),
                            (minx, 350), (minx, miny)])
            short_stairs.append(ElementFootprint(
                "STAIR", poly, opening.label, poly.bounds, poly.area))
        decision = {
            "pt_concrete_slab_ids": ["floor_pt_001"],
            "other_floor_system_ids": [],
            "opening_ids": ["floor_opening_001", "floor_opening_002"],
            "non_floor_ids": [],
            "unknown_ids": ["floor_review_001", "floor_review_002"],
            "confidence_by_id": {}, "reason_by_id": {},
        }
        renderer = type("Renderer", (), {"out_dir": "."})()
        with patch(
                "src.slab_v2.floor_system_resolver._profile_with_gemini",
                return_value=self.profile), patch(
                "src.slab_v2.floor_system_resolver._judge",
                return_value=decision):
            resolution, candidates, _profile = resolve_floor_systems(
                self.page, self.paths, [],
                [{"polygon_pdf": self.gross}], [], self.cfg, renderer,
                use_ai=True, scale=100, context_objects=short_stairs)
        self.assertTrue(any(c.id.startswith("floor_review")
                            for c in candidates))
        self.assertAlmostEqual(resolution.pt_gross_geometry.area,
                               self.gross.area)
        self.assertEqual(resolution.status, "review")


if __name__ == "__main__":
    unittest.main()
