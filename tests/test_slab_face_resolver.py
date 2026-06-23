import unittest
from unittest.mock import patch

import fitz
from shapely.geometry import Polygon

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ElementFootprint, Face, FaceGraph
from src.slab_v2.slab_face_resolver import resolve_slab_faces


class _Renderer:
    out_dir = "."


class SlabFaceResolverTests(unittest.TestCase):
    def setUp(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=200, height=200)
        self.content = fitz.Rect(0, 0, 200, 200)
        self.gross = Polygon([(10, 10), (190, 10), (190, 190), (10, 190)])
        self.cfg = SlabV2Config()

    def tearDown(self):
        self.doc.close()

    def test_filled_face_cannot_be_removed_without_negative_evidence(self):
        face = Face(0, self.gross, self.gross.area, source="fill")
        fg = FaceGraph([face])
        judged = {"selected_slab_ids": [], "appendage_ids": [],
                  "opening_ids": [], "non_slab_ids": ["face_0000"],
                  "review_ids": [], "confidence": 0.99,
                  "reason": "model guessed outside"}
        with patch("src.slab_v2.slab_face_resolver._judge",
                   return_value=judged):
            resolution, _ = resolve_slab_faces(
                self.page, fg, [{"polygon_pdf": self.gross}], [],
                self.content, self.cfg, _Renderer(), use_ai=True)
        self.assertAlmostEqual(resolution.gross_geometry.area,
                               self.gross.area)
        self.assertIn("face_0000", resolution.review_ids)

    def test_explicit_no_slab_closed_face_can_be_removed(self):
        hole = Polygon([(70, 70), (130, 70), (130, 130), (70, 130)])
        self.page.insert_text((82, 102), "NO SLAB")
        fg = FaceGraph([
            Face(0, self.gross, self.gross.area),
            Face(1, hole, hole.area),
        ])
        judged = {"selected_slab_ids": ["face_0000"],
                  "appendage_ids": [], "opening_ids": [],
                  "non_slab_ids": ["face_0001"], "review_ids": [],
                  "confidence": 0.95, "reason": "explicit NO SLAB"}
        with patch("src.slab_v2.slab_face_resolver._judge",
                   return_value=judged):
            resolution, _ = resolve_slab_faces(
                self.page, fg, [{"polygon_pdf": self.gross}], [],
                self.content, self.cfg, _Renderer(), use_ai=True)
        self.assertLess(resolution.gross_geometry.area, self.gross.area)
        self.assertIn("face_0001", resolution.non_slab_ids)

    def test_confirmed_opening_is_net_only_and_not_review_noise(self):
        hole = Polygon([(70, 70), (130, 70), (130, 130), (70, 130)])
        fg = FaceGraph([
            Face(0, self.gross, self.gross.area),
            Face(1, hole, hole.area),
        ])
        opening = ElementFootprint("STAIR", hole, "STAIR 01", hole.bounds,
                                   hole.area)
        judged = {"selected_slab_ids": ["face_0000"],
                  "appendage_ids": [], "opening_ids": ["face_0001"],
                  "non_slab_ids": [], "review_ids": [],
                  "confidence": 0.95, "reason": "matches confirmed stair"}
        with patch("src.slab_v2.slab_face_resolver._judge",
                   return_value=judged):
            resolution, _ = resolve_slab_faces(
                self.page, fg, [{"polygon_pdf": self.gross}], [],
                self.content, self.cfg, _Renderer(),
                resolved_openings=[opening], use_ai=True)
        self.assertAlmostEqual(resolution.gross_geometry.area, self.gross.area)
        self.assertLess(resolution.net_geometry.area, self.gross.area)
        self.assertEqual(resolution.status, "verified")


if __name__ == "__main__":
    unittest.main()
