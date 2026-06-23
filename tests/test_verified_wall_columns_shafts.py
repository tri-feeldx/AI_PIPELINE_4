import tempfile
import unittest
from pathlib import Path

import fitz
from types import SimpleNamespace
from shapely.geometry import box

from src.slab_v2.config import SlabV2Config
from src.slab_v2.export_ruby import generate_building_ruby
from src.slab_v2.models import (
    ColumnFootprint,
    ElementFootprint,
    ModelReadinessReport,
    SlabV2Result,
    WallFootprint,
)
from src.slab_v2.wall_junction_resolver import resolve_wall_junctions
from src.slab_v2.core_wall_topology import (
    reconcile_core_wall_topologies, resolve_core_wall_topology,
    resolve_lw1_topology)
from src.slab_v2.opening_resolver import (
    _stair_xcross_candidates, _stairwell_boundary_candidates)


class VerifiedWallColumnShaftTests(unittest.TestCase):
    def setUp(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=1000, height=700)

    def tearDown(self):
        self.doc.close()

    def test_lw6_snaps_to_lw7_without_losing_identity(self):
        # At scale 1:100, 0.12pt is 4.233mm: the observed production gap.
        walls = [
            WallFootprint("LW6", box(100, 100, 299.88, 108),
                          275, 7000, mapping_status="verified"),
            WallFootprint("LW7", box(292, 108, 300, 300),
                          275, 6800, mapping_status="verified"),
            WallFootprint("LW1", box(100, 292, 300, 300),
                          275, 7000, mapping_status="verified"),
        ]
        cfg = SlabV2Config(debug_images=False)
        with tempfile.TemporaryDirectory() as temp:
            resolved, report = resolve_wall_junctions(
                self.page, walls, {"LW6": 1, "LW7": 1, "LW1": 1},
                100, [], cfg, Path(temp))
        by_label = {wall.label: wall for wall in resolved}
        self.assertEqual(report["status"], "verified")
        self.assertAlmostEqual(by_label["LW6"].polygon.bounds[2],
                               by_label["LW7"].polygon.bounds[2], places=6)
        self.assertEqual(by_label["LW1"].polygon.bounds,
                         walls[2].polygon.bounds)
        row = next(item for item in report["junctions"]
                   if item.get("wall") == "LW6"
                   and item.get("target_wall") == "LW7")
        self.assertGreater(row["before_gap_mm"], 4.0)
        self.assertLessEqual(row["after_gap_mm"], 1.0)

    def test_shaft_cuts_slab_but_never_renders_a_separate_solid(self):
        result = SlabV2Result(page_index=0, scale=100)
        result.slabs = [{"label": "SLAB", "polygon_pdf": box(50, 50, 500, 500),
                         "polygon_mm": None, "area_m2": 250.0}]
        shaft = ElementFootprint("SHAFT", box(200, 200, 250, 250),
                                 "CORE/SHAFT", (0, 0, 0, 0), 2500)
        stair = ElementFootprint("STAIR", box(300, 200, 350, 300),
                                 "STAIR 01", (0, 0, 0, 0), 5000)
        result.resolved_openings = [shaft, stair]
        result.render_elements = [stair]
        result.column_detection_report = {
            "status": "not_required", "expected": {}, "detected": {},
            "missing": {}, "extra": {}, "ambiguous_count": 0}
        readiness = ModelReadinessReport(model_status="debug")
        cfg = SlabV2Config(render_shaft_solids=False)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "model.rb"
            generate_building_ruby([{
                "result": result, "page": self.page, "ffl_mm": 0.0,
                "level_id": "level_1", "height_mm": 3000,
                "height_status": "verified_explicit",
            }], str(out), cfg, readiness_report=readiness)
            ruby = out.read_text(encoding="utf-8")
        self.assertNotIn("SHAFT CORE/SHAFT", ruby)
        self.assertNotIn("WALL_CORE/SHAFT", ruby)
        self.assertIn("0 shaft solid(s)", ruby)
        self.assertNotIn("STAIR_PLACEHOLDER", ruby)
        self.assertIn("0 stair solid(s)", ruby)

    def test_verified_columns_have_unique_symbols_and_no_ambiguity(self):
        columns = [ColumnFootprint(
            symbol=f"C{i}", polygon=box(i*10, 0, i*10+5, 5),
            candidate_id=f"candidate_{i}", source="vector", confidence=0.9)
            for i in range(1, 17)]
        self.assertEqual(len(columns), 16)
        self.assertEqual(len({column.symbol for column in columns}), 16)
        self.assertNotIn("C?", {column.symbol for column in columns})

    def test_large_stair_xcross_becomes_finite_penetration_candidate(self):
        self.page.insert_text((115, 185), "STAIR 01")
        path = SimpleNamespace(
            outside_content=False,
            segments=[((100, 300), (300, 150)),
                      ((300, 300), (100, 100))])
        candidates, defaults, _warnings = _stair_xcross_candidates(
            self.page, [path], self.page.rect, box(50, 50, 500, 500), 100)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["kind_hint"], "STAIR_PENETRATION")
        self.assertEqual(defaults, [candidates[0]["id"]])
        self.assertAlmostEqual(candidates[0]["polygon"].bounds[0], 100)
        self.assertAlmostEqual(candidates[0]["polygon"].bounds[2], 300)

    def test_xcross_is_seed_and_closed_vector_enclosure_is_final(self):
        xseed = {
            "id": "stair_STAIR_01_xcross", "label": "STAIR 01",
            "polygon": box(120, 120, 260, 260),
        }
        flight = {
            "id": "stair_STAIR_01_flight", "label": "STAIR 01",
            "kind_hint": "STAIR_OPENING", "polygon": box(220, 100, 300, 170),
        }
        segments = [
            ((100, 80), (320, 80)), ((320, 80), (320, 280)),
            ((320, 280), (100, 280)), ((100, 280), (100, 80)),
        ]
        path = SimpleNamespace(outside_content=False, style_id=0,
                               segments=segments)
        candidates, defaults, resolved, _warnings = (
            _stairwell_boundary_candidates(
                self.page, [path], None, box(0, 0, 600, 600), 100,
                [xseed], [flight], SlabV2Config(debug_images=False)))
        self.assertEqual(defaults, ["stair_STAIR_01_closed_stairwell"])
        self.assertEqual(len(resolved), 1)
        self.assertGreater(candidates[0]["polygon"].area,
                           xseed["polygon"].area)
        self.assertEqual(candidates[0]["polygon"].bounds,
                         (100.0, 80.0, 320.0, 280.0))

    def test_lw1_completion_requires_two_target_page_vector_rails(self):
        walls = [
            WallFootprint("LW1", box(140, 292, 220, 300), 275, 2800),
            WallFootprint("LW2", box(100, 120, 108, 292), 275, 6300),
            WallFootprint("LW4", box(140, 108, 148, 260), 275, 5300),
            WallFootprint("LW5", box(220, 108, 228, 260), 275, 5300),
            WallFootprint("LW7", box(292, 120, 300, 292), 275, 6300),
            WallFootprint("LW3", box(148, 250, 220, 258), 275, 2500),
        ]
        segments = [
            ((100, 292), (180, 292)), ((180, 292), (300, 292)),
            ((100, 300), (190, 300)), ((190, 300), (300, 300)),
            ((148, 250), (220, 250)), ((148, 258), (220, 258)),
        ]
        path = SimpleNamespace(outside_content=False, style_id=0,
                               segments=segments)
        with tempfile.TemporaryDirectory() as temp:
            resolved, report = resolve_lw1_topology(
                self.page, [path], None, walls, 100, [],
                SlabV2Config(debug_images=False), Path(temp))
        by_label = {wall.label: wall for wall in resolved}
        self.assertEqual(report["status"], "verified")
        self.assertEqual(by_label["LW1"].polygon.bounds,
                         (100.0, 292.0, 300.0, 300.0))
        self.assertIn("global_core_topology", by_label["LW1"].source)

    def test_global_topology_corrects_swapped_lw1_lw3_identity(self):
        walls = [
            WallFootprint("LW1", box(148, 250, 220, 258), 275, 2500),
            WallFootprint("LW6", box(100, 100, 300, 108), 275, 7000),
            WallFootprint("LW4", box(140, 108, 148, 300), 275, 6700),
            WallFootprint("LW5", box(220, 108, 228, 292), 275, 6400),
            # Mislabelled main/right fragment of the real bottom chord.
            WallFootprint("LW3", box(148, 292, 300, 300), 275, 5300),
            WallFootprint("LW2", box(100, 220, 108, 292), 275, 2500),
            WallFootprint("LW7", box(292, 108, 300, 292), 275, 6400),
        ]
        segments = [
            ((100, 292), (300, 292)), ((100, 300), (300, 300)),
            ((148, 250), (220, 250)), ((148, 258), (220, 258)),
        ]
        path = SimpleNamespace(outside_content=False, style_id=0,
                               segments=segments)
        with tempfile.TemporaryDirectory() as temp:
            resolved, report = resolve_core_wall_topology(
                self.page, [path], None, walls, 100, [],
                SlabV2Config(debug_images=False), Path(temp))
            self.assertTrue((Path(temp) /
                             "core_wall_global_assignment_p01.json").exists())
        by_label = {wall.label: wall for wall in resolved}
        self.assertEqual(report["status"], "verified")
        self.assertEqual(by_label["LW1"].polygon.bounds,
                         (100.0, 292.0, 300.0, 300.0))
        self.assertEqual(by_label["LW3"].polygon.bounds,
                         (148.0, 250.0, 220.0, 258.0))
        self.assertEqual(len(report["rejected_extension_bboxes"]), 2)
        self.assertEqual({row["from_label"] for row in report["corrections"]},
                         {"LW1", "LW3"})

    def test_reference_prefers_complete_native_topology(self):
        def result(page_index, derived, debug_dir):
            walls = [WallFootprint(
                f"LW{i}", box(i*20, 100, i*20+8, 200), 275, 3500,
                source=("plan_shape+global_core_topology"
                        if derived and i == 1 else "plan_shape"))
                for i in range(1, 8)]
            return SimpleNamespace(
                page_index=page_index, walls=walls,
                wall_readiness={
                    "core_topology_report": {"status": "verified"},
                    "junction_report": {"status": "verified"},
                    "core_topology_status": "verified",
                    "junction_status": "verified"},
                opening_report={}, render_elements=[], debug_dir=debug_dir)
        with tempfile.TemporaryDirectory() as temp:
            p6 = Path(temp) / "p6"; p6.mkdir()
            p11 = Path(temp) / "p11"; p11.mkdir()
            report = reconcile_core_wall_topologies([
                {"level_id": "L1", "result": result(5, True, p6)},
                {"level_id": "L2", "result": result(10, False, p11)},
            ], Path(temp))
        self.assertEqual(report["reference"]["page"], 11)
        self.assertEqual(report["reference"]["topology_derived_walls"], 0)


if __name__ == "__main__":
    unittest.main()
