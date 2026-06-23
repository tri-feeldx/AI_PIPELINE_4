import unittest

from src.slab_v2.config import SlabV2Config
from src.slab_v2.height_reconcile import solve_level_datums
from src.slab_v2.height_source_planner import build_consensus
from src.slab_v2.models import (
    BuildingInfo, DocAnalysis, FloorInfo, HeightEvidence,
)


class HeightDatumGraphTests(unittest.TestCase):
    def setUp(self):
        self.building = BuildingInfo(name="Building A", floors=[
            FloorInfo(level_id="level_1"),
            FloorInfo(level_id="level_2"),
            FloorInfo(level_id="level_3"),
        ])
        self.analysis = DocAnalysis(buildings=[self.building])
        self.cfg = SlabV2Config()

    def test_explicit_ffls_are_verified(self):
        evidence = [
            HeightEvidence(f"e{i}", "Building A", None, f"level_{i}",
                           "explicit_datum", value, confidence=0.95,
                           extraction_method="page_text_regex",
                           is_absolute_datum=True)
            for i, value in ((1, 0), (2, 3600), (3, 7200))
        ]
        result = solve_level_datums(evidence, self.analysis, self.cfg)
        self.assertAlmostEqual(result.level_datums[0].ffl_mm, 0, delta=1)
        self.assertAlmostEqual(result.level_datums[1].ffl_mm, 3600, delta=1)
        self.assertEqual(result.level_datums[0].status, "verified_explicit")
        self.assertEqual(result.level_datums[1].storey_height_mm, 3600)

    def test_relative_only_is_not_absolute_verified(self):
        evidence = [HeightEvidence(
            "r1", "Building A", "level_1", "level_2",
            "scaled_elevation_spacing", 3800,
            extraction_method="elevation_vector_spacing",
            confidence=0.8, is_absolute_datum=False)]
        result = solve_level_datums(evidence, self.analysis, self.cfg)
        self.assertEqual(result.level_datums[0].status, "inferred_relative")
        self.assertAlmostEqual(result.level_datums[1].ffl_mm, 3800, delta=1)

    def test_strong_disagreement_surfaces_conflict(self):
        evidence = [
            HeightEvidence("a1", "Building A", None, "level_1",
                           "explicit_datum", 0, confidence=0.95,
                           extraction_method="page_text_regex",
                           is_absolute_datum=True),
            HeightEvidence("a2", "Building A", None, "level_2",
                           "explicit_datum", 3500, confidence=0.95,
                           extraction_method="page_text_regex",
                           is_absolute_datum=True),
            HeightEvidence("r1", "Building A", "level_1", "level_2",
                           "explicit_storey_height", 4000, confidence=0.9,
                           extraction_method="section_text_regex",
                           is_absolute_datum=False),
        ]
        result = solve_level_datums(evidence, self.analysis, self.cfg)
        self.assertTrue(result.conflicts)
        self.assertIn("conflict", {d.status for d in result.level_datums})

    def test_three_independent_measurements_promote_consensus(self):
        rows = [HeightEvidence(
            f"m{i}", "Building A", "level_1", "level_2",
            "scaled_elevation_measurement", value,
            extraction_method="datum_line_spacing", confidence=0.8,
            scale_ratio=100, scale_status="verified_local_text",
            source_fingerprint=f"view{i}", independence_group=f"view{i}")
            for i, value in enumerate((3598, 3601, 3603), 1)]
        evidence, report = build_consensus(rows)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].evidence_type, "verified_consensus")
        self.assertEqual(report[0]["status"], "verified_consensus")

    def test_duplicate_views_do_not_form_consensus(self):
        rows = [HeightEvidence(
            f"m{i}", "Building A", "level_1", "level_2",
            "scaled_elevation_measurement", 3600,
            extraction_method="datum_line_spacing", confidence=0.8,
            scale_ratio=100, scale_status="verified_local_text",
            source_fingerprint="same", independence_group="same")
            for i in range(3)]
        evidence, report = build_consensus(rows)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].evidence_type,
                         "scaled_elevation_measurement")
        self.assertEqual(report[0]["independent_count"], 1)

    def test_terminal_datum_sets_top_storey_height_without_new_floor(self):
        evidence = [
            HeightEvidence("r12", "Building A", "level_1", "level_2",
                           "verified_consensus", 3600, confidence=0.95),
            HeightEvidence("r23", "Building A", "level_2", "level_3",
                           "verified_consensus", 3700, confidence=0.95),
            HeightEvidence("r34", "Building A", "level_3", "level_4",
                           "verified_consensus", 3500, confidence=0.95),
        ]
        result = solve_level_datums(evidence, self.analysis, self.cfg)
        self.assertEqual(len(result.level_datums), 3)
        self.assertEqual(result.level_datums[-1].storey_height_mm, 3500)


if __name__ == "__main__":
    unittest.main()
