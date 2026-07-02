"""TDD tests for SLAB_OPENING classification path in opening_resolver.

These tests verify the new classification path that allows X-crosses
inside the slab (≥90% containment, ≤5% structural overlap) to be
classified as openings even WITHOUT "SLAB PENETRATION" legend text.

Test PDF: 2381_MSCP_STR_Combine.pdf pages 5,6,7,8,9
(these have X marks but NO legend text → currently all stuck in review)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import OpeningIntent


# ---------------------------------------------------------------------------
# Unit tests: _raw_candidates classification logic (synthetic, no PDF)
# ---------------------------------------------------------------------------

class TestSlabOpeningClassification:
    """Test the new SLAB_OPENING elif branch in _raw_candidates()."""

    def _run_raw_candidates(self, element, walls=None, slab_union=None,
                            page_text="", scale=100, columns=None,
                            words=None):
        """Helper: call _raw_candidates with a single synthetic element."""
        from unittest.mock import MagicMock
        from shapely.geometry import box
        import fitz

        page = MagicMock()
        def _get_text(mode="text"):
            if mode == "text":
                return page_text
            if mode == "words":
                return words or []
            return page_text
        page.get_text = _get_text

        content_rect = fitz.Rect(0, 0, 1000, 700)

        if slab_union is None:
            slab_union = box(0, 0, 1000, 700)

        from src.slab_v2.opening_resolver import _raw_candidates
        candidates, default_ids, warnings = _raw_candidates(
            [element], walls or [], page, content_rect,
            slab_union=slab_union, scale=scale, columns=columns)
        return candidates, default_ids

    def _make_element(self, polygon, etype="VOID", label=""):
        """Create a minimal ElementFootprint-like object."""
        from unittest.mock import MagicMock
        elem = MagicMock()
        elem.polygon = polygon
        elem.type = etype
        elem.label = label
        elem.area_pt2 = polygon.area
        elem.anchor_bbox = polygon.bounds
        return elem

    def test_xcross_inside_slab_no_legend_becomes_opening(self):
        """X-cross 100% inside slab, no legend text → should be SLAB_OPENING
        with action='opening', not stuck in 'review'."""
        from shapely.geometry import box
        slab = box(0, 0, 1000, 700)
        xcross = box(200, 200, 230, 230)  # ~1.0 m² at scale=100
        elem = self._make_element(xcross)

        cands, defaults = self._run_raw_candidates(
            elem, slab_union=slab, page_text="GROUND FLOOR PLAN")

        assert len(cands) == 1
        c = cands[0]
        assert c["kind_hint"] == "SLAB_OPENING", (
            f"Expected SLAB_OPENING, got {c['kind_hint']}")
        assert c["default_action"] == "opening"
        assert c["confidence"] >= 0.70
        assert c["destructive_allowed"] is True

    def test_xcross_outside_slab_stays_review(self):
        """X-cross <90% inside slab → should stay as review."""
        from shapely.geometry import box
        slab = box(0, 0, 500, 500)
        xcross = box(480, 480, 510, 510)  # mostly outside slab
        elem = self._make_element(xcross)

        cands, _ = self._run_raw_candidates(elem, slab_union=slab)

        assert len(cands) == 1
        c = cands[0]
        assert c["default_action"] == "review", (
            f"Expected review for outside-slab X, got {c['default_action']}")

    def test_xcross_near_stair_stays_review(self):
        """X-cross near STAIR text → should NOT become SLAB_OPENING."""
        from shapely.geometry import box
        slab = box(0, 0, 1000, 700)
        xcross = box(200, 200, 230, 230)
        elem = self._make_element(xcross)

        # "STAIR" word at center (215,215) — inside element's 40pt buffer zone
        stair_words = [(210, 210, 220, 220, "STAIR", 0, 0)]

        cands, _ = self._run_raw_candidates(
            elem, slab_union=slab,
            page_text="GROUND FLOOR",
            words=stair_words)

        assert len(cands) == 1
        c = cands[0]
        assert c["kind_hint"] != "SLAB_OPENING" or c["default_action"] == "review"

    def test_xcross_too_much_structural_overlap_stays_review(self):
        """X-cross >5% wall overlap → should NOT become SLAB_OPENING."""
        from shapely.geometry import box
        from unittest.mock import MagicMock

        slab = box(0, 0, 1000, 700)
        xcross = box(200, 200, 230, 230)  # 30x30
        elem = self._make_element(xcross)

        # Wall overlapping >5% of xcross
        wall = MagicMock()
        wall.polygon = box(200, 200, 215, 230)  # ~50% overlap
        wall.label = "W1"

        cands, _ = self._run_raw_candidates(
            elem, walls=[wall], slab_union=slab)

        c = cands[0]
        assert c["kind_hint"] != "SLAB_OPENING" or c["default_action"] != "opening"

    def test_shaft_takes_priority_over_slab_opening(self):
        """X near LW wall → should be SHAFT, not SLAB_OPENING."""
        from shapely.geometry import box
        from unittest.mock import MagicMock

        slab = box(0, 0, 1000, 700)
        xcross = box(200, 200, 230, 230)
        elem = self._make_element(xcross)

        # LW wall very close (within 35pt)
        wall = MagicMock()
        wall.polygon = box(190, 190, 195, 240)  # adjacent, not overlapping
        wall.label = "LW1"

        cands, _ = self._run_raw_candidates(
            elem, walls=[wall], slab_union=slab)

        c = cands[0]
        assert c["kind_hint"] == "SHAFT", (
            f"Expected SHAFT (LW priority), got {c['kind_hint']}")
        assert c["default_action"] == "opening"

    def test_legend_slab_penetration_takes_priority(self):
        """With legend text 'SLAB PENETRATION' → should be SLAB_PENETRATION,
        not SLAB_OPENING."""
        from shapely.geometry import box

        slab = box(0, 0, 1000, 700)
        xcross = box(200, 200, 230, 230)
        elem = self._make_element(xcross)

        cands, _ = self._run_raw_candidates(
            elem, slab_union=slab,
            page_text="SLAB PENETRATION\nGROUND FLOOR PLAN")

        c = cands[0]
        assert c["kind_hint"] == "SLAB_PENETRATION", (
            f"Expected SLAB_PENETRATION with legend, got {c['kind_hint']}")


# ---------------------------------------------------------------------------
# Unit tests: _apply_multi_intent_policy for SLAB_OPENING kind
# ---------------------------------------------------------------------------

class TestSlabOpeningIntentPolicy:
    """Verify SLAB_OPENING gets proper intent and evidence in policy."""

    def _make_candidate(self, kind="SLAB_OPENING", action="opening",
                        confidence=0.88):
        from src.slab_v2.opening_resolver import _candidate
        from shapely.geometry import box
        from unittest.mock import MagicMock

        page = MagicMock()
        page.get_text_blocks.return_value = []

        return _candidate(
            "test_01_slab_opening", kind, "SLAB OPENING",
            "x_cross_vector", box(100, 100, 250, 250),
            page, confidence, action)

    def test_slab_opening_gets_slab_penetration_intent(self):
        """SLAB_OPENING kind → intent=SLAB_PENETRATION."""
        from src.slab_v2.opening_resolver import _apply_multi_intent_policy
        candidate = self._make_candidate()
        policy = _apply_multi_intent_policy([candidate])

        assert candidate["opening_intent"] == OpeningIntent.SLAB_PENETRATION.value
        assert "closed_x_cross_vector_signature" in candidate["opening_evidence_ids"]
        assert "slab_containment_guard" in candidate["opening_evidence_ids"]

    def test_slab_opening_is_cut_eligible(self):
        """SLAB_OPENING with action=opening, conf=0.88 → cut_eligible=True."""
        from src.slab_v2.opening_resolver import _apply_multi_intent_policy
        candidate = self._make_candidate()
        policy = _apply_multi_intent_policy([candidate])

        assert candidate["cut_eligible"] is True
        assert candidate["destructive_allowed"] is True
        assert candidate["id"] in policy["verified_cut_ids"]

    def test_slab_opening_review_not_cut_eligible(self):
        """SLAB_OPENING with action=review → cut_eligible=False."""
        from src.slab_v2.opening_resolver import _apply_multi_intent_policy
        candidate = self._make_candidate(action="review", confidence=0.55)
        policy = _apply_multi_intent_policy([candidate])

        assert candidate["cut_eligible"] is False


# ---------------------------------------------------------------------------
# Integration test: 2381 MSCP (the PDF that was stuck)
# ---------------------------------------------------------------------------

_MSCP_PDF = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")

@pytest.mark.skipif(not _MSCP_PDF.exists(),
                    reason="2381 MSCP test PDF not found")
class TestMscpXcrossIntegration:
    """After fix, 2381 MSCP pages 5-9 should have verified cuts."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return SlabV2Config(
            debug_images=False,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    @pytest.mark.parametrize("page_index", [4, 5, 6, 7, 8],
                             ids=["p5", "p6", "p7", "p8", "p9"])
    def test_xcross_not_stuck_in_review(self, page_index, cfg):
        """X-crosses should be classified as openings, not stuck in review."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_MSCP_PDF), page_index, cfg, use_ai=True)

        assert result.status == "OK"
        cands = list(result.opening_candidates)
        raw_cands = [c for c in cands if c["id"].startswith("raw_")]
        assert len(raw_cands) > 0, "No raw X-cross candidates found"

        opening_count = sum(1 for c in raw_cands
                           if c["default_action"] == "opening")
        assert opening_count > 0, (
            f"All {len(raw_cands)} raw X-crosses stuck in review — "
            f"fix not applied")

    @pytest.mark.parametrize("page_index", [4, 5, 6, 7, 8],
                             ids=["p5", "p6", "p7", "p8", "p9"])
    def test_openings_detected_on_loading_plans(self, page_index, cfg):
        """2381 p5-p9 are LOADING PLANs — evidence pages, not geometry
        authority (user decision 2026-07-02: cut-opening golden moves to
        the GA sheets).  Openings must still be DETECTED and classified
        here; verified cuts are only required on geometry pages."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_MSCP_PDF), page_index, cfg, use_ai=True)

        assert result.status == "OK"
        assert len(result.slabs) >= 1
        voids = [e for e in result.elements if e.type == "VOID"]
        assert voids, (
            f"Page {page_index+1}: no VOID openings detected on loading plan")


# ---------------------------------------------------------------------------
# Regression: Structural.pdf (already working) must stay working
# ---------------------------------------------------------------------------

_STRUCTURAL_PDF = Path(r"C:\Users\LENOVO\Downloads\Structural.pdf")

@pytest.mark.skipif(not _STRUCTURAL_PDF.exists(),
                    reason="Structural.pdf not found")
class TestStructuralRegressionXcross:
    """Structural.pdf has legend text → must still classify as SLAB_PENETRATION."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return SlabV2Config(
            debug_images=False,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    @pytest.mark.parametrize("page_index", [7, 9, 10],
                             ids=["p8", "p10", "p11"])
    def test_still_has_slab_penetration_kind(self, page_index, cfg):
        """With legend, X-crosses must still be SLAB_PENETRATION (not SLAB_OPENING)."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_STRUCTURAL_PDF), page_index, cfg, use_ai=True)

        assert result.status == "OK"
        cands = [c for c in result.opening_candidates if c["id"].startswith("raw_")]
        penetration_cands = [c for c in cands if c["kind_hint"] == "SLAB_PENETRATION"]
        assert len(penetration_cands) > 0, (
            f"Page {page_index+1}: no SLAB_PENETRATION candidates — regression!")

    @pytest.mark.parametrize("page_index", [7, 9, 10],
                             ids=["p8", "p10", "p11"])
    def test_still_has_verified_cuts(self, page_index, cfg):
        """Verified cuts count must not decrease."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_STRUCTURAL_PDF), page_index, cfg, use_ai=True)
        assert len(result.verified_cut_openings) >= 1

    def test_p6_steel_bracing_excluded_but_detected(self, cfg):
        """p6's X-cross sits among steel labels — excluding it from cuts is
        the CORRECT 3.6 behaviour (bracing symbol, not an opening).  The
        raw candidate must still be detected and stair evidence resolved."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_STRUCTURAL_PDF), 5, cfg, use_ai=True)
        assert result.status == "OK"
        raw = [c for c in result.opening_candidates
               if c["id"].startswith("raw_")]
        assert raw, "p6: raw X-cross candidates must still be detected"
