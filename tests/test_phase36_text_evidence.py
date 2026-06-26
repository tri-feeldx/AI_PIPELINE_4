"""TDD tests for Phase 3.6 Fix C: Relaxed face fallback for VOID/PENETRATION.

Problem: CW2 shaft has stair (upper) + VOID (lower). The VOID portion has
non-standard diagonal geometry (triangular fan, not clean 2-diagonal X-cross).
Tier 1 geometry detection CANNOT find it. Need text "VOID" + face fallback
with relaxed thresholds (400mm min side instead of 1200mm).

User requirement: CUT only the VOID portion, keep the stair portion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest
from shapely.geometry import box, Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config


def _make_face(x0, y0, x1, y1):
    f = MagicMock()
    f.polygon = box(x0, y0, x1, y1)
    f.area_pt2 = f.polygon.area
    return f


def _make_fg(faces):
    fg = MagicMock()
    fg.faces = faces
    return fg


def _make_page_with_words(words):
    """Create a mock page that returns words from get_text('words').

    words: list of (x0, y0, x1, y1, text) tuples.
    """
    page = MagicMock()
    word_tuples = [(x0, y0, x1, y1, text, 0, 0, 0)
                   for x0, y0, x1, y1, text in words]
    page.get_text = MagicMock(return_value=word_tuples)
    return page


# ---------------------------------------------------------------------------
# Unit: Relaxed face fallback thresholds for VOID
# ---------------------------------------------------------------------------

class TestVoidFaceFallback:
    """VOID text should find smaller faces (400mm min) than STAIR (1200mm min)."""

    def test_void_finds_500mm_face(self):
        """A 500mm face near VOID text should be found (400mm min for VOID)."""
        from src.slab_v2.elements import extract_elements

        scale = 100
        to_mm = 25.4 / 72 * scale
        side_mm = 500
        side_pt = side_mm / to_mm

        face = _make_face(100, 100, 100 + side_pt, 100 + side_pt)
        fg = _make_fg([face])

        page = _make_page_with_words([
            (100, 95, 120, 102, "VOID"),
        ])

        cfg = SlabV2Config(
            void_fallback_min_side_mm=400.0,
            text_evidence_search_radius_pt=120.0,
        )
        content_rect = fitz.Rect(0, 0, 500, 500)
        content_area = 500 * 500

        elems, warnings = extract_elements(
            page, fg, cfg, content_rect, content_area,
            paths=[], scale=scale,
        )

        void_elems = [e for e in elems if e.type == "VOID"]
        assert len(void_elems) >= 1, (
            f"Expected VOID element from 500mm face, got {len(void_elems)}. "
            f"All elements: {[(e.type, e.label) for e in elems]}")

    def test_stair_rejects_500mm_face(self):
        """A 500mm face near STAIR text should NOT be found (1200mm min for STAIR)."""
        from src.slab_v2.elements import extract_elements

        scale = 100
        to_mm = 25.4 / 72 * scale
        side_mm = 500
        side_pt = side_mm / to_mm

        face = _make_face(100, 100, 100 + side_pt, 100 + side_pt)
        fg = _make_fg([face])

        page = _make_page_with_words([
            (100, 95, 140, 102, "STAIR"),
        ])

        cfg = SlabV2Config(
            void_fallback_min_side_mm=400.0,
            text_evidence_search_radius_pt=120.0,
        )
        content_rect = fitz.Rect(0, 0, 500, 500)
        content_area = 500 * 500

        elems, warnings = extract_elements(
            page, fg, cfg, content_rect, content_area,
            paths=[], scale=scale,
        )

        stair_elems = [e for e in elems if e.type == "STAIR"]
        assert len(stair_elems) == 0, (
            f"STAIR should reject 500mm face (needs 1200mm+), "
            f"got {len(stair_elems)}")

    def test_void_wider_search_radius(self):
        """VOID should search 120pt radius, not default 80pt."""
        from src.slab_v2.elements import extract_elements

        scale = 100
        to_mm = 25.4 / 72 * scale
        side_mm = 600
        side_pt = side_mm / to_mm

        # Face is 100pt away from text (beyond default 80pt, within 120pt)
        face = _make_face(200, 100, 200 + side_pt, 100 + side_pt)
        fg = _make_fg([face])

        page = _make_page_with_words([
            (80, 95, 100, 102, "VOID"),
        ])

        cfg = SlabV2Config(
            void_fallback_min_side_mm=400.0,
            text_evidence_search_radius_pt=120.0,
        )
        content_rect = fitz.Rect(0, 0, 500, 500)
        content_area = 500 * 500

        elems, warnings = extract_elements(
            page, fg, cfg, content_rect, content_area,
            paths=[], scale=scale,
        )

        void_elems = [e for e in elems if e.type == "VOID"]
        assert len(void_elems) >= 1, (
            f"Expected VOID element with 120pt search radius, got {len(void_elems)}")


# ---------------------------------------------------------------------------
# Unit: Config parameters exist
# ---------------------------------------------------------------------------

class TestConfigParams:
    """Config should have void fallback parameters."""

    def test_void_fallback_min_side_default(self):
        cfg = SlabV2Config()
        assert hasattr(cfg, "void_fallback_min_side_mm")
        assert cfg.void_fallback_min_side_mm == 400.0

    def test_text_evidence_search_radius_default(self):
        cfg = SlabV2Config()
        assert hasattr(cfg, "text_evidence_search_radius_pt")
        assert cfg.text_evidence_search_radius_pt == 120.0


# ---------------------------------------------------------------------------
# Integration: Combined Structural p18 CW2 VOID detection
# ---------------------------------------------------------------------------

_COMBINED_PDF = Path(r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")


@pytest.mark.skipif(not _COMBINED_PDF.exists(),
                    reason="Combined Structural PDF not found")
class TestP18CW2Void:
    """p18 CW2 VOID should be detected via text evidence + face fallback."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return SlabV2Config(
            debug_images=False,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    def test_p18_has_void_candidate_in_cw2(self, cfg):
        """p18 should have >=2 verified cuts (CW1 LIFT VOID + CW2 VOID).

        Note: Top VOID and ON HOLD are already excluded from slab by AI face
        selection (slab has holes there), so they don't appear as elements.
        The slab assembly on this page covers ~1% — a separate issue.
        """
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_COMBINED_PDF), 17, cfg, use_ai=True)
        assert result.status == "OK"
        assert len(result.verified_cut_openings) >= 2, (
            f"Expected >=2 verified cuts, got {len(result.verified_cut_openings)}")
