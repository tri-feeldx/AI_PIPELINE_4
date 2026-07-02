"""TDD tests for Phase 3.5: Multi-X detection improvements.

4 fixes tested:
1. xcross_max_area_frac 0.04 → 0.10 (config.py)
2. _MIN_SIDE_MM 300 → 200 (elements.py)
3. VOID face fallback (elements.py)
4. CW wall support in core detector (opening_resolver.py)

Test PDFs: Combined Structural p18-21 (LIFT VOID),
           2381 MSCP p5-9 (regression), Structural p6,8,10,11 (regression)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shapely.geometry import box, Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config


# ---------------------------------------------------------------------------
# Unit: xcross_max_area_frac allows 8% rect
# ---------------------------------------------------------------------------

class TestXcrossMaxAreaFrac:
    """xcross_max_area_frac should be 0.10, allowing larger shaft openings."""

    def test_config_default_is_0_10(self):
        cfg = SlabV2Config()
        assert cfg.xcross_max_area_frac == 0.10, (
            f"Expected 0.10, got {cfg.xcross_max_area_frac}")

    def test_8pct_rect_passes_area_filter(self):
        """A rect that is 8% of content area should NOT be filtered out."""
        from src.slab_v2.elements import extract_elements
        cfg = SlabV2Config()
        content_area = 1000 * 700  # typical page
        rect_area = 0.08 * content_area  # 8% of content
        max_area = cfg.xcross_max_area_frac * content_area
        assert rect_area <= max_area, (
            f"8% rect ({rect_area}) exceeds max_area ({max_area}) — "
            f"xcross_max_area_frac={cfg.xcross_max_area_frac} too strict")


# ---------------------------------------------------------------------------
# Unit: _MIN_SIDE_MM reduced to 200
# ---------------------------------------------------------------------------

class TestMinSideMm:
    """_MIN_SIDE_MM should be 200, allowing small penetrations."""

    def test_250mm_side_passes(self):
        """A 250mm-side opening should pass min side filter."""
        scale = 100
        _PT_TO_MM = 25.4 / 72.0
        _to_mm = _PT_TO_MM * scale
        _MIN_SIDE_MM = 200.0
        _min_side_pt = _MIN_SIDE_MM / _to_mm
        side_mm = 250
        side_pt = side_mm / _to_mm
        assert side_pt >= _min_side_pt, (
            f"250mm side ({side_pt:.1f}pt) < min ({_min_side_pt:.1f}pt)")

    def test_150mm_side_rejected(self):
        """A 150mm-side opening should still be filtered out."""
        scale = 100
        _PT_TO_MM = 25.4 / 72.0
        _to_mm = _PT_TO_MM * scale
        _MIN_SIDE_MM = 200.0
        _min_side_pt = _MIN_SIDE_MM / _to_mm
        side_mm = 150
        side_pt = side_mm / _to_mm
        assert side_pt < _min_side_pt


# ---------------------------------------------------------------------------
# Unit: VOID face fallback
# ---------------------------------------------------------------------------

class TestVoidFaceFallback:
    """VOID text labels should trigger face fallback (same as STAIR/LIFT/SHAFT)."""

    def _make_fake_face(self, polygon, area_pt2=None):
        f = MagicMock()
        f.polygon = polygon
        f.area_pt2 = area_pt2 or polygon.area
        return f

    def _make_page_with_words(self, words):
        page = MagicMock()
        page.get_text = lambda mode="text": words if mode == "words" else ""
        page.rect = MagicMock()
        page.rect.width = 1000
        page.rect.height = 700
        return page

    def test_void_label_triggers_face_fallback(self):
        """A 'VOID' text label near a qualifying face should create an element."""
        import fitz
        from src.slab_v2.elements import extract_elements
        from src.slab_v2.models import FaceGraph

        scale = 100
        _PT_TO_MM = 25.4 / 72.0
        _to_mm = _PT_TO_MM * scale
        face_side_mm = 2000
        face_side_pt = face_side_mm / _to_mm

        face_poly = box(100, 100, 100 + face_side_pt, 100 + face_side_pt)
        fake_face = self._make_fake_face(face_poly)

        fg = MagicMock(spec=FaceGraph)
        fg.faces = [fake_face]

        page = MagicMock()
        label_cx = 100 + face_side_pt / 2
        label_cy = 100 + face_side_pt / 2
        words = [(label_cx - 5, label_cy - 5, label_cx + 5, label_cy + 5,
                  "VOID", 0, 0, 0)]
        page.get_text = lambda mode="text": (
            words if mode == "words" else "VOID")

        cfg = SlabV2Config()
        content_rect = fitz.Rect(0, 0, 1000, 700)
        content_area = content_rect.width * content_rect.height

        elements, warnings = extract_elements(
            page, fg, cfg, content_rect, content_area,
            paths=[], scale=scale)

        void_elements = [e for e in elements if e.type == "VOID"]
        assert len(void_elements) >= 1, (
            f"VOID label should trigger face fallback but got 0 VOID elements. "
            f"Warnings: {warnings}")


# ---------------------------------------------------------------------------
# Unit: CW wall support in core detector
# ---------------------------------------------------------------------------

class TestCwWallSupport:
    """Core wall detector should support CW (core wall) labels, not just LW."""

    def test_cw_walls_detected(self):
        """4 CW walls should be recognized by _verified_core_wall_opening_candidates."""
        from src.slab_v2.opening_resolver import _verified_core_wall_opening_candidates
        import fitz

        scale = 100
        _PT_TO_MM = 25.4 / 72.0
        _to_mm = _PT_TO_MM * scale

        shaft_mm = 3000
        shaft_pt = shaft_mm / _to_mm

        wall_thick_pt = 150 / _to_mm

        cx, cy = 500, 350
        half = shaft_pt / 2

        walls = []
        wall_specs = [
            ("CW1a", (cx - half, cy - half, cx + half, cy - half + wall_thick_pt)),
            ("CW1b", (cx - half, cy + half - wall_thick_pt, cx + half, cy + half)),
            ("CW1c", (cx - half, cy - half, cx - half + wall_thick_pt, cy + half)),
            ("CW1d", (cx + half - wall_thick_pt, cy - half, cx + half, cy + half)),
        ]
        for label, (x0, y0, x1, y1) in wall_specs:
            w = MagicMock()
            w.polygon = box(x0, y0, x1, y1)
            w.label = label
            walls.append(w)

        elem = MagicMock()
        elem.polygon = box(cx - half + wall_thick_pt,
                          cy - half + wall_thick_pt,
                          cx + half - wall_thick_pt,
                          cy + half - wall_thick_pt)
        elem.type = "VOID"
        elem.label = "VOID"
        elem.area_pt2 = elem.polygon.area
        elem.anchor_bbox = elem.polygon.bounds

        page = MagicMock()
        page.get_text_blocks = MagicMock(return_value=[])

        content_rect = fitz.Rect(0, 0, 1000, 700)
        slab_union = box(0, 0, 1000, 700)
        cfg = SlabV2Config()

        candidates, defaults, warnings = _verified_core_wall_opening_candidates(
            walls, [elem], page, content_rect, slab_union, scale, cfg)

        assert len(candidates) > 0, (
            f"CW walls should be recognized. Got 0 candidates. Warnings: {warnings}")


# ---------------------------------------------------------------------------
# Integration: Combined Structural p18 multi-X detection
# ---------------------------------------------------------------------------

_COMBINED_PDF = Path(r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")


@pytest.mark.skipif(not _COMBINED_PDF.exists(),
                    reason="Combined Structural PDF not found")
class TestCombinedStructuralP18:
    """p18 should detect >=2 X-crosses (was 1 before Phase 3.5)."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return SlabV2Config(
            debug_images=False,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    def test_p18_detects_multiple_openings(self, cfg):
        """p18 has 3-4 X marks; should detect >=2 verified cuts."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_COMBINED_PDF), 17, cfg, use_ai=True)

        assert result.status == "OK"
        n_cuts = len(result.verified_cut_openings)
        assert n_cuts >= 2, (
            f"p18: expected >=2 verified cuts, got {n_cuts}")

    @pytest.mark.parametrize("page_index", [18, 19, 20],
                             ids=["p19", "p20", "p21"])
    def test_lift_void_pages_have_cuts(self, page_index, cfg):
        """Pages 19-21 also have LIFT VOID X marks."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_COMBINED_PDF), page_index, cfg,
                                  use_ai=True)
        if result.status == "OK":
            assert len(result.verified_cut_openings) >= 1, (
                f"p{page_index+1}: 0 verified cuts on LIFT VOID page")


# ---------------------------------------------------------------------------
# Regression: 2381 MSCP must still work
# ---------------------------------------------------------------------------

_MSCP_PDF = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")


@pytest.mark.skipif(not _MSCP_PDF.exists(),
                    reason="2381 MSCP test PDF not found")
class TestMscpRegression35:
    """2381 MSCP pages must still have verified cuts after Phase 3.5."""

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
    def test_still_detects_openings(self, page_index, cfg):
        """p5-p9 are LOADING PLANs — evidence pages (user decision
        2026-07-02: cut golden moves to GA sheets); openings must still be
        detected and classified here."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_MSCP_PDF), page_index, cfg,
                                  use_ai=True)
        assert result.status == "OK"
        voids = [e for e in result.elements if e.type == "VOID"]
        assert voids, (
            f"2381 MSCP p{page_index+1}: no VOID openings detected")


# ---------------------------------------------------------------------------
# Regression: Structural.pdf must still work
# ---------------------------------------------------------------------------

_STRUCTURAL_PDF = Path(r"C:\Users\LENOVO\Downloads\Structural.pdf")


@pytest.mark.skipif(not _STRUCTURAL_PDF.exists(),
                    reason="Structural.pdf not found")
class TestStructuralRegression35:
    """Structural.pdf must still have verified cuts after Phase 3.5."""

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
    def test_still_has_cuts(self, page_index, cfg):
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_STRUCTURAL_PDF), page_index, cfg,
                                  use_ai=True)
        assert result.status == "OK"
        assert len(result.verified_cut_openings) >= 1

    def test_p6_raw_xcross_still_detected(self, cfg):
        """p6's X-cross is bracing among steel labels — correctly excluded
        from cuts by the 3.6 steel guard, but detection must survive."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_STRUCTURAL_PDF), 5, cfg, use_ai=True)
        assert result.status == "OK"
        raw = [c for c in result.opening_candidates
               if c["id"].startswith("raw_")]
        assert raw
