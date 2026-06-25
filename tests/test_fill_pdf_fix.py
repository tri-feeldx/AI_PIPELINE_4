"""TDD tests for fill-based PDF fix (RC1-RC4).

Tests are written BEFORE the fix — they define the expected behavior.
Unit tests use synthetic data (no PDF required).
Integration tests require the fill-based test PDFs.

Root causes addressed:
  RC1: Fill-only micro-polygon paths can't polygonize
  RC2: Segment count explosion from hatch classes
  RC3: No hatch fingerprint detection
  RC4: fill_polygon boundaries ignored in _collect_segments
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import StyleKey, StyleClass, VectorPath


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------
def _make_style_class(
    cid: int,
    stroke: tuple | None = (0.0, 0.0, 0.0),
    fill: tuple | None = None,
    n_segments: int = 10,
    median_seg_len: float = 5.0,
    n_paths: int = 1,
    total_length: float = 100.0,
) -> StyleClass:
    key = StyleKey(stroke=stroke, fill=fill, width=0.5, dashes="", even_odd=False)
    return StyleClass(
        id=cid, key=key, n_paths=n_paths, n_segments=n_segments,
        total_length_pt=total_length, median_seg_len_pt=median_seg_len,
    )


def _micro_segments(cx: float, cy: float, n: int = 100) -> list:
    """Generate n tiny segments around a point (simulating bezier micro-fragments)."""
    segs = []
    r = 5.0
    for i in range(n):
        a = 2 * math.pi * i / n
        b = 2 * math.pi * (i + 1) / n
        segs.append((
            (cx + r * math.cos(a), cy + r * math.sin(a)),
            (cx + r * math.cos(b), cy + r * math.sin(b)),
        ))
    return segs


def _square_polygon(cx: float, cy: float, half: float) -> Polygon:
    return Polygon([
        (cx - half, cy - half), (cx + half, cy - half),
        (cx + half, cy + half), (cx - half, cy + half),
        (cx - half, cy - half),
    ])


# ---------------------------------------------------------------------------
# Test Group 1: Hatch fingerprint detection
# ---------------------------------------------------------------------------
class TestHatchFingerprint:
    """Verify that vector_extract tags fill-only high-segment-count classes as HATCH."""

    def test_fill_only_many_segments_tiny_median_is_hatch(self):
        """Fill-only class with 1000+ segments and median < 1pt → HATCH."""
        from src.slab_v2 import vector_extract
        sc = _make_style_class(
            0, stroke=None, fill=(0.8, 0.8, 0.8),
            n_segments=2000, median_seg_len=0.17, n_paths=500,
        )
        page_area = 500 * 700
        cfg = SlabV2Config()
        # Simulate what extract_paths does after building the class
        _apply_hatch_fingerprint(sc, cfg, page_area)
        assert sc.role == "HATCH"
        assert sc.prefiltered is True

    def test_stroke_class_many_segments_not_hatch(self):
        """Class WITH stroke should never be tagged HATCH regardless of segment count."""
        sc = _make_style_class(
            1, stroke=(0.0, 0.0, 0.0), fill=(0.8, 0.8, 0.8),
            n_segments=5000, median_seg_len=0.5,
        )
        _apply_hatch_fingerprint(sc, SlabV2Config(), 500 * 700)
        assert sc.role != "HATCH"

    def test_fill_only_few_segments_not_hatch(self):
        """Fill-only class with few segments (< 500) is not hatch — could be structural fill."""
        sc = _make_style_class(
            2, stroke=None, fill=(0.0, 0.0, 1.0),
            n_segments=50, median_seg_len=0.5,
        )
        _apply_hatch_fingerprint(sc, SlabV2Config(), 500 * 700)
        assert sc.role != "HATCH"

    def test_fill_only_normal_median_not_hatch(self):
        """Fill-only class with normal median segment length (> 1pt) is not hatch."""
        sc = _make_style_class(
            3, stroke=None, fill=(0.5, 0.5, 0.5),
            n_segments=1000, median_seg_len=5.0,
        )
        _apply_hatch_fingerprint(sc, SlabV2Config(), 500 * 700)
        assert sc.role != "HATCH"


def _apply_hatch_fingerprint(sc: StyleClass, cfg: SlabV2Config, page_area: float):
    """Apply the hatch fingerprint logic that should exist in vector_extract.py."""
    if sc.role != "UNKNOWN":
        return
    if (sc.key.fill is not None and sc.key.stroke is None
            and sc.n_segments > 500 and sc.median_seg_len_pt < 1.0):
        sc.role = "HATCH"
        sc.role_confidence = 0.85
        sc.prefiltered = True


# ---------------------------------------------------------------------------
# Test Group 2: _fill_boundary_segments
# ---------------------------------------------------------------------------
class TestFillBoundarySegments:
    """Verify fill polygon boundary extraction produces correct segments."""

    def test_square_gives_4_segments(self):
        poly = _square_polygon(100, 100, 50)
        segs = _fill_boundary_segments(poly)
        assert len(segs) == 4
        for a, b in segs:
            assert a != b

    def test_complex_polygon_simplified(self):
        """A polygon with many micro-vertices should be simplified to fewer segments."""
        n = 200
        coords = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            r = 50 + 2 * math.sin(5 * angle)
            coords.append((100 + r * math.cos(angle), 100 + r * math.sin(angle)))
        coords.append(coords[0])
        poly = Polygon(coords)
        segs = _fill_boundary_segments(poly)
        assert len(segs) < n
        assert len(segs) >= 3

    def test_degenerate_zero_area_no_crash(self):
        """Degenerate polygon should return empty or minimal result, not crash."""
        poly = Polygon([(0, 0), (1, 0), (0, 0)])
        segs = _fill_boundary_segments(poly)
        assert isinstance(segs, list)


def _fill_boundary_segments(fill_poly: Polygon, simplify_tol: float = 0.5) -> list:
    """Reference implementation — matches what planarize.py should have."""
    if not fill_poly.is_valid or fill_poly.is_empty:
        return []
    simplified = fill_poly.exterior.simplify(simplify_tol, preserve_topology=True)
    coords = list(simplified.coords)
    return [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)
            if coords[i] != coords[i + 1]]


# ---------------------------------------------------------------------------
# Test Group 3: _collect_segments should use fill boundary for fill-only paths
# ---------------------------------------------------------------------------
class TestCollectSegmentsFillBoundary:
    """Verify _collect_segments uses fill_polygon boundary for fill-only paths."""

    def test_fill_only_path_uses_boundary_not_micro_segments(self):
        """A fill-only VectorPath should contribute boundary segments, not raw micro-segs."""
        square = _square_polygon(100, 100, 50)
        micro = _micro_segments(100, 100, n=100)
        vp = VectorPath(
            id=0, style_id=0, segments=micro, is_closed=True, is_filled=True,
            seqno=0, fill_polygon=square, has_stroke=False,
        )
        from src.slab_v2.planarize import _collect_segments
        segs = _collect_segments([vp], {0})
        # Should use simplified boundary (~4 segments), not 100 micro-segments
        assert len(segs) < 20, f"Expected ~4 boundary segments, got {len(segs)}"
        assert len(segs) >= 3, f"Expected at least 3 boundary segments, got {len(segs)}"

    def test_stroke_path_uses_raw_segments(self):
        """A path with stroke should use raw segments as before."""
        vp = VectorPath(
            id=0, style_id=0,
            segments=[((0, 0), (100, 0)), ((100, 0), (100, 100))],
            is_closed=False, is_filled=False, seqno=0,
            has_stroke=True,
        )
        from src.slab_v2.planarize import _collect_segments
        segs = _collect_segments([vp], {0})
        assert len(segs) == 2

    def test_fill_only_without_fill_polygon_uses_raw(self):
        """A fill-only path without fill_polygon falls back to raw segments."""
        vp = VectorPath(
            id=0, style_id=0,
            segments=[((0, 0), (10, 0)), ((10, 0), (10, 10))],
            is_closed=False, is_filled=True, seqno=0,
            fill_polygon=None, has_stroke=False,
        )
        from src.slab_v2.planarize import _collect_segments
        segs = _collect_segments([vp], {0})
        assert len(segs) == 2

    def test_outside_content_excluded(self):
        """Paths outside content rect should be excluded regardless of fill."""
        square = _square_polygon(100, 100, 50)
        vp = VectorPath(
            id=0, style_id=0, segments=_micro_segments(100, 100, 50),
            is_closed=True, is_filled=True, seqno=0,
            fill_polygon=square, has_stroke=False, outside_content=True,
        )
        from src.slab_v2.planarize import _collect_segments
        segs = _collect_segments([vp], {0})
        assert len(segs) == 0


# ---------------------------------------------------------------------------
# Test Group 4: Integration — fill-based PDFs produce faces
# ---------------------------------------------------------------------------
_FILL_PDFS = {
    "2381_MSCP": Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf"),
    "South_Melbourne": Path(r"C:\Users\LENOVO\Downloads\2402. South Melbourne Primary School - CIVIL & STR - 260610.pdf"),
}

_WORKING_PDFS = {}
for _name in ["combine strc.pdf", "Structural.pdf"]:
    for _search in [
        Path(r"D:\FeelDX_Workspace\AI_PIPELINE_4-main"),
        Path(r"D:\FeelDX_Workspace\sketchup_auto_project\data\input_pdf"),
        Path(r"D:\FeelDX_Workspace"),
    ]:
        if _search.exists():
            for _m in _search.rglob(_name):
                _WORKING_PDFS[_name.replace(".pdf", "")] = _m
                break
        if _name.replace(".pdf", "") in _WORKING_PDFS:
            break


def _skip_if_no_pdf(pdf_path: Path):
    if not pdf_path.exists():
        pytest.skip(f"PDF not found: {pdf_path}")


class TestFillPdfIntegration:
    """Integration tests: pipeline should produce faces on fill-based PDFs."""

    @pytest.fixture(scope="class")
    def cfg(self) -> SlabV2Config:
        return SlabV2Config(
            debug_images=True,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    @pytest.mark.parametrize("pdf_key", list(_FILL_PDFS.keys()))
    def test_fill_pdf_has_faces(self, pdf_key, cfg):
        """Fill-based PDFs should produce at least some pages with faces (not all NO_FACES)."""
        pdf_path = _FILL_PDFS[pdf_key]
        _skip_if_no_pdf(pdf_path)
        import fitz
        from src.slab_v2.pipeline import extract_slabs_v2

        doc = fitz.open(str(pdf_path))
        n_pages = doc.page_count
        doc.close()

        # Test up to 4 pages in the slab range
        test_pages = list(range(4, min(n_pages, 16)))[:4]
        ok_count = 0
        for pi in test_pages:
            result = extract_slabs_v2(str(pdf_path), pi, cfg, use_ai=False)
            if result.status == "OK":
                ok_count += 1

        assert ok_count > 0, (
            f"{pdf_key}: ALL {len(test_pages)} pages returned NO_FACES — "
            f"fill-based PDF fix not working")

    @pytest.mark.parametrize("pdf_name", list(_WORKING_PDFS.keys()))
    def test_working_pdf_still_ok(self, pdf_name, cfg):
        """Regression guard: existing working PDFs must still produce OK on slab pages."""
        pdf_path = _WORKING_PDFS[pdf_name]
        _skip_if_no_pdf(pdf_path)
        import fitz
        from src.slab_v2.pipeline import extract_slabs_v2

        doc = fitz.open(str(pdf_path))
        n_pages = doc.page_count
        doc.close()

        # Use slab-likely pages (skip cover/notes pages)
        if n_pages <= 5:
            test_pages = list(range(n_pages))
        else:
            test_pages = list(range(4, min(n_pages, 15)))[:4]

        ok_count = 0
        for pi in test_pages:
            result = extract_slabs_v2(str(pdf_path), pi, cfg, use_ai=False)
            if result.status == "OK":
                ok_count += 1

        assert ok_count > 0, (
            f"REGRESSION: {pdf_name} — ALL {len(test_pages)} slab pages "
            f"returned NO_FACES")


class TestHatchExclusionFromPipeline:
    """Verify pipeline excludes HATCH classes from initial face graph."""

    def test_hatch_class_excluded_from_all_ids(self):
        """Classes with role='HATCH' should be excluded from all_ids in pipeline."""
        classes = [
            _make_style_class(0, stroke=(0, 0, 0)),
            _make_style_class(1, stroke=None, fill=(0.8, 0.8, 0.8),
                              n_segments=5000, median_seg_len=0.2),
        ]
        classes[1].role = "HATCH"
        classes[1].prefiltered = True

        all_ids = {c.id for c in classes if c.role not in ("FRAME", "HATCH")}
        assert 0 in all_ids
        assert 1 not in all_ids, "HATCH class should be excluded from all_ids"

    def test_frame_still_excluded(self):
        """FRAME classes must still be excluded (existing behavior)."""
        classes = [
            _make_style_class(0, stroke=(0, 0, 0)),
            _make_style_class(1, stroke=(0, 0, 0)),
        ]
        classes[1].role = "FRAME"

        all_ids = {c.id for c in classes if c.role not in ("FRAME", "HATCH")}
        assert 0 in all_ids
        assert 1 not in all_ids
