"""Unit tests for X-cross (slab penetration) detection.

All tests use synthetic data — no PDF files required.
Tests verify the deterministic geometry algorithms in elements.py:
  _diagonal_segments()  — filters segments by angle
  _detect_xcross_rects() — pairs crossing diagonals into rectangles
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pytest
from shapely.geometry import Polygon

from src.slab_v2.elements import _diagonal_segments, _detect_xcross_rects


# ---------------------------------------------------------------------------
# Synthetic VectorPath (mirrors src.slab_v2.models.VectorPath)
# ---------------------------------------------------------------------------
@dataclass
class FakePath:
    id: int = 0
    style_id: int = 0
    segments: list = field(default_factory=list)
    is_closed: bool = False
    is_filled: bool = False
    seqno: int = 0
    fill_polygon: Optional[Polygon] = None
    outside_content: bool = False
    layer: str = ""


def _make_x_paths(cx: float, cy: float, half_w: float, half_h: float) -> list[FakePath]:
    """Create 2 diagonal segments forming an X centered at (cx, cy)."""
    return [
        FakePath(segments=[
            ((cx - half_w, cy - half_h), (cx + half_w, cy + half_h)),
        ]),
        FakePath(segments=[
            ((cx - half_w, cy + half_h), (cx + half_w, cy - half_h)),
        ]),
    ]


def _make_hv_paths(x0, y0, x1, y1) -> list[FakePath]:
    """Create horizontal and vertical segments (NOT diagonal)."""
    return [
        FakePath(segments=[((x0, y0), (x1, y0))]),  # horizontal
        FakePath(segments=[((x0, y0), (x0, y1))]),  # vertical
    ]


# ---------------------------------------------------------------------------
# _diagonal_segments tests
# ---------------------------------------------------------------------------
class TestDiagonalSegments:
    def test_45_degree_diag_is_included(self):
        paths = [FakePath(segments=[((0, 0), (100, 100))])]
        diags = _diagonal_segments(paths)
        assert len(diags) == 1

    def test_horizontal_line_excluded(self):
        paths = [FakePath(segments=[((0, 0), (100, 0))])]
        diags = _diagonal_segments(paths)
        assert len(diags) == 0

    def test_vertical_line_excluded(self):
        paths = [FakePath(segments=[((0, 0), (0, 100))])]
        diags = _diagonal_segments(paths)
        assert len(diags) == 0

    def test_near_horizontal_14deg_excluded(self):
        # 14 degrees — below _DIAG_MIN_DEG = 15
        dx = 100
        dy = dx * math.tan(math.radians(14))
        paths = [FakePath(segments=[((0, 0), (dx, dy))])]
        diags = _diagonal_segments(paths)
        assert len(diags) == 0

    def test_near_vertical_76deg_excluded(self):
        # 76 degrees — above _DIAG_MAX_DEG = 75
        dx = 100
        dy = dx * math.tan(math.radians(76))
        paths = [FakePath(segments=[((0, 0), (dx, dy))])]
        diags = _diagonal_segments(paths)
        assert len(diags) == 0

    def test_30_degree_included(self):
        dx = 100
        dy = dx * math.tan(math.radians(30))
        paths = [FakePath(segments=[((0, 0), (dx, dy))])]
        diags = _diagonal_segments(paths)
        assert len(diags) == 1

    def test_very_short_segment_excluded(self):
        # length < 2.0 pt
        paths = [FakePath(segments=[((0, 0), (1, 1))])]
        diags = _diagonal_segments(paths)
        assert len(diags) == 0

    def test_outside_content_excluded(self):
        paths = [FakePath(
            segments=[((0, 0), (100, 100))],
            outside_content=True)]
        diags = _diagonal_segments(paths)
        assert len(diags) == 0

    def test_multiple_segments_in_one_path(self):
        paths = [FakePath(segments=[
            ((0, 0), (50, 50)),       # diagonal
            ((50, 50), (100, 50)),    # horizontal
            ((100, 50), (150, 100)),  # diagonal
        ])]
        diags = _diagonal_segments(paths)
        assert len(diags) == 2


# ---------------------------------------------------------------------------
# _detect_xcross_rects tests
# ---------------------------------------------------------------------------
class TestDetectXcrossRects:
    def test_two_crossing_diags_produce_one_rect(self):
        """Standard X: two 45° diags crossing near midpoint → 1 rectangle."""
        # At scale 100, 1pt = 35.28mm. A 10pt × 10pt X = 352.8mm side — valid
        paths = _make_x_paths(250, 250, 5, 5)
        diags = _diagonal_segments(paths)
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=500 * 500)
        assert len(rects) == 1
        r = rects[0]
        assert r.geom_type == "Polygon"
        assert r.area > 0

    def test_parallel_diags_no_crossing(self):
        """Two parallel diagonal lines should NOT produce an X-cross."""
        paths = [
            FakePath(segments=[((0, 0), (100, 100))]),
            FakePath(segments=[((10, 0), (110, 100))]),
        ]
        diags = _diagonal_segments(paths)
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=500 * 500)
        assert len(rects) == 0

    def test_hv_lines_no_xcross(self):
        """Horizontal and vertical lines → no X-cross."""
        paths = _make_hv_paths(100, 100, 200, 200)
        diags = _diagonal_segments(paths)
        assert len(diags) == 0
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=500 * 500)
        assert len(rects) == 0

    def test_x_too_small_filtered_out(self):
        """X with sides < 250mm real → should be filtered.
        At scale 100: 250mm / 35.28 = 7.09pt diagonal minimum.
        half-diagonal = 250*sqrt(2)/2 / 35.28 = 5.01pt.
        Make X with 2pt half → 4pt side → ~141mm real → too small.
        """
        paths = _make_x_paths(250, 250, 2, 2)
        diags = _diagonal_segments(paths)
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=500 * 500)
        assert len(rects) == 0

    def test_x_too_large_filtered_out(self):
        """X with sides > 4000mm real → should be filtered.
        At scale 100: 4000mm / 35.28 = 113.4pt.
        half = 80pt → side = 160pt → 5644mm → too large.
        """
        paths = _make_x_paths(250, 250, 80, 80)
        diags = _diagonal_segments(paths)
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=1000 * 1000)
        assert len(rects) == 0

    def test_two_separate_x_produce_two_rects(self):
        """Two X-crosses far apart → 2 separate rectangles."""
        # X1 at (100, 100), X2 at (400, 400), half=5pt each
        paths = _make_x_paths(100, 100, 5, 5) + _make_x_paths(400, 400, 5, 5)
        diags = _diagonal_segments(paths)
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=600 * 600)
        assert len(rects) == 2

    def test_overlapping_x_dedup_handled_downstream(self):
        """Two X-crosses at same position — _detect_xcross_rects may find both;
        dedup happens in extract_elements(), not here. This tests raw detection."""
        paths = _make_x_paths(200, 200, 5, 5) + _make_x_paths(200, 200, 5.5, 5.5)
        diags = _diagonal_segments(paths)
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=500 * 500)
        # May produce 1 or 2 depending on pairing — both are valid
        assert len(rects) >= 1

    def test_valid_x_rect_bounds_match_endpoints(self):
        """The detected rectangle should tightly bound the diagonal endpoints."""
        cx, cy, hw, hh = 250, 250, 8, 6
        paths = _make_x_paths(cx, cy, hw, hh)
        diags = _diagonal_segments(paths)
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=500 * 500)
        assert len(rects) == 1
        bx = rects[0].bounds
        assert abs(bx[0] - (cx - hw)) < 1.0  # xmin
        assert abs(bx[1] - (cy - hh)) < 1.0  # ymin
        assert abs(bx[2] - (cx + hw)) < 1.0  # xmax
        assert abs(bx[3] - (cy + hh)) < 1.0  # ymax

    def test_aspect_ratio_5_to_1_rejected(self):
        """An X-cross with aspect > 5:1 should be rejected (not a valid opening)."""
        # half_w=5, half_h=30 → side 10 vs 60 → ratio 6:1
        paths = _make_x_paths(250, 250, 5, 30)
        diags = _diagonal_segments(paths)
        rects = _detect_xcross_rects(diags, scale=100, content_area_pt2=500 * 500)
        assert len(rects) == 0

    def test_different_scales(self):
        """Same physical X at different scales: scale 50 vs 200.
        At scale 50: 1pt = 17.64mm; 10pt side = 176mm → too small (< 200mm).
        At scale 200: 1pt = 70.56mm; 10pt side = 706mm → valid.
        """
        paths = _make_x_paths(250, 250, 5, 5)

        diags = _diagonal_segments(paths)
        rects_50 = _detect_xcross_rects(diags, scale=50, content_area_pt2=500 * 500)
        rects_200 = _detect_xcross_rects(diags, scale=200, content_area_pt2=500 * 500)
        # At scale 50, the X is too small physically
        assert len(rects_50) == 0
        # At scale 200, the X is valid
        assert len(rects_200) == 1
