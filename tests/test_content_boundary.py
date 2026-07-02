"""Tests for smart content boundary closing in planarize.

When slab boundary lines terminate at the content_rect edge (common in
multi-sheet structural plans), the face graph should inject content_rect
edges to close the boundary and produce large faces.
"""
from __future__ import annotations

import fitz
import pytest
from shapely.geometry import box
from unittest.mock import MagicMock

from src.slab_v2.config import SlabV2Config


# ── _boundary_edges_to_inject tests ─────────────────────────────────────────

class TestBoundaryEdgesToInject:

    def _call(self, segments, content_rect, proximity_pt=3.0, min_touches=2):
        from src.slab_v2.planarize import _boundary_edges_to_inject
        return _boundary_edges_to_inject(
            segments, content_rect,
            proximity_pt=proximity_pt, min_touches=min_touches)

    def _rect(self, x0=0, y0=0, x1=1000, y1=700):
        return fitz.Rect(x0, y0, x1, y1)

    def test_L_shape_touches_left_and_bottom(self):
        """L-shape: vertical line on left edge + horizontal on bottom edge.
        Both edges should be injected."""
        rect = self._rect()
        segments = [
            # vertical line on left edge (x=0)
            ((0.0, 100.0), (0.0, 600.0)),
            ((0.0, 600.0), (500.0, 600.0)),
            # horizontal line on bottom edge (y=700)
            ((500.0, 600.0), (500.0, 700.0)),
            ((500.0, 700.0), (900.0, 700.0)),
        ]
        edges = self._call(segments, rect)
        # left edge: endpoints at x=0 → touches left
        # bottom edge: endpoints at y=700 → touches bottom
        assert len(edges) >= 2

    def test_closed_box_no_injection(self):
        """A closed rectangle in the middle → no endpoints near edges."""
        rect = self._rect()
        segments = [
            ((200.0, 200.0), (800.0, 200.0)),
            ((800.0, 200.0), (800.0, 500.0)),
            ((800.0, 500.0), (200.0, 500.0)),
            ((200.0, 500.0), (200.0, 200.0)),
        ]
        edges = self._call(segments, rect)
        assert len(edges) == 0

    def test_none_content_rect(self):
        """content_rect=None → empty list."""
        edges = self._call([((0, 0), (100, 0))], None)
        assert edges == []

    def test_min_touches_threshold(self):
        """Only 1 endpoint near an edge (below min_touches=2) → no injection."""
        rect = self._rect()
        # single endpoint at x=0
        segments = [((0.0, 350.0), (500.0, 350.0))]
        edges = self._call(segments, rect, min_touches=2)
        assert len(edges) == 0

    def test_proximity_within_range(self):
        """Endpoint 2pt from edge → within proximity_pt=3 → detected."""
        rect = self._rect()
        segments = [
            ((2.0, 100.0), (2.0, 600.0)),   # near left edge (x=0, dist=2)
            ((1.0, 300.0), (500.0, 300.0)),  # another near left
        ]
        edges = self._call(segments, rect, proximity_pt=3.0, min_touches=2)
        left_edge = ((0.0, 700.0), (0.0, 0.0))
        has_left = any(
            abs(e[0][0] - rect.x0) < 0.1 and abs(e[1][0] - rect.x0) < 0.1
            for e in edges)
        assert has_left

    def test_proximity_out_of_range(self):
        """Endpoint 5pt from edge → outside proximity_pt=3 → not detected."""
        rect = self._rect()
        segments = [
            ((5.0, 100.0), (5.0, 600.0)),
            ((5.0, 300.0), (500.0, 300.0)),
        ]
        edges = self._call(segments, rect, proximity_pt=3.0, min_touches=2)
        has_left = any(
            abs(e[0][0] - rect.x0) < 0.1 and abs(e[1][0] - rect.x0) < 0.1
            for e in edges)
        assert not has_left


# ── build_face_graph with content_rect ──────────────────────────────────────

class TestBuildFaceGraphBoundary:

    def test_open_U_closed_by_content_rect(self):
        """U-shape touching top edge at 2 points → content_rect injects
        top edge → polygonize closes into a large face."""
        from src.slab_v2.planarize import build_face_graph
        from src.slab_v2.models import VectorPath

        cfg = SlabV2Config(debug_images=False)
        rect = fitz.Rect(0, 0, 1000, 700)
        content_area = rect.width * rect.height

        # U-shape: two vertical legs touching top edge (y=0)
        paths = [
            VectorPath(id=0, style_id=0, segments=[
                ((200.0, 0.0), (200.0, 500.0)),
                ((200.0, 500.0), (800.0, 500.0)),
                ((800.0, 500.0), (800.0, 0.0)),
            ], is_closed=False, is_filled=False, seqno=0,
               fill_polygon=None, outside_content=False, has_stroke=True),
        ]

        # Without content_rect: open U → no closed faces
        fg_no_rect = build_face_graph(paths, {0}, cfg, content_area,
                                       content_rect=None)
        max_no = max((f.area_pt2 for f in fg_no_rect.faces), default=0)

        # With content_rect: top edge injected → U closes into rectangle
        fg_with_rect = build_face_graph(paths, {0}, cfg, content_area,
                                         content_rect=rect)
        max_with = max((f.area_pt2 for f in fg_with_rect.faces), default=0)

        # U interior = 600 * 500 = 300,000 pt²
        assert max_with > max_no, (
            f"content_rect should create larger face: "
            f"with={max_with:.0f} vs without={max_no:.0f}")
        assert max_with > 0.1 * content_area, (
            f"Expected large face (>10% of content area), got {max_with:.0f} "
            f"({max_with/content_area:.1%})")

    def test_closed_shape_unaffected(self):
        """Already-closed boundary → content_rect doesn't change max face."""
        from src.slab_v2.planarize import build_face_graph
        from src.slab_v2.models import VectorPath, StyleKey

        cfg = SlabV2Config(debug_images=False)
        rect = fitz.Rect(0, 0, 1000, 700)
        content_area = rect.width * rect.height

        # closed rectangle in center
        paths = [
            VectorPath(id=0, style_id=0, segments=[
                ((200, 200), (800, 200)),
                ((800, 200), (800, 500)),
                ((800, 500), (200, 500)),
                ((200, 500), (200, 200)),
            ], is_closed=True, is_filled=False, seqno=0,
               fill_polygon=None, outside_content=False, has_stroke=True),
        ]

        fg_no = build_face_graph(paths, {0}, cfg, content_area,
                                  content_rect=None)
        fg_yes = build_face_graph(paths, {0}, cfg, content_area,
                                   content_rect=rect)

        max_no = max((f.area_pt2 for f in fg_no.faces), default=0)
        max_yes = max((f.area_pt2 for f in fg_yes.faces), default=0)

        # The interior rectangle face should exist in both
        assert max_no > 0
        # With content_rect, the box face should still be close to original
        assert abs(max_yes - max_no) / max(max_no, 1) < 0.5 or max_yes >= max_no


# ── Config defaults ─────────────────────────────────────────────────────────

class TestContentBoundaryConfig:

    def test_defaults(self):
        cfg = SlabV2Config()
        assert cfg.use_content_boundary is True
        assert cfg.content_boundary_proximity_pt == 3.0
        assert cfg.content_boundary_min_touches == 2
