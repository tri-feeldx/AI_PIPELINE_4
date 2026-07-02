"""Dash bridging at corners/junctions (Phase 1.2).

Revit exports dashed boundaries as one path per dash.  _bridge_collinear
already closes gaps ALONG a straight run, but a gap at a corner (last dash
of one side vs first dash of the perpendicular side) is not collinear and
stayed open — measured on 2381 p17: thousands of free endpoints spaced
2.2/4.5pt.  _bridge_endpoints must connect mutually-nearest free endpoints
of the same style class within cfg.dash_bridge_tol_pt.
"""
from __future__ import annotations

import pytest

from src.slab_v2.config import SlabV2Config
from src.slab_v2.planarize import _bridge_endpoints, _polygonize


def _dashed_side(a, b, dash=6.0, gap=3.0):
    """Dashes along a->b, starting at a, never reaching past b."""
    import math
    ax, ay = a
    bx, by = b
    L = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / L, (by - ay) / L
    segs, t = [], 0.0
    while t + dash <= L:
        segs.append(((ax + ux * t, ay + uy * t),
                     (ax + ux * (t + dash), ay + uy * (t + dash))))
        t += dash + gap
    return segs


def _dashed_rect(w=100.0, h=80.0, dash=6.0, gap=3.0):
    return (_dashed_side((0, 0), (w, 0), dash, gap)
            + _dashed_side((w, 0), (w, h), dash, gap)
            + _dashed_side((w, h), (0, h), dash, gap)
            + _dashed_side((0, h), (0, 0), dash, gap))


def test_config_has_dash_bridge_tol():
    cfg = SlabV2Config()
    assert cfg.dash_bridge_tol_pt == 6.0


def test_corner_gaps_bridged():
    segs = _dashed_rect()
    bridges = _bridge_endpoints(segs, tol_pt=6.0)
    assert bridges, "free endpoints within tol must be bridged"
    polys, dangles, cuts = _polygonize(segs + bridges, snap_grid=0.05)
    assert polys, "bridged dashed rectangle must close at least one face"
    assert max(p.area for p in polys) == pytest.approx(100 * 80, rel=0.05)


def test_without_endpoint_bridge_rect_stays_open():
    """Guard: proves the corner gap is the thing being fixed."""
    segs = _dashed_rect()
    polys, dangles, cuts = _polygonize(segs, snap_grid=0.05)
    assert not polys or max(p.area for p in polys) < 0.5 * 100 * 80


def test_gap_beyond_tol_not_bridged():
    segs = _dashed_rect(dash=6.0, gap=3.0)
    # corner gap along the last side is ~3pt; with tol 1pt nothing bridges
    bridges = _bridge_endpoints(segs, tol_pt=1.0)
    assert bridges == []


def test_parallel_dash_pairs_do_not_ladder():
    """Revit slab edges are PAIRS of parallel dashed lines ~1.4pt apart.
    Endpoints across the pair are closer than the dash gap along the line;
    bridging must continue ALONG each line, never rung across the pair
    (measured on 2381 p17: ladder rungs shredded the boundary into 105
    micro-faces)."""
    upper = _dashed_side((0, 0), (60, 0))
    lower = _dashed_side((0, 1.4), (60, 1.4))
    bridges = _bridge_endpoints(upper + lower, tol_pt=6.0)
    for (a, b) in bridges:
        assert abs(a[1] - b[1]) < 0.1, f"cross-pair rung bridged: {a}->{b}"


class TestAdaptiveTolerance:
    """Robustness rule: thresholds derive from the page's own statistics.
    The bridge tolerance must come from the measured free-endpoint gap
    distribution of each style class, so solid-line sets never bridge and
    any dash rhythm (2.2pt, 4.5pt, ...) bridges itself — no per-file
    tuning."""

    def test_dashed_class_gets_tolerance_above_its_gap(self):
        from src.slab_v2.planarize import _adaptive_bridge_tol
        segs = _dashed_rect(dash=6.0, gap=3.0)
        tol = _adaptive_bridge_tol(segs, cap=8.0)
        assert 3.0 <= tol <= 8.0

    def test_solid_lines_get_zero_tolerance(self):
        from src.slab_v2.planarize import _adaptive_bridge_tol
        # four long solid lines, ends 50+ pt apart — nothing dash-like
        segs = [((0, 0), (100, 0)), ((0, 60), (100, 60)),
                ((200, 0), (300, 0)), ((200, 60), (300, 60))]
        assert _adaptive_bridge_tol(segs, cap=8.0) == 0.0

    def test_cap_is_respected(self):
        from src.slab_v2.planarize import _adaptive_bridge_tol
        segs = _dashed_rect(dash=6.0, gap=7.0)   # gaps larger than cap
        assert _adaptive_bridge_tol(segs, cap=5.0) <= 5.0

    def test_collect_segments_closes_dashed_rect_end_to_end(self):
        from unittest.mock import MagicMock
        from src.slab_v2.planarize import _collect_segments, _polygonize
        paths = []
        for i, seg in enumerate(_dashed_rect()):
            p = MagicMock()
            p.style_id = 7
            p.outside_content = False
            p.has_stroke = True
            p.fill_polygon = None
            p.segments = [seg]
            paths.append(p)
        segs = _collect_segments(paths, {7}, dash_bridge_tol_pt=8.0)
        polys, _, _ = _polygonize(segs, snap_grid=0.05)
        assert polys and max(p.area for p in polys) > 0.9 * 100 * 80


def test_bridge_is_mutual_nearest_only():
    # three collinear free points: A(0,0)-B(2,0) mutual; C(5,0) nearest to B
    # but B is taken -> C stays free (no chain bridging)
    segs = [((-10, 0), (0, 0)), ((2, 0), (3, 0)), ((5, 0), (15, 0))]
    bridges = _bridge_endpoints(segs, tol_pt=4.0)
    assert (((0, 0), (2, 0)) in bridges) or (((2, 0), (0, 0)) in bridges)
    # C..B gap is 2pt but B already used; only one bridge in that cluster
    assert len(bridges) <= 2
