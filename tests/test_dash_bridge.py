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


def test_bridge_is_mutual_nearest_only():
    # three collinear free points: A(0,0)-B(2,0) mutual; C(5,0) nearest to B
    # but B is taken -> C stays free (no chain bridging)
    segs = [((-10, 0), (0, 0)), ((2, 0), (3, 0)), ((5, 0), (15, 0))]
    bridges = _bridge_endpoints(segs, tol_pt=4.0)
    assert (((0, 0), (2, 0)) in bridges) or (((2, 0), (0, 0)) in bridges)
    # C..B gap is 2pt but B already used; only one bridge in that cluster
    assert len(bridges) <= 2
