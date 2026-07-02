from __future__ import annotations

from types import SimpleNamespace

import fitz

from src.slab_v2.models import VectorPath
from src.slab_v2.plan_viewport import assemble_irregular_no_fill_slab_boundary


def _path(points, *, dashed=False, pid=0):
    segments = [
        (points[i], points[i + 1])
        for i in range(len(points) - 1)
        if points[i] != points[i + 1]
    ]
    p = VectorPath(
        id=pid,
        style_id=1,
        segments=segments,
        is_closed=points[0] == points[-1],
        is_filled=False,
        seqno=0,
        fill_polygon=None,
        outside_content=False,
        has_stroke=True,
    )
    if dashed:
        p.key = SimpleNamespace(dashes="[3 2] 0", stroke=(0, 0, 0))
    return p


def _viewport():
    return fitz.Rect(0, 0, 1000, 800)


def test_irregular_l_shape_is_accepted():
    viewport = _viewport()
    pts = [
        (100, 100), (900, 100), (900, 380), (650, 380),
        (650, 700), (100, 700), (100, 100),
    ]
    poly, audit = assemble_irregular_no_fill_slab_boundary(
        [_path(pts)], viewport, viewport.width * viewport.height)

    assert poly is not None
    assert audit["status"] == "verified"
    assert audit["method"] == "irregular_structural_outline_assembly"
    assert audit["selected_candidate"]["area_fraction_of_viewport"] > 0.40


def test_fragmented_perimeter_small_gaps_are_audited_and_closed():
    viewport = _viewport()
    paths = [
        _path([(50, 50), (480, 50)], pid=1),
        _path([(490, 50), (950, 50)], pid=2),
        _path([(950, 50), (950, 750)], pid=3),
        _path([(950, 750), (50, 750)], pid=4),
        _path([(50, 750), (50, 50)], pid=5),
    ]
    poly, audit = assemble_irregular_no_fill_slab_boundary(
        paths, viewport, viewport.width * viewport.height)

    assert poly is not None
    assert audit["status"] == "verified"
    assert audit["gap_closure_count"] >= 1
    assert any(g["source"] == "gap_closure" for g in audit["gap_closures"])


def test_fragmented_ga_perimeter_uses_relaxed_supported_envelope():
    viewport = _viewport()
    # Mimics no-fill GA pages where the perimeter is supported by real edge
    # fragments, but top/bottom runs are interrupted enough that deriving the
    # side extents from them is too strict.
    paths = [
        _path([(180, 160), (420, 160)], pid=1),
        _path([(560, 160), (880, 160)], pid=2),
        _path([(140, 650), (900, 650)], pid=3),
        _path([(120, 170), (120, 640)], pid=4),
        _path([(910, 260), (910, 630)], pid=5),
        # Internal structural fragments should not be chosen as the slab.
        _path([(300, 300), (700, 300)], pid=6),
        _path([(300, 520), (700, 520)], pid=7),
    ]
    poly, audit = assemble_irregular_no_fill_slab_boundary(
        paths, viewport, viewport.width * viewport.height)

    assert poly is not None
    assert audit["status"] == "verified"
    assert audit["selected_candidate"]["id"] == "supported_envelope"
    assert audit["relaxed_supported_envelope_candidate"]["status"] == "verified"
    assert audit["selected_candidate"]["area_fraction_of_viewport"] > 0.40


def test_large_gap_stays_fail_closed():
    viewport = _viewport()
    paths = [
        _path([(50, 50), (420, 50)], pid=1),
        _path([(560, 50), (950, 50)], pid=2),
        _path([(950, 50), (950, 750)], pid=3),
        _path([(950, 750), (50, 750)], pid=4),
        _path([(50, 750), (50, 50)], pid=5),
    ]
    poly, audit = assemble_irregular_no_fill_slab_boundary(
        paths, viewport, viewport.width * viewport.height)

    assert poly is None
    assert audit["status"] == "unresolved"
    assert audit["reason"] in {
        "polygonize_produced_no_slab_like_candidates",
        "best_candidate_below_confidence_gate",
        "ambiguous_multiple_outline_candidates",
    }


def test_small_annotation_rectangle_is_not_selected_as_slab():
    viewport = _viewport()
    pts = [(100, 100), (220, 100), (220, 190), (100, 190), (100, 100)]
    poly, audit = assemble_irregular_no_fill_slab_boundary(
        [_path(pts)], viewport, viewport.width * viewport.height)

    assert poly is None
    assert audit["status"] == "unresolved"
    assert audit["reason"] in {
        "polygonize_produced_no_slab_like_candidates",
        "best_candidate_below_confidence_gate",
    }


def test_dashed_reference_edges_are_blocked():
    viewport = _viewport()
    pts = [(50, 50), (950, 50), (950, 750), (50, 750), (50, 50)]
    poly, audit = assemble_irregular_no_fill_slab_boundary(
        [_path(pts, dashed=True)], viewport, viewport.width * viewport.height)

    assert poly is None
    assert audit["rejected_edge_counts"].get("dashed_or_reference") == 1
    assert audit["reason"] == "not_enough_solid_edges_in_viewport"
