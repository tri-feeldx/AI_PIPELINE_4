"""Topology-based no-fill slab assembly (Phase 1.3).

On no-fill GA pages the slab is fragmented into hundreds of small faces by
gridlines, bay markings and text boxes (p17: 1920 faces, largest 1%).  The
slab is the union of faces NOT reachable from the sheet border when
region-growing is blocked at edges drawn by the thick SLAB_EDGE classes.
All face coordinates come from polygonize on real PDF vectors.
"""
from __future__ import annotations

import pytest
from shapely.geometry import box

from src.slab_v2.models import Face
from src.slab_v2.planarize import _polygonize
from src.slab_v2.nofill_topology import slab_faces_by_topology


def _rect_segs(x0, y0, x1, y1):
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def _make_faces(segments):
    polys, dangles, cuts = _polygonize(segments, snap_grid=0.05)
    return [Face(id=i, polygon=p, area_pt2=p.area,
                 label_anchor=tuple(p.representative_point().coords[0]))
            for i, p in enumerate(polys)]


@pytest.fixture
def fragmented_sheet():
    """Viewport 120x120; thick slab boundary 10,10..110,110; thin gridlines
    at x/y = 40, 70 crossing the whole viewport (fragmenting inside AND
    outside the slab)."""
    viewport = box(0, 0, 120, 120)
    blocking = _rect_segs(10, 10, 110, 110)
    thin = []
    for t in (40.0, 70.0):
        thin.append(((t, 0.0), (t, 120.0)))
        thin.append(((0.0, t), (120.0, t)))
    frame = _rect_segs(0, 0, 120, 120)
    all_segs = frame + blocking + thin
    return viewport, blocking, all_segs


def test_slab_is_union_of_interior_faces(fragmented_sheet):
    viewport, blocking, all_segs = fragmented_sheet
    faces = _make_faces(all_segs)
    assert len(faces) > 9  # fragmented inside and outside

    slab, audit = slab_faces_by_topology(faces, blocking, viewport.bounds)
    assert slab is not None, audit
    assert slab.area == pytest.approx(100 * 100, rel=0.01)
    # nothing outside the blocking rect may leak in
    assert slab.difference(box(10, 10, 110, 110)).area < 1.0


def test_no_blocking_edges_means_no_slab(fragmented_sheet):
    viewport, _, all_segs = fragmented_sheet
    faces = _make_faces(all_segs)
    slab, audit = slab_faces_by_topology(faces, [], viewport.bounds)
    assert slab is None
    assert audit["status"] == "unresolved"


def test_open_boundary_leaks_and_fails_closed():
    """If the thick boundary has a big hole (> bridge), outside floods in
    and the resolver must NOT return a slab."""
    viewport = box(0, 0, 120, 120)
    blocking = [((10, 10), (110, 10)), ((110, 10), (110, 110)),
                ((110, 110), (10, 110))]  # left side fully missing
    frame = _rect_segs(0, 0, 120, 120)
    faces = _make_faces(frame + blocking)
    slab, audit = slab_faces_by_topology(faces, blocking, viewport.bounds)
    assert slab is None or slab.area < 0.2 * 100 * 100


def test_audit_reports_counts(fragmented_sheet):
    viewport, blocking, all_segs = fragmented_sheet
    faces = _make_faces(all_segs)
    slab, audit = slab_faces_by_topology(faces, blocking, viewport.bounds)
    assert audit["schema"] == "nofill_topology_v1"
    assert audit["n_faces"] == len(faces)
    assert audit["n_outside"] > 0
    assert audit["n_slab_faces"] >= 9
