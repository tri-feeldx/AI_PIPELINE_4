"""Topology-based slab assembly for no-fill GA plans (Phase 1.3).

The polygonized arrangement of a GA sheet is a planar subdivision: hundreds
of small faces separated by gridlines, bay markings and annotation boxes.
The slab is defined topologically — the set of faces that CANNOT be reached
from the sheet border without crossing an edge drawn by a thick SLAB_EDGE
style class.  Region-growing starts at the faces touching the viewport ring
and floods through every face boundary that is NOT covered by blocking
segments; whatever stays unreached is slab.

Every output vertex comes from the polygonized PDF vectors — nothing is
invented; openings are subtracted later by the existing opening resolvers.
"""
from __future__ import annotations

from collections import deque

import shapely
from shapely.geometry import MultiLineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

# a shared face border counts as blocked when at least this fraction of its
# length lies on blocking (thick-class) segments
_BLOCK_COVER_FRAC = 0.5
_TOUCH_TOL_PT = 1.0
_MIN_SHARED_LEN_PT = 0.5


def slab_faces_by_topology(
    faces: list,
    blocking_segs: list,
    viewport_bounds: tuple,
    min_slab_frac: float = 0.05,
) -> tuple:
    """Returns (slab geometry | None, audit dict)."""
    audit = {
        "schema": "nofill_topology_v1",
        "status": "unresolved",
        "n_faces": len(faces),
        "n_blocking_segments": len(blocking_segs),
        "n_outside": 0,
        "n_slab_faces": 0,
        "reason": "",
    }
    if not faces:
        audit["reason"] = "no faces"
        return None, audit
    if not blocking_segs:
        audit["reason"] = "no blocking (slab-edge) segments"
        return None, audit

    blocking = MultiLineString(
        [s for s in blocking_segs if s[0] != s[1]]).buffer(0.1)

    boundaries = [f.polygon.exterior for f in faces]
    tree = STRtree(boundaries)

    x0, y0, x1, y1 = viewport_bounds
    ring = box(x0, y0, x1, y1).exterior.buffer(_TOUCH_TOL_PT)

    # seeds: faces touching the viewport ring
    outside = set()
    for i, b in enumerate(boundaries):
        if b.intersects(ring):
            outside.add(i)
    audit["n_outside"] = len(outside)
    if not outside:
        audit["reason"] = "no face touches the viewport ring"
        return None, audit

    # region-grow: flood into neighbours across non-blocked shared borders
    visited = set(outside)
    queue = deque(outside)
    while queue:
        i = queue.popleft()
        for j in tree.query(boundaries[i]):
            j = int(j)
            if j in visited or j == i:
                continue
            shared = boundaries[i].intersection(boundaries[j])
            if shared.is_empty or shared.length < _MIN_SHARED_LEN_PT:
                continue
            covered = shared.intersection(blocking).length
            if covered / shared.length >= _BLOCK_COVER_FRAC:
                continue                     # blocked by slab edge
            visited.add(j)
            queue.append(j)

    slab_faces = [f for k, f in enumerate(faces) if k not in visited]
    audit["n_slab_faces"] = len(slab_faces)
    if not slab_faces:
        audit["reason"] = "outside flooded every face (boundary not closed)"
        return None, audit

    slab = shapely.make_valid(unary_union([f.polygon for f in slab_faces]))
    viewport_area = max((x1 - x0) * (y1 - y0), 1.0)
    if slab.is_empty or slab.area < min_slab_frac * viewport_area:
        audit["reason"] = (
            f"unreached area {slab.area / viewport_area:.1%} below "
            f"{min_slab_frac:.0%} of viewport")
        return None, audit

    audit["status"] = "verified"
    audit["slab_area_frac"] = round(slab.area / viewport_area, 4)
    return slab, audit


def resolve_no_fill_topology(paths, classes, cfg, viewport_rect, area_ref):
    """Pipeline wrapper: pick blocking classes by stroke width, polygonize
    the full non-frame network (dash-bridged) and run the topology resolver.

    Returns (slab geometry | None, audit dict).
    """
    from src.slab_v2 import planarize

    min_w = getattr(cfg, "nofill_blocking_min_width_pt", 1.0)
    nonframe = {c.id for c in classes if c.role != "FRAME"}
    blocking_ids = {
        c.id for c in classes
        if c.id in nonframe and (c.key.width or 0.0) >= min_w
        and c.key.stroke is not None
    }
    audit_extra = {
        "blocking_class_ids": sorted(blocking_ids),
        "blocking_min_width_pt": min_w,
    }
    if not blocking_ids:
        return None, {"schema": "nofill_topology_v1", "status": "unresolved",
                      "reason": "no stroke class reaches blocking width",
                      **audit_extra}

    fg = planarize.build_face_graph(
        paths, nonframe, cfg, area_ref, content_rect=viewport_rect)
    blocking_segs = planarize._collect_segments(
        paths, blocking_ids,
        dash_bridge_tol_pt=getattr(cfg, "dash_bridge_tol_pt", 0.0))

    slab, audit = slab_faces_by_topology(
        fg.faces, blocking_segs,
        (viewport_rect.x0, viewport_rect.y0,
         viewport_rect.x1, viewport_rect.y1),
        min_slab_frac=0.10)
    audit.update(audit_extra)
    return slab, audit
