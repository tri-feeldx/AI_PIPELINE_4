"""
Element extraction — stairs, lifts, shafts, voids via the X-CROSS symbol.

On structural drawings an opening (stair/lift shaft, penetration) is drawn
as a rectangle with corner-to-corner diagonals (an "X"). Detection finds
CROSSING DIAGONAL PAIRS — two diagonal segments that intersect near their
midpoints with endpoints near a common bounding rectangle's corners.

This segment-pair approach is independent of the face graph (which
subdivides X-cross rectangles into triangles, breaking face-based
detection). Text labels like "STAIR 01" assign the type; an unlabeled
X-cross is cut as VOID. Adjacent X-crosses merge. A label with no X-cross
nearby uses a nearest-face fallback (same pattern as column detection).
"""

from __future__ import annotations

import math
import re

import fitz
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import FaceGraph, ElementFootprint

_KEYWORD_TYPES = [
    (re.compile(r"\bSTAIRS?\b|\bST[- ]?\d{1,2}\b", re.I), "STAIR"),
    (re.compile(r"\bLIFTS?\b|\bELEV(ATOR)?\b|\bHOIST\b|\bLV ?\d{1,2}\b", re.I),
     "LIFT"),
    (re.compile(r"\bSHAFT\b", re.I), "SHAFT"),
    (re.compile(r"\bVOID\b|\bOPENING\b|\bPENETRATIONS?\b", re.I), "VOID"),
    (re.compile(r"\bDUCTS?\b|\bRISER\b", re.I), "DUCT"),
]

_DIAG_MIN_DEG = 15.0
_DIAG_MAX_DEG = 75.0
_CORNER_TOL_PT = 3.0


def _diagonal_segments(paths) -> list:
    """All segments at a diagonal angle (15-75° from the axes)."""
    out = []
    for p in paths:
        if p.outside_content:
            continue
        for (a, b) in p.segments:
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            if L < 2.0:
                continue
            ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            ang = min(ang, 180.0 - ang)
            if _DIAG_MIN_DEG <= ang <= _DIAG_MAX_DEG:
                out.append((a, b, L))
    return out


def _detect_xcross_rects(
    diags: list,
    scale: int | None,
    content_area_pt2: float,
) -> list:
    """Find X-cross rectangles from crossing diagonal segment pairs.

    Returns list of shapely Polygons (axis-aligned bounding rectangles).
    """
    if len(diags) < 2:
        return []

    _PT_TO_MM = 25.4 / 72.0
    _to_mm = _PT_TO_MM * (scale or 100)

    _MIN_DIAG_MM = 250 * math.sqrt(2)
    _MAX_DIAG_MM = 4000 * math.sqrt(2)
    _min_diag_pt = _MIN_DIAG_MM / _to_mm
    _max_diag_pt = _MAX_DIAG_MM / _to_mm

    geoms = []
    lens = []
    for a, b, L in diags:
        if L < _min_diag_pt or L > _max_diag_pt:
            continue
        geoms.append(LineString([a, b]))
        lens.append(L)

    if len(geoms) < 2:
        return []

    tree = STRtree(geoms)
    used: set[int] = set()
    rects: list = []

    for i in range(len(geoms)):
        if i in used:
            continue
        seg_a = geoms[i]
        len_a = lens[i]
        mid_a = seg_a.interpolate(0.5, normalized=True)
        ca = list(seg_a.coords)

        best_j = None
        best_score = float("inf")

        for j in tree.query(seg_a):
            j = int(j)
            if j <= i or j in used:
                continue
            len_b = lens[j]
            if min(len_a, len_b) / max(len_a, len_b) < 0.55:
                continue
            seg_b = geoms[j]
            if not seg_a.crosses(seg_b):
                continue

            inter = seg_a.intersection(seg_b)
            if inter.is_empty or inter.geom_type != "Point":
                continue
            mid_b = seg_b.interpolate(0.5, normalized=True)
            tol = 0.25 * max(len_a, len_b)
            if inter.distance(mid_a) > tol or inter.distance(mid_b) > tol:
                continue

            cb = list(seg_b.coords)
            all_pts = [ca[0], ca[-1], cb[0], cb[-1]]
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            minx, miny = min(xs), min(ys)
            maxx, maxy = max(xs), max(ys)
            w_pt, h_pt = maxx - minx, maxy - miny
            if w_pt < 2 or h_pt < 2:
                continue

            corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
            corner_tol = _CORNER_TOL_PT * 3
            ok = True
            for pt in all_pts:
                if not any(math.hypot(pt[0] - cx, pt[1] - cy) <= corner_tol
                           for cx, cy in corners):
                    ok = False
                    break
            if not ok:
                continue

            w_mm = w_pt * _to_mm
            h_mm = h_pt * _to_mm
            short, long = min(w_mm, h_mm), max(w_mm, h_mm)
            if short < 200 or long > 4000:
                continue
            if short > 0 and long / short > 5.0:
                continue

            score = inter.distance(mid_a) + inter.distance(mid_b)
            if score < best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            used.add(i)
            used.add(best_j)
            ca2 = list(geoms[i].coords)
            cb2 = list(geoms[best_j].coords)
            all_pts = [ca2[0], ca2[-1], cb2[0], cb2[-1]]
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            rects.append(box(min(xs), min(ys), max(xs), max(ys)))

    return rects


def extract_elements(
    page: fitz.Page,
    fg_all: FaceGraph,
    cfg: SlabV2Config,
    content_rect: fitz.Rect,
    content_area_pt2: float,
    paths: list | None = None,
    scale: int | None = None,
) -> tuple[list[ElementFootprint], list[str]]:
    """X-cross opening detection. Returns (elements, warnings)."""
    warnings: list[str] = []
    if paths is None:
        return [], ["element extraction skipped: no paths provided"]

    diags = _diagonal_segments(paths)

    _PT_TO_MM_e = 25.4 / 72.0
    _to_mm_e = _PT_TO_MM_e * (scale or 100)
    _MIN_SIDE_MM = 200.0
    _min_side_pt = _MIN_SIDE_MM / _to_mm_e

    max_area = cfg.xcross_max_area_frac * content_area_pt2
    xcross_rects = _detect_xcross_rects(diags, scale, content_area_pt2)

    footprints = []
    if xcross_rects:
        deduped: list = []
        for r in xcross_rects:
            is_dup = False
            for existing in deduped:
                inter = r.intersection(existing)
                if inter.area / max(r.area, 1e-9) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                bx = r.bounds
                w_pt = bx[2] - bx[0]
                h_pt = bx[3] - bx[1]
                if min(w_pt, h_pt) >= _min_side_pt and r.area <= max_area:
                    deduped.append(r)
        footprints = deduped

    anchors = []
    for w in page.get_text("words"):
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if not content_rect.contains(fitz.Point(cx, cy)):
            continue
        for rx, etype in _KEYWORD_TYPES:
            if rx.search(text):
                anchors.append((etype, text, (x0, y0, x1, y1), Point(cx, cy)))
                break

    elements: list[ElementFootprint] = []
    used_anchors = set()
    for fp in footprints:
        etype, label, bbox = "VOID", "", (0, 0, 0, 0)
        best_d = cfg.element_text_radius_pt
        best_i = None
        for i, (atype, atext, abbox, apt) in enumerate(anchors):
            d = fp.distance(apt)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            etype, label, bbox, _ = anchors[best_i]
            used_anchors.add(best_i)
        elements.append(ElementFootprint(
            type=etype, polygon=fp, label=label or etype,
            anchor_bbox=bbox, area_pt2=fp.area))

    # ── fallback: text anchor + nearest stair-sized face (same pattern as columns) ──
    _PT_TO_MM = 25.4 / 72.0
    _to_mm = _PT_TO_MM * (scale or 100)
    _STAIR_MIN_SIDE_MM = 1200
    _STAIR_MAX_SIDE_MM = 5000
    _STAIR_MAX_AREA_MM2 = 20_000_000
    _MAX_ASPECT = 4.0
    _SEARCH_RADIUS = cfg.element_text_radius_pt
    for i, (atype, atext, abbox, apt) in enumerate(anchors):
        if i in used_anchors:
            continue
        if atype not in ("STAIR", "LIFT", "SHAFT", "VOID"):
            continue
        if atype == "VOID":
            _min_side = getattr(cfg, "void_fallback_min_side_mm", 400.0)
            _radius = getattr(cfg, "text_evidence_search_radius_pt", 120.0)
        else:
            _min_side = _STAIR_MIN_SIDE_MM
            _radius = _SEARCH_RADIUS
        best_face = None
        best_dist = _radius + 1
        for f in fg_all.faces:
            dist = f.polygon.distance(apt)
            if dist > _radius:
                continue
            bx = f.polygon.bounds
            w_mm = (bx[2] - bx[0]) * _to_mm
            h_mm = (bx[3] - bx[1]) * _to_mm
            area_mm2 = f.area_pt2 * (_to_mm ** 2)
            short, long = min(w_mm, h_mm), max(w_mm, h_mm)
            if short < _min_side or long > _STAIR_MAX_SIDE_MM:
                continue
            if area_mm2 > _STAIR_MAX_AREA_MM2:
                continue
            if short > 0 and long / short > _MAX_ASPECT:
                continue
            if dist < best_dist:
                best_face = f
                best_dist = dist
        if best_face is None:
            continue
        already = any(
            best_face.polygon.intersects(e.polygon)
            and best_face.polygon.intersection(e.polygon).area
            / max(best_face.polygon.area, 1e-9) > 0.5
            for e in elements)
        if already:
            continue
        elements.append(ElementFootprint(
            type=atype, polygon=best_face.polygon,
            label=atext or atype,
            anchor_bbox=abbox, area_pt2=best_face.area_pt2))
        used_anchors.add(i)
        bx = best_face.polygon.bounds
        w_mm = (bx[2] - bx[0]) * _to_mm
        h_mm = (bx[3] - bx[1]) * _to_mm
        warnings.append(
            f"label '{atext}' ({atype}): no X-cross, fallback face "
            f"{w_mm:.0f}x{h_mm:.0f}mm")

    for i, (atype, atext, _bbox, apt) in enumerate(anchors):
        if i not in used_anchors and atype in ("STAIR", "LIFT", "SHAFT", "VOID"):
            warnings.append(
                f"label '{atext}' ({atype}) at ({apt.x:.0f},{apt.y:.0f})pt "
                f"has no X-cross opening or qualifying face — nothing cut")

    if xcross_rects:
        warnings.insert(0,
            f"X-cross segment-pair detection: {len(xcross_rects)} raw, "
            f"{len(footprints)} after merge, {len(elements)} with type")

    return elements, warnings
