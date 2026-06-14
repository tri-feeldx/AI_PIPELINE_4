"""
Element extraction — stairs, lifts, shafts, voids via the X-CROSS symbol.

On structural drawings an opening (stair/lift shaft, penetration) is drawn
as a small rectangle with corner-to-corner diagonals (an "X"). That symbol
— not the text label — defines the opening footprint: labels like
"STAIR 01" usually sit OUTSIDE the shaft with a leader line, so anchoring
on text alone grabs whole rooms (the v2 bug on Structural.pdf p10/p11).

Detection: a face is an opening candidate when at least two DIAGONAL
segments span it near corner-to-corner. Text within the face or within
cfg.element_text_radius_pt only assigns the type/label; an unlabeled
X-face is still cut as VOID. Adjacent X-faces merge into one element
(double lifts, multi-cell cores). A label with no X-face nearby produces
a warning and cuts nothing — never cut a large face.
"""

from __future__ import annotations

import math
import re

import fitz
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import FaceGraph, ElementFootprint

# keyword -> element type (checked in order, word-boundary)
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


def _is_xcross_face(face, diag_tree: STRtree, diag_geoms: list,
                    diag_lens: list) -> bool:
    """Face has >=2 diagonal segments spanning it near corner-to-corner."""
    minx, miny, maxx, maxy = face.polygon.bounds
    face_diag = math.hypot(maxx - minx, maxy - miny)
    if face_diag < 4.0:
        return False
    corners = [Point(minx, miny), Point(maxx, miny),
               Point(maxx, maxy), Point(minx, maxy)]
    span_count = 0
    poly = face.polygon.buffer(_CORNER_TOL_PT)
    for i in diag_tree.query(face.polygon):
        seg = diag_geoms[i]
        if diag_lens[i] < 0.5 * face_diag:
            continue
        if not seg.intersects(poly):
            continue
        mid = seg.interpolate(0.5, normalized=True)
        if not face.polygon.contains(mid):
            continue
        (ax, ay), (bx, by) = seg.coords[0], seg.coords[-1]
        a_near = any(c.distance(Point(ax, ay)) <= _CORNER_TOL_PT * 2
                     for c in corners)
        b_near = any(c.distance(Point(bx, by)) <= _CORNER_TOL_PT * 2
                     for c in corners)
        if a_near and b_near:
            span_count += 1
            if span_count >= 2:
                return True
    return False


def extract_elements(
    page: fitz.Page,
    fg_all: FaceGraph,
    cfg: SlabV2Config,
    content_rect: fitz.Rect,
    content_area_pt2: float,
    paths: list | None = None,
) -> tuple[list[ElementFootprint], list[str]]:
    """X-cross opening detection. Returns (elements, warnings)."""
    warnings: list[str] = []
    if paths is None:
        return [], ["element extraction skipped: no paths provided"]

    diags = _diagonal_segments(paths)
    if not diags:
        return [], []
    diag_geoms = [LineString([a, b]) for a, b, _l in diags]
    diag_lens = [l for _a, _b, l in diags]
    diag_tree = STRtree(diag_geoms)

    max_area = cfg.xcross_max_area_frac * content_area_pt2
    xfaces = [f for f in fg_all.faces
              if f.area_pt2 <= max_area
              and _is_xcross_face(f, diag_tree, diag_geoms, diag_lens)]
    if not xfaces:
        return [], warnings

    # merge adjacent X-faces into single footprints
    merged = unary_union([f.polygon.buffer(2.0) for f in xfaces])
    footprints = []
    for g in getattr(merged, "geoms", [merged]):
        shrunk = g.buffer(-2.0)
        for part in getattr(shrunk, "geoms", [shrunk]):
            if not part.is_empty and part.area > 0:
                footprints.append(part)

    # collect text anchors for typing/labeling
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

    for i, (atype, atext, _bbox, apt) in enumerate(anchors):
        if i not in used_anchors and atype in ("STAIR", "LIFT", "SHAFT"):
            warnings.append(
                f"label '{atext}' ({atype}) at ({apt.x:.0f},{apt.y:.0f})pt "
                f"has no X-cross opening within "
                f"{cfg.element_text_radius_pt:.0f}pt — nothing cut")

    return elements, warnings
