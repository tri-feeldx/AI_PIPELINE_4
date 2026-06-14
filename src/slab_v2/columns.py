"""
Column detection — shape-first, the column schedule as a size filter.

v1 was label-first: a column without a text mark nearby was skipped, and a
schedule row without geometry got the LABEL bbox as its footprint (wrong
size by construction). Here it is inverted, same philosophy as the slab
kernel: geometry comes from the vector data (exact coordinates), the
Gemini-read schedule only says WHICH rectangle sizes are columns, and text
marks only ASSIGN symbols.

A candidate is a small closed near-rectangular path (filled or outlined)
whose sides match a scheduled width x depth (swap allowed) at the page
scale. Without a census, repeated identical sizes form anonymous types.
Candidates inside detected openings (stair/lift X-crosses) are dropped.
"""

from __future__ import annotations

import math
from collections import defaultdict

import fitz
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ColumnFootprint, ColumnType

PT_TO_MM = 25.4 / 72.0


def _path_polygon(p) -> Polygon | None:
    if p.fill_polygon is not None:
        poly = p.fill_polygon
        return poly if poly.is_valid and poly.area > 0 else None
    if not p.is_closed or not 3 <= len(p.segments) <= 8:
        return None
    pts = [seg[0] for seg in p.segments]
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
    except (ValueError, TypeError):
        return None
    return poly if poly.is_valid and poly.area > 0 else None


def _rect_sides_pt(poly: Polygon) -> tuple | None:
    """(long_pt, short_pt) when the polygon is essentially a rectangle."""
    mrr = poly.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon" or mrr.area <= 0:
        return None
    if poly.area / mrr.area < 0.80:
        return None
    c = list(mrr.exterior.coords)
    s1 = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
    s2 = math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
    return (max(s1, s2), min(s1, s2))


def _size_match(w_mm: float, d_mm: float, t: ColumnType,
                tol: float) -> bool:
    if not t.width_mm or not t.depth_mm:
        return False
    a, b = max(t.width_mm, t.depth_mm), min(t.width_mm, t.depth_mm)
    return abs(w_mm - a) <= tol and abs(d_mm - b) <= tol


def extract_columns(
    page: fitz.Page,
    paths: list,
    slab_union,
    scale: float,
    column_types: dict[str, ColumnType],
    cfg: SlabV2Config,
    elements: list | None = None,
) -> tuple[list[ColumnFootprint], list[str]]:
    """Returns (columns, warnings). scale = final (precise) page scale."""
    warnings: list[str] = []
    if not scale:
        return [], ["column detection skipped: no scale"]
    to_mm = PT_TO_MM * scale

    openings = unary_union([e.polygon for e in elements]) \
        if elements else None

    # ── candidates: small near-rectangles with measured mm sides ──────────
    cands = []           # (poly, w_mm, d_mm)
    for p in paths:
        if p.outside_content:
            continue
        poly = _path_polygon(p)
        if poly is None:
            continue
        sides = _rect_sides_pt(poly)
        if sides is None:
            continue
        w_mm, d_mm = sides[0] * to_mm, sides[1] * to_mm
        if not (100.0 <= d_mm and w_mm <= cfg.column_max_side_mm):
            continue
        if openings is not None and poly.intersects(openings):
            continue
        if slab_union is not None:
            dist_mm = poly.distance(slab_union) * to_mm
            if not poly.intersects(slab_union) and dist_mm > w_mm:
                continue
        cands.append((poly, w_mm, d_mm))

    if not cands:
        return [], warnings

    # ── census matching (or anonymous repeated-size types) ────────────────
    matched = []         # (poly, w_mm, d_mm, [symbols])
    if column_types:
        for poly, w, d in cands:
            syms = [t.symbol for t in column_types.values()
                    if _size_match(w, d, t, cfg.column_size_tol_mm)]
            if syms:
                matched.append((poly, w, d, syms))
    else:
        groups = defaultdict(list)
        for poly, w, d in cands:
            key = (int(round(w / 25.0) * 25), int(round(d / 25.0) * 25))
            groups[key].append((poly, w, d))
        for (kw, kd), members in sorted(groups.items()):
            if len(members) >= cfg.column_min_repeat:
                sym = f"COL{kw}x{kd}"
                matched.extend((poly, w, d, [sym])
                               for poly, w, d in members)
        if matched:
            warnings.append(
                "no column schedule in the document census — using "
                f"{len({m[3][0] for m in matched})} repeated-size "
                f"anonymous column type(s)")

    if not matched:
        return [], warnings

    # ── dedupe overlapping candidates (nested outlines, fills over strokes)
    matched.sort(key=lambda m: m[0].area)
    geoms = [m[0] for m in matched]
    tree = STRtree(geoms)
    kept: set[int] = set()
    kept_idx: list[int] = []
    for i, (poly, _w, _d, _s) in enumerate(matched):
        dup = False
        for j in tree.query(poly):
            j = int(j)
            if j == i or j not in kept:
                continue
            inter = poly.intersection(geoms[j]).area
            union = poly.area + geoms[j].area - inter
            if union > 0 and inter / union > 0.5:
                dup = True
                break
        if not dup:
            kept.add(i)
            kept_idx.append(i)

    # ── text marks assign symbols ──────────────────────────────────────────
    symbols_upper = {s.upper(): s for s in column_types}
    anchors = []
    if symbols_upper:
        for w in page.get_text("words"):
            txt = w[4].strip().upper().rstrip(".,:")
            if txt in symbols_upper:
                cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                anchors.append((symbols_upper[txt], (cx, cy)))

    columns: list[ColumnFootprint] = []
    ambiguous = 0
    for i in kept_idx:
        poly, w, d, syms = matched[i]
        symbol, labeled = (syms[0] if len(syms) == 1 else "C?"), False
        best = cfg.column_label_radius_pt
        for sym, (cx, cy) in anchors:
            if sym not in syms:
                continue
            dist = poly.distance(Point(cx, cy))
            if dist < best:
                best, symbol, labeled = dist, sym, True
        if symbol == "C?":
            ambiguous += 1
        columns.append(ColumnFootprint(
            symbol=symbol, polygon=poly, w_mm=round(w, 0),
            d_mm=round(d, 0), labeled=labeled))

    if ambiguous:
        warnings.append(
            f"{ambiguous} column(s) match several schedule sizes and have "
            f"no nearby mark — exported as 'C?'")
    return columns, warnings
