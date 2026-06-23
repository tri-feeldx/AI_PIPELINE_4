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
import re
from collections import defaultdict

import fitz
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ColumnFootprint, ColumnType

PT_TO_MM = 25.4 / 72.0

def _normalize_label(text: str) -> str:
    """Normalize a PDF word/column symbol for robust label matching.

    Strips everything except A-Z and 0-9 so that 'CH*35c' and 'CH*'
    become 'CH35C' and 'CH' — asterisks, dashes, spaces all removed.
    """
    return re.sub(r"[^A-Z0-9]+", "", str(text or "").upper())


def _steel_label_matches(word: str, steel_symbols: set[str]) -> bool:
    """True when *word* (page text) looks like a variant of a census steel symbol.

    Only uses symbols the Gemini census classified as STEEL — no hardcoded
    prefix list, so SH/CH/etc. are only treated as steel when the census
    says so for THIS PDF.
    """
    norm = _normalize_label(word)
    if not norm:
        return False
    if norm in steel_symbols:
        return True
    for sym in steel_symbols:
        if not sym or len(sym) < 2:
            continue
        if norm.startswith(sym):
            return True
    return False


def _collect_steel_exclusion_zones(
    page: fitz.Page,
    steel_symbols: set[str],
    radius_pt: float,
    slab_union=None,
) -> list:
    """Return buffered text-label zones that should not become RC columns."""
    if not steel_symbols:
        return []
    slab_buffer = slab_union.buffer(radius_pt * 3) if slab_union is not None else None
    zones = []
    for w in page.get_text("words"):
        if len(w) < 5 or not _steel_label_matches(w[4], steel_symbols):
            continue
        x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if slab_buffer is not None and not slab_buffer.contains(Point(cx, cy)):
            continue
        zones.append(box(x0, y0, x1, y1).buffer(radius_pt))
    return zones


def _in_steel_exclusion(poly: Polygon, zones: list) -> bool:
    """True when a candidate rectangle is too close to a steel text label."""
    return any(poly.intersects(zone) or poly.centroid.within(zone)
               for zone in zones)


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
    steel_symbols = {
        _normalize_label(sym) for sym, ct in (column_types or {}).items()
        if str(getattr(ct, "material", "") or "").upper() == "STEEL"
    }
    steel_symbols.discard("")
    steel_skipped = sorted(
        sym for sym, ct in (column_types or {}).items()
        if str(getattr(ct, "material", "") or "").upper() == "STEEL"
    )
    if steel_skipped:
        warnings.append(
            "steel column types skipped in RC-only phase: "
            + ", ".join(steel_skipped)
        )
    column_types = {
        sym: ct for sym, ct in (column_types or {}).items()
        if str(getattr(ct, "material", "") or "").upper() != "STEEL"
    }
    to_mm = PT_TO_MM * scale

    openings = unary_union([e.polygon for e in elements]) \
        if elements else None
    steel_exclusion_radius = max(cfg.steel_exclusion_radius_pt, 40.0)
    steel_exclusion_zones = _collect_steel_exclusion_zones(
        page, steel_symbols, steel_exclusion_radius, slab_union)
    if steel_exclusion_zones:
        warnings.append(
            f"steel exclusion zones active: {len(steel_exclusion_zones)} label(s)")

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
        if _in_steel_exclusion(poly, steel_exclusion_zones):
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
    symbols_norm = {_normalize_label(s): s for s in column_types}
    anchors = []
    if symbols_norm:
        all_words = page.get_text("words")
        for w in all_words:
            txt = _normalize_label(w[4])
            if txt in symbols_norm:
                cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                anchors.append((symbols_norm[txt], (cx, cy)))

        # merge split labels: "C" + "9" → "C9"
        _SPLIT_MERGE_DIST = 20.0
        letter_words = []
        digit_words = []
        for w in all_words:
            norm = _normalize_label(w[4])
            if not norm:
                continue
            if norm.isalpha() and len(norm) <= 3:
                letter_words.append((norm, (w[0]+w[2])/2, (w[1]+w[3])/2))
            elif norm.isdigit() and len(norm) <= 2:
                digit_words.append((norm, (w[0]+w[2])/2, (w[1]+w[3])/2))
        for ltxt, lx, ly in letter_words:
            for dtxt, dx, dy in digit_words:
                if ((lx - dx)**2 + (ly - dy)**2)**0.5 > _SPLIT_MERGE_DIST:
                    continue
                combined = ltxt + dtxt
                if combined in symbols_norm:
                    mx, my = (lx + dx) / 2, (ly + dy) / 2
                    anchors.append((symbols_norm[combined], (mx, my)))

        # dedupe: same symbol within 30pt → keep first
        _seen: dict[str, tuple[float, float]] = {}
        _deduped: list[tuple[str, tuple[float, float]]] = []
        for sym, pos in anchors:
            if sym in _seen:
                ox, oy = _seen[sym]
                if ((pos[0] - ox)**2 + (pos[1] - oy)**2)**0.5 < 30:
                    continue
            _seen[sym] = pos
            _deduped.append((sym, pos))
        anchors = _deduped

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
