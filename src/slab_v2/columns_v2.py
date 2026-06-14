"""
Column detection v2 — text-anchor-then-shape.

Pass 1: find text labels matching column symbols on the page, search for
rectangular shapes near each label, assign symbol precisely.
Pass 2: shape-first fallback (columns.py logic) for unlabeled columns,
skipping polygons already claimed by Pass 1.

Solves the C1/C2/C3-all-600x600 problem: when multiple schedule types share
dimensions, the text label disambiguates which symbol belongs where.
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
from src.slab_v2.columns import (
    _path_polygon, _rect_sides_pt, _size_match, PT_TO_MM,
)


def extract_columns_v2(
    page: fitz.Page,
    paths: list,
    slab_union,
    scale: float,
    column_types: dict[str, ColumnType],
    cfg: SlabV2Config,
    elements: list | None = None,
    columns_per_floor_census: dict[str, int] | None = None,
) -> tuple[list[ColumnFootprint], list[str]]:
    """Text-anchor-then-shape column detection with shape-first fallback."""
    warnings: list[str] = []
    if not scale:
        return [], ["column detection skipped: no scale"]
    to_mm = PT_TO_MM * scale

    openings = unary_union([e.polygon for e in elements]) \
        if elements else None

    # ── build all rectangular candidates from vector paths ───────────────
    all_cands = []  # (poly, w_mm, d_mm)
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
        all_cands.append((poly, w_mm, d_mm))

    if not all_cands:
        return [], warnings

    # build STRtree for spatial queries
    cand_geoms = [c[0] for c in all_cands]
    cand_tree = STRtree(cand_geoms)

    # ── PASS 1: text-anchor-then-shape ───────────────────────────────────
    symbols_upper = {s.upper(): s for s in column_types}
    text_anchors = []  # (symbol, cx, cy)
    if symbols_upper:
        slab_buffer = slab_union.buffer(50) if slab_union is not None else None
        for w in page.get_text("words"):
            txt = w[4].strip().upper().rstrip(".,:")
            if txt in symbols_upper:
                cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
                # only anchors within/near the slab area (skip legend/notes)
                if slab_buffer is not None and \
                        not slab_buffer.contains(Point(cx, cy)):
                    continue
                text_anchors.append((symbols_upper[txt], cx, cy))

    claimed: set[int] = set()  # indices into all_cands
    pass1_columns: list[ColumnFootprint] = []
    search_r = cfg.column_text_search_radius_pt

    for sym, cx, cy in text_anchors:
        ct = column_types.get(sym)
        if ct is None:
            continue
        # search for rectangles near this text
        search_box = Point(cx, cy).buffer(search_r)
        hits = cand_tree.query(search_box)
        best_idx, best_dist = None, search_r + 1
        for idx in hits:
            idx = int(idx)
            if idx in claimed:
                continue
            poly, w, d = all_cands[idx]
            # size must match THIS SPECIFIC symbol's dimensions
            if not _size_match(w, d, ct, cfg.column_size_tol_mm):
                continue
            dist = poly.distance(Point(cx, cy))
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        if best_idx is not None:
            poly, w, d = all_cands[best_idx]
            claimed.add(best_idx)
            pass1_columns.append(ColumnFootprint(
                symbol=sym, polygon=poly,
                w_mm=round(w, 0), d_mm=round(d, 0), labeled=True))

    # dedupe pass1 (same polygon claimed by multiple nearby text anchors)
    if len(pass1_columns) > 1:
        p1_geoms = [c.polygon for c in pass1_columns]
        p1_tree = STRtree(p1_geoms)
        p1_keep = set()
        for i, col in enumerate(pass1_columns):
            dup = False
            for j in p1_tree.query(col.polygon):
                j = int(j)
                if j == i or j not in p1_keep:
                    continue
                inter = col.polygon.intersection(p1_geoms[j]).area
                union = col.polygon.area + p1_geoms[j].area - inter
                if union > 0 and inter / union > 0.5:
                    dup = True
                    break
            if not dup:
                p1_keep.add(i)
        pass1_columns = [pass1_columns[i] for i in sorted(p1_keep)]

    # ── PASS 2: shape-first fallback for unclaimed candidates ────────────
    unclaimed = [(i, poly, w, d) for i, (poly, w, d) in enumerate(all_cands)
                 if i not in claimed]

    matched = []  # (poly, w_mm, d_mm, [symbols])
    if column_types:
        for _i, poly, w, d in unclaimed:
            syms = [t.symbol for t in column_types.values()
                    if _size_match(w, d, t, cfg.column_size_tol_mm)]
            if syms:
                matched.append((poly, w, d, syms))
    else:
        # no census: anonymous repeated-size types
        groups = defaultdict(list)
        for _i, poly, w, d in unclaimed:
            key = (int(round(w / 25.0) * 25), int(round(d / 25.0) * 25))
            groups[key].append((poly, w, d))
        for (kw, kd), members in sorted(groups.items()):
            if len(members) >= cfg.column_min_repeat:
                sym = f"COL{kw}x{kd}"
                matched.extend((poly, w, d, [sym])
                               for poly, w, d in members)
        if matched:
            warnings.append(
                "no column schedule — using "
                f"{len({m[3][0] for m in matched})} anonymous type(s)")

    # dedupe pass2
    if matched:
        matched.sort(key=lambda m: m[0].area)
        m_geoms = [m[0] for m in matched]
        m_tree = STRtree(m_geoms)
        kept_idx = []
        kept_set: set[int] = set()
        for i, (poly, _w, _d, _s) in enumerate(matched):
            dup = False
            for j in m_tree.query(poly):
                j = int(j)
                if j == i or j not in kept_set:
                    continue
                inter = poly.intersection(m_geoms[j]).area
                union = poly.area + m_geoms[j].area - inter
                if union > 0 and inter / union > 0.5:
                    dup = True
                    break
            if not dup:
                # also check against pass1 claimed polygons
                if pass1_columns:
                    for p1col in pass1_columns:
                        inter = poly.intersection(p1col.polygon).area
                        union = poly.area + p1col.polygon.area - inter
                        if union > 0 and inter / union > 0.5:
                            dup = True
                            break
                if not dup:
                    kept_set.add(i)
                    kept_idx.append(i)

        ambiguous = 0
        for i in kept_idx:
            poly, w, d, syms = matched[i]
            symbol = syms[0] if len(syms) == 1 else "C?"
            if symbol == "C?":
                ambiguous += 1
            pass1_columns.append(ColumnFootprint(
                symbol=symbol, polygon=poly,
                w_mm=round(w, 0), d_mm=round(d, 0), labeled=False))
        if ambiguous:
            warnings.append(
                f"{ambiguous} unlabeled column(s) match multiple schedule "
                f"sizes — exported as 'C?'")

    columns = pass1_columns

    # ── census cross-check ───────────────────────────────────────────────
    if columns_per_floor_census:
        detected: dict[str, int] = defaultdict(int)
        for c in columns:
            detected[c.symbol] += 1
        for sym, expected in columns_per_floor_census.items():
            got = detected.get(sym, 0)
            if got != expected:
                warnings.append(
                    f"census expects {expected}× {sym}, detected {got}")
        for sym, got in detected.items():
            if sym not in columns_per_floor_census and sym != "C?":
                warnings.append(
                    f"detected {got}× {sym} not in census for this floor")

    labeled_count = sum(1 for c in columns if c.labeled)
    if text_anchors and labeled_count:
        warnings.insert(0,
            f"text-anchor pass: {labeled_count} column(s) identified by "
            f"label, {len(columns) - labeled_count} by shape fallback")

    return columns, warnings
