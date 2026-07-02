"""
Stage A — vector extraction.

Pulls every drawing from fitz.Page.get_drawings(), flattens curves to straight
segments (adaptive De Casteljau, bounded deviation), and groups paths into
style classes by (stroke color, fill color, width, dash pattern).

Coordinates are kept verbatim from the PDF. Bezier flattening introduces at
most cfg.bezier_tol_pt (0.2 pt = 0.07 mm paper) deviation.
"""

from __future__ import annotations

import math
from collections import defaultdict

import fitz
from shapely.geometry import Polygon

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import StyleKey, StyleClass, VectorPath


# ── helpers ────────────────────────────────────────────────────────────────────

def _norm_color(c) -> tuple | None:
    if c is None:
        return None
    try:
        return tuple(round(float(v), 3) for v in c)
    except TypeError:
        return None


def _norm_dashes(d) -> str:
    """fitz reports dashes like '[] 0' (solid) or '[3 2] 0'."""
    if not d:
        return ""
    s = str(d).strip()
    if s in ("[] 0", "[]", "0", ""):
        return ""
    return s


def _flatten_bezier(p0, p1, p2, p3, tol: float, depth: int = 0) -> list:
    """Adaptive De Casteljau subdivision of a cubic bezier.

    Returns interior+end points (excluding p0) such that the polyline deviates
    from the true curve by less than tol.
    """
    # flatness test: max distance of control points from chord p0-p3
    dx, dy = p3[0] - p0[0], p3[1] - p0[1]
    chord = math.hypot(dx, dy)
    if chord < 1e-9:
        d1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        d2 = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        flat = max(d1, d2)
    else:
        d1 = abs((p1[0] - p0[0]) * dy - (p1[1] - p0[1]) * dx) / chord
        d2 = abs((p2[0] - p0[0]) * dy - (p2[1] - p0[1]) * dx) / chord
        flat = max(d1, d2)

    if flat <= tol or depth >= 16:
        return [p3]

    # subdivide at t=0.5
    m01 = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    m12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    m23 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
    m012 = ((m01[0] + m12[0]) / 2, (m01[1] + m12[1]) / 2)
    m123 = ((m12[0] + m23[0]) / 2, (m12[1] + m23[1]) / 2)
    mid = ((m012[0] + m123[0]) / 2, (m012[1] + m123[1]) / 2)

    left = _flatten_bezier(p0, m01, m012, mid, tol, depth + 1)
    right = _flatten_bezier(mid, m123, m23, p3, tol, depth + 1)
    return left + right


def _pt(p) -> tuple:
    return (float(p.x), float(p.y))


def _items_to_points(items, tol: float) -> list:
    """Convert one fitz drawing's items into a list of polylines (point lists)."""
    polylines = []
    current = []

    def flush():
        nonlocal current
        if len(current) >= 2:
            polylines.append(current)
        current = []

    for item in items:
        op = item[0]
        if op == "l":                       # line: (op, p1, p2)
            p1, p2 = _pt(item[1]), _pt(item[2])
            if current and current[-1] == p1:
                current.append(p2)
            else:
                flush()
                current = [p1, p2]
        elif op == "c":                     # cubic bezier: (op, p1, p2, p3, p4)
            p0, c1, c2, p3 = (_pt(item[1]), _pt(item[2]),
                              _pt(item[3]), _pt(item[4]))
            pts = _flatten_bezier(p0, c1, c2, p3, tol)
            if current and current[-1] == p0:
                current.extend(pts)
            else:
                flush()
                current = [p0] + pts
        elif op == "re":                    # rectangle: (op, rect, orientation)
            flush()
            r = item[1]
            polylines.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1),
                              (r.x0, r.y1), (r.x0, r.y0)])
        elif op == "qu":                    # quad: (op, quad)
            flush()
            q = item[1]
            pts = [_pt(q.ul), _pt(q.ur), _pt(q.lr), _pt(q.ll)]
            polylines.append(pts + [pts[0]])
    flush()
    return polylines


def _polyline_to_segments(polyline: list) -> list:
    return [
        (polyline[i], polyline[i + 1])
        for i in range(len(polyline) - 1)
        if polyline[i] != polyline[i + 1]
    ]


def _seg_len(seg) -> float:
    (x1, y1), (x2, y2) = seg
    return math.hypot(x2 - x1, y2 - y1)


# ── public API ─────────────────────────────────────────────────────────────────

def extract_paths(
    page: fitz.Page,
    cfg: SlabV2Config,
    content_rect: fitz.Rect | None = None,
) -> tuple[list[VectorPath], list[StyleClass]]:
    """Extract all vector paths with style classes from one page.

    content_rect: drawing area (excl. title block / legend). Paths fully
    outside are flagged outside_content=True (kept for debug rendering,
    excluded from polygonization).
    """
    drawings = page.get_drawings()
    paths: list[VectorPath] = []
    by_key: dict[StyleKey, list[int]] = defaultdict(list)

    for d in drawings:
        key = StyleKey(
            stroke=_norm_color(d.get("color")),
            fill=_norm_color(d.get("fill")),
            width=round(d.get("width") or 0.0, 2),
            dashes=_norm_dashes(d.get("dashes")),
            even_odd=bool(d.get("even_odd")),
        )
        polylines = _items_to_points(d["items"], cfg.bezier_tol_pt)
        segments = []
        for pl in polylines:
            segments.extend(_polyline_to_segments(pl))
        if not segments:
            continue

        closed = bool(d.get("closePath")) or any(
            len(pl) >= 4 and pl[0] == pl[-1] for pl in polylines
        )
        filled = key.fill is not None

        fill_poly = None
        if filled:
            # largest closed polyline becomes the fill polygon candidate
            best = None
            for pl in polylines:
                ring = pl if pl[0] == pl[-1] else pl + [pl[0]]
                if len(ring) >= 4:
                    try:
                        poly = Polygon(ring)
                        if poly.is_valid and poly.area > 0 and (
                                best is None or poly.area > best.area):
                            best = poly
                    except Exception:
                        pass
            fill_poly = best

        outside = False
        if content_rect is not None:
            xs = [p[0] for s in segments for p in s]
            ys = [p[1] for s in segments for p in s]
            bbox = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
            outside = not bbox.intersects(content_rect)

        vp = VectorPath(
            id=len(paths),
            style_id=-1,                    # assigned below
            segments=segments,
            is_closed=closed,
            is_filled=filled,
            seqno=d.get("seqno") or 0,
            fill_polygon=fill_poly,
            outside_content=outside,
            has_stroke=key.stroke is not None,
        )
        paths.append(vp)
        by_key[key].append(vp.id)

    # build style classes, ids by total stroke length desc (stable ordering)
    stats = []
    for key, ids in by_key.items():
        total_len = 0.0
        n_segs = 0
        seg_lens = []
        xs, ys = [], []
        for pid in ids:
            for seg in paths[pid].segments:
                L = _seg_len(seg)
                total_len += L
                seg_lens.append(L)
                n_segs += 1
                xs.extend((seg[0][0], seg[1][0]))
                ys.extend((seg[0][1], seg[1][1]))
        seg_lens.sort()
        median = seg_lens[len(seg_lens) // 2] if seg_lens else 0.0
        stats.append((key, ids, total_len, n_segs, median,
                      (min(xs), min(ys), max(xs), max(ys))))

    stats.sort(key=lambda s: -s[2])

    classes: list[StyleClass] = []
    page_area = page.rect.width * page.rect.height
    for cid, (key, ids, total_len, n_segs, median, bbox) in enumerate(stats):
        sc = StyleClass(
            id=cid, key=key, n_paths=len(ids), n_segments=n_segs,
            total_length_pt=total_len, bbox=bbox, median_seg_len_pt=median,
        )
        # frame fingerprint: nearly page-sized bbox and few long paths
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bw * bh >= cfg.frame_area_frac * page_area and len(ids) <= 6:
            sc.role = "FRAME"
            sc.role_confidence = 0.9
        # slab edge fingerprint: dark solid stroke, medium+ weight, top-3
        elif (cid <= 2
              and key.stroke is not None
              and max(key.stroke) <= 0.3
              and key.fill is None
              and not key.dashes
              and key.width >= 0.5):
            sc.role = "SLAB_EDGE"
            sc.role_confidence = 0.75
        # hatch fingerprint: fill-only, many micro-segments
        elif (key.fill is not None and key.stroke is None
              and n_segs > 500 and median < 1.0):
            sc.role = "HATCH"
            sc.role_confidence = 0.85
            sc.prefiltered = True
        # Revit column fingerprint: solid dark stroke, no fill, medium weight
        elif (key.stroke is not None and max(key.stroke) <= 0.15
              and key.fill is None and not key.dashes
              and 0.9 <= key.width <= 1.5):
            sc.role = "COLUMN"
            sc.role_confidence = 0.70
        # annotation fingerprint: thin + (dashed or colored)
        elif (key.stroke is not None and key.width <= 0.5
              and (key.dashes or max(key.stroke) > 0.5)):
            sc.role = "ANNOTATION"
            sc.role_confidence = 0.65
        for pid in ids:
            paths[pid].style_id = cid
        classes.append(sc)

    return paths, classes


def class_summary_table(classes: list[StyleClass]) -> list[dict]:
    """JSON-friendly summary used by the profile CLI and Round-1 prompt."""
    return [
        {
            "id": c.id,
            "stroke": c.key.stroke,
            "fill": c.key.fill,
            "width_pt": c.key.width,
            "dashes": c.key.dashes or "solid",
            "n_paths": c.n_paths,
            "n_segments": c.n_segments,
            "total_length_pt": round(c.total_length_pt, 1),
            "median_seg_len_pt": round(c.median_seg_len_pt, 2),
            "role": c.role,
        }
        for c in classes
    ]
