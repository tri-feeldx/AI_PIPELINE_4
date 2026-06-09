"""
Column & Foundation Location Detection - LABEL-FIRST approach.
Find text labels matching census keys, then search nearby for matching vector shapes.
"""
import math
from collections import defaultdict
from typing import Optional
import re
import fitz
from shapely.geometry import Polygon, Point
from src.structural_elements import ColumnRegion, FoundationRegion

_PT_TO_MM = 25.4 / 72.0
SIZE_TOLERANCE_MM = 60.0
LABEL_SEARCH_RADIUS_PT = 120.0
MIN_LINE_LENGTH_PT = 0.5


def _pts_to_mm(pts, scale):
    return pts * _PT_TO_MM * scale


def _rect_from_item(item):
    kind = item[0]
    if kind == "re":
        return item[1]
    if kind == "qu":
        q = item[1]
        return fitz.Rect(
            min(q.ul.x, q.ur.x, q.ll.x, q.lr.x),
            min(q.ul.y, q.ur.y, q.ll.y, q.lr.y),
            max(q.ul.x, q.ur.x, q.ll.x, q.lr.x),
            max(q.ul.y, q.ur.y, q.ll.y, q.lr.y),
        )
    return None


def _extract_text_blocks(page):
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    result = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    result.append({"text": t, "bbox": span["bbox"], "size": span.get("size", 0)})
    return result


def _extract_all_rectangles(page):
    rects = []
    for path in page.get_drawings():
        fill = path.get("fill")
        for item in path["items"]:
            rect = _rect_from_item(item)
            if rect is None or rect.width < 2 or rect.height < 2:
                continue
            rects.append({"rect": rect, "fill": fill})
    return rects


def _extract_line_segments(page):
    segments = []
    for path in page.get_drawings():
        for item in path["items"]:
            kind = item[0]
            if kind == "l":
                p1, p2 = item[1], item[2]
                if math.hypot(p2.x - p1.x, p2.y - p1.y) >= MIN_LINE_LENGTH_PT:
                    segments.append({"x1": p1.x, "y1": p1.y, "x2": p2.x, "y2": p2.y})
            elif kind == "qu":
                q = item[1]
                corners = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
                for i in range(4):
                    x1, y1 = corners[i]
                    x2, y2 = corners[(i + 1) % 4]
                    if math.hypot(x2 - x1, y2 - y1) >= MIN_LINE_LENGTH_PT:
                        segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return segments


def _find_nearby_rectangles(cx, cy, all_rects, radius_pt=LABEL_SEARCH_RADIUS_PT):
    candidates = []
    for r in all_rects:
        rect = r["rect"]
        rx, ry = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
        dist = math.hypot(rx - cx, ry - cy)
        if dist <= radius_pt:
            candidates.append((r, dist))
    candidates.sort(key=lambda x: x[1])
    return candidates


def _find_nearby_closed_polylines(cx, cy, segments, radius_pt=LABEL_SEARCH_RADIUS_PT):
    nearby = []
    for seg in segments:
        d1 = math.hypot(seg["x1"] - cx, seg["y1"] - cy)
        d2 = math.hypot(seg["x2"] - cx, seg["y2"] - cy)
        if d1 <= radius_pt or d2 <= radius_pt:
            nearby.append(seg)
    if len(nearby) < 4:
        return []
    points = defaultdict(list)
    for seg in nearby:
        points[(round(seg["x1"], 1), round(seg["y1"], 1))].append(seg)
        points[(round(seg["x2"], 1), round(seg["y2"], 1))].append(seg)
    corners = [pt for pt, segs in points.items() if len(segs) >= 2]
    if len(corners) < 4:
        return []
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w < 2 or h < 2:
        return []
    if w > 0 and h > 0 and max(w, h) / min(w, h) > 10:
        return []
    rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
    rcx, rcy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    dist = math.hypot(rcx - cx, rcy - cy)
    return [{"rect": rect, "fill": None, "dist": dist}]


def _find_labels(text_blocks, census_keys):
    """Find text blocks matching census keys (exact or prefix like C1a->C1)."""
    upper_map = {k.upper(): k for k in census_keys.keys()}
    candidates = []
    for block in text_blocks:
        text = block["text"].strip().upper()
        if not text:
            continue
        if text in upper_map:
            candidates.append({
                "text": upper_map[text],
                "cx": (block["bbox"][0] + block["bbox"][2]) / 2,
                "cy": (block["bbox"][1] + block["bbox"][3]) / 2,
                "bbox": block["bbox"],
            })
            continue
        for key in census_keys.keys():
            if text.startswith(key.upper()) and len(text) <= len(key) + 2:
                suffix = text[len(key):]
                if suffix == "" or suffix.isalpha():
                    candidates.append({
                        "text": key,
                        "cx": (block["bbox"][0] + block["bbox"][2]) / 2,
                        "cy": (block["bbox"][1] + block["bbox"][3]) / 2,
                        "bbox": block["bbox"],
                    })
                    break
    # Concrete column grammar: PDF text often extracts circled C/10 as two spans.
    # Join a "C" span with a nearby number span below/above it, normalizing to C10.
    concrete_keys = {k.upper(): k for k in census_keys.keys() if re.fullmatch(r"C\d+", k.upper())}
    if concrete_keys:
        c_blocks = [b for b in text_blocks if b["text"].strip().upper() == "C"]
        n_blocks = [b for b in text_blocks if re.fullmatch(r"\d{1,3}", b["text"].strip())]
        existing = {
            (c["text"].upper(), round(c["cx"], 1), round(c["cy"], 1))
            for c in candidates
        }
        for cb in c_blocks:
            ccx = (cb["bbox"][0] + cb["bbox"][2]) / 2
            ccy = (cb["bbox"][1] + cb["bbox"][3]) / 2
            best = None
            best_dist = float("inf")
            for nb in n_blocks:
                ncx = (nb["bbox"][0] + nb["bbox"][2]) / 2
                ncy = (nb["bbox"][1] + nb["bbox"][3]) / 2
                dx = abs(ncx - ccx)
                dy = abs(ncy - ccy)
                if dx <= 8 and 4 <= dy <= 18:
                    dist = math.hypot(dx, dy)
                    if dist < best_dist:
                        best = nb
                        best_dist = dist
            if not best:
                continue
            symbol_u = f"C{best['text'].strip()}".upper()
            if symbol_u not in concrete_keys:
                continue
            x0 = min(cb["bbox"][0], best["bbox"][0])
            y0 = min(cb["bbox"][1], best["bbox"][1])
            x1 = max(cb["bbox"][2], best["bbox"][2])
            y1 = max(cb["bbox"][3], best["bbox"][3])
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            key = (symbol_u, round(cx, 1), round(cy, 1))
            if key in existing:
                continue
            candidates.append({
                "text": concrete_keys[symbol_u],
                "cx": cx,
                "cy": cy,
                "bbox": (x0, y0, x1, y1),
            })
            existing.add(key)
    return candidates


def _match_shape_to_label(cx, cy, expected_w, expected_d, scale,
                          all_rects, all_segments, used_rects):
    """Find the best matching shape near (cx,cy) for given expected dimensions."""
    best_shape, best_err, best_polygon = None, float("inf"), None

    for r_info, dist in _find_nearby_rectangles(cx, cy, all_rects):
        rect = r_info["rect"]
        rk = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
        if rk in used_rects:
            continue
        w_mm, h_mm = _pts_to_mm(rect.width, scale), _pts_to_mm(rect.height, scale)
        if expected_w > 0 and expected_d > 0:
            err = min(abs(w_mm - expected_w) + abs(h_mm - expected_d),
                      abs(w_mm - expected_d) + abs(h_mm - expected_w))
        else:
            err = 0
        if err < best_err:
            best_err, best_shape = err, rk
            best_polygon = Polygon([
                (rect.x0, rect.y0), (rect.x1, rect.y0),
                (rect.x1, rect.y1), (rect.x0, rect.y1),
            ])

    if best_shape is None:
        for pr in _find_nearby_closed_polylines(cx, cy, all_segments):
            rect = pr["rect"]
            rk = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
            if rk in used_rects:
                continue
            w_mm, h_mm = _pts_to_mm(rect.width, scale), _pts_to_mm(rect.height, scale)
            if expected_w > 0 and expected_d > 0:
                err = min(abs(w_mm - expected_w) + abs(h_mm - expected_d),
                          abs(w_mm - expected_d) + abs(h_mm - expected_w))
            else:
                err = 0
            if err < best_err:
                best_err, best_shape = err, rk
                best_polygon = Polygon([
                    (rect.x0, rect.y0), (rect.x1, rect.y0),
                    (rect.x1, rect.y1), (rect.x0, rect.y1),
                ])
    return best_shape, best_err, best_polygon


def _symbol_info(column_types: dict, symbol: str) -> dict:
    return column_types.get(symbol) or column_types.get(symbol.upper()) or {}


def build_column_types_from_intelligence(document_intelligence: dict) -> dict:
    """Convert document intelligence column_symbols into detector input."""
    out = {}
    for symbol, info in (document_intelligence or {}).get("column_symbols", {}).items():
        if not isinstance(info, dict):
            continue
        if str(symbol).strip().upper() == "C":
            # Legend family marker only; real concrete instances are C<number>.
            continue
        out[str(symbol)] = {
            "width_mm": info.get("width_mm"),
            "depth_mm": info.get("depth_mm"),
            "count_total": info.get("count_total"),
            "family": info.get("family", "unknown_column"),
            "status": info.get("status", "unknown"),
            "source": info.get("source", "document_intelligence"),
        }
    return out


def build_foundation_types_from_intelligence(document_intelligence: dict) -> dict:
    """Convert document intelligence foundation_symbols into detector input."""
    out = {}
    for symbol, info in (document_intelligence or {}).get("foundation_symbols", {}).items():
        if not isinstance(info, dict):
            continue
        depth_below = info.get("depth_below_gl_mm")
        thickness = info.get("thickness_mm")
        out[str(symbol)] = {
            "width_mm": info.get("width_mm"),
            "depth_mm": info.get("depth_mm"),
            "type": info.get("type", "unknown"),
            "depth_below_gl_mm": depth_below if depth_below is not None else 0,
            "thickness_mm": thickness if thickness is not None else 0,
            "source": info.get("source", "document_intelligence"),
        }
    return out


# =============================================================================
# Public API
# =============================================================================

def detect_columns_on_page(page, column_types, scale, page_index,
                           building="", level="", is_detail_page=False):
    """LABEL-FIRST column detection on a single page."""
    if not column_types:
        return []
    text_blocks = _extract_text_blocks(page)
    all_rects = _extract_all_rectangles(page)
    all_segments = _extract_line_segments(page)
    labels = _find_labels(text_blocks, column_types)
    labels = [
        c for c in labels
        if c["cx"] < page.rect.width * 0.82 and c["cy"] < page.rect.height * 0.88
    ]
    print(f"[ColumnDetector] P{page_index + 1}: {len(labels)} labels in {len(text_blocks)} text blocks")
    columns, used = [], set()
    for i, c in enumerate(labels):
        exp = _symbol_info(column_types, c["text"])
        ew = float(exp.get("width_mm") or 0)
        ed = float(exp.get("depth_mm") or 0)
        shape, err, poly = _match_shape_to_label(c["cx"], c["cy"], ew, ed, scale,
                                                  all_rects, all_segments, used)
        if shape is not None and err <= SIZE_TOLERANCE_MM * 2:
            used.add(shape)
        elif shape is None and ew == 0:
            bb = c["bbox"]
            poly = Polygon([(bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3])])
        elif shape is None:
            continue
        if ew == 0 and poly is not None:
            ew = _pts_to_mm(poly.bounds[2] - poly.bounds[0], scale)
        if ed == 0 and poly is not None:
            ed = _pts_to_mm(poly.bounds[3] - poly.bounds[1], scale)
        columns.append(ColumnRegion(
            id=i, polygon=poly, symbol=c["text"],
            width_mm=ew, depth_mm=ed, building=building,
            level=level, page_index=page_index,
            is_detail_only=is_detail_page,
            family=exp.get("family", "unknown_column"),
            status=exp.get("status", "unknown"),
            detection_confidence=0.85 if shape is not None else 0.55,
            source=exp.get("source", "geometry"),
        ))
    print(f"[ColumnDetector] P{page_index + 1}: {len(columns)} columns detected")
    return columns


def detect_foundations_on_page(page, foundation_types, scale, page_index):
    """LABEL-FIRST foundation detection on a single page."""
    if not foundation_types:
        return []
    text_blocks = _extract_text_blocks(page)
    all_rects = _extract_all_rectangles(page)
    all_segments = _extract_line_segments(page)
    labels = _find_labels(text_blocks, foundation_types)
    print(f"[FoundationDetector] P{page_index + 1}: {len(labels)} labels")
    foundations, used = [], set()
    for i, c in enumerate(labels):
        exp = foundation_types.get(c["text"], {})
        ew = float(exp.get("width_mm") or 0)
        ed = float(exp.get("depth_mm") or 0)
        shape, err, poly = _match_shape_to_label(c["cx"], c["cy"], ew, ed, scale,
                                                  all_rects, all_segments, used)
        if shape is not None and err <= SIZE_TOLERANCE_MM * 2:
            used.add(shape)
        elif shape is None and ew == 0:
            bb = c["bbox"]
            poly = Polygon([(bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3])])
        elif shape is None:
            continue
        if ew == 0 and poly is not None:
            ew = _pts_to_mm(poly.bounds[2] - poly.bounds[0], scale)
        if ed == 0 and poly is not None:
            ed = _pts_to_mm(poly.bounds[3] - poly.bounds[1], scale)
        foundations.append(FoundationRegion(
            id=i, polygon=poly, symbol=c["text"],
            fdn_type=exp.get("type", "pad"),
            width_mm=ew, depth_mm=ed,
            depth_below_gl_mm=float(exp.get("depth_below_gl_mm") or 0),
            page_index=page_index,
            thickness_mm=float(exp.get("thickness_mm") or 0),
            detection_confidence=0.85 if shape is not None else 0.55,
            source=exp.get("source", "geometry"),
        ))
    print(f"[FoundationDetector] P{page_index + 1}: {len(foundations)} foundations detected")
    return foundations


def assign_columns_to_regions(
    columns: list,
    column_census: dict,
    ai_floor_result: dict = None,
) -> list:
    """
    Cross-reference detected columns with Gemini census to fill in
    building and level info where column_detector couldn't infer it.
    """
    if not ai_floor_result or not column_census:
        return columns

    page_to_info: dict[int, tuple[str, str]] = {}
    for bldg in ai_floor_result.get("buildings", []):
        bname = bldg.get("name", "")
        for floor in bldg.get("floors", []):
            lname = floor.get("level_name", "")
            for pg in floor.get("slab_plan_pages", []):
                if isinstance(pg, int):
                    page_to_info[pg - 1] = (bname, lname)

    for col in columns:
        if not col.building and col.page_index in page_to_info:
            col.building, col.level = page_to_info[col.page_index]

    return columns
