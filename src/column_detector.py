"""
Column & Foundation Location Detection — vector path extraction only.
No Vision API used. Relies on PDF filled rectangles matched against
the column/foundation census from column_analyzer.py.

Public API:
    detect_columns_on_page(page, column_types, scale, page_index, ...) -> list[ColumnRegion]
    detect_foundations_on_page(page, foundation_types, scale, page_index) -> list[FoundationRegion]
    assign_columns_to_regions(columns, column_census, ai_floor_result) -> list[ColumnRegion]
"""

import math
from typing import Optional

import fitz
from shapely.geometry import Polygon, box as shapely_box

from src.structural_elements import ColumnRegion, FoundationRegion

# PDF: 1 pt = 1/72 inch. Scale 1:N means 1 pt = N/72 inch real-world.
# pts_to_mm(pts, scale) = pts * (25.4 / 72) * scale
_PT_TO_MM = 25.4 / 72.0

SIZE_TOLERANCE_MM = 60.0   # ±mm tolerance when matching rect to column type
MIN_FILL_DARKNESS  = 0.5   # fill color darkness threshold (0=white, 1=black)
LABEL_SEARCH_RADIUS_PT = 60.0   # radius (PDF pts) to look for column label text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pts_to_mm(pts: float, scale: int) -> float:
    return pts * _PT_TO_MM * scale


def _rect_from_item(item) -> Optional[fitz.Rect]:
    """Extract fitz.Rect from a drawing item (kind 're' or 'qu')."""
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


def _is_dark_fill(color) -> bool:
    """Return True if fill color is dark (indicates a solid filled column symbol)."""
    if color is None:
        return False
    if isinstance(color, (int, float)):
        return color < (1.0 - MIN_FILL_DARKNESS)
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        return brightness < (1.0 - MIN_FILL_DARKNESS)
    return False


def _match_symbol(w_mm: float, d_mm: float, column_types: dict) -> Optional[str]:
    """
    Find the best-matching column/foundation symbol for given dimensions.
    Returns symbol string or None if no match within tolerance.
    """
    best_sym, best_err = None, float("inf")
    for sym, info in column_types.items():
        cw = info.get("width_mm")
        cd = info.get("depth_mm")
        if cw is None or cd is None:
            continue   # skip types where Gemini couldn't determine dimensions
        cw = float(cw)
        cd = float(cd)
        # Try both orientations (rotated 90°)
        err1 = abs(w_mm - cw) + abs(d_mm - cd)
        err2 = abs(w_mm - cd) + abs(d_mm - cw)
        err  = min(err1, err2)
        if err < best_err:
            best_err = err
            best_sym = sym
    if best_err <= SIZE_TOLERANCE_MM * 2:
        return best_sym
    return None


def _find_nearby_label(cx: float, cy: float, text_blocks: list,
                       radius_pt: float = LABEL_SEARCH_RADIUS_PT) -> str:
    """Return the nearest text label within radius_pt of center (cx, cy)."""
    best_text, best_dist = "", float("inf")
    for block in text_blocks:
        bx = (block["bbox"][0] + block["bbox"][2]) / 2
        by = (block["bbox"][1] + block["bbox"][3]) / 2
        dist = math.hypot(bx - cx, by - cy)
        if dist < best_dist and dist <= radius_pt:
            best_dist = dist
            best_text = block.get("text", "").strip()
    return best_text


def _extract_text_blocks(page: fitz.Page) -> list:
    """Fast text block extraction — reuse pattern from pdf_processor."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    result = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t:
                    result.append({"text": t, "bbox": span["bbox"],
                                   "size": span.get("size", 0)})
    return result


# ── Column detection ──────────────────────────────────────────────────────────

def detect_columns_on_page(
    page: fitz.Page,
    column_types: dict,
    scale: int,
    page_index: int,
    building: str = "",
    level: str = "",
    is_detail_page: bool = False,
) -> list:
    """
    Detect column locations on a single PDF page using vector path analysis.

    Strategy:
      1. Collect all filled rectangles from page.get_drawings()
      2. Convert size to real-world mm using drawing scale
      3. Match against column_types census (±tolerance)
      4. Find nearest text label to confirm/assign symbol
      5. Return list[ColumnRegion]

    No Vision API — pure vector path extraction.
    """
    if not column_types:
        return []

    text_blocks = _extract_text_blocks(page)
    columns: list[ColumnRegion] = []
    col_id = 0

    for path in page.get_drawings():
        fill = path.get("fill")
        if not _is_dark_fill(fill):
            continue   # skip unfilled / light-colored paths

        for item in path["items"]:
            rect = _rect_from_item(item)
            if rect is None:
                continue

            w_pt = rect.width
            h_pt = rect.height
            if w_pt < 2 or h_pt < 2:
                continue   # degenerate

            w_mm = _pts_to_mm(w_pt, scale)
            h_mm = _pts_to_mm(h_pt, scale)

            symbol = _match_symbol(w_mm, h_mm, column_types)
            if symbol is None:
                continue

            cx, cy = rect.x0 + w_pt / 2, rect.y0 + h_pt / 2
            label  = _find_nearby_label(cx, cy, text_blocks)
            # If nearby text matches a column symbol, prefer it
            if label.upper() in column_types:
                symbol = label.upper()

            poly = Polygon([
                (rect.x0, rect.y0), (rect.x1, rect.y0),
                (rect.x1, rect.y1), (rect.x0, rect.y1),
            ])
            info = column_types[symbol]
            columns.append(ColumnRegion(
                id=col_id,
                polygon=poly,
                symbol=symbol,
                width_mm=info.get("width_mm", w_mm),
                depth_mm=info.get("depth_mm", h_mm),
                building=building,
                level=level,
                page_index=page_index,
                is_detail_only=is_detail_page,
            ))
            col_id += 1

    print(f"[ColumnDetector] Page {page_index + 1}: {len(columns)} column(s) found "
          f"(detail={is_detail_page})")
    return columns


# ── Foundation detection ──────────────────────────────────────────────────────

def detect_foundations_on_page(
    page: fitz.Page,
    foundation_types: dict,
    scale: int,
    page_index: int,
) -> list:
    """
    Detect foundation/footing locations on a footing plan page.
    Uses same vector path strategy as column detection.
    No Vision API.
    """
    if not foundation_types:
        return []

    text_blocks = _extract_text_blocks(page)
    foundations: list[FoundationRegion] = []
    fdn_id = 0

    for path in page.get_drawings():
        fill = path.get("fill")
        # Foundations may be lighter (hatched) — relax darkness filter
        if fill is None:
            continue

        for item in path["items"]:
            rect = _rect_from_item(item)
            if rect is None:
                continue

            w_pt = rect.width
            h_pt = rect.height
            if w_pt < 4 or h_pt < 4:
                continue

            w_mm = _pts_to_mm(w_pt, scale)
            h_mm = _pts_to_mm(h_pt, scale)

            symbol = _match_symbol(w_mm, h_mm, foundation_types)
            if symbol is None:
                continue

            cx, cy = rect.x0 + w_pt / 2, rect.y0 + h_pt / 2
            label  = _find_nearby_label(cx, cy, text_blocks)
            if label.upper() in foundation_types:
                symbol = label.upper()

            info = foundation_types[symbol]
            poly = Polygon([
                (rect.x0, rect.y0), (rect.x1, rect.y0),
                (rect.x1, rect.y1), (rect.x0, rect.y1),
            ])
            foundations.append(FoundationRegion(
                id=fdn_id,
                polygon=poly,
                symbol=symbol,
                fdn_type=info.get("type", "pad"),
                width_mm=info.get("width_mm", w_mm),
                depth_mm=info.get("depth_mm", h_mm),
                depth_below_gl_mm=info.get("depth_below_gl_mm", 0.0),
                page_index=page_index,
            ))
            fdn_id += 1

    print(f"[ColumnDetector] Page {page_index + 1}: {len(foundations)} foundation(s) found")
    return foundations


# ── Building/level assignment ─────────────────────────────────────────────────

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

    # Build page_index → (building, level) map from ai_floor_result
    page_to_info: dict[int, tuple[str, str]] = {}
    for bldg in ai_floor_result.get("buildings", []):
        bname = bldg.get("name", "")
        for floor in bldg.get("floors", []):
            lname = floor.get("level_name", "")
            for pg in floor.get("slab_plan_pages", []):
                page_to_info[pg - 1] = (bname, lname)   # convert to 0-indexed

    for col in columns:
        if not col.building and col.page_index in page_to_info:
            col.building, col.level = page_to_info[col.page_index]

    return columns
