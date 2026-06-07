"""
Building location detection from structural PDF site plans.

Pipeline:
  1. find_location_page()  — ask Gemini which page is the site/location plan
  2. extract_building_polygons() — detect filled polygon per building on that page,
     matched to building names by proximity to text labels.
     Falls back to Gemini Vision if polygon matching is ambiguous.
"""

import pathlib
import sys

# Ensure project root is on sys.path whether this file is imported or run directly
_project_root = str(pathlib.Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import re
from typing import Optional

import fitz
from shapely.geometry import Point, Polygon

from src.ai_floor_analyzer import (
    _get_client,
    _get_model_name,
    _parse_json_response,
    extract_pdf_text_for_ai,
)
from src.pdf_processor import extract_text_blocks
from src.slab_extractor import build_polygons_from_drawings


# ── Step 1: find site plan page ───────────────────────────────────────────────

def find_location_page(
    pdf_path: str,
    page_indices: list[int],
    building_names: list[str],
) -> Optional[int]:
    """
    Ask Gemini which page contains the site plan / location plan.
    Returns 1-indexed page number, or None if not found.
    """
    page_texts = extract_pdf_text_for_ai(pdf_path, page_indices)
    names_str = ", ".join(building_names)
    n = len(building_names)

    prompt = f"""You are analyzing a structural PDF with {n} buildings: {names_str}.

From the page texts below, identify the page number of the SITE PLAN or LOCATION PLAN —
the overview drawing that shows WHERE each building sits on the site relative to each other.

Typical drawing titles: "SITE PLAN", "OVERALL LOCATION SITE PLAN", "KEY PLAN",
"LOCATION PLAN", "OVERALL SITE PLAN", "SITE LAYOUT", "SITE AND LOCATION PLAN"

{page_texts}

Respond ONLY with JSON (no markdown, no explanation):
{{
  "location_page": <1-indexed integer page number, or null if not found>,
  "page_title": "<title text on that page>",
  "confidence": "high|medium|low",
  "notes": "<one sentence explaining your choice>"
}}"""

    from google.genai import types as genai_types
    client = _get_client()
    model = _get_model_name()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0.0),
    )

    raw = response.text or ""
    parsed = _parse_json_response(raw)
    if not parsed:
        print(f"[building_locator] Gemini parse failed. Raw: {raw[:200]}")
        return None

    page = parsed.get("location_page")
    confidence = parsed.get("confidence", "low")
    print(f"[building_locator] Location page: {page}  confidence={confidence}")
    print(f"[building_locator] Title: {parsed.get('page_title')}")
    print(f"[building_locator] Notes: {parsed.get('notes')}")

    if page is None:
        return None
    if confidence == "low":
        print("[building_locator] Low confidence — treat result with caution")
    return int(page)


# ── Step 2a: polygon-based matching ───────────────────────────────────────────

# 1% of page area minimum — excludes text label boxes (~5000 pt²) while keeping
# building footprints (typically 50,000–600,000 pt² on a site plan).
_MIN_AREA_FRACTION = 0.01
_MAX_AREA_FRACTION = 0.90


def _match_polygons_to_buildings(
    candidates: list[Polygon],
    text_blocks: list[dict],
    building_names: list[str],
) -> tuple[dict, dict]:
    """
    Match building names to candidate polygons via label proximity.
    Returns (result_dict, name_to_idx) where result_dict = {name: Polygon}
    and name_to_idx = {name: candidate_index} (used for ambiguity check).
    """
    name_patterns = {
        name: re.compile(re.escape(name.upper()))
        for name in building_names
    }

    label_points: list[tuple[str, float, float]] = []
    for block in text_blocks:
        text_upper = block["text"].strip().upper()
        for name, pattern in name_patterns.items():
            if pattern.search(text_upper):
                bbox = block["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                label_points.append((name, cx, cy))
                break

    result: dict = {}
    name_to_idx: dict = {}

    for name in building_names:
        pts = [(cx, cy) for n, cx, cy in label_points if n == name]
        if not pts:
            continue

        containing: list[tuple[float, int]] = []
        nearest_idx = None
        nearest_dist = float("inf")

        for label_cx, label_cy in pts:
            label_pt = Point(label_cx, label_cy)
            for i, poly in enumerate(candidates):
                if poly.contains(label_pt):
                    containing.append((poly.area, i))
                else:
                    d = poly.centroid.distance(label_pt)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_idx = i

        if containing:
            # Smallest containing polygon = tightest-fitting building footprint
            best_idx = min(containing, key=lambda t: t[0])[1]
        else:
            best_idx = nearest_idx

        if best_idx is not None:
            result[name] = candidates[best_idx]
            name_to_idx[name] = best_idx

    return result, name_to_idx


def _is_ambiguous(name_to_idx: dict, building_names: list[str]) -> bool:
    """True if multiple buildings mapped to the same polygon, or too few matched."""
    matched = [name for name in building_names if name in name_to_idx]
    unique_polys = len(set(name_to_idx.values()))
    if unique_polys < len(matched):
        return True  # collisions
    if len(matched) < max(1, len(building_names) // 2):
        return True  # too few buildings found
    return False


# ── Step 2b: Gemini Vision fallback ───────────────────────────────────────────

_VISION_MAX_PX = 2048   # longest edge cap for Gemini Vision input


def _gemini_vision_fallback(
    pdf_path: str,
    location_page: int,
    building_names: list[str],
    page_width_pt: float,
    page_height_pt: float,
) -> dict:
    """
    Render the site plan page and ask Gemini Vision to locate each building.
    Returns {name: Polygon} in PDF coordinate space (points).
    """
    # Cap resolution so the longest edge <= _VISION_MAX_PX
    longest_pt = max(page_width_pt, page_height_pt)
    dpi = min(150, int(_VISION_MAX_PX / longest_pt * 72))
    dpi = max(dpi, 48)   # floor at 48 DPI so tiny pages still render legibly
    scale = dpi / 72.0
    print(f"[building_locator] Vision render: {dpi} DPI  (scale={scale:.2f})")

    # Render page to PNG bytes
    doc = fitz.open(pdf_path)
    page = doc[location_page - 1]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()

    img_bytes = pix.tobytes("png")
    img_w, img_h = pix.width, pix.height
    names_str = ", ".join(building_names)

    prompt = (
        f"This is a structural engineering SITE PLAN showing buildings: {names_str}.\n\n"
        "For each building label visible in the image, provide a tight bounding box "
        "around the BUILDING FOOTPRINT (the filled/outlined shape representing the "
        "floor area of that building) — not just the text label itself.\n\n"
        f"Image size: {img_w} x {img_h} pixels (width x height). "
        "Coordinates are in pixels measured from the top-left corner.\n\n"
        "Respond ONLY with JSON:\n"
        '{\n'
        '  "buildings": [\n'
        '    {\n'
        '      "name": "Building A",\n'
        '      "bbox_px": [x_min, y_min, x_max, y_max],\n'
        '      "confidence": "high|medium|low",\n'
        '      "found": true\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        'Set "found": false (and omit bbox_px) for any building not clearly visible.'
    )

    from google.genai import types as genai_types
    client = _get_client()
    model = _get_model_name()

    image_part = genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    response = client.models.generate_content(
        model=model,
        contents=[image_part, prompt],
        config=genai_types.GenerateContentConfig(temperature=0.0),
    )

    raw = response.text or ""
    parsed = _parse_json_response(raw)
    if not parsed:
        print(f"[building_locator] Vision fallback parse failed. Raw: {raw[:300]}")
        return {"unmatched": []}

    result: dict = {}
    for bld in parsed.get("buildings", []):
        name = bld.get("name", "")
        if not bld.get("found", True):
            print(f"[building_locator] Vision: {name} — not found in image")
            continue
        bbox_px = bld.get("bbox_px")
        if not bbox_px or len(bbox_px) != 4:
            continue

        x0_px, y0_px, x1_px, y1_px = bbox_px
        # Convert pixel → PDF points, clamped to page
        x0 = max(0.0, min(x0_px / scale, page_width_pt))
        y0 = max(0.0, min(y0_px / scale, page_height_pt))
        x1 = max(0.0, min(x1_px / scale, page_width_pt))
        y1 = max(0.0, min(y1_px / scale, page_height_pt))

        poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        conf = bld.get("confidence", "medium")
        print(f"[building_locator] Vision: {name}  bbox_px={[int(v) for v in bbox_px]}  conf={conf}")
        result[name] = poly

    result["unmatched"] = []
    return result


# ── Step 2 (public): extract building polygons ─────────────────────────────────

def extract_building_polygons(
    pdf_path: str,
    location_page: int,
    building_names: list[str],
) -> dict:
    """
    From the site plan page, extract one Shapely Polygon per building.

    First tries polygon-based matching (fast, deterministic).
    If matching is ambiguous (buildings share a polygon), automatically falls back
    to Gemini Vision which reads the rendered page image.

    Returns:
      {
        "Building A": Polygon(...),   # PDF coordinate space (points)
        "Building B": Polygon(...),
        ...
        "unmatched": [Polygon, ...],  # polygons with no label match
        "_method": "polygon" | "vision",
      }
    """
    doc = fitz.open(pdf_path)
    page_idx = location_page - 1
    if page_idx < 0 or page_idx >= doc.page_count:
        doc.close()
        raise ValueError(f"location_page={location_page} out of range ({doc.page_count} pages)")

    page = doc[page_idx]
    page_w = page.rect.width
    page_h = page.rect.height
    page_area = page_w * page_h
    min_area = page_area * _MIN_AREA_FRACTION
    max_area = page_area * _MAX_AREA_FRACTION

    drawings = page.get_drawings()
    all_pairs = build_polygons_from_drawings(drawings)
    candidates = [p for p, _ in all_pairs if min_area <= p.area <= max_area]

    text_blocks = extract_text_blocks(page)
    doc.close()

    if not candidates:
        print("[building_locator] No polygon candidates — using Vision fallback")
        result = _gemini_vision_fallback(pdf_path, location_page, building_names, page_w, page_h)
        result["_method"] = "vision"
        return result

    result, name_to_idx = _match_polygons_to_buildings(candidates, text_blocks, building_names)

    if _is_ambiguous(name_to_idx, building_names):
        matched = len(name_to_idx)
        unique = len(set(name_to_idx.values()))
        print(
            f"[building_locator] Polygon match ambiguous "
            f"({matched} buildings -> {unique} unique polygons) -- using Vision fallback"
        )
        vision_result = _gemini_vision_fallback(pdf_path, location_page, building_names, page_w, page_h)
        vision_result["_method"] = "vision"
        return vision_result

    # Polygon match succeeded
    matched_indices = set(name_to_idx.values())
    unmatched = [candidates[i] for i in range(len(candidates)) if i not in matched_indices]
    result["unmatched"] = unmatched
    result["_method"] = "polygon"
    print(f"[building_locator] Polygon match: {len(name_to_idx)}/{len(building_names)} buildings matched")
    return result


# ── Visualization ─────────────────────────────────────────────────────────────

_BUILDING_COLORS = [
    (255, 80,  80),    # red
    (80,  160, 255),   # blue
    (80,  220, 80),    # green
    (255, 200, 50),    # yellow
    (200, 80,  255),   # purple
    (255, 140, 40),    # orange
    (40,  220, 220),   # cyan
    (255, 80,  180),   # pink
]


def render_building_polygons(
    pdf_path: str,
    location_page: int,
    building_polygons: dict,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Render site plan page with building polygons overlaid. Saves PNG."""
    from PIL import Image, ImageDraw, ImageFont

    doc = fitz.open(pdf_path)
    page = doc[location_page - 1]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    doc.close()

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    names = [k for k in building_polygons if k not in ("unmatched", "_method")]
    for idx, name in enumerate(names):
        poly = building_polygons.get(name)
        if poly is None or not hasattr(poly, "exterior"):
            continue

        r, g, b = _BUILDING_COLORS[idx % len(_BUILDING_COLORS)]
        pts = [(x * scale, y * scale) for x, y in poly.exterior.coords]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(r, g, b, 55))
            draw.line(pts + [pts[0]], fill=(r, g, b, 220), width=4)

        cx = poly.centroid.x * scale
        cy = poly.centroid.y * scale
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((cx + dx, cy + dy), name, fill=(255, 255, 255, 220), font=font)
        draw.text((cx, cy), name, fill=(r, g, b, 255), font=font)

    method = building_polygons.get("_method", "")
    if method:
        draw.text((10, 10), f"method: {method}", fill=(50, 50, 50, 200), font=font)

    img.save(output_path)
    print(f"[building_locator] Saved: {output_path}")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    if len(sys.argv) < 2:
        print("Usage: python src/building_locator.py <pdf> [Building A] [Building B] ...")
        sys.exit(1)

    pdf = sys.argv[1]
    building_names = sys.argv[2:] if len(sys.argv) > 2 else [
        "Building A", "Building B", "Building C", "Building D"
    ]

    doc = fitz.open(pdf)
    n_pages = doc.page_count
    doc.close()

    print(f"PDF: {pdf}  ({n_pages} pages)")
    print(f"Buildings: {building_names}\n")

    pg = find_location_page(pdf, list(range(n_pages)), building_names)
    print(f"\nLocation page: {pg}")

    if pg:
        polys = extract_building_polygons(pdf, pg, building_names)
        method = polys.get("_method", "?")
        print(f"\nBuilding polygons  [method={method}]:")
        for name, poly in polys.items():
            if name in ("unmatched", "_method"):
                continue
            if hasattr(poly, "area"):
                c = poly.centroid
                print(f"  {name}: area={poly.area:.0f} pt²  centroid=({c.x:.0f}, {c.y:.0f})")
            else:
                print(f"  {name}: not found")
        unmatched = polys.get("unmatched", [])
        print(f"  unmatched: {len(unmatched)} polygon(s)")

        out = str(pathlib.Path(pdf).with_name("building_location_check.png"))
        render_building_polygons(pdf, pg, polys, out, dpi=150)
        print(f"\nCheck the image: {out}")
