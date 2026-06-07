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

_VISION_MAX_PX = 4096   # longest edge cap for Gemini Vision input


def _gemini_vision_fallback(
    pdf_path: str,
    location_page: int,
    building_names: list[str],
    page_width_pt: float,
    page_height_pt: float,
    candidates: list = None,
) -> dict:
    """
    Hybrid fallback using Gemini Vision:
      1. Ask Gemini where each building LABEL text is located (pixel coordinate).
      2. Convert label pixel → PDF point, then assign the nearest/containing PDF polygon.
      3. If no PDF polygon found for a building, fall back to Gemini's bbox_px rectangle.

    This is more accurate than asking Gemini to draw bboxes directly, because
    Gemini reliably reads text positions but struggles with tight footprint geometry.
    """
    longest_pt = max(page_width_pt, page_height_pt)
    dpi = min(150, int(_VISION_MAX_PX / longest_pt * 72))
    dpi = max(dpi, 48)
    scale = dpi / 72.0
    print(f"[building_locator] Vision render: {dpi} DPI  (scale={scale:.2f})")

    doc = fitz.open(pdf_path)
    page = doc[location_page - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    doc.close()

    img_bytes = pix.tobytes("png")
    img_w, img_h = pix.width, pix.height
    names_str = ", ".join(building_names)

    prompt = f"""This is a structural engineering SITE PLAN image.

Buildings to find: {names_str}

For each building, I need TWO things:
A) LABEL POSITION: the pixel coordinate of the CENTER of the building's text label
   (labels may be horizontal or vertical/rotated, e.g. "BUILDING A REFER TO DRAWING S100-S199")
B) FOOTPRINT BBOX: a tight bounding box around the building's floor area shape
   (the filled coloured shape or dashed outline that shows where the building sits on the site)

Image size: {img_w} x {img_h} pixels, top-left origin.

Important notes:
- Labels are often placed OUTSIDE or BELOW the footprint shape, pointing to it
- Labels may be written vertically (rotated 90 degrees)
- Stage 2 / future buildings may have only a dashed outline -- still provide their bbox
- Do NOT divide the drawing into equal strips; follow the actual shapes

Respond ONLY with JSON (no markdown):
{{
  "reasoning": "<1-2 sentences describing the building shapes and label positions you see>",
  "buildings": [
    {{
      "name": "Building A",
      "label_px": [cx, cy],
      "bbox_px": [x_min, y_min, x_max, y_max],
      "confidence": "high|medium|low",
      "found": true
    }}
  ]
}}

Set "found": false and omit label_px / bbox_px for buildings not visible in the image."""

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
        print(f"[building_locator] Vision parse failed. Raw: {raw[:300]}")
        return {"unmatched": []}

    reasoning = parsed.get("reasoning", "")
    if reasoning:
        print(f"[building_locator] Vision reasoning: {reasoning}")

    result: dict = {}
    bld_data = {b.get("name", ""): b for b in parsed.get("buildings", [])}

    for name in building_names:
        bld = bld_data.get(name, {})
        if not bld.get("found", True):
            print(f"[building_locator] Vision: {name} - not found")
            continue

        conf = bld.get("confidence", "medium")
        label_px = bld.get("label_px")
        bbox_px = bld.get("bbox_px")

        # --- Strategy A: use label position to assign a PDF polygon ---
        matched_poly = None
        if label_px and candidates:
            lx_pt = label_px[0] / scale
            ly_pt = label_px[1] / scale
            label_pt = Point(lx_pt, ly_pt)

            # Find smallest polygon that contains the label point
            containing = [(p.area, i) for i, p in enumerate(candidates) if p.contains(label_pt)]
            if containing:
                best_idx = min(containing, key=lambda t: t[0])[1]
                matched_poly = candidates[best_idx]
            else:
                # Nearest centroid within 500pt; beyond that use bbox fallback
                nearest_idx = min(range(len(candidates)),
                                  key=lambda i: candidates[i].centroid.distance(label_pt))
                dist = candidates[nearest_idx].centroid.distance(label_pt)
                if dist < 500:
                    matched_poly = candidates[nearest_idx]

        if matched_poly is not None:
            print(f"[building_locator] Vision+polygon: {name}  label_px={label_px}  "
                  f"poly_area={matched_poly.area:.0f}  conf={conf}")
            result[name] = matched_poly
            continue

        # --- Strategy B: use Gemini bbox rectangle directly ---
        if bbox_px and len(bbox_px) == 4:
            x0_px, y0_px, x1_px, y1_px = bbox_px
            x0 = max(0.0, min(x0_px / scale, page_width_pt))
            y0 = max(0.0, min(y0_px / scale, page_height_pt))
            x1 = max(0.0, min(x1_px / scale, page_width_pt))
            y1 = max(0.0, min(y1_px / scale, page_height_pt))
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            print(f"[building_locator] Vision bbox: {name}  bbox_px={bbox_px}  conf={conf}")
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
        result = _gemini_vision_fallback(pdf_path, location_page, building_names, page_w, page_h, [])
        result["_method"] = "vision"
        save_debug_outputs(pdf_path, location_page, result)
        return result

    result, name_to_idx = _match_polygons_to_buildings(candidates, text_blocks, building_names)

    if _is_ambiguous(name_to_idx, building_names):
        matched = len(name_to_idx)
        unique = len(set(name_to_idx.values()))
        print(
            f"[building_locator] Polygon match ambiguous "
            f"({matched} buildings -> {unique} unique polygons) -- using Vision fallback"
        )
        vision_result = _gemini_vision_fallback(pdf_path, location_page, building_names, page_w, page_h, candidates)
        vision_result["_method"] = "vision"
        save_debug_outputs(pdf_path, location_page, vision_result)
        return vision_result

    # Polygon match succeeded
    matched_indices = set(name_to_idx.values())
    unmatched = [candidates[i] for i in range(len(candidates)) if i not in matched_indices]
    result["unmatched"] = unmatched
    result["_method"] = "polygon"
    print(f"[building_locator] Polygon match: {len(name_to_idx)}/{len(building_names)} buildings matched")
    save_debug_outputs(pdf_path, location_page, result)
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


# ── Debug output ──────────────────────────────────────────────────────────────

def save_debug_outputs(
    pdf_path: str,
    location_page: int,
    building_polygons: dict,
    debug_dir: str = "debug_ai",
) -> None:
    """
    Save three debug artifacts to debug_dir/:
      - building_location_page.png   raw site plan page (no overlay)
      - building_location_result.png site plan + colored polygon overlay
      - building_location.json       JSON summary with polygon coordinates
    """
    import json
    from datetime import datetime
    from pathlib import Path

    Path(debug_dir).mkdir(exist_ok=True)
    pdf_name = Path(pdf_path).name

    # Compute render scale (same cap as vision: longest edge <= 4096px)
    doc = fitz.open(pdf_path)
    page = doc[location_page - 1]
    longest_pt = max(page.rect.width, page.rect.height)
    doc.close()
    dpi = max(72, min(150, int(4096 / longest_pt * 72)))
    scale = dpi / 72.0

    # 1. Raw page PNG
    doc = fitz.open(pdf_path)
    page = doc[location_page - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    doc.close()
    raw_path = f"{debug_dir}/building_location_page.png"
    pix.save(raw_path)

    # 2. Overlay PNG
    result_path = f"{debug_dir}/building_location_result.png"
    render_building_polygons(pdf_path, location_page, building_polygons, result_path, dpi=dpi)

    # 3. JSON summary
    buildings_out = []
    for name, poly in building_polygons.items():
        if name in ("unmatched", "_method"):
            continue
        if hasattr(poly, "exterior"):
            pts = [[round(x, 1), round(y, 1)] for x, y in poly.exterior.coords]
            c = poly.centroid
            b = poly.bounds
            buildings_out.append({
                "name": name,
                "found": True,
                "polygon_pts": pts,
                "area_pt2": round(poly.area, 1),
                "centroid": [round(c.x, 1), round(c.y, 1)],
                "bbox": [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)],
            })
        else:
            buildings_out.append({"name": name, "found": False})

    summary = {
        "pdf": pdf_name,
        "location_page": location_page,
        "method": building_polygons.get("_method", "unknown"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "buildings": buildings_out,
        "unmatched_count": len(building_polygons.get("unmatched", [])),
    }
    json_path = f"{debug_dir}/building_location.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[building_locator] Debug: {raw_path}")
    print(f"[building_locator] Debug: {result_path}")
    print(f"[building_locator] Debug: {json_path}")


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
                print(f"  {name}: area={poly.area:.0f} pt2  centroid=({c.x:.0f}, {c.y:.0f})")
            else:
                print(f"  {name}: not found")
        unmatched = polys.get("unmatched", [])
        print(f"  unmatched: {len(unmatched)} polygon(s)")
        print("\nDebug files saved to: debug_ai/")
