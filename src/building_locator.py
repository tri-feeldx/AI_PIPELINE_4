"""
Building location detection from structural PDF site plans.

Pipeline:
  1. find_location_page()  — ask Gemini which page is the site/location plan
  2. extract_building_polygons() — detect filled polygon per building on that page,
     matched to building names by proximity to text labels
"""

import pathlib
import sys

# Ensure project root is on sys.path whether this file is imported or run directly
_project_root = str(pathlib.Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import math
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

    Returns 1-indexed page number, or None if not found / low confidence.
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

    client = _get_client()
    model = _get_model_name()

    from google.genai import types as genai_types
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
        print("[building_locator] Low confidence — returning result but treat with caution")

    return int(page)


# ── Step 2: extract building polygons ─────────────────────────────────────────

# Polygon must be > 1% of page area — filters out text label boxes and tiny annotations.
# Building footprints on a site plan are always a significant fraction of the page.
_MIN_AREA_FRACTION = 0.01
_MAX_AREA_FRACTION = 0.90    # skip full-page border


def extract_building_polygons(
    pdf_path: str,
    location_page: int,
    building_names: list[str],
) -> dict:
    """
    From the site plan page, extract one Shapely Polygon per building.

    Returns dict:
      {
        "Building A": Polygon(...),   # PDF coordinate space (points)
        "Building B": Polygon(...),
        ...
        "unmatched": [Polygon, ...]   # polygons with no nearby building label
      }
    """
    doc = fitz.open(pdf_path)
    page_idx = location_page - 1   # convert to 0-indexed
    if page_idx < 0 or page_idx >= doc.page_count:
        doc.close()
        raise ValueError(f"location_page={location_page} out of range (doc has {doc.page_count} pages)")

    page = doc[page_idx]
    page_area = page.rect.width * page.rect.height
    min_area = page_area * _MIN_AREA_FRACTION
    max_area = page_area * _MAX_AREA_FRACTION

    drawings = page.get_drawings()
    all_pairs = build_polygons_from_drawings(drawings)   # list[(Polygon, color)]

    # Filter by area — keep candidates large enough to be building footprints
    candidates = [
        poly for poly, _color in all_pairs
        if min_area <= poly.area <= max_area
    ]

    # Get text blocks with their bounding boxes
    text_blocks = extract_text_blocks(page)
    doc.close()

    if not candidates:
        return {"unmatched": []}

    # Match building names to polygons
    result = {}
    matched_poly_indices = set()

    # Normalise building names for matching: "Building A" → "BUILDING A" etc.
    name_patterns = {
        name: re.compile(re.escape(name.upper()))
        for name in building_names
    }

    # For each text block, collect label positions
    label_points: list[tuple[str, float, float]] = []  # (name, cx, cy)
    for block in text_blocks:
        text_upper = block["text"].strip().upper()
        for name, pattern in name_patterns.items():
            if pattern.search(text_upper):
                bbox = block["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                label_points.append((name, cx, cy))
                break  # each block → first matching name

    # For each building name, find best polygon.
    # Strategy: among polygons that *contain* the label point, pick the smallest
    # (actual building footprint, not the larger site boundary that encloses everything).
    # If no polygon contains the label, fall back to nearest centroid distance.
    for name in building_names:
        pts = [(cx, cy) for n, cx, cy in label_points if n == name]
        if not pts:
            continue

        containing: list[tuple[float, int]] = []   # (area, idx) for polys containing label
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
            matched_poly_indices.add(best_idx)

    # Collect unmatched polygons
    unmatched = [
        candidates[i] for i in range(len(candidates))
        if i not in matched_poly_indices
    ]
    result["unmatched"] = unmatched

    return result


# ── Visualization ─────────────────────────────────────────────────────────────

# Colors for up to 8 buildings: (R, G, B) 0–255
_BUILDING_COLORS = [
    (255, 80, 80),    # red
    (80, 160, 255),   # blue
    (80, 220, 80),    # green
    (255, 200, 50),   # yellow
    (200, 80, 255),   # purple
    (255, 140, 40),   # orange
    (40, 220, 220),   # cyan
    (255, 80, 180),   # pink
]


def render_building_polygons(
    pdf_path: str,
    location_page: int,
    building_polygons: dict,
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Render the site plan page with building polygons overlaid as colored outlines.
    Saves a PNG to output_path.
    """
    from PIL import Image, ImageDraw, ImageFont

    doc = fitz.open(pdf_path)
    page = doc[location_page - 1]

    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    doc.close()

    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")

    names = [k for k in building_polygons if k != "unmatched"]
    for idx, name in enumerate(names):
        poly = building_polygons.get(name)
        if poly is None or not hasattr(poly, "exterior"):
            continue

        r, g, b = _BUILDING_COLORS[idx % len(_BUILDING_COLORS)]
        fill_color = (r, g, b, 60)     # semi-transparent fill
        outline_color = (r, g, b, 220) # solid outline

        # Scale PDF coords → pixel coords
        pts = [(x * scale, y * scale) for x, y in poly.exterior.coords]
        if len(pts) >= 3:
            draw.polygon(pts, fill=fill_color)
            draw.line(pts + [pts[0]], fill=outline_color, width=3)

        # Label at centroid
        cx = poly.centroid.x * scale
        cy = poly.centroid.y * scale
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # White shadow + colored text
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((cx + dx, cy + dy), name, fill=(255, 255, 255, 220), font=font)
        draw.text((cx, cy), name, fill=(r, g, b, 255), font=font)

    img.save(output_path)
    print(f"[building_locator] Saved: {output_path}")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import pathlib
    # Allow `python src/building_locator.py` from project root
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    if len(sys.argv) < 2:
        print("Usage: python src/building_locator.py <path-to-pdf> [Building A] [Building B] ...")
        sys.exit(1)

    pdf = sys.argv[1]
    building_names = sys.argv[2:] if len(sys.argv) > 2 else [
        "Building A", "Building B", "Building C", "Building D"
    ]

    doc = fitz.open(pdf)
    n_pages = doc.page_count
    doc.close()
    pages = list(range(n_pages))

    print(f"PDF: {pdf}  ({n_pages} pages)")
    print(f"Buildings: {building_names}")
    print()

    pg = find_location_page(pdf, pages, building_names)
    print(f"\nLocation page: {pg}")

    if pg:
        polys = extract_building_polygons(pdf, pg, building_names)
        print("\nBuilding polygons:")
        for name, poly in polys.items():
            if name == "unmatched":
                print(f"  unmatched: {len(poly)} polygon(s)")
            elif hasattr(poly, "area"):
                centroid = poly.centroid
                print(f"  {name}: area={poly.area:.0f} pt²  centroid=({centroid.x:.0f}, {centroid.y:.0f})")
            else:
                print(f"  {name}: not found")

        # Save visualization PNG next to the PDF
        out = str(pathlib.Path(pdf).with_name("building_location_check.png"))
        render_building_polygons(pdf, pg, polys, out, dpi=150)
        print(f"\nCheck the image: {out}")
