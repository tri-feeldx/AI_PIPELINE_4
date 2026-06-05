"""
Coordinate mapping: PDF point space → real-world millimeters.

PDF coordinate system:
  - Origin at bottom-left of page
  - 1 PDF point = 1/72 inch
  - fitz uses top-left origin (y increases downward)

Real-world coordinate system:
  - Origin at bottom-left of drawing extents
  - Units: mm
  - Y-axis flipped (PDF y-down → real y-up)

Scale: if drawing scale is 1:100, then 1mm on paper = 100mm in reality.
  1 PDF point = (25.4/72) mm on paper = (25.4/72) * scale mm in reality
"""

import copy
import math
from shapely.geometry import Polygon
from shapely.affinity import scale as shapely_scale, translate, affine_transform
import fitz

from src.pipeline_logger import log_warn, log_slab


PT_TO_MM = 25.4 / 72.0  # 1 PDF point in mm (on paper)


def pts_to_real_mm(pdf_pts: float, scale: int) -> float:
    """Convert PDF point distance to real-world mm."""
    return pdf_pts * PT_TO_MM * scale


def pdf_point_to_real(
    x_pdf: float,
    y_pdf: float,
    page_height_pts: float,
    scale: int,
    origin_x_pdf: float = 0.0,
    origin_y_pdf: float = 0.0,
) -> tuple[float, float]:
    """
    Convert a single PDF point (x, y) to real-world mm.
    Flips Y axis (PDF: y-down → real: y-up).
    origin_x/y_pdf: PDF coordinate of the real-world origin (0,0).
    """
    # fitz top-left origin → standard bottom-left: flip Y
    y_flipped = page_height_pts - y_pdf
    origin_y_flipped = page_height_pts - origin_y_pdf

    real_x = (x_pdf - origin_x_pdf) * PT_TO_MM * scale
    real_y = (y_flipped - origin_y_flipped) * PT_TO_MM * scale
    return real_x, real_y


def transform_polygon(
    polygon: Polygon,
    page: fitz.Page,
    scale: int,
    origin_x_pdf: float = 0.0,
    origin_y_pdf: float = 0.0,
) -> Polygon:
    """Transform a Shapely polygon from PDF coords to real-world mm."""
    page_height = page.rect.height

    def transform_pt(x, y):
        return pdf_point_to_real(x, y, page_height, scale, origin_x_pdf, origin_y_pdf)

    exterior = [transform_pt(x, y) for x, y in polygon.exterior.coords]
    interiors = [
        [transform_pt(x, y) for x, y in interior.coords]
        for interior in polygon.interiors
    ]
    try:
        return Polygon(exterior, interiors)
    except Exception:
        return Polygon(exterior)


def transform_all_slabs(slabs, page: fitz.Page, scale: int) -> list:
    """
    Transform all SlabRegion polygons to real-world mm.
    MultiPolygon slabs are split into separate SlabRegion objects (no silent data loss).
    Origin is fixed to the page bottom-left corner for cross-page consistency.
    """
    from shapely.geometry import MultiPolygon

    # Expand MultiPolygon slabs into individual SlabRegion objects
    expanded = []
    for slab in slabs:
        if isinstance(slab.polygon, MultiPolygon):
            components = sorted(slab.polygon.geoms, key=lambda g: g.area, reverse=True)
            log_warn(slab.page_index,
                     f"Slab {slab.label}: MultiPolygon {len(components)} parts → splitting into separate slabs")
            for j, comp in enumerate(components):
                new_slab = copy.copy(slab)
                new_slab.polygon = comp
                if j > 0:
                    new_slab.label = f"{slab.label}_{j + 1}"
                expanded.append(new_slab)
        else:
            expanded.append(slab)

    slabs = expanded

    if not slabs:
        return slabs

    # Fixed origin: bottom-left corner of the PDF page (consistent across all pages of same size)
    # fitz uses top-left origin (y increases downward), so bottom-left = (x0, y1)
    origin_x_pdf = page.rect.x0   # = 0 for standard PDFs
    origin_y_pdf = page.rect.y1   # = page height in fitz coords (bottom of page)

    for slab in slabs:
        real_poly = transform_polygon(slab.polygon, page, scale, origin_x_pdf, origin_y_pdf)
        slab.real_polygon = real_poly
        slab.area_m2 = real_poly.area / 1_000_000.0
        log_slab(slab.page_index, slab)

    return slabs


def estimate_origin_from_grid(text_blocks: list[dict], page: fitz.Page) -> tuple[float, float]:
    """
    Try to find the drawing origin by looking for '0,0' or 'A/1' grid intersection.
    Falls back to bottom-left of page.
    """
    # Simple fallback: use page bottom-left corner
    return 0.0, page.rect.height


def auto_scale_from_drawing(
    text_blocks: list[dict],
    drawings: list[dict],
    page: fitz.Page,
) -> int | None:
    """
    Attempt to auto-detect scale by finding a dimension annotation
    near a line and computing: annotation_value_mm / line_length_pts * PT_TO_MM
    Returns scale int or None.
    """
    import re
    # Look for dimension numbers (standalone numbers like "3600", "5400")
    dim_pattern = re.compile(r"^\d{3,5}$")
    dim_blocks = [b for b in text_blocks if dim_pattern.match(b["text"].strip())]
    if not dim_blocks:
        return None

    # For each dimension text, find the nearest line segment and compute scale
    scales = []
    for db in dim_blocks[:10]:
        cx = (db["bbox"][0] + db["bbox"][2]) / 2
        cy = (db["bbox"][1] + db["bbox"][3]) / 2
        dim_val = float(db["text"].strip())

        best_len = None
        best_dist = float("inf")
        for d in drawings:
            for item in d.get("items", []):
                if item[0] == "l":
                    p1 = item[1]
                    p2 = item[2]
                    mid_x = (p1.x + p2.x) / 2
                    mid_y = (p1.y + p2.y) / 2
                    dist = math.hypot(cx - mid_x, cy - mid_y)
                    seg_len = math.dist((p1.x, p1.y), (p2.x, p2.y))
                    if dist < best_dist and seg_len > 5:
                        best_dist = dist
                        best_len = seg_len

        if best_len and best_dist < 50:
            paper_mm = best_len * PT_TO_MM
            if paper_mm > 0:
                computed_scale = dim_val / paper_mm
                if 10 <= computed_scale <= 2000:
                    scales.append(round(computed_scale / 10) * 10)

    if scales:
        # Return most common scale
        from collections import Counter
        return Counter(scales).most_common(1)[0][0]
    return None
