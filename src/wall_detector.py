"""
Semantic wall extraction from legend-guided boundary evidence.

V1 exports only high-confidence wall labels/semantic wall objects, not every
generic cyan geometry line. This keeps the model conservative and reviewable.
"""

from __future__ import annotations

from typing import Any

import fitz
from shapely.ops import unary_union

from src.coordinate_mapper import transform_structural_elements
from src.structural_boundary_detector import detect_structural_boundary_objects
from src.structural_elements import WallRegion


def _is_exportable_wall_source(source: str) -> bool:
    source = str(source or "")
    return source == "wall_label" or source.startswith("legend_semantic")


def _dedupe_walls(walls: list[WallRegion]) -> list[WallRegion]:
    kept: list[WallRegion] = []
    for wall in walls:
        poly = wall.polygon
        if poly is None or poly.is_empty:
            continue
        duplicate = False
        for existing in kept:
            if existing.polygon is None or existing.polygon.is_empty:
                continue
            inter = poly.intersection(existing.polygon).area
            denom = max(min(poly.area, existing.polygon.area), 1.0)
            if inter / denom > 0.82 and wall.label == existing.label:
                duplicate = True
                break
        if not duplicate:
            kept.append(wall)
    return kept


def detect_walls_on_page(
    page: fitz.Page,
    drawings: list[dict],
    scale: int,
    page_index: int,
    legend_semantics: dict | None = None,
    text_blocks: list[dict] | None = None,
    building: str = "",
    level: str = "",
) -> tuple[list[WallRegion], Any]:
    structural = detect_structural_boundary_objects(
        page,
        drawings,
        text_blocks=text_blocks,
        auto_cut_voids=False,
        legend_semantics=legend_semantics,
    )
    walls: list[WallRegion] = []
    for obj in structural.walls:
        if not _is_exportable_wall_source(obj.source):
            continue
        if not obj.label:
            continue
        walls.append(WallRegion(
            id=len(walls),
            polygon=obj.polygon,
            label=obj.label,
            wall_type="retaining_wall" if "RETAINING" in obj.label.upper() else "wall",
            page_index=page_index,
            building=building,
            level=level,
            detection_confidence=obj.confidence,
            source=obj.source,
        ))
    walls = _dedupe_walls(walls)
    for i, wall in enumerate(walls):
        wall.id = i
    return transform_structural_elements(walls, page, scale), structural


def detect_walls_for_pages(
    pdf_path: str,
    page_indices: list[int],
    scale: int,
    legend_semantics: dict | None = None,
) -> tuple[list[WallRegion], dict[int, Any]]:
    import fitz as _fitz

    all_walls: list[WallRegion] = []
    structural_by_page: dict[int, Any] = {}
    doc = _fitz.open(pdf_path)
    try:
        for page_index in page_indices:
            if page_index < 0 or page_index >= doc.page_count:
                continue
            page = doc[page_index]
            walls, structural = detect_walls_on_page(
                page,
                page.get_drawings(),
                scale,
                page_index,
                legend_semantics=legend_semantics,
            )
            structural_by_page[page_index] = structural
            for wall in walls:
                wall.id = len(all_walls)
                all_walls.append(wall)
    finally:
        doc.close()
    return all_walls, structural_by_page
