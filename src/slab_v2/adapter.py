"""
Adapter: SlabV2Result -> list[SlabRegion] so the existing app.py review/
export steps consume slab_v2 output without modification.

The v1 path remains the default; this adapter is only used when the
SLAB_V2 toggle is on.
"""

from __future__ import annotations

import fitz

from src.slab_extractor import SlabRegion
from src.slab_v2.models import SlabV2Result


def to_slab_regions(
    result: SlabV2Result,
    page: fitz.Page,
    ffl_values: list | None = None,
) -> list[SlabRegion]:
    """Convert slab_v2 output to v1 SlabRegion objects (PDF-space polygons;
    coordinate_mapper.transform_all_slabs handles the mm conversion exactly
    as for v1 regions)."""
    regions: list[SlabRegion] = []
    ffl_m = None
    if ffl_values:
        try:
            ffl_m = float(ffl_values[0])
        except (TypeError, ValueError, IndexError):
            ffl_m = None

    for s in result.slabs:
        geom = s["polygon_pdf"]
        geoms = getattr(geom, "geoms", [geom])
        for j, g in enumerate(geoms):
            label = s["label"] or f"SLAB_V2_{len(regions) + 1}"
            if j > 0:
                label = f"{label}_{j + 1}"
            regions.append(SlabRegion(
                id=len(regions),
                polygon=g,
                label=label,
                ffl_m=ffl_m,
                area_pdf=g.area,
                page_index=result.page_index,
                source="slab_v2",
            ))
    return regions
