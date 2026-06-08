"""
Dataclasses for structural elements beyond slabs: columns and foundations.
"""

from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import Polygon


@dataclass
class ColumnRegion:
    id: int
    polygon: Polygon                   # column footprint in PDF coordinate space (pts)
    symbol: str                        # e.g. "SH", "PG1", "C1"
    width_mm: float = 0.0
    depth_mm: float = 0.0
    building: str = ""
    level: str = ""
    page_index: int = 0
    is_detail_only: bool = False       # True = from detail/section page, not real location
    real_polygon: Optional[Polygon] = None   # in mm after coordinate mapping
    area_m2: float = 0.0


@dataclass
class FoundationRegion:
    id: int
    polygon: Polygon                   # footing footprint in PDF coordinate space (pts)
    symbol: str                        # e.g. "PF1", "PC-500"
    fdn_type: str = "pad"              # "pad" | "pile_cap" | "raft" | "strip"
    width_mm: float = 0.0
    depth_mm: float = 0.0
    depth_below_gl_mm: float = 0.0    # depth below ground level (positive = downward)
    page_index: int = 0
    real_polygon: Optional[Polygon] = None
    area_m2: float = 0.0
