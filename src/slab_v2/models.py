"""
Data model for the slab_v2 pipeline.

All geometry is stored in PDF points (fitz top-left origin) until the final
mm conversion in pipeline.py. Coordinates are never synthesized — every vertex
traces back to the PDF vector data (or a GEOS snap-round of it, ≤0.025 pt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import Polygon


@dataclass(frozen=True)
class StyleKey:
    """Hashable identity of a line/fill style class."""
    stroke: Optional[tuple]   # RGB rounded to 3 dp, None when no stroke
    fill: Optional[tuple]     # RGB rounded to 3 dp, None when no fill
    width: float              # stroke width, round(w or 0.0, 2)
    dashes: str               # normalized fitz dash string, "" = solid
    even_odd: bool

    def describe(self) -> str:
        parts = []
        if self.stroke is not None:
            parts.append(f"stroke RGB{self.stroke}")
        if self.fill is not None:
            parts.append(f"fill RGB{self.fill}")
        parts.append(f"width {self.width}pt")
        parts.append("dashed " + self.dashes if self.dashes else "solid")
        return ", ".join(parts)


@dataclass
class StyleClass:
    """A group of paths sharing one StyleKey. ids sorted by total length desc."""
    id: int
    key: StyleKey
    n_paths: int = 0
    n_segments: int = 0
    total_length_pt: float = 0.0
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)     # (x0, y0, x1, y1) union
    median_seg_len_pt: float = 0.0
    role: str = "UNKNOWN"   # SLAB_EDGE|WALL|COLUMN|GRID|DIMENSION|HATCH|VOID_EDGE|ANNOTATION|FRAME|OTHER
    role_confidence: float = 0.0
    prefiltered: bool = False              # excluded from Round-1 prompt by fingerprint


@dataclass
class VectorPath:
    """One fitz drawing, flattened to straight segments."""
    id: int
    style_id: int
    segments: list                          # [((x1,y1),(x2,y2)), ...] exact PDF pts
    is_closed: bool
    is_filled: bool
    seqno: int                              # paint order from fitz
    fill_polygon: Optional[Polygon] = None  # for filled closed paths
    outside_content: bool = False           # outside the drawing content rect
    layer: str = ""


@dataclass
class Face:
    """An atomic closed region produced by polygonize."""
    id: int
    polygon: Polygon                        # exact PDF-point coordinates
    area_pt2: float = 0.0
    style_ids: frozenset = frozenset()      # classes whose segments border this face
    parent_id: Optional[int] = None         # smallest containing face
    depth: int = 0                          # nesting depth
    label_anchor: tuple = (0.0, 0.0)        # representative_point for ID rendering
    source: str = "polygonize"              # polygonize | fill


@dataclass
class FaceGraph:
    faces: list                             # list[Face]
    dangles: list = field(default_factory=list)    # LineStrings that failed to close
    cut_edges: list = field(default_factory=list)
    snap_used_pt: float = 0.0               # gap-ladder rung that succeeded (0 = none needed)
    source_style_ids: tuple = ()            # classes polygonized in this pass
    n_segments_in: int = 0


@dataclass
class ClassElection:
    """Stage C Round 1 — which style classes bound the slab."""
    slab_edge_classes: list
    supporting_classes: list                # classes allowed to close gaps (walls etc.)
    roles: dict                             # class_id -> role string
    reasoning: str = ""
    raw_response: str = ""
    warning: str = ""                       # e.g. low bbox coverage


@dataclass
class FaceSelection:
    """Stage C Round 2 — which faces are slab / void."""
    slabs: list                             # [{"face_ids":[...], "void_face_ids":[...], "label":str}]
    confidence: float = 0.0
    reasoning: str = ""
    raw_response: str = ""


@dataclass
class ElementFootprint:
    """A stair/lift/shaft/void element anchored by drawing text.

    The footprint polygon is the deepest face of the all-classes face graph
    containing the anchor — exact vector coordinates. Openings are cut from
    the slab only at Ruby-export time (BIM-style host/opening), never in 2D
    extraction.
    """
    type: str                               # STAIR | LIFT | SHAFT | VOID | DUCT
    polygon: Polygon
    label: str                              # the anchor text as drawn
    anchor_bbox: tuple                      # text bbox in PDF pts
    area_pt2: float = 0.0


@dataclass
class DimensionAnnotation:
    text: str
    value_mm: float
    bbox: tuple                             # text bbox in PDF pts
    rotation_deg: float = 0.0
    dim_line: Optional[tuple] = None        # ((x1,y1),(x2,y2)) associated segment
    measured_pt: Optional[float] = None     # its length


@dataclass
class VerificationReport:
    passed: bool = False
    scale_used: int = 0
    scale_precise: float = 0.0              # continuous median scale; set only
                                            # when the consensus is strong
                                            # (>= precise_scale_min_dims distinct
                                            # values, spread <= max_spread)
    scale_consistency: float = 0.0          # fraction of dims agreeing with scale
    n_dims_associated: int = 0
    edge_matches: list = field(default_factory=list)   # {edge, dim_value_mm, edge_mm, rel_err}
    extent_check: dict = field(default_factory=dict)
    area_fraction_of_content: float = 0.0
    failures: list = field(default_factory=list)       # feedback strings for retry


@dataclass
class ColumnFootprint:
    """A column cross-section detected shape-first from the vector data."""
    symbol: str                             # census mark ("C1") or "C?" when ambiguous
    polygon: Polygon                        # exact PDF-point coordinates
    w_mm: float = 0.0                       # measured size at the page scale
    d_mm: float = 0.0
    labeled: bool = False                   # True when a text mark confirmed it


@dataclass
class ColumnType:
    """One row of the column schedule (from the Gemini document census)."""
    symbol: str
    width_mm: float = 0.0
    depth_mm: float = 0.0
    count_total: int = 0


@dataclass
class FloorInfo:
    level_name: str = ""
    level_id: str = ""
    ffl_m: Optional[float] = None
    pages: list = field(default_factory=list)       # 0-based page indices
    titles: list = field(default_factory=list)
    storey_height_mm: float = 0.0
    columns: dict = field(default_factory=dict)     # symbol -> count
    total_columns: int = 0


@dataclass
class BuildingInfo:
    name: str = ""
    floors: list = field(default_factory=list)      # list[FloorInfo]


@dataclass
class DocAnalysis:
    """One Gemini text-only call over the whole document (v1 step-2 strategy)."""
    buildings: list = field(default_factory=list)   # list[BuildingInfo]
    column_types: dict = field(default_factory=dict)  # symbol -> ColumnType
    column_schedule_pages: list = field(default_factory=list)   # 0-based
    columns_per_floor: list = field(default_factory=list)
    # [{"building": str, "level_id": str, "counts": {symbol: n}}]
    stair_detail_pages: list = field(default_factory=list)      # parked for later
    lift_detail_pages: list = field(default_factory=list)
    foundation_detail_pages: list = field(default_factory=list)
    foundation_types: dict = field(default_factory=dict)
    footing_plan_pages: list = field(default_factory=list)      # 0-based
    orphan_columns: dict = field(default_factory=dict)
    detail_pages: list = field(default_factory=list)            # 0-based
    confidence: str = ""
    notes: str = ""
    warnings: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass
class FloorHeight:
    """Reconciled height data for one floor level."""
    level_id: str
    building: str
    ffl_m: float
    storey_height_mm: float = 3000.0
    sources: dict = field(default_factory=dict)
    confidence: str = "low"


@dataclass
class HeightReconciliation:
    """Multi-source FFL reconciliation result."""
    floors: list = field(default_factory=list)      # list[FloorHeight]
    method: str = ""
    warnings: list = field(default_factory=list)
    debug_log: list = field(default_factory=list)

    def get_ffl(self, building: str, level_id: str) -> Optional[float]:
        for f in self.floors:
            if f.building == building and f.level_id == level_id:
                return f.ffl_m
        return None

    def get_height(self, building: str, level_id: str) -> Optional[float]:
        for f in self.floors:
            if f.building == building and f.level_id == level_id:
                return f.storey_height_mm
        return None


@dataclass
class SlabV2Result:
    page_index: int                         # 0-based
    status: str = "OK"                      # OK | VERIFY_FAILED | NO_FACES | AI_ERROR | NO_AI
    slabs: list = field(default_factory=list)
    # each: {"label": str, "polygon_pdf": Polygon, "polygon_mm": Polygon|None,
    #        "area_m2": float|None, "void_count": int}
    style_classes: list = field(default_factory=list)
    election: Optional[ClassElection] = None
    selection: Optional[FaceSelection] = None
    verification: Optional[VerificationReport] = None
    elements: list = field(default_factory=list)        # list[ElementFootprint]
    columns: list = field(default_factory=list)         # list[ColumnFootprint]
    warnings: list = field(default_factory=list)
    scale: Optional[float] = None           # nominal int or precise float
    attempts: int = 0
    gemini_calls: int = 0
    debug_dir: str = ""
    timings: dict = field(default_factory=dict)
