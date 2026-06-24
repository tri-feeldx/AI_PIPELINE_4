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
class SlabFaceCandidate:
    """One code-generated atomic face offered to the semantic slab judge."""
    id: str
    polygon: Polygon
    area_pt2: float
    source: str = "polygonize"
    fill_style: dict = field(default_factory=dict)
    boundary_style_ids: list = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    nearby_text: list = field(default_factory=list)
    contained_columns: int = 0
    contained_walls: int = 0
    intersects_openings: list = field(default_factory=list)
    positive_evidence: list = field(default_factory=list)
    negative_evidence: list = field(default_factory=list)
    deterministic_score: float = 0.0


@dataclass
class SlabResolution:
    """Validated semantic decision over slab face candidates."""
    selected_slab_ids: list = field(default_factory=list)
    appendage_ids: list = field(default_factory=list)
    opening_ids: list = field(default_factory=list)
    non_slab_ids: list = field(default_factory=list)
    review_ids: list = field(default_factory=list)
    gross_geometry: object = None
    net_geometry: object = None
    confidence: float = 0.0
    status: str = "deterministic_fallback"
    reason: str = ""
    warnings: list = field(default_factory=list)


@dataclass
class FloorSystemSemanticProfile:
    """Document/page evidence used to distinguish concrete from floor extent."""
    concrete_slab_terms: list = field(default_factory=list)
    floor_extent_terms: list = field(default_factory=list)
    steel_floor_terms: list = field(default_factory=list)
    opening_terms: list = field(default_factory=list)
    fill_rules: list = field(default_factory=list)
    boundary_rules: list = field(default_factory=list)
    symbol_families: list = field(default_factory=list)
    confidence: float = 0.0
    warnings: list = field(default_factory=list)


@dataclass
class FloorSystemCandidate:
    """A code-generated partition of the gross floor extent."""
    id: str
    polygon: Polygon
    source_face_ids: list = field(default_factory=list)
    fill_role: str = "UNKNOWN"
    boundary_styles: list = field(default_factory=list)
    nearby_text: list = field(default_factory=list)
    steel_symbols: list = field(default_factory=list)
    concrete_symbols: list = field(default_factory=list)
    adjacent_openings: list = field(default_factory=list)
    touches_outer_edge: bool = False
    separator_evidence: list = field(default_factory=list)
    positive_pt_evidence: list = field(default_factory=list)
    negative_pt_evidence: list = field(default_factory=list)
    deterministic_score: float = 0.0
    separator_segment: object = None
    terminal_cap_segment: object = None
    terminal_source: str = ""
    terminal_alignment_error_pt: Optional[float] = None
    extension_direction: str = ""
    bounded_cut_area_pt2: float = 0.0
    rejected_extension_area_pt2: float = 0.0
    rejected_extension_geometry: object = None
    cut_status: str = "candidate"


@dataclass
class FloorSystemResolution:
    """Validated partition of floor extent into concrete and other systems."""
    pt_slab_ids: list = field(default_factory=list)
    other_floor_ids: list = field(default_factory=list)
    opening_ids: list = field(default_factory=list)
    non_floor_ids: list = field(default_factory=list)
    unknown_ids: list = field(default_factory=list)
    pt_gross_geometry: object = None
    other_floor_geometry: object = None
    pt_net_geometry: object = None
    status: str = "review"
    confidence: float = 0.0
    warnings: list = field(default_factory=list)
    reason: str = ""


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
class ResolvedPenetration:
    """Auditable, vector-derived opening selected from semantic seeds."""
    id: str
    kind: str
    polygon: Polygon
    source_candidate_ids: list = field(default_factory=list)
    contained_seed_ids: list = field(default_factory=list)
    boundary_coverage: float = 0.0
    confidence: float = 0.0
    status: str = "review"
    warnings: list = field(default_factory=list)
    geometry_audit: dict = field(default_factory=dict)


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
class WallFootprint:
    """A wall cross-section extracted from WALL-tagged style classes."""
    label: str                              # "W1", "SW1", or "WALL_1" if unlabeled
    polygon: Polygon                        # PDF-point coordinates
    w_mm: float = 0.0                       # short side (thickness)
    l_mm: float = 0.0                       # long side (length)
    wall_type: str = "wall"                 # "wall" | "retaining_wall"
    centerline: list = field(default_factory=list)  # [(x, y)] in PDF points
    source: str = "plan_shape"
    confidence: float = 0.0
    profile_id: str = ""
    grid_start: str = ""
    grid_end: str = ""
    mapping_status: str = "review"


@dataclass
class WallElevationProfile:
    """Wall elevation geometry in normalized station and real Z millimetres."""
    profile_id: str
    wall_symbol: str
    source_page: int
    source_view_bbox: tuple
    panels: list = field(default_factory=list)
    from_level: str = ""
    to_level: str = ""
    grid_start: str = ""
    grid_end: str = ""
    grid_sequence: list = field(default_factory=list)
    grid_stations: dict = field(default_factory=dict)
    scale_ratio: float = 0.0
    scale_status: str = ""
    confidence: float = 0.0
    status: str = "review"
    warnings: list = field(default_factory=list)


@dataclass
class ColumnFootprint:
    """A column cross-section detected shape-first from the vector data."""
    symbol: str                             # census mark ("C1") or "C?" when ambiguous
    polygon: Polygon                        # exact PDF-point coordinates
    w_mm: float = 0.0                       # measured size at the page scale
    d_mm: float = 0.0
    labeled: bool = False                   # True when a text mark confirmed it
    candidate_id: str = ""
    source: str = "shape"
    confidence: float = 0.0
    grid_id: str = ""


@dataclass
class ColumnType:
    """One row of the column schedule (from the Gemini document census)."""
    symbol: str
    width_mm: float = 0.0
    depth_mm: float = 0.0
    count_total: int = 0
    material: str = "UNKNOWN"              # RC | STEEL | UNKNOWN


@dataclass
class WallType:
    """One row of the wall schedule (from the Gemini document census)."""
    symbol: str
    thickness_mm: float = 0.0
    height_mm: float = 0.0
    material: str = ""
    wall_category: str = "wall"
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
    walls: dict = field(default_factory=dict)       # symbol -> count
    total_walls: int = 0


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
    column_census_report: dict = field(default_factory=dict)
    # [{"building": str, "level_id": str, "counts": {symbol: n}}]
    stair_detail_pages: list = field(default_factory=list)      # parked for later
    lift_detail_pages: list = field(default_factory=list)
    foundation_detail_pages: list = field(default_factory=list)
    foundation_types: dict = field(default_factory=dict)
    footing_plan_pages: list = field(default_factory=list)      # 0-based
    wall_types: dict = field(default_factory=dict)             # symbol -> WallType
    wall_schedule_pages: list = field(default_factory=list)    # 0-based
    wall_elevation_pages: list = field(default_factory=list)   # 0-based
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
class HeightEvidence:
    """A traceable absolute datum or relative level-height constraint."""
    id: str
    building: str
    from_level: Optional[str]
    to_level: str
    evidence_type: str
    value_mm: float
    page_index: int = -1
    bbox: Optional[tuple] = None
    source_text: str = ""
    extraction_method: str = ""
    confidence: float = 0.0
    is_absolute_datum: bool = False
    viewport_id: str = ""
    source_fingerprint: str = ""
    independence_group: str = ""
    datum_line_from: Optional[tuple] = None
    datum_line_to: Optional[tuple] = None
    scale_ratio: Optional[float] = None
    scale_status: str = ""
    scale_evidence_ids: list = field(default_factory=list)
    duplicate_of: str = ""


@dataclass
class LevelDatum:
    """Solved level elevation with provenance and engineering readiness."""
    building: str
    level_id: str
    ffl_mm: Optional[float]
    storey_height_mm: Optional[float]
    status: str = "default_unsafe"
    confidence: float = 0.0
    supporting_evidence_ids: list = field(default_factory=list)
    rejected_evidence_ids: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


@dataclass
class ModelReadinessReport:
    slab_status: str = "review"
    height_status: str = "default_unsafe"
    opening_status: str = "review"
    wall_status: str = "review"
    column_status: str = "review"
    wall_junction_status: str = "review"
    shaft_render_status: str = "review"
    stair_render_status: str = "review"
    model_status: str = "debug"
    reasons: list = field(default_factory=list)


@dataclass
class HeightReconciliation:
    """Multi-source FFL reconciliation result."""
    floors: list = field(default_factory=list)      # list[FloorHeight]
    method: str = ""
    warnings: list = field(default_factory=list)
    debug_log: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    level_datums: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    source_planner: dict = field(default_factory=dict)
    consensus_report: list = field(default_factory=list)
    candidate_pages: list = field(default_factory=list)

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
    resolved_openings: list = field(default_factory=list)  # list[ElementFootprint]
    resolved_penetrations: list = field(default_factory=list)
    render_elements: list = field(default_factory=list)  # 3D elements, separate from cuts
    opening_report: dict = field(default_factory=dict)
    walls: list = field(default_factory=list)           # list[WallFootprint]
    wall_detection_report: dict = field(default_factory=dict)
    wall_profiles: dict = field(default_factory=dict)
    wall_readiness: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)         # list[ColumnFootprint]
    column_candidates: list = field(default_factory=list)
    column_detection_report: dict = field(default_factory=dict)
    column_readiness: dict = field(default_factory=dict)
    opening_candidates: list = field(default_factory=list)
    opening_judgement: dict = field(default_factory=dict)
    slab_candidates: list = field(default_factory=list)
    slab_resolution: Optional[SlabResolution] = None
    slab_readiness: dict = field(default_factory=dict)
    floor_system_candidates: list = field(default_factory=list)
    floor_system_profile: Optional[FloorSystemSemanticProfile] = None
    floor_system_resolution: Optional[FloorSystemResolution] = None
    other_floor_systems: list = field(default_factory=list)
    floor_system_readiness: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    scale: Optional[float] = None           # nominal int or precise float
    attempts: int = 0
    gemini_calls: int = 0
    debug_dir: str = ""
    timings: dict = field(default_factory=dict)
