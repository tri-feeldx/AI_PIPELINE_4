"""
SlabV2Config — every tunable of the slab_v2 pipeline in one place.

Units: "pt" = PDF points (1 pt = 1/72 inch = 0.3528 mm on paper).
At drawing scale 1:100, 1 pt on paper = 35.28 mm real-world.
"""

from dataclasses import dataclass, field


@dataclass
class SlabV2Config:
    # ── Stage A: vector extraction ────────────────────────────────────────────
    bezier_tol_pt: float = 0.2
    """Max sagitta deviation when flattening cubic beziers (De Casteljau).
    0.2 pt = 0.07 mm paper = 7 mm real at 1:100 — far below line width."""

    frame_area_frac: float = 0.97
    """Style classes made only of rectangles covering more than this fraction
    of the page are tagged FRAME (sheet border) and excluded from faces."""

    # ── Stage B: planarization ────────────────────────────────────────────────
    snap_grid_pt: float = 0.05
    """GEOS snap-rounding grid. Coordinates move at most half a cell
    (0.025 pt = 0.0088 mm paper). Resolves micro-gaps without distortion."""

    gap_ladder_pt: tuple = (0.25, 0.5, 1.0)
    """Dangle-endpoint snap distances tried in order when polygonize yields
    too few faces. Endpoints move ONTO existing nodes, never to midpoints."""

    min_face_area_frac: float = 0.005
    """A face counts as 'significant' if its area exceeds this fraction of
    the drawing content area (used to judge whether polygonize succeeded)."""

    min_faces: int = 1
    """Minimum number of significant faces before the gap ladder kicks in."""

    max_polygonize_segments: int = 200_000
    """Safety cap on segment count fed to shapely (A1 sheets stay well under)."""

    # ── Stage C: AI selection ─────────────────────────────────────────────────
    gemini_model: str = ""
    """Override Gemini model name; empty = GEMINI_MODEL env or gemini-2.5-flash."""

    max_classes_in_prompt: int = 25
    """Style classes shown individually in Round 1. The rest are prefiltered
    (hatch/frame fingerprints) and summarized in one line."""

    max_faces_in_prompt: int = 30
    """Faces labeled in the Round-2 overview image. Beyond this, quadrant
    zoom tiles are added (same global ids)."""

    min_class_coverage_frac: float = 0.30
    """Round-1 validation: elected slab-edge classes must jointly cover at
    least this fraction of the content rect bbox."""

    min_area_frac: float = 0.05
    max_area_frac: float = 0.90
    """Round-2 validation: selected slab area as fraction of content area."""

    max_ai_retries: int = 3
    """Retries per election round (with validation feedback appended)."""

    max_total_calls: int = 8
    """Hard budget of Gemini calls per page."""

    prompt_dpi: int = 150
    """DPI for images sent to Gemini and for debug renders."""

    # ── deterministic assembly (replaces AI Round 2) ──────────────────────────
    min_keep_face_frac: float = 0.001
    """Faces above this fraction of the content area are unioned into the
    gross slab ("better too much than too little" — architect's rule).
    Below = noise slivers."""

    min_component_frac: float = 0.02
    """After unioning faces, disconnected components smaller than this
    fraction of the largest component are dropped (stray column boxes,
    symbols) while detached slab pads/wings are kept ("đừng thiếu")."""

    element_max_area_frac: float = 0.15
    """An element footprint (stair/lift/shaft) larger than this fraction of
    the content area is considered a mis-anchor and skipped with a warning."""

    xcross_max_area_frac: float = 0.04
    """X-cross opening candidates (rect with corner-to-corner diagonals)
    must be at most this fraction of the content area — shafts are small;
    anything bigger is a drawing region, never an opening."""

    element_text_radius_pt: float = 80.0
    """Max distance from a STAIR/LIFT/... label to the X-cross face it
    names (labels usually sit outside the shaft with a leader line)."""

    slab_thickness_mm: float = 200.0
    """Extrusion depth for exported slab faces (same default as v1)."""

    element_height_mm: float = 3000.0
    """Element volume height for single-page export (no storey above)."""

    # ── multi-storey building export ──────────────────────────────────────────
    default_storey_height_mm: float = 3000.0
    """Storey height assumed when no FFL above exists (top floor, or pages
    without FFL annotations stacked in page order)."""

    shaft_wall_thickness_mm: float = 150.0
    """Wall thickness of the LIFT/SHAFT/DUCT ring volume. Footprints too
    narrow for the inward buffer fall back to a solid volume."""

    stair_max_riser_mm: float = 175.0
    """Riser count = ceil(storey_height / this); actual riser = height/n."""

    stair_min_going_mm: float = 240.0
    """Minimum tread going. If the footprint run is too short for the riser
    count, steps are reduced to honour this (with a warning)."""

    shaft_pair_min_iou: float = 0.30
    """Min footprint IoU (in mm space) for same-type elements on consecutive
    storeys to count as one continuous shaft; unpaired ones get a warning."""

    # ── Stage D: verification (optional, informational only) ─────────────────
    dim_assoc_radius_pt: float = 25.0
    """Max distance from dimension-text center to its dimension line."""

    dim_parallel_tol_deg: float = 5.0
    """Dimension line must be parallel to the text direction within this."""

    dim_rel_tol: float = 0.015
    """Relative tolerance when comparing polygon edge length to a dimension
    value (covers snap grid, tick width, extension-line offsets)."""

    min_scale_consistency: float = 0.6
    """Fraction of associated dimensions whose implied scale must agree with
    the page scale for verification to pass."""

    min_edge_matches: int = 2
    """Minimum matched edges (or one extent match) for a confident pass."""

    precise_scale_min_dims: int = 4
    """Distinct dimension values that must agree before the continuous
    (median) scale replaces the nominal one for the mm conversion. Catches
    non-integer viewport scales (A1 sheet exported at A3 = 1:141.42 while
    the text still says 1:100)."""

    precise_scale_max_spread: float = 0.01
    """Max relative spread (max-min)/median of the consensus group's implied
    scales for the precise scale to be trusted."""

    precise_scale_min_len_pt: float = 60.0
    """Only dims whose dim line is at least this long feed the precise
    scale — short detail dims carry tick noise and 5mm label rounding."""

    scale_sanity_min: int = 5
    scale_sanity_max: int = 1250
    """Dimension-measured scales outside this range are coincidental
    consensus (pages with few real dims) and never override the text."""

    # ── columns (shape-first, schedule as size filter) ───────────────────────
    column_size_tol_mm: float = 25.0
    """A candidate rectangle matches a schedule column type when both sides
    are within this of the scheduled width/depth (w<->d swap allowed)."""

    column_label_radius_pt: float = 60.0
    """Max distance from a column-mark text (C1, SH...) to the rectangle it
    names; labels only ASSIGN the symbol, never create footprints."""

    column_min_repeat: int = 4
    """Without a schedule census: rectangles of one repeated size become an
    anonymous column type when at least this many occurrences exist."""

    column_max_side_mm: float = 1500.0
    """Rectangles with a side beyond this are never column candidates."""

    column_text_search_radius_pt: float = 40.0
    """columns_v2: radius around a column text label to search for matching
    rectangular shapes. Smaller = more precise, larger = tolerates offset
    labels with leader lines."""

    # ── Manual overrides ────────────────────────────────────────────────────
    manual_scale: int | None = None
    """User-supplied scale denominator (e.g. 100 for 1:100). When set,
    bypasses both text-based and dimension-based scale detection."""

    # ── Output ────────────────────────────────────────────────────────────────
    debug_dir: str = "debug_slab_v2"
    debug_dpi: int = 150
    save_prompt_images: bool = True
    """Write the byte-identical images sent to Gemini into the debug folder."""
