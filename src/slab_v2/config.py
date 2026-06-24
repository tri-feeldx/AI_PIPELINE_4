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

    enable_opening_judge: bool = True
    """Call Gemini after code generates opening candidates. No result cache is
    used while this development flag is enabled."""

    opening_judge_min_confidence: float = 0.70
    """Minimum semantic-judge confidence before selected IDs replace the
    deterministic opening policy."""

    enable_slab_face_judge: bool = True
    """Judge code-generated atomic slab faces. The gross deterministic slab
    remains the fallback whenever the response or geometry validation fails."""

    slab_judge_min_confidence: float = 0.80
    slab_subtract_min_confidence: float = 0.85
    """A semantic decision may classify at 0.80, but destructive non-slab
    subtraction requires 0.85 plus deterministic negative evidence."""

    slab_max_net_area_loss_frac: float = 0.35
    """Reject a judge decision that removes more than this fraction of gross
    slab unless every removed face has explicit negative text evidence."""

    enable_floor_system_judge: bool = True
    floor_system_judge_min_confidence: float = 0.80
    floor_system_other_min_confidence: float = 0.85
    floor_system_separator_min_coverage: float = 0.45
    floor_system_stair_proximity_pt: float = 12.0
    floor_system_terminal_tolerance_mm: float = 150.0
    floor_system_terminal_tolerance_max_pt: float = 6.0
    """Floor-system separation is conservative: an OTHER region needs a
    substantial vector separator, verified stair adjacency, and independent
    semantic/system evidence. Gemini may select IDs but cannot create lines."""

    height_conflict_tolerance_mm: float = 100.0
    height_reject_residual_mm: float = 150.0
    """Datum graph thresholds for strong-source conflicts and outliers."""

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

    xcross_min_area_frac: float = 0.0002
    """Merged X-cross footprints smaller than this fraction of the content
    area are dropped — they are annotation circles (e.g. slab thickness
    '400'), grid intersection marks, or other false positives."""

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

    render_shaft_solids: bool = False
    """Keep verified shaft/core footprints as slab cuts, but do not create
    duplicate 3D shaft rings when the detected LW walls already model core."""

    keep_verified_shaft_openings: bool = True
    render_stair_solids: bool = False
    """Customer output is opening-only: stairs cut slabs but are not modelled."""

    keep_verified_stair_openings: bool = True
    penetration_min_boundary_coverage: float = 0.55
    penetration_min_confidence: float = 0.85
    penetration_axis_tolerance_mm: float = 150.0
    penetration_edge_snap_max_mm: float = 600.0
    penetration_edge_snap_min_overlap: float = 0.90
    penetration_edge_snap_min_endpoint_coverage: float = 0.65
    penetration_edge_snap_max_protected_ratio: float = 0.01
    core_opening_min_boundary_coverage: float = 0.70
    core_opening_max_wall_intersection_ratio: float = 0.01

    lw1_min_vector_coverage: float = 0.35
    """Both LW1 rails must have this target-page vector coverage before recovery."""
    extraction_max_workers: int = 10
    """Max parallel page-extraction threads."""

    wall_junction_snap_max_mm: float = 25.0
    wall_junction_verified_gap_mm: float = 1.0
    """Only close small wall endpoint quantization gaps; larger discontinuities
    remain review items rather than being bridged destructively."""

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

    steel_exclusion_radius_pt: float = 40.0
    """Buffer radius around steel text labels (CH, SH, UB, etc.) to block
    RC column detection. Independent of column_text_search_radius_pt so
    steel zones remain effective even when text search radius is reduced."""

    # ── Manual overrides ────────────────────────────────────────────────────
    manual_scale: int | None = None
    """User-supplied scale denominator (e.g. 100 for 1:100). When set,
    bypasses both text-based and dimension-based scale detection."""

    # ── Performance ────────────────────────────────────────────────────────────
    max_parallel_pages: int = 10
    """Number of pages to extract in parallel (ThreadPoolExecutor workers)."""

    speed_mode: bool = False
    """When True, disable optional Gemini judges and debug images for faster
    batch processing.  Gross slab geometry (Round 1 + deterministic assembly)
    is unaffected.  Re-run with speed_mode=False for pages needing review."""

    debug_images: bool = True
    """When False, skip non-essential debug images (step_00, step_03-11).
    Prompt images (step_01, step_02) are always generated for Gemini."""

    # ── Output ────────────────────────────────────────────────────────────────
    debug_dir: str = "debug_slab_v2"
    debug_dpi: int = 150
    save_prompt_images: bool = True
    """Write the byte-identical images sent to Gemini into the debug folder."""
