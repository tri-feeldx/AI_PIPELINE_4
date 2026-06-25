# AI Pipeline SketchUp — System Documentation

> **Purpose**: This document describes the full PDF-to-SketchUp 3D pipeline so that any
> developer joining the project can understand the architecture, data flow, every module,
> and how to debug / operate the system.
>
> Last updated: 2026-06-24

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture-tổng-quan)
2. [Data Flow](#2-data-flow-luồng-dữ-liệu)
3. [Module Reference](#3-module-reference-chi-tiết-module)
4. [Configuration Reference](#4-configuration-reference-cấu-hình)
5. [Debug & Troubleshooting](#5-debug--troubleshooting)
6. [Key Design Decisions](#6-key-design-decisions-quyết-định-thiết-kế)
7. [Testing](#7-testing-kiểm-thử)
8. [Operational Guide](#8-operational-guide-vận-hành)

---

## 1. Overview & Architecture (Tổng quan)

### What the system does

This system takes **structural engineering PDF drawings** (slab plans, column schedules,
wall schedules, elevations) and produces a **SketchUp 3D model** (`.rb` Ruby script)
containing:

- Slab polygons with correct openings (stair, lift, shaft, void)
- Column footprints (RC and steel)
- Wall footprints (boundary + core/LW walls)
- Multi-storey stacking at correct FFL (Finished Floor Level) elevations

### Core Principle — Vector-First

```
┌─────────────────────────────────────────────────────────────┐
│  ALL geometry comes from PDF vector paths.                  │
│  AI (Gemini) only SELECTS and CLASSIFIES — never creates    │
│  coordinates. If Gemini fails, the deterministic fallback   │
│  still produces geometry.                                   │
└─────────────────────────────────────────────────────────────┘
```

### System Diagram

```
 ┌──────────┐
 │  PDF     │
 │ (upload) │
 └────┬─────┘
      │
      ▼
 ┌──────────────────── Phase 2: Document Analysis ────────────────────┐
 │                                                                     │
 │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
 │  │ Floor Analyzer   │  │ Column Census     │  │ Wall Census      │  │
 │  │ (buildings,      │  │ (symbols,         │  │ (symbols,        │  │
 │  │  floors, FFLs)   │  │  dimensions)      │  │  thicknesses)    │  │
 │  └────────┬─────────┘  └────────┬──────────┘  └────────┬─────────┘  │
 │           └──────────────┬──────┘───────────────────────┘            │
 │                          ▼                                          │
 │                   DocAnalysis                                       │
 │           (buildings, schedules, census)                             │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
 ┌──────────────── Phase 2.5: Site Placement (multi-building) ────────┐
 │  Coordinate audit — relative building positions (dx, dy)            │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
 ┌──────────────── Phase 3: Extraction & Export ──────────────────────┐
 │                                                                     │
 │  1. Height Reconciliation (multi-source datum graph → FFLs)         │
 │  2. Wall Source Registry (elevation pages → wall profiles)          │
 │  3. Per-Page Extraction (ThreadPoolExecutor, min 10 workers)        │
 │     ┌──────────────────────────────────────────────────────┐        │
 │     │ Stages A→M per page (see Section 2 below)            │        │
 │     │ → SlabV2Result (slab + columns + walls + openings)    │        │
 │     └──────────────────────────────────────────────────────┘        │
 │  4. Export Readiness Assessment                                     │
 │  5. Ruby Generation → .rb file download                             │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘
```

### UI Phases (Streamlit — `app_v2.py`)

| Phase | Name | What happens |
|-------|------|-------------|
| 1 | Upload | PDF file selection, page count validation |
| 2 | Analyzing | 3 parallel Gemini calls (floor / column / wall census) |
| 2.5 | Site Placement | Multi-building coordinate audit (skipped for single building) |
| 3 | Extract & Export | Parallel page extraction → height reconciliation → Ruby generation → download |

---

## 2. Data Flow (Luồng dữ liệu)

### Per-Page Pipeline Stages (A → M)

Each floor page runs through these stages in order inside `pipeline.py`:

| Stage | Module | What it does | Input | Output |
|-------|--------|-------------|-------|--------|
| **A** | `vector_extract.py` | Extract PDF vector paths, group by style (color/width/dash) | `fitz.Page` | `VectorPath[]`, `StyleClass[]` |
| **B** | `planarize.py` | Snap-round segments, polygonize into atomic faces | paths + class IDs | `FaceGraph` (all faces) |
| **C1** | `ai_select.py` | Gemini elects which style classes form the slab edge | page image + classes | `ClassElection` |
| **C2** | `planarize.py` | Build face graph from elected classes only | elected class IDs | `FaceGraph` (selected) |
| **C3** | `planarize.py` | Union faces into gross slab polygon | selected faces | `Polygon` (gross slab) |
| **retry** | — | If slab < 10% of content area → fallback to all classes, then re-elect (max 3 attempts) | — | — |
| **D** | `elements.py` | Find X-cross symbols → STAIR / LIFT / SHAFT / VOID / DUCT | page vectors | `ElementFootprint[]` |
| **E** | `verify.py` | Parse dimension annotations (informational) | page text + lines | `DimensionAnnotation[]` |
| **F** | — | Finalize scale: manual > dimension-consensus > text-detected > fallback | — | `final_scale: int` |
| **G** | `columns_v2.py` | Text-anchor column detection + schedule size matching | page + `ColumnType{}` | `ColumnFootprint[]` |
| **H** | `walls_v2.py` | Census-aware wall detection (only detect what Gemini census says) | page + `WallType{}` | `WallFootprint[]` |
| **I** | `wall_profile_resolver.py` | Assign wall topology from elevation/section evidence | walls + registry | `WallFootprint[]` (enriched) |
| **J** | `core_wall_topology.py` | Assign LW core-wall identities | page + walls | `WallFootprint[]` (LW labeled) |
| **K** | `wall_junction_resolver.py` | Snap small orthogonal wall endpoint gaps (< 25mm) | LW walls | `WallFootprint[]` (snapped) |
| **L** | `opening_resolver.py` | Validate stair/lift/core/shaft penetration candidates + optional AI judge | elements + walls + slab | `OpeningResolution` |
| **M** | `debug_render.py` | Generate step-by-step debug PNG images | all geometry | PNG files |

### Scale Determination (Stage F — Priority Order)

1. `cfg.manual_scale` — user override (highest priority)
2. Dimension-measured scale — if >= 4 consistent dimension values, spread <= 1%
3. Text-detected scale — regex match for "1:N" on the page
4. Fallback — warning issued

### Multi-Storey Assembly (Export)

After all pages are extracted:

1. `height_reconcile.py` solves FFLs from dimension annotations, text datums, and elevation pages
2. `export_ruby.py` → `generate_building_ruby()` stacks floors at FFL elevations
3. Same-type elements on consecutive storeys are paired by IoU (≥ 0.30) for shaft continuity
4. Slab face at Z=FFL, extruded downward by `slab_thickness_mm` (default 200mm)

---

## 3. Module Reference (Chi tiết module)

### 3.1 `app_v2.py` — UI Orchestrator

Entry point. Streamlit application managing the 4-phase workflow.

**Key session state**: `pdf_path`, `doc_analysis`, `storeys`, `ruby_bytes`, `height_result`, `wall_source_registry`, `model_readiness`

**Sidebar config**: slab thickness, storey height, column text radius, manual scale override, Gemini model override

**Phase 3 worker pool**: `max(10, cfg.extraction_max_workers)` threads via `ThreadPoolExecutor`

---

### 3.2 `src/slab_v2/pipeline.py` — Stage Orchestrator

Runs stages A→M for a single page. Contains retry logic (up to 3 attempts if slab coverage is too low).

```
extract_slabs_v2(
    pdf_path, page_index, cfg, use_ai, scale,
    column_types, columns_per_floor,
    wall_types, walls_per_floor,
    wall_source_registry
) → SlabV2Result
```

**Retry logic**: elected classes → assemble slab → if < 10% content area → fallback to all-class graph → if still < 10% and `use_ai=True` → re-call Gemini with feedback → if still fails → `status=NO_FACES`

**Thread safety**: `_RUN_DIRS` lock ensures unique upload directories for parallel pages.

---

### 3.3 `src/slab_v2/doc_analyze.py` — Document Analysis

Whole-document semantic analysis. Runs 3 parallel Gemini calls:
1. **Floor analyzer** (`ai_floor_analyzer.py`) → `BuildingInfo`, `FloorInfo`
2. **Column census** (`column_census.py`) → column symbols + dimensions
3. **Wall census** (`wall_census.py`) → wall symbols + thicknesses

**Census ground-truth filtering**: The wall census output determines which wall symbols exist on each floor. During per-floor text search, only symbols present in the census are searched. If census ran but a floor is missing → treat as empty (no walls), not fallback to all types. This prevents boundary walls (W1/W2/W3) from leaking to floors where they don't exist.

**Fallback**: If column census fails, falls back to `document_intelligence.py` (v1 legacy).

**Output**: `DocAnalysis` dataclass with `buildings`, `floors`, `column_types`, `wall_types`, `foundation_types`, `walls_per_floor`, `columns_per_floor`, `confidence`, `warnings`

---

### 3.4 `src/slab_v2/vector_extract.py` — PDF Vector Extraction (Stage A)

Extracts all vector paths from a PDF page using `fitz.Page.get_drawings()`.

**Process**:
1. Get raw paths (stroked/filled) from PyMuPDF
2. Flatten cubic Beziers via De Casteljau (tolerance = `bezier_tol_pt`)
3. Group by `StyleKey` (stroke RGB, fill RGB, width, dash pattern, even_odd)
4. Classify each class by fingerprint: FRAME (sheet border), HATCH, ANNOTATION, or UNKNOWN

**Output**: `VectorPath[]` (id, segments, style_id, is_closed, fill_polygon) + `StyleClass[]` (id, role, n_paths, total_length)

---

### 3.5 `src/slab_v2/planarize.py` — Planarization (Stage B)

Converts line segments into atomic closed polygon faces.

**Process**:
1. Collect deduplicated segments from selected style classes
2. Bridge collinear dashes (gaps up to 30pt)
3. GEOS snap-rounding at `snap_grid_pt` (0.05pt = 0.0088mm paper)
4. `shapely.ops.polygonize_full()` → face graph
5. Gap ladder: if too few faces, snap dangle endpoints onto existing nodes (0.25 → 0.5 → 1.0pt)
6. Face union: collect faces ≥ `min_keep_face_frac` of content area
7. Component filtering: drop components < `min_component_frac` of largest

**Output**: `FaceGraph` (faces, dangles, snap_used_pt, n_segments_in)

---

### 3.6 `src/slab_v2/ai_select.py` — Gemini Class Election (Stage C)

Gemini selects which style classes bound the slab.

**Round 1 — Class Election**:
1. Render page with classes color-coded → send to Gemini
2. Render style legend swatch sheet → send to Gemini
3. Gemini returns: `slab_edge_classes[]`, `supporting_classes[]`, `roles{}`
4. Validate: elected classes cover ≥ 30% of content bbox
5. Retry with feedback if validation fails (max 3 retries)

**Budget**: `max_total_calls = 8` Gemini calls per page (hard cap)

**Output**: `ClassElection` (slab_edge_classes, supporting_classes, roles, reasoning, warning)

---

### 3.7 `src/slab_v2/elements.py` — Element Detection (Stage D)

Finds stair/lift/shaft/void elements via X-CROSS symbol detection.

**Detection method**:
1. Find diagonal segment pairs (15–75° from axes)
2. Detect crossing diagonals (intersection near midpoints)
3. Text label classification: STAIR, LIFT, SHAFT, VOID, DUCT (regex)
4. Unlabeled X-cross → VOID
5. Text without X-cross → nearest-face fallback

**Size limits**: `xcross_min_area_frac` (0.0002) to `xcross_max_area_frac` (0.04) of content area

**Output**: `ElementFootprint[]` (type, polygon, label, anchor_bbox)

---

### 3.8 `src/slab_v2/columns_v2.py` — Column Detection v2 (Stage G)

Text-anchor-first column detection with schedule type matching.

**Two-pass detection**:
- **Pass 1**: Find column text labels (C1, SH, UC200) → search nearby for rectangles → size-match to schedule (tolerance ± `column_size_tol_mm`)
- **Pass 2**: Shape-first fallback for unlabeled rectangles

**Features**: steel exclusion zones (CH/SH/UB labels block RC detection), segmented dashed-box recovery, symbol assignment from text anchor

**Output**: `ColumnFootprint[]` (symbol, polygon, width_mm, depth_mm, grid_label)

---

### 3.9 `src/slab_v2/walls_v2.py` — Wall Detection v2 (Stage H)

Census-aware wall detection. Only detects wall symbols that the Gemini wall census says exist on this floor.

**Two-pass detection**:
- **Pass 1**: Text-anchor walls → nearby long rectangles (aspect ratio ≥ 3.0)
- **Pass 2**: WALL-class face fallback

**Census cross-check**: detected counts vs. expected per-floor → warnings if mismatch

**Output**: `WallFootprint[]` (label, polygon, w_mm, l_mm, wall_type, source, profile_id)

---

### 3.10 `src/slab_v2/opening_resolver.py` — Opening Resolution (Stage L)

Validates and classifies stair/core/shaft penetration candidates.

**Deterministic phase**:
1. Generate candidates from elements (stair + lift footprints)
2. `_core_wall_opening_candidates()`: convex hull of LW-prefix walls → SHAFT candidate
3. Geometry validation: boundary coverage, area bounds, structural intersection limits

**Optional AI judge** (`opening_judge.py`): sends image to Gemini, selects which candidate IDs to resolve. Min confidence: 0.70.

**Geometry guard**: after AI judge, force core/LW candidates with `default_action="opening"` back into `judged_ids` — prevents AI from dropping high-confidence deterministic candidates.

**Output**: `OpeningResolution` (resolved_openings, resolved_penetrations, judgement, warnings)

---

### 3.11 `src/slab_v2/export_ruby.py` — Ruby Export (Stage M)

Generates SketchUp Ruby scripts for 3D model construction.

**Single-page**: `generate_ruby()` → slab face + elements + walls + columns
**Multi-storey**: `generate_building_ruby()` → stack floors at FFL elevations

**Coordinate system**: real-world mm, page bottom-left origin, Y-up. Slab at Z=FFL, extruded downward.

**Centroid-normalized wall dedup**: walls on different PDF pages have different Y coordinates in mm space. Before comparing IoU, both polygons are translated to origin (`_centered()`). Threshold: IoU > 0.90. LW-prefix walls are exempt (they appear at every floor).

**Output**: Ruby bytecode string (downloadable from UI)

---

### 3.12 `src/slab_v2/wall_profile_resolver.py` — Wall Profiles

Extracts elevation/section page wall profiles and assigns topology to floor-plan walls.

`build_wall_source_registry()` scans all pages for elevation/section views, extracts profile geometry, optionally uses Gemini profile judge.

`resolve_plan_wall_topology()` assigns detected plan walls to topology entries. Only recovers symbols present in the census.

---

### 3.13 `src/slab_v2/core_wall_topology.py` — Core Wall Topology (Stage J)

Assigns LW (lightweight/core) wall identities from topology evidence. Cross-checks with elevation pages. Corrects mislabeled walls based on position.

---

### 3.14 `src/slab_v2/wall_junction_resolver.py` — Wall Junctions (Stage K)

Snaps small orthogonal wall endpoint gaps (≤ `wall_junction_snap_max_mm` = 25mm). Larger discontinuities remain as review items — not bridged destructively.

---

### 3.15 `src/slab_v2/height_reconcile.py` — Height Reconciliation

Multi-source storey-height datum collection and graph solving.

**Evidence sources**:
1. Dimension annotations (page-wide parsed)
2. FFL/RL/EL/NGL annotations (strict datum regex)
3. Text "storey height = NNN mm" annotations
4. Elevation/section page topology
5. Manual user overrides (session state)

**Datum graph solving**: vertices = levels, edges = height constraints. Detects conflicts (spread > 100mm). Assigns confidence (single source < multiple consistent sources).

**Output**: `HeightReconciliation` (level_datums, warnings, debug_log)

---

### 3.16 `src/slab_v2/gemini_client.py` — Gemini Integration

Vertex AI service account auth + structured-output JSON calls.

**Auth** (from `.env`):
- `GOOGLE_APPLICATION_CREDENTIALS`: GCP service account JSON path
- `GOOGLE_CLOUD_PROJECT`: GCP project ID
- `VERTEX_LOCATION`: region (default `us-central1`)
- `GEMINI_MODEL`: model name (default `gemini-2.5-flash`)

**Structured output**: `response_mime_type=application/json` + `response_schema` — no regex parsing.

**Concurrency**: semaphore limits concurrent Gemini calls (default 10, configurable).

---

### 3.17 `src/slab_v2/debug_render.py` — Debug Rendering

Step-by-step visual debug images (PNG per stage). `PageRenderer` class with methods:
- `step00_page_raster()` — page background
- `step01_paths_by_style()` — colored by class (also sent to Gemini)
- `step02_style_legend_sheet()` — swatch legend (also sent to Gemini)
- `step03_planarized()` — all faces
- `step06_elected_classes()` — Round 1 result
- `step08_assembled_slab()` — final gross slab
- `step09_elements()` — X-crosses
- `step11_columns()` — column boxes

DPI: `cfg.debug_dpi` (default 150). Images are byte-identical to those sent to Gemini.

---

### 3.18 `src/slab_v2/config.py` — Configuration

All tunable parameters in one dataclass. See [Section 4](#4-configuration-reference-cấu-hình) for the full table.

---

### 3.19 `src/slab_v2/models.py` — Data Models

Key dataclasses (no business logic):

| Class | Purpose |
|-------|---------|
| `StyleKey` | Hash identity of a line/fill style (stroke RGB, fill RGB, width, dashes) |
| `StyleClass` | Group of paths sharing a StyleKey |
| `VectorPath` | One PDF path (flattened segments) |
| `Face` | Atomic closed polygon region from planarization |
| `FaceGraph` | Output of planarization (faces, dangles, cut_edges) |
| `ClassElection` | Stage C result (slab_edge_classes, supporting_classes, roles) |
| `ElementFootprint` | Stair/lift/shaft/void footprint |
| `ColumnFootprint` | Column detection result (symbol, polygon, dimensions) |
| `WallFootprint` | Wall detection result (label, polygon, thickness, length) |
| `ResolvedPenetration` | Validated opening (id, kind, polygon, confidence) |
| `SlabV2Result` | Single-page extraction output (all geometry + metadata) |
| `DocAnalysis` | Whole-document analysis (buildings, schedules, census) |
| `BuildingInfo` | Building structure (name, floors) |
| `FloorInfo` | Floor mapping (level_id, ffl_m, pages, walls, column_counts) |
| `ColumnType` | Column schedule entry (symbol, width, depth, material) |
| `WallType` | Wall schedule entry (symbol, thickness, height, material) |
| `HeightReconciliation` | Multi-source height solution (level_datums, warnings) |
| `LevelDatum` | Solved level height (ffl_mm, storey_height_mm, confidence) |

---

### 3.20 `src/slab_v2/verify.py` — Dimension Verification (Stage E)

Parses dimension annotations from page text + lines. Used for scale cross-validation and informational overlays.

---

### 3.21 `src/slab_v2/slab_face_resolver.py` — Slab Face Judge

Optional Gemini judge for code-generated atomic slab faces. The gross deterministic slab remains the fallback when the judge fails.

---

### 3.22 `src/slab_v2/floor_system_resolver.py` — Floor System Resolver

Classifies floor system type: post-tensioned (PT) slab vs. conventional. Uses vector separators, stair adjacency, and Gemini evidence.

---

### 3.23 `src/slab_v2/opening_judge.py` — Opening Judge

Gemini-based semantic judge for opening candidates. Sends annotated page image, receives selected opening IDs with confidence scores.

---

### 3.24 `src/slab_v2/readiness.py` — Export Readiness

Aggregates per-page quality metrics into building-level readiness reports:
- Slab verification status
- Opening resolution completeness
- Wall detection + topology match
- Column count match
- Shaft continuity across storeys

**Output**: `ModelReadinessReport` (overall status per building: `verified` | `review`)

---

### 3.25 `src/slab_v2/wall_census.py` — Wall Census Prompt

Gemini prompt that extracts wall symbols + schedule properties:
- Goal A: Find wall schedule/legend (symbol, thickness, height, material)
- Goal B: Scan plan pages for wall labels + counts per floor
- Collects elevation/section pages with wall profiles

---

### 3.26 `src/slab_v2/column_census.py` — Column Census Prompt

Gemini prompt that extracts column symbols + schedule dimensions:
- Searches column schedule tables
- Scans plan pages for column labels
- Infers material (RC / STEEL / UNKNOWN)
- Maps symbols to buildings + floors with counts

---

### 3.27 Other `slab_v2` Modules

| Module | Purpose |
|--------|---------|
| `adapter.py` | Compatibility adapter between v1 and v2 interfaces |
| `wall_extract.py` | Legacy face-based wall extraction (Pass 2 fallback) |
| `columns.py` | Legacy shape-first column detection (Pass 2 fallback for `columns_v2`) |
| `element_geometry.py` | Geometry helpers for element footprint calculations |
| `height_source_planner.py` | Plans which pages to scan for height evidence |
| `column_reconciler.py` | Reconciles column detection across multiple pages |

---

### 3.28 Legacy Modules (`src/`)

These are v1 modules. Most are replaced by `slab_v2` but some are still called:

| Module | Status | Purpose |
|--------|--------|---------|
| `ai_floor_analyzer.py` | **Active** — called by `doc_analyze.py` | Building/floor structure extraction via Gemini |
| `document_intelligence.py` | **Active** — fallback | Column/foundation schedule extraction (if census fails) |
| `coordinate_mapper.py` | **Active** — called by `export_ruby.py` | PDF point → real-world mm transformation |
| `column_detector.py` | **Active** — helpers | Column bounds, size matching, label search |
| `building_site_placement.py` | **Active** — Phase 2.5 | Multi-building relative positioning |
| `column_analyzer.py` | Legacy | v1 column analysis |
| `floor_detector.py` | Legacy | v1 floor detection |
| `floor_alignment.py` | Legacy | v1 floor alignment |
| `wall_detector.py` | Legacy | v1 wall detection |
| `slab_extractor.py` | Legacy | v1 slab extraction |
| `slab_semantic_detector.py` | Legacy | v1 semantic slab detection |
| `legend_locator.py` | Legacy | v1 legend finding |
| `legend_semantic_analyzer.py` | Legacy | v1 legend analysis |
| `boundary_slab_extractor.py` | Legacy | v1 boundary slab |
| `interior_slab_resolver.py` | Legacy | v1 interior slab |
| `structural_boundary_detector.py` | Legacy | v1 boundary detection |
| `line_semantic_analyzer.py` | Legacy | v1 line analysis |
| `building_registry.py` | Legacy | v1 building registry |
| `building_audit.py` | Legacy | v1 building audit |
| `model_builder.py` | Legacy | v1 model building |
| `visualizer.py` | Legacy | v1 visualization |
| `vision_refiner.py` | Legacy | v1 vision refinement |
| `structural_elements.py` | Legacy | v1 structural elements |
| `pdf_processor.py` | Legacy | v1 PDF processing |
| `pipeline_logger.py` | Legacy | v1 logging |

---

## 4. Configuration Reference (Cấu hình)

### 4.1 `SlabV2Config` Fields

All parameters live in `src/slab_v2/config.py`. Units: "pt" = PDF points (1pt = 0.3528mm paper). At scale 1:100, 1pt = 35.28mm real-world.

#### Stage A — Vector Extraction

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `bezier_tol_pt` | 0.2 | Max sagitta for Bezier flattening (De Casteljau) |
| `frame_area_frac` | 0.97 | Rectangles covering > this fraction → FRAME (excluded) |

#### Stage B — Planarization

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `snap_grid_pt` | 0.05 | GEOS snap-rounding grid size |
| `gap_ladder_pt` | (0.25, 0.5, 1.0) | Dangle snap distances tried in order |
| `min_face_area_frac` | 0.005 | Face "significant" threshold (fraction of content area) |
| `min_faces` | 1 | Min faces before gap ladder kicks in |
| `max_polygonize_segments` | 200,000 | Safety cap for shapely memory |

#### Stage C — AI Selection

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `gemini_model` | "" | Override model (empty = env or gemini-2.5-flash) |
| `max_classes_in_prompt` | 25 | Classes shown individually in Round 1 |
| `max_faces_in_prompt` | 30 | Faces labeled in Round 2 image |
| `min_class_coverage_frac` | 0.30 | Elected classes must cover ≥ 30% of content bbox |
| `min_area_frac` / `max_area_frac` | 0.05 / 0.90 | Slab area validation bounds |
| `max_ai_retries` | 3 | Per-round retries with feedback |
| `max_total_calls` | 8 | Hard Gemini call budget per page |
| `enable_opening_judge` | True | Call Gemini for opening validation |
| `opening_judge_min_confidence` | 0.70 | Min confidence for judge override |
| `enable_slab_face_judge` | True | Judge code-generated slab faces |
| `slab_judge_min_confidence` | 0.80 | Slab face classification threshold |
| `slab_subtract_min_confidence` | 0.85 | Destructive subtraction requires higher confidence |
| `slab_max_net_area_loss_frac` | 0.35 | Max slab area loss from judge decisions |
| `enable_floor_system_judge` | True | Judge PT vs. other floor systems |

#### Assembly

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_keep_face_frac` | 0.001 | Faces above this → union (noise filter) |
| `min_component_frac` | 0.02 | Drop disconnected components < this fraction |
| `element_max_area_frac` | 0.15 | Element max area (mis-anchor filter) |
| `xcross_min_area_frac` | 0.0002 | X-cross min area (filters annotation circles) |
| `xcross_max_area_frac` | 0.04 | X-cross max area |
| `element_text_radius_pt` | 80.0 | Label-to-element anchor distance |

#### Export

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `slab_thickness_mm` | 200.0 | Slab extrusion depth |
| `element_height_mm` | 3000.0 | Default element volume height |
| `default_storey_height_mm` | 3000.0 | Fallback storey height |
| `shaft_wall_thickness_mm` | 150.0 | Shaft ring wall thickness |
| `render_shaft_solids` | False | Create shaft ring volumes |
| `cut_stair_openings` | False | Stairs never cut slab |
| `cut_verified_lift_voids` | True | Lift shafts cut slab holes |
| `shaft_pair_min_iou` | 0.30 | Min IoU for shaft continuity pairing |

#### Columns

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `column_size_tol_mm` | 25.0 | Size match tolerance (both sides) |
| `column_label_radius_pt` | 60.0 | Label-to-shape distance |
| `column_text_search_radius_pt` | 40.0 | v2: label search radius |
| `column_min_repeat` | 4 | Min occurrences for anonymous type |
| `column_max_side_mm` | 1500.0 | Max rectangle side |
| `steel_exclusion_radius_pt` | 40.0 | Buffer around steel labels |

#### Penetration / Opening Thresholds

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `penetration_min_boundary_coverage` | 0.55 | Min vector boundary coverage for opening |
| `penetration_min_confidence` | 0.85 | Min confidence for penetration |
| `core_opening_min_boundary_coverage` | 0.70 | Min boundary for core opening |
| `slab_penetration_min_area_m2` | 0.05 | Min penetration area |
| `slab_penetration_max_area_m2` | 10.0 | Max penetration area |
| `stair_opening_min_area_m2` | 0.25 | Min stair opening area |
| `stair_opening_max_area_m2` | 40.0 | Max stair opening area |
| `wall_junction_snap_max_mm` | 25.0 | Max wall endpoint gap to snap |

#### Height Reconciliation

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `height_conflict_tolerance_mm` | 100.0 | Conflict detection threshold |
| `height_reject_residual_mm` | 150.0 | Outlier rejection threshold |

#### Verification

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `dim_assoc_radius_pt` | 25.0 | Dim text ↔ dim line distance |
| `dim_parallel_tol_deg` | 5.0 | Dim line angle tolerance |
| `dim_rel_tol` | 0.015 | Dimension error tolerance (relative) |
| `precise_scale_min_dims` | 4 | Dims needed for consensus scale |
| `precise_scale_max_spread` | 0.01 | Max spread for consensus |

#### Performance

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `extraction_max_workers` | 10 | Max parallel extraction threads |
| `max_parallel_pages` | 10 | ThreadPoolExecutor worker count |
| `speed_mode` | False | Skip optional judges + debug images for batch |
| `debug_images` | True | Write step PNG images |

#### Output

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `debug_dir` | "debug_slab_v2" | Output folder name |
| `debug_dpi` | 150 | Image DPI |
| `prompt_dpi` | 150 | Gemini prompt image DPI |
| `save_prompt_images` | True | Save Gemini images to debug folder |

### 4.2 Environment Variables (`.env`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | Path to GCP service account JSON |
| `GOOGLE_CLOUD_PROJECT` | Yes | GCP project ID |
| `VERTEX_LOCATION` | No | Vertex AI region (default: `us-central1`) |
| `GEMINI_MODEL` | No | Model name (default: `gemini-2.5-flash`) |

---

## 5. Debug & Troubleshooting

### 5.1 Output Directory Structure

```
debug_slab_v2/
└── <pdf_stem>/
    └── upload<N>/
        ├── doc_analysis.json          # Full document analysis
        ├── wall_census_*_raw.txt      # Raw Gemini wall census response
        ├── column_census_*_raw.txt    # Raw Gemini column census response
        └── page_<P>/
            ├── step_00_page_raster.png          # Page background
            ├── step_01_paths_by_style.png        # Paths colored by class (→ Gemini)
            ├── step_02_style_legend_sheet.png    # Style swatch legend (→ Gemini)
            ├── step_03_planarized.png            # All faces from polygonize
            ├── step_04_faces_all.png             # Faces numbered
            ├── step_06_elected_classes.png       # Round 1 class election result
            ├── step_07_faces_candidates.png      # Round 2 face candidates
            ├── step_08_assembled_slab.png        # Final gross slab polygon
            ├── step_09_elements.png              # X-cross elements detected
            ├── step_09b_dimensions.png           # Dimension annotations
            ├── step_10b_columns.png              # Column detection
            ├── step_11_columns.png               # Final column footprints
            ├── prompts.log                       # Full Gemini conversation transcript
            └── result.json                       # SlabV2Result (all geometry + metadata)
```

### 5.2 Key Fields in `result.json`

| Field | What to check |
|-------|--------------|
| `status` | Should be `OK`. `NO_FACES` means slab assembly failed |
| `scale` | Should match drawing title block (e.g., 100 for 1:100) |
| `slabs[0].area_m2` | Gross slab area — should match expected floor area |
| `columns` | Count should match schedule census |
| `walls` | Count should match wall census for this floor |
| `resolved_penetrations` | Core/lift openings should appear here |
| `warnings` | Review any warnings — scale conflicts, missing walls, etc. |
| `election.slab_edge_classes` | Which style classes Gemini selected as slab boundary |

### 5.3 Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `status=NO_FACES` | Gemini elected wrong classes, or PDF has unusual line styles | Check `step_06` — are the colored lines actually slab edges? Try `manual_scale` override |
| Slab too small (< 50% expected) | Elected classes don't include all slab boundary segments | Check `step_01` — find the correct class visually, verify it's in the election |
| Walls appearing on wrong floors | Census override bug — page text search overriding census | Check `doc_analysis.json` → `walls_per_floor`. Should match `wall_census_raw.txt` |
| Core shaft not cut | AI judge dropping core candidate | Check `result.json` → `resolved_penetrations`. Geometry guard should force core back |
| Wrong scale | PDF has non-standard viewport or no text scale | Use `manual_scale` config override. Check `step_09b_dimensions` for measurement consistency |
| Columns not found | Labels too far from rectangles, or schedule missing | Increase `column_text_search_radius_pt`. Check column census output |
| Walls too thick/thin | Schedule thickness doesn't match drawn thickness | Check wall census vs. actual page text. May need schedule page re-scan |

### 5.4 Reading `prompts.log`

Each Gemini call is logged with:
- `[PROMPT]` — the text prompt sent
- `[IMAGES]` — image byte sizes
- `[RESPONSE]` — full JSON response
- `[TOKENS]` — token usage

This is the primary tool for diagnosing why Gemini made a specific classification decision.

---

## 6. Key Design Decisions (Quyết định thiết kế)

### 6.1 Why Vector-First (not OCR/raster)

PDF structural drawings contain exact vector geometry. OCR/raster approaches lose precision and introduce noise. By extracting vector paths directly from PyMuPDF, we get sub-millimeter accuracy without any image processing. AI is only needed to decide *which* paths matter, not *where* they are.

### 6.2 Why Gemini-as-Selector (not coordinate source)

Gemini is excellent at visual classification but poor at precise coordinate extraction. Our architecture uses Gemini to answer "which style classes form the slab boundary?" — a classification task it handles well. All coordinates come from the PDF, so Gemini errors only affect selection (recoverable via retry/fallback), never geometry accuracy.

### 6.3 Why Gross+Holes Model

The slab polygon stays gross (no holes) throughout extraction. Openings (stair, lift, shaft) are tracked separately as `ElementFootprint` and `ResolvedPenetration`. Holes are only cut at Ruby export time. This is the standard BIM approach (host element + hosted openings) and prevents premature geometry corruption if opening detection has errors.

### 6.4 Why Census Ground Truth

The Gemini wall/column census determines which symbols exist on each floor. This is the single source of truth for "what should we look for." If the census says Level 02 has only LW1-LW7, we only search for those — never W1/W2/W3. This prevents boundary walls from leaking to upper floors, which was a critical bug caused by page-text search overriding correct census data.

### 6.5 Why Centroid-Normalized IoU for Wall Dedup

The same wall (e.g., W1) drawn on different PDF pages has different Y-coordinates in mm space because `coordinate_mapper.py` uses `page.rect.y1` as the origin and each page has different absolute coordinates. Standard IoU comparison returns ~0 even for identical walls. By translating both polygons to the centroid origin before comparison, we get correct IoU regardless of page offset.

### 6.6 Why Geometry Guard for Core Penetration

The convex hull of LW walls is a high-confidence deterministic candidate for core shaft opening. However, the AI opening judge (`opening_judge.py`) replaces the entire `judgement` dict at line 621, which can drop this candidate. The geometry guard forces core candidates with `default_action="opening"` back into `judged_ids` after the AI judge runs, preserving deterministic confidence while still allowing the AI to override other candidates.

---

## 7. Testing (Kiểm thử)

### 7.1 Test Files

| Test file | What it covers |
|-----------|---------------|
| `tests/test_verified_opening_geometry.py` | Stair/lift/shaft penetration boundary validation |
| `tests/test_verified_wall_columns_shafts.py` | Multi-storey element continuity (IoU pairing) |
| `tests/test_height_datum_graph.py` | Level FFL solving from multi-source evidence |
| `tests/test_floor_system_resolver.py` | PT slab vs. other floor system classification |
| `tests/test_wall_profile_export.py` | Elevation profile to Ruby export |
| `tests/test_slab_face_resolver.py` | Slab face semantic judgment |
| `tests/test_wall_vertical_scope.py` | Under-only / over-only wall detection |

### 7.2 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test
python -m pytest tests/test_height_datum_graph.py -v

# Run with coverage
python -m pytest tests/ --cov=src/slab_v2 --cov-report=term-missing
```

### 7.3 Adding a Test Page

1. Run extraction on the new PDF to produce `debug_slab_v2/.../<page>/result.json`
2. Inspect the result — identify expected values (wall count, column count, opening types)
3. Copy the `result.json` to `tests/fixtures/` or reference it from test
4. Write assertions checking key fields against expected values

---

## 8. Operational Guide (Vận hành)

### 8.1 Single-Building Extraction

1. Start the app: `streamlit run app_v2.py`
2. **Phase 1**: Upload the structural PDF
3. **Phase 2**: Wait for document analysis (typically 15-30 seconds for 3 parallel Gemini calls)
4. Review the building/floor/schedule summaries displayed
5. Adjust sidebar config if needed (slab thickness, scale override)
6. **Phase 3**: Click Extract — pages are processed in parallel (min 10 workers)
7. Review readiness report — green = verified, yellow = needs review
8. Download the `.rb` file
9. In SketchUp: `Window → Ruby Console → load "<path>.rb"`

### 8.2 Multi-Building Project

Same as above, but Phase 2.5 automatically audits relative building positions. Export may include coordinate transforms. Check for warnings about unverified site placement.

### 8.3 Batch Mode (Speed Mode)

For processing many PDFs quickly:

```python
cfg = SlabV2Config(speed_mode=True)
```

This disables:
- Optional Gemini judges (opening, slab face, floor system)
- Debug images (except prompt images needed for Gemini)

The gross slab geometry (Round 1 election + deterministic assembly) is unaffected. Re-run problem pages with `speed_mode=False` for full diagnostics.

### 8.4 Worker Configuration

The worker pool always uses at least 10 threads:

```python
max_w = max(10, int(cfg.extraction_max_workers))
```

This is set in `app_v2.py` and ensures adequate parallelism even with default config.

### 8.5 Key Dependencies

| Package | Purpose |
|---------|---------|
| `PyMuPDF` (`fitz`) | PDF rasterization + vector extraction |
| `shapely` | Computational geometry (polygonize, snap, union, buffer) |
| `google-cloud-aiplatform` | Vertex AI Gemini API |
| `streamlit` | Web UI framework |
| `Pillow` (`PIL`) | Image rendering for debug + prompts |
| `pandas` | Data display in UI |
| `numpy` | Numerical operations |

---

## Appendix: Coordinate System

```
PDF Space (fitz)              Real-World Space (mm)
┌───────────────┐             ┌───────────────┐
│ origin: top-left            │ origin: bottom-left
│ Y increases downward        │ Y increases upward
│ units: PDF points           │ units: millimeters
│ (1pt = 1/72 inch)          │ (scale-dependent)
└───────────────┘             └───────────────┘

Transform (coordinate_mapper.py):
  real_x = (x_pdf - page.rect.x0) * PT_TO_MM * scale
  real_y = (page.rect.y1 - y_pdf) * PT_TO_MM * scale
  where PT_TO_MM = 25.4 / 72 ≈ 0.3528
```

---

*This document describes the system as of commit `b9a5fd2` (2026-06-23). For AI agent instructions, see `CLAUDE.md`.*
