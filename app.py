"""
Feeldx Slab Extractor — Streamlit App
Phase 1: PDF Structural → SketchUp 3D Slabs
"""

import io
import os
import uuid
import math
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image

# ── compatibility shim ─────────────────────────────────────────────────────────
def _rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Feeldx Slab Extractor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  html, body, [class*="css"] { font-size: 15px; }
  .main .block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1400px; }

  /* Step header */
  .step-header {
    font-size: 1.6rem; font-weight: 800; color: #4FC3F7;
    padding: 10px 0 4px 0; letter-spacing: 0.02em;
    border-bottom: 2px solid #1e3a5f; margin-bottom: 12px;
  }

  /* Info boxes */
  .info-box {
    background: #0f2035; border-left: 5px solid #4FC3F7;
    padding: 14px 18px; border-radius: 6px; margin: 10px 0;
    font-size: 1.0rem; line-height: 1.6;
  }
  .success-box {
    background: #0a2218; border-left: 5px solid #66BB6A;
    padding: 14px 18px; border-radius: 6px; margin: 10px 0;
    font-size: 1.0rem; line-height: 1.6;
  }
  .warn-box {
    background: #2a1f00; border-left: 5px solid #FFA726;
    padding: 14px 18px; border-radius: 6px; margin: 10px 0;
    font-size: 1.0rem; line-height: 1.6;
  }
  .error-box {
    background: #2a0a0a; border-left: 5px solid #EF5350;
    padding: 14px 18px; border-radius: 6px; margin: 10px 0;
    font-size: 1.0rem;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #0f2035; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 12px 16px;
  }
  div[data-testid="metric-container"] label {
    font-size: 0.85rem !important; color: #90CAF9 !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important; color: #E3F2FD !important; font-weight: 700 !important;
  }

  /* Buttons */
  div[data-testid="stButton"] > button {
    font-size: 1.0rem; font-weight: 600; border-radius: 8px;
    padding: 10px 24px; transition: all 0.2s;
  }
  div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1565C0, #0288D1);
    border: none; color: white;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1976D2, #039BE5);
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(41,182,246,0.3);
  }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #060e1a; }
  section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

  /* Progress bar */
  div[data-testid="stProgress"] > div > div { background: #29B6F6 !important; }

  /* Data editor */
  div[data-testid="stDataEditor"] { border: 1px solid #1e3a5f; border-radius: 8px; }

  /* Expander */
  details summary { font-size: 1.05rem; font-weight: 600; color: #90CAF9; }

  /* Tabs */
  button[data-baseweb="tab"] { font-size: 0.9rem; }
  button[data-baseweb="tab"][aria-selected="true"] { color: #4FC3F7 !important; }
</style>
""", unsafe_allow_html=True)


# ── constants ──────────────────────────────────────────────────────────────────
DEBUG_DIR = Path("debug_images")
OUTPUT_DIR = Path("output")
DEBUG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SLAB_THICKNESS_MM = 200


# ── session helpers ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "step": 1,
        "pdf_path": None,
        "pdf_metadata": None,
        "page_infos": None,
        "selected_pages": [],
        "scale": None,
        "slab_results": {},
        "debug_images": {},
        "final_slabs": [],
        "ruby_script": None,
        "ruby_path": None,
        "csv_path": None,
        "log_path": None,
        "smart_detect_result": None,
        "smart_detect_done": False,
        "ai_floor_result": None,
        "ai_floor_output_path": None,
        "ai_floor_pdf": None,
        "vision_backend": "gemini",
        "column_census": None,
        "column_regions": [],
        "foundation_regions": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_session()


init_session()
sid = st.session_state["session_id"]
sess_debug_dir = DEBUG_DIR / sid
sess_debug_dir.mkdir(exist_ok=True)


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.4rem;font-weight:800;color:#4FC3F7;padding:4px 0 2px 0;">'
        '🏗️ Feeldx Slab Extractor</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div style="color:#546E7A;font-size:0.85rem;margin-bottom:12px;">'
        'Phase 1 — PDF → SketchUp 3D</div>', unsafe_allow_html=True
    )
    st.divider()

    steps = [
        ("1", "Upload PDF"),
        ("2", "Select Pages"),
        ("3", "Detect Slabs"),
        ("4", "Review"),
        ("5", "Generate 3D"),
        ("6", "Export to SketchUp"),
    ]
    current_step = st.session_state["step"]
    for num, name in steps:
        n = int(num)
        is_done = n < current_step
        is_active = n == current_step
        if is_done:
            icon, color, bg = "✅", "#66BB6A", "#0a2218"
        elif is_active:
            icon, color, bg = "▶️", "#4FC3F7", "#0f2035"
        else:
            icon, color, bg = "⬜", "#546E7A", "transparent"
        st.markdown(
            f'<div style="padding:7px 10px;margin:2px 0;color:{color};'
            f'background:{bg};border-radius:6px;font-size:0.95rem;">'
            f'{icon}&nbsp; <b>Step {num}</b>: {name}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        reset_session()
        _rerun()

    st.markdown(
        f'<div style="color:#37474F;font-size:0.75rem;margin-top:8px;">Session: {sid}</div>',
        unsafe_allow_html=True,
    )


# ── STEP 1: Upload ─────────────────────────────────────────────────────────────
def step1_upload():
    st.markdown('<div class="step-header">📂 Step 1 — Upload Structural PDF</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Upload Australian structural drawings (vector PDF from Bluebeam/AutoCAD). '
        'The system automatically detects slabs, extracts FFL elevations, '
        'and generates a dimensionally accurate SketchUp 3D model.'
        '</div>', unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drag & drop PDF here or click to browse",
        type=["pdf"],
        help="Vector PDF supported. Scanned PDFs yield lower accuracy.",
        label_visibility="visible",
    )

    if uploaded:
        tmp_path = Path(tempfile.gettempdir()) / f"feeldx_{sid}_{uploaded.name}"
        tmp_path.write_bytes(uploaded.read())
        st.session_state["pdf_path"] = str(tmp_path)

        with st.spinner("Reading PDF..."):
            import fitz
            from src.pdf_processor import get_pdf_metadata, classify_pages, load_pdf
            doc = load_pdf(str(tmp_path))
            meta = get_pdf_metadata(doc)
            page_infos = classify_pages(doc)
            doc.close()

        st.session_state["pdf_metadata"] = meta
        st.session_state["page_infos"] = page_infos

        floor_plan_count = sum(1 for p in page_infos if p["is_floor_plan"])
        w_pts, h_pts = meta.get("page_size_pts", (0, 0))
        page_size_label = "A1" if w_pts > 2300 else ("A3" if w_pts > 1100 else "Other")

        st.markdown("---")
        st.markdown("#### 📊 PDF Info")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total pages", meta["page_count"])
        c2.metric("Floor Plan (auto)", floor_plan_count)
        c3.metric("Paper size", page_size_label)
        c4.metric("File", uploaded.name[:18] + ("…" if len(uploaded.name) > 18 else ""))

        if meta.get("creator"):
            st.markdown(
                f'<div class="success-box">✅ Vector PDF confirmed — Creator: '
                f'<b>{meta["creator"]}</b></div>', unsafe_allow_html=True,
            )

        # ── Auto-run Gemini AI floor analysis on new upload ────────────────────
        pdf_path = str(tmp_path)
        if st.session_state.get("ai_floor_pdf") != pdf_path:
            with st.spinner("Gemini AI analysing floor structure... (15-30s)"):
                try:
                    import fitz as _fitz_ai
                    from src.ai_floor_analyzer import analyze_floor_structure
                    _doc_ai = _fitz_ai.open(pdf_path)
                    _all_pages = list(range(_doc_ai.page_count))
                    _doc_ai.close()
                    ai_result, ai_path = analyze_floor_structure(pdf_path, _all_pages)
                    st.session_state["ai_floor_result"] = ai_result
                    st.session_state["ai_floor_pdf"]    = pdf_path
                    st.session_state["ai_floor_output_path"] = ai_path
                    smart_res = _ai_result_to_floor_detect(ai_result, _all_pages)
                    st.session_state["smart_detect_result"] = smart_res
                    st.session_state["smart_detect_done"] = False
                    slab_pages = sorted({
                        pg - 1
                        for b in ai_result.get("buildings", [])
                        for f in b.get("floors", [])
                        for pg in f.get("slab_plan_pages", [])
                        if isinstance(pg, int)
                    })
                    st.session_state["selected_pages"] = slab_pages
                except Exception as _e:
                    st.warning(f"Gemini floor analysis failed: {_e}. Configure manually in Step 2.")

        st.markdown("---")
        if st.button("Next: Configure →", type="primary", use_container_width=True):
            st.session_state["step"] = 2
            _rerun()


# ── STEP 2: Configure ────────────────────────────────────────────────────────────
def step2_select_pages():
    st.markdown("""<div class="step-header">⚙️ Step 2 — Configure Detection</div>""",
                unsafe_allow_html=True)

    page_infos = st.session_state.get("page_infos") or []
    smart_result = st.session_state.get("smart_detect_result")

    # ── Scale selector ──────────────────────────────────────────────────────────
    scale_auto = next((p["scale"] for p in page_infos if p.get("scale")), None)
    st.markdown("#### ⚙️ Drawing Scale")
    col_s, col_hint = st.columns([1, 2])
    with col_s:
        scale_input = st.number_input(
            "Scale (1 : N)",
            min_value=10, max_value=2000,
            value=scale_auto or 100,
            step=10,
            help="Check the drawing title block. Common: 1:100, 1:200, 1:50",
        )
        st.session_state["scale"] = int(scale_input)
    with col_hint:
        if scale_auto:
            st.markdown(
                f"""<div class="success-box" style="margin-top:28px;">✅ Auto-detected scale: <b>1:{scale_auto}</b></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="warn-box" style="margin-top:28px;">⚠️ Scale not detected — enter manually from the PDF title block.</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Vision backend ──────────────────────────────────────────────────────────
    st.markdown("#### 👁️ Vision Backend")
    vision_backend = st.radio(
        "Vision LLM for slab boundary tracing",
        ["gemini", "openai"],
        index=0 if st.session_state.get("vision_backend", "gemini") == "gemini" else 1,
        horizontal=True,
    )
    st.session_state["vision_backend"] = vision_backend

    st.markdown("---")

    # ── Detected floors summary ─────────────────────────────────────────────────
    st.markdown("#### 🏢 Detected Floors")
    if smart_result and smart_result.groups:
        n_pages = len(st.session_state.get("selected_pages", []))
        st.markdown(
            f"""<div class="success-box">✅ Gemini detected <b>{smart_result.floor_count} floors</b> across <b>{n_pages} plan pages</b></div>""",
            unsafe_allow_html=True,
        )
        rows = []
        for g in smart_result.groups:
            rows.append({
                "Floor": g.floor_label,
                "Canonical": f"Page {g.canonical_page + 1}",
                "Supplement": ", ".join(f"P{p+1}" for p in g.supplemental_pages) or "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ai_path = st.session_state.get("ai_floor_output_path")
        if ai_path:
            try:
                with open(ai_path, "rb") as f:
                    st.download_button(
                        "📥 Download gemini_floors.json",
                        f,
                        file_name="gemini_floors.json",
                        mime="application/json",
                    )
            except FileNotFoundError:
                pass
    else:
        st.markdown(
            """<div class="warn-box">⚠️ No floor structure detected. Gemini analysis may have failed — check your API credentials.</div>""",
            unsafe_allow_html=True,
        )

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 1
            _rerun()
    with col_next:
        selected = st.session_state.get("selected_pages", [])
        if selected:
            if st.button(
                f"▶ Detect Slabs on {len(selected)} pages",
                type="primary", use_container_width=True,
            ):
                st.session_state["smart_detect_done"] = True
                st.session_state["slab_results"] = {}
                st.session_state["step"] = 3
                _rerun()
        else:
            st.warning("No floor plan pages detected. Go back and re-upload.")

# ── AI result converter ──────────────────────────────────────────────────────────

def _ai_result_to_floor_detect(ai_result: dict, selected_pages: list):
    """Convert Gemini AI JSON result → FloorDetectResult for use in step3_detect()."""
    from src.floor_detector import FloorDetectResult, FloorGroup

    pages_to_process = ai_result.get("pages_to_process", list(selected_pages))
    all_page_set = set(selected_pages)
    skipped = sorted(all_page_set - set(pages_to_process))

    groups = []
    for bld in ai_result.get("buildings", []):
        bld_name = bld.get("name", "Building")
        for floor in bld.get("floors", []):
            raw_pages = floor.get("slab_plan_pages", [])
            # Convert 1-indexed → 0-indexed, filter to valid selected pages
            page_0idx = sorted(p - 1 for p in raw_pages if isinstance(p, int) and (p - 1) in all_page_set)
            if not page_0idx:
                continue
            ffl = floor.get("ffl_m")
            label = f"{bld_name} — {floor.get('level_name', '?')}"
            if ffl is not None:
                label += f" (FFL {float(ffl):.3f}m)"
            groups.append(FloorGroup(
                floor_key=floor.get("level_id", f"floor_{len(groups)}"),
                floor_label=label,
                canonical_page=page_0idx[0],
                supplemental_pages=page_0idx[1:],
                skipped_pages=[],
            ))

    confidence = ai_result.get("detection_confidence", "low")
    notes = ai_result.get("notes", "")
    warnings = [f"Gemini confidence: {confidence}"] + ([notes] if notes else [])

    return FloorDetectResult(
        groups=groups,
        pages_to_process=pages_to_process,
        skipped_pages=skipped,
        detection_basis="ai_gemini",
        floor_count=ai_result.get("total_unique_floors", len(groups)),
        warnings=warnings,
    )


# ── Floor group merge (AI-guided) ───────────────────────────────────────────────

_LEVEL_ORDER = {
    "basement_2": -7.0, "basement_b2": -7.0, "carpark_b2": -7.0,
    "basement_1": -3.5, "basement_b1": -3.5, "carpark_b1": -3.5,
    "basement":   -3.5, "carpark":     -3.5,
    "ground":      0.0, "level_1":      0.0,
    "podium":      3.5,
    "level_2":     3.5, "level_3":  7.0, "level_4": 10.5, "level_5": 14.0,
    "level_6":    17.5, "level_7": 21.0, "level_8": 24.5, "level_9": 28.0,
    "level_10":   31.5, "mezzanine": 2.0,
    "lower_roof": 14.0, "upper_roof": 17.5, "roof": 14.0,
}


def _estimate_ffl(level_id: str, level_name: str) -> float:
    """Last-resort FFL estimate when no elevation data exists in the PDF."""
    import re
    key = level_id.lower().strip()
    if key in _LEVEL_ORDER:
        return _LEVEL_ORDER[key]
    # Try extracting a floor number: "level_3" → 3, "floor_4" → 4
    m = re.search(r"(\d+)", key)
    if m:
        n = int(m.group(1))
        return round((n - 1) * 3.5, 3)
    return 0.0


def _merge_slabs_by_ai_floors(all_slabs: list, ai_floor_result: dict) -> list:
    """
    Given Gemini's floor grouping, merge slabs from the same level into one polygon.

    Benefits:
    - Cross-page duplicates: pages 9+10 both show full floor → unary_union of identical
      polygons = same polygon (deduplication is automatic)
    - Zone A+B splits: each shows a partial floor → unary_union gives the true combined
      shape, even if L-shaped or T-shaped
    - Non-rectangular slabs: Shapely unary_union preserves exact geometry
    """
    from collections import Counter, defaultdict
    from shapely.ops import unary_union
    from shapely.geometry import MultiPolygon
    from src.slab_extractor import SlabRegion

    # Build 0-indexed page → floor info
    page_to_floor: dict = {}
    for bld in ai_floor_result.get("buildings", []):
        bld_name = bld.get("name", "Building")
        for floor in bld.get("floors", []):
            info = {
                "level_id":   floor.get("level_id", "unknown"),
                "level_name": floor.get("level_name", "Floor"),
                "ffl_m":      floor.get("ffl_m"),
                "building":   bld_name,
            }
            for p_1idx in floor.get("slab_plan_pages", []):
                if isinstance(p_1idx, int) and p_1idx >= 1:
                    page_to_floor[p_1idx - 1] = info

    # Group slabs by building-unique key.
    # MUST include building name: "level_3" appears in Building A, B, C, D — without the
    # building prefix they would all collapse into one wrongly merged slab.
    floor_slab_map: dict = defaultdict(list)
    floor_info_map: dict = {}
    ungrouped: list = []

    for slab in all_slabs:
        info = page_to_floor.get(slab.page_index)
        if info:
            lid = f"{info['building']}__{info['level_id']}"
            floor_slab_map[lid].append(slab)
            floor_info_map[lid] = info
        else:
            ungrouped.append(slab)

    merged_slabs = []
    for lid, group_slabs in floor_slab_map.items():
        info = floor_info_map[lid]
        polys = [
            getattr(s, "real_polygon", None)
            for s in group_slabs
            if getattr(s, "real_polygon", None) is not None
            and not getattr(s, "real_polygon", None).is_empty
        ]
        if not polys:
            ungrouped.extend(group_slabs)
            continue

        merged = unary_union(polys)

        # Resolve FFL (priority order):
        # 1. Gemini explicit value  2. majority vote from page FFL regex  3. estimate from level name
        ffl_m = info["ffl_m"]
        if ffl_m is None:
            ffl_counts = Counter(s.ffl_m for s in group_slabs if s.ffl_m is not None)
            ffl_m = ffl_counts.most_common(1)[0][0] if ffl_counts else None
        if ffl_m is None:
            ffl_m = _estimate_ffl(info["level_id"], info["level_name"])

        ref = group_slabs[0]

        def _make_slab(poly, suffix=""):
            ms = SlabRegion(
                id=ref.id,
                polygon=ref.polygon,
                label=f"{info['building']} — {info['level_name']}{suffix}",
                ffl_m=ffl_m,
                ffl_mm=(ffl_m * 1000) if ffl_m is not None else ref.ffl_mm,
                area_m2=poly.area / 1_000_000.0,
                page_index=ref.page_index,
                source="merged",
            )
            ms.real_polygon = poly
            return ms

        if isinstance(merged, MultiPolygon):
            # Disconnected parts → keep each as a separate slab (e.g. separate buildings)
            parts = sorted(merged.geoms, key=lambda g: g.area, reverse=True)
            for i, part in enumerate(parts):
                merged_slabs.append(_make_slab(part, f" (Part {i+1})" if len(parts) > 1 else ""))
        else:
            merged_slabs.append(_make_slab(merged))

    return merged_slabs + ungrouped


# ── STEP 3 helpers ──────────────────────────────────────────────────────────────

def _process_page_worker(args: tuple) -> tuple:
    """
    Worker for parallel page processing.
    Opens its own fitz document — fitz.Document is NOT thread-safe when shared.
    """
    pdf_path, page_idx, scale, debug_base = args

    import fitz
    from src.pdf_processor import extract_text_blocks, extract_ffl_values, extract_slab_labels
    from src.slab_extractor import (
        extract_slabs_from_page, build_polygons_from_drawings,
        reconstruct_closed_polygons, filter_slab_candidates,
    )
    from src.coordinate_mapper import transform_all_slabs
    from src.visualizer import (
        save_step1_raw_paths, save_step2_polygons, save_step3_filtered,
        save_step4_labeled, save_step5_final,
    )

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        text_blocks = extract_text_blocks(page)
        ffl_values = extract_ffl_values(text_blocks)
        slab_labels = extract_slab_labels(text_blocks)

        slab_regions, drawings = extract_slabs_from_page(page, text_blocks, ffl_values, slab_labels)

        if slab_regions:
            slab_regions = transform_all_slabs(slab_regions, page, scale)

        # Build polygon lists once — reuse for visualization steps
        filled_pairs = build_polygons_from_drawings(drawings)   # list[(Polygon, color)]
        recon        = reconstruct_closed_polygons(drawings)    # list[Polygon]
        recon_pairs  = [(p, None) for p in recon]
        filtered     = filter_slab_candidates(filled_pairs + recon_pairs, page)

        filled_polys = [p for p, _ in filled_pairs]            # strip color for visualizer

        page_debug = {}
        for step_fn, key, step_args in [
            (save_step1_raw_paths, "step1", (page, drawings,             f"{debug_base}_step1_raw.png")),
            (save_step2_polygons,  "step2", (page, filled_polys + recon, f"{debug_base}_step2_polys.png")),
            (save_step3_filtered,  "step3", (page, filtered,             f"{debug_base}_step3_filtered.png")),
            (save_step4_labeled,   "step4", (page, slab_regions,   f"{debug_base}_step4_labeled.png")),
            (save_step5_final,     "step5", (page, slab_regions,   f"{debug_base}_step5_final.png")),
        ]:
            try:
                page_debug[key] = step_fn(*step_args)
            except Exception:
                pass

        return page_idx, slab_regions, page_debug
    finally:
        doc.close()


def _render_step3_results():
    """Display Step 3 results and navigation buttons (reads from session_state)."""
    selected_pages = st.session_state["selected_pages"]
    results = st.session_state["slab_results"]
    debug_imgs = st.session_state["debug_images"]
    all_slabs = st.session_state["final_slabs"]
    total = len(all_slabs)
    n = len(selected_pages)

    st.markdown(
        f'<div class="success-box">✅ Complete — detected <b>{total} slab{"s" if total != 1 else ""}</b> '
        f'across {n} page{"s" if n != 1 else ""}.</div>', unsafe_allow_html=True,
    )

    # Log file download
    log_path = st.session_state.get("log_path")
    if log_path and Path(log_path).exists():
        with open(log_path, "rb") as lf:
            st.download_button(
                "📋 Download Log File (for debugging)",
                lf.read(),
                file_name=Path(log_path).name,
                mime="text/plain",
                key="dl_log_step3",
            )

    # Debug image viewer
    st.markdown("---")
    st.markdown("#### 📸 Debug Images — processing steps")
    for page_idx in selected_pages:
        page_imgs = debug_imgs.get(page_idx, {})
        n_slabs = len(results.get(page_idx, []))
        with st.expander(f"📄 Page {page_idx + 1} — {n_slabs} slab{'s' if n_slabs != 1 else ''}", expanded=(n_slabs > 0)):
            tabs = st.tabs(["① Raw Paths", "② Polygons", "③ Filtered", "④ Labeled", "⑤ Final"])
            step_keys = ["step1", "step2", "step3", "step4", "step5"]
            for tab, key in zip(tabs, step_keys):
                with tab:
                    img_path = page_imgs.get(key)
                    if img_path and Path(img_path).exists():
                        st.image(img_path, use_container_width=True)
                        with open(img_path, "rb") as f:
                            st.download_button(
                                f"⬇ Download {key}.png", f.read(),
                                file_name=Path(img_path).name, mime="image/png",
                                key=f"dl_{page_idx}_{key}",
                            )
                    else:
                        st.info("Image not available.")

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            # Clear cache so re-entering step 3 will re-process with any new settings
            st.session_state["slab_results"] = {}
            st.session_state["step"] = 2
            _rerun()
    with col_next:
        if total > 0:
            if st.button("Next: Review →", type="primary", use_container_width=True):
                st.session_state["step"] = 4
                _rerun()
        else:
            st.markdown(
                '<div class="warn-box">⚠️ No slabs found. Try: adjusting scale, '
                'selecting different pages, or reviewing debug images ①②③.</div>',
                unsafe_allow_html=True,
            )


# ── STEP 3: Detect Slabs ────────────────────────────────────────────────────────
def step3_detect():
    st.markdown('<div class="step-header">🔍 Step 3 — Slab Detection</div>',
                unsafe_allow_html=True)

    selected_pages = st.session_state["selected_pages"]
    scale = st.session_state["scale"] or 100
    pdf_path = st.session_state["pdf_path"]

    # Smart detect override: use only canonical/supplemental pages if confirmed
    smart_result = st.session_state.get("smart_detect_result")
    if smart_result and st.session_state.get("smart_detect_done"):
        pages_to_process = smart_result.pages_to_process
        n_floors = smart_result.floor_count
        n_skipped = len(smart_result.skipped_pages)
        st.markdown(
            f'<div class="info-box">🧠 Smart mode — {n_floors} floors / '
            f'{len(pages_to_process)} pages (skipping {n_skipped} duplicate pages)</div>',
            unsafe_allow_html=True,
        )
    else:
        pages_to_process = selected_pages

    # BUG 0 FIX: Cache guard — skip re-processing if all pages already done
    existing = st.session_state.get("slab_results", {})
    if existing and all(idx in existing for idx in pages_to_process):
        _render_step3_results()
        return

    # Setup logger for this run
    from src.pipeline_logger import setup_logger, log_session_start, log_summary
    _, log_path = setup_logger(OUTPUT_DIR)
    st.session_state["log_path"] = str(log_path)
    log_session_start(Path(pdf_path).name, selected_pages, scale)

    # Parallel processing with ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(pages_to_process)
    progress = st.progress(0, text=f"Processing {n} pages in parallel...")
    status_placeholder = st.empty()

    worker_args = [
        (pdf_path, page_idx, scale, str(sess_debug_dir / f"p{page_idx + 1:02d}"))
        for page_idx in pages_to_process
    ]

    results = {}
    debug_imgs = {}
    done_count = 0

    with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
        futures = {executor.submit(_process_page_worker, args): args[1] for args in worker_args}
        for future in as_completed(futures):
            page_idx = futures[future]
            try:
                p_idx, slab_regions, page_debug = future.result()
                results[p_idx] = slab_regions
                debug_imgs[p_idx] = page_debug
            except Exception as e:
                results[page_idx] = []
                debug_imgs[page_idx] = {}
                status_placeholder.markdown(
                    f'<div class="warn-box">⚠️ Error on page {page_idx + 1}: {e}</div>',
                    unsafe_allow_html=True,
                )
            done_count += 1
            n_found = len(results.get(page_idx, []))
            progress.progress(
                done_count / n,
                text=f"Page {page_idx + 1}: {n_found} slab{'s' if n_found != 1 else ''} — ({done_count}/{n} pages done)",
            )

    # Update session state only after all workers finish (thread-safe)
    st.session_state["slab_results"] = results
    st.session_state["debug_images"] = debug_imgs

    # ── Vision Refinement (Stage 4) ─────────────────────────────────────────────
    import fitz as _fitz
    from src.vision_refiner import get_vision_client, refine_page_slabs
    backend = st.session_state.get("vision_backend", "gemini")
    try:
        v_client, v_model = get_vision_client(backend)
        doc_v = _fitz.open(pdf_path)
        v_prog = st.progress(0, text="Vision Refinement (Stage 4)...")
        pages_with_slabs = [(idx, slabs) for idx, slabs in results.items() if slabs]
        for vi, (page_idx, page_slabs) in enumerate(pages_with_slabs):
            results[page_idx] = refine_page_slabs(
                page_slabs, doc_v[page_idx], v_client, v_model, backend
            )
            v_prog.progress((vi + 1) / max(len(pages_with_slabs), 1),
                            text=f"Vision: page {page_idx + 1} done ({vi+1}/{len(pages_with_slabs)})")
        doc_v.close()
        v_prog.empty()
        st.session_state["slab_results"] = results
    except Exception as ve:
        st.warning(f"Vision Refinement failed: {ve} — using standard detection results")

    all_slabs = [s for page_slabs in results.values() for s in page_slabs]

    # ── Column & Foundation Detection ───────────────────────────────────────────
    try:
        from src.column_analyzer import analyze_columns_and_foundations
        from src.column_detector import (
            detect_columns_on_page, detect_foundations_on_page,
            assign_columns_to_regions,
        )
        import fitz as _fitz2

        ai_floor_res = st.session_state.get("ai_floor_result")
        c_prog = st.progress(0, text="Column & Foundation Census (Gemini)...")
        census = analyze_columns_and_foundations(
            pdf_path, pages_to_process, ai_floor_res or {}
        )
        st.session_state["column_census"] = census
        c_prog.progress(0.4, text="Detecting column positions (vector)...")

        footing_pages_1idx = set(census.get("footing_plan_pages", []))
        col_types = census.get("column_types", {})
        fdn_types = census.get("foundation_types", {})

        _page_job_map: dict = {}
        for _bldg in census.get("buildings", []):
            for _floor in _bldg.get("floors", []):
                for _pg1 in _floor.get("slab_plan_pages", []):
                    _idx = _pg1 - 1
                    _relevant = {
                        sym: col_types[sym]
                        for sym in _floor.get("columns", {})
                        if sym in col_types and col_types[sym].get("width_mm") is not None
                    }
                    if _idx not in _page_job_map:
                        _page_job_map[_idx] = {
                            "building": _bldg["name"],
                            "level": _floor["level_name"],
                            "col_types": {},
                        }
                    _page_job_map[_idx]["col_types"].update(_relevant)

        doc_c = _fitz2.open(pdf_path)
        all_columns, all_foundations = [], []
        for page_idx, job in sorted(_page_job_map.items()):
            page_c = doc_c[page_idx]
            all_columns.extend(
                detect_columns_on_page(
                    page_c, job["col_types"], scale, page_idx,
                    building=job["building"], level=job["level"],
                )
            )
            if (page_idx + 1) in footing_pages_1idx:
                all_foundations.extend(
                    detect_foundations_on_page(page_c, fdn_types, scale, page_idx)
                )
        doc_c.close()

        all_columns = assign_columns_to_regions(all_columns, census, ai_floor_res)
        st.session_state["column_regions"]     = all_columns
        st.session_state["foundation_regions"] = all_foundations
        c_prog.empty()

        # Per-floor column / foundation summary
        st.markdown("#### 🏛️ Column & Foundation Schedule")
        for _bldg in census.get("buildings", []):
            bldg_rows = []
            for _floor in _bldg.get("floors", []):
                for sym, count in _floor.get("columns", {}).items():
                    if sym == "UNSCHEDULED_COLUMN":
                        continue
                    info = col_types.get(sym, {})
                    bldg_rows.append({
                        "Floor": _floor.get("level_name", ""),
                        "Symbol": sym,
                        "Width (mm)": info.get("width_mm", "?"),
                        "Depth (mm)": info.get("depth_mm", "?"),
                        "Count": count,
                    })
            if bldg_rows:
                st.markdown(f"**{_bldg['name']} — Columns**")
                st.dataframe(pd.DataFrame(bldg_rows), use_container_width=True, hide_index=True)

        fdn_types_info = census.get("foundation_types", {})
        if fdn_types_info:
            fdn_rows = [
                {
                    "Symbol": sym,
                    "Type": info.get("type", "pad"),
                    "Width (mm)": info.get("width_mm", "?"),
                    "Depth (mm)": info.get("depth_mm", "?"),
                    "Depth below GL (mm)": info.get("depth_below_gl_mm", "?"),
                }
                for sym, info in fdn_types_info.items()
            ]
            st.markdown("**Foundations**")
            st.dataframe(pd.DataFrame(fdn_rows), use_container_width=True, hide_index=True)

    except Exception as ce:
        st.warning(f"Column & Foundation detection failed: {ce}")

    # AI floor merge: combine all polygons from the same Gemini floor group.
    # Handles: (1) cross-page duplicates (same polygon on pages 9+10 → union = one),
    # (2) zone splits (A+B → full floor shape), (3) non-rectangular/L-shape floors.
    ai_floor_result = st.session_state.get("ai_floor_result")
    merge_msg = ""
    if ai_floor_result and st.session_state.get("smart_detect_done"):
        before_merge = len(all_slabs)
        all_slabs = _merge_slabs_by_ai_floors(all_slabs, ai_floor_result)
        merge_msg = f"✅ AI merge: {before_merge} raw slabs → {len(all_slabs)} floor slab(s)"

    st.session_state["final_slabs"] = all_slabs
    status_placeholder.empty()
    if merge_msg:
        st.markdown(
            f'<div class="success-box">{merge_msg}</div>', unsafe_allow_html=True
        )

    unique_ffls = len({s.ffl_m for s in all_slabs if s.ffl_m is not None})
    log_summary(len(all_slabs), n, unique_ffls)

    _render_step3_results()


# ── STEP 4: Review ──────────────────────────────────────────────────────────────
def step4_review():
    st.markdown('<div class="step-header">📋 Step 4 — Review & Edit</div>',
                unsafe_allow_html=True)

    all_slabs = st.session_state["final_slabs"]
    debug_imgs = st.session_state["debug_images"]
    selected_pages = st.session_state["selected_pages"]

    if not all_slabs:
        st.markdown('<div class="warn-box">⚠️ No slabs to review. Go back to step 3.</div>',
                    unsafe_allow_html=True)
        if st.button("← Back"):
            st.session_state["step"] = 3
            _rerun()
        return

    total_area = sum(s.area_m2 for s in all_slabs if s.area_m2 > 0)
    ffls = [s.ffl_m for s in all_slabs if s.ffl_m is not None]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Slabs", len(all_slabs))
    c2.metric("Total Area", f"{total_area:.1f} m²")
    c3.metric("Min FFL", f"{min(ffls):.3f}m" if ffls else "N/A")
    c4.metric("Max FFL", f"{max(ffls):.3f}m" if ffls else "N/A")

    st.markdown("---")
    st.markdown("#### ✏️ Edit Slab Data")
    st.markdown(
        '<div class="info-box">Double-click a cell to edit Label or FFL. '
        'Thickness is fixed at 200mm.</div>', unsafe_allow_html=True,
    )

    df_data = []
    for slab in all_slabs:
        df_data.append({
            "ID": slab.id,
            "Label": slab.label,
            "Page": slab.page_index + 1,
            "FFL (m)": round(slab.ffl_m, 3) if slab.ffl_m is not None else None,
            "Thickness (mm)": SLAB_THICKNESS_MM,
            "Area (m²)": round(slab.area_m2, 2) if slab.area_m2 > 0 else None,
            "Source": slab.source,
        })

    df = pd.DataFrame(df_data)
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "Label": st.column_config.TextColumn("Label", width="medium"),
            "Page": st.column_config.NumberColumn("Page", disabled=True, width="small"),
            "FFL (m)": st.column_config.NumberColumn("FFL (m)", format="%.3f", width="medium"),
            "Thickness (mm)": st.column_config.NumberColumn("Thickness (mm)", disabled=True, width="small"),
            "Area (m²)": st.column_config.NumberColumn("Area (m²)", format="%.2f", width="medium"),
            "Source": st.column_config.TextColumn("Source", disabled=True, width="small"),
        },
        key="slab_editor",
        height=min(400, 60 + len(all_slabs) * 38),
    )

    # Apply edits
    for i, row in edited_df.iterrows():
        if i < len(all_slabs):
            if pd.notna(row["Label"]):
                all_slabs[i].label = str(row["Label"])
            if pd.notna(row["FFL (m)"]):
                all_slabs[i].ffl_m = float(row["FFL (m)"])
                all_slabs[i].ffl_mm = float(row["FFL (m)"]) * 1000
    st.session_state["final_slabs"] = all_slabs

    # Final images
    st.markdown("---")
    st.markdown("#### 🖼️ Final Result Images")
    for page_idx in selected_pages:
        img_path = debug_imgs.get(page_idx, {}).get("step5")
        if img_path and Path(img_path).exists():
            with st.expander(f"Page {page_idx+1}", expanded=True):
                st.image(img_path, use_container_width=True)

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 3
            _rerun()
    with col_next:
        if st.button("Next: Generate 3D Model →", type="primary", use_container_width=True):
            st.session_state["step"] = 5
            _rerun()


# ── STEP 5: Generate ────────────────────────────────────────────────────────────
def step5_generate():
    st.markdown('<div class="step-header">⚙️ Step 5 — Generate SketchUp 3D Model</div>',
                unsafe_allow_html=True)

    all_slabs = st.session_state["final_slabs"]
    if not all_slabs:
        st.error("No slabs to export. Go back to step 3.")
        return

    from src.model_builder import generate_ruby_script, generate_slab_csv

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruby_path = str(OUTPUT_DIR / f"slabs_{ts}.rb")
    csv_path  = str(OUTPUT_DIR / f"slabs_{ts}.csv")

    with st.spinner("Generating Ruby script for SketchUp..."):
        ruby_content = generate_ruby_script(all_slabs, ruby_path)
        generate_slab_csv(all_slabs, csv_path)

    st.session_state["ruby_script"] = ruby_content
    st.session_state["ruby_path"]   = ruby_path
    st.session_state["csv_path"]    = csv_path

    c1, c2, c3 = st.columns(3)
    c1.metric("Slabs in script", len(all_slabs))
    c2.metric("Script size", f"{len(ruby_content)/1024:.1f} KB")
    c3.metric("Thickness", "200 mm (fixed)")

    st.markdown("---")
    st.markdown("#### ⬇️ Download Files")
    col_rb, col_csv = st.columns(2)
    with col_rb:
        st.download_button(
            "📥 Download .rb Script (SketchUp Ruby)",
            ruby_content.encode("utf-8"),
            file_name=Path(ruby_path).name,
            mime="text/plain",
            use_container_width=True,
        )
    with col_csv:
        with open(csv_path, "rb") as f:
            st.download_button(
                "📊 Download Slab Data (.csv)",
                f.read(),
                file_name=Path(csv_path).name,
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("#### 👁️ Preview Ruby Script")
    with st.expander("View code (first 80 lines)", expanded=False):
        preview = "\n".join(ruby_content.split("\n")[:80])
        st.code(preview, language="ruby")

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 4
            _rerun()
    with col_next:
        if st.button("Next: SketchUp Guide →", type="primary", use_container_width=True):
            st.session_state["step"] = 6
            _rerun()


# ── STEP 6: Export / Instructions ──────────────────────────────────────────────
def step6_done():
    st.markdown('<div class="step-header">🎉 Step 6 — Import into SketchUp</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="success-box">🎉 <b>Done!</b> Follow the instructions below '
        'to import slabs into SketchUp 2026.</div>', unsafe_allow_html=True,
    )

    st.markdown("""
---
### 🚀 How to import into SketchUp 2026

| Step | Action |
|------|--------|
| 1 | Open **SketchUp** → `Window` → `Ruby Console` |
| 2 | **Download** the `.rb` file from Step 5 |
| 3 | Open the `.rb` file in Notepad → **Copy all** |
| 4 | **Paste** into Ruby Console → press **Enter** |
| 5 | Press **Z** (Zoom Extents) to view the full model |

---
### 📦 What you get in SketchUp
- Each slab = a 3D solid **200mm** thick
- Each floor = its own **Layer** (named by FFL)
- Each floor has a **unique material colour**
- Model units: **mm** (set automatically)

---
### 🔍 Verify accuracy
1. Use **Tape Measure** (`T`) to measure slab dimensions
2. Compare with annotations on the PDF
3. If incorrect → go back to Step 2 and adjust **Scale**

---
### 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Slabs not visible | Check Ruby Console for errors |
| Wrong dimensions | Adjust Scale in Step 2 |
| Wrong elevation | Edit FFL in Step 4 |
| Missing slabs | Select additional pages in Step 2 |
| Overlapping slabs | Check debug image Step ③ |
""")

    ruby_content = st.session_state.get("ruby_script", "")
    if ruby_content:
        st.markdown("---")
        with st.expander("📋 Quick copy script"):
            st.code(ruby_content, language="ruby")
            st.download_button(
                "⬇️ Re-download .rb",
                ruby_content.encode("utf-8"),
                file_name="slabs.rb",
                mime="text/plain",
            )

    st.markdown("---")
    st.markdown(
        '<div class="info-box">💡 <b>Next phase:</b> Once slabs are confirmed, '
        'columns, beams, walls and reinforcement will be added.</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄 Process another PDF", use_container_width=True, type="primary"):
        reset_session()
        _rerun()


# ── Main Router ─────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        '<h1 style="font-size:2.2rem;font-weight:800;color:#E3F2FD;margin-bottom:0;">'
        '🏗️ Feeldx Structural Slab Extractor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#546E7A;font-size:1.05rem;margin-top:2px;">'
        'Phase 1 — PDF → SketchUp 3D Slabs &nbsp;|&nbsp; '
        'Australian Structural Drawings &nbsp;|&nbsp; '
        'Thickness: 200mm fixed</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    step = st.session_state["step"]
    dispatch = {1: step1_upload, 2: step2_select_pages, 3: step3_detect,
                4: step4_review, 5: step5_generate, 6: step6_done}
    dispatch.get(step, step1_upload)()


if __name__ == "__main__":
    main()
