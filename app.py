"""
Feeldx Slab Extractor — Streamlit App
Phase 1: PDF Structural → SketchUp 3D Slabs
"""

import io
import os
import uuid
import math
import json
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
VISION_MAX_WORKERS = 6


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
        "slab_debug_stats": {},
        "final_slabs": [],
        "ruby_script": None,
        "ruby_path": None,
        "csv_path": None,
        "log_path": None,
        "smart_detect_result": None,
        "smart_detect_done": False,
        "ai_floor_result": None,
        "ai_floor_output_path": None,
        "document_intelligence": None,
        "document_intelligence_path": None,
        "document_intelligence_raw_path": None,
        "document_intelligence_parse_report_path": None,
        "column_validation": {},
        "storey_height_overrides": {},
        "storey_height_report": [],
        "storey_height_by_page_mm": {},
        "building_model_registry": {},
        "building_polygon_image": None,
        "floor_alignment_report": [],
        "floor_alignment_offsets": {},
        "floor_alignment_preview": None,
        "legend_semantics": None,
        "legend_semantics_path": None,
        "legend_semantics_raw_path": None,
        "legend_semantics_report_path": None,
        "legend_semantics_cache_key": None,
        "slab_semantic_previews": {},
        "slab_semantic_surface_images": {},
        "slab_semantic_boundary_images": {},
        "slab_semantic_cut_images": {},
        "wall_regions": [],
        "wall_polygon_images": {},
        "semantic_wall_images": {},
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
        st.session_state["document_intelligence"] = None
        st.session_state["document_intelligence_path"] = None
        st.session_state["document_intelligence_raw_path"] = None
        st.session_state["document_intelligence_parse_report_path"] = None
        st.session_state["column_validation"] = {}
        st.session_state["storey_height_overrides"] = {}
        st.session_state["storey_height_report"] = []
        st.session_state["storey_height_by_page_mm"] = {}
        st.session_state["building_model_registry"] = {}
        st.session_state["building_polygon_image"] = None
        st.session_state["floor_alignment_report"] = []
        st.session_state["floor_alignment_offsets"] = {}
        st.session_state["floor_alignment_preview"] = None
        st.session_state["legend_semantics"] = None
        st.session_state["legend_semantics_path"] = None
        st.session_state["legend_semantics_raw_path"] = None
        st.session_state["legend_semantics_report_path"] = None
        st.session_state["legend_semantics_cache_key"] = None
        st.session_state["slab_semantic_previews"] = {}
        st.session_state["slab_semantic_surface_images"] = {}
        st.session_state["slab_semantic_boundary_images"] = {}
        st.session_state["slab_semantic_cut_images"] = {}
        st.session_state["wall_regions"] = []
        st.session_state["wall_polygon_images"] = {}
        st.session_state["semantic_wall_images"] = {}

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

        st.markdown("---")
        if st.button("Next: Select Pages →", type="primary", use_container_width=True):
            st.session_state["step"] = 2
            _rerun()


# ── STEP 2: Select Pages ────────────────────────────────────────────────────────
def step2_select_pages():
    st.markdown('<div class="step-header">🗂️ Step 2 — Select Floor Plan Pages</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Floor plan pages have been auto-detected. '
        'Check/uncheck to customise. Set the correct drawing scale for accurate 3D dimensions.'
        '</div>', unsafe_allow_html=True,
    )

    page_infos = st.session_state["page_infos"]
    if not page_infos:
        st.error("No page information found.")
        return

    import fitz
    from src.pdf_processor import load_pdf, get_page_thumbnail

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
                f'<div class="success-box" style="margin-top:28px;">✅ Auto-detect scale: <b>1:{scale_auto}</b></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warn-box" style="margin-top:28px;">⚠️ Auto-detect failed — '
                'enter manually. Check the PDF title block.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 📄 Select Pages to Process")

    doc = load_pdf(st.session_state["pdf_path"])
    floor_plan_idxs = {p["index"] for p in page_infos if p["is_floor_plan"]}
    current_selected = set(
        st.session_state.get("selected_pages") or list(floor_plan_idxs)
    )

    new_selected = set()
    cols_per_row = 5
    rows = [page_infos[i:i+cols_per_row] for i in range(0, len(page_infos), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, page_info in zip(cols, row):
            with col:
                try:
                    thumb = get_page_thumbnail(doc[page_info["index"]], dpi=50)
                    st.image(thumb, use_container_width=True)
                except Exception:
                    st.markdown("_(no preview)_")

                badge = "🔵" if page_info["is_floor_plan"] else "⬜"
                checked = st.checkbox(
                    f'{badge} P{page_info["index"]+1}',
                    value=page_info["index"] in current_selected,
                    key=f"pg_{page_info['index']}",
                )
                title = page_info.get("title", "")
                st.caption(title[:22] + "…" if len(title) > 22 else title or f"Page {page_info['index']+1}")
                if checked:
                    new_selected.add(page_info["index"])

    doc.close()
    st.session_state["selected_pages"] = sorted(new_selected)

    st.markdown("---")
    n_sel = len(new_selected)
    st.markdown(
        f'<div class="{"success-box" if n_sel > 0 else "warn-box"}">'
        f'{"✅" if n_sel > 0 else "⚠️"} <b>{n_sel} page{"s" if n_sel != 1 else ""}</b> selected for analysis.'
        '</div>', unsafe_allow_html=True,
    )

    # ── Smart Floor Detection (AI + keyword fallback) ───────────────────────
    if n_sel > 0:
        st.markdown("---")
        st.markdown("#### 🧠 AI Floor Detection")
        st.markdown(
            '<div class="info-box">Gemini reads all PDF text, identifies buildings/floors/FFLs, '
            'and selects the correct pages to process (skipping sections, details, elevations).</div>',
            unsafe_allow_html=True,
        )

        smart_result = st.session_state.get("smart_detect_result")

        col_ai, col_kw, col_reset = st.columns([2, 2, 1])
        with col_ai:
            if st.button("🤖 AI Analysis (Gemini)", use_container_width=True, type="primary"):
                from src.ai_floor_analyzer import analyze_floor_structure
                from src.floor_detector import FloorDetectResult, FloorGroup
                with st.spinner(f"Gemini reading {n_sel} pages (~5–10 seconds)..."):
                    try:
                        ai_result, ai_path = analyze_floor_structure(
                            st.session_state["pdf_path"],
                            sorted(new_selected),
                        )
                        st.session_state["ai_floor_result"] = ai_result
                        st.session_state["ai_floor_output_path"] = ai_path
                        # Convert AI result → FloorDetectResult for step3
                        smart_res = _ai_result_to_floor_detect(ai_result, sorted(new_selected))
                        st.session_state["smart_detect_result"] = smart_res
                        st.session_state["smart_detect_done"] = False
                        # Auto-activate Vision + Column + Foundation detection
                        st.session_state["vision_refine_enabled"] = True
                        st.session_state["column_detect_enabled"] = True
                        st.session_state["foundation_detect_enabled"] = True
                    except Exception as e:
                        st.error(f"Gemini error: {e}")
                _rerun()
        with col_kw:
            if st.button("🔍 Keyword detection (fast)", use_container_width=True):
                from src.floor_detector import detect_unique_floors
                with st.spinner(f"Scanning {n_sel} pages..."):
                    result = detect_unique_floors(
                        st.session_state["pdf_path"],
                        sorted(new_selected),
                    )
                st.session_state["smart_detect_result"] = result
                st.session_state["ai_floor_result"] = None
                st.session_state["smart_detect_done"] = False
                # Auto-activate Column + Foundation detection (no Vision for keyword mode)
                st.session_state["column_detect_enabled"] = True
                st.session_state["foundation_detect_enabled"] = True
                _rerun()
        with col_reset:
            if smart_result and st.button("↺ Reset", use_container_width=True):
                st.session_state["smart_detect_result"] = None
                st.session_state["smart_detect_done"] = False
                st.session_state["ai_floor_result"] = None
                st.session_state["ai_floor_output_path"] = None
                # Deactivate Vision + Column + Foundation on reset
                st.session_state["vision_refine_enabled"] = False
                st.session_state["column_detect_enabled"] = False
                st.session_state["foundation_detect_enabled"] = False
                _rerun()

        # ── Show AI raw JSON for review ──────────────────────────────────
        ai_result = st.session_state.get("ai_floor_result")
        ai_path   = st.session_state.get("ai_floor_output_path")
        if ai_result:
            with st.expander("🔍 View Gemini output (raw JSON)"):
                st.json(ai_result)
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

        if smart_result:
            basis_labels = {
                "ffl": "FFL values ✅ (keyword)",
                "title": "Title keywords ⚠️ (keyword)",
                "all_pages": "Keyword: insufficient signal — processing all pages",
                "ai_gemini": "Gemini AI ✅ (highest confidence)",
                "page_type": "Page classification ✅ (keyword)",
            }
            basis_txt = basis_labels.get(smart_result.detection_basis, smart_result.detection_basis)

            if smart_result.detection_basis == "all_pages":
                st.markdown(
                    f'<div class="warn-box">⚠️ {smart_result.warnings[0] if smart_result.warnings else ""}'
                    f'<br>Basis: {basis_txt}</div>',
                    unsafe_allow_html=True,
                )
            else:
                n_proc = len(smart_result.pages_to_process)
                n_skip = len(smart_result.skipped_pages)
                st.markdown(
                    f'<div class="success-box">✅ Detected <b>{smart_result.floor_count} floors</b> '
                    f'— processing <b>{n_proc} pages</b>, skipping <b>{n_skip} pages</b>'
                    f'<br><small>Basis: {basis_txt}</small></div>',
                    unsafe_allow_html=True,
                )

                # Results table
                import pandas as pd
                rows = []
                for g in smart_result.groups:
                    rows.append({
                        "Floor": g.floor_label,
                        "Canonical": f"Page {g.canonical_page + 1}",
                        "Supplement": ", ".join(f"P{p+1}" for p in g.supplemental_pages) or "—",
                        "Skipped": ", ".join(f"P{p+1}" for p in g.skipped_pages) or "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                for w in (smart_result.warnings or []):
                    if w:
                        st.warning(w)

            # Confirm buttons
            st.markdown("")
            ca, cb = st.columns(2)
            with ca:
                label = (f"⚡ Process {len(smart_result.pages_to_process)} pages (smart)"
                         if smart_result.detection_basis != "all_pages"
                         else f"Continue with all {n_sel} pages")
                if st.button(label, type="primary", use_container_width=True):
                    st.session_state["smart_detect_done"] = True
                    st.session_state["slab_results"] = {}   # clear cache → re-process
                    st.session_state["step"] = 3
                    _rerun()
            with cb:
                if smart_result.detection_basis != "all_pages":
                    if st.button(f"Use all {n_sel} pages (skip smart mode)", use_container_width=True):
                        st.session_state["smart_detect_done"] = False
                        st.session_state["slab_results"] = {}
                        st.session_state["step"] = 3
                        _rerun()

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 1
            _rerun()
    with col_next:
        if n_sel > 0:
            if not st.session_state.get("smart_detect_result"):
                if st.button("Next: Detect Slabs →", type="primary", use_container_width=True):
                    st.session_state["slab_results"] = {}
                    st.session_state["step"] = 3
                    _rerun()
        else:
            st.warning("Select at least 1 page.")


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

def _assign_ffl_from_elevation(all_slabs: list, pdf_path: str, page_infos: list) -> None:
    """
    Assign ffl_m/ffl_mm to slabs that have no FFL annotation, using storey heights
    derived from the elevation/section drawings in the PDF.

    Level 01 = 0.000m (project datum). Each higher level adds the storey height.
    Only patches slabs where ffl_m is currently None.
    """
    import re, fitz as _fitz
    from src.pdf_processor import extract_storey_heights_from_elevation

    # Skip if all slabs already have FFL
    if all(s.ffl_m is not None for s in all_slabs):
        return

    try:
        doc = _fitz.open(pdf_path)
        storey_heights = extract_storey_heights_from_elevation(doc)
        doc.close()
    except Exception:
        return

    if not storey_heights:
        return

    # Compute cumulative FFL map: level_num → ffl_m (Level 01 = 0.000m datum)
    level_nums = sorted(storey_heights.keys())
    ffl_map: dict[int, float] = {level_nums[0]: 0.0}
    for lvl in level_nums:
        ffl_map[lvl + 1] = round(ffl_map.get(lvl, 0.0) + storey_heights[lvl], 4)

    # Map page_index → level_num from page titles ("LEVEL 01 OUTLINE PLAN" → 1)
    _lvl_re = re.compile(r"LEVEL\s*0*(\d+)", re.IGNORECASE)
    page_level: dict[int, int] = {}
    for pi in page_infos:
        m = _lvl_re.search(pi.get("title", "") or pi.get("label", ""))
        if m:
            page_level[pi["index"]] = int(m.group(1))

    for slab in all_slabs:
        if slab.ffl_m is not None:
            continue
        lvl_num = page_level.get(slab.page_index)
        if lvl_num and lvl_num in ffl_map:
            slab.ffl_m = ffl_map[lvl_num]
            slab.ffl_mm = ffl_map[lvl_num] * 1000.0

    import streamlit as _st
    _st.session_state["storey_heights_map"] = storey_heights


def _find_page_info(page_infos: list, page_index: int) -> dict:
    for info in page_infos or []:
        if info.get("index") == page_index:
            return info
    return {}


def _height_sources_from_intelligence(intel: dict) -> tuple[dict, list]:
    """Return explicit page-level elevations and useful height evidence from Gemini."""
    explicit_by_page: dict[int, list] = {}
    evidence_rows = []
    for src in (intel or {}).get("height_sources", []) or []:
        page_1 = src.get("page")
        page_idx = int(page_1) - 1 if isinstance(page_1, int) and page_1 > 0 else None
        row = {
            "Type": src.get("type", "unknown"),
            "Page": page_1,
            "Level": src.get("level"),
            "Elevation (m)": src.get("elevation_m"),
            "Height (mm)": src.get("height_mm"),
            "Action": src.get("recommended_action"),
            "Confidence": src.get("confidence"),
            "Text": src.get("source_text", ""),
        }
        evidence_rows.append(row)
        if page_idx is not None and src.get("elevation_m") is not None:
            explicit_by_page.setdefault(page_idx, []).append(src)
    return explicit_by_page, evidence_rows


def _explicit_storey_heights_from_intelligence(intel: dict) -> dict[int, dict]:
    by_page: dict[int, dict] = {}
    for item in (intel or {}).get("storey_heights", []) or []:
        h = item.get("height_mm")
        if h is None:
            continue
        try:
            height_mm = float(h)
        except (TypeError, ValueError):
            continue
        for page_1 in item.get("source_pages", []) or []:
            if isinstance(page_1, int) and page_1 > 0:
                by_page[page_1 - 1] = {
                    "height_mm": height_mm,
                    "source": item.get("source", "explicit_text"),
                    "confidence": item.get("confidence", 0.85),
                }
    return by_page


def _build_storey_height_report(all_slabs: list, page_infos: list | None = None,
                                document_intelligence: dict | None = None) -> tuple[list, dict]:
    """
    Build one review row per slab-plan page.

    Priority:
      manual override -> explicit FFL/RL/EL text -> FFL difference/elevation inferred -> missing/default.
    """
    from src.model_builder import compute_storey_heights

    overrides = st.session_state.get("storey_height_overrides", {}) or {}
    explicit_by_page, _ = _height_sources_from_intelligence(document_intelligence or {})
    explicit_height_by_page = _explicit_storey_heights_from_intelligence(document_intelligence or {})
    computed_m = compute_storey_heights(all_slabs)
    measured_m = st.session_state.get("storey_heights_map", {}) or {}

    page_slabs: dict[int, list] = {}
    for slab in all_slabs:
        page_slabs.setdefault(slab.page_index, []).append(slab)

    rows = []
    height_by_page_mm: dict[int, float] = {}
    for page_idx in sorted(page_slabs):
        slabs = page_slabs[page_idx]
        info = _find_page_info(page_infos or [], page_idx)
        ffl_candidates = [s.ffl_m for s in slabs if s.ffl_m is not None]
        ffl_m = ffl_candidates[0] if ffl_candidates else None
        page_override = overrides.get(str(page_idx), {})
        if page_override.get("ffl_m") not in (None, ""):
            try:
                ffl_m = float(page_override["ffl_m"])
            except (TypeError, ValueError):
                pass

        height_mm = None
        source = "missing"
        status = "missing"
        confidence = 0.0
        warning = "No reliable height source. Default export fallback only."

        if page_override.get("height_mm") not in (None, ""):
            try:
                height_mm = float(page_override["height_mm"])
                source = "manual"
                status = "verified"
                confidence = 1.0
                warning = ""
            except (TypeError, ValueError):
                pass

        if height_mm is None and explicit_height_by_page.get(page_idx):
            item = explicit_height_by_page[page_idx]
            height_mm = item["height_mm"]
            source = item.get("source") or "explicit_text"
            status = "verified"
            confidence = float(item.get("confidence") or 0.85)
            warning = ""

        if height_mm is None and computed_m.get(page_idx) is not None:
            height_mm = float(computed_m[page_idx]) * 1000.0
            if explicit_by_page.get(page_idx) or info.get("ffl_values"):
                source = "explicit_ffl_difference"
                status = "verified"
                confidence = 0.9
                warning = ""
            elif measured_m:
                source = "elevation_measured"
                status = "inferred"
                confidence = 0.75
                warning = "Height inferred from elevation/section spacing. Review before final model."
            else:
                source = "ffl_difference"
                status = "inferred"
                confidence = 0.7
                warning = "Height computed from available FFL values. Review source drawings."

        if height_mm is not None and 2500 <= height_mm <= 8000:
            height_by_page_mm[page_idx] = height_mm
        elif height_mm is not None:
            status = "missing"
            warning = "Height outside sanity range 2500-8000mm. Not used for export."

        rows.append({
            "Page": page_idx + 1,
            "Floor / Level": slabs[0].label if slabs else info.get("title", ""),
            "FFL/RL/EL (m)": round(ffl_m, 3) if ffl_m is not None else None,
            "Storey Height (mm)": round(height_mm, 0) if height_mm is not None else None,
            "Source": source,
            "Status": status,
            "Confidence": confidence,
            "Warning": warning,
        })
    return rows, height_by_page_mm


def _render_storey_height_report(all_slabs: list, editable: bool = False) -> tuple[list, dict]:
    page_infos = st.session_state.get("page_infos", []) or []
    intel = st.session_state.get("document_intelligence") or {}
    rows, height_by_page_mm = _build_storey_height_report(all_slabs, page_infos, intel)
    st.session_state["storey_height_report"] = rows
    st.session_state["storey_height_by_page_mm"] = height_by_page_mm

    verified = sum(1 for r in rows if r["Status"] == "verified")
    inferred = sum(1 for r in rows if r["Status"] == "inferred")
    missing = sum(1 for r in rows if r["Status"] in ("missing", "default"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verified heights", verified)
    c2.metric("Inferred heights", inferred)
    c3.metric("Missing/default", missing)
    c4.metric("Usable height pages", len(height_by_page_mm))

    explicit_by_page, evidence_rows = _height_sources_from_intelligence(intel)
    if evidence_rows:
        with st.expander("Height evidence from Document Intelligence", expanded=False):
            st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

    df = pd.DataFrame(rows)
    if editable:
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Page": st.column_config.NumberColumn("Page", disabled=True, width="small"),
                "Floor / Level": st.column_config.TextColumn("Floor / Level", disabled=True, width="large"),
                "FFL/RL/EL (m)": st.column_config.NumberColumn("FFL/RL/EL (m)", format="%.3f", width="medium"),
                "Storey Height (mm)": st.column_config.NumberColumn("Storey Height (mm)", format="%.0f", width="medium"),
                "Source": st.column_config.TextColumn("Source", disabled=True, width="medium"),
                "Status": st.column_config.TextColumn("Status", disabled=True, width="small"),
                "Confidence": st.column_config.NumberColumn("Confidence", disabled=True, format="%.2f", width="small"),
                "Warning": st.column_config.TextColumn("Warning", disabled=True, width="large"),
            },
            key="storey_height_editor",
            height=min(420, 70 + max(len(rows), 1) * 38),
        )
        overrides = {}
        for _, row in edited.iterrows():
            page = int(row["Page"]) - 1
            original = next((r for r in rows if r["Page"] == int(row["Page"])), {})
            changed_ffl = row.get("FFL/RL/EL (m)") != original.get("FFL/RL/EL (m)")
            changed_h = row.get("Storey Height (mm)") != original.get("Storey Height (mm)")
            if changed_ffl or changed_h:
                overrides[str(page)] = {
                    "ffl_m": None if pd.isna(row.get("FFL/RL/EL (m)")) else float(row.get("FFL/RL/EL (m)")),
                    "height_mm": None if pd.isna(row.get("Storey Height (mm)")) else float(row.get("Storey Height (mm)")),
                }
        if overrides:
            st.session_state["storey_height_overrides"] = overrides
            rows, height_by_page_mm = _build_storey_height_report(all_slabs, page_infos, intel)
            st.session_state["storey_height_report"] = rows
            st.session_state["storey_height_by_page_mm"] = height_by_page_mm
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    if missing:
        st.warning("Some levels do not have verified/inferred height evidence. Export will use fallback only for those levels.")
    overrides = st.session_state.get("storey_height_overrides", {}) or {}
    for slab in all_slabs:
        ov = overrides.get(str(slab.page_index), {})
        if ov.get("ffl_m") not in (None, ""):
            try:
                slab.ffl_m = float(ov["ffl_m"])
                slab.ffl_mm = slab.ffl_m * 1000.0
            except (TypeError, ValueError):
                pass
    return rows, height_by_page_mm


def _semantic_mapping_ready() -> bool:
    """Only trust Gemini building/floor mapping when Document Intelligence parsed cleanly."""
    intel = st.session_state.get("document_intelligence")
    if not intel:
        return False
    metadata = intel.get("_metadata", {}) or {}
    status = intel.get("_parse_status") or metadata.get("parse_status")
    return status == "ok"


def _trusted_ai_floor_result():
    return st.session_state.get("ai_floor_result") if _semantic_mapping_ready() else None


def _is_reliable_visible_slab_fill(color, polygons: list, page) -> bool:
    """True only for visible material fills, not white/gray PDF masks."""
    if color is None or not polygons:
        return False
    if len(color) < 3:
        return False
    r, g, b = [float(v) for v in color[:3]]
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    chroma = max_c - min_c
    brightness = (r + g + b) / 3.0
    if brightness > 0.92:
        return False
    if brightness < 0.08:
        return False
    if chroma < 0.045:
        return False
    page_area = max(page.rect.width * page.rect.height, 1.0)
    fill_area = sum(p.area for p in polygons)
    if fill_area / page_area < 0.01:
        return False
    return True


def _refresh_building_registry(all_slabs: list, all_columns: list | None = None,
                               all_foundations: list | None = None) -> dict:
    from src.building_registry import build_building_registry
    from src.visualizer import save_building_footprints

    registry = build_building_registry(
        all_slabs,
        all_columns or st.session_state.get("column_regions", []),
        all_foundations or st.session_state.get("foundation_regions", []),
        _trusted_ai_floor_result(),
    )
    if st.session_state.get("ai_floor_result") and not _semantic_mapping_ready():
        registry.setdefault("warnings", []).append(
            "Building/floor mapping disabled because Document Intelligence parse_status is not ok."
        )
    st.session_state["building_model_registry"] = registry
    out = sess_debug_dir / "building_footprints.png"
    try:
        st.session_state["building_polygon_image"] = save_building_footprints(registry, str(out))
    except Exception as exc:
        st.session_state["building_polygon_image"] = None
        registry.setdefault("warnings", []).append(f"Building footprint preview failed: {exc}")
    return registry


def _apply_floor_alignment(all_slabs: list, all_columns: list | None = None,
                           all_foundations: list | None = None) -> tuple[list[dict], dict]:
    from src.floor_alignment import align_floors
    from src.visualizer import save_floor_alignment_preview

    rows, offsets = align_floors(
        all_slabs,
        all_columns or st.session_state.get("column_regions", []),
        all_foundations or st.session_state.get("foundation_regions", []),
        _trusted_ai_floor_result(),
    )
    st.session_state["floor_alignment_report"] = rows
    st.session_state["floor_alignment_offsets"] = offsets
    out = sess_debug_dir / "floor_alignment_report.png"
    try:
        st.session_state["floor_alignment_preview"] = save_floor_alignment_preview(rows, str(out))
    except Exception:
        st.session_state["floor_alignment_preview"] = None
    return rows, offsets


def _render_floor_alignment_report(expanded: bool = True) -> None:
    rows = st.session_state.get("floor_alignment_report") or []
    with st.expander("Floor Alignment Report", expanded=expanded):
        applied = sum(1 for r in rows if r.get("Applied"))
        warnings = sum(1 for r in rows if r.get("Warning") and r.get("Warning") != "reference floor")
        c1, c2, c3 = st.columns(3)
        c1.metric("Alignment rows", len(rows))
        c2.metric("Offsets applied", applied)
        c3.metric("Warnings", warnings)
        img_path = st.session_state.get("floor_alignment_preview")
        if img_path and Path(img_path).exists():
            st.image(img_path, use_container_width=True)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No alignment rows yet. Run Step 3 detection first.")


def _render_building_registry(registry: dict | None = None, expanded: bool = True) -> None:
    from src.building_registry import building_registry_rows

    registry = registry or st.session_state.get("building_model_registry") or {}
    rows = building_registry_rows(registry)
    with st.expander("Building Polygon Preview / Position Report", expanded=expanded):
        b1, b2, b3, b4 = st.columns(4)
        buildings = registry.get("buildings", {})
        b1.metric("Buildings", len(buildings))
        b2.metric("Slabs", sum(b.get("slab_count", 0) for b in buildings.values()))
        b3.metric("Columns", sum(b.get("column_count", 0) for b in buildings.values()))
        b4.metric("Foundations", sum(b.get("foundation_count", 0) for b in buildings.values()))

        img_path = st.session_state.get("building_polygon_image")
        if img_path and Path(img_path).exists():
            st.image(img_path, use_container_width=True)
            with open(img_path, "rb") as f:
                st.download_button(
                    "Download building footprint preview",
                    f.read(),
                    file_name=Path(img_path).name,
                    mime="image/png",
                    key="dl_building_footprints",
                )
        else:
            st.warning("No building footprint preview image available.")

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No building registry rows available.")

        for warning in registry.get("warnings", []) or []:
            st.warning(warning)


def _render_element_overlay_gallery(title: str, image_map: dict, empty_message: str,
                                    key_prefix: str) -> None:
    if not image_map:
        st.warning(empty_message)
        return
    for page_idx, img_path in sorted(image_map.items()):
        if not img_path or not Path(img_path).exists():
            continue
        with st.expander(f"{title} - Page {page_idx + 1}", expanded=True):
            st.image(img_path, use_container_width=True)
            with open(img_path, "rb") as f:
                st.download_button(
                    f"Download {title.lower()} p{page_idx + 1}",
                    f.read(),
                    file_name=Path(img_path).name,
                    mime="image/png",
                    key=f"{key_prefix}_{page_idx}",
                )


def _process_page_worker(args: tuple) -> tuple:
    """
    Worker for parallel page processing.
    Opens its own fitz document — fitz.Document is NOT thread-safe when shared.
    """
    if len(args) >= 5:
        pdf_path, page_idx, scale, debug_base, legend_semantics = args[:5]
    else:
        pdf_path, page_idx, scale, debug_base = args
        legend_semantics = None

    import fitz
    from src.pdf_processor import extract_text_blocks, extract_ffl_values, extract_slab_labels
    from src.slab_extractor import (
        extract_slabs_from_page, build_polygons_from_drawings,
        reconstruct_closed_polygons, filter_slab_candidates_structured, assign_labels,
    )
    from src.boundary_slab_extractor import extract_boundary_first_slabs
    from src.coordinate_mapper import transform_all_slabs
    from src.visualizer import (
        save_step1_raw_paths, save_step2_polygons, save_step3_filtered,
        save_step4_labeled, save_step5_final, save_gross_net_slab_debug,
        save_boundary_first_debug, save_wall_evidence_only,
        save_slab_candidates_only, save_wall_guided_final,
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
        extraction_result = filter_slab_candidates_structured(
            filled_pairs + recon_pairs,
            page,
            text_blocks=text_blocks,
            recover_slab_appendages=True,
            auto_cut_voids=True,
            cut_walls=False,
            min_void_confidence=0.75,
        )
        filtered = extraction_result.net_slabs
        reliable_visible_fill = _is_reliable_visible_slab_fill(
            extraction_result.dominant_fill,
            filtered,
            page,
        )
        has_fill = bool(slab_regions and reliable_visible_fill)
        extraction_mode = "fill_first" if reliable_visible_fill else "boundary_first"
        if reliable_visible_fill:
            mode_reason = "visible_dominant_fill"
            fill_confidence = 0.85
        elif extraction_result.dominant_fill is not None:
            mode_reason = "dominant_fill_not_visible_or_neutral"
            fill_confidence = 0.35
        else:
            mode_reason = "no_dominant_fill"
            fill_confidence = 0.25

        boundary_result = extract_boundary_first_slabs(
            page,
            drawings,
            text_blocks=text_blocks,
            legend_semantics=legend_semantics,
        )
        boundary_confidence = getattr(boundary_result, "confidence", 0.0)
        structural_debug = getattr(getattr(boundary_result, "structural_objects", None), "debug", {}) or {}
        fill_area = sum(p.area for p in filtered)
        boundary_area = sum(p.area for p in getattr(boundary_result, "final_regions", []) or [])
        use_boundary = (
            boundary_result.final_regions
            and (
                not reliable_visible_fill
                or not filtered
                or (boundary_confidence >= 0.70 and boundary_area > fill_area * 1.15)
            )
        )
        if use_boundary:
            filtered = boundary_result.final_regions
            extraction_mode = "evidence_guided_no_fill_boundary" if not reliable_visible_fill else "hybrid"
            mode_reason = boundary_result.mode_reason
            slab_regions = assign_labels(filtered, text_blocks, ffl_values, slab_labels)
            for i, r in enumerate(slab_regions):
                r.source = extraction_mode
                r.page_index = page.number
            slab_regions = transform_all_slabs(slab_regions, page, scale)
        elif reliable_visible_fill and boundary_result.final_regions:
            extraction_mode = "hybrid"
            mode_reason = "dominant_fill_with_boundary_evidence"

        slab_stats = {
            "gross_slab_count": len(extraction_result.gross_slabs),
            "net_slab_count": len(extraction_result.net_slabs),
            "final_slab_count": len(filtered),
            "gross_area_pdf": float(sum(p.area for p in extraction_result.gross_slabs)),
            "net_area_pdf": float(sum(p.area for p in extraction_result.net_slabs)),
            "final_area_pdf": float(sum(p.area for p in filtered)),
            "recovered_appendage_count": len(extraction_result.appendages),
            "void_candidate_count": len(extraction_result.void_candidates),
            "auto_cut_count": sum(1 for c in extraction_result.void_candidates if c.get("auto_cut")),
            "dominant_fill": extraction_result.dominant_fill,
            "reliable_visible_fill": reliable_visible_fill,
            "extraction_mode": extraction_mode,
            "mode_reason": mode_reason,
            "fill_confidence": fill_confidence,
            "boundary_confidence": boundary_confidence,
            "boundary_signature_count": boundary_result.debug.get("boundary_signature_count", 0),
            "boundary_evidence_count": boundary_result.debug.get("boundary_evidence_count", 0),
            "wall_count": structural_debug.get("walls", 0),
            "core_count": structural_debug.get("cores", 0),
            "stair_count": structural_debug.get("stairs", 0),
            "opening_count": structural_debug.get("openings", 0),
            "boundary_cut_count": structural_debug.get("cut_candidates", 0),
            "ignored_boundary_count": len(getattr(getattr(boundary_result, "structural_objects", None), "ignored_regions", []) or []),
            "boundary_debug": boundary_result.debug,
            "debug": extraction_result.debug,
        }

        filled_polys = [p for p, _ in filled_pairs]            # strip color for visualizer

        page_debug = {}
        for step_fn, key, step_args in [
            (save_step1_raw_paths, "step1", (page, drawings,             f"{debug_base}_step1_raw.png")),
            (save_step2_polygons,  "step2", (page, filled_polys + recon, f"{debug_base}_step2_polys.png")),
            (save_step3_filtered,  "step3", (page, filtered,             f"{debug_base}_step3_filtered.png")),
            (save_gross_net_slab_debug, "gross_net", (page, extraction_result, f"{debug_base}_gross_net.png")),
            (save_boundary_first_debug, "boundary", (page, boundary_result, f"{debug_base}_boundary_first.png")),
            (save_wall_evidence_only, "wall_evidence", (page, boundary_result, f"{debug_base}_wall_evidence.png")),
            (save_slab_candidates_only, "wall_candidates", (page, boundary_result, f"{debug_base}_wall_candidates.png")),
            (save_wall_guided_final, "wall_final", (page, boundary_result, f"{debug_base}_wall_final.png")),
            (save_step4_labeled,   "step4", (page, slab_regions,   f"{debug_base}_step4_labeled.png")),
            (save_step5_final,     "step5", (page, slab_regions,   f"{debug_base}_step5_final.png")),
        ]:
            try:
                page_debug[key] = step_fn(*step_args)
            except Exception as dbg_err:
                page_debug[f"{key}_error"] = str(dbg_err)
                pass

        return page_idx, slab_regions, page_debug, has_fill, slab_stats
    finally:
        doc.close()


def _run_legend_detection(pdf_path: str, pages_to_process: list[int]) -> dict:
    """Detect and render review crops for side-strip legends."""
    from collections import defaultdict
    import fitz
    from src.legend_locator import locate_legends_for_pages
    from src.visualizer import save_legend_crop, save_legend_overlay

    if len(pages_to_process) > 12:
        step = max(1, len(pages_to_process) // 12)
        sampled_pages = list(pages_to_process[::step][:12])
    else:
        sampled_pages = list(pages_to_process)

    cache_key = {
        "pdf_path": str(pdf_path),
        "pages": sampled_pages,
    }
    if st.session_state.get("legend_detection_cache_key") == cache_key:
        cached = st.session_state.get("legend_detection")
        if cached:
            return cached

    result = locate_legends_for_pages(pdf_path, sampled_pages, dpi=144)
    result["sampled_pages"] = [p + 1 for p in sampled_pages]
    out_dir = sess_debug_dir / "legend_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_by_page: dict[int, list[dict]] = defaultdict(list)
    for cand in result.get("candidates", []):
        candidates_by_page[int(cand.get("page_index", -1))].append(cand)

    overlays: dict[int, str] = {}
    crops: dict[int, list[dict]] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_idx, candidates in sorted(candidates_by_page.items()):
            if page_idx < 0 or page_idx >= doc.page_count:
                continue
            page = doc[page_idx]
            overlay_path = out_dir / f"p{page_idx + 1:02d}_legend_overlay.png"
            save_legend_overlay(page, candidates, str(overlay_path), dpi=100)
            overlays[page_idx] = str(overlay_path)
            page_crops = []
            for i, cand in enumerate(candidates, start=1):
                crop_path = out_dir / f"p{page_idx + 1:02d}_legend_{cand.get('side', 'side')}_{i}.png"
                save_legend_crop(page, cand.get("bbox", []), str(crop_path), dpi=160)
                crop_item = dict(cand)
                crop_item["image_path"] = str(crop_path)
                page_crops.append(crop_item)
            crops[page_idx] = page_crops
    finally:
        doc.close()

    result["overlays"] = overlays
    result["crops"] = crops
    st.session_state["legend_detection"] = result
    st.session_state["legend_detection_cache_key"] = cache_key
    return result


def _sample_pages_for_semantics(pages_to_process: list[int]) -> list[int]:
    if len(pages_to_process) > 6:
        step = max(1, len(pages_to_process) // 6)
        return list(pages_to_process[::step][:6])
    return list(pages_to_process)


def _run_legend_semantics(pdf_path: str, pages_to_process: list[int]) -> dict:
    """Call Gemini on indexed legend crop text and cache semantic rules."""
    from src.legend_semantic_analyzer import analyze_legend_semantics

    sampled_pages = _sample_pages_for_semantics(pages_to_process)
    cache_key = {"pdf_path": str(pdf_path), "pages": sampled_pages}
    if st.session_state.get("legend_semantics_cache_key") == cache_key:
        cached = st.session_state.get("legend_semantics")
        if cached:
            return cached

    result, json_path, raw_path, report_path = analyze_legend_semantics(
        pdf_path,
        sampled_pages,
        OUTPUT_DIR,
    )
    st.session_state["legend_semantics"] = result
    st.session_state["legend_semantics_path"] = json_path
    st.session_state["legend_semantics_raw_path"] = raw_path
    st.session_state["legend_semantics_report_path"] = report_path
    st.session_state["legend_semantics_cache_key"] = cache_key
    return result


def _run_wall_detection_preview(pdf_path: str, pages_to_process: list[int], scale: int) -> list:
    """Detect semantic wall objects and render review overlays."""
    import fitz
    from src.wall_detector import detect_walls_for_pages
    from src.visualizer import save_semantic_wall_overlay, save_wall_polygons

    sem_result = st.session_state.get("legend_semantics") or {}
    legend_semantics = sem_result.get("gemini_result") if isinstance(sem_result, dict) else None
    if not legend_semantics:
        st.session_state["wall_regions"] = []
        return []

    walls, structural_by_page = detect_walls_for_pages(
        pdf_path,
        pages_to_process,
        scale,
        legend_semantics=legend_semantics,
    )
    page_context: dict[int, tuple[str, str]] = {}
    ai_floor = _trusted_ai_floor_result()
    if ai_floor:
        for b in ai_floor.get("buildings", []) if isinstance(ai_floor, dict) else []:
            for floor in b.get("floors", []):
                for p1 in floor.get("slab_plan_pages", []) or []:
                    if isinstance(p1, int):
                        page_context[p1 - 1] = (b.get("name", ""), floor.get("level_name", ""))
    for wall in walls:
        bld, lvl = page_context.get(wall.page_index, ("(unknown)", "(unknown)"))
        wall.building = wall.building or bld
        wall.level = wall.level or lvl
    out_dir = sess_debug_dir / "walls"
    out_dir.mkdir(parents=True, exist_ok=True)
    walls_by_page: dict[int, list] = {}
    for wall in walls:
        walls_by_page.setdefault(wall.page_index, []).append(wall)

    semantic_imgs: dict[int, str] = {}
    wall_imgs: dict[int, str] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_idx in pages_to_process:
            if page_idx < 0 or page_idx >= doc.page_count:
                continue
            page = doc[page_idx]
            structural = structural_by_page.get(page_idx)
            if structural is not None:
                p = out_dir / f"p{page_idx + 1:02d}_semantic_wall_overlay.png"
                semantic_imgs[page_idx] = save_semantic_wall_overlay(page, structural, str(p), dpi=130)
            page_walls = walls_by_page.get(page_idx, [])
            if page_walls:
                p = out_dir / f"p{page_idx + 1:02d}_wall_model_polygons.png"
                wall_imgs[page_idx] = save_wall_polygons(page, page_walls, str(p), dpi=150)
    finally:
        doc.close()

    st.session_state["wall_regions"] = walls
    st.session_state["semantic_wall_images"] = semantic_imgs
    st.session_state["wall_polygon_images"] = wall_imgs
    return walls


def _run_slab_semantic_preview(pdf_path: str, pages_to_process: list[int]) -> dict:
    """Render Step 2.1 slab semantic overlays for review."""
    import fitz
    from src.slab_semantic_detector import detect_slab_semantics_for_pages
    from src.visualizer import (
        save_slab_semantic_boundary_cues,
        save_slab_semantic_cut_candidates,
        save_slab_semantic_surface,
    )

    sem_result = st.session_state.get("legend_semantics") or {}
    legend_semantics = sem_result.get("gemini_result") if isinstance(sem_result, dict) else None
    if not legend_semantics:
        st.session_state["slab_semantic_previews"] = {}
        return {}

    previews = detect_slab_semantics_for_pages(
        pdf_path,
        pages_to_process,
        legend_semantics=legend_semantics,
    )
    out_dir = sess_debug_dir / "slab_semantics"
    out_dir.mkdir(parents=True, exist_ok=True)
    surface_imgs: dict[int, str] = {}
    boundary_imgs: dict[int, str] = {}
    cut_imgs: dict[int, str] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_idx, preview in previews.items():
            if page_idx < 0 or page_idx >= doc.page_count:
                continue
            page = doc[page_idx]
            p = out_dir / f"p{page_idx + 1:02d}_slab_surface.png"
            surface_imgs[page_idx] = save_slab_semantic_surface(page, preview, str(p), dpi=130)
            p = out_dir / f"p{page_idx + 1:02d}_slab_boundary_cues.png"
            boundary_imgs[page_idx] = save_slab_semantic_boundary_cues(page, preview, str(p), dpi=130)
            p = out_dir / f"p{page_idx + 1:02d}_slab_cut_candidates.png"
            cut_imgs[page_idx] = save_slab_semantic_cut_candidates(page, preview, str(p), dpi=130)
    finally:
        doc.close()

    st.session_state["slab_semantic_previews"] = previews
    st.session_state["slab_semantic_surface_images"] = surface_imgs
    st.session_state["slab_semantic_boundary_images"] = boundary_imgs
    st.session_state["slab_semantic_cut_images"] = cut_imgs
    return previews


def _render_legend_detection():
    legend = st.session_state.get("legend_detection")
    if not legend:
        return
    rows = legend.get("rows", []) or []
    consensus = legend.get("consensus", {}) or {}
    overlays = legend.get("overlays", {}) or {}
    crops = legend.get("crops", {}) or {}

    with st.expander("Legend Detection / Crop Preview", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Candidates", len(legend.get("candidates", []) or []))
        c2.metric("Consensus side", consensus.get("side") or "N/A")
        c3.metric("Coverage", consensus.get("coverage", 0))
        c4.metric("Status", consensus.get("status", "unknown"))
        if legend.get("sampled_pages"):
            st.caption(f"Sampled pages: {', '.join(map(str, legend.get('sampled_pages', [])))}")
        if consensus.get("status") == "weak":
            st.warning("Legend consensus is weak across selected pages. Review crops before using this as a template.")
        elif consensus.get("status") == "missing":
            st.warning("No reliable legend crop found on selected pages.")

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for page_idx in sorted(set(list(overlays.keys()) + list(crops.keys()))):
            with st.expander(f"Legend crop - Page {page_idx + 1}", expanded=False):
                overlay_path = overlays.get(page_idx)
                if overlay_path and Path(overlay_path).exists():
                    st.markdown("##### Overlay")
                    st.image(overlay_path, use_container_width=True)
                    with open(overlay_path, "rb") as f:
                        st.download_button(
                            f"Download legend overlay P{page_idx + 1}",
                            f.read(),
                            file_name=Path(overlay_path).name,
                            mime="image/png",
                            key=f"dl_legend_overlay_{page_idx}",
                        )
                for i, item in enumerate(crops.get(page_idx, []), start=1):
                    img_path = item.get("image_path")
                    if img_path and Path(img_path).exists():
                        st.markdown(
                            f"##### Crop {i}: {item.get('side')} | "
                            f"confidence={item.get('confidence')}"
                        )
                        st.image(img_path, use_container_width=True)
                        st.caption(item.get("text_preview", ""))
                        with open(img_path, "rb") as f:
                            st.download_button(
                                f"Download legend crop P{page_idx + 1}-{i}",
                                f.read(),
                                file_name=Path(img_path).name,
                                mime="image/png",
                                key=f"dl_legend_crop_{page_idx}_{i}",
                            )


def _render_legend_semantics_and_walls():
    sem = st.session_state.get("legend_semantics")
    walls = st.session_state.get("wall_regions", []) or []
    if not sem and not walls:
        return
    metadata = (sem or {}).get("_metadata", {}) if isinstance(sem, dict) else {}
    parsed = (sem or {}).get("gemini_result", {}) if isinstance(sem, dict) else {}

    with st.expander("Step 2.2 - Legend Semantics / Wall Preview", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Legend parse", metadata.get("parse_status", "N/A"))
        c2.metric("Wall rules", len((parsed.get("rules_for_code") or {}).get("wall_keywords", []) if parsed else []))
        c3.metric("Walls for model", len(walls))
        c4.metric("Cut rules", len((parsed.get("rules_for_code") or {}).get("net_slab_cut_keywords", []) if parsed else []))

        tabs = st.tabs(["Rules", "Wall Evidence", "Wall Model Polygons", "Raw JSON"])
        with tabs[0]:
            rules = parsed.get("rules_for_code", {}) if parsed else {}
            if rules:
                st.json(rules)
            if parsed.get("wall_detection_items"):
                st.markdown("##### Wall Detection Items")
                st.dataframe(pd.DataFrame(parsed.get("wall_detection_items", [])), use_container_width=True, hide_index=True)
            if parsed.get("slab_detection_items"):
                st.markdown("##### Slab Detection Items")
                st.dataframe(pd.DataFrame(parsed.get("slab_detection_items", [])), use_container_width=True, hide_index=True)
            if parsed.get("warnings"):
                for w in parsed.get("warnings", []):
                    st.warning(w)
        with tabs[1]:
            _render_element_overlay_gallery(
                "Semantic Wall Evidence",
                st.session_state.get("semantic_wall_images", {}) or {},
                "No semantic wall evidence overlays yet.",
                "dl_semantic_wall_overlay",
            )
        with tabs[2]:
            _render_element_overlay_gallery(
                "Wall Model Polygons",
                st.session_state.get("wall_polygon_images", {}) or {},
                "No wall polygons selected for model export yet.",
                "dl_wall_model_overlay",
            )
        with tabs[3]:
            st.json(sem or {})
            for label, path_key, mime in [
                ("Download legend semantics JSON", "legend_semantics_path", "application/json"),
                ("Download legend semantics raw", "legend_semantics_raw_path", "text/plain"),
                ("Download legend semantics parse report", "legend_semantics_report_path", "application/json"),
            ]:
                p = st.session_state.get(path_key)
                if p and Path(p).exists():
                    with open(p, "rb") as f:
                        st.download_button(
                            label,
                            f.read(),
                            file_name=Path(p).name,
                            mime=mime,
                            key=f"dl_{path_key}",
                        )


def _render_slab_semantic_preview():
    sem = st.session_state.get("legend_semantics")
    previews = st.session_state.get("slab_semantic_previews") or {}
    if not sem and not previews:
        return
    metadata = (sem or {}).get("_metadata", {}) if isinstance(sem, dict) else {}
    parsed = (sem or {}).get("gemini_result", {}) if isinstance(sem, dict) else {}
    rules = parsed.get("rules_for_code", {}) if parsed else {}
    surface_count = sum(len(getattr(p, "surface_regions", []) or []) for p in previews.values())
    cue_count = sum(len(getattr(p, "boundary_cues", []) or []) for p in previews.values())
    cut_count = sum(len(getattr(p, "cut_candidates", []) or []) for p in previews.values())
    page_rows = []
    for page_idx, preview in sorted(previews.items()):
        warnings = getattr(preview, "warnings", []) or []
        page_rows.append({
            "Page": page_idx + 1,
            "Surface Count": len(getattr(preview, "surface_regions", []) or []),
            "Boundary Cue Count": len(getattr(preview, "boundary_cues", []) or []),
            "Cut Candidate Count": len(getattr(preview, "cut_candidates", []) or []),
            "Gemini Policy": getattr(preview, "gemini_fallback_policy", None) or rules.get("fallback_policy"),
            "Effective Source": getattr(preview, "effective_surface_source", "unknown"),
            "Fallback Policy": getattr(preview, "fallback_policy", "unknown"),
            "Warning / Status": " | ".join(warnings) if warnings else "ok",
        })
    uses_no_fill_boundary = any(
        getattr(p, "effective_surface_source", "") in (
            "white_no_fill_boundary",
            "evidence_guided_no_fill_boundary",
        )
        for p in previews.values()
    )

    with st.expander("Step 2.1 - Legend Slab Semantics Preview", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Legend parse", metadata.get("parse_status", "N/A"))
        c2.metric("Slab surfaces", surface_count)
        c3.metric("Boundary cues", cue_count)
        c4.metric("Cut candidates", cut_count)
        fallback = rules.get("fallback_policy")
        if fallback in ("use_white_no_fill_boundary", "use_evidence_guided_no_fill_boundary") or uses_no_fill_boundary:
            st.warning("One or more pages use evidence-guided no-fill boundary slab candidates for review. This uses learned boundary signatures/wall evidence, not the full white page background.")

        tabs = st.tabs(["Rules", "Slab Surface Evidence", "Slab Boundary Cues", "Slab Cut Candidates", "Raw JSON"])
        with tabs[0]:
            st.json({
                "slab_surface_keywords": rules.get("slab_surface_keywords", []),
                "slab_fill_keywords": rules.get("slab_fill_keywords", []),
                "slab_boundary_keywords": rules.get("slab_boundary_keywords", []),
                "net_slab_cut_keywords": rules.get("net_slab_cut_keywords", []),
                "gemini_fallback_policy": rules.get("fallback_policy"),
                "notes": rules.get("notes"),
            })
            if page_rows:
                st.markdown("##### Page Slab Semantic Source Report")
                st.dataframe(pd.DataFrame(page_rows), use_container_width=True, hide_index=True)
            for key, title in [
                ("slab_surface_items", "Slab Surface Items"),
                ("slab_boundary_items", "Slab Boundary Items"),
                ("slab_cut_items", "Slab Cut Items"),
                ("slab_detection_items", "Legacy Slab Detection Items"),
            ]:
                if parsed.get(key):
                    st.markdown(f"##### {title}")
                    st.dataframe(pd.DataFrame(parsed.get(key, [])), use_container_width=True, hide_index=True)
        with tabs[1]:
            _render_element_overlay_gallery(
                "Slab Surface Evidence",
                st.session_state.get("slab_semantic_surface_images", {}) or {},
                "No slab surface semantic overlays yet.",
                "dl_slab_semantic_surface",
            )
        with tabs[2]:
            _render_element_overlay_gallery(
                "Slab Boundary Cues",
                st.session_state.get("slab_semantic_boundary_images", {}) or {},
                "No slab boundary cue overlays yet.",
                "dl_slab_semantic_boundary",
            )
        with tabs[3]:
            _render_element_overlay_gallery(
                "Slab Cut Candidates",
                st.session_state.get("slab_semantic_cut_images", {}) or {},
                "No slab cut candidate overlays yet.",
                "dl_slab_semantic_cut",
            )
        with tabs[4]:
            st.json(sem or {})


def _render_document_intelligence():
    intel = st.session_state.get("document_intelligence")
    if not intel:
        return
    summary = intel.get("document_summary", {})
    schedule_pages = intel.get("schedule_pages", {})
    column_symbols = intel.get("column_symbols", {})
    foundation_symbols = intel.get("foundation_symbols", {})
    metadata = intel.get("_metadata", {}) or {}
    parse_status = intel.get("_parse_status") or metadata.get("parse_status", "unknown")
    parse_error = intel.get("_parse_error") or metadata.get("parse_error")

    with st.expander("Document Intelligence (Gemini full PDF)", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Parse", parse_status)
        c2.metric("Confidence", summary.get("detection_confidence", "N/A"))
        c3.metric("Column symbols", len(column_symbols))
        c4.metric("Foundation symbols", len(foundation_symbols))
        c5.metric("PDF pages", summary.get("page_count", "N/A"))

        if parse_status != "ok":
            st.error(
                "Gemini raw response saved, but JSON parse failed or semantic schema is empty. "
                "Do not trust empty building/column/foundation results until the raw response is reviewed."
            )
            if parse_error:
                st.code(str(parse_error), language="text")

        tabs = st.tabs(["Summary", "Audit", "Columns", "Foundations", "Heights", "Buildings/Floors", "Warnings", "Raw JSON"])
        with tabs[0]:
            st.json({
                "project_name": summary.get("project_name"),
                "notes": summary.get("notes"),
                "schedule_pages": schedule_pages,
                "legend_rules": intel.get("legend_rules", {}),
            })
        with tabs[1]:
            intel_path = st.session_state.get("document_intelligence_path")
            raw_path = st.session_state.get("document_intelligence_raw_path") or metadata.get("raw_response_path")
            report_path = (
                st.session_state.get("document_intelligence_parse_report_path")
                or metadata.get("parse_report_path")
            )
            st.json({
                "parse_status": parse_status,
                "parse_error": parse_error,
                "raw_response_length": metadata.get("raw_response_length"),
                "cleaned_response_length": metadata.get("cleaned_response_length"),
                "response_ended_cleanly": metadata.get("response_ended_cleanly"),
                "parsed_json_path": intel_path or metadata.get("parsed_json_path"),
                "raw_response_path": raw_path,
                "parse_report_path": report_path,
            })
            if intel_path and Path(intel_path).exists():
                with open(intel_path, "rb") as f:
                    st.download_button(
                        "Download document_intelligence.json",
                        f.read(),
                        file_name=Path(intel_path).name,
                        mime="application/json",
                        key="dl_document_intelligence_json",
                    )
            if raw_path and Path(raw_path).exists():
                raw_text = Path(raw_path).read_text(encoding="utf-8", errors="replace")
                with open(raw_path, "rb") as f:
                    st.download_button(
                        "Download Gemini raw response",
                        f.read(),
                        file_name=Path(raw_path).name,
                        mime="text/plain",
                        key="dl_document_intelligence_raw",
                    )
                st.markdown("##### Raw preview - first 3000 chars")
                st.code(raw_text[:3000], language="json")
                st.markdown("##### Raw preview - last 3000 chars")
                st.code(raw_text[-3000:], language="json")
            if report_path and Path(report_path).exists():
                with open(report_path, "rb") as f:
                    st.download_button(
                        "Download parse report",
                        f.read(),
                        file_name=Path(report_path).name,
                        mime="application/json",
                        key="dl_document_intelligence_parse_report",
                    )
        with tabs[2]:
            rows = []
            for sym, info in column_symbols.items():
                rows.append({
                    "Symbol": sym,
                    "Family": info.get("family"),
                    "Status": info.get("status"),
                    "Width": info.get("width_mm"),
                    "Depth": info.get("depth_mm"),
                    "Count": info.get("count_total"),
                    "Source": info.get("source"),
                    "Pages": ", ".join(map(str, info.get("source_pages", []))),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with tabs[3]:
            rows = []
            for sym, info in foundation_symbols.items():
                rows.append({
                    "Symbol": sym,
                    "Type": info.get("type"),
                    "Width": info.get("width_mm"),
                    "Depth": info.get("depth_mm"),
                    "Thickness": info.get("thickness_mm"),
                    "Depth below GL": info.get("depth_below_gl_mm"),
                    "Pile count": info.get("pile_count"),
                    "Pile dia": info.get("pile_diameter_mm"),
                    "Source": info.get("source"),
                    "Pages": ", ".join(map(str, info.get("source_pages", []))),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info(f"No foundation symbols extracted yet. Foundation schedule pages: {schedule_pages.get('foundation_schedule_pages', [])}")
        with tabs[4]:
            height_sources = intel.get("height_sources", []) or []
            storey_heights = intel.get("storey_heights", []) or []
            if height_sources:
                rows = []
                for src in height_sources:
                    rows.append({
                        "Type": src.get("type"),
                        "Page": src.get("page"),
                        "Level": src.get("level"),
                        "Elevation (m)": src.get("elevation_m"),
                        "Height (mm)": src.get("height_mm"),
                        "Action": src.get("recommended_action"),
                        "Confidence": src.get("confidence"),
                        "Text": src.get("source_text", ""),
                    })
                st.markdown("##### Height sources")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No height sources extracted by Gemini yet.")
            if storey_heights:
                st.markdown("##### Storey heights")
                st.dataframe(pd.DataFrame(storey_heights), use_container_width=True, hide_index=True)

        with tabs[5]:
            floor_rows = []
            for b in intel.get("buildings", []):
                for f in b.get("floors", []):
                    col_summary = f.get("column_summary", {}) or {}
                    fdn_summary = f.get("foundation_summary", {}) or {}
                    floor_rows.append({
                        "Building": b.get("name"),
                        "Floor": f.get("level_name"),
                        "Slab pages": ", ".join(map(str, f.get("slab_plan_pages", []))),
                        "Column total": col_summary.get("total_columns"),
                        "Column symbols": json.dumps(col_summary.get("by_symbol", {}), ensure_ascii=False),
                        "Foundation total": fdn_summary.get("total_foundations"),
                    })
            st.dataframe(pd.DataFrame(floor_rows), use_container_width=True, hide_index=True)
        with tabs[6]:
            warnings = intel.get("warnings", [])
            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("No document-intelligence warnings.")
        with tabs[7]:
            st.json(intel)


def _compute_column_validation(document_intelligence: dict, detected_columns: list) -> dict:
    expected_rows = []
    for b in (document_intelligence or {}).get("buildings", []):
        for floor in b.get("floors", []):
            pages = [p - 1 for p in floor.get("slab_plan_pages", []) if isinstance(p, int)]
            by_symbol = (floor.get("column_summary") or {}).get("by_symbol", {}) or {}
            for sym, expected in by_symbol.items():
                expected_rows.append({
                    "building": b.get("name", ""),
                    "floor": floor.get("level_name", ""),
                    "pages": pages,
                    "symbol": sym,
                    "expected": expected,
                })

    rows = []
    for row in expected_rows:
        detected = sum(
            1 for c in detected_columns
            if c.symbol == row["symbol"] and c.page_index in row["pages"]
        )
        rows.append({
            "Building": row["building"],
            "Floor": row["floor"],
            "Symbol": row["symbol"],
            "Expected": row["expected"],
            "Detected": detected,
            "Delta": detected - int(row["expected"] or 0),
        })
    return {"rows": rows}


def _render_step3_results():
    """Display Step 3 results and navigation buttons (reads from session_state)."""
    selected_pages = st.session_state["selected_pages"]
    results = st.session_state["slab_results"]
    debug_imgs = st.session_state["debug_images"]
    slab_stats = st.session_state.get("slab_debug_stats", {})
    all_slabs = st.session_state["final_slabs"]
    total = len(all_slabs)
    n = len(selected_pages)

    st.markdown(
        f'<div class="success-box">✅ Complete — detected <b>{total} slab{"s" if total != 1 else ""}</b> '
        f'across {n} page{"s" if n != 1 else ""}.</div>', unsafe_allow_html=True,
    )

    # ── Column & Foundation Census JSON ────────────────────────────────────
    if slab_stats:
        gross_count = sum(s.get("gross_slab_count", 0) for s in slab_stats.values())
        appendage_count = sum(s.get("recovered_appendage_count", 0) for s in slab_stats.values())
        void_count = sum(s.get("void_candidate_count", 0) for s in slab_stats.values())
        auto_cut_count = sum(s.get("auto_cut_count", 0) for s in slab_stats.values())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross slab regions", gross_count)
        c2.metric("Recovered appendages", appendage_count)
        c3.metric("Void candidates", void_count)
        c4.metric("Auto cuts", auto_cut_count)
        mode_rows = []
        for page_idx, stats in sorted(slab_stats.items()):
            mode_rows.append({
                "Page": page_idx + 1,
                "Mode": stats.get("extraction_mode", "fill_first"),
                "Reason": stats.get("mode_reason", ""),
                "Reliable fill": bool(stats.get("reliable_visible_fill", False)),
                "Fill confidence": round(float(stats.get("fill_confidence", 0.0)), 2),
                "Boundary confidence": round(float(stats.get("boundary_confidence", 0.0)), 2),
                "Walls": stats.get("wall_count", 0),
                "Cores": stats.get("core_count", 0),
                "Stairs": stats.get("stair_count", 0),
                "Openings": stats.get("opening_count", 0),
                "Boundary cuts": stats.get("boundary_cut_count", 0),
                "Ignored zones": stats.get("ignored_boundary_count", 0),
                "Final slabs": stats.get("final_slab_count", stats.get("net_slab_count", 0)),
            })
        with st.expander("Extraction Mode Report", expanded=True):
            st.dataframe(pd.DataFrame(mode_rows), use_container_width=True, hide_index=True)

    _render_legend_detection()
    _render_slab_semantic_preview()
    _render_legend_semantics_and_walls()
    _render_document_intelligence()
    _render_floor_alignment_report(expanded=True)
    _render_building_registry(expanded=True)

    col_validation = st.session_state.get("column_validation") or {}
    if col_validation.get("rows"):
        with st.expander("Column Expected vs Detected", expanded=True):
            st.dataframe(pd.DataFrame(col_validation["rows"]), use_container_width=True, hide_index=True)

    census = st.session_state.get("column_census")
    if census:
        with st.expander("🔍 Column & Foundation Census (Gemini JSON)", expanded=True):
            st.json(census)
            census_path = st.session_state.get("column_census_path")
            if census_path and Path(census_path).exists():
                with open(census_path, "rb") as cf:
                    st.download_button(
                        "Download Column Census JSON",
                        cf.read(),
                        file_name=Path(census_path).name,
                        mime="application/json",
                        key="dl_column_census_json",
                    )

    with st.expander("Column / Foundation Polygon Overlays", expanded=True):
        tab_cols, tab_fdns = st.tabs(["Columns", "Foundations"])
        with tab_cols:
            _render_element_overlay_gallery(
                "Columns",
                st.session_state.get("column_polygon_images", {}) or {},
                "No column polygon overlay images. Column detection may be disabled or detected 0 columns.",
                "dl_step3_col_overlay",
            )
        with tab_fdns:
            _render_element_overlay_gallery(
                "Foundations",
                st.session_state.get("foundation_polygon_images", {}) or {},
                "No foundation polygon overlay images. Foundation symbols/instances may not be detected yet.",
                "dl_step3_fdn_overlay",
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
            stats = slab_stats.get(page_idx)
            if stats:
                st.caption(
                    " | ".join([
                        f"gross={stats.get('gross_slab_count', 0)}",
                        f"net={stats.get('net_slab_count', 0)}",
                        f"appendages={stats.get('recovered_appendage_count', 0)}",
                        f"voids={stats.get('void_candidate_count', 0)}",
                        f"auto_cuts={stats.get('auto_cut_count', 0)}",
                        f"dominant_fill={stats.get('dominant_fill')}",
                        f"mode={stats.get('extraction_mode')}",
                        f"boundary_signatures={stats.get('boundary_signature_count', 0)}",
                        f"boundary_evidence={stats.get('boundary_evidence_count', 0)}",
                    ])
                )
            tabs = st.tabs(["① Raw Paths", "② Polygons", "③ Filtered", "Gross/Net", "Wall/Boundary", "④ Labeled", "⑤ Final"])
            step_keys = ["step1", "step2", "step3", "gross_net", "boundary", "step4", "step5"]
            for tab, key in zip(tabs, step_keys):
                with tab:
                    if key == "boundary":
                        wall_items = [
                            ("Wall Evidence Only", "wall_evidence"),
                            ("Slab Candidates Only", "wall_candidates"),
                            ("Final Wall-Guided Result", "wall_final"),
                            ("Combined Legacy Overlay", "boundary"),
                        ]
                        for label, wall_key in wall_items:
                            st.markdown(f"##### {label}")
                            img_path = page_imgs.get(wall_key)
                            if img_path and Path(img_path).exists():
                                st.image(img_path, use_container_width=True)
                                with open(img_path, "rb") as f:
                                    st.download_button(
                                        f"Download {wall_key}.png", f.read(),
                                        file_name=Path(img_path).name, mime="image/png",
                                        key=f"dl_{page_idx}_{wall_key}",
                                    )
                            else:
                                st.info(f"{label} image not available.")
                        continue
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

    try:
        _run_legend_detection(pdf_path, pages_to_process)
    except Exception as le:
        st.warning(f"Legend detection preview failed: {le}")

    try:
        if not st.session_state.get("legend_semantics"):
            with st.spinner("Step 2.2 - Gemini reading legend rules..."):
                _run_legend_semantics(pdf_path, pages_to_process)
        _run_slab_semantic_preview(pdf_path, pages_to_process)
        _run_wall_detection_preview(pdf_path, pages_to_process, scale)
    except Exception as we:
        st.warning(f"Legend semantic / wall preview failed: {we}")

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

    sem_result = st.session_state.get("legend_semantics") or {}
    legend_semantics = sem_result.get("gemini_result") if isinstance(sem_result, dict) else None
    worker_args = [
        (pdf_path, page_idx, scale, str(sess_debug_dir / f"p{page_idx + 1:02d}"), legend_semantics)
        for page_idx in pages_to_process
    ]

    results = {}
    debug_imgs = {}
    slab_debug_stats = {}
    page_fill_detected = {}
    done_count = 0

    with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
        futures = {executor.submit(_process_page_worker, args): args[1] for args in worker_args}
        for future in as_completed(futures):
            page_idx = futures[future]
            try:
                tup = future.result()
                if len(tup) == 5:
                    p_idx, slab_regions, page_debug, has_fill, page_stats = tup
                    page_fill_detected[p_idx] = has_fill
                    slab_debug_stats[p_idx] = page_stats
                elif len(tup) == 4:
                    p_idx, slab_regions, page_debug, has_fill = tup
                    page_fill_detected[p_idx] = has_fill
                else:
                    p_idx, slab_regions, page_debug = tup
                    page_fill_detected[p_idx] = False
                results[p_idx] = slab_regions
                debug_imgs[p_idx] = page_debug
            except Exception as e:
                import traceback
                results[page_idx] = []
                debug_imgs[page_idx] = {"error": str(e), "traceback": traceback.format_exc()}
                page_fill_detected[page_idx] = False
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
    st.session_state["slab_debug_stats"] = slab_debug_stats

    # ── Vision Refinement (Stage 4) — skip pages where slab fill color was already detected ──
    if st.session_state.get("vision_refine_enabled"):
        import fitz as _fitz
        from src.vision_refiner import get_vision_client, refine_page_slabs
        backend = st.session_state.get("vision_backend", "gemini")

        pages_with_slabs = [(idx, slabs) for idx, slabs in results.items() if slabs]
        fill_pages   = {idx for idx, has_f in page_fill_detected.items() if has_f}
        no_fill_pages = {idx for idx, _ in pages_with_slabs if idx not in fill_pages}
        skip_pages    = {idx for idx, _ in pages_with_slabs if idx in fill_pages}

        if skip_pages:
            st.markdown(
                f'<div class="info-box">🖌️ <b>Fill-based detection</b> already found slab polygons '
                f'on {len(skip_pages)} page(s) — Vision skipped '
                f'(P{", P".join(str(p+1) for p in sorted(skip_pages))}). '
                f'Vision only runs on {len(no_fill_pages)} SOG / line-drawn page(s).</div>',
                unsafe_allow_html=True,
            )

        if no_fill_pages:
            to_refine = [(idx, slabs) for idx, slabs in pages_with_slabs if idx in no_fill_pages]
            max_workers = min(len(to_refine), VISION_MAX_WORKERS)
            v_prog = st.progress(
                0,
                text=f"Vision Refinement (Stage 4) parallel x{max_workers}...",
            )

            def _vision_refine_worker(page_idx: int, page_slabs: list) -> tuple[int, list, str | None]:
                doc_v = None
                try:
                    v_client, v_model = get_vision_client(backend)
                    doc_v = _fitz.open(pdf_path)
                    refined = refine_page_slabs(
                        page_slabs, doc_v[page_idx], v_client, v_model, backend
                    )
                    return page_idx, refined, None
                except Exception as exc:
                    return page_idx, page_slabs, str(exc)
                finally:
                    if doc_v is not None:
                        try:
                            doc_v.close()
                        except Exception:
                            pass

            done_v = 0
            vision_errors = []
            with ThreadPoolExecutor(max_workers=max_workers) as v_executor:
                v_futures = {
                    v_executor.submit(_vision_refine_worker, page_idx, page_slabs): page_idx
                    for page_idx, page_slabs in to_refine
                }
                for v_future in as_completed(v_futures):
                    page_idx = v_futures[v_future]
                    try:
                        refined_page_idx, refined_slabs, err = v_future.result()
                    except Exception as exc:
                        refined_page_idx, refined_slabs, err = page_idx, results.get(page_idx, []), str(exc)
                    results[refined_page_idx] = refined_slabs
                    done_v += 1
                    if err:
                        vision_errors.append((refined_page_idx, err))
                    v_prog.progress(
                        done_v / len(to_refine),
                        text=(
                            f"Vision parallel: page {refined_page_idx + 1} done "
                            f"({done_v}/{len(to_refine)})"
                        ),
                    )
            v_prog.empty()
            st.session_state["slab_results"] = results
            if vision_errors:
                for page_idx, err in vision_errors[:5]:
                    st.warning(
                        f"Vision refinement failed on page {page_idx + 1}: {err} "
                        "— kept standard/boundary-first polygon."
                    )
                if len(vision_errors) > 5:
                    st.warning(f"Vision refinement had {len(vision_errors) - 5} more page error(s).")
        else:
            st.markdown(
                '<div class="success-box">✅ All pages have fill-based slab detection — '
                'Vision refinement not needed.</div>',
                unsafe_allow_html=True,
            )

    all_slabs = [s for page_slabs in results.values() for s in page_slabs]

    # ── Column & Foundation Detection (optional) ────────────────────────────────
    if st.session_state.get("column_detect_enabled"):
        try:
            from src.column_analyzer import analyze_columns_and_foundations
            from src.column_detector import (
                detect_columns_on_page, detect_foundations_on_page,
                assign_columns_to_regions, build_column_types_from_intelligence,
                build_foundation_types_from_intelligence,
            )
            from src.document_intelligence import analyze_document_intelligence
            from src.coordinate_mapper import transform_structural_elements
            import fitz as _fitz2

            ai_floor_res = st.session_state.get("ai_floor_result")
            c_prog = st.progress(0, text="Document Intelligence (Gemini full PDF)...")
            document_intelligence = st.session_state.get("document_intelligence")
            if not document_intelligence:
                document_intelligence, intel_path, raw_path = analyze_document_intelligence(pdf_path, OUTPUT_DIR)
                st.session_state["document_intelligence"] = document_intelligence
                st.session_state["document_intelligence_path"] = intel_path
                st.session_state["document_intelligence_raw_path"] = raw_path
                st.session_state["document_intelligence_parse_report_path"] = (
                    (document_intelligence.get("_metadata") or {}).get("parse_report_path")
                )

            c_prog.progress(0.35, text="Preparing column/foundation symbols...")
            intel_status = (
                document_intelligence.get("_parse_status")
                or (document_intelligence.get("_metadata") or {}).get("parse_status")
                or "ok"
            )
            if intel_status != "ok":
                raise RuntimeError(
                    "Document Intelligence JSON is not valid/usable. "
                    "Raw Gemini response and parse report were saved for audit."
                )
            col_types = build_column_types_from_intelligence(document_intelligence)
            fdn_types = build_foundation_types_from_intelligence(document_intelligence)
            schedule_pages = document_intelligence.get("schedule_pages", {})
            census = {
                "column_types": col_types,
                "foundation_types": fdn_types,
                "buildings": document_intelligence.get("buildings", []),
                "footing_plan_pages": schedule_pages.get("footing_plan_pages", []) or schedule_pages.get("foundation_schedule_pages", []),
                "detection_confidence": document_intelligence.get("document_summary", {}).get("detection_confidence", "low"),
            }
            if not col_types and not fdn_types:
                census = analyze_columns_and_foundations(pdf_path, pages_to_process, ai_floor_res or {})
                col_types = census.get("column_types", {})
                fdn_types = census.get("foundation_types", {})

            st.session_state["column_census"] = census
            census_path = OUTPUT_DIR / f"column_census_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            census_path.write_text(
                json.dumps({"pdf": Path(pdf_path).name, "pages_processed": [p + 1 for p in pages_to_process], "column_census": census}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            st.session_state["column_census_path"] = str(census_path)
            c_prog.progress(0.5, text="Detecting column/foundation positions (vector)...")

            footing_pages_1idx = set(census.get("footing_plan_pages", []))

            # Build slab-page-scoped job map — only scan pages Gemini confirmed as floor plans
            _page_job_map: dict = {}
            for _bldg in census.get("buildings", []):
                for _floor in _bldg.get("floors", []):
                    for _pg1 in _floor.get("slab_plan_pages", []):
                        if not isinstance(_pg1, int):
                            continue
                        _idx = _pg1 - 1
                        by_symbol = ((_floor.get("column_summary") or {}).get("by_symbol") or _floor.get("columns", {}) or {})
                        _relevant = {sym: col_types[sym] for sym in by_symbol if sym in col_types}
                        if not _relevant:
                            _relevant = col_types
                        if _idx not in _page_job_map:
                            _page_job_map[_idx] = {
                                "building": _bldg.get("name", ""),
                                "level": _floor.get("level_name", ""),
                                "col_types": {},
                            }
                        _page_job_map[_idx]["col_types"].update(_relevant)

            doc_c = _fitz2.open(pdf_path)
            all_columns, all_foundations = [], []
            scanned_col_pages = set()

            # Phase 1: scan Gemini-confirmed floor-plan pages with scoped column types
            for page_idx, job in sorted(_page_job_map.items()):
                if page_idx < 0 or page_idx >= doc_c.page_count:
                    continue
                page_c = doc_c[page_idx]
                page_cols = detect_columns_on_page(
                    page_c, job["col_types"], scale, page_idx,
                    building=job["building"], level=job["level"],
                )
                all_columns.extend(transform_structural_elements(page_cols, page_c, scale))
                scanned_col_pages.add(page_idx)

            # Phase 2: scan remaining pages with ALL column types (fallback)
            # These are pages that have column symbols but weren't picked up by Gemini
            # floor-plan classification — e.g. combined plans, detail pages, unlabeled pages.
            if col_types:
                for page_idx in pages_to_process:
                    if page_idx in scanned_col_pages or page_idx < 0 or page_idx >= doc_c.page_count:
                        continue
                    page_c = doc_c[page_idx]
                    page_cols = detect_columns_on_page(
                        page_c, col_types, scale, page_idx,
                        building="(unknown)", level="(unknown)",
                    )
                    all_columns.extend(transform_structural_elements(page_cols, page_c, scale))
            foundation_pages = {p - 1 for p in footing_pages_1idx if isinstance(p, int)} or set(pages_to_process)
            if fdn_types:
                for page_idx in sorted(foundation_pages):
                    if page_idx < 0 or page_idx >= doc_c.page_count:
                        continue
                    page_c = doc_c[page_idx]
                    page_fdns = detect_foundations_on_page(page_c, fdn_types, scale, page_idx)
                    all_foundations.extend(transform_structural_elements(page_fdns, page_c, scale))
            doc_c.close()

            all_columns = assign_columns_to_regions(all_columns, census, ai_floor_res)
            st.session_state["column_regions"]     = all_columns
            st.session_state["foundation_regions"] = all_foundations
            st.session_state["column_validation"] = _compute_column_validation(
                document_intelligence, all_columns
            )

            # --- Generate column/foundation polygon images (per page) ---
            from src.visualizer import save_column_polygons, save_foundation_polygons
            col_img_map: dict = {}
            fdn_img_map: dict = {}
            debug_dir = sess_debug_dir
            debug_dir.mkdir(parents=True, exist_ok=True)
            col_by_page: dict = {}
            for c in all_columns:
                col_by_page.setdefault(c.page_index, []).append(c)
            fdn_by_page: dict = {}
            for f in all_foundations:
                fdn_by_page.setdefault(f.page_index, []).append(f)
            import fitz as _fitz_img
            fitz_for_img = _fitz_img.open(pdf_path)
            for p_idx in sorted(set(list(col_by_page) + list(fdn_by_page))):
                page_img = fitz_for_img[p_idx]
                cols = col_by_page.get(p_idx, [])
                fdns = fdn_by_page.get(p_idx, [])
                if cols:
                    p = debug_dir / f"columns_p{p_idx + 1}.png"
                    save_column_polygons(page_img, cols, str(p))
                    col_img_map[p_idx] = str(p)
                if fdns:
                    p = debug_dir / f"foundations_p{p_idx + 1}.png"
                    save_foundation_polygons(page_img, fdns, str(p))
                    fdn_img_map[p_idx] = str(p)
            fitz_for_img.close()
            st.session_state["column_polygon_images"] = col_img_map
            st.session_state["foundation_polygon_images"] = fdn_img_map

            # Build FFL → height_mm map from column census
            from src.model_builder import _compute_ffl_height_map
            st.session_state["column_height_map"] = _compute_ffl_height_map(
                all_slabs, census
            )

            c_prog.empty()
            st.success(
                f"Column detection: {len(all_columns)} column(s) | "
                f"{len(all_foundations)} foundation(s) found. "
                f"Confidence: {census.get('detection_confidence','?')}"
            )
        except Exception as ce:
            st.warning(f"Column detection failed: {ce}")

    # AI floor merge: combine all polygons from the same Gemini floor group.
    # Handles: (1) cross-page duplicates (same polygon on pages 9+10 → union = one),
    # (2) zone splits (A+B → full floor shape), (3) non-rectangular/L-shape floors.
    ai_floor_result = _trusted_ai_floor_result()
    merge_msg = ""
    if ai_floor_result and st.session_state.get("smart_detect_done"):
        before_merge = len(all_slabs)
        all_slabs = _merge_slabs_by_ai_floors(all_slabs, ai_floor_result)
        merge_msg = f"✅ AI merge: {before_merge} raw slabs → {len(all_slabs)} floor slab(s)"
    elif st.session_state.get("ai_floor_result") and not _semantic_mapping_ready():
        merge_msg = "Document Intelligence parse is not ok - skipped AI building/floor merge to avoid wrong grouping."

    # Assign FFL values from elevation drawing storey heights when FFL annotations are absent
    _assign_ffl_from_elevation(all_slabs, pdf_path, st.session_state.get("page_infos", []))
    _apply_floor_alignment(
        all_slabs,
        st.session_state.get("column_regions", []),
        st.session_state.get("foundation_regions", []),
    )
    _refresh_building_registry(
        all_slabs,
        st.session_state.get("column_regions", []),
        st.session_state.get("foundation_regions", []),
    )

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
    st.markdown("#### Storey Height Report")
    st.markdown(
        '<div class="info-box">Review FFL/storey height evidence before export. '
        'Manual edits are treated as verified and override inferred/default values.</div>',
        unsafe_allow_html=True,
    )
    _render_storey_height_report(all_slabs, editable=True)
    st.session_state["final_slabs"] = all_slabs

    st.markdown("---")
    st.markdown("#### Edit Slab Data")
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

    # ── Column editor (optional) ───────────────────────────────────────────────
    all_columns = st.session_state.get("column_regions", [])
    if all_columns:
        st.markdown("---")
        st.markdown("#### 🏛️ Column Data")
        st.markdown(
            '<div class="info-box">Double-click a cell to edit Width, Depth, Height or Level. '
            'Height_mm = 0 means auto-compute from FFL difference.</div>',
            unsafe_allow_html=True,
        )

        col_df_data = []
        height_map = st.session_state.get("column_height_map", {})
        height_by_page_mm = st.session_state.get("storey_height_by_page_mm", {})
        for col in all_columns:
            # Pre-compute height from FFL map if not already set
            if col.height_mm == 0 and col.symbol in height_map:
                col.height_mm = height_map[col.symbol]
            if col.height_mm == 0 and col.page_index in height_by_page_mm:
                col.height_mm = height_by_page_mm[col.page_index]
            col_df_data.append({
                "ID": col.id,
                "Symbol": col.symbol,
                "Building": col.building,
                "Level": col.level,
                "Width (mm)": round(col.width_mm, 0) if col.width_mm else 0,
                "Depth (mm)": round(col.depth_mm, 0) if col.depth_mm else 0,
                "Height (mm)": round(col.height_mm, 0) if col.height_mm else 0,
                "Page": col.page_index + 1,
            })

        col_df = pd.DataFrame(col_df_data)
        col_edited_df = st.data_editor(
            col_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "Symbol": st.column_config.TextColumn("Symbol", disabled=True, width="small"),
                "Building": st.column_config.TextColumn("Building", width="medium"),
                "Level": st.column_config.TextColumn("Level", width="medium"),
                "Width (mm)": st.column_config.NumberColumn("Width (mm)", format="%.0f", width="medium"),
                "Depth (mm)": st.column_config.NumberColumn("Depth (mm)", format="%.0f", width="medium"),
                "Height (mm)": st.column_config.NumberColumn("Height (mm)", format="%.0f", width="medium",
                    help="Column height in mm. 0 = auto-compute from FFL difference between floors."),
                "Page": st.column_config.NumberColumn("Page", disabled=True, width="small"),
            },
            key="col_editor",
            height=min(400, 60 + len(all_columns) * 38),
        )

        # Apply column edits
        for i, row in col_edited_df.iterrows():
            if i < len(all_columns):
                if pd.notna(row["Width (mm)"]):
                    all_columns[i].width_mm = float(row["Width (mm)"])
                if pd.notna(row["Depth (mm)"]):
                    all_columns[i].depth_mm = float(row["Depth (mm)"])
                if pd.notna(row["Height (mm)"]):
                    all_columns[i].height_mm = float(row["Height (mm)"])
                if pd.notna(row["Building"]):
                    all_columns[i].building = str(row["Building"])
                if pd.notna(row["Level"]):
                    all_columns[i].level = str(row["Level"])
        st.session_state["column_regions"] = all_columns
        _refresh_building_registry(
            all_slabs,
            all_columns,
            st.session_state.get("foundation_regions", []),
        )

    # ── Foundation editor (optional) ──────────────────────────────────────────
    all_foundations = st.session_state.get("foundation_regions", [])
    if all_foundations:
        st.markdown("---")
        st.markdown("#### 🧱 Foundation Data")
        st.markdown(
            '<div class="info-box">Double-click a cell to edit Width, Depth, Depth below GL, or Type. '
            'Foundation types: pad, pile_cap, raft, strip.</div>',
            unsafe_allow_html=True,
        )

        fdn_df_data = []
        for fdn in all_foundations:
            fdn_df_data.append({
                "ID": fdn.id,
                "Symbol": fdn.symbol,
                "Type": fdn.fdn_type,
                "Width (mm)": round(fdn.width_mm, 0) if fdn.width_mm else 0,
                "Depth (mm)": round(fdn.depth_mm, 0) if fdn.depth_mm else 0,
                "Depth below GL (mm)": round(fdn.depth_below_gl_mm, 0) if fdn.depth_below_gl_mm else 0,
                "Page": fdn.page_index + 1,
            })

        fdn_df = pd.DataFrame(fdn_df_data)
        fdn_edited_df = st.data_editor(
            fdn_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "Symbol": st.column_config.TextColumn("Symbol", disabled=True, width="small"),
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["pad", "pile_cap", "raft", "strip"],
                    width="medium",
                    help="Foundation type: pad (spread), pile_cap, raft, or strip footing.",
                ),
                "Width (mm)": st.column_config.NumberColumn("Width (mm)", format="%.0f", width="medium"),
                "Depth (mm)": st.column_config.NumberColumn("Depth (mm)", format="%.0f", width="medium"),
                "Depth below GL (mm)": st.column_config.NumberColumn(
                    "Depth below GL (mm)", format="%.0f", width="medium",
                    help="Depth below ground level in mm (positive = downward).",
                ),
                "Page": st.column_config.NumberColumn("Page", disabled=True, width="small"),
            },
            key="fdn_editor",
            height=min(400, 60 + len(all_foundations) * 38),
        )

        # Apply foundation edits
        for i, row in fdn_edited_df.iterrows():
            if i < len(all_foundations):
                if pd.notna(row["Type"]):
                    all_foundations[i].fdn_type = str(row["Type"])
                if pd.notna(row["Width (mm)"]):
                    all_foundations[i].width_mm = float(row["Width (mm)"])
                if pd.notna(row["Depth (mm)"]):
                    all_foundations[i].depth_mm = float(row["Depth (mm)"])
                if pd.notna(row["Depth below GL (mm)"]):
                    all_foundations[i].depth_below_gl_mm = float(row["Depth below GL (mm)"])
        st.session_state["foundation_regions"] = all_foundations
        _refresh_building_registry(
            all_slabs,
            st.session_state.get("column_regions", []),
            all_foundations,
        )

    # Final images
    st.markdown("---")
    _render_floor_alignment_report(expanded=False)
    st.markdown("---")
    st.markdown("#### Final Result Images")
    tabs = st.tabs(["Slabs", "Slab Semantics", "Buildings", "Walls", "Columns", "Foundations"])
    with tabs[0]:
        for page_idx in selected_pages:
            img_path = debug_imgs.get(page_idx, {}).get("step5")
            if img_path and Path(img_path).exists():
                with st.expander(f"Page {page_idx+1}", expanded=True):
                    st.image(img_path, use_container_width=True)
    with tabs[1]:
        _render_slab_semantic_preview()
    with tabs[2]:
        _render_building_registry(expanded=True)
    with tabs[3]:
        _render_legend_semantics_and_walls()
    with tabs[4]:
        _render_element_overlay_gallery(
            "Columns",
            st.session_state.get("column_polygon_images", {}) or {},
            "No column polygon overlay images. Column detection may be disabled or detected 0 columns.",
            "dl_step4_col_overlay",
        )
    with tabs[5]:
        _render_element_overlay_gallery(
            "Foundations",
            st.session_state.get("foundation_polygon_images", {}) or {},
            "No foundation polygon overlay images. Foundation symbols/instances may not be detected yet.",
            "dl_step4_fdn_overlay",
        )

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

    from src.model_builder import generate_full_ruby_script, generate_slab_csv, compute_storey_heights

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruby_path = str(OUTPUT_DIR / f"structural_model_{ts}.rb")
    csv_path  = str(OUTPUT_DIR / f"slabs_{ts}.csv")

    storey_heights_map = st.session_state.get("storey_heights_map", {})
    all_columns     = st.session_state.get("column_regions", [])
    all_foundations = st.session_state.get("foundation_regions", [])
    all_walls       = st.session_state.get("wall_regions", [])
    height_report, height_by_page_mm = _build_storey_height_report(
        all_slabs,
        st.session_state.get("page_infos", []) or [],
        st.session_state.get("document_intelligence") or {},
    )
    st.session_state["storey_height_report"] = height_report
    st.session_state["storey_height_by_page_mm"] = height_by_page_mm
    building_registry = _refresh_building_registry(all_slabs, all_columns, all_foundations)

    with st.spinner("Generating Ruby script for SketchUp..."):
        ruby_content = generate_full_ruby_script(
            all_slabs, all_columns, all_foundations, ruby_path,
            storey_height_by_page_mm=height_by_page_mm,
            storey_height_report=height_report,
            building_registry=building_registry,
            single_model=True,
            preserve_native_building_position=True,
            generated_by="Feeldx Structural Pipeline",
            wall_regions=all_walls,
        )
        generate_slab_csv(all_slabs, csv_path, storey_heights=storey_heights_map)

    st.session_state["ruby_script"] = ruby_content
    st.session_state["ruby_path"]   = ruby_path
    st.session_state["csv_path"]    = csv_path

    # ── Column & Foundation export (optional) ───────────────────────────────
    if all_columns or all_foundations:
        from src.model_builder import (
            generate_columns_ruby, generate_foundations_ruby, generate_columns_csv,
        )
        if all_columns:
            col_rb_path  = str(Path(OUTPUT_DIR) / f"columns_{ts}.rb")
            col_csv_path = str(Path(OUTPUT_DIR) / f"columns_{ts}.csv")
            generate_columns_ruby(
                all_columns, col_rb_path,
                storey_heights=storey_heights_map,
                height_map=st.session_state.get("column_height_map", {}),
            )
            generate_columns_csv(all_columns, col_csv_path)
            st.session_state["col_ruby_path"] = col_rb_path
            st.session_state["col_csv_path"]  = col_csv_path
        if all_foundations:
            fdn_rb_path = str(Path(OUTPUT_DIR) / f"foundations_{ts}.rb")
            generate_foundations_ruby(all_foundations, fdn_rb_path)
            st.session_state["fdn_ruby_path"] = fdn_rb_path

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Slabs in script", len(all_slabs))
    c2.metric("Columns", len(all_columns))
    c3.metric("Foundations", len(all_foundations))
    c4.metric("Walls", len(all_walls))
    c5.metric("Script size", f"{len(ruby_content)/1024:.1f} KB")

    st.markdown("---")
    st.markdown("#### Final Model Readiness Report")
    _render_floor_alignment_report(expanded=True)
    _render_building_registry(building_registry, expanded=True)
    _render_legend_semantics_and_walls()

    st.markdown("---")
    st.markdown("#### Storey Height Report")
    st.dataframe(pd.DataFrame(height_report), use_container_width=True, hide_index=True)
    unsafe_heights = [r for r in height_report if r.get("Status") in ("missing", "default")]
    if unsafe_heights:
        st.warning(
            f"{len(unsafe_heights)} level(s) are missing verified/inferred height evidence. "
            "Ruby export will use the fallback column height only for those levels."
        )

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

    # Column & Foundation download buttons
    col_rb  = st.session_state.get("col_ruby_path")
    col_csv_f = st.session_state.get("col_csv_path")
    fdn_rb  = st.session_state.get("fdn_ruby_path")
    if col_rb or fdn_rb:
        st.markdown("---")
        st.markdown("#### 🏛️ Column & Foundation Files")
        dc1, dc2, dc3 = st.columns(3)
        if col_rb and Path(col_rb).exists():
            with dc1, open(col_rb, "rb") as f:
                st.download_button("📥 Columns .rb", f.read(),
                                   file_name=Path(col_rb).name, mime="text/plain",
                                   use_container_width=True)
        if col_csv_f and Path(col_csv_f).exists():
            with dc2, open(col_csv_f, "rb") as f:
                st.download_button("📊 Columns .csv", f.read(),
                                   file_name=Path(col_csv_f).name, mime="text/csv",
                                   use_container_width=True)
        if fdn_rb and Path(fdn_rb).exists():
            with dc3, open(fdn_rb, "rb") as f:
                st.download_button("📥 Foundations .rb", f.read(),
                                   file_name=Path(fdn_rb).name, mime="text/plain",
                                   use_container_width=True)

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
