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
        'Upload bản vẽ kết cấu Úc (vector PDF từ Bluebeam/AutoCAD). '
        'Hệ thống sẽ tự động nhận diện sàn, trích xuất cao độ FFL, '
        'và tạo mô hình 3D SketchUp với kích thước chính xác 100%.'
        '</div>', unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Kéo thả PDF vào đây hoặc nhấp để chọn file",
        type=["pdf"],
        help="Hỗ trợ vector PDF. Scanned PDF cho độ chính xác thấp hơn.",
        label_visibility="visible",
    )

    if uploaded:
        tmp_path = Path(tempfile.gettempdir()) / f"feeldx_{sid}_{uploaded.name}"
        tmp_path.write_bytes(uploaded.read())
        st.session_state["pdf_path"] = str(tmp_path)

        with st.spinner("Đang đọc PDF..."):
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
        st.markdown("#### 📊 Thông tin PDF")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng số trang", meta["page_count"])
        c2.metric("Floor Plan (auto)", floor_plan_count)
        c3.metric("Khổ giấy", page_size_label)
        c4.metric("File", uploaded.name[:18] + ("…" if len(uploaded.name) > 18 else ""))

        if meta.get("creator"):
            st.markdown(
                f'<div class="success-box">✅ Vector PDF xác nhận — Creator: '
                f'<b>{meta["creator"]}</b></div>', unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("Tiếp theo: Chọn trang →", type="primary", use_container_width=True):
            st.session_state["step"] = 2
            _rerun()


# ── STEP 2: Select Pages ────────────────────────────────────────────────────────
def step2_select_pages():
    st.markdown('<div class="step-header">🗂️ Step 2 — Chọn trang Floor Plan</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Các trang floor plan đã được tự động đánh dấu. '
        'Tick/bỏ tick để tùy chỉnh. Đặt đúng tỉ lệ bản vẽ để kích thước 3D chính xác.'
        '</div>', unsafe_allow_html=True,
    )

    page_infos = st.session_state["page_infos"]
    if not page_infos:
        st.error("Không tìm thấy thông tin trang.")
        return

    import fitz
    from src.pdf_processor import load_pdf, get_page_thumbnail

    # ── Scale selector ──────────────────────────────────────────────────────────
    scale_auto = next((p["scale"] for p in page_infos if p.get("scale")), None)

    st.markdown("#### ⚙️ Tỉ lệ bản vẽ")
    col_s, col_hint = st.columns([1, 2])
    with col_s:
        scale_input = st.number_input(
            "Scale (1 : N)",
            min_value=10, max_value=2000,
            value=scale_auto or 100,
            step=10,
            help="Xem title block của bản vẽ. Phổ biến: 1:100, 1:200, 1:50",
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
                '<div class="warn-box" style="margin-top:28px;">⚠️ Không tự detect được — '
                'hãy nhập thủ công. Xem title block PDF.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 📄 Chọn trang cần xử lý")

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
        f'{"✅" if n_sel > 0 else "⚠️"} <b>{n_sel} trang</b> được chọn để phân tích.'
        '</div>', unsafe_allow_html=True,
    )

    # ── Smart Floor Detection (AI + keyword fallback) ───────────────────────
    if n_sel > 0:
        st.markdown("---")
        st.markdown("#### 🧠 AI Floor Detection")
        st.markdown(
            '<div class="info-box">Gemini đọc toàn bộ text trong PDF, tự nhận dạng '
            'building/tầng/FFL, chọn đúng pages cần xử lý (bỏ sections, details, elevations).</div>',
            unsafe_allow_html=True,
        )

        smart_result = st.session_state.get("smart_detect_result")

        col_ai, col_kw, col_reset = st.columns([2, 2, 1])
        with col_ai:
            if st.button("🤖 Phân tích AI (Gemini)", use_container_width=True, type="primary"):
                from src.ai_floor_analyzer import analyze_floor_structure
                from src.floor_detector import FloorDetectResult, FloorGroup
                with st.spinner(f"Gemini đang đọc {n_sel} trang (~5-10 giây)..."):
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
                    except Exception as e:
                        st.error(f"Gemini error: {e}")
                _rerun()
        with col_kw:
            if st.button("🔍 Keyword detection (nhanh)", use_container_width=True):
                from src.floor_detector import detect_unique_floors
                with st.spinner(f"Đang scan {n_sel} trang..."):
                    result = detect_unique_floors(
                        st.session_state["pdf_path"],
                        sorted(new_selected),
                    )
                st.session_state["smart_detect_result"] = result
                st.session_state["ai_floor_result"] = None
                st.session_state["smart_detect_done"] = False
                _rerun()
        with col_reset:
            if smart_result and st.button("↺ Reset", use_container_width=True):
                st.session_state["smart_detect_result"] = None
                st.session_state["smart_detect_done"] = False
                st.session_state["ai_floor_result"] = None
                st.session_state["ai_floor_output_path"] = None
                _rerun()

        # ── Show AI raw JSON for review ──────────────────────────────────
        ai_result = st.session_state.get("ai_floor_result")
        ai_path   = st.session_state.get("ai_floor_output_path")
        if ai_result:
            with st.expander("🔍 Xem kết quả Gemini (raw JSON)"):
                st.json(ai_result)
            if ai_path:
                try:
                    with open(ai_path, "rb") as f:
                        st.download_button(
                            "📥 Tải gemini_floors.json",
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
                "all_pages": "Keyword: không đủ tín hiệu — xử lý tất cả",
                "ai_gemini": "Gemini AI ✅ (độ tin cậy cao nhất)",
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
                    f'<div class="success-box">✅ Phát hiện <b>{smart_result.floor_count} tầng</b> '
                    f'— xử lý <b>{n_proc} pages</b>, bỏ qua <b>{n_skip} pages</b>'
                    f'<br><small>Basis: {basis_txt}</small></div>',
                    unsafe_allow_html=True,
                )

                # Results table
                import pandas as pd
                rows = []
                for g in smart_result.groups:
                    rows.append({
                        "Tầng": g.floor_label,
                        "Canonical": f"Page {g.canonical_page + 1}",
                        "Supplement": ", ".join(f"P{p+1}" for p in g.supplemental_pages) or "—",
                        "Bỏ qua": ", ".join(f"P{p+1}" for p in g.skipped_pages) or "—",
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
                         else f"Tiếp với tất cả {n_sel} pages")
                if st.button(label, type="primary", use_container_width=True):
                    st.session_state["smart_detect_done"] = True
                    st.session_state["slab_results"] = {}   # clear cache → re-process
                    st.session_state["step"] = 3
                    _rerun()
            with cb:
                if smart_result.detection_basis != "all_pages":
                    if st.button(f"Dùng tất cả {n_sel} pages (bỏ qua smart)", use_container_width=True):
                        st.session_state["smart_detect_done"] = False
                        st.session_state["slab_results"] = {}
                        st.session_state["step"] = 3
                        _rerun()

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Quay lại", use_container_width=True):
            st.session_state["step"] = 1
            _rerun()
    with col_next:
        if n_sel > 0:
            if not st.session_state.get("smart_detect_result"):
                if st.button("Tiếp: Detect Slabs →", type="primary", use_container_width=True):
                    st.session_state["slab_results"] = {}
                    st.session_state["step"] = 3
                    _rerun()
        else:
            st.warning("Chọn ít nhất 1 trang.")


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
        filled = build_polygons_from_drawings(drawings)
        recon = reconstruct_closed_polygons(drawings)
        filtered = filter_slab_candidates(filled + recon, page)

        page_debug = {}
        for step_fn, key, step_args in [
            (save_step1_raw_paths, "step1", (page, drawings,      f"{debug_base}_step1_raw.png")),
            (save_step2_polygons,  "step2", (page, filled + recon, f"{debug_base}_step2_polys.png")),
            (save_step3_filtered,  "step3", (page, filtered,       f"{debug_base}_step3_filtered.png")),
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
        f'<div class="success-box">✅ Hoàn tất — phát hiện <b>{total} slab</b> '
        f'trên {n} trang.</div>', unsafe_allow_html=True,
    )

    # Log file download
    log_path = st.session_state.get("log_path")
    if log_path and Path(log_path).exists():
        with open(log_path, "rb") as lf:
            st.download_button(
                "📋 Tải Log File (gửi cho engineer để debug)",
                lf.read(),
                file_name=Path(log_path).name,
                mime="text/plain",
                key="dl_log_step3",
            )

    # Debug image viewer
    st.markdown("---")
    st.markdown("#### 📸 Debug Images — từng bước xử lý")
    for page_idx in selected_pages:
        page_imgs = debug_imgs.get(page_idx, {})
        n_slabs = len(results.get(page_idx, []))
        with st.expander(f"📄 Trang {page_idx + 1} — {n_slabs} slab", expanded=(n_slabs > 0)):
            tabs = st.tabs(["① Raw Paths", "② Polygons", "③ Filtered", "④ Labeled", "⑤ Final"])
            step_keys = ["step1", "step2", "step3", "step4", "step5"]
            for tab, key in zip(tabs, step_keys):
                with tab:
                    img_path = page_imgs.get(key)
                    if img_path and Path(img_path).exists():
                        st.image(img_path, use_container_width=True)
                        with open(img_path, "rb") as f:
                            st.download_button(
                                f"⬇ Tải {key}.png", f.read(),
                                file_name=Path(img_path).name, mime="image/png",
                                key=f"dl_{page_idx}_{key}",
                            )
                    else:
                        st.info("Ảnh không khả dụng.")

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Quay lại", use_container_width=True):
            # Clear cache so re-entering step 3 will re-process with any new settings
            st.session_state["slab_results"] = {}
            st.session_state["step"] = 2
            _rerun()
    with col_next:
        if total > 0:
            if st.button("Tiếp: Review →", type="primary", use_container_width=True):
                st.session_state["step"] = 4
                _rerun()
        else:
            st.markdown(
                '<div class="warn-box">⚠️ Không tìm thấy slab. Thử: điều chỉnh scale, '
                'chọn trang khác, hoặc xem debug ①②③ để phân tích.</div>',
                unsafe_allow_html=True,
            )


# ── STEP 3: Detect Slabs ────────────────────────────────────────────────────────
def step3_detect():
    st.markdown('<div class="step-header">🔍 Step 3 — Nhận diện Slab</div>',
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
            f'<div class="info-box">🧠 Smart mode — {n_floors} tầng / '
            f'{len(pages_to_process)} pages (bỏ qua {n_skipped} pages trùng)</div>',
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
    progress = st.progress(0, text=f"Đang xử lý {n} trang song song...")
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
                    f'<div class="warn-box">⚠️ Lỗi trang {page_idx + 1}: {e}</div>',
                    unsafe_allow_html=True,
                )
            done_count += 1
            n_found = len(results.get(page_idx, []))
            progress.progress(
                done_count / n,
                text=f"Trang {page_idx + 1}: {n_found} slab — ({done_count}/{n} trang xong)",
            )

    # Update session state only after all workers finish (thread-safe)
    st.session_state["slab_results"] = results
    st.session_state["debug_images"] = debug_imgs
    all_slabs = [s for page_slabs in results.values() for s in page_slabs]

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
    st.markdown('<div class="step-header">📋 Step 4 — Review & Chỉnh sửa</div>',
                unsafe_allow_html=True)

    all_slabs = st.session_state["final_slabs"]
    debug_imgs = st.session_state["debug_images"]
    selected_pages = st.session_state["selected_pages"]

    if not all_slabs:
        st.markdown('<div class="warn-box">⚠️ Không có slab để review. Quay lại bước 3.</div>',
                    unsafe_allow_html=True)
        if st.button("← Quay lại"):
            st.session_state["step"] = 3
            _rerun()
        return

    total_area = sum(s.area_m2 for s in all_slabs if s.area_m2 > 0)
    ffls = [s.ffl_m for s in all_slabs if s.ffl_m is not None]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng Slab", len(all_slabs))
    c2.metric("Tổng Diện Tích", f"{total_area:.1f} m²")
    c3.metric("Cao độ min", f"{min(ffls):.3f}m" if ffls else "N/A")
    c4.metric("Cao độ max", f"{max(ffls):.3f}m" if ffls else "N/A")

    st.markdown("---")
    st.markdown("#### ✏️ Chỉnh sửa dữ liệu Slab")
    st.markdown(
        '<div class="info-box">Double-click vào ô để sửa Label hoặc FFL. '
        'Thickness cố định 200mm.</div>', unsafe_allow_html=True,
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
            "Page": st.column_config.NumberColumn("Trang", disabled=True, width="small"),
            "FFL (m)": st.column_config.NumberColumn("FFL (m)", format="%.3f", width="medium"),
            "Thickness (mm)": st.column_config.NumberColumn("Dày (mm)", disabled=True, width="small"),
            "Area (m²)": st.column_config.NumberColumn("Diện tích (m²)", format="%.2f", width="medium"),
            "Source": st.column_config.TextColumn("Nguồn", disabled=True, width="small"),
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
    st.markdown("#### 🖼️ Ảnh kết quả cuối")
    for page_idx in selected_pages:
        img_path = debug_imgs.get(page_idx, {}).get("step5")
        if img_path and Path(img_path).exists():
            with st.expander(f"Trang {page_idx+1}", expanded=True):
                st.image(img_path, use_container_width=True)

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Quay lại", use_container_width=True):
            st.session_state["step"] = 3
            _rerun()
    with col_next:
        if st.button("Tiếp: Tạo 3D Model →", type="primary", use_container_width=True):
            st.session_state["step"] = 5
            _rerun()


# ── STEP 5: Generate ────────────────────────────────────────────────────────────
def step5_generate():
    st.markdown('<div class="step-header">⚙️ Step 5 — Tạo SketchUp 3D Model</div>',
                unsafe_allow_html=True)

    all_slabs = st.session_state["final_slabs"]
    if not all_slabs:
        st.error("Không có slab để xuất. Quay lại bước 3.")
        return

    from src.model_builder import generate_ruby_script, generate_slab_csv

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruby_path = str(OUTPUT_DIR / f"slabs_{ts}.rb")
    csv_path  = str(OUTPUT_DIR / f"slabs_{ts}.csv")

    with st.spinner("Đang tạo Ruby script cho SketchUp..."):
        ruby_content = generate_ruby_script(all_slabs, ruby_path)
        generate_slab_csv(all_slabs, csv_path)

    st.session_state["ruby_script"] = ruby_content
    st.session_state["ruby_path"]   = ruby_path
    st.session_state["csv_path"]    = csv_path

    c1, c2, c3 = st.columns(3)
    c1.metric("Slabs trong script", len(all_slabs))
    c2.metric("Kích thước script", f"{len(ruby_content)/1024:.1f} KB")
    c3.metric("Thickness", "200 mm (cố định)")

    st.markdown("---")
    st.markdown("#### ⬇️ Download Files")
    col_rb, col_csv = st.columns(2)
    with col_rb:
        st.download_button(
            "📥 Tải .rb Script (SketchUp Ruby)",
            ruby_content.encode("utf-8"),
            file_name=Path(ruby_path).name,
            mime="text/plain",
            use_container_width=True,
        )
    with col_csv:
        with open(csv_path, "rb") as f:
            st.download_button(
                "📊 Tải Slab Data (.csv)",
                f.read(),
                file_name=Path(csv_path).name,
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("#### 👁️ Preview Ruby Script")
    with st.expander("Xem code (80 dòng đầu)", expanded=False):
        preview = "\n".join(ruby_content.split("\n")[:80])
        st.code(preview, language="ruby")

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Quay lại", use_container_width=True):
            st.session_state["step"] = 4
            _rerun()
    with col_next:
        if st.button("Tiếp: Hướng dẫn SketchUp →", type="primary", use_container_width=True):
            st.session_state["step"] = 6
            _rerun()


# ── STEP 6: Export / Instructions ──────────────────────────────────────────────
def step6_done():
    st.markdown('<div class="step-header">🎉 Step 6 — Import vào SketchUp</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="success-box">🎉 <b>Xong rồi!</b> Làm theo hướng dẫn bên dưới '
        'để import slab vào SketchUp 2026.</div>', unsafe_allow_html=True,
    )

    st.markdown("""
---
### 🚀 Hướng dẫn import vào SketchUp 2026

| Bước | Hành động |
|------|-----------|
| 1 | Mở **SketchUp** → `Window` → `Ruby Console` |
| 2 | **Download** file `.rb` từ Step 5 |
| 3 | Mở file `.rb` bằng Notepad → **Copy toàn bộ** |
| 4 | **Paste** vào Ruby Console → nhấn **Enter** |
| 5 | Nhấn **Z** (Zoom Extents) để xem toàn bộ model |

---
### 📦 Kết quả trong SketchUp
- Mỗi slab = khối 3D solid dày **200mm**
- Mỗi tầng = 1 **Layer riêng** (tên theo FFL)
- Mỗi tầng có **màu material riêng**
- Đơn vị mô hình: **mm** (tự động set)

---
### 🔍 Kiểm tra độ chính xác
1. Dùng **Tape Measure** (`T`) đo kích thước slab
2. So sánh với annotation trên PDF
3. Nếu sai → quay Step 2, điều chỉnh **Scale**

---
### 🛠️ Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| Không thấy slab | Xem Ruby Console có lỗi không |
| Kích thước sai | Điều chỉnh Scale ở Step 2 |
| Cao độ sai | Sửa FFL ở Step 4 |
| Thiếu slab | Chọn thêm trang ở Step 2 |
| Slab chồng lên nhau | Kiểm tra debug Step ③ |
""")

    ruby_content = st.session_state.get("ruby_script", "")
    if ruby_content:
        st.markdown("---")
        with st.expander("📋 Copy Script nhanh tại đây"):
            st.code(ruby_content, language="ruby")
            st.download_button(
                "⬇️ Download lại .rb",
                ruby_content.encode("utf-8"),
                file_name="slabs.rb",
                mime="text/plain",
            )

    st.markdown("---")
    st.markdown(
        '<div class="info-box">💡 <b>Phase tiếp theo:</b> Sau khi slab OK, '
        'sẽ thêm cột, dầm, vách, và cốt thép.</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔄 Xử lý PDF khác", use_container_width=True, type="primary"):
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
