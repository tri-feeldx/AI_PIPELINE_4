"""
FeelDX Slab Extractor v2  --  Streamlit app.

Upload â†’ Gemini doc analysis â†’ slab+column extraction â†’ .rb download.
Uses the slab_v2 pipeline exclusively. Does NOT touch app.py.
"""

import copy
import json
import sys
import os
import tempfile
from pathlib import Path

import streamlit as st
import fitz

# ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.slab_v2.config import SlabV2Config
from src.slab_v2.doc_analyze import analyze_document
from src.slab_v2.pipeline import extract_slabs_v2, run_dir
from src.slab_v2.export_ruby import generate_building_ruby
from src.slab_v2.models import ColumnFootprint
from src.slab_v2.readiness import build_model_readiness
from src.slab_v2.height_reconcile import reconcile_heights
from src.column_detector import detect_columns_on_page
from src.building_site_placement import run_building_site_placement_audit

st.set_page_config(page_title="FeelDX Slab v2", layout="wide")


def _save_column_debug_image(page, cols_raw, cols_kept, census_dict, debug_dir):
    """Draw column detection debug image: green=kept, red=filtered."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        draw = ImageDraw.Draw(img)
        scale_x = img.width / page.rect.width
        scale_y = img.height / page.rect.height
        kept_syms = {id(c) for c in cols_kept}
        for c in cols_raw:
            color = "green" if id(c) in kept_syms else "red"
            if c.polygon is None:
                continue
            bx = c.polygon.bounds
            x0, y0 = bx[0] * scale_x, bx[1] * scale_y
            x1, y1 = bx[2] * scale_x, bx[3] * scale_y
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0, y0 - 12), c.symbol, fill=color)
        debug_dir.mkdir(parents=True, exist_ok=True)
        img.save(str(debug_dir / "step_10b_columns.png"))
    except Exception:
        pass


# ---- session state defaults ----------------------------------------------------------------------------------------------
def _display_number(value, digits: int = 0):
    """Return table-safe text for optional numeric values."""
    if value is None or value == "":
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value == 0:
        return "N/A"
    if digits <= 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


_DEFAULTS = {
    "pdf_path": None,
    "pdf_name": None,
    "doc_analysis": None,
    "storeys": None,        # {building_name: [{result, ffl_mm, page_idx}]}
    "ruby_bytes": None,     # {building_name: bytes}
    "site_placement": None, # building_site_placement audit result
    "height_result": None,
    "height_overrides": {},
    "model_readiness": {},
    "wall_source_registry": None,
    "phase": "upload",
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---- sidebar ----------------------------------------------------------------------------------------------------------------------------
def _sidebar_config() -> SlabV2Config:
    cfg = SlabV2Config()
    with st.sidebar:
        st.header("Settings")
        cfg.slab_thickness_mm = st.number_input(
            "Slab thickness (mm)", value=200.0, step=25.0)
        cfg.default_storey_height_mm = st.number_input(
            "Default storey height (mm)", value=3000.0, step=100.0)
        cfg.column_text_search_radius_pt = st.number_input(
            "Column text search radius (pt)", value=40.0, step=5.0)
        st.divider()
        st.subheader("Manual Overrides")
        manual_scale = st.number_input(
            "Scale override (0 = auto)", value=0, step=10,
            help="VD: 100 cho 1:100. Äá»ƒ 0 = tá»± detect.")
        if manual_scale > 0:
            cfg.manual_scale = int(manual_scale)
        model = st.text_input("Gemini model (blank=default)", value="")
        if model:
            cfg.gemini_model = model
        st.divider()
        if st.button("Reset"):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()
    return cfg


# ---- Phase 1: Upload ------------------------------------------------------------------------------------------------------------
def _phase_upload():
    st.header("1. Upload Structural PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded is None:
        return

    tmp = Path(tempfile.gettempdir()) / f"feeldx_v2_{uploaded.name}"
    tmp.write_bytes(uploaded.read())
    st.session_state["pdf_path"] = str(tmp)
    st.session_state["pdf_name"] = uploaded.name

    doc = fitz.open(str(tmp))
    st.success(f"**{uploaded.name}**  --  {doc.page_count} pages")
    doc.close()

    if st.button("Analyze Document", type="primary"):
        st.session_state["phase"] = "analyzing"
        st.session_state["doc_analysis"] = None
        st.session_state["site_placement"] = None
        st.session_state["storeys"] = None
        st.session_state["ruby_bytes"] = None
        st.session_state["height_result"] = None
        st.session_state["height_overrides"] = {}
        st.session_state["model_readiness"] = {}
        st.session_state["wall_source_registry"] = None
        st.rerun()


# ---- Phase 2: Gemini Analysis ------------------------------------------------------------------------------------------
def _phase_analyze(cfg: SlabV2Config):
    st.header("2. Document Analysis (Gemini)")

    if st.session_state["doc_analysis"] is None:
        with st.spinner("Running Gemini document analysis..."):
            try:
                ana = analyze_document(st.session_state["pdf_path"], cfg)
                st.session_state["doc_analysis"] = ana
                st.session_state["phase"] = "analyzed"
            except Exception as e:
                st.error(f"Gemini analysis failed: {e}")
                st.session_state["phase"] = "upload"
                return

    ana = st.session_state["doc_analysis"]

    # buildings & floors
    for b in ana.buildings:
        with st.expander(f"Building: {b.name}", expanded=True):
            rows = []
            for f in b.floors:
                pages_str = ", ".join(str(p + 1) for p in f.pages)
                rows.append({
                    "Level": f.level_id,
                    "FFL (m)": f"{f.ffl_m:.3f}" if f.ffl_m is not None else "N/A",
                    "Pages": pages_str or "N/A",
                })
            st.table(rows)

    # column schedule
    if ana.column_types:
        st.subheader("Column Schedule")
        ct_rows = [{"Symbol": t.symbol,
                     "Width (mm)": _display_number(t.width_mm),
                     "Depth (mm)": _display_number(t.depth_mm),
                     "Material": t.material or "UNKNOWN",
                    "Count": _display_number(t.count_total)}
                    for t in ana.column_types.values()]
        st.table(ct_rows)

    # columns per floor
    if ana.columns_per_floor:
        st.subheader("Columns per Floor")
        cpf_rows = []
        for e in ana.columns_per_floor:
            counts_str = ", ".join(f"{s}:{n}" for s, n in e["counts"].items())
            cpf_rows.append({
                "Building": e.get("building", ""),
                "Level": e["level_id"],
                "Columns": counts_str,
            })
        st.table(cpf_rows)

    # foundation types
    if ana.foundation_types:
        st.subheader("Foundation Schedule")
        fdn_rows = []
        for sym, info in ana.foundation_types.items():
            row = {"Symbol": sym}
            for key, value in info.items():
                row[key] = _display_number(value) if key.endswith("_mm") else value
            fdn_rows.append(row)
        st.table(fdn_rows)

    # orphan columns
    if ana.orphan_columns:
        st.subheader("Orphan Columns")
        st.json(ana.orphan_columns)

    # parked detail pages
    parked = []
    if ana.stair_detail_pages:
        parked.append(f"Stair: pages {[p+1 for p in ana.stair_detail_pages]}")
    if ana.lift_detail_pages:
        parked.append(f"Lift: pages {[p+1 for p in ana.lift_detail_pages]}")
    if ana.foundation_detail_pages:
        parked.append(f"Foundation: pages {[p+1 for p in ana.foundation_detail_pages]}")
    if ana.footing_plan_pages:
        parked.append(f"Footing plans: pages {[p+1 for p in ana.footing_plan_pages]}")
    if parked:
        st.info("Detail pages (parked): " + " | ".join(parked))

    # warnings
    for w in ana.warnings:
        st.warning(w)

    st.success(f"Confidence: {ana.confidence}")

    # ---- Phase 2.5: Building Site Placement ------------------------------------------------------
    if st.session_state["site_placement"] is None:
        if len(ana.buildings) > 1:
            with st.spinner("Running site placement audit..."):
                try:
                    _upload_dir = run_dir(cfg,
                                         st.session_state["pdf_path"])
                    site_report = run_building_site_placement_audit(
                        st.session_state["pdf_path"], str(_upload_dir))
                    st.session_state["site_placement"] = site_report
                except Exception as e:
                    st.warning(f"Site placement failed: {e}")
                    st.session_state["site_placement"] = {}
        else:
            st.session_state["site_placement"] = {}

    site_report = st.session_state["site_placement"]
    if site_report:
        transforms = site_report.get("site_transform", {})
        bld_transforms = transforms.get("building_transforms", {})
        if bld_transforms:
            with st.expander("Site Placement", expanded=True):
                for bname, t in bld_transforms.items():
                    dx = t.get("dx_mm")
                    dy = t.get("dy_mm")
                    status = t.get("status", "not_verified")
                    st.write(f"**{bname}**: dx={dx}mm, dy={dy}mm "
                             f"[{status}]")

    if st.button("Extract All Floors", type="primary"):
        st.session_state["phase"] = "extracting"
        st.session_state["storeys"] = None
        st.session_state["ruby_bytes"] = None
        st.rerun()


# ---- Phase 3: Extraction + Export ------------------------------------------------------------------------------------
def _phase_extract(cfg: SlabV2Config):
    st.header("3. Extraction & Export")
    ana = st.session_state["doc_analysis"]
    pdf_path = st.session_state["pdf_path"]

    if st.session_state["storeys"] is not None:
        _show_results(cfg)
        return

    all_storeys = {}  # building_name -> [{result, ffl_mm, page_idx}]
    total_pages = sum(len(f.pages) for b in ana.buildings for f in b.floors)
    progress = st.progress(0.0)

    # ---- Column types from merged Gemini analysis (no 2nd call) --------------
    all_col_types = {
        sym: {
            "width_mm": t.width_mm,
            "depth_mm": t.depth_mm,
            "material": (t.material or "UNKNOWN").upper(),
        }
        for sym, t in ana.column_types.items()
    }
    steel_column_symbols = sorted(
        sym for sym, t in ana.column_types.items()
        if (t.material or "").upper() == "STEEL"
    )
    v1_col_types = {
        sym: data for sym, data in all_col_types.items()
        if data.get("material") != "STEEL"
    }

    with st.expander("Step: Column Schedule (from doc analysis)", expanded=True):
        if all_col_types:
            st.markdown(
                f"**{len(v1_col_types)} RC/unknown column type(s) used** "
                f"of {len(all_col_types)} total"
            )
            if steel_column_symbols:
                st.info(
                    "Steel column types skipped in RC-only phase: "
                    + ", ".join(steel_column_symbols)
                )
            ct_rows = [{"Symbol": sym, "Width (mm)": _display_number(t["width_mm"]),
                        "Depth (mm)": _display_number(t["depth_mm"]),
                        "Material": t.get("material") or "UNKNOWN",
                        "Count": _display_number(ana.column_types[sym].count_total)}
                       for sym, t in all_col_types.items()]
            st.table(ct_rows)

            if ana.columns_per_floor:
                st.markdown("**Per-floor column counts:**")
                floor_rows = []
                for e in ana.columns_per_floor:
                    cols_str = ", ".join(
                        f"{s}: {n}" for s, n in e["counts"].items())
                    floor_rows.append({
                        "Building": e.get("building", ""),
                        "Level": e["level_id"],
                        "Columns": cols_str,
                    })
                if floor_rows:
                    st.table(floor_rows)
        else:
            st.warning("No column types found in analysis  --  "
                       "columns will be skipped for this PDF.")
        if ana.column_census_report:
            report = ana.column_census_report
            st.caption(
                "Census consistency: "
                f"{report.get('status', 'unknown')} | confidence "
                f"{report.get('requested_confidence', 'unknown')} -> "
                f"{report.get('effective_confidence', 'unknown')}")
            if report.get("backfilled_types"):
                st.info("Recovered from per-floor counts: "
                        + ", ".join(report["backfilled_types"]))

    # ---- Wall types from merged Gemini analysis --------------------------------------------
    v2_wall_types = dict(ana.wall_types)  # symbol -> WallType

    with st.expander("Step: Wall Schedule (from doc analysis)", expanded=True):
        if v2_wall_types:
            st.markdown(f"**{len(v2_wall_types)} wall type(s)**")
            wt_rows = [{"Symbol": sym,
                        "Thickness": _display_number(t.thickness_mm),
                        "Height": _display_number(t.height_mm),
                        "Material": t.material or "N/A",
                        "Category": t.wall_category,
                        "Count": _display_number(t.count_total)}
                       for sym, t in v2_wall_types.items()]
            import pandas as pd
            st.table(pd.DataFrame(wt_rows).astype(str))

            wall_floor_rows = []
            for b in ana.buildings:
                for f in b.floors:
                    if f.walls:
                        walls_str = ", ".join(
                            f"{s}: {n}" for s, n in f.walls.items())
                        wall_floor_rows.append({
                            "Building": b.name,
                            "Level": f.level_id,
                            "Walls": walls_str,
                            "Total": f.total_walls,
                        })
            if wall_floor_rows:
                st.markdown("**Per-floor wall counts:**")
                st.table(wall_floor_rows)
        else:
            st.info("No wall types found in analysis  --  "
                    "walls will use face-based fallback detection.")

    # ---- Save doc_analysis.json into upload folder ----------------------------------------
    _upload_dir = run_dir(cfg, pdf_path)
    try:
        (_upload_dir / "doc_analysis_raw.json").write_text(
            json.dumps(ana.raw, indent=2, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass

    # ---- Height reconciliation (multi-source) ------------------------------------------------------
    with st.spinner("Reconciling storey heights..."):
        height_result = reconcile_heights(
            pdf_path, ana, cfg,
            manual_overrides=st.session_state.get("height_overrides") or {})
        st.session_state["height_result"] = height_result

    with st.expander("Storey Heights (verify before export)", expanded=True):
        ht_rows = []
        for datum in height_result.level_datums:
            ht_rows.append({
                "Building": datum.building,
                "Level": datum.level_id,
                "FFL (m)": f"{(datum.ffl_mm or 0) / 1000:.3f}",
                "Height (mm)": f"{(datum.storey_height_mm or 0):.0f}",
                "Status": datum.status,
                "Confidence": f"{datum.confidence:.2f}",
                "Evidence": ", ".join(datum.supporting_evidence_ids),
            })
        if ht_rows:
            st.table(ht_rows)
        for w in height_result.warnings:
            st.warning(f"Height: {w}")
        if height_result.debug_log:
            with st.expander("Height debug log", expanded=False):
                for line in height_result.debug_log:
                    st.caption(line)

    # One document-level wall source registry is shared by all parallel
    # floor-plan workers. It contains only serializable topology/profile data.
    if st.session_state.get("wall_source_registry") is None:
        from src.slab_v2.wall_profile_resolver import build_wall_source_registry
        with st.spinner("Resolving wall key plan and elevation profiles..."):
            st.session_state["wall_source_registry"] = build_wall_source_registry(
                pdf_path, ana, cfg, run_dir(cfg, pdf_path), use_ai=True)
    wall_source_registry = st.session_state.get("wall_source_registry") or {}
    with st.expander("Wall Source Intelligence", expanded=True):
        st.write(f"Status: **{wall_source_registry.get('status', 'review')}**")
        st.json({
            "source_pages": wall_source_registry.get("source_pages", []),
            "keyplan": wall_source_registry.get("keyplan", {}),
            "profiles": wall_source_registry.get("profiles", {}),
            "warnings": wall_source_registry.get("warnings", []),
        })
        for name in ("wall_keyplan_topology.png",):
            image = run_dir(cfg, pdf_path) / name
            if image.exists():
                st.image(str(image), caption=image.stem,
                         use_container_width=True)
        for image in sorted(run_dir(cfg, pdf_path).glob(
                "wall_elevation_candidates_p*.png")):
            st.image(str(image), caption=image.stem,
                     use_container_width=True)

    # ---- Build flat task list for parallel extraction ----------------------------------
    tasks = []
    for b in ana.buildings:
        for f in b.floors:
            if "roof" in f.level_id.lower():
                continue
            ffl_m = height_result.get_ffl(b.name, f.level_id)
            if ffl_m is None:
                ffl_m = f.ffl_m or 0.0
            for pi in f.pages:
                tasks.append({
                    "pi": pi, "building": b.name, "level_id": f.level_id,
                    "ffl_m": ffl_m, "floor_columns": f.columns or {},
                    "floor_walls": f.walls or {},
                })

    # ---- Parallel page extraction --------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    _progress_lock = threading.Lock()
    _done_count = [0]

    def _extract_one_page(task):
        pi = task["pi"]
        try:
            result = extract_slabs_v2(
                pdf_path, pi, cfg, use_ai=True,
                column_types=ana.column_types if ana.column_types else None,
                columns_per_floor=task.get("floor_columns") or None,
                wall_types=v2_wall_types if v2_wall_types else None,
                walls_per_floor=task.get("floor_walls") or None,
                wall_source_registry=wall_source_registry)
        except Exception as e:
            return {**task, "result": None, "error": str(e),
                    "v1_cols": [], "v1_cols_raw": [], "col_logs": []}

        col_logs = []
        v1_cols_raw_out = []
        v1_cols = []

        if result.columns:
            cc = {}
            for c in result.columns:
                cc[c.symbol] = cc.get(c.symbol, 0) + 1
            detail = ", ".join(f"{s}: {n}" for s, n in sorted(cc.items()))
            col_logs.append(
                ("success", f"Census-aware columns: {len(result.columns)} "
                 f"({detail})"))
        elif (result.status == "OK" and result.slabs
                and v1_col_types and result.scale):
            tdoc = fitz.open(pdf_path)
            tpage = tdoc[pi]
            v1_scale = int(round(result.scale))
            try:
                v1_cols_raw_out = detect_columns_on_page(
                    tpage, v1_col_types, v1_scale, pi,
                    building=task["building"], level=task["level_id"])
                fc = task["floor_columns"]
                if fc:
                    v1_cols_in = [c for c in v1_cols_raw_out
                                  if c.symbol in fc]
                    v1_cols_out = [c for c in v1_cols_raw_out
                                   if c.symbol not in fc]
                    extra = [c for c in v1_cols_out
                             if c.symbol in v1_col_types]
                    v1_cols = v1_cols_in + extra
                    truly_dropped = [c.symbol for c in v1_cols_out
                                     if c.symbol not in v1_col_types]
                    if extra:
                        col_logs.append(
                            ("caption",
                             f"{len(extra)} col(s) not in floor census "
                             f"but in schedule  --  kept: "
                             f"{', '.join(sorted(set(c.symbol for c in extra)))}"))
                    if truly_dropped:
                        col_logs.append(
                            ("caption",
                             f"Dropped {len(truly_dropped)} col(s) "
                             f"unknown to schedule: "
                             f"{', '.join(sorted(set(truly_dropped)))}"))
                else:
                    v1_cols = list(v1_cols_raw_out)

                result.columns = [
                    ColumnFootprint(
                        symbol=c.symbol, polygon=c.polygon,
                        w_mm=c.width_mm, d_mm=c.depth_mm, labeled=True)
                    for c in v1_cols
                ]
                _save_column_debug_image(
                    tpage, v1_cols_raw_out, v1_cols,
                    fc, Path(result.debug_dir))

                if v1_cols:
                    cc = {}
                    for c in v1_cols:
                        cc[c.symbol] = cc.get(c.symbol, 0) + 1
                    detail = ", ".join(f"{s}: {n}"
                                       for s, n in sorted(cc.items()))
                    col_logs.append(
                        ("success",
                         f"Columns detected: {len(v1_cols)} ({detail})"))
                else:
                    col_logs.append(("warning", "v1 detector found 0 columns"))
            except Exception as e:
                col_logs.append(("error", f"Column detection failed: {e}"))
            tdoc.close()
        elif not v1_col_types:
            col_logs.append(("caption", "No column types  --  skipping detection"))
        elif result.status == "OK" and result.slabs and not result.scale:
            col_logs.append(("caption", "No scale  --  skipping column detection"))

        with _progress_lock:
            _done_count[0] += 1
        print(f"[Parallel] Page {pi + 1} done ({_done_count[0]}/{total_pages})")

        return {**task, "result": result, "error": None,
                "v1_cols": v1_cols, "v1_cols_raw": v1_cols_raw_out,
                "col_logs": col_logs}

    max_w = max(1, min(len(tasks), int(cfg.extraction_max_workers)))
    st.info(f"Extracting {len(tasks)} pages with {max_w} parallel workers...")
    page_results = {}
    with ThreadPoolExecutor(max_workers=max_w) as executor:
        futures = {executor.submit(_extract_one_page, t): t for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            page_results[r["pi"]] = r
            progress.progress(_done_count[0] / max(total_pages, 1))

    # ---- Render results on main thread (Streamlit UI) ----------------------------
    for b in ana.buildings:
        storeys = []
        for f in b.floors:
            if "roof" in f.level_id.lower():
                continue
            ffl_m = height_result.get_ffl(b.name, f.level_id)
            if ffl_m is None:
                ffl_m = f.ffl_m or 0.0
            for pi in f.pages:
                r = page_results.get(pi)
                if not r:
                    continue

                with st.status(f"Page {pi + 1}  --  {f.level_id}",
                               expanded=False) as status:
                    if r["error"]:
                        st.error(f"Page {pi + 1} failed: {r['error']}")
                        status.update(label=f"Page {pi + 1} FAILED",
                                      state="error")
                        continue

                    result = r["result"]
                    if result.status != "OK" or not result.slabs:
                        st.warning(f"Page {pi + 1}: status={result.status}")
                        status.update(label=f"Page {pi + 1} SKIP",
                                      state="error")
                        continue

                    for log_type, log_msg in r["col_logs"]:
                        getattr(st, log_type)(log_msg)

                    area = sum(s.get("area_m2") or 0 for s in result.slabs)
                    other_area = sum(
                        s.get("area_m2") or 0
                        for s in result.other_floor_systems)
                    st.write(f"Area: {area:.1f} m\u00b2 | "
                             f"Other floor: {other_area:.1f} m\u00b2 | "
                             f"Openings: {len(result.resolved_openings)} | "
                             f"Walls: {len(result.walls)} | "
                             f"Columns: {len(result.columns)} | "
                             f"Scale: 1:{result.scale}")
                    if result.opening_judgement:
                        st.caption(
                            "Opening Judge: "
                            f"{result.opening_judgement.get('status', 'n/a')} "
                            f"({result.opening_judgement.get('confidence', 0):.2f})")
                    if result.floor_system_readiness:
                        fs = result.floor_system_readiness
                        st.caption(
                            "Floor System Resolver: "
                            f"{fs.get('status', 'review')} | "
                            f"PT={len(fs.get('pt_slab_ids', []))} | "
                            f"Other={len(fs.get('other_floor_ids', []))} | "
                            f"Unknown={len(fs.get('unknown_ids', []))}")
                        system_rows = []
                        area_factor = ((25.4 / 72.0 * float(result.scale)) ** 2
                                       / 1_000_000.0) if result.scale else 0.0
                        for candidate in result.floor_system_candidates:
                            if candidate.separator_segment is None:
                                continue
                            system_rows.append({
                                "Candidate": candidate.id,
                                "Status": candidate.cut_status,
                                "Stair/cap source": candidate.terminal_source,
                                "Terminal error (pt)": (
                                    f"{candidate.terminal_alignment_error_pt:.3f}"
                                    if candidate.terminal_alignment_error_pt
                                    is not None else "N/A"),
                                "Bounded cut (m²)": f"{candidate.bounded_cut_area_pt2 * area_factor:.2f}",
                                "Prevented overcut (m²)": f"{candidate.rejected_extension_area_pt2 * area_factor:.2f}",
                                "Direction": candidate.extension_direction,
                            })
                        if system_rows:
                            with st.expander("Floor-system cut audit",
                                             expanded=False):
                                st.table(system_rows)
                    missing_cols = result.column_detection_report.get("missing", {})
                    if missing_cols:
                        st.warning("Missing expected RC columns: "
                                   + ", ".join(
                                       f"{s}:{n}" for s, n in missing_cols.items()))

                    for w in result.warnings:
                        st.caption(f"WARN: {w}")

                    debug_dir = Path(result.debug_dir)
                    imgs = sorted(debug_dir.glob("step_*.png"))
                    if imgs:
                        with st.expander("Debug images", expanded=False):
                            for img in imgs:
                                st.image(str(img), caption=img.stem,
                                         use_container_width=True)

                    status.update(
                        label=f"Page {pi + 1}  --  {area:.0f} m\u00b2, "
                              f"{len(result.columns)} cols",
                        state="complete")

                    datum = next((d for d in height_result.level_datums
                                  if d.building == b.name
                                  and d.level_id == f.level_id), None)
                    storeys.append({
                        "result": result, "ffl_mm": ffl_m * 1000.0,
                        "page_idx": pi, "level_id": f.level_id,
                        "level_name": f.level_name,
                        "height_mm": (datum.storey_height_mm
                                      if datum else None),
                        "height_status": (datum.status
                                          if datum else "default_unsafe"),
                    })

        if not storeys:
            st.error(f"Building {b.name}: no valid floors extracted")
            continue

        # Recover only symbol identity from adjacent floors. The target floor
        # must already contain a real vector rectangle at the projected grid
        # location; no column geometry is invented here.
        try:
            from src.slab_v2.column_reconciler import reconcile_columns_across_floors
            cross_floor = reconcile_columns_across_floors(
                pdf_path, storeys, run_dir(cfg, pdf_path))
            if cross_floor.get("recoveries"):
                st.info(f"{b.name}: recovered {len(cross_floor['recoveries'])} "
                        "RC column assignment(s) from cross-floor vector evidence.")
        except Exception as exc:
            st.warning(f"{b.name}: cross-floor column reconciliation failed: {exc}")

        try:
            from src.slab_v2.core_wall_topology import (
                reconcile_core_wall_topologies)
            wall_reconciliation = reconcile_core_wall_topologies(
                storeys, run_dir(cfg, pdf_path))
            reference = wall_reconciliation.get("reference") or {}
            if reference:
                st.caption(
                    f"Core topology reference: page {reference.get('page')} "
                    f"({reference.get('completeness')}/7 LW symbols); "
                    "target-page vectors remain geometry authority.")
        except Exception as exc:
            st.warning(f"{b.name}: core wall topology audit failed: {exc}")

        # junk filter with slab fallback
        def _max_slab(st_entry):
            return max((s.get("area_m2") or 0 for s in
                        st_entry["result"].slabs), default=0)

        biggest = max(_max_slab(s) for s in storeys) if storeys else 0
        kept = []
        deferred = []
        for s in storeys:
            a = _max_slab(s)
            if biggest > 0 and a < 0.10 * biggest:
                deferred.append(s)
            else:
                kept.append(s)

        # fallback: clone slab from nearest valid level
        for s in deferred:
            a = _max_slab(s)
            if kept:
                nearest = min(kept,
                              key=lambda k: abs(k["ffl_mm"] - s["ffl_mm"]))
                cloned = copy.copy(s["result"])
                cloned.slabs = copy.deepcopy(nearest["result"].slabs)
                cloned.scale = nearest["result"].scale or s["result"].scale
                s["result"] = cloned
                kept.append(s)
                st.warning(
                    f"Page {s['page_idx'] + 1}: slab {a:.1f} m\u00b2 too small "
                    f"(<10% of {biggest:.1f} m\u00b2)  --  using slab shape from "
                    f"page {nearest['page_idx'] + 1} (fallback)")
            else:
                st.warning(
                    f"Page {s['page_idx'] + 1}: slab {a:.1f} m\u00b2  --  "
                    f"excluded, no fallback available")

        all_storeys[b.name] = kept

    progress.progress(1.0)

    # generate .rb files
    ruby_bytes = {}
    for bname, storeys in all_storeys.items():
        if not storeys:
            continue
        # re-open doc for page objects needed by generate_building_ruby
        doc = fitz.open(pdf_path)
        readiness = build_model_readiness(
            storeys, height_result.level_datums, bname)
        st.session_state["model_readiness"][bname] = readiness
        storey_dicts = [{"result": s["result"], "page": doc[s["page_idx"]],
                         "ffl_mm": s["ffl_mm"],
                         "level_id": s["level_id"],
                         "level_name": s.get("level_name", ""),
                         "height_mm": s.get("height_mm"),
                         "height_status": s.get("height_status")}
                        for s in storeys]
        stem = Path(pdf_path).stem
        bid = "".join(ch if ch.isalnum() else "_" for ch in bname).strip("_")
        out_dir = run_dir(cfg, pdf_path)
        try:
            from dataclasses import asdict
            (out_dir / f"model_readiness_{bid}.json").write_text(
                json.dumps(asdict(readiness), indent=2,
                           ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        prefix = "final_model" if readiness.model_status == "final" \
            else "debug_model"
        out_path = str(out_dir / f"{prefix}_{stem}_{bid}.rb")
        # site placement offset for this building
        site_report = st.session_state.get("site_placement") or {}
        bld_transforms = site_report.get("site_transform", {}).get(
            "building_transforms", {})
        t = bld_transforms.get(bname, {})
        offset = (t.get("dx_mm") or 0.0, t.get("dy_mm") or 0.0)

        try:
            path, warnings = generate_building_ruby(
                storey_dicts, out_path, cfg, site_offset_mm=offset,
                building_name=bname, readiness_report=readiness)
            for w in warnings:
                st.warning(f"Export: {w}")
            ruby_bytes[bname] = Path(path).read_bytes()
        except Exception as e:
            st.error(f"Ruby export failed for {bname}: {e}")
        doc.close()

    st.session_state["storeys"] = all_storeys
    st.session_state["ruby_bytes"] = ruby_bytes
    st.session_state["phase"] = "extracted"
    st.rerun()


def _show_results(cfg: SlabV2Config):
    """Display extraction results and download buttons."""
    all_storeys = st.session_state["storeys"]
    ruby_bytes = st.session_state["ruby_bytes"] or {}

    height_result = st.session_state.get("height_result")
    if height_result:
        with st.expander("Level Datum Report", expanded=True):
            tabs = st.tabs(["Final Levels", "Source Planner", "Measurements",
                            "Consensus", "Conflicts", "Raw JSON",
                            "Manual Overrides"])
            with tabs[0]:
                st.table([{
                    "Building": d.building,
                    "Level": d.level_id,
                    "FFL (m)": f"{(d.ffl_mm or 0) / 1000:.3f}",
                    "Height (mm)": f"{(d.storey_height_mm or 0):.0f}",
                    "Status": d.status,
                    "Confidence": f"{d.confidence:.2f}",
                    "Evidence": ", ".join(d.supporting_evidence_ids),
                } for d in height_result.level_datums])
            with tabs[1]:
                planner = height_result.source_planner or {}
                st.json(planner)
                candidate_image = (run_dir(
                    cfg, st.session_state["pdf_path"]) /
                    "height_candidate_pages.png")
                if candidate_image.exists():
                    st.image(str(candidate_image),
                             caption="CPU candidate pages sent to Gemini",
                             use_container_width=True)
                viewport_images = sorted(run_dir(
                    cfg, st.session_state["pdf_path"]).glob(
                        "height_viewports_p*.png"))
                for image in viewport_images:
                    st.image(str(image), caption=image.stem,
                             use_container_width=True)
            with tabs[2]:
                st.dataframe([{
                    "ID": e.id, "Building": e.building,
                    "From": e.from_level or "ABS", "To": e.to_level,
                    "Type": e.evidence_type, "Value (mm)": e.value_mm,
                    "Page": e.page_index + 1 if e.page_index >= 0 else "N/A",
                    "Viewport": e.viewport_id or "N/A",
                    "Scale": (f"1:{e.scale_ratio:g}" if e.scale_ratio else "N/A"),
                    "Scale status": e.scale_status or "N/A",
                    "Duplicate of": e.duplicate_of or "",
                    "Confidence": e.confidence, "Source": e.source_text,
                } for e in height_result.evidence],
                    use_container_width=True)
                evidence_images = sorted(
                    run_dir(cfg, st.session_state["pdf_path"]).glob(
                        "height_datum_measurements_p*.png"))
                for image in evidence_images:
                    st.image(str(image), caption=image.stem,
                             use_container_width=True)
            with tabs[3]:
                if height_result.consensus_report:
                    st.dataframe(height_result.consensus_report,
                                 use_container_width=True)
                else:
                    st.info("No independent measured-height consensus available.")
            with tabs[4]:
                if height_result.conflicts:
                    st.dataframe(height_result.conflicts,
                                 use_container_width=True)
                else:
                    st.success("No height evidence conflicts.")
            with tabs[5]:
                from dataclasses import asdict
                st.json({
                    "levels": [asdict(d) for d in height_result.level_datums],
                    "evidence": [asdict(e) for e in height_result.evidence],
                    "conflicts": height_result.conflicts,
                    "source_planner": height_result.source_planner,
                    "consensus": height_result.consensus_report,
                })
            with tabs[6]:
                existing = st.session_state.get("height_overrides") or {}
                override_rows = [{
                    "Building": d.building, "Level": d.level_id,
                    "Solved FFL (m)": (d.ffl_mm or 0) / 1000.0,
                    "Manual FFL (m)": (existing.get(
                        f"{d.building}/{d.level_id}", "") / 1000.0
                        if existing.get(f"{d.building}/{d.level_id}")
                        is not None else ""),
                } for d in height_result.level_datums]
                edited = st.data_editor(
                    override_rows, use_container_width=True,
                    disabled=["Building", "Level", "Solved FFL (m)"],
                    key="height_override_editor")
                if st.button("Apply Height Overrides", type="primary"):
                    overrides = {}
                    records = (edited.to_dict("records")
                               if hasattr(edited, "to_dict") else edited)
                    for row in records:
                        value = row.get("Manual FFL (m)")
                        if value not in (None, ""):
                            overrides[f"{row['Building']}/{row['Level']}"] = (
                                float(value) * 1000.0)
                    st.session_state["height_overrides"] = overrides
                    st.session_state["storeys"] = None
                    st.session_state["ruby_bytes"] = None
                    st.session_state["phase"] = "analyzing"
                    st.rerun()

    for bname, storeys in all_storeys.items():
        st.subheader(f"Building: {bname}")
        readiness = (st.session_state.get("model_readiness") or {}).get(bname)
        if readiness:
            if readiness.model_status == "final":
                st.success("Model readiness: FINAL VERIFIED")
            else:
                st.error("Model readiness: UNVERIFIED / DEBUG ONLY")
            st.table([{
                "Slab": readiness.slab_status,
                "Openings": readiness.opening_status,
                "Walls": readiness.wall_status,
                "Wall junctions": readiness.wall_junction_status,
                "RC columns": readiness.column_status,
                "Shaft solids": readiness.shaft_render_status,
                "Height": readiness.height_status,
                "Model": readiness.model_status,
                "Reasons": "; ".join(readiness.reasons),
            }])

        rows = []
        col_summary = {}  # symbol -> {floors, total}
        for s in storeys:
            r = s["result"]
            area = sum(sl.get("area_m2") or 0 for sl in r.slabs)
            other_area = sum(
                sl.get("area_m2") or 0 for sl in r.other_floor_systems)
            # per-page column breakdown
            col_breakdown = {}
            for c in r.columns:
                col_breakdown[c.symbol] = col_breakdown.get(c.symbol, 0) + 1
                col_summary.setdefault(c.symbol, {"total": 0, "floors": []})
                col_summary[c.symbol]["total"] += 1
            col_str = ", ".join(f"{sym}:{n}" for sym, n in
                                sorted(col_breakdown.items())) or "N/A"
            if col_breakdown:
                col_summary_entry = col_str
                for sym in col_breakdown:
                    ffl_label = f"{s['ffl_mm'] / 1000:.1f}m"
                    col_summary[sym]["floors"].append(ffl_label)
            rows.append({
                "Page": r.page_index + 1,
                "FFL (m)": f"{s['ffl_mm'] / 1000:.3f}",
                "Area (m\u00b2)": f"{area:.1f}",
                "Other floor (m\u00b2)": f"{other_area:.1f}",
                "Openings": len(r.resolved_openings),
                "Walls": len(r.walls),
                "Columns": col_str,
                "Scale": f"1:{r.scale}" if r.scale else "N/A",
                "Columns (s)": f"{r.timings.get('columns', 0):.1f}",
                "Walls (s)": f"{r.timings.get('walls', 0):.1f}",
                "Openings (s)": f"{r.timings.get('openings', 0):.1f}",
                "Total (s)": f"{r.timings.get('total', 0):.1f}",
            })
        st.table(rows)

        with st.expander("Wall Topology + Elevation Profiles", expanded=True):
            wall_tabs = st.tabs([
                "Plan Topology", "Elevation Profiles",
                "Expected vs Detected", "3D Mapping", "Raw JSON"])
            with wall_tabs[0]:
                for s in storeys:
                    debug_dir = Path(s["result"].debug_dir)
                    for pattern in ("wall_plan_candidates_p*.png",
                                    "wall_grid_registration_p*.png",
                                    "wall_plan_resolved_p*.png",
                                    "wall_junction_candidates_p*.png",
                                    "wall_junction_resolved_p*.png"):
                        for image in sorted(debug_dir.glob(pattern)):
                            st.image(str(image), caption=image.stem,
                                     use_container_width=True)
            with wall_tabs[1]:
                source_dir = run_dir(cfg, st.session_state["pdf_path"])
                profile_json = source_dir / "wall_elevation_profiles.json"
                if profile_json.exists():
                    st.json(json.loads(profile_json.read_text(encoding="utf-8")))
                for image in sorted(source_dir.glob(
                        "wall_elevation_candidates_p*.png")):
                    st.image(str(image), caption=image.stem,
                             use_container_width=True)
            with wall_tabs[2]:
                wall_rows = []
                for s in storeys:
                    report = s["result"].wall_readiness or {}
                    wall_rows.append({
                        "Page": s["result"].page_index + 1,
                        "Status": report.get("status", "review"),
                        "Expected": report.get("expected", {}),
                        "Detected": report.get("detected", {}),
                        "Missing": report.get("missing", {}),
                    })
                st.dataframe(wall_rows, use_container_width=True)
            with wall_tabs[3]:
                mapping_rows = []
                for s in storeys:
                    report = s["result"].wall_readiness or {}
                    for row in report.get("instances", []):
                        mapping_rows.append({
                            "Page": s["result"].page_index + 1,
                            "Wall": row.get("symbol"),
                            "Status": row.get("status"),
                            "Length (mm)": row.get("length_mm"),
                            "Recovered (mm)": row.get(
                                "recovered_missing_length_mm"),
                            "Grid scope": (f"{row.get('grid_start', '?')} -> "
                                           f"{row.get('grid_end', '?')}"),
                            "Profile": row.get("profile_id"),
                            "Evidence": ", ".join(row.get("evidence", [])),
                        })
                if mapping_rows:
                    st.dataframe(mapping_rows, use_container_width=True)
                else:
                    st.info("No dedicated wall profile mapping on these pages.")
            with wall_tabs[4]:
                st.json({
                    "source_registry": st.session_state.get(
                        "wall_source_registry") or {},
                    "pages": [{
                        "page": s["result"].page_index + 1,
                        "wall_readiness": s["result"].wall_readiness,
                    } for s in storeys],
                })

        # column detection summary
        if col_summary:
            with st.expander("Column Detection Summary", expanded=True):
                total_all = sum(v["total"] for v in col_summary.values())
                st.markdown(f"**Total columns detected: {total_all}**")
                sum_rows = []
                for sym, info in sorted(col_summary.items()):
                    sum_rows.append({
                        "Symbol": sym,
                        "Count": info["total"],
                        "Floors": ", ".join(info["floors"]),
                    })
                st.table(sum_rows)
                page_reports = []
                for s in storeys:
                    result = s["result"]
                    report = result.column_detection_report or {}
                    page_reports.append({
                        "Page": result.page_index + 1,
                        "Status": report.get("status", "review"),
                        "Expected": report.get("expected", {}),
                        "Detected": report.get("detected", {}),
                        "Missing": report.get("missing", {}),
                        "Extra": report.get("extra", {}),
                        "Ambiguous": report.get("ambiguous_count", 0),
                    })
                    debug_dir = Path(result.debug_dir)
                    for pattern in ("column_candidates_p*.png",
                                    "column_assignment_p*.png"):
                        for image in sorted(debug_dir.glob(pattern)):
                            st.image(str(image), caption=image.stem,
                                     use_container_width=True)
                st.dataframe(page_reports, use_container_width=True)

        if bname in ruby_bytes:
            stem = Path(st.session_state["pdf_name"]).stem
            bid = "".join(ch if ch.isalnum() else "_" for ch in bname).strip("_")
            prefix = ("final_model" if readiness
                      and readiness.model_status == "final"
                      else "debug_model")
            file_name = f"{prefix}_{stem}_{bid}.rb"
            st.download_button(
                f"Download {file_name}",
                data=ruby_bytes[bname],
                file_name=file_name,
                mime="text/plain",
            )
            st.caption("SketchUp Ruby Console command:")
            st.code(
                f"load File.join(Dir.home, 'Downloads', '{file_name}')",
                language="ruby")

    # doc_analysis.json download
    ana = st.session_state["doc_analysis"]
    if ana and ana.raw:
        st.download_button(
            "Download doc_analysis.json",
            data=json.dumps(ana.raw, indent=2, ensure_ascii=False),
            file_name="doc_analysis.json",
            mime="application/json",
        )

    if st.button("Start Over"):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ---- main routing ------------------------------------------------------------------------------------------------------------------
def main():
    st.title("FeelDX Slab Extractor v2")
    cfg = _sidebar_config()
    phase = st.session_state["phase"]

    if phase == "upload":
        _phase_upload()
    elif phase in ("analyzing", "analyzed"):
        _phase_analyze(cfg)
    elif phase in ("extracting", "extracted"):
        _phase_extract(cfg)


if __name__ == "__main__":
    main()
