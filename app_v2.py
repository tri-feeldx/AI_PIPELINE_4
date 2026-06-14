"""
FeelDX Slab Extractor v2 — Streamlit app.

Upload → Gemini doc analysis → slab+column extraction → .rb download.
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
from src.slab_v2.height_reconcile import reconcile_heights
from src.column_detector import detect_columns_on_page
from src.building_site_placement import run_building_site_placement_audit

st.set_page_config(page_title="FeelDX Slab v2", layout="wide")


# ── session state defaults ───────────────────────────────────────────────
_DEFAULTS = {
    "pdf_path": None,
    "pdf_name": None,
    "doc_analysis": None,
    "storeys": None,        # {building_name: [{result, ffl_mm, page_idx}]}
    "ruby_bytes": None,     # {building_name: bytes}
    "site_placement": None, # building_site_placement audit result
    "phase": "upload",
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── sidebar ──────────────────────────────────────────────────────────────
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
            help="VD: 100 cho 1:100. Để 0 = tự detect.")
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


# ── Phase 1: Upload ──────────────────────────────────────────────────────
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
    st.success(f"**{uploaded.name}** — {doc.page_count} pages")
    doc.close()

    if st.button("Analyze Document", type="primary"):
        st.session_state["phase"] = "analyzing"
        st.session_state["doc_analysis"] = None
        st.session_state["storeys"] = None
        st.session_state["ruby_bytes"] = None
        st.rerun()


# ── Phase 2: Gemini Analysis ─────────────────────────────────────────────
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
                    "FFL (m)": f.ffl_m if f.ffl_m is not None else "—",
                    "Pages": pages_str,
                })
            st.table(rows)

    # column schedule
    if ana.column_types:
        st.subheader("Column Schedule")
        ct_rows = [{"Symbol": t.symbol,
                     "Width (mm)": t.width_mm,
                     "Depth (mm)": t.depth_mm,
                     "Count": t.count_total or "—"}
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
        fdn_rows = [{"Symbol": sym, **info}
                     for sym, info in ana.foundation_types.items()]
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

    # ── Phase 2.5: Building Site Placement ───────────────────────────
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


# ── Phase 3: Extraction + Export ──────────────────────────────────────────
def _phase_extract(cfg: SlabV2Config):
    st.header("3. Extraction & Export")
    ana = st.session_state["doc_analysis"]
    pdf_path = st.session_state["pdf_path"]

    if st.session_state["storeys"] is not None:
        _show_results(cfg)
        return

    doc = fitz.open(pdf_path)
    all_storeys = {}  # building_name -> [{result, ffl_mm, page_idx}]
    total_pages = sum(len(f.pages) for b in ana.buildings for f in b.floors)
    progress = st.progress(0.0)
    done = 0

    # ── Column types from merged Gemini analysis (no 2nd call) ───────
    v1_col_types = {
        sym: {"width_mm": t.width_mm, "depth_mm": t.depth_mm}
        for sym, t in ana.column_types.items()
    }

    with st.expander("Step: Column Schedule (from doc analysis)", expanded=True):
        if v1_col_types:
            st.markdown(f"**{len(v1_col_types)} column type(s)**")
            ct_rows = [{"Symbol": sym, "Width (mm)": t["width_mm"],
                        "Depth (mm)": t["depth_mm"],
                        "Count": ana.column_types[sym].count_total or "—"}
                       for sym, t in v1_col_types.items()]
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
            st.error("No column types found in analysis!")

    # ── Save doc_analysis.json into upload folder ────────────────────
    _upload_dir = run_dir(cfg, pdf_path)
    try:
        (_upload_dir / "doc_analysis.json").write_text(
            json.dumps(ana.raw, indent=2, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass

    # ── Height reconciliation (multi-source) ───────────────────────────
    with st.spinner("Reconciling storey heights..."):
        height_result = reconcile_heights(pdf_path, ana, cfg)

    with st.expander("Storey Heights (verify before export)", expanded=True):
        ht_rows = []
        for fh in height_result.floors:
            src_str = ", ".join(
                f"{k}={v:.3f}" for k, v in fh.sources.items())
            ht_rows.append({
                "Level": fh.level_id,
                "FFL (m)": f"{fh.ffl_m:.3f}",
                "Height (mm)": f"{fh.storey_height_mm:.0f}",
                "Confidence": fh.confidence,
                "Sources": src_str,
            })
        if ht_rows:
            st.table(ht_rows)
        for w in height_result.warnings:
            st.warning(f"Height: {w}")
        if height_result.debug_log:
            with st.expander("Height debug log", expanded=False):
                for line in height_result.debug_log:
                    st.caption(line)

    for b in ana.buildings:
        storeys = []

        for f in b.floors:
            ffl_m = height_result.get_ffl(b.name, f.level_id)
            if ffl_m is None:
                ffl_m = f.ffl_m or 0.0

            for pi in f.pages:
                done += 1
                progress.progress(done / max(total_pages, 1))

                with st.status(f"Page {pi + 1} — {f.level_id}",
                               expanded=False) as status:
                    # ── slab extraction (no columns — v2 pipeline) ───────
                    try:
                        result = extract_slabs_v2(
                            pdf_path, pi, cfg, use_ai=True)
                    except Exception as e:
                        st.error(f"Page {pi + 1} failed: {e}")
                        status.update(label=f"Page {pi + 1} FAILED",
                                      state="error")
                        continue

                    if result.status != "OK" or not result.slabs:
                        st.warning(f"Page {pi + 1}: status={result.status}")
                        status.update(label=f"Page {pi + 1} SKIP",
                                      state="error")
                        continue

                    # ── v1 column detection on this page ─────────────────
                    if v1_col_types and result.scale:
                        page = doc[pi]
                        v1_scale = int(round(result.scale))
                        try:
                            v1_cols = detect_columns_on_page(
                                page, v1_col_types, v1_scale, pi,
                                building=b.name, level=f.level_id)
                            result.columns = [
                                ColumnFootprint(
                                    symbol=c.symbol,
                                    polygon=c.polygon,
                                    w_mm=c.width_mm,
                                    d_mm=c.depth_mm,
                                    labeled=True)
                                for c in v1_cols
                            ]
                            # column detection debug
                            if v1_cols:
                                col_counts = {}
                                for c in v1_cols:
                                    col_counts[c.symbol] = \
                                        col_counts.get(c.symbol, 0) + 1
                                col_detail = ", ".join(
                                    f"{s}: {n}" for s, n in
                                    sorted(col_counts.items()))
                                st.success(
                                    f"Columns detected: {len(v1_cols)} "
                                    f"({col_detail})")
                            else:
                                st.warning("v1 detector found 0 columns")
                        except Exception as e:
                            st.error(f"Column detection failed: {e}")
                    elif not v1_col_types:
                        st.caption("No column types — skipping detection")
                    elif not result.scale:
                        st.caption("No scale — skipping column detection")

                    area = sum(s.get("area_m2") or 0 for s in result.slabs)
                    st.write(f"Area: {area:.1f} m² | "
                             f"Openings: {len(result.elements)} | "
                             f"Columns: {len(result.columns)} | "
                             f"Scale: 1:{result.scale}")

                    for w in result.warnings:
                        st.caption(f"WARN: {w}")

                    # debug images
                    debug_dir = Path(result.debug_dir)
                    imgs = sorted(debug_dir.glob("step_*.png"))
                    if imgs:
                        with st.expander("Debug images", expanded=False):
                            for img in imgs:
                                st.image(str(img), caption=img.stem,
                                         use_container_width=True)

                    status.update(
                        label=f"Page {pi + 1} — {area:.0f} m², "
                              f"{len(result.columns)} cols",
                        state="complete")

                    storeys.append({
                        "result": result, "ffl_mm": ffl_m * 1000.0,
                        "page_idx": pi, "level_id": f.level_id,
                    })

        if not storeys:
            st.error(f"Building {b.name}: no valid floors extracted")
            continue

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
                    f"Page {s['page_idx'] + 1}: slab {a:.1f} m² too small "
                    f"(<10% of {biggest:.1f} m²) — using slab shape from "
                    f"page {nearest['page_idx'] + 1} (fallback)")
            else:
                st.warning(
                    f"Page {s['page_idx'] + 1}: slab {a:.1f} m² — "
                    f"excluded, no fallback available")

        all_storeys[b.name] = kept

    doc.close()
    progress.progress(1.0)

    # generate .rb files
    ruby_bytes = {}
    for bname, storeys in all_storeys.items():
        if not storeys:
            continue
        # re-open doc for page objects needed by generate_building_ruby
        doc = fitz.open(pdf_path)
        storey_dicts = [{"result": s["result"], "page": doc[s["page_idx"]],
                         "ffl_mm": s["ffl_mm"],
                         "level_id": s["level_id"]} for s in storeys]
        stem = Path(pdf_path).stem
        bid = "".join(ch if ch.isalnum() else "_" for ch in bname).strip("_")
        out_dir = run_dir(cfg, pdf_path)
        out_path = str(out_dir / f"{stem}_{bid}.rb")
        # site placement offset for this building
        site_report = st.session_state.get("site_placement") or {}
        bld_transforms = site_report.get("site_transform", {}).get(
            "building_transforms", {})
        t = bld_transforms.get(bname, {})
        offset = (t.get("dx_mm") or 0.0, t.get("dy_mm") or 0.0)

        try:
            path, warnings = generate_building_ruby(
                storey_dicts, out_path, cfg, site_offset_mm=offset)
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

    for bname, storeys in all_storeys.items():
        st.subheader(f"Building: {bname}")

        rows = []
        col_summary = {}  # symbol -> {floors, total}
        for s in storeys:
            r = s["result"]
            area = sum(sl.get("area_m2") or 0 for sl in r.slabs)
            # per-page column breakdown
            col_breakdown = {}
            for c in r.columns:
                col_breakdown[c.symbol] = col_breakdown.get(c.symbol, 0) + 1
                col_summary.setdefault(c.symbol, {"total": 0, "floors": []})
                col_summary[c.symbol]["total"] += 1
            col_str = ", ".join(f"{sym}:{n}" for sym, n in
                                sorted(col_breakdown.items())) or "—"
            if col_breakdown:
                col_summary_entry = col_str
                for sym in col_breakdown:
                    ffl_label = f"{s['ffl_mm'] / 1000:.1f}m"
                    col_summary[sym]["floors"].append(ffl_label)
            rows.append({
                "Page": r.page_index + 1,
                "FFL (m)": f"{s['ffl_mm'] / 1000:.3f}",
                "Area (m²)": f"{area:.1f}",
                "Openings": len(r.elements),
                "Columns": col_str,
                "Scale": f"1:{r.scale}" if r.scale else "—",
            })
        st.table(rows)

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

        if bname in ruby_bytes:
            stem = Path(st.session_state["pdf_name"]).stem
            bid = "".join(ch if ch.isalnum() else "_" for ch in bname).strip("_")
            st.download_button(
                f"Download {stem}_{bid}.rb",
                data=ruby_bytes[bname],
                file_name=f"{stem}_{bid}.rb",
                mime="text/plain",
            )

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


# ── main routing ─────────────────────────────────────────────────────────
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
