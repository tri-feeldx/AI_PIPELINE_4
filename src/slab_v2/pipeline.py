"""
slab_v2 orchestrator.

extract_slabs_v2(pdf_path, page_index, cfg, use_ai=True)
  Stage A  vector extraction (style classes)
  Stage B  planarize ALL non-frame classes -> face graph (step_04)
  Stage C  Gemini Round 1 elects the slab linework classes (step_06 — the
           "blue lines"), augment closes gaps; then take EVERY face those
           classes enclose — all of them, and nothing from other classes
           (user rule: "lấy hết line xanh, đừng lấy thêm"). Union ->
           components >= min_component_frac of largest -> exterior rings.
  Elements X-cross opening symbols (stair/lift/shaft) -> footprints;
           the 2D slab stays gross, holes are cut at Ruby export.
  Dims     dimension annotations parsed page-wide; the modal implied scale
           cross-checks (and on conflict overrides) the text scale, so the
           exported millimetres match what the drawing actually measures.
  Stage E  debug images at every step + result.json

use_ai=False skips the election and unions faces of ALL classes — debug /
fallback mode; over-includes on pages with annotation loops.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import fitz

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import SlabV2Result
from src.slab_v2 import vector_extract, planarize
from src.slab_v2.debug_render import PageRenderer

# one numbered run folder per CLI invocation / upload:
# debug_slab_v2/<stem>/upload<N>/page_<P>/... — runs never overwrite each
# other, N is resolved once per process per document
_RUN_DIRS: dict[str, Path] = {}


def run_dir(cfg: SlabV2Config, pdf_path: str) -> Path:
    stem = Path(pdf_path).stem.replace(" ", "_")
    key = f"{cfg.debug_dir}|{stem}"
    if key not in _RUN_DIRS:
        root = Path(cfg.debug_dir) / stem
        nums = []
        if root.exists():
            for d in root.iterdir():
                m = re.fullmatch(r"upload(\d+)", d.name)
                if m and d.is_dir():
                    nums.append(int(m.group(1)))
        path = root / f"upload{max(nums, default=0) + 1}"
        path.mkdir(parents=True, exist_ok=True)
        _RUN_DIRS[key] = path
    return _RUN_DIRS[key]


def _content_rect(page: fitz.Page) -> fitz.Rect:
    from src.vision_refiner import find_legend_rect, find_drawing_content_rect
    legend = find_legend_rect(page)
    return find_drawing_content_rect(page, legend)


def _detect_scale(doc: fitz.Document, page_index: int) -> int | None:
    from src.pdf_processor import extract_text_blocks, detect_scale_from_blocks
    blocks = extract_text_blocks(doc[page_index])
    return detect_scale_from_blocks(blocks)


def extract_slabs_v2(
    pdf_path: str,
    page_index: int,
    cfg: SlabV2Config | None = None,
    use_ai: bool = True,
    scale: int | None = None,
    column_types: dict | None = None,
    columns_per_floor: dict | None = None,
) -> SlabV2Result:
    cfg = cfg or SlabV2Config()
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    out_dir = run_dir(cfg, pdf_path) / f"page_{page_index + 1}"
    result = SlabV2Result(page_index=page_index, debug_dir=str(out_dir))
    rend = PageRenderer(page, cfg, str(out_dir))
    t0 = time.time()

    # ── Stage A ───────────────────────────────────────────────────────────
    content = _content_rect(page)
    content_area = content.width * content.height
    paths, classes = vector_extract.extract_paths(page, cfg, content)
    result.style_classes = classes
    result.timings["stage_a"] = time.time() - t0

    rend.step00_page_raster()
    rend.step01_paths_by_style(paths, classes)
    rend.step02_style_legend_sheet(classes)

    # ── Stage B ───────────────────────────────────────────────────────────
    t1 = time.time()
    all_ids = {c.id for c in classes if c.role != "FRAME"}
    fg_all = planarize.build_face_graph(paths, all_ids, cfg, content_area)
    result.timings["stage_b"] = time.time() - t1

    rend.step03_planarized(fg_all)
    rend.faces_numbered(fg_all, "step_04_faces_all.png",
                        content_area_pt2=content_area)

    if not fg_all.faces:
        result.status = "NO_FACES"
        _write_result_json(result, out_dir)
        return result

    if cfg.manual_scale:
        text_scale = cfg.manual_scale
    else:
        text_scale = scale if scale is not None else _detect_scale(doc, page_index)

    # ── face source: deterministic default, optional Gemini election ──────
    fg_src = fg_all
    if use_ai:
        from src.slab_v2 import ai_select
        words = page.get_text("words")
        title_words = [w[4] for w in words
                       if w[0] > page.rect.width * 0.78
                       or w[1] > page.rect.height * 0.88]
        ctx = ai_select.SelectionContext(
            page=page, paths=paths, classes=classes, cfg=cfg,
            content_rect=content, content_area_pt2=content_area,
            renderer=rend, fg_all=fg_all, scale=text_scale,
            title_text=" ".join(title_words))
        try:
            election = ai_select.elect_classes(ctx, [])
            if election.warning:
                result.warnings.append(election.warning)
            base_ids = (set(election.slab_edge_classes)
                        | set(election.supporting_classes))
            fg_sel, _used, added = planarize.augment_until_closed(
                paths, base_ids, classes, cfg, content_area)
            if added:
                election.supporting_classes = sorted(
                    set(election.supporting_classes) | set(added))
            result.election = election
            rend.step06_elected_classes(paths, classes,
                                        election.slab_edge_classes,
                                        election.supporting_classes)
            if fg_sel.faces:
                fg_src = fg_sel
            else:
                result.warnings.append(
                    "elected classes produced no faces — falling back to "
                    "all-classes graph")
        except ai_select.AIError as e:
            result.warnings.append(
                f"class election failed ({e}) — falling back to "
                f"all-classes graph")
        result.gemini_calls = ctx.calls_used

    rend.faces_numbered(fg_src, "step_07_faces_candidates.png",
                        content_area_pt2=content_area)

    # ── deterministic assembly ────────────────────────────────────────────
    t2 = time.time()
    keep_ids = [f.id for f in fg_src.faces
                if f.area_pt2 >= cfg.min_keep_face_frac * content_area]
    if not keep_ids:
        result.status = "NO_FACES"
        result.warnings.append("no face above min_keep_face_frac")
        _write_result_json(result, out_dir)
        return result

    gross, err = planarize.assemble_slab_polygon(
        fg_src.faces, keep_ids, [],
        min_component_frac=cfg.min_component_frac)
    if gross is None:
        result.status = "NO_FACES"
        result.warnings.append(f"assembly failed: {err}")
        _write_result_json(result, out_dir)
        return result

    frac = gross.area / max(content_area, 1.0)
    if frac < 0.05:
        result.warnings.append(
            f"assembled slab covers only {frac:.0%} of the drawing area — "
            f"check step_04/step_07")
    elif frac > 0.90:
        result.warnings.append(
            f"assembled slab covers {frac:.0%} of the drawing area — may "
            f"include annotation loops, check step_07")

    parts = sorted(getattr(gross, "geoms", [gross]), key=lambda g: -g.area)
    slabs = [{"label": f"SLAB_{j + 1}" if len(parts) > 1 else "SLAB",
              "polygon_pdf": g, "void_count": 0}
             for j, g in enumerate(parts)]
    result.timings["assembly"] = time.time() - t2

    rend.step08_assembled_slab(slabs, keep_ids)

    # ── elements: X-cross openings ────────────────────────────────────────
    from src.slab_v2 import elements as elements_mod
    elems, elem_warnings = elements_mod.extract_elements(
        page, fg_all, cfg, content, content_area, paths=paths)
    # keep only openings that intersect a slab
    from shapely.ops import unary_union
    slab_union = unary_union([s["polygon_pdf"] for s in slabs])
    result.elements = [e for e in elems
                       if e.polygon.intersects(slab_union)]
    result.warnings.extend(elem_warnings)
    rend.step09_elements(result.elements)

    # ── dimension verification + scale self-check (always on) ─────────────
    from src.slab_v2 import verify
    t3 = time.time()
    dims = verify.parse_dimensions(page, classes, cfg, content)
    report = verify.verify_selection(slabs, dims, text_scale, content, cfg)
    result.verification = report
    rend.step_dimensions(dims, "step_09b_dimensions.png")
    result.timings["verify"] = time.time() - t3

    final_scale = text_scale
    if cfg.manual_scale:
        pass  # manual override — skip dimension-based reconciliation
    elif report.scale_precise:
        # strong consensus: the continuous measured scale calibrates the mm
        # conversion (catches non-integer viewport scales the nominal bucket
        # absorbs — "khách hàng yêu cầu đúng kích thước")
        final_scale = report.scale_precise
        if text_scale and \
                abs(report.scale_precise - text_scale) / text_scale > 0.001:
            result.warnings.append(
                f"using dimension-calibrated scale 1:{report.scale_precise:.2f} "
                f"(text says 1:{text_scale})")
    elif report.scale_used and text_scale and \
            abs(report.scale_used - text_scale) / text_scale > 0.05:
        result.warnings.append(
            f"text scale 1:{text_scale} contradicts dimension-measured "
            f"scale 1:{report.scale_used} — using the measured scale")
        final_scale = report.scale_used
    elif report.scale_used and not text_scale:
        final_scale = report.scale_used
    elif not report.n_dims_associated:
        result.warnings.append(
            "scale unverified: no dimension annotations associated")
    result.scale = final_scale

    result.status = "OK"

    # ── columns (text-anchor-then-shape v2; fallback to shape-first v1) ───
    if column_types is not None:
        from src.slab_v2 import columns_v2 as columns_mod
        cols, col_warnings = columns_mod.extract_columns_v2(
            page, paths, slab_union, final_scale, column_types, cfg,
            elements=result.elements,
            columns_per_floor_census=columns_per_floor)
        result.columns = cols
        result.warnings.extend(col_warnings)
        rend.step11_columns(cols)

    # ── Stage E: mm conversion + final image ──────────────────────────────
    from src.coordinate_mapper import transform_polygon
    for s in slabs:
        if final_scale:
            mm = transform_polygon(s["polygon_pdf"], page, final_scale,
                                   page.rect.x0, page.rect.y1)
            s["polygon_mm"] = mm
            s["area_m2"] = mm.area / 1_000_000.0
        else:
            s["polygon_mm"] = None
            s["area_m2"] = None
    result.slabs = slabs

    rend.step10_final(slabs, result.elements)
    _write_result_json(result, out_dir)
    return result


def _write_result_json(result: SlabV2Result, out_dir: Path) -> None:
    def poly_coords(geom):
        if geom is None:
            return None
        geoms = getattr(geom, "geoms", [geom])
        return [
            {"exterior": [list(c) for c in g.exterior.coords],
             "holes": [[list(c) for c in h.coords] for h in g.interiors]}
            for g in geoms
        ]

    v = result.verification
    payload = {
        "page_index": result.page_index,
        "page_number": result.page_index + 1,
        "status": result.status,
        "scale": result.scale,
        "gemini_calls": result.gemini_calls,
        "warnings": result.warnings,
        "timings_s": {k: round(v_, 2) for k, v_ in result.timings.items()},
        "style_classes": vector_extract.class_summary_table(
            result.style_classes),
        "election": (
            {"slab_edge_classes": result.election.slab_edge_classes,
             "supporting_classes": result.election.supporting_classes,
             "reasoning": result.election.reasoning}
            if result.election else None),
        "verification": (
            {"scale_used": v.scale_used,
             "scale_precise": round(v.scale_precise, 4) or None,
             "scale_consistency": v.scale_consistency,
             "n_dims_associated": v.n_dims_associated,
             "dimension_report": {
                 "n_matched_edges": len(v.edge_matches),
                 "max_rel_err": (max(m["rel_err"] for m in v.edge_matches)
                                 if v.edge_matches else None),
             },
             "edge_matches": [
                 {k: x for k, x in m.items() if k != "edge"}
                 for m in v.edge_matches],
             "extent_check": v.extent_check,
             "failures": v.failures}
            if v else None),
        "elements": [
            {"type": e.type, "label": e.label,
             "area_pt2": round(e.area_pt2, 1),
             "polygon_pdf_pts": poly_coords(e.polygon)}
            for e in result.elements],
        "columns": [
            {"symbol": c.symbol, "w_mm": c.w_mm, "d_mm": c.d_mm,
             "labeled": c.labeled,
             "polygon_pdf_pts": poly_coords(c.polygon)}
            for c in result.columns],
        "slabs": [
            {"label": s["label"],
             "area_m2": s.get("area_m2"),
             "polygon_pdf_pts": poly_coords(s["polygon_pdf"])}
            for s in result.slabs],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
