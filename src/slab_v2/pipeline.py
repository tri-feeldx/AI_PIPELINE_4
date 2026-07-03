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
import threading
import time
from dataclasses import asdict
from pathlib import Path

import fitz

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import ElementFootprint, SlabV2Result
from src.slab_v2 import vector_extract, planarize, plan_viewport
from src.slab_v2.debug_render import PageRenderer
from src.slab_v2 import trace as trace_mod

# one numbered run folder per CLI invocation / upload:
# debug_slab_v2/<stem>/upload<N>/page_<P>/... — runs never overwrite each
# other, N is resolved once per process per document
_RUN_DIRS: dict[str, Path] = {}
_RUN_DIRS_LOCK = threading.Lock()


def run_dir(cfg: SlabV2Config, pdf_path: str) -> Path:
    stem = Path(pdf_path).stem.replace(" ", "_")
    key = f"{cfg.debug_dir}|{stem}"
    if key in _RUN_DIRS:
        return _RUN_DIRS[key]
    with _RUN_DIRS_LOCK:
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


def _page_text_audits(doc: fitz.Document, page_index: int) -> tuple[int | None, dict, dict]:
    from src.pdf_processor import (
        classify_page_role_from_blocks,
        collect_scale_candidates_from_blocks,
        extract_text_blocks,
    )
    blocks = extract_text_blocks(doc[page_index])
    scale_audit = collect_scale_candidates_from_blocks(blocks)
    role_audit = classify_page_role_from_blocks(blocks)
    return scale_audit.get("chosen_scale"), scale_audit, role_audit


def extract_slabs_v2(
    pdf_path: str,
    page_index: int,
    cfg: SlabV2Config | None = None,
    use_ai: bool = True,
    scale: int | None = None,
    column_types: dict | None = None,
    columns_per_floor: dict | None = None,
    wall_types: dict | None = None,
    walls_per_floor: dict | None = None,
    wall_source_registry: dict | None = None,
    steel_census: dict | None = None,
) -> SlabV2Result:
    cfg = cfg or SlabV2Config()
    if cfg.speed_mode:
        cfg.enable_opening_judge = False
        cfg.enable_slab_face_judge = False
        cfg.enable_floor_system_judge = False
        cfg.debug_images = False
        cfg.save_prompt_images = False
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    run_root = run_dir(cfg, pdf_path)
    out_dir = run_root / f"page_{page_index + 1}"
    result = SlabV2Result(page_index=page_index, debug_dir=str(out_dir))
    rend = PageRenderer(page, cfg, str(out_dir))
    t0 = time.time()
    trace_enabled = str(getattr(cfg, "trace_level", "full")).lower() != "off"
    trace_stages: list[dict] = []
    if trace_enabled:
        trace_mod.init_run_trace(run_root, pdf_path, cfg, doc.page_count)

    def _trace(stage: str, status: str = "ok", **payload) -> None:
        if not trace_enabled:
            return
        row = {
            "stage": stage,
            "status": status,
            "elapsed_total_s": round(time.time() - t0, 3),
            **payload,
        }
        trace_stages.append(row)
        trace_mod.append_event(run_root, out_dir, stage, status, payload)

    def _write_page_trace(extra: dict | None = None) -> None:
        if trace_enabled:
            trace_mod.write_page_trace(result, page, out_dir, trace_stages, extra)

    _trace(
        "input_page",
        pdf_path=pdf_path,
        page_count=doc.page_count,
        page=trace_mod._page_snapshot(page),
    )

    # ── Stage A ───────────────────────────────────────────────────────────
    content = _content_rect(page)
    content_area = content.width * content.height
    if content_area <= 0:
        content = page.rect
        content_area = content.width * content.height
    paths, classes = vector_extract.extract_paths(page, cfg, content)
    result.style_classes = classes
    result.timings["stage_a"] = time.time() - t0
    _trace(
        "vector_extract",
        duration_s=round(result.timings["stage_a"], 3),
        content_rect=trace_mod.content_rect_snapshot(content),
        content_area_pt2=round(content_area, 3),
        vector_stats=trace_mod.vector_stats(paths, classes),
    )

    need_prompt_images = bool(use_ai and getattr(cfg, "save_prompt_images", True))
    if cfg.debug_images:
        rend.step00_page_raster()
    if cfg.debug_images or need_prompt_images:
        rend.step01_paths_by_style(paths, classes)
        rend.step02_style_legend_sheet(classes)

    # ── Stage B ───────────────────────────────────────────────────────────
    detected_scale, scale_audit, role_audit = _page_text_audits(doc, page_index)
    result.scale_audit = scale_audit
    result.page_role_classification = role_audit
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"scale_candidates_p{page_index + 1:02d}.json").write_text(
            json.dumps(scale_audit, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out_dir / f"page_role_classification_p{page_index + 1:02d}.json").write_text(
            json.dumps(role_audit, indent=2, ensure_ascii=False),
            encoding="utf-8")
    except Exception as exc:
        result.warnings.append(f"text audit output failed: {exc}")
    _trace(
        "text_audit",
        scale_status=scale_audit.get("status"),
        chosen_scale=scale_audit.get("chosen_scale"),
        scale_candidates=scale_audit.get("candidates", []),
        page_role=role_audit,
    )

    if role_audit.get("role") in {"evidence_only", "foundation_plan"}:
        # informational only — golden pages (loading plans with X-cross
        # openings, foundation sheets) must still extract geometry; export
        # gating decides downstream from result.page_role_classification
        result.warnings.append(
            f"page role classified as {role_audit.get('role')}; geometry "
            f"extracted anyway, downstream export gating applies")
        _trace("page_role_flag", role_audit.get("role"), page_role=role_audit)

    viewport_rect, viewport_audit = plan_viewport.detect_plan_viewport(
        page, paths, content, role_audit)
    result.plan_viewport = viewport_audit
    try:
        plan_viewport.write_plan_viewport_artifacts(
            page, rend, out_dir, page_index + 1, viewport_rect, content,
            viewport_audit)
    except Exception as exc:
        result.warnings.append(f"plan viewport artifact output failed: {exc}")
    area_ref = (
        float(viewport_rect.width * viewport_rect.height)
        if viewport_audit.get("status") == "detected"
        else float(content_area)
    )
    if area_ref <= 0:
        area_ref = content_area
    area_ref_label = (
        "plan_viewport"
        if viewport_audit.get("status") == "detected"
        else "content_rect"
    )
    _trace(
        "plan_viewport",
        status=viewport_audit.get("status", "unknown"),
        area_reference=area_ref_label,
        viewport=viewport_audit,
        analysis_area_pt2=round(area_ref, 3),
    )

    t1 = time.time()
    all_ids = {c.id for c in classes if c.role not in ("FRAME", "HATCH")}
    _boundary_rect = content if cfg.use_content_boundary else None
    fg_all = planarize.build_face_graph(paths, all_ids, cfg, area_ref,
                                         content_rect=_boundary_rect)
    result.timings["stage_b"] = time.time() - t1
    _trace(
        "face_graph_all",
        duration_s=round(result.timings["stage_b"], 3),
        selected_style_ids=sorted(all_ids),
        use_content_boundary=bool(cfg.use_content_boundary),
        face_graph=trace_mod.face_graph_stats(fg_all, area_ref),
        area_reference=area_ref_label,
    )

    if cfg.debug_images:
        rend.step03_planarized(fg_all)
        rend.faces_numbered(fg_all, "step_04_faces_all.png",
                            content_area_pt2=area_ref)

    no_initial_faces = not fg_all.faces
    if no_initial_faces:
        result.warnings.append(
            "all-classes face graph produced no closed faces; continuing to "
            "no-fill boundary resolver if this is a geometry plan")
        _trace("face_graph_all", "no_closed_faces_continue",
               reason="face graph has no faces")
        if not (
            role_audit.get("role") == "geometry_plan"
            and viewport_audit.get("status") == "detected"
        ):
            result.status = "NO_FACES"
            _trace("early_exit", "no_faces",
                   reason="face graph has no faces and page is not a "
                   "detected geometry plan")
            _write_result_json(result, out_dir)
            _write_page_trace({"exit": "NO_FACES"})
            return result

    if cfg.manual_scale:
        text_scale = cfg.manual_scale
        result.scale_audit = {
            **scale_audit,
            "chosen_scale": text_scale,
            "status": "manual_override",
            "reason": f"Manual scale override selected 1:{text_scale}.",
        }
    else:
        text_scale = scale if scale is not None else detected_scale
        if scale is not None:
            result.scale_audit = {
                **scale_audit,
                "chosen_scale": scale,
                "status": "provided_override",
                "reason": f"Caller-provided scale selected 1:{scale}.",
            }
    result.scale = text_scale
    try:
        (out_dir / f"scale_candidates_p{page_index + 1:02d}.json").write_text(
            json.dumps(result.scale_audit, indent=2, ensure_ascii=False),
            encoding="utf-8")
    except Exception:
        pass

    # ── face source: deterministic default, optional Gemini election ──────
    fg_src = fg_all
    ctx = None
    if use_ai:
        from src.slab_v2 import ai_select
        words = page.get_text("words")
        title_words = [w[4] for w in words
                       if w[0] > page.rect.width * 0.78
                       or w[1] > page.rect.height * 0.88]
        ctx = ai_select.SelectionContext(
            page=page, paths=paths, classes=classes, cfg=cfg,
            content_rect=content, content_area_pt2=area_ref,
            renderer=rend, fg_all=fg_all, scale=text_scale,
            title_text=" ".join(title_words))
        try:
            election = ai_select.elect_classes(ctx, [])
            if election.warning:
                result.warnings.append(election.warning)
            base_ids = (set(election.slab_edge_classes)
                        | set(election.supporting_classes))
            fg_sel, _used, added = planarize.augment_until_closed(
                paths, base_ids, classes, cfg, area_ref,
                content_rect=_boundary_rect)
            if added:
                election.supporting_classes = sorted(
                    set(election.supporting_classes) | set(added))
            # face count validation gate
            if (fg_sel.faces
                    and len(fg_sel.faces) < cfg.min_faces_for_election):
                result.warnings.append(
                    f"face count gate: {len(fg_sel.faces)} faces < "
                    f"{cfg.min_faces_for_election} — requesting re-election")
                try:
                    face_fb = [
                        f"Your class selection produced only "
                        f"{len(fg_sel.faces)} closed faces. A structural "
                        f"slab plan typically has 20-100+ faces. Your "
                        f"selected classes (slab_edge="
                        f"{election.slab_edge_classes}) likely included "
                        f"wall/annotation lines instead of the structural "
                        f"slab boundary. Re-examine and select classes "
                        f"whose lines form closed rectangular regions."]
                    election2 = ai_select.elect_classes(ctx, face_fb)
                    base2 = (set(election2.slab_edge_classes)
                             | set(election2.supporting_classes))
                    fg_sel2, _, added2 = planarize.augment_until_closed(
                        paths, base2, classes, cfg, area_ref,
                        content_rect=_boundary_rect)
                    if (fg_sel2.faces
                            and len(fg_sel2.faces) > len(fg_sel.faces)):
                        fg_sel = fg_sel2
                        election = election2
                        if added2:
                            election.supporting_classes = sorted(
                                set(election.supporting_classes)
                                | set(added2))
                except ai_select.AIError:
                    pass
            result.election = election
            if cfg.debug_images:
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

    _trace(
        "ai_class_election",
        status="used" if result.election else "fallback",
        gemini_calls=result.gemini_calls,
        election=(
            {
                "slab_edge_classes": result.election.slab_edge_classes,
                "supporting_classes": result.election.supporting_classes,
                "warning": result.election.warning,
            } if result.election else None
        ),
        candidate_face_graph=trace_mod.face_graph_stats(fg_src, area_ref),
        area_reference=area_ref_label,
    )

    if cfg.debug_images:
        rend.faces_numbered(fg_src, "step_07_faces_candidates.png",
                            content_area_pt2=area_ref)

    # ── deterministic assembly with retry (up to 3 attempts) ─────────────
    _SLAB_MIN_FRAC = 0.10
    t2 = time.time()
    gross, frac = None, 0.0

    def _try_assemble(fg):
        threshold = cfg.min_keep_face_frac * area_ref
        bg_max = cfg.background_face_max_frac * area_ref
        ids = [f.id for f in fg.faces
               if f.area_pt2 >= threshold
               and not (f.depth == 0 and f.parent_id is None
                        and f.area_pt2 > bg_max)]
        if not ids:
            return None, 0.0
        g, _err = planarize.assemble_slab_polygon(
            fg.faces, ids, [],
            min_component_frac=cfg.min_component_frac,
            sliver_heal_pt=cfg.sliver_heal_pt)
        if g is None:
            return None, 0.0
        return g, g.area / max(area_ref, 1.0)

    # Attempt 1: elected classes
    gross, frac = _try_assemble(fg_src)

    # Attempt 2: all-classes face graph
    if (gross is None or frac < _SLAB_MIN_FRAC) and fg_src is not fg_all:
        result.warnings.append(
            f"attempt 1 (elected classes): slab {frac:.0%} of content — "
            f"retrying with all-classes graph")
        g2, f2 = _try_assemble(fg_all)
        if g2 is not None and f2 > frac:
            gross, frac, fg_src = g2, f2, fg_all
            if cfg.debug_images:
                rend.faces_numbered(fg_src, "step_07_faces_candidates.png",
                                    content_area_pt2=area_ref)

    # Attempt 3: re-call Gemini with feedback
    if (gross is None or frac < _SLAB_MIN_FRAC) and use_ai and ctx:
        result.warnings.append(
            f"attempt 2 (all classes): slab {frac:.0%} — "
            f"re-calling Gemini with feedback")
        try:
            fb = [f"Your previous class selection produced a slab covering "
                  f"only {frac:.0%} of the drawing. The slab should cover "
                  f"at least 10%. The selected classes likely picked up the "
                  f"title block or a stamp instead of the actual slab. "
                  f"Please re-examine the drawing and choose different "
                  f"slab_edge_classes."]
            election2 = ai_select.elect_classes(ctx, fb)
            base2 = (set(election2.slab_edge_classes)
                     | set(election2.supporting_classes))
            fg_sel2, _, _ = planarize.augment_until_closed(
                paths, base2, classes, cfg, area_ref,
                content_rect=_boundary_rect)
            if fg_sel2.faces:
                g3, f3 = _try_assemble(fg_sel2)
                if g3 is not None and f3 > frac:
                    gross, frac, fg_src = g3, f3, fg_sel2
                    result.election = election2
                    if cfg.debug_images:
                        rend.step06_elected_classes(
                            paths, classes,
                            election2.slab_edge_classes,
                            election2.supporting_classes)
                        rend.faces_numbered(
                            fg_src, "step_07_faces_candidates.png",
                            content_area_pt2=area_ref)
            result.gemini_calls = ctx.calls_used
        except Exception as e:
            result.warnings.append(f"attempt 3 (re-elect) failed: {e}")

    no_fill_boundary_audit = None
    if (
        (gross is None or frac < _SLAB_MIN_FRAC)
        and role_audit.get("role") == "geometry_plan"
        and viewport_audit.get("status") == "detected"
    ):
        try:
            no_fill_poly, no_fill_boundary_audit = (
                plan_viewport.assemble_irregular_no_fill_slab_boundary(
                    paths, viewport_rect, area_ref,
                    snap_grid=getattr(cfg, "snap_grid_pt", 0.05))
            )
            try:
                plan_viewport.write_no_fill_irregular_artifacts(
                    page, rend, Path(out_dir), page_index + 1,
                    no_fill_poly, viewport_rect, no_fill_boundary_audit)
            except Exception:
                pass
            if no_fill_poly is None:
                # Keep the older four-side resolver as a narrow fallback for
                # simple rectangular no-fill sheets.  The richer v2 audit stays
                # nested so the failure reason remains traceable.
                v2_audit = no_fill_boundary_audit
                no_fill_poly, v1_audit = (
                    plan_viewport.assemble_no_fill_slab_boundary(
                        paths, viewport_rect, area_ref)
                )
                no_fill_boundary_audit = {
                    "schema": "no_fill_slab_boundary_combined_v1",
                    "status": (
                        "verified" if no_fill_poly is not None
                        else "unresolved"
                    ),
                    "primary_v2": v2_audit,
                    "fallback_v1": v1_audit,
                    "reason": (
                        "v2 unresolved; v1 rectangular fallback accepted"
                        if no_fill_poly is not None
                        else "v2 and v1 no-fill resolvers unresolved"
                    ),
                }
            if no_fill_poly is not None:
                gross = no_fill_poly
                frac = gross.area / max(area_ref, 1.0)
                result.warnings.append(
                    "no-fill GA boundary resolver accepted a slab boundary "
                    "from closed vector outline evidence")
                try:
                    plan_viewport.write_slab_boundary_failure_artifacts(
                        page, rend, Path(out_dir), page_index + 1, gross,
                        viewport_rect, no_fill_boundary_audit)
                except Exception:
                    pass
                _trace(
                    "no_fill_slab_boundary",
                    "accepted",
                    slab_fraction_of_area_ref=round(frac, 6),
                    audit=no_fill_boundary_audit,
                    gross_geometry=trace_mod.geometry_summary(gross),
                )
            else:
                _trace(
                    "no_fill_slab_boundary",
                    "unresolved",
                    audit=no_fill_boundary_audit,
                )
        except Exception as exc:
            no_fill_boundary_audit = {
                "schema": "no_fill_slab_boundary_v1",
                "status": "error",
                "reason": str(exc),
            }
            result.warnings.append(
                f"no-fill slab boundary resolver failed: {exc}")
            _trace(
                "no_fill_slab_boundary",
                "error",
                audit=no_fill_boundary_audit,
            )

    if gross is None:
        result.status = "NO_FACES"
        result.warnings.append(
            "all 3 assembly attempts produced no geometry — skipping page")
        _trace(
            "assembly",
            "failed",
            slab_fraction_of_content=round(frac, 4),
            slab_fraction_of_area_ref=round(frac, 4),
            area_reference=area_ref_label,
            face_graph=trace_mod.face_graph_stats(fg_src, area_ref),
        )
        _write_result_json(result, out_dir)
        _write_page_trace({"exit": "NO_FACES_AFTER_ASSEMBLY"})
        return result

    if frac < _SLAB_MIN_FRAC:
        result.status = "NO_EXPORT_TINY_SLAB"
        boundary_audit = plan_viewport.analyze_slab_boundary_failure(
            gross, viewport_rect, paths, area_ref)
        if no_fill_boundary_audit:
            boundary_audit["no_fill_boundary_audit"] = no_fill_boundary_audit
        result.slab_readiness = {
            "status": "blocked",
            "reason": "tiny_slab_fail_closed",
            "slab_fraction_of_content": round(
                float(getattr(gross, "area", 0.0)) / max(content_area, 1.0),
                6),
            "slab_fraction_of_area_ref": round(frac, 6),
            "area_reference": area_ref_label,
            "plan_viewport": viewport_audit,
            "boundary_audit": boundary_audit,
            "min_required_fraction": _SLAB_MIN_FRAC,
            "warnings": list(result.warnings),
        }
        result.warnings.append(
            f"assembled slab covers only {frac:.0%} after 3 attempts; "
            "fail-closed and no slab geometry will be exported")
        try:
            (Path(out_dir) / f"slab_boundary_candidates_p{page_index + 1:02d}.json").write_text(
                json.dumps(boundary_audit, indent=2, ensure_ascii=False),
                encoding="utf-8")
            (Path(out_dir) / f"slab_missing_boundary_sides_p{page_index + 1:02d}.json").write_text(
                json.dumps(boundary_audit, indent=2, ensure_ascii=False),
                encoding="utf-8")
            plan_viewport.write_slab_boundary_failure_artifacts(
                page, rend, Path(out_dir), page_index + 1, gross,
                viewport_rect, boundary_audit)
        except Exception:
            pass
        _trace(
            "assembly",
            "tiny_slab_fail_closed",
            slab_fraction_of_content=result.slab_readiness.get(
                "slab_fraction_of_content"),
            slab_fraction_of_area_ref=round(frac, 6),
            area_reference=area_ref_label,
            min_required_fraction=_SLAB_MIN_FRAC,
            boundary_audit=boundary_audit,
            face_graph=trace_mod.face_graph_stats(fg_src, area_ref),
            gross_geometry=trace_mod.geometry_summary(gross),
        )
        try:
            (Path(out_dir) / "slab_readiness_report.json").write_text(
                json.dumps(result.slab_readiness, indent=2,
                           ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        _write_result_json(result, out_dir)
        _write_page_trace({"exit": "NO_EXPORT_TINY_SLAB"})
        return result
    elif frac > 0.90:
        result.warnings.append(
            f"assembled slab covers {frac:.0%} of the drawing area — may "
            f"include annotation loops, check step_07")

    parts = sorted(getattr(gross, "geoms", [gross]), key=lambda g: -g.area)
    slabs = [{"label": f"SLAB_{j + 1}" if len(parts) > 1 else "SLAB",
              "polygon_pdf": g, "void_count": 0}
             for j, g in enumerate(parts)]
    result.timings["assembly"] = time.time() - t2
    _trace(
        "assembly",
        duration_s=round(result.timings["assembly"], 3),
        slab_fraction_of_content=round(
            float(getattr(gross, "area", 0.0)) / max(content_area, 1.0), 4),
        slab_fraction_of_area_ref=round(frac, 4),
        area_reference=area_ref_label,
        slab_component_count=len(parts),
        gross_geometry=trace_mod.geometry_summary(gross),
    )

    if cfg.debug_images:
        final_keep_ids = [f.id for f in fg_src.faces
                          if f.area_pt2 >= cfg.min_keep_face_frac * area_ref]
        rend.step08_assembled_slab(slabs, final_keep_ids)

    # ── elements: X-cross openings ────────────────────────────────────────
    from src.slab_v2 import elements as elements_mod
    hatch_ids = {c.id for c in classes if c.role == "HATCH"}
    elem_paths = [p for p in paths if p.style_id not in hatch_ids]
    elems, elem_warnings = elements_mod.extract_elements(
        page, fg_all, cfg, content, area_ref, paths=elem_paths,
        scale=text_scale)

    # Per-page fallback: if page has VOID/PENETRATION text but strict
    # thresholds found nothing, retry with relaxed thresholds.
    # This ensures Cairns Hospital-style PDFs still detect large voids
    # while Combined Structural keeps its strict detection.
    _void_keywords = {"VOID", "PENETRATION", "PENETRATIONS", "OPENING"}
    page_words = [w[4].strip().upper().rstrip(".")
                  for w in page.get_text("words")]
    has_void_text = bool(_void_keywords & set(page_words))
    void_elems = [e for e in elems if e.type == "VOID"]
    if has_void_text and not void_elems and cfg.xcross_max_area_frac < 0.10:
        from dataclasses import replace as _cfg_replace
        relaxed_cfg = _cfg_replace(cfg, xcross_max_area_frac=0.10)
        elems_retry, retry_warnings = elements_mod.extract_elements(
            page, fg_all, relaxed_cfg, content, area_ref,
            paths=elem_paths, scale=text_scale)
        new_voids = [e for e in elems_retry if e.type == "VOID"
                     and e not in elems]
        if new_voids:
            elems = elems + new_voids
            elem_warnings.append(
                f"per-page fallback: relaxed xcross threshold found "
                f"{len(new_voids)} additional VOID element(s)")

    # keep only openings that intersect a slab
    from shapely.ops import unary_union
    slab_union = unary_union([s["polygon_pdf"] for s in slabs])
    result.elements = [e for e in elems
                       if e.polygon.intersects(slab_union)]
    result.warnings.extend(elem_warnings)
    _trace(
        "raw_elements",
        detected=len(elems),
        kept=len(result.elements),
        discarded=len(elems) - len(result.elements),
        has_void_text=has_void_text,
        warnings=list(elem_warnings),
    )
    _to_mm_log = 25.4 / 72.0 * float(text_scale or 100)
    xcross_audit = []
    for e in elems:
        bx = e.polygon.bounds
        w_mm = (bx[2] - bx[0]) * _to_mm_log
        h_mm = (bx[3] - bx[1]) * _to_mm_log
        intersects = e.polygon.intersects(slab_union)
        xcross_audit.append({
            "type": e.type, "label": e.label,
            "bounds_pt": [round(b, 1) for b in bx],
            "size_mm": f"{w_mm:.0f}x{h_mm:.0f}",
            "intersects_slab": intersects,
            "kept": intersects,
        })
    (Path(out_dir) / f"xcross_audit_p{page.number+1:02d}.json").write_text(
        json.dumps({"total_detected": len(elems),
                    "kept": len(result.elements),
                    "discarded": len(elems) - len(result.elements),
                    "slab_area_m2": round(sum(
                        s.get("area_m2", 0) if isinstance(s, dict)
                        else 0 for s in slabs), 1),
                    "elements": xcross_audit},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    if cfg.debug_images:
        rend.step09_elements(result.elements)

    # ── walls: moved after columns (needs column_polys) ──────────────────

    # ── dimension verification + scale self-check (always on) ─────────────
    from src.slab_v2 import verify
    t3 = time.time()
    dims = verify.parse_dimensions(page, classes, cfg, content)
    report = verify.verify_selection(slabs, dims, text_scale, content, cfg)
    result.verification = report
    rend.step_dimensions(dims, "step_09b_dimensions.png")
    result.timings["verify"] = time.time() - t3
    _trace(
        "dimension_verify",
        duration_s=round(result.timings["verify"], 3),
        text_scale=text_scale,
        scale_used=report.scale_used,
        scale_precise=round(report.scale_precise, 4) if report.scale_precise else None,
        scale_consistency=report.scale_consistency,
        n_dims_detected=len(dims),
        n_dims_associated=report.n_dims_associated,
        failure_count=len(report.failures or []),
    )

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
    _trace(
        "scale_final",
        text_scale=text_scale,
        final_scale=final_scale,
        manual_scale=cfg.manual_scale,
    )

    result.status = "OK"

    # ── columns (text-anchor-then-shape v2; fallback to shape-first v1) ───
    # Only exclude column candidates from elements that have specific text
    # evidence (a named label, not just the type name). Raw X-cross detections
    # without text anchors are unreliable and must not suppress columns.
    column_t0 = time.time()
    evidenced_elements = [
        e for e in result.elements
        if e.label and e.label != e.type
    ]
    if column_types is None:
        # deterministic source: the sheet's own CONCRETE COLUMN SCHEDULE
        # (Gemini census, when present, still takes precedence upstream)
        try:
            from src.slab_v2 import schedule_parser
            from src.slab_v2.models import ColumnType as _CT
            _sched = schedule_parser.parse_schedules(page)
            if _sched.columns:
                column_types = {}
                for mark, ct in _sched.columns.items():
                    w, d = ct.size_mm or (ct.diameter_mm or 0,
                                          ct.diameter_mm or 0)
                    column_types[mark] = _CT(
                        symbol=mark, width_mm=w, depth_mm=d,
                        material=ct.material)
                result.warnings.append(
                    "column types from on-page schedule: "
                    f"{sorted(column_types)}")
        except Exception as exc:               # noqa: BLE001
            result.warnings.append(f"on-page schedule parse failed: {exc}")
    if column_types is not None:
        from src.slab_v2 import columns_v2 as columns_mod
        cols, col_warnings, column_audit = columns_mod.extract_columns_v2(
            page, paths, slab_union, final_scale, column_types, cfg,
            elements=evidenced_elements,
            columns_per_floor_census=columns_per_floor,
            classes=classes, audit_out_dir=Path(out_dir))
        result.columns = cols
        result.column_candidates = column_audit.get("candidates", [])
        result.column_readiness = column_audit
        result.warnings.extend(col_warnings)
        if cfg.debug_images:
            rend.step11_columns(cols)

    expected_rc_columns = {}
    for sym, expected in (columns_per_floor or {}).items():
        clean_sym = str(sym).strip().rstrip("*")
        ct = (column_types or {}).get(sym) or (column_types or {}).get(clean_sym)
        material = str(getattr(ct, "material", "UNKNOWN") or "UNKNOWN").upper()
        if material == "STEEL":
            continue
        expected_rc_columns[clean_sym] = (
            expected_rc_columns.get(clean_sym, 0) + int(expected or 0))
    detected_column_counts = {}
    for col in result.columns:
        detected_column_counts[col.symbol] = (
            detected_column_counts.get(col.symbol, 0) + 1)
    missing_columns = {
        sym: max(expected - detected_column_counts.get(sym, 0), 0)
        for sym, expected in expected_rc_columns.items()
        if detected_column_counts.get(sym, 0) < expected
    }
    extra_columns = {
        sym: got for sym, got in detected_column_counts.items()
        if sym not in expected_rc_columns
    }
    column_status = ("not_required" if not expected_rc_columns else
                     "verified" if not missing_columns and not extra_columns
                     and not detected_column_counts.get("C?", 0) else "review")
    result.column_detection_report = {
        "status": column_status,
        "expected": expected_rc_columns,
        "detected": detected_column_counts,
        "missing": missing_columns,
        "extra": extra_columns,
        "ambiguous_count": detected_column_counts.get("C?", 0),
        "assignments": result.column_readiness.get("assignments", []),
    }
    result.column_readiness.update(result.column_detection_report)
    (Path(out_dir) / "column_readiness_report.json").write_text(
        json.dumps(result.column_detection_report, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    result.timings["columns"] = time.time() - column_t0
    _trace(
        "columns",
        duration_s=round(result.timings["columns"], 3),
        column_status=column_status,
        raw_count=len(result.columns),
        expected_count=sum(expected_rc_columns.values()),
        detected_counts=detected_column_counts,
        missing_count=sum(missing_columns.values()),
        extra_count=sum(extra_columns.values()),
        ambiguous_count=detected_column_counts.get("C?", 0),
        candidate_count=len(result.column_candidates or []),
    )

    # Steel is a separate subsystem. RC detection intentionally excludes steel
    # symbols; this stage audits and exports only verified steel geometry.
    steel_t0 = time.time()
    if column_types is not None or steel_census:
        try:
            from src.slab_v2 import steel_detector
            steel_result = steel_detector.detect_steel(
                page, paths, classes, slab_union, final_scale, column_types,
                cfg, Path(out_dir), renderer=rend, steel_census=steel_census)
            result.steel_members = steel_result.members
            result.steel_candidates = steel_result.candidates
            result.steel_assignment_report = steel_result.assignment
            result.steel_readiness = steel_result.readiness
            result.warnings.extend(steel_result.warnings)
        except Exception as exc:
            result.steel_readiness = {
                "status": "error",
                "warnings": [str(exc)],
                "export_policy": "verified_only",
            }
            result.warnings.append(f"steel detection failed: {exc}")
    else:
        result.steel_readiness = {
            "status": "steel_source_missing",
            "warnings": [],
            "export_policy": "verified_only",
            "zero_steel_reason": "Steel source planner was not run.",
        }
    result.timings["steel"] = time.time() - steel_t0
    _trace(
        "steel",
        duration_s=round(result.timings["steel"], 3),
        member_count=len(result.steel_members or []),
        candidate_count=len(result.steel_candidates or []),
        readiness=result.steel_readiness,
    )

    # ── walls: census-aware v2 or WALL-class face fallback ────────────────
    wall_t0 = time.time()
    col_polys = [c.polygon for c in result.columns] if result.columns else None
    if wall_types is None:
        # deterministic source: the sheet's own WALL SCHEDULE (same rule
        # as columns — Gemini census, when present, wins upstream)
        try:
            from src.slab_v2 import schedule_parser as _sp
            from src.slab_v2.models import WallType as _WT
            _wsched = _sp.parse_schedules(page)
            if _wsched.walls:
                wall_types = {
                    mark: _WT(symbol=mark,
                              thickness_mm=float(wt.thickness_mm or 0),
                              material=wt.description)
                    for mark, wt in _wsched.walls.items()}
                result.warnings.append(
                    "wall types from on-page schedule: "
                    f"{sorted(wall_types)}")
        except Exception as exc:               # noqa: BLE001
            result.warnings.append(
                f"on-page wall schedule parse failed: {exc}")
    if wall_types is not None:
        from src.slab_v2 import walls_v2
        result.walls, wall_warns = walls_v2.extract_walls_v2(
            page, paths, slab_union, final_scale, wall_types, cfg,
            fg_all=fg_all, election=result.election,
            elements=result.elements, column_polys=col_polys,
            walls_per_floor_census=walls_per_floor,
            classes=classes)
        result.warnings.extend(wall_warns)
    elif result.election:
        from src.slab_v2 import wall_extract
        result.walls = wall_extract.extract_walls(
            page, fg_all, result.election, cfg, area_ref, text_scale,
            column_polys=col_polys)

    if wall_source_registry and wall_types is not None:
        from src.slab_v2.wall_profile_resolver import resolve_plan_wall_topology
        result.walls, topology_report = resolve_plan_wall_topology(
            page, slab_union, result.walls, wall_types,
            walls_per_floor or {}, wall_source_registry, final_scale,
            Path(out_dir))
        result.wall_profiles = dict(wall_source_registry.get("profiles", {}))
        result.wall_readiness = topology_report
        result.warnings.extend(topology_report.get("warnings", []))
    if result.walls:
        from src.slab_v2.core_wall_topology import resolve_core_wall_topology
        result.walls, core_topology_report = resolve_core_wall_topology(
            page, paths, classes, result.walls, final_scale,
            [],
            cfg, Path(out_dir))
        if not result.wall_readiness:
            result.wall_readiness = {}
        result.wall_readiness["core_topology_status"] = (
            core_topology_report.get("status", "review"))
        result.wall_readiness["core_topology_report"] = core_topology_report
        result.warnings.extend(core_topology_report.get("warnings", []))

    if result.walls:
        from src.slab_v2.wall_junction_resolver import resolve_wall_junctions
        junction_expected = dict(walls_per_floor or {})
        if not junction_expected:
            for wall in result.walls:
                if wall.label.upper().startswith("LW"):
                    junction_expected[wall.label.upper()] = 1
        result.walls, junction_report = resolve_wall_junctions(
            page, result.walls, junction_expected, final_scale,
            result.elements, cfg, Path(out_dir))
        if not result.wall_readiness:
            result.wall_readiness = {}
        result.wall_readiness["junction_status"] = junction_report.get(
            "status", "review")
        result.wall_readiness["junction_report"] = junction_report
        result.warnings.extend(junction_report.get("warnings", []))
    result.timings["walls"] = time.time() - wall_t0

    opening_t0 = time.time()
    from src.slab_v2 import opening_resolver
    opening_resolution = opening_resolver.resolve_openings(
        page, paths, classes, result.elements, result.walls, slabs,
        final_scale, content, cfg=cfg, renderer=rend, use_ai=use_ai,
        columns=result.columns)
    result.verified_cut_openings = opening_resolution.verified_cut_openings
    result.resolved_openings = list(result.verified_cut_openings)
    result.opening_context_objects = opening_resolution.context_objects
    result.opening_review_candidates = opening_resolution.review_candidates
    result.opening_policy_version = getattr(
        cfg, "opening_policy_version", "penetration_only_v2")
    result.resolved_penetrations = opening_resolution.resolved_penetrations
    # Openings are subtraction metadata, never 3D solids. Dedicated element
    # renderers may populate this later, but must not reuse cut geometry.
    result.render_elements = []
    result.opening_candidates = opening_resolution.candidates
    result.opening_judgement = opening_resolution.judgement
    result.opening_report = opening_resolution.report
    if opening_resolution.report.get("judge_status") in {"accepted", "rejected"}:
        result.gemini_calls += 1
    if opening_resolution.report.get("stairs", 0) > 0:
        result.warnings = [
            w for w in result.warnings
            if not ("has no X-cross opening" in w and "STAIR" in w)
        ]
    result.warnings.extend(opening_resolution.warnings)
    result.timings["openings"] = time.time() - opening_t0
    _trace(
        "openings",
        duration_s=round(result.timings["openings"], 3),
        candidate_count=len(opening_resolution.candidates or []),
        verified_cut_openings=len(result.verified_cut_openings or []),
        context_objects=len(result.opening_context_objects or []),
        review_candidates=len(result.opening_review_candidates or []),
        resolved_penetrations=len(result.resolved_penetrations or []),
        judgement=opening_resolution.judgement,
        report=opening_resolution.report,
    )

    detected_wall_counts = {}
    for w in result.walls:
        detected_wall_counts[w.label] = detected_wall_counts.get(w.label, 0) + 1
    base_wall_report = {
        "expected": dict(walls_per_floor or {}),
        "detected": detected_wall_counts,
        "missing": {
            sym: expected for sym, expected in (walls_per_floor or {}).items()
            if detected_wall_counts.get(sym, 0) < expected
        },
        "extra": {
            sym: got for sym, got in detected_wall_counts.items()
            if walls_per_floor and sym not in walls_per_floor
            and not sym.startswith("WALL_")
        },
    }
    if result.wall_readiness:
        result.wall_detection_report = {
            **base_wall_report,
            "topology_status": result.wall_readiness.get("status", "review"),
            "instances": result.wall_readiness.get("instances", []),
        }
        result.wall_readiness["detected"] = detected_wall_counts
        result.wall_readiness["missing"] = base_wall_report["missing"]
        if base_wall_report["missing"]:
            result.wall_readiness["status"] = "review"
    else:
        result.wall_detection_report = base_wall_report
        result.wall_readiness = {
            "status": "review" if (walls_per_floor or {}) else "not_required",
            **base_wall_report,
            "warnings": ["Wall source registry unavailable; shape detector used."],
        }
    _trace(
        "walls",
        duration_s=round(result.timings.get("walls", 0.0), 3),
        wall_count=len(result.walls or []),
        detected_counts=detected_wall_counts,
        readiness=result.wall_readiness,
        detection_report=result.wall_detection_report,
    )

    if cfg.debug_images:
        if not (Path(out_dir) / "step_09c_opening_candidates.png").exists():
            rend.step09_candidates(opening_resolution.candidates)
        rend.step09_elements(
            opening_resolution.stair_footprints
            + opening_resolution.core_shaft_footprints,
            "step_09a_stair_core_candidates.png")
        rend.step09_elements(result.resolved_openings,
                             "step_09b_resolved_openings.png")
        rend.step09_opening_guards(
            opening_resolution.candidates, result.walls,
            set(opening_resolution.judgement.get("opening_ids", [])),
            "opening_geometry_guards_p%02d.png" % (page.number + 1))
        penetration_candidates = [
            candidate for candidate in opening_resolution.candidates
            if candidate.get("kind_hint") in {
                "STAIRWELL", "STAIR_PENETRATION", "STAIR_OPENING",
                "STAIR_LANDING"}]
        rend.step09_candidates(
            penetration_candidates, "penetration_candidates_p%02d.png" %
            (page.number + 1))
        rend.step09_candidates(
            [candidate for candidate in penetration_candidates
             if candidate.get("kind_hint") == "STAIRWELL"],
            "penetration_boundary_graph_p%02d.png" % (page.number + 1))
        rend.step09_elements(
            [ElementFootprint(
                type="STAIR", polygon=penetration.polygon,
                label=penetration.id, anchor_bbox=penetration.polygon.bounds,
                area_pt2=penetration.polygon.area)
             for penetration in opening_resolution.resolved_penetrations],
            "resolved_penetrations_p%02d.png" % (page.number + 1))
        rend.step09_candidates(
            [candidate for candidate in penetration_candidates
             if "rejected_as_final_hull" in candidate.get("source", "")],
            "rejected_convex_hull_candidates_p%02d.png" %
            (page.number + 1))
        rend.step10c_walls(result.walls)
    try:
        public_penetrations = [{
            "id": item.id, "kind": item.kind,
            "source_candidate_ids": item.source_candidate_ids,
            "contained_seed_ids": item.contained_seed_ids,
            "boundary_coverage": item.boundary_coverage,
            "confidence": item.confidence, "status": item.status,
            "warnings": item.warnings,
            "geometry_audit": item.geometry_audit,
            "bbox": list(item.polygon.bounds),
        } for item in opening_resolution.resolved_penetrations]
        penetration_candidates = [{
            key: value for key, value in candidate.items()
            if key != "polygon"
        } for candidate in opening_resolution.candidates
            if candidate.get("kind_hint") in {
                "STAIRWELL", "STAIR_PENETRATION", "STAIR_OPENING",
                "STAIR_LANDING"}]
        (Path(out_dir) / f"penetration_candidates_p{page.number+1:02d}.json").write_text(
            json.dumps(penetration_candidates, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (Path(out_dir) / f"penetration_boundary_graph_p{page.number+1:02d}.json").write_text(
            json.dumps(public_penetrations, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (Path(out_dir) / f"resolved_penetrations_p{page.number+1:02d}.json").write_text(
            json.dumps(public_penetrations, indent=2, ensure_ascii=False),
            encoding="utf-8")
        guard_candidates = [{
            key: value for key, value in candidate.items()
            if key != "polygon"
        } for candidate in opening_resolution.candidates]
        (Path(out_dir) / f"opening_geometry_guards_p{page.number+1:02d}.json").write_text(
            json.dumps({
                "selected_ids": opening_resolution.judgement.get(
                    "opening_ids", []),
                "report": opening_resolution.report,
                "candidates": guard_candidates,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        result.warnings.append(f"penetration audit output failed: {exc}")

    # Partition gross FLOOR STRUCTURE into PT/concrete and other floor
    # systems. Fill is extent evidence only; material is resolved separately.
    from src.slab_v2 import floor_system_resolver
    floor_resolution, floor_candidates, floor_profile = (
        floor_system_resolver.resolve_floor_systems(
            page, paths, classes, slabs, result.verified_cut_openings, cfg, rend,
            use_ai=use_ai, scale=final_scale,
            context_objects=result.opening_context_objects))
    result.floor_system_candidates = floor_candidates
    result.floor_system_profile = floor_profile
    result.floor_system_resolution = floor_resolution
    result.floor_system_readiness = {
        "status": floor_resolution.status,
        "confidence": floor_resolution.confidence,
        "pt_slab_ids": floor_resolution.pt_slab_ids,
        "other_floor_ids": floor_resolution.other_floor_ids,
        "unknown_ids": floor_resolution.unknown_ids,
        "reason": floor_resolution.reason,
        "warnings": floor_resolution.warnings,
    }
    result.slab_readiness = {
        "status": floor_resolution.status,
        "confidence": floor_resolution.confidence,
        "review_ids": floor_resolution.unknown_ids,
        "reason": floor_resolution.reason,
        "warnings": floor_resolution.warnings,
    }
    quality_warnings = []
    try:
        if frac is not None and float(frac) < 0.03:
            quality_warnings.append(
                f"QA gate: slab covers only {float(frac):.1%} of content; "
                "floor/slab readiness forced to review.")
    except Exception:
        pass
    if any("using best result anyway" in str(w).lower()
           or "covers only 1%" in str(w).lower()
           or "covers only 2%" in str(w).lower()
           for w in result.warnings):
        quality_warnings.append(
            "QA gate: tiny slab fallback warning detected; "
            "floor/slab readiness forced to review.")
    if quality_warnings:
        floor_resolution.status = "review"
        merged = list(floor_resolution.warnings or [])
        for warning in quality_warnings:
            if warning not in merged:
                merged.append(warning)
            if warning not in result.warnings:
                result.warnings.append(warning)
        floor_resolution.warnings = merged
        result.floor_system_readiness["status"] = "review"
        result.floor_system_readiness["warnings"] = merged
        result.slab_readiness["status"] = "review"
        result.slab_readiness["warnings"] = merged
        _trace("quality_gate", "review", warnings=quality_warnings,
               slab_fraction_of_content=frac)
    result.warnings.extend(floor_resolution.warnings)
    result.gemini_calls += sum((Path(out_dir) / name).exists() for name in (
        "step_08b_floor_system_profile_raw.txt",
        "step_08c_floor_system_judge_raw.txt"))
    _trace(
        "floor_system",
        candidate_count=len(floor_candidates or []),
        floor_system_status=floor_resolution.status,
        confidence=floor_resolution.confidence,
        pt_slab_ids=floor_resolution.pt_slab_ids,
        other_floor_ids=floor_resolution.other_floor_ids,
        opening_ids=floor_resolution.opening_ids,
        unknown_ids=floor_resolution.unknown_ids,
        reason=floor_resolution.reason,
        warnings=floor_resolution.warnings,
        pt_gross_geometry=trace_mod.geometry_summary(
            floor_resolution.pt_gross_geometry),
        pt_net_geometry=trace_mod.geometry_summary(
            floor_resolution.pt_net_geometry),
        other_floor_geometry=trace_mod.geometry_summary(
            floor_resolution.other_floor_geometry),
    )

    resolved_parts = [g for g in getattr(
        floor_resolution.pt_gross_geometry, "geoms",
        [floor_resolution.pt_gross_geometry])
        if hasattr(g, "area") and g.area > 0]
    if resolved_parts:
        slabs = [{"label": (f"SLAB_{i + 1}"
                            if len(resolved_parts) > 1 else "SLAB"),
                  "polygon_pdf": g, "void_count": 0}
                 for i, g in enumerate(resolved_parts)]

    other_parts = [g for g in getattr(
        floor_resolution.other_floor_geometry, "geoms",
        [floor_resolution.other_floor_geometry])
        if hasattr(g, "area") and g.area > 0]
    result.other_floor_systems = [
        {"label": f"OTHER_FLOOR_SYSTEM_{i + 1}", "polygon_pdf": g}
        for i, g in enumerate(other_parts)]

    if cfg.debug_images:
        rend.step08_separator_endpoints(floor_candidates)
        rend.step08_floor_system_decision(floor_candidates, floor_resolution)
        rend.step08_overcut_guard(floor_candidates, floor_resolution)
        rend.step10_floor_system_geometry(
            floor_resolution.pt_gross_geometry, "PT concrete gross slab",
            "step_10_pt_gross_slab.png")
        rend.step10_floor_system_geometry(
            floor_resolution.pt_net_geometry, "PT concrete net slab",
            "step_10_pt_net_slab.png")
    try:
        (Path(out_dir) / "step_08a_floor_system_candidates.json").write_text(
            json.dumps(floor_system_resolver.candidate_payload(floor_candidates),
                       indent=2, ensure_ascii=False), encoding="utf-8")
        endpoint_payload = floor_system_resolver.candidate_payload(
            floor_candidates)
        (Path(out_dir) / "step_08a_separator_endpoints.json").write_text(
            json.dumps(endpoint_payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (Path(out_dir) / "step_08b_bounded_cut_candidates.json").write_text(
            json.dumps(endpoint_payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (Path(out_dir) / "floor_system_readiness.json").write_text(
            json.dumps(result.floor_system_readiness, indent=2,
                       ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        result.warnings.append(f"slab audit output failed: {exc}")

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
    for other in result.other_floor_systems:
        if final_scale:
            mm = transform_polygon(other["polygon_pdf"], page, final_scale,
                                   page.rect.x0, page.rect.y1)
            other["polygon_mm"] = mm
            other["area_m2"] = mm.area / 1_000_000.0
        else:
            other["polygon_mm"] = None
            other["area_m2"] = None
    result.slabs = slabs

    if cfg.debug_images:
        rend.step10_final(slabs, result.verified_cut_openings)
    result.timings["total"] = time.time() - t0
    _trace(
        "final_page",
        total_s=round(result.timings["total"], 3),
        counts=trace_mod.result_counts(result),
        statuses=trace_mod.report_statuses(result),
    )
    _write_result_json(result, out_dir)
    _write_page_trace({"exit": "OK"})
    return result


def _write_result_json(result: SlabV2Result, out_dir: Path) -> None:
    from src.slab_v2 import slab_face_resolver, floor_system_resolver

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
        "scale_audit": result.scale_audit,
        "page_role_classification": result.page_role_classification,
        "plan_viewport": result.plan_viewport,
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
        "resolved_openings": [
            {"type": e.type, "label": e.label,
             "opening_intent": e.opening_intent,
             "object_roles": e.object_roles,
             "evidence_ids": e.evidence_ids,
             "candidate_id": e.candidate_id,
             "area_pt2": round(e.area_pt2, 1),
             "polygon_pdf_pts": poly_coords(e.polygon)}
            for e in result.resolved_openings],
        "opening_policy_version": result.opening_policy_version,
        "verified_cut_openings": [
            {"type": e.type, "label": e.label,
             "opening_intent": e.opening_intent,
             "object_roles": e.object_roles,
             "evidence_ids": e.evidence_ids,
             "candidate_id": e.candidate_id,
             "area_pt2": round(e.area_pt2, 1),
             "polygon_pdf_pts": poly_coords(e.polygon)}
            for e in result.verified_cut_openings],
        "opening_context_objects": [
            {"type": e.type, "label": e.label,
             "opening_intent": e.opening_intent,
             "object_roles": e.object_roles,
             "candidate_id": e.candidate_id,
             "area_pt2": round(e.area_pt2, 1),
             "polygon_pdf_pts": poly_coords(e.polygon)}
            for e in result.opening_context_objects],
        "resolved_penetrations": [
            {"id": p.id, "kind": p.kind,
             "source_candidate_ids": p.source_candidate_ids,
             "contained_seed_ids": p.contained_seed_ids,
             "boundary_coverage": p.boundary_coverage,
             "confidence": p.confidence, "status": p.status,
             "warnings": p.warnings,
             "geometry_audit": p.geometry_audit,
             "polygon_pdf_pts": poly_coords(p.polygon)}
            for p in result.resolved_penetrations],
        "render_elements": [
            {"type": e.type, "label": e.label,
             "area_pt2": round(e.area_pt2, 1),
             "polygon_pdf_pts": poly_coords(e.polygon)}
            for e in result.render_elements],
        "opening_report": result.opening_report,
        "opening_candidates": [
            {**{k: value for k, value in c.items() if k != "polygon"},
             "polygon_pdf_pts": poly_coords(c.get("polygon"))}
            for c in result.opening_candidates
        ],
        "opening_judgement": result.opening_judgement,
        "slab_candidates": [
            {**slab_face_resolver._public(c),
             "polygon_pdf_pts": poly_coords(c.polygon)}
            for c in result.slab_candidates
        ],
        "slab_readiness": result.slab_readiness,
        "slab_resolution": (
            {"selected_slab_ids": result.slab_resolution.selected_slab_ids,
             "appendage_ids": result.slab_resolution.appendage_ids,
             "opening_ids": result.slab_resolution.opening_ids,
             "non_slab_ids": result.slab_resolution.non_slab_ids,
             "review_ids": result.slab_resolution.review_ids,
             "confidence": result.slab_resolution.confidence,
             "status": result.slab_resolution.status,
             "reason": result.slab_resolution.reason,
             "warnings": result.slab_resolution.warnings}
            if result.slab_resolution else None),
        "floor_system_profile": (
            asdict(result.floor_system_profile)
            if result.floor_system_profile else None),
        "floor_system_candidates": [
            {**floor_system_resolver._candidate_public(c),
             "polygon_pdf_pts": poly_coords(c.polygon)}
            for c in result.floor_system_candidates
        ],
        "floor_system_readiness": result.floor_system_readiness,
        "floor_system_resolution": (
            {"pt_slab_ids": result.floor_system_resolution.pt_slab_ids,
             "other_floor_ids": result.floor_system_resolution.other_floor_ids,
             "opening_ids": result.floor_system_resolution.opening_ids,
             "non_floor_ids": result.floor_system_resolution.non_floor_ids,
             "unknown_ids": result.floor_system_resolution.unknown_ids,
             "confidence": result.floor_system_resolution.confidence,
             "status": result.floor_system_resolution.status,
             "reason": result.floor_system_resolution.reason,
             "warnings": result.floor_system_resolution.warnings}
            if result.floor_system_resolution else None),
        "walls": [
            {"label": w.label, "w_mm": w.w_mm, "l_mm": w.l_mm,
             "wall_type": w.wall_type, "centerline": w.centerline,
             "source": w.source, "confidence": w.confidence,
             "profile_id": w.profile_id,
             "mapping_status": w.mapping_status,
             "polygon_pdf_pts": poly_coords(w.polygon)}
            for w in result.walls],
        "wall_detection_report": result.wall_detection_report,
        "wall_readiness": result.wall_readiness,
        "wall_profiles": result.wall_profiles,
        "columns": [
            {"symbol": c.symbol, "w_mm": c.w_mm, "d_mm": c.d_mm,
             "labeled": c.labeled, "candidate_id": c.candidate_id,
             "source": c.source, "confidence": c.confidence,
             "grid_id": c.grid_id,
             "polygon_pdf_pts": poly_coords(c.polygon)}
            for c in result.columns],
        "column_candidates": result.column_candidates,
        "column_readiness": result.column_readiness,
        "column_detection_report": result.column_detection_report,
        "steel_members": [
            {"id": m.id, "symbol": m.symbol,
             "member_type": m.member_type, "section": m.section,
             "source": m.source, "confidence": m.confidence,
             "status": m.status, "nearby_text": m.nearby_text,
             "evidence": m.evidence, "reject_reason": m.reject_reason,
             "polygon_pdf_pts": poly_coords(m.polygon)}
            for m in result.steel_members],
        "steel_candidates": result.steel_candidates,
        "steel_assignment_report": result.steel_assignment_report,
        "steel_readiness": result.steel_readiness,
        "slabs": [
            {"label": s["label"],
             "area_m2": s.get("area_m2"),
             "polygon_pdf_pts": poly_coords(s["polygon_pdf"])}
            for s in result.slabs],
        "other_floor_systems": [
            {"label": s["label"], "area_m2": s.get("area_m2"),
             "polygon_pdf_pts": poly_coords(s["polygon_pdf"])}
            for s in result.other_floor_systems],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
