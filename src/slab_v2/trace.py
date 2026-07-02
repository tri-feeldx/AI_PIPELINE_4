"""Best-effort flight recorder for slab_v2 runs.

The trace layer is intentionally observational: failures here must never
change geometry decisions or stop an extraction/export.  It indexes the many
existing debug artifacts and records compact stage summaries so a bad SketchUp
model can be traced back to raster/vector/AI/geometry/export decisions.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from statistics import median
from pathlib import Path
from typing import Any


_TRACE_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "bounds"):
        try:
            return {
                "bounds": [round(float(x), 3) for x in value.bounds],
                "area": round(float(getattr(value, "area", 0.0)), 3),
                "geom_type": getattr(value, "geom_type", type(value).__name__),
            }
        except Exception:
            return repr(value)
    return repr(value)


def safe_write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        # Trace must not affect extraction.
        return


def file_sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def artifact_summary(path: Path) -> dict:
    item = {"path": str(path), "name": path.name}
    try:
        stat = path.stat()
        item["bytes"] = stat.st_size
        item["mtime"] = stat.st_mtime
        item["sha256_12"] = (file_sha256(path) or "")[:12]
    except Exception:
        pass
    return item


def collect_artifacts(page_dir: Path) -> list[dict]:
    try:
        files = [
            p for p in page_dir.iterdir()
            if p.is_file()
            and (p.suffix.lower() in {".json", ".png", ".txt", ".rb"})
            and p.name not in {"page_trace.json"}
        ]
    except Exception:
        return []
    return [artifact_summary(p) for p in sorted(files, key=lambda x: x.name)]


def init_run_trace(run_root: Path, pdf_path: str, cfg: Any, page_count: int | None) -> None:
    pdf = Path(pdf_path)
    payload = {
        "schema": "slab_v2_run_trace_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pdf": {
            "path": str(pdf),
            "name": pdf.name,
            "stem": pdf.stem,
            "sha256": file_sha256(pdf),
            "page_count": page_count,
        },
        "config": _config_snapshot(cfg),
        "trace_files": {
            "run_trace": str(run_root / "run_trace.json"),
            "stage_trace_jsonl": str(run_root / "stage_trace.jsonl"),
        },
    }
    safe_write_json(run_root / "run_trace.json", payload)


def append_event(run_root: Path, page_dir: Path | None, stage: str,
                 status: str, payload: dict | None = None) -> None:
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": stage,
        "status": status,
        "page_dir": str(page_dir) if page_dir else None,
        "payload": _jsonable(payload or {}),
    }
    line = json.dumps(event, ensure_ascii=False)
    with _TRACE_LOCK:
        try:
            run_root.mkdir(parents=True, exist_ok=True)
            with (run_root / "stage_trace.jsonl").open(
                    "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            return


def write_page_trace(result: Any, page: Any, out_dir: Path,
                     stage_snapshots: list[dict], extra: dict | None = None) -> None:
    payload = {
        "schema": "slab_v2_page_trace_v1",
        "page_index": getattr(result, "page_index", None),
        "page_number": getattr(result, "page_index", 0) + 1,
        "status": getattr(result, "status", None),
        "page": _page_snapshot(page),
        "scale": getattr(result, "scale", None),
        "timings_s": {
            str(k): round(float(v), 3)
            for k, v in (getattr(result, "timings", {}) or {}).items()
        },
        "warnings": list(getattr(result, "warnings", []) or []),
        "counts": result_counts(result),
        "plan_viewport": getattr(result, "plan_viewport", {}) or {},
        "stage_snapshots": stage_snapshots,
        "artifacts": collect_artifacts(out_dir),
    }
    if extra:
        payload["extra"] = _jsonable(extra)
    safe_write_json(out_dir / "page_trace.json", payload)


def _page_snapshot(page: Any) -> dict:
    try:
        rect = page.rect
        return {
            "number": page.number + 1,
            "width_pt": round(float(rect.width), 3),
            "height_pt": round(float(rect.height), 3),
            "rotation": int(getattr(page, "rotation", 0) or 0),
        }
    except Exception:
        return {}


def _config_snapshot(cfg: Any) -> dict:
    try:
        raw = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    except Exception:
        raw = {}
    interesting = {
        k: raw.get(k) for k in sorted(raw)
        if k in {
            "debug_dir", "debug_dpi", "debug_images", "trace_level",
            "speed_mode", "max_parallel_pages", "manual_scale",
            "fast_disable_page_ai", "extraction_max_workers",
            "opening_policy_version", "debug_export_verified_only",
            "export_foundations", "enable_opening_judge",
            "enable_floor_system_judge", "enable_slab_face_judge",
        }
    }
    return interesting


def content_rect_snapshot(rect: Any) -> dict:
    try:
        return {
            "x0": round(float(rect.x0), 3),
            "y0": round(float(rect.y0), 3),
            "x1": round(float(rect.x1), 3),
            "y1": round(float(rect.y1), 3),
            "width": round(float(rect.width), 3),
            "height": round(float(rect.height), 3),
            "area": round(float(rect.width * rect.height), 3),
        }
    except Exception:
        return {}


def vector_stats(paths: list, classes: list) -> dict:
    by_style = Counter()
    outside = Counter()
    segs = 0
    for p in paths or []:
        sid = getattr(p, "style_id", None)
        by_style[str(sid)] += 1
        if getattr(p, "outside_content", False):
            outside[str(sid)] += 1
        segs += len(getattr(p, "segments", []) or [])
    roles = Counter(str(getattr(c, "role", "")) for c in classes or [])
    return {
        "path_count": len(paths or []),
        "segment_count": segs,
        "style_class_count": len(classes or []),
        "paths_by_style": dict(by_style),
        "outside_paths_by_style": dict(outside),
        "classes_by_role": dict(roles),
        "classes": _class_rows(classes),
    }


def _class_rows(classes: list) -> list[dict]:
    rows = []
    for c in classes or []:
        key = getattr(c, "key", None)
        rows.append({
            "id": getattr(c, "id", None),
            "role": getattr(c, "role", None),
            "n_segments": getattr(c, "n_segments", None),
            "total_length_pt": round(float(getattr(c, "total_length_pt", 0.0)), 2),
            "width": getattr(key, "width", None),
            "dashes": bool(getattr(key, "dashes", False)),
            "prefiltered": bool(getattr(c, "prefiltered", False)),
        })
    return rows


def face_graph_stats(fg: Any, content_area: float | None = None) -> dict:
    faces = list(getattr(fg, "faces", []) or [])
    areas = sorted([float(getattr(f, "area_pt2", 0.0)) for f in faces],
                   reverse=True)
    payload = {
        "face_count": len(faces),
        "largest_areas_pt2": [round(a, 2) for a in areas[:10]],
        "total_area_pt2": round(sum(areas), 2),
        "depth_counts": dict(Counter(str(getattr(f, "depth", "")) for f in faces)),
    }
    if content_area:
        payload["largest_area_frac_of_content"] = (
            round(areas[0] / content_area, 4) if areas else 0.0)
    return payload


def geometry_summary(geom: Any) -> dict:
    if geom is None:
        return {"present": False}
    geoms = list(getattr(geom, "geoms", [geom]) or [])
    return {
        "present": True,
        "geom_type": getattr(geom, "geom_type", type(geom).__name__),
        "component_count": len(geoms),
        "area_pt2": round(float(getattr(geom, "area", 0.0)), 3),
        "bounds": [round(float(v), 3) for v in getattr(geom, "bounds", [])],
    }


def result_counts(result: Any) -> dict:
    return {
        "slabs": len(getattr(result, "slabs", []) or []),
        "raw_elements": len(getattr(result, "elements", []) or []),
        "verified_cut_openings": len(
            getattr(result, "verified_cut_openings", []) or []),
        "opening_context_objects": len(
            getattr(result, "opening_context_objects", []) or []),
        "opening_review_candidates": len(
            getattr(result, "opening_review_candidates", []) or []),
        "columns": len(getattr(result, "columns", []) or []),
        "walls": len(getattr(result, "walls", []) or []),
        "steel_members": len(getattr(result, "steel_members", []) or []),
        "other_floor_systems": len(
            getattr(result, "other_floor_systems", []) or []),
    }


def report_statuses(result: Any) -> dict:
    def _status(name: str) -> Any:
        row = getattr(result, name, {}) or {}
        return row.get("status") or row.get("model_status")
    return {
        "column": _status("column_readiness"),
        "steel": _status("steel_readiness"),
        "wall": _status("wall_readiness"),
        "floor_system": _status("floor_system_readiness"),
        "slab": _status("slab_readiness"),
        "opening_policy": getattr(result, "opening_policy_version", None),
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stage_payload(page_trace: dict, stage: str) -> dict:
    for row in page_trace.get("stage_snapshots", []) or []:
        if row.get("stage") == stage:
            return row
    return {}


def _result_area_m2(result: dict | None) -> float:
    if not result:
        return 0.0
    return float(sum((s.get("area_m2") or 0.0)
                     for s in result.get("slabs", []) or []))


def _page_audit_row(page_dir: Path) -> dict:
    page_trace = _read_json(page_dir / "page_trace.json") or {}
    result = _read_json(page_dir / "result.json") or {}
    page_num = (
        page_trace.get("page_number")
        or result.get("page_number")
        or _page_number_from_dir(page_dir)
    )
    assembly = _stage_payload(page_trace, "assembly")
    floor_system = result.get("floor_system_readiness") or {}
    slab_readiness = result.get("slab_readiness") or {}
    column_report = result.get("column_detection_report") or {}
    steel_readiness = result.get("steel_readiness") or {}
    wall_readiness = result.get("wall_readiness") or {}
    opening_report = result.get("opening_report") or {}
    counts = page_trace.get("counts") or {}
    if result:
        counts = {
            **counts,
            "slabs": len(result.get("slabs", []) or []),
            "columns": len(result.get("columns", []) or []),
            "walls": len(result.get("walls", []) or []),
            "steel_members": len(result.get("steel_members", []) or []),
            "verified_cut_openings": len(
                result.get("verified_cut_openings", []) or []),
        }
    return {
        "page": page_num,
        "page_dir": str(page_dir),
        "status": page_trace.get("status") or result.get("status"),
        "exit": (page_trace.get("extra") or {}).get("exit"),
        "scale": page_trace.get("scale") or result.get("scale"),
        "area_m2": round(_result_area_m2(result), 3),
        "slab_fraction_of_content": assembly.get(
            "slab_fraction_of_content"),
        "slab_fraction_of_area_ref": assembly.get(
            "slab_fraction_of_area_ref"),
        "area_reference": assembly.get("area_reference"),
        "plan_viewport": page_trace.get("plan_viewport") or _stage_payload(
            page_trace, "plan_viewport").get("viewport"),
        "counts": counts,
        "readiness": {
            "slab": slab_readiness.get("status"),
            "floor_system": floor_system.get("status"),
            "columns": column_report.get("status"),
            "walls": wall_readiness.get("status"),
            "steel": steel_readiness.get("status"),
        },
        "column_expected": column_report.get("expected", {}),
        "column_detected": column_report.get("detected", {}),
        "column_missing": column_report.get("missing", {}),
        "steel": {
            "expected": steel_readiness.get("expected_count"),
            "verified": steel_readiness.get("verified_count"),
            "review": steel_readiness.get("review_count"),
            "zero_reason": steel_readiness.get("zero_or_low_steel_reason")
                           or steel_readiness.get("zero_steel_reason"),
        },
        "openings": {
            "verified_cuts": opening_report.get("verified_cuts")
                             or counts.get("verified_cut_openings"),
            "review": opening_report.get("unresolved_candidate_ids", []),
        },
        "warnings": list(page_trace.get("warnings") or result.get("warnings")
                         or []),
        "artifacts": page_trace.get("artifacts", []),
        "trace_missing": not bool(page_trace),
        "result_missing": not bool(result),
    }


def _page_number_from_dir(page_dir: Path) -> int | None:
    try:
        if page_dir.name.startswith("page_"):
            return int(page_dir.name.split("_", 1)[1])
    except Exception:
        pass
    return None


def collect_audit_ledger(run_root: Path) -> dict:
    run_trace = _read_json(run_root / "run_trace.json") or {}
    page_dirs = sorted(
        [p for p in run_root.glob("page_*") if p.is_dir()],
        key=lambda p: _page_number_from_dir(p) or 0,
    )
    pages = [_page_audit_row(p) for p in page_dirs]
    return {
        "schema": "slab_v2_audit_ledger_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_root": str(run_root),
        "pdf": run_trace.get("pdf", {}),
        "config": run_trace.get("config", {}),
        "pages": pages,
        "summary": _audit_summary(pages),
    }


def _audit_summary(pages: list[dict]) -> dict:
    ok = [p for p in pages if p.get("status") == "OK"]
    areas = [float(p.get("area_m2") or 0.0) for p in ok
             if float(p.get("area_m2") or 0.0) > 0.0]
    scales = Counter(str(p.get("scale")) for p in ok if p.get("scale"))
    return {
        "page_count_with_trace": len(pages),
        "ok_page_count": len(ok),
        "skip_page_count": len([p for p in pages
                                if p.get("status") != "OK"]),
        "median_slab_area_m2": round(median(areas), 3) if areas else None,
        "scale_histogram": dict(scales),
        "total_columns": sum(int((p.get("counts") or {}).get("columns") or 0)
                             for p in pages),
        "total_walls": sum(int((p.get("counts") or {}).get("walls") or 0)
                           for p in pages),
        "total_steel_members": sum(int((p.get("counts") or {}).get(
            "steel_members") or 0) for p in pages),
        "total_verified_cut_openings": sum(int((p.get("counts") or {}).get(
            "verified_cut_openings") or 0) for p in pages),
    }


def build_quality_gate_report(ledger: dict,
                              model_readiness: dict | None = None) -> dict:
    pages = ledger.get("pages", []) or []
    areas = [float(p.get("area_m2") or 0.0) for p in pages
             if p.get("status") == "OK"
             and float(p.get("area_m2") or 0.0) > 0.0]
    med_area = median(areas) if areas else 0.0
    scale_counts = Counter(str(p.get("scale")) for p in pages
                           if p.get("status") == "OK" and p.get("scale"))
    dominant_scale = scale_counts.most_common(1)[0][0] if scale_counts else None
    issues: list[dict] = []

    for p in pages:
        page = p.get("page")
        status = p.get("status")
        warnings = " ".join(str(w).lower() for w in p.get("warnings", []))
        area = float(p.get("area_m2") or 0.0)
        frac = p.get("slab_fraction_of_content")
        page_issues = []
        if p.get("trace_missing") or p.get("result_missing"):
            page_issues.append(("critical", "missing_trace_or_result",
                                "Missing page_trace.json or result.json."))
        if status != "OK":
            page_issues.append(("warning", "page_not_ok",
                                f"Page status is {status}."))
        if area > 0 and med_area and area < 0.10 * med_area:
            page_issues.append((
                "critical", "slab_area_outlier",
                f"Slab area {area:.3f}m2 is below 10% of median "
                f"{med_area:.3f}m2."))
        if frac is not None and float(frac) < 0.03:
            page_issues.append((
                "critical", "tiny_slab_fraction",
                f"Slab covers only {float(frac):.1%} of content."))
        if ("using best result anyway" in warnings
                or "covers only 1%" in warnings
                or "covers only 2%" in warnings):
            page_issues.append((
                "critical", "fail_open_slab_warning",
                "Page warning says tiny slab was accepted anyway."))
        if (p.get("scale") and dominant_scale
                and str(p.get("scale")) != dominant_scale
                and area > 0 and med_area and area < 0.25 * med_area):
            page_issues.append((
                "critical", "scale_area_mismatch",
                f"Scale 1:{p.get('scale')} differs from dominant "
                f"1:{dominant_scale} on an area outlier."))
        readiness = p.get("readiness") or {}
        if any(item[1] in {
            "slab_area_outlier", "tiny_slab_fraction",
            "fail_open_slab_warning", "scale_area_mismatch",
        } for item in page_issues):
            if readiness.get("slab") == "verified" or readiness.get(
                    "floor_system") == "verified":
                page_issues.append((
                    "critical", "verified_status_contradicts_trace",
                    "Slab/floor readiness is verified despite critical "
                    "geometry evidence."))
        missing_cols = p.get("column_missing") or {}
        if missing_cols:
            page_issues.append((
                "warning", "missing_rc_columns",
                f"Missing expected RC columns: {missing_cols}"))
        steel = p.get("steel") or {}
        if steel.get("review"):
            page_issues.append((
                "warning", "steel_review_items",
                f"Steel has {steel.get('review')} review items."))

        for severity, code, message in page_issues:
            issues.append({
                "severity": severity,
                "page": page,
                "code": code,
                "message": message,
                "area_m2": area,
                "scale": p.get("scale"),
                "readiness": readiness,
            })

    readiness = model_readiness or {}
    if readiness:
        if readiness.get("model_status") != "final":
            issues.append({
                "severity": "critical",
                "page": None,
                "code": "model_readiness_not_final",
                "message": "Model readiness is not final.",
                "readiness": readiness,
            })
        if readiness.get("height_status") in {"default_unsafe", "conflict"}:
            issues.append({
                "severity": "critical",
                "page": None,
                "code": "height_not_verified",
                "message": f"Height status is {readiness.get('height_status')}.",
                "readiness": readiness,
            })

    critical = [x for x in issues if x.get("severity") == "critical"]
    warnings = [x for x in issues if x.get("severity") == "warning"]
    return {
        "schema": "slab_v2_quality_gate_report_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "blocked" if critical else ("review" if warnings else "pass"),
        "ready_for_delivery": not critical,
        "dominant_scale": dominant_scale,
        "median_slab_area_m2": round(med_area, 3) if med_area else None,
        "critical_count": len(critical),
        "warning_count": len(warnings),
        "issues": issues,
    }


def build_delivery_readiness_report(
        ledger: dict, quality: dict,
        model_readiness: dict | None = None) -> dict:
    blockers = [x for x in quality.get("issues", [])
                if x.get("severity") == "critical"]
    warnings = [x for x in quality.get("issues", [])
                if x.get("severity") == "warning"]
    model_status = (model_readiness or {}).get("model_status")
    ready = not blockers and (not model_status or model_status == "final")
    return {
        "schema": "slab_v2_delivery_readiness_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "delivery_status": "ready" if ready else "not_ready",
        "ready_for_client_or_boss": ready,
        "summary": ledger.get("summary", {}),
        "model_readiness": model_readiness or {},
        "quality_status": quality.get("status"),
        "blockers": blockers,
        "warnings": warnings,
        "required_human_review": [b.get("message") for b in blockers],
        "reports": {
            "audit_ledger": "audit_ledger.json",
            "quality_gate": "quality_gate_report.json",
            "delivery_readiness": "delivery_readiness_report.json",
        },
    }


def write_run_audit_reports(run_root: Path,
                            model_readiness: dict | None = None) -> dict:
    try:
        ledger = collect_audit_ledger(run_root)
        quality = build_quality_gate_report(ledger, model_readiness)
        delivery = build_delivery_readiness_report(
            ledger, quality, model_readiness)
        safe_write_json(run_root / "audit_ledger.json", ledger)
        safe_write_json(run_root / "quality_gate_report.json", quality)
        safe_write_json(run_root / "delivery_readiness_report.json", delivery)
        return {
            "audit_ledger": ledger,
            "quality_gate": quality,
            "delivery_readiness": delivery,
        }
    except Exception as exc:
        payload = {
            "schema": "slab_v2_audit_error_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error": repr(exc),
        }
        safe_write_json(run_root / "audit_error.json", payload)
        return {"error": payload}
