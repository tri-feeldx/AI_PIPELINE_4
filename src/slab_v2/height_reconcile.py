"""Traceable level-datum evidence collection and robust graph solving."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import fitz
import numpy as np

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import (
    DocAnalysis, FloorHeight, HeightEvidence, HeightReconciliation, LevelDatum,
)
from src.pdf_processor import (
    detect_scale_from_blocks, extract_ffl_values, extract_text_blocks,
)


_LEVEL_RE = re.compile(r"\b(?:LEVEL|LVL|L)\s*0*(\d{1,2})\b", re.I)
_HEIGHT_RE = re.compile(
    r"\b(?:STOREY|STORY|FLOOR[ -]?TO[ -]?FLOOR|LEVEL)\s*(?:HEIGHT)?"
    r"\s*[:=]?\s*(\d{3,5}(?:\.\d+)?)\s*(MM|M)?\b", re.I)
_BAD_CONTEXT_RE = re.compile(
    r"\b(SETDOWN|STEP|REBATE|THICK|THICKNESS|GRID|BAND|SLAB)\b", re.I)
_STRICT_DATUM_RE = re.compile(
    r"\b(FFL|RL|EL|AHD|NGL)\b\s*(?:=|:|\+)?\s*"
    r"([+-]?\d{1,5}(?:\.\d{1,3})?)\s*(MM|M|MAHD)?\b", re.I)


def _norm_level(text: str) -> str:
    m = re.search(r"(\d+)", text or "")
    return f"level_{int(m.group(1)):02d}" if m else (text or "").lower()


def _level_num(level_id: str) -> Optional[int]:
    m = re.search(r"(\d+)", level_id or "")
    return int(m.group(1)) if m else None


def _building_level_id(building, number: int) -> str:
    for floor in building.floors:
        if _level_num(floor.level_id) == number:
            return floor.level_id
    return f"level_{number:02d}"


def _sane_ffl_m(value: float, source: str) -> bool:
    # -50/-100 setdowns parsed without units are the common false positives.
    return -20.0 <= value <= 500.0 and not _BAD_CONTEXT_RE.search(source or "")


def _evidence_id(prefix: str, counter: int) -> str:
    return f"{prefix}_{counter:04d}"


def _strict_datums(blocks: list[dict]) -> list[dict]:
    """Datum parser that cannot confuse LEVEL/FLOOR/address text with FL."""
    out = []
    for block in blocks:
        text = str(block.get("text") or "")
        if _BAD_CONTEXT_RE.search(text):
            continue
        for match in _STRICT_DATUM_RE.finditer(text):
            value = float(match.group(2))
            unit = (match.group(3) or "").upper()
            if unit == "MM" or (not unit and abs(value) >= 1000):
                value_m = value / 1000.0
            else:
                value_m = value
            if _sane_ffl_m(value_m, text):
                out.append({"ffl_m": value_m,
                            "bbox": block.get("bbox"),
                            "source_text": text})
    return out


def _consolidate_elevation_evidence(evidence: list[HeightEvidence]) -> list:
    """A repeated elevation view is corroboration, not extra solver votes."""
    groups: dict[tuple, list[HeightEvidence]] = {}
    passthrough = []
    for e in evidence:
        if e.evidence_type != "scaled_elevation_spacing":
            passthrough.append(e)
            continue
        key = (e.building, e.from_level, e.to_level, e.evidence_type)
        groups.setdefault(key, []).append(e)
    for rows in groups.values():
        bins: dict[int, list[HeightEvidence]] = {}
        for e in rows:
            bins.setdefault(round(e.value_mm / 100.0), []).append(e)
        dominant = max(bins.values(), key=lambda x: (len(x),
                                                       sum(e.confidence for e in x)))
        values = sorted(e.value_mm for e in dominant)
        median = values[len(values) // 2]
        best = max(dominant, key=lambda e: e.confidence)
        pages = sorted({e.page_index + 1 for e in dominant})
        passthrough.append(HeightEvidence(
            id=best.id, building=best.building,
            from_level=best.from_level, to_level=best.to_level,
            evidence_type=best.evidence_type, value_mm=median,
            page_index=best.page_index, bbox=best.bbox,
            source_text=(f"{best.source_text}; corroborated on pages {pages}"),
            extraction_method=best.extraction_method,
            confidence=min(0.85, best.confidence + 0.05 * (len(pages) - 1)),
            is_absolute_datum=False))
    return passthrough


def _candidate_height_pages(doc: fitz.Document, analysis: DocAnalysis) -> set[int]:
    pages = set(analysis.wall_elevation_pages or [])
    raw = analysis.raw or {}
    for key in ("elevation_pages", "section_pages", "height_schedule_pages",
                "storey_height_pages"):
        for p in raw.get(key, []) or []:
            try:
                pi = int(p) - 1
                if 0 <= pi < len(doc):
                    pages.add(pi)
            except (TypeError, ValueError):
                pass
    for i, page in enumerate(doc):
        txt = page.get_text("text")[:8000].upper()
        if (("ELEVATION" in txt or "SECTION" in txt)
                and ("LEVEL" in txt or "FFL" in txt or "RL" in txt)):
            pages.add(i)
    return pages


def _page_building(page_text: str, buildings: list) -> Optional[str]:
    hits = [b.name for b in buildings
            if b.name and b.name.upper() in page_text.upper()]
    if len(hits) == 1:
        return hits[0]
    if len(buildings) == 1:
        return buildings[0].name
    return None


def collect_height_evidence(pdf_path: str, analysis: DocAnalysis,
                            cfg: SlabV2Config,
                            manual_overrides: dict | None = None) -> list[HeightEvidence]:
    """Collect provenance-rich absolute and relative level constraints."""
    doc = fitz.open(pdf_path)
    evidence: list[HeightEvidence] = []
    n = 0

    # Manual overrides are pinned constraints: {"Building/level_01": ffl_mm}
    for key, value in (manual_overrides or {}).items():
        if "/" not in key:
            continue
        building, level = key.split("/", 1)
        n += 1
        evidence.append(HeightEvidence(
            _evidence_id("manual", n), building, None, level,
            "manual", float(value), source_text="manual override",
            extraction_method="manual", confidence=1.0,
            is_absolute_datum=True))

    # Floor-plan mapping is useful context, but its Gemini ffl_m is not height
    # evidence. Older cached responses may contain the former 3.5m estimate.
    for building in analysis.buildings:
        for floor in building.floors:
            # Explicit datum text on the mapped plan page.
            for pi in floor.pages:
                if not (0 <= pi < len(doc)):
                    continue
                for found in _strict_datums(extract_text_blocks(doc[pi])):
                    source = str(found.get("source_text") or "")
                    value_m = float(found["ffl_m"])
                    if not _sane_ffl_m(value_m, source):
                        continue
                    n += 1
                    evidence.append(HeightEvidence(
                        _evidence_id("explicit_datum", n), building.name,
                        None, floor.level_id, "explicit_datum",
                        value_m * 1000.0, page_index=pi,
                        bbox=tuple(found.get("bbox") or ()),
                        source_text=source, extraction_method="page_text_regex",
                        confidence=0.95, is_absolute_datum=True))

    # A single compact Gemini call selects source pages. Code then locates
    # real vector datum lines and measures them with a verified viewport scale.
    from src.slab_v2.height_source_planner import (
        build_consensus, extract_planned_absolute_datums,
        measure_planned_sources, plan_height_sources, write_measurement_overlays,
    )
    from src.slab_v2.pipeline import run_dir
    out_dir = run_dir(cfg, pdf_path)
    planner, candidate_pages = plan_height_sources(doc, analysis, cfg, out_dir)
    measured, viewport_reports = measure_planned_sources(
        doc, analysis, planner, out_dir)
    evidence.extend(extract_planned_absolute_datums(doc, analysis, planner))
    write_measurement_overlays(doc, measured, out_dir, viewport_reports)
    consensus_evidence, consensus_report = build_consensus(measured)
    evidence.extend(consensus_evidence)
    analysis.raw["height_source_planner"] = planner
    analysis.raw["height_candidate_pages"] = candidate_pages
    analysis.raw["height_viewports"] = viewport_reports
    analysis.raw["height_consensus_report"] = consensus_report

    # Explicit storey-height text from selected source pages remains stronger
    # than geometric spacing and does not require scale.
    building_lookup = {b.name: b for b in analysis.buildings}
    for source in planner.get("sources", []):
        pi = int(source.get("page", 0)) - 1
        building_name = str(source.get("building") or "")
        if pi < 0 or pi >= len(doc) or building_name not in building_lookup:
            continue
        page = doc[pi]
        page_text = page.get_text("text")
        for block in extract_text_blocks(page):
            match = _HEIGHT_RE.search(block["text"])
            if not match:
                continue
            raw_value = float(match.group(1))
            unit = (match.group(2) or "MM").upper()
            value = raw_value * 1000.0 if unit == "M" else raw_value
            if not 2000 <= value <= 8000:
                continue
            nearby = page_text[max(0, page_text.find(block["text"]) - 100):]
            lm = _LEVEL_RE.search(nearby[:300])
            if not lm:
                continue
            lower = int(lm.group(1))
            n += 1
            building_obj = building_lookup[building_name]
            evidence.append(HeightEvidence(
                _evidence_id("explicit_height", n), building_name,
                _building_level_id(building_obj, lower),
                _building_level_id(building_obj, lower + 1),
                "explicit_storey_height", value, page_index=pi,
                bbox=tuple(block["bbox"]), source_text=block["text"],
                extraction_method="section_text_regex", confidence=0.90,
                is_absolute_datum=False))

    doc.close()
    return evidence


def _weight(e: HeightEvidence) -> float:
    factors = {
        "manual": 100.0, "explicit_datum": 25.0,
        "explicit_storey_height": 16.0,
        "verified_consensus": 14.0,
        "scaled_elevation_measurement": 5.0,
        "scaled_elevation_spacing": 5.0,
        "default": 0.01,
    }
    return factors.get(e.evidence_type, 1.0) * max(e.confidence, 0.05) ** 2


def _solve_building(building, evidence: list[HeightEvidence],
                    cfg: SlabV2Config) -> tuple[list[LevelDatum], list[dict]]:
    levels = [f.level_id for f in building.floors
              if "roof" not in f.level_id.lower()]
    levels = list(dict.fromkeys(levels))
    if not levels:
        return [], []
    idx = {level: i for i, level in enumerate(levels)}
    building_evidence = [e for e in evidence if e.building == building.name]
    usable = [e for e in building_evidence
              if e.to_level in idx
              and (e.is_absolute_datum or e.from_level in idx)]
    terminal_by_level = {
        level: [e for e in building_evidence
                if not e.is_absolute_datum and e.from_level == level
                and e.to_level not in idx and 2000 <= e.value_mm <= 8000]
        for level in levels
    }
    has_absolute = any(e.is_absolute_datum for e in usable)
    synthetic_anchor = False

    if not usable:
        datums = []
        for i, level in enumerate(levels):
            datums.append(LevelDatum(
                building.name, level, i * cfg.default_storey_height_mm,
                cfg.default_storey_height_mm, "default_unsafe", 0.0,
                warnings=["No reliable height evidence; debug default used."]))
        return datums, []

    rows, values, weights, row_evidence = [], [], [], []
    for e in usable:
        row = np.zeros(len(levels), dtype=float)
        if e.is_absolute_datum:
            row[idx[e.to_level]] = 1.0
        else:
            row[idx[e.to_level]] = 1.0
            row[idx[e.from_level]] = -1.0
        rows.append(row)
        values.append(e.value_mm)
        weights.append(_weight(e))
        row_evidence.append(e)
    if has_absolute:
        # A datum graph needs a stable origin. Pin the strongest absolute
        # source (manual > explicit text > Gemini) and let other absolutes
        # remain auditable conflict checks rather than shifting the datum.
        anchor = max((e for e in usable if e.is_absolute_datum), key=_weight)
        row = np.zeros(len(levels), dtype=float)
        row[idx[anchor.to_level]] = 1.0
        rows.append(row)
        values.append(anchor.value_mm)
        weights.append(1000.0)
        row_evidence.append(None)
    else:
        row = np.zeros(len(levels), dtype=float)
        row[0] = 1.0
        rows.append(row)
        values.append(0.0)
        weights.append(100.0)
        row_evidence.append(None)
        synthetic_anchor = True

    A = np.asarray(rows)
    b = np.asarray(values)
    base_w = np.asarray(weights)
    w = base_w.copy()
    x = np.zeros(len(levels))
    for _ in range(6):
        sw = np.sqrt(np.maximum(w, 1e-9))
        x, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        residual = A @ x - b
        delta = cfg.height_reject_residual_mm
        huber = np.ones_like(residual)
        mask = np.abs(residual) > delta
        huber[mask] = delta / np.abs(residual[mask])
        w = base_w * huber

    residual = A @ x - b
    conflicts = []
    rejected = set()
    support_by_level = {level: [] for level in levels}
    reject_by_level = {level: [] for level in levels}
    for e, r in zip(row_evidence, residual):
        if e is None:
            continue
        targets = [e.to_level] + ([] if e.is_absolute_datum else [e.from_level])
        if abs(r) > cfg.height_reject_residual_mm:
            rejected.add(e.id)
            for level in targets:
                reject_by_level[level].append(e.id)
        else:
            for level in targets:
                support_by_level[level].append(e.id)
        if (abs(r) > cfg.height_conflict_tolerance_mm
                and e.confidence >= 0.5):
            conflicts.append({"evidence_id": e.id, "building": building.name,
                              "residual_mm": round(float(r), 1),
                              "source_text": e.source_text})

    datums = []
    for i, level in enumerate(levels):
        incident = [e for e in usable
                    if e.to_level == level or e.from_level == level]
        if i == len(levels) - 1:
            incident += terminal_by_level.get(level, [])
        accepted = [e for e in incident if e.id not in rejected]
        if i == len(levels) - 1:
            for terminal in terminal_by_level.get(level, []):
                if terminal.id not in support_by_level[level]:
                    support_by_level[level].append(terminal.id)
        manual = any(e.evidence_type == "manual" for e in accepted)
        explicit = any(e.evidence_type == "explicit_datum" for e in accepted)
        consensus = any(e.evidence_type == "verified_consensus"
                        for e in accepted)
        level_conflicts = [c for c in conflicts
                           if c["evidence_id"] in {e.id for e in incident}]
        if manual:
            status, confidence = "manual", 1.0
        elif level_conflicts:
            status, confidence = "conflict", 0.3
        elif synthetic_anchor:
            status, confidence = "inferred_relative", 0.55
        elif explicit:
            status, confidence = "verified_explicit", 0.95
        elif consensus:
            status, confidence = "verified_consensus", 0.90
        else:
            status, confidence = "inferred", 0.6

        if i + 1 < len(levels):
            height = float(x[i + 1] - x[i])
        else:
            outgoing = [e for e in accepted
                        if e.from_level == level and not e.is_absolute_datum]
            if outgoing:
                height = float(np.average(
                    [e.value_mm for e in outgoing],
                    weights=[_weight(e) for e in outgoing]))
            elif i > 0:
                height = float(x[i] - x[i - 1])
                if status in {"verified_explicit", "verified_consensus"}:
                    status, confidence = "inferred", min(confidence, 0.6)
            else:
                height = cfg.default_storey_height_mm
                status, confidence = "default_unsafe", 0.0
        warns = []
        if not 2000 <= height <= 8000:
            warns.append(f"Suspicious storey height {height:.0f}mm.")
            status = "conflict"
        if status == "inferred_relative":
            warns.append("Relative levels only; Level 01 anchored to 0.000m.")
        datums.append(LevelDatum(
            building.name, level, round(float(x[i]), 1), round(height, 1),
            status, confidence, support_by_level[level],
            reject_by_level[level], warns))
    return datums, conflicts


def solve_level_datums(evidence: list[HeightEvidence], analysis: DocAnalysis,
                       cfg: SlabV2Config) -> HeightReconciliation:
    datums, conflicts = [], []
    for building in analysis.buildings:
        bd, bc = _solve_building(building, evidence, cfg)
        datums.extend(bd)
        conflicts.extend(bc)

    floors = [FloorHeight(
        level_id=d.level_id, building=d.building,
        ffl_m=(d.ffl_mm or 0.0) / 1000.0,
        storey_height_mm=d.storey_height_mm or cfg.default_storey_height_mm,
        sources={"evidence_count": float(len(d.supporting_evidence_ids))},
        confidence=d.status) for d in datums]
    warnings = []
    for d in datums:
        warnings.extend(f"{d.building}/{d.level_id}: {w}" for w in d.warnings)
        if d.status in {"conflict", "default_unsafe"}:
            warnings.append(f"{d.building}/{d.level_id}: height status={d.status}")
    return HeightReconciliation(
        floors=floors, method="level_datum_graph", warnings=warnings,
        debug_log=[f"evidence={len(evidence)} datums={len(datums)} "
                   f"conflicts={len(conflicts)}"],
        evidence=evidence, level_datums=datums, conflicts=conflicts)


def _write_audit(pdf_path: str, cfg: SlabV2Config,
                 result: HeightReconciliation) -> None:
    try:
        from src.slab_v2.pipeline import run_dir
        out = run_dir(cfg, pdf_path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "height_evidence.json").write_text(json.dumps(
            [asdict(e) for e in result.evidence], indent=2,
            ensure_ascii=False), encoding="utf-8")
        (out / "height_solver_report.json").write_text(json.dumps(
            [asdict(d) for d in result.level_datums], indent=2,
            ensure_ascii=False), encoding="utf-8")
        (out / "height_conflicts.json").write_text(json.dumps(
            result.conflicts, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "height_consensus_report.json").write_text(json.dumps(
            result.consensus_report, indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out / "level_datum_report.json").write_text(json.dumps({
            "method": result.method, "warnings": result.warnings,
            "levels": [asdict(d) for d in result.level_datums],
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        # Visual provenance: every source bbox used or rejected stays visible.
        from PIL import Image, ImageDraw
        doc = fitz.open(pdf_path)
        rejected = {eid for d in result.level_datums
                    for eid in d.rejected_evidence_ids}
        by_page: dict[int, list[HeightEvidence]] = {}
        for e in result.evidence:
            if e.page_index >= 0 and e.bbox:
                by_page.setdefault(e.page_index, []).append(e)
        for pi, rows in by_page.items():
            if not (0 <= pi < len(doc)):
                continue
            scale = 150 / 72.0
            pix = doc[pi].get_pixmap(matrix=fitz.Matrix(scale, scale),
                                     alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            dr = ImageDraw.Draw(img)
            for e in rows:
                x0, y0, x1, y1 = e.bbox
                color = (220, 40, 40) if e.id in rejected else (20, 150, 60)
                box = [x0 * scale, y0 * scale, x1 * scale, y1 * scale]
                dr.rectangle(box, outline=color, width=4)
                dr.text((box[0], max(0, box[1] - 14)), e.id, fill=color)
            img.save(out / f"height_evidence_p{pi + 1:02d}.png")
        doc.close()
    except Exception as exc:
        result.warnings.append(f"height audit output failed: {exc}")


def reconcile_heights(pdf_path: str, doc_analysis: DocAnalysis,
                      cfg: SlabV2Config,
                      manual_overrides: dict | None = None) -> HeightReconciliation:
    """Backward-compatible facade over evidence collection and graph solve."""
    evidence = collect_height_evidence(
        pdf_path, doc_analysis, cfg, manual_overrides=manual_overrides)
    result = solve_level_datums(evidence, doc_analysis, cfg)
    result.source_planner = (doc_analysis.raw or {}).get(
        "height_source_planner", {})
    result.consensus_report = (doc_analysis.raw or {}).get(
        "height_consensus_report", [])
    result.candidate_pages = (doc_analysis.raw or {}).get(
        "height_candidate_pages", [])
    for warning in result.source_planner.get("warnings", []) or []:
        result.warnings.append(f"Height source planner: {warning}")
    _write_audit(pdf_path, cfg, result)
    return result
