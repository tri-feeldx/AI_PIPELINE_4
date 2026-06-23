"""Recover missing RC columns from verified cross-floor vector evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import fitz
import numpy as np
from shapely.geometry import Point, Polygon

from src.slab_v2.models import ColumnFootprint
from src.slab_v2.wall_profile_resolver import (
    _grid_anchors,
    _solve_grid_registration,
)


def _apply(point: tuple[float, float], matrix: list[list[float]]) -> Point:
    value = np.array([point[0], point[1], 1.0]) @ np.asarray(matrix)
    return Point(float(value[0]), float(value[1]))


def _refresh_report(result) -> None:
    report = result.column_detection_report
    expected = report.get("expected", {})
    detected = defaultdict(int)
    for column in result.columns:
        detected[column.symbol] += 1
    missing = {symbol: count-detected.get(symbol, 0)
               for symbol, count in expected.items()
               if detected.get(symbol, 0) < count}
    extra = {symbol: count for symbol, count in detected.items()
             if symbol not in expected}
    ambiguous = detected.get("C?", 0)
    if not expected:
        status = "not_required"
    else:
        status = ("verified" if not missing and not extra and not ambiguous
                  else "review")
    report.update({"status": status, "detected": dict(detected),
                   "missing": missing, "extra": extra,
                   "ambiguous_count": ambiguous})
    result.column_readiness.update(report)


def reconcile_columns_across_floors(
    pdf_path: str,
    storeys: list[dict],
    output_dir: Path,
) -> dict:
    """Use adjacent-floor locations only to select real target-page candidates.

    No geometry is invented: a recovery requires verified grid registration and
    an unclaimed vector rectangle on the target page.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "not_required", "recoveries": [], "rejections": [],
              "registrations": []}
    if len(storeys) < 2:
        (output_dir / "column_cross_floor_evidence.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        return report

    doc = fitz.open(pdf_path)
    try:
        anchors = {entry["page_idx"]: _grid_anchors(doc[entry["page_idx"]])
                   for entry in storeys}
        for target in storeys:
            target_result = target["result"]
            missing = dict(target_result.column_detection_report.get(
                "missing", {}))
            if not missing:
                continue
            used_ids = {column.candidate_id for column in target_result.columns
                        if column.candidate_id}
            candidates = []
            for row in target_result.column_candidates:
                exterior = row.get("exterior") or []
                if len(exterior) < 4 or row.get("id") in used_ids:
                    continue
                poly = Polygon(exterior)
                if not poly.is_valid or poly.area <= 0:
                    continue
                candidates.append((row, poly))

            for symbol, needed in list(missing.items()):
                sources = [entry for entry in storeys if entry is not target
                           for column in entry["result"].columns
                           if column.symbol == symbol]
                sources.sort(key=lambda entry: abs(
                    float(entry.get("ffl_mm", 0))-float(target.get("ffl_mm", 0))))
                recovered = 0
                for source in sources:
                    if recovered >= needed:
                        break
                    registration = _solve_grid_registration(
                        anchors.get(source["page_idx"], {}),
                        anchors.get(target["page_idx"], {}))
                    report["registrations"].append({
                        "from_page": source["page_idx"]+1,
                        "to_page": target["page_idx"]+1,
                        "symbol": symbol,
                        **registration,
                    })
                    if registration.get("status") != "verified" or not registration.get("matrix"):
                        continue
                    source_columns = [column for column in source["result"].columns
                                      if column.symbol == symbol]
                    for source_column in source_columns:
                        projected = _apply(
                            (source_column.polygon.centroid.x,
                             source_column.polygon.centroid.y),
                            registration["matrix"])
                        ranked = sorted(
                            ((poly.distance(projected), row, poly)
                             for row, poly in candidates
                             if row.get("id") not in used_ids),
                            key=lambda item: item[0])
                        if not ranked:
                            continue
                        distance, row, poly = ranked[0]
                        # 25 PDF points is deliberately conservative; it only
                        # resolves symbol identity for an existing rectangle.
                        if distance > 25.0:
                            report["rejections"].append({
                                "symbol": symbol,
                                "target_page": target["page_idx"]+1,
                                "reason": "no vector candidate near projected grid position",
                                "distance_pt": round(distance, 3),
                            })
                            continue
                        target_result.columns.append(ColumnFootprint(
                            symbol=symbol, polygon=poly,
                            w_mm=float(row.get("w_mm") or 0),
                            d_mm=float(row.get("d_mm") or 0), labeled=False,
                            candidate_id=str(row.get("id") or ""),
                            source="cross_floor_vector_recovery",
                            confidence=0.90))
                        used_ids.add(row.get("id"))
                        recovered += 1
                        evidence = {
                            "symbol": symbol,
                            "source_page": source["page_idx"]+1,
                            "target_page": target["page_idx"]+1,
                            "candidate_id": row.get("id"),
                            "distance_pt": round(distance, 3),
                            "registration_rms_pt": registration.get("rms_error_pt"),
                            "source": "adjacent_floor_grid_projection+target_vector",
                        }
                        report["recoveries"].append(evidence)
                        target_result.column_detection_report.setdefault(
                            "assignments", []).append(evidence)
                        break
                _refresh_report(target_result)

        for entry in storeys:
            _refresh_report(entry["result"])
        statuses = {entry["result"].column_detection_report.get("status")
                    for entry in storeys}
        report["status"] = ("verified" if statuses <= {"verified", "not_required"}
                            else "review")
    finally:
        doc.close()
    (output_dir / "column_cross_floor_evidence.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
