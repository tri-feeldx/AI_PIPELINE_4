"""
Site/keyplan building placement audit.

Gemini finds pages that may contain building-position evidence. Deterministic
code then extracts label/polygon evidence from those pages. This module is
deliberately conservative: it records proposed transforms, but only marks site
placement verified when all expected buildings have unique, non-overlapping
evidence and scale confidence is acceptable.
"""

from __future__ import annotations

from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any

import fitz
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping

from src.document_intelligence import _load_gemini_client, _strip_json_fence


PROMPT_BUILDING_POSITION_SOURCES = r"""
You are a principal structural drawing analyst.
You receive compact extracted text for every page of one structural PDF.

Find pages that can be used to locate buildings relative to each other on site:
KEYPLAN, SITE PLAN, OVERALL GA plan, overall carpark/building plan, or any page
where Building A/B/C/D or BLD A/B/C/D are visible together in one coordinate frame.

Return ONLY valid compact JSON:
{
  "primary_building_position_source": {
    "page": number,
    "title": string|null,
    "source_type": "keyplan"|"site_plan"|"overall_plan"|"ga_plan_overall"|"title_block_keyplan"|"unknown",
    "buildings_visible": ["Building A"],
    "recommended_for_site_placement": true,
    "why_chosen": string,
    "why_better_than_other_candidates": string,
    "confidence": number
  },
  "expected_buildings": [
    {
      "canonical_name": "Building A",
      "aliases": ["Building A", "BLD A", "Proposed Building A"],
      "short_labels": ["A"]
    }
  ],
  "building_position_sources": [
    {
      "page": number,
      "title": string|null,
      "source_type": "keyplan"|"site_plan"|"overall_plan"|"ga_plan_overall"|"title_block_keyplan"|"unknown",
      "buildings_visible": ["Building A"],
      "recommended_for_site_placement": true,
      "reason": string,
      "confidence": number
    }
  ],
  "warnings": [string]
}

Rules:
- Prefer pages where multiple buildings are shown together in one small keyplan/site/overall view.
- You MUST choose exactly one primary_building_position_source.
- The primary page MUST show every expected building in the same coordinate frame. If the project has Building A/B/C/D/E,
  the primary page must show A, B, C, D, and E. If no single page does, set page to null and explain why.
- Choose the strongest single page in this order: full site plan, overall location site plan, overall GA/key plan,
  title-block keyplan. Prefer a full-page/site drawing over a tiny title-block keyplan.
- In why_chosen, explicitly explain why this one page is the best page to polygon for building placement.
- In why_better_than_other_candidates, compare it against other candidate pages.
- Do not choose per-building detailed plan pages as final site placement unless their title block keyplan shows all buildings.
- Do not invent coordinates. Only identify candidate source pages and visible building labels.
- expected_buildings must list all building names you expect in the project. Include aliases/short_labels exactly as they may
  appear on keyplans/site plans, including Building 1/2/3, Tower AB, Block AV, BLD C, Proposed Building A, etc.
"""


BUILDING_LABEL_RE = re.compile(r"\b(?:BLD(?:G)?|BUILDING)\b\s*([A-Z0-9]{1,4})\b", re.IGNORECASE)
SCALE_RE = re.compile(
    r"\bSCALE\b\s*(?:AT\s*A\d\s*)?:?\s*(?:1\s*[:=]\s*|1\s+TO\s+)(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
LOOSE_SCALE_RE = re.compile(r"\b1\s*[:=]\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
PT_TO_MM = 25.4 / 72.0


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "geom_type"):
        return mapping(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return str(path)


def _parse_json(text: str) -> tuple[dict, dict]:
    cleaned = _strip_json_fence(text or "")
    report = {"parse_status": "ok", "parse_error": None}
    try:
        return json.loads(cleaned), report
    except Exception as exc:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                return json.loads(m.group(0)), report
            except Exception as exc2:
                exc = exc2
        report["parse_status"] = "invalid_json"
        report["parse_error"] = str(exc)
        return {"building_position_sources": [], "expected_buildings": [], "warnings": [str(exc)]}, report


def _page_title(page: fitz.Page) -> str:
    rect = page.rect
    clips = [
        fitz.Rect(rect.x0, rect.y1 * 0.72, rect.x1, rect.y1),
        fitz.Rect(rect.x1 * 0.55, rect.y1 * 0.55, rect.x1, rect.y1),
    ]
    chunks = []
    for clip in clips:
        txt = page.get_text("text", clip=clip).strip()
        if txt:
            chunks.append(" ".join(txt.split()))
    return " | ".join(chunks)[:900]


def _parse_page_scale(page: fitz.Page) -> dict:
    text = " ".join(page.get_text("text").split())
    matches = [float(m.group(1)) for m in SCALE_RE.finditer(text)]
    if not matches:
        lines = page.get_text("text").splitlines()
        for idx, line in enumerate(lines):
            window = " ".join(lines[idx:idx + 4])
            if "SCALE" not in window.upper():
                continue
            loose = LOOSE_SCALE_RE.search(window)
            if loose:
                matches.append(float(loose.group(1)))
                break
    if not matches:
        return {
            "scale_status": "not_verified",
            "scale_ratio": None,
            "source": None,
            "reason": "No explicit SCALE 1:n text found on primary source page.",
        }
    ratio = matches[0]
    return {
        "scale_status": "verified",
        "scale_ratio": ratio,
        "source": f"SCALE 1:{ratio:g}",
        "reason": "Explicit scale text found on primary source page.",
    }


def _compact_page_catalog(pdf_path: str) -> tuple[str, int]:
    doc = fitz.open(pdf_path)
    parts = []
    try:
        for i, page in enumerate(doc):
            text = " ".join(page.get_text("text").split())
            upper = text.upper()
            interesting = []
            for kw in ("KEYPLAN", "KEY PLAN", "SITE PLAN", "OVERALL", "BUILDING A", "BUILDING B", "BUILDING C", "BUILDING D", "BLD A", "BLD B", "BLD C", "BLD D"):
                if kw in upper:
                    interesting.append(kw)
            title = _page_title(page)
            snippet = text[:1800]
            parts.append(
                f"[Page {i + 1}] TITLE_BLOCK={title}\nKEYWORDS={', '.join(sorted(set(interesting)))}\nTEXT={snippet}"
            )
        return "\n\n".join(parts), doc.page_count
    finally:
        doc.close()


def analyze_building_position_sources(pdf_path: str, out_dir: str | Path) -> tuple[dict, str, str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog, page_count = _compact_page_catalog(pdf_path)
    client, model = _load_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=[PROMPT_BUILDING_POSITION_SOURCES, f"PDF page_count={page_count}\nPAGE_CATALOG:\n{catalog}"],
    )
    raw = (response.text or "").strip()
    raw_path = out / "01_gemini_building_position_sources_raw.txt"
    json_path = out / "01_gemini_building_position_sources.json"
    report_path = out / "01_gemini_building_position_sources_parse_report.json"
    raw_path.write_text(raw, encoding="utf-8")
    parsed, report = _parse_json(raw)
    report.update({"raw_response_path": str(raw_path), "parsed_json_path": str(json_path), "page_count": page_count})
    parsed["_metadata"] = report
    json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return parsed, str(json_path), str(raw_path), str(report_path)


def _default_aliases(canonical_name: str) -> list[str]:
    name = str(canonical_name or "").strip()
    aliases = [name]
    m = re.search(r"\b(?:BLD(?:G)?|BUILDING|TOWER|BLOCK)\b\s*([A-Z0-9]{1,4})\b", name, re.IGNORECASE)
    if m:
        code = m.group(1).upper()
        aliases.extend([
            code,
            f"Building {code}",
            f"BLD {code}",
            f"BLDG {code}",
            f"Tower {code}",
            f"Block {code}",
            f"Proposed Building {code}",
        ])
    return list(dict.fromkeys(a for a in aliases if a))


def _expected_building_specs(parsed: dict) -> list[dict]:
    raw = parsed.get("expected_buildings") or []
    specs = []
    if raw and isinstance(raw[0], dict):
        for item in raw:
            canonical = item.get("canonical_name") or item.get("name") or ""
            aliases = []
            aliases.extend(item.get("aliases") or [])
            aliases.extend(item.get("short_labels") or [])
            aliases.extend(_default_aliases(canonical))
            specs.append({
                "canonical_name": canonical or (aliases[0] if aliases else ""),
                "aliases": list(dict.fromkeys(str(a).strip() for a in aliases if str(a).strip())),
            })
    else:
        for name in raw:
            canonical = str(name).strip()
            specs.append({"canonical_name": canonical, "aliases": _default_aliases(canonical)})

    if specs:
        return specs
    seen = set()
    for src in parsed.get("building_position_sources", []) or []:
        seen.update(_norm_building_names(src.get("buildings_visible") or []))
    names = sorted(seen)
    return [{"canonical_name": name, "aliases": _default_aliases(name)} for name in names]


def _extract_text_labels(page: fitz.Page, building_specs: list[dict]) -> list[dict]:
    labels = []
    seen = set()
    for spec in building_specs:
        building = spec.get("canonical_name") or ""
        aliases = sorted(spec.get("aliases") or [], key=len, reverse=True)
        long_aliases = [a for a in aliases if len(str(a).strip()) > 2]
        short_aliases = [a for a in aliases if len(str(a).strip()) <= 2]
        building_found = False
        for alias in long_aliases + short_aliases:
            if building_found and alias in short_aliases:
                continue
            if not alias or len(alias.strip()) < 1:
                continue
            try:
                hits = page.search_for(alias, quads=False)
            except Exception:
                hits = []
            for rect in hits:
                key = (building, alias.lower(), round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
                if key in seen:
                    continue
                seen.add(key)
                labels.append({
                    "building": building,
                    "alias": alias,
                    "text": alias,
                    "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                })
                building_found = True

    if labels:
        return labels

    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                for m in BUILDING_LABEL_RE.finditer(text):
                    code = m.group(1).upper()
                    labels.append({
                        "building": f"Building {code}",
                        "alias": m.group(0),
                        "text": text.strip(),
                        "bbox": list(span.get("bbox", block.get("bbox"))),
                    })
    return labels


def _rect_to_poly(bbox, pad=0.0) -> Polygon:
    x0, y0, x1, y1 = bbox
    return Polygon([
        (x0 - pad, y0 - pad),
        (x1 + pad, y0 - pad),
        (x1 + pad, y1 + pad),
        (x0 - pad, y1 + pad),
    ])


def _color_tuple(value):
    if value is None:
        return None
    try:
        return tuple(round(float(v), 3) for v in value[:3])
    except Exception:
        return None


def _is_whiteish(color) -> bool:
    if color is None:
        return False
    return min(color) > 0.92


def _is_colored_fill(color) -> bool:
    if color is None or _is_whiteish(color):
        return False
    return max(color) - min(color) > 0.08 or max(color) < 0.90


def _drawing_polygons(page: fitz.Page) -> list[dict]:
    candidates = []
    for d in page.get_drawings():
        fill = _color_tuple(d.get("fill"))
        stroke = _color_tuple(d.get("color"))
        rect = d.get("rect")
        if rect and rect.width > 6 and rect.height > 4:
            poly = _rect_to_poly((rect.x0, rect.y0, rect.x1, rect.y1))
            candidates.append({"polygon": poly, "fill": fill, "stroke": stroke, "source": "drawing_rect", "items": len(d.get("items", []))})
        pts = []
        for item in d.get("items", []):
            if item and item[0] == "l":
                pts.extend([(item[1].x, item[1].y), (item[2].x, item[2].y)])
            elif item and item[0] == "re":
                r = item[1]
                if r.width > 6 and r.height > 4:
                    poly = _rect_to_poly((r.x0, r.y0, r.x1, r.y1))
                    candidates.append({"polygon": poly, "fill": fill, "stroke": stroke, "source": "path_rect", "items": len(d.get("items", []))})
        if len(pts) >= 4:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if max(xs) - min(xs) > 6 and max(ys) - min(ys) > 4:
                poly = _rect_to_poly((min(xs), min(ys), max(xs), max(ys)))
                candidates.append({"polygon": poly, "fill": fill, "stroke": stroke, "source": "path_bbox", "items": len(d.get("items", []))})
    return candidates


def _best_polygon_for_label(label: dict, drawing_polys: list[dict], page: fitz.Page) -> tuple[Polygon, str, float, str | None]:
    lb = _rect_to_poly(label["bbox"], pad=12)
    centroid = lb.centroid
    candidates = []
    page_area = max(page.rect.width * page.rect.height, 1.0)
    for cand in drawing_polys:
        poly = cand["polygon"]
        area_ratio = poly.area / page_area
        min_dim = min(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1])
        if area_ratio > 0.20:
            continue
        if area_ratio < 0.00025 or min_dim < 12:
            continue
        dist = poly.distance(centroid)
        intersects = poly.intersects(lb)
        contains = poly.contains(centroid)
        if intersects or contains or dist < 120:
            fill = cand.get("fill")
            score = 0.0
            score += 0.55 if intersects else 0.0
            score += 0.45 if contains else 0.0
            score += max(0.0, 0.35 - dist / 260.0)
            score += 0.70 if _is_colored_fill(fill) else 0.0
            score -= 0.45 if _is_whiteish(fill) else 0.0
            score -= 0.35 if area_ratio < 0.0012 else 0.0
            score += min(0.25, area_ratio * 8.0)
            reason = []
            if contains:
                reason.append("contains_label")
            elif intersects:
                reason.append("intersects_label")
            else:
                reason.append("near_label")
            if _is_colored_fill(fill):
                reason.append("colored_fill")
            if _is_whiteish(fill):
                reason.append("white_fill_penalty")
            candidates.append((score, dist, -poly.area, poly, cand, ";".join(reason)))
    if candidates:
        candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
        score, _, _, poly, cand, reason = candidates[0]
        source = "colored_fill_polygon" if _is_colored_fill(cand.get("fill")) else "nearby_vector_polygon"
        return poly, source, max(0.1, min(0.98, score / 2.0)), reason
    return _rect_to_poly(label["bbox"], pad=18), "label_bbox_fallback", 0.45, "no_nearby_polygon"


def _save_candidate_page_overlay(page: fitz.Page, save_path: Path, labels: list[dict], polygons: list[dict]) -> str:
    dpi = 150
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    import numpy as np
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    fig, ax = plt.subplots(figsize=(pix.width / 100, pix.height / 100), dpi=100)
    ax.imshow(img, origin="upper")
    colors = {"Building A": "#00C853", "Building B": "#2962FF", "Building C": "#FF6D00", "Building D": "#D500F9"}
    for item in polygons:
        poly = item["polygon"]
        bld = item["building"]
        color = colors.get(bld, "#00B8D4")
        xs, ys = poly.exterior.xy
        ax.fill([x * dpi / 72 for x in xs], [y * dpi / 72 for y in ys], facecolor=color, edgecolor=color, alpha=0.28, linewidth=2.0)
        c = poly.centroid
        ax.text(c.x * dpi / 72, c.y * dpi / 72, f"{bld}\n{item['confidence']:.2f}", color="white", fontsize=8,
                ha="center", va="center", bbox=dict(facecolor="black", alpha=0.75, edgecolor=color, pad=2))
    for label in labels:
        x0, y0, x1, y1 = label["bbox"]
        ax.add_patch(plt.Rectangle((x0 * dpi / 72, y0 * dpi / 72), (x1 - x0) * dpi / 72, (y1 - y0) * dpi / 72,
                                   fill=False, edgecolor="yellow", linewidth=1.2))
    ax.set_xlim(0, pix.width)
    ax.set_ylim(pix.height, 0)
    ax.axis("off")
    ax.set_title("Building position source / label-polygon evidence", fontsize=8, color="white", backgroundcolor="black")
    fig.tight_layout(pad=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return str(save_path)


def _norm_building_names(names: list[str] | None) -> set[str]:
    out = set()
    for name in names or []:
        if isinstance(name, dict):
            name = name.get("canonical_name") or name.get("name") or ""
        m = re.search(r"\b(?:BLD(?:G)?|BUILDING|TOWER|BLOCK)\b\s*([A-Z0-9]{1,4})\b", str(name), re.IGNORECASE)
        if m:
            out.add(f"Building {m.group(1).upper()}")
    return out


def _expected_from_sources(parsed: dict) -> list[str]:
    expected = [spec["canonical_name"] for spec in _expected_building_specs(parsed) if spec.get("canonical_name")]
    if expected:
        return sorted(set(expected))
    seen = set()
    for src in parsed.get("building_position_sources", []) or []:
        seen.update(_norm_building_names(src.get("buildings_visible") or []))
    return sorted(seen)


def _source_pages(parsed: dict, page_count: int, expected_buildings: list[str]) -> list[int]:
    primary_src = parsed.get("primary_building_position_source") or {}
    primary_page = primary_src.get("page")
    primary_visible = _norm_building_names(primary_src.get("buildings_visible") or [])
    expected = set(expected_buildings)
    if (
        isinstance(primary_page, int)
        and 1 <= primary_page <= page_count
        and (not expected or expected.issubset(primary_visible))
    ):
        return [primary_page - 1]

    primary = []
    fallback = []
    for src in parsed.get("building_position_sources", []) or []:
        p = src.get("page")
        visible = _norm_building_names(src.get("buildings_visible") or [])
        if expected and not expected.issubset(visible):
            continue
        if not isinstance(p, int) or not (1 <= p <= page_count):
            continue
        source_type = str(src.get("source_type") or "").lower()
        target = primary if source_type in {"site_plan", "overall_plan", "ga_plan_overall"} else fallback
        if p - 1 not in target:
            target.append(p - 1)
    pages = primary or fallback
    return pages[:10]


def _local_candidate_pages(pdf_path: str, expected_buildings: list[str]) -> list[dict]:
    doc = fitz.open(pdf_path)
    rows = []
    expected = set(expected_buildings)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            upper = text.upper()
            labels = sorted({f"Building {m.group(1).upper()}" for m in BUILDING_LABEL_RE.finditer(text)})
            if expected and not expected.issubset(set(labels)):
                continue
            score = 0
            if "KEYPLAN" in upper or "KEY PLAN" in upper:
                score += 3
            if "OVERALL" in upper or "SITE PLAN" in upper:
                score += 2
            score += min(len(labels), 4)
            has_placement_keyword = (
                "KEYPLAN" in upper or "KEY PLAN" in upper or
                "OVERALL" in upper or "SITE PLAN" in upper
            )
            if has_placement_keyword and score >= 3:
                rows.append({
                    "page": i + 1,
                    "title": _page_title(page),
                    "labels_found": labels,
                    "local_score": score,
                })
        rows.sort(key=lambda r: (-r["local_score"], r["page"]))
        return rows[:12]
    finally:
        doc.close()


def run_building_site_placement_audit(pdf_path: str, output_root: str | Path,
                                      existing_registry: dict | None = None) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_root) / f"feeldx_building_site_placement_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    parsed, parsed_path, raw_path, report_path = analyze_building_position_sources(pdf_path, out)
    building_specs = _expected_building_specs(parsed)
    expected_buildings = _expected_from_sources(parsed)
    local_candidates = _local_candidate_pages(pdf_path, expected_buildings)
    candidate_payload = {
        "primary_building_position_source": parsed.get("primary_building_position_source"),
        "gemini_sources": parsed.get("building_position_sources", []),
        "local_candidates": local_candidates,
    }
    _write_json(out / "02_candidate_pages.json", candidate_payload)

    doc = fitz.open(pdf_path)
    try:
        source_pages = _source_pages(parsed, doc.page_count, expected_buildings)
        if not source_pages:
            for row in local_candidates[:6]:
                p0 = row["page"] - 1
                if p0 not in source_pages:
                    source_pages.append(p0)
        source_pages = [p for p in source_pages if 0 <= p < doc.page_count]

        all_polygons = []
        label_report = []
        image_outputs = {}
        scale_evidence = None
        for page_idx in source_pages:
            page = doc[page_idx]
            if scale_evidence is None:
                scale_evidence = _parse_page_scale(page)
            labels = _extract_text_labels(page, building_specs)
            drawing_polys = _drawing_polygons(page)
            page_polygons = []
            seen = set()
            for label in labels:
                bld = label["building"]
                if bld in seen:
                    continue
                seen.add(bld)
                poly, source, conf, reason = _best_polygon_for_label(label, drawing_polys, page)
                item = {
                    "page": page_idx + 1,
                    "building": bld,
                    "alias_matched": label.get("alias", label["text"]),
                    "label_text": label["text"],
                    "label_bbox": label["bbox"],
                    "polygon": poly,
                    "bbox": list(poly.bounds),
                    "centroid": [float(poly.centroid.x), float(poly.centroid.y)],
                    "source": source,
                    "confidence": round(conf, 3),
                    "evidence": reason,
                    "reject_reason": None if conf >= 0.70 else "low_confidence_polygon",
                }
                page_polygons.append(item)
                all_polygons.append(item)
                label_report.append({k: v for k, v in item.items() if k != "polygon"})
            img = _save_candidate_page_overlay(
                page,
                out / f"03_building_polygons_p{page_idx + 1:02d}.png",
                labels,
                page_polygons,
            )
            image_outputs[f"p{page_idx + 1:02d}_building_polygons"] = img
            # Alias for requested stage-2 candidate page output.
            image_outputs[f"p{page_idx + 1:02d}_candidate_page"] = img

    finally:
        doc.close()

    by_building = {}
    for item in sorted(all_polygons, key=lambda r: (-r["confidence"], r["page"])):
        by_building.setdefault(item["building"], item)

    building_polygons_json = {
        "buildings": {
            bld: {k: v for k, v in item.items() if k != "polygon"}
            for bld, item in sorted(by_building.items())
        },
        "all_candidates": [{k: v for k, v in item.items() if k != "polygon"} for item in all_polygons],
    }
    building_polygons_path = _write_json(out / "03_building_polygons.json", building_polygons_json)
    label_report_path = _write_json(out / "04_label_match_report.json", {"labels": label_report})

    scale_evidence = scale_evidence or {
        "scale_status": "not_verified",
        "scale_ratio": None,
        "source": None,
        "reason": "No primary source page was available for scale parsing.",
    }
    scale_ratio = scale_evidence.get("scale_ratio")
    all_bounds = None
    if by_building:
        minx = min(float(item["bbox"][0]) for item in by_building.values())
        miny = min(float(item["bbox"][1]) for item in by_building.values())
        maxx = max(float(item["bbox"][2]) for item in by_building.values())
        maxy = max(float(item["bbox"][3]) for item in by_building.values())
        all_bounds = [minx, miny, maxx, maxy]

    transforms = {}
    registry_buildings = (existing_registry or {}).get("buildings", {})
    for bld, item in by_building.items():
        target = item["centroid"]
        current = ((registry_buildings.get(bld) or {}).get("centroid_mm") or {})
        site_centroid_mm = None
        if scale_ratio and all_bounds:
            site_centroid_mm = [
                (target[0] - all_bounds[0]) * PT_TO_MM * float(scale_ratio),
                (target[1] - all_bounds[1]) * PT_TO_MM * float(scale_ratio),
            ]
        if current:
            model_centroid = [current.get("x_mm"), current.get("y_mm")]
            has_model_centroid = model_centroid[0] is not None and model_centroid[1] is not None
            dx = site_centroid_mm[0] - model_centroid[0] if site_centroid_mm and has_model_centroid else None
            dy = site_centroid_mm[1] - model_centroid[1] if site_centroid_mm and has_model_centroid else None
            transforms[bld] = {
                "site_centroid_source": "keyplan_page_scaled_mm",
                "source_page": item["page"],
                "source_centroid_pt": target,
                "site_centroid_mm_relative": site_centroid_mm,
                "model_centroid_mm": model_centroid,
                "dx_mm": dx,
                "dy_mm": dy,
                "status": "verified" if dx is not None and dy is not None else "not_verified",
                "reason": "Centroid translation from scaled primary site/keyplan page."
                if dx is not None and dy is not None else "Missing source scale or model centroid.",
            }
        else:
            transforms[bld] = {
                "site_centroid_source": "keyplan_page_scaled_mm",
                "source_page": item["page"],
                "source_centroid_pt": target,
                "site_centroid_mm_relative": site_centroid_mm,
                "model_centroid_mm": None,
                "dx_mm": None,
                "dy_mm": None,
                "status": "not_verified",
                "reason": "No building model centroid available from detected slab footprint.",
            }
    transform_payload = {
        "scale": scale_evidence,
        "source_bounds_pt": all_bounds,
        "unit_conversion": {
            "pdf_point_to_mm": PT_TO_MM,
            "formula": "(point - source_min_point) * 25.4 / 72 * scale_ratio",
        },
        "building_transforms": transforms,
    }
    transform_path = _write_json(out / "05_site_placement_transform.json", transform_payload)

    expected = expected_buildings or parsed.get("expected_buildings") or ["Building A", "Building B", "Building C", "Building D"]
    found = sorted(by_building)
    warnings = []
    status = "verified"
    if not all(b in by_building for b in expected):
        missing = [b for b in expected if b not in by_building]
        warnings.append(f"Missing building placement polygons for: {', '.join(missing)}")
        status = "not_verified"
    if any(item["confidence"] < 0.70 for item in by_building.values()):
        warnings.append("One or more building placement polygons are label-only or low confidence.")
        status = "not_verified"
    if scale_evidence.get("scale_status") != "verified":
        warnings.append(scale_evidence.get("reason") or "Keyplan/site placement scale is not verified.")
        status = "not_verified"
    transform_missing = [
        bld for bld in expected
        if (transforms.get(bld) or {}).get("status") != "verified"
    ]
    if transform_missing:
        warnings.append(f"Missing verified model transform for: {', '.join(transform_missing)}")
        status = "not_verified"

    names = sorted(by_building)
    for i, a in enumerate(names):
        pa = by_building[a]["polygon"]
        for b in names[i + 1:]:
            pb = by_building[b]["polygon"]
            inter = pa.intersection(pb).area
            smaller = min(pa.area, pb.area)
            if smaller > 0 and inter / smaller > 0.25:
                warnings.append(f"Building placement polygons overlap on source page: {a} / {b}")
                status = "not_verified"

    readiness = {
        "site_placement_status": status,
        "expected_buildings": expected,
        "found_buildings": found,
        "source_pages": [p + 1 for p in source_pages],
        "primary_source_page": (
            (parsed.get("primary_building_position_source") or {}).get("page")
        ),
        "primary_source_why": (
            (parsed.get("primary_building_position_source") or {}).get("why_chosen")
        ),
        "primary_source_comparison": (
            (parsed.get("primary_building_position_source") or {}).get("why_better_than_other_candidates")
        ),
        "recommended_pages": [
            src.get("page") for src in parsed.get("building_position_sources", [])
            if src.get("recommended_for_site_placement")
        ],
        "scale": scale_evidence,
        "transform_verified_count": sum(1 for t in transforms.values() if t.get("status") == "verified"),
        "warnings": warnings,
    }
    readiness_path = _write_json(out / "06_readiness_report.json", readiness)

    summary = {
        "audit_dir": str(out),
        "site_placement_status": status,
        "json_outputs": {
            "gemini_sources": parsed_path,
            "gemini_raw": raw_path,
            "gemini_parse_report": report_path,
            "candidate_pages": str(out / "02_candidate_pages.json"),
            "building_polygons": building_polygons_path,
            "label_match_report": label_report_path,
            "site_placement_transform": transform_path,
            "readiness_report": readiness_path,
        },
        "image_outputs": image_outputs,
        "readiness": readiness,
        "site_transform": transform_payload,
    }
    _write_json(out / "summary.json", summary)
    return summary
