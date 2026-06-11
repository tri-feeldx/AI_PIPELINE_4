"""
Legend-guided slab semantic preview.

This module does not replace final slab extraction. It classifies visible slab
surface candidates, boundary cues, and net-slab cut candidates for review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import fitz
from shapely.geometry import Polygon, box

from src.boundary_slab_extractor import extract_boundary_first_slabs
from src.slab_extractor import (
    build_polygons_from_drawings,
    filter_slab_candidates_structured,
    reconstruct_closed_polygons,
)


@dataclass
class SlabSemanticObject:
    kind: str
    polygon: Polygon
    label: str = ""
    confidence: float = 0.0
    source: str = "semantic"
    auto_cut: bool = False


@dataclass
class SlabSemanticPreview:
    surface_regions: list[SlabSemanticObject] = field(default_factory=list)
    boundary_cues: list[SlabSemanticObject] = field(default_factory=list)
    cut_candidates: list[SlabSemanticObject] = field(default_factory=list)
    fallback_policy: str = "review_manually"
    gemini_fallback_policy: str = "review_manually"
    effective_surface_source: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _rules(legend_semantics: dict | None) -> dict:
    return (legend_semantics or {}).get("rules_for_code", {}) or {}


def _keywords(rules: dict, *keys: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        for value in rules.get(key, []) or []:
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            if cleaned and cleaned.upper() not in seen:
                seen.add(cleaned.upper())
                out.append(cleaned)
    return out


def _hit(text: str, keywords: list[str]) -> str | None:
    upper = (text or "").upper()
    for kw in keywords:
        if kw.upper() in upper:
            return kw
    return None


def _text_blocks(page: fitz.Page) -> list[dict]:
    blocks = []
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            text = " ".join(s.get("text", "").strip() for s in spans)
            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)
            blocks.append({"text": text, "bbox": [x0, y0, x1, y1]})
    return blocks


def _bbox_poly(bbox) -> Polygon:
    return box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _drawing_area(page: fitz.Page) -> Polygon:
    w, h = page.rect.width, page.rect.height
    return box(0, 0, w * 0.84, h * 0.90)


def _is_visible_fill(color) -> bool:
    if color is None or len(color) < 3:
        return False
    r, g, b = [float(v) for v in color[:3]]
    brightness = (r + g + b) / 3.0
    chroma = max(r, g, b) - min(r, g, b)
    if brightness >= 0.985 and chroma < 0.02:
        return False
    return 0.08 <= brightness <= 0.985 and chroma >= 0.035


def _visible_fill_surface(page: fitz.Page, drawings: list[dict], text_blocks: list[dict]) -> list[Polygon]:
    filled_pairs = build_polygons_from_drawings(drawings)
    visible_pairs = [(p, c) for p, c in filled_pairs if _is_visible_fill(c)]
    if not visible_pairs:
        return []
    result = filter_slab_candidates_structured(
        visible_pairs,
        page,
        text_blocks=text_blocks,
        recover_slab_appendages=True,
        auto_cut_voids=False,
        cut_walls=False,
    )
    return result.gross_slabs or result.net_slabs


def detect_slab_semantics_on_page(
    page: fitz.Page,
    drawings: list[dict],
    legend_semantics: dict | None = None,
    line_semantics: dict | None = None,
    text_blocks: list[dict] | None = None,
) -> SlabSemanticPreview:
    rules = _rules(legend_semantics)
    fallback_policy = rules.get("fallback_policy") or "review_manually"
    surface_keywords = _keywords(rules, "slab_surface_keywords", "slab_fill_keywords")
    boundary_keywords = _keywords(rules, "slab_boundary_keywords")
    cut_keywords = _keywords(rules, "net_slab_cut_keywords")
    blocks = text_blocks if text_blocks is not None else _text_blocks(page)
    drawing_area = _drawing_area(page)
    preview = SlabSemanticPreview(
        fallback_policy=fallback_policy,
        gemini_fallback_policy=fallback_policy,
    )

    # Surface candidates: use fill-based gross regions if present; otherwise boundary-first fallback.
    surface_polys = _visible_fill_surface(page, drawings, blocks)
    surface_source = "semantic_fill_or_material"
    surface_conf = 0.78 if surface_keywords else 0.62
    boundary_debug: dict[str, Any] = {}
    boundary_confidence = 0.0
    if not surface_polys:
        boundary = extract_boundary_first_slabs(
            page,
            drawings,
            text_blocks=blocks,
            legend_semantics=legend_semantics,
            line_semantics=line_semantics,
        )
        boundary_debug = boundary.debug or {}
        boundary_confidence = getattr(boundary, "confidence", 0.0)
        surface_polys = boundary.gross_regions or boundary.final_regions
        surface_source = "evidence_guided_no_fill_boundary"
        surface_conf = 0.55
        if surface_polys:
            preview.fallback_policy = "use_evidence_guided_no_fill_boundary"
            preview.warnings.append(
                "No explicit slab fill/material rule; using evidence-guided no-fill boundary candidate for review."
            )
        else:
            preview.fallback_policy = "review_manually"
            preview.warnings.append(
                "No explicit slab fill/material rule and no reliable boundary evidence; review manually."
            )
    preview.effective_surface_source = surface_source

    for poly in surface_polys:
        if poly is None or poly.is_empty:
            continue
        if not drawing_area.buffer(5).intersects(poly.centroid):
            continue
        preview.surface_regions.append(SlabSemanticObject(
            "slab_surface",
            poly,
            label=surface_source,
            confidence=surface_conf,
            source=surface_source,
        ))

    for block in blocks:
        text = block.get("text", "")
        bbox = block.get("bbox")
        if not bbox:
            continue
        poly = _bbox_poly(bbox).buffer(10)
        if not drawing_area.buffer(5).intersects(poly.centroid):
            continue
        cut_hit = _hit(text, cut_keywords)
        boundary_hit = _hit(text, boundary_keywords)
        surface_hit = _hit(text, surface_keywords)
        if cut_hit:
            preview.cut_candidates.append(SlabSemanticObject(
                "slab_cut",
                poly,
                label=text,
                confidence=0.86,
                source=f"legend_semantic:{cut_hit}",
                auto_cut=True,
            ))
        elif boundary_hit:
            preview.boundary_cues.append(SlabSemanticObject(
                "slab_boundary_cue",
                poly,
                label=text,
                confidence=0.74,
                source=f"legend_semantic:{boundary_hit}",
                auto_cut=False,
            ))
        elif surface_hit:
            preview.boundary_cues.append(SlabSemanticObject(
                "slab_surface_label",
                poly,
                label=text,
                confidence=0.68,
                source=f"legend_semantic:{surface_hit}",
                auto_cut=False,
            ))

    preview.debug = {
        "surface_regions": len(preview.surface_regions),
        "boundary_cues": len(preview.boundary_cues),
        "cut_candidates": len(preview.cut_candidates),
        "gemini_fallback_policy": preview.gemini_fallback_policy,
        "fallback_policy": preview.fallback_policy,
        "effective_surface_source": preview.effective_surface_source,
        "boundary_evidence_count": boundary_debug.get("boundary_evidence_count", 0),
        "boundary_signature_count": boundary_debug.get("boundary_signature_count", 0),
        "line_semantic_rule_count": boundary_debug.get("line_semantic_rule_count", 0),
        "boundary_envelope_confidence": boundary_confidence,
        "excluded_non_boundary_count": boundary_debug.get("excluded_non_boundary_count", 0),
        "wall_count": boundary_debug.get("walls", 0),
        "load_bearing_count": boundary_debug.get("load_bearing_elements", 0),
        "column_or_pile_count": boundary_debug.get("columns_or_piles", 0),
        "footing_count": boundary_debug.get("footings", 0),
        "surface_keywords": surface_keywords,
        "boundary_keywords": boundary_keywords,
        "cut_keywords": cut_keywords,
        "warnings": preview.warnings,
    }
    return preview


def detect_slab_semantics_for_pages(
    pdf_path: str,
    page_indices: list[int],
    legend_semantics: dict | None = None,
    line_semantics: dict | None = None,
) -> dict[int, SlabSemanticPreview]:
    previews: dict[int, SlabSemanticPreview] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_index in page_indices:
            if page_index < 0 or page_index >= doc.page_count:
                continue
            page = doc[page_index]
            previews[page_index] = detect_slab_semantics_on_page(
                page,
                page.get_drawings(),
                legend_semantics=legend_semantics,
                line_semantics=line_semantics,
            )
    finally:
        doc.close()
    return previews
