"""
Interior slab resolver for no-fill structural drawings.

Line semantics can identify which line styles are valid enclosure evidence, but
the final no-fill slab still needs an "inside" decision. This module scores
candidate faces using structural inside seeds and outside masks, then returns
reviewable selected/rejected geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import fitz
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union


@dataclass
class InteriorSeed:
    kind: str
    polygon: Polygon
    label: str = ""
    confidence: float = 0.0


@dataclass
class OutsideMask:
    kind: str
    polygon: Polygon
    label: str = ""
    confidence: float = 0.0


@dataclass
class CandidateDecision:
    polygon: Polygon
    score: float
    selected: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class InteriorSlabResolution:
    selected_inside_slabs: list[Polygon] = field(default_factory=list)
    rejected_candidates: list[CandidateDecision] = field(default_factory=list)
    inside_seeds: list[InteriorSeed] = field(default_factory=list)
    outside_masks: list[OutsideMask] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


_INSIDE_TEXT_RE = re.compile(
    r"\b(PC\d+|PF\d+|C-?CC\d+|C\d{1,3}|S\.?\s*O\.?\s*G\.?|SOG|SLAB|R\.?\s*C\.?\s+WALL|WALL)\b",
    re.I,
)
_OUTSIDE_TEXT_RE = re.compile(
    r"\b(LEGEND|NOTES?|SCHEDULE|TITLE|REVISION|KEYPLAN|SCALE|DRAWING|PROJECT|BUILDER|ARCHITECT|REFER TO)\b",
    re.I,
)


def _bbox_poly(bbox) -> Polygon:
    return box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _page_drawing_area(page: fitz.Page) -> Polygon:
    w, h = page.rect.width, page.rect.height
    return box(0, 0, w * 0.84, h * 0.90)


def _valid_polygon_parts(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom.buffer(0)]
    if isinstance(geom, MultiPolygon):
        return [g.buffer(0) for g in geom.geoms if not g.is_empty and g.area > 1]
    return [
        g.buffer(0) for g in getattr(geom, "geoms", [])
        if isinstance(g, Polygon) and not g.is_empty and g.area > 1
    ]


def _text_lines(text_blocks: list[dict]) -> list[dict]:
    out = []
    for block in text_blocks or []:
        text = str(block.get("text", "")).strip()
        bbox = block.get("bbox")
        if text and bbox:
            out.append({"text": text, "polygon": _bbox_poly(bbox)})
    return out


def _seed_envelope(page: fitz.Page, seeds: list[InteriorSeed]) -> Polygon | None:
    seed_polys = [s.polygon for s in seeds if s.polygon is not None and not s.polygon.is_empty]
    if not seed_polys:
        return None
    bounds = unary_union(seed_polys).bounds
    pad_x = page.rect.width * 0.055
    pad_y = page.rect.height * 0.055
    env = box(bounds[0] - pad_x, bounds[1] - pad_y, bounds[2] + pad_x, bounds[3] + pad_y)
    return env.intersection(_page_drawing_area(page).buffer(page.rect.width * 0.03))


def _collect_inside_seeds(page: fitz.Page, text_blocks: list[dict], structural_objects) -> list[InteriorSeed]:
    seeds: list[InteriorSeed] = []
    for attr, kind, conf in [
        ("columns_or_piles", "column_or_pile", 0.82),
        ("footings", "footing", 0.78),
        ("walls", "wall", 0.68),
        ("load_bearing_elements", "load_bearing", 0.72),
        ("cores", "core", 0.70),
    ]:
        for obj in getattr(structural_objects, attr, []) or []:
            poly = getattr(obj, "polygon", None)
            if poly is not None and not poly.is_empty:
                seeds.append(InteriorSeed(kind, poly, getattr(obj, "label", ""), conf))

    drawing_area = _page_drawing_area(page)
    for item in _text_lines(text_blocks):
        text = item["text"]
        if not _INSIDE_TEXT_RE.search(text):
            continue
        poly = item["polygon"].buffer(8)
        if drawing_area.buffer(3).intersects(poly.centroid):
            seeds.append(InteriorSeed("inside_text", poly, text[:40], 0.66))
    return seeds


def _collect_outside_masks(page: fitz.Page, text_blocks: list[dict], structural_objects) -> list[OutsideMask]:
    masks: list[OutsideMask] = []
    w, h = page.rect.width, page.rect.height
    masks.extend([
        OutsideMask("right_sheet_zone", box(w * 0.84, 0, w, h), "legend/title side", 0.80),
        OutsideMask("bottom_title_zone", box(0, h * 0.90, w, h), "title block side", 0.68),
    ])
    for obj in getattr(structural_objects, "ignored_regions", []) or []:
        poly = getattr(obj, "polygon", None)
        if poly is not None and not poly.is_empty:
            masks.append(OutsideMask("ignored_region", poly.buffer(8), getattr(obj, "label", ""), 0.85))
    for item in _text_lines(text_blocks):
        text = item["text"]
        if not _OUTSIDE_TEXT_RE.search(text):
            continue
        poly = item["polygon"].buffer(16)
        if len(text) > 60 or poly.centroid.x >= w * 0.72 or poly.centroid.y >= h * 0.84:
            masks.append(OutsideMask("outside_text", poly, text[:50], 0.74))
    return masks


def _outside_penalty(poly: Polygon, masks: list[OutsideMask]) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons = []
    area = max(poly.area, 1.0)
    for mask in masks:
        try:
            inter_area = poly.intersection(mask.polygon).area
        except Exception:
            inter_area = 0.0
        frac = inter_area / area
        if frac > 0.002:
            p = min(0.42, frac * 3.0) * max(mask.confidence, 0.35)
            penalty += p
            reasons.append(f"outside_{mask.kind}={frac:.3f}")
    return penalty, reasons


def _inside_score(poly: Polygon, seeds: list[InteriorSeed]) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    contains = 0
    near = 0
    for seed in seeds:
        if seed.polygon is None or seed.polygon.is_empty:
            continue
        if poly.buffer(2).intersects(seed.polygon.centroid):
            contains += 1
            score += 0.035 * max(seed.confidence, 0.35)
        elif poly.buffer(15).intersects(seed.polygon):
            near += 1
            score += 0.010 * max(seed.confidence, 0.35)
    if contains:
        reasons.append(f"inside_seeds={contains}")
    if near:
        reasons.append(f"near_seeds={near}")
    return min(score, 0.38), reasons


def _clip_to_inside_envelope(poly: Polygon, envelope: Polygon | None, score: float) -> tuple[Polygon, bool]:
    if envelope is None or poly.is_empty:
        return poly, False
    try:
        clipped = poly.intersection(envelope)
    except Exception:
        return poly, False
    if clipped.is_empty or clipped.area < poly.area * 0.20:
        return poly, False
    # Only trim likely spillover regions; otherwise leave high-confidence faces intact.
    if clipped.area < poly.area * 0.92 and score < 0.86:
        parts = _valid_polygon_parts(clipped)
        if parts:
            return max(parts, key=lambda g: g.area), True
    return poly, False


def resolve_interior_slabs(
    page: fitz.Page,
    candidates: list[tuple[float, Polygon, list[str]]] | list[Polygon],
    text_blocks: list[dict] | None = None,
    structural_objects=None,
    min_confidence: float = 0.55,
) -> InteriorSlabResolution:
    """Choose the inside slab faces from no-fill polygon candidates."""
    structural_objects = structural_objects or object()
    text_blocks = text_blocks or []
    seeds = _collect_inside_seeds(page, text_blocks, structural_objects)
    masks = _collect_outside_masks(page, text_blocks, structural_objects)
    envelope = _seed_envelope(page, seeds)
    drawing_area = _page_drawing_area(page)

    normalized: list[tuple[float, Polygon, list[str]]] = []
    for item in candidates or []:
        if isinstance(item, tuple) and len(item) >= 2:
            base_score, poly = float(item[0]), item[1]
            reasons = list(item[2]) if len(item) >= 3 else []
        else:
            base_score, poly, reasons = 0.45, item, []
        if poly is not None and not poly.is_empty:
            normalized.append((base_score, poly, reasons))

    decisions: list[CandidateDecision] = []
    warnings: list[str] = []
    for base_score, poly, reasons in normalized:
        if not drawing_area.buffer(page.rect.width * 0.04).intersects(poly.centroid):
            decisions.append(CandidateDecision(poly, 0.0, False, reasons + ["outside_drawing_area"]))
            continue
        score = min(base_score, 0.62)
        seed_score, seed_reasons = _inside_score(poly, seeds)
        outside_penalty, outside_reasons = _outside_penalty(poly, masks)
        score = max(0.0, min(0.98, score + seed_score - outside_penalty))
        out_poly, clipped = _clip_to_inside_envelope(poly, envelope, score)
        out_reasons = reasons + seed_reasons + outside_reasons
        if clipped:
            out_reasons.append("clipped_to_inside_seed_envelope")
            score = max(score, min(0.78, score + 0.05))
        decisions.append(CandidateDecision(out_poly, score, False, out_reasons))

    if not decisions:
        return InteriorSlabResolution(
            inside_seeds=seeds,
            outside_masks=masks,
            confidence=0.0,
            warnings=["No candidate faces for interior resolver."],
            debug={"candidate_count": 0, "inside_seed_count": len(seeds), "outside_mask_count": len(masks)},
        )

    decisions.sort(key=lambda d: (d.score, d.polygon.area), reverse=True)
    top = decisions[0]
    selected: list[Polygon] = []
    for decision in decisions:
        keep = decision.score >= max(min_confidence, top.score - 0.16)
        keep = keep and decision.polygon.area >= max(top.polygon.area * 0.05, page.rect.width * page.rect.height * 0.003)
        decision.selected = keep
        if keep:
            selected.append(decision.polygon)
    if top.score < min_confidence:
        warnings.append("Interior confidence below threshold; review before trusting no-fill slab.")
        selected = []
        for decision in decisions:
            decision.selected = False
    if not seeds:
        warnings.append("No inside structural seeds found; resolver relied on boundary geometry only.")

    try:
        selected_union = unary_union(selected) if selected else None
        selected_polys = _valid_polygon_parts(selected_union)
    except Exception:
        selected_polys = selected

    return InteriorSlabResolution(
        selected_inside_slabs=selected_polys,
        rejected_candidates=[d for d in decisions if not d.selected],
        inside_seeds=seeds,
        outside_masks=masks,
        confidence=top.score,
        warnings=warnings,
        debug={
            "candidate_count": len(normalized),
            "selected_count": len(selected_polys),
            "rejected_count": len([d for d in decisions if not d.selected]),
            "inside_seed_count": len(seeds),
            "outside_mask_count": len(masks),
            "top_score": top.score,
            "top_reasons": top.reasons,
        },
    )
