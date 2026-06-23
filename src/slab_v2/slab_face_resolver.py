"""Evidence-driven semantic resolver for code-generated slab faces.

The model may classify candidate IDs, but only deterministic geometry and
hard evidence are allowed to remove material from the gross slab.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from shapely.geometry import Point
from shapely.ops import unary_union

from src.slab_v2.models import SlabFaceCandidate, SlabResolution
from src.slab_v2 import gemini_client


_POSITIVE_RE = re.compile(
    r"\b(SLAB|S\.O\.G|FLOOR STRUCTURE|POST[ -]?TENSION|THICKNESS)\b", re.I)
_NEGATIVE_RE = re.compile(
    r"\b(NO SLAB|VOID|OPENING|PENETRATION|STAIRWELL|LIFT SHAFT|CORE SHAFT)\b",
    re.I)
_EQUIPMENT_RE = re.compile(r"\b(FB|FLOOR BOX|REBATE|SETDOWN|PLINTH)\b", re.I)

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "selected_slab_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "appendage_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "opening_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "non_slab_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "review_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence": {"type": "NUMBER"},
        "reason": {"type": "STRING"},
    },
    "required": ["selected_slab_ids", "appendage_ids", "opening_ids",
                 "non_slab_ids", "review_ids", "confidence", "reason"],
}


def _words_for_polygon(words: list, poly, max_words: int = 16) -> list[str]:
    hits = []
    expanded = poly.buffer(4.0)
    for w in words:
        p = Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
        if expanded.contains(p):
            hits.append(str(w[4]))
            if len(hits) >= max_words:
                break
    return hits


def build_candidate_registry(page, face_graph, gross_geometry, content_rect,
                             columns=None, walls=None, openings=None) -> list:
    """Build auditable evidence records for every useful atomic face."""
    words = page.get_text("words")
    columns = columns or []
    walls = walls or []
    openings = openings or []
    gross_area = max(gross_geometry.area, 1.0)
    page_area = max(page.rect.width * page.rect.height, 1.0)
    candidates = []

    for face in face_graph.faces:
        poly = face.polygon
        if poly.is_empty or poly.area < page_area * 0.00005:
            continue
        # Faces unrelated to the assembled slab remain useful only when they
        # are large enough to explain title/legend/schedule false positives.
        touches_gross = poly.buffer(0.5).intersects(gross_geometry)
        if not touches_gross and poly.area < page_area * 0.002:
            continue

        texts = _words_for_polygon(words, poly)
        joined = " ".join(texts)
        positive, negative = [], []
        if face.source == "fill":
            positive.append("filled_closed_region")
        if _POSITIVE_RE.search(joined):
            positive.append("explicit_slab_text")
        if touches_gross:
            positive.append("intersects_gross_slab")
        ncols = sum(1 for c in columns
                    if poly.buffer(1.0).contains(c.polygon.representative_point()))
        nwalls = sum(1 for w in walls
                     if poly.buffer(1.0).intersects(w.polygon))
        if ncols:
            positive.append("contains_structural_columns")
        if nwalls:
            positive.append("supported_by_walls")

        if _NEGATIVE_RE.search(joined):
            negative.append("explicit_negative_text")
        if _EQUIPMENT_RE.search(joined):
            negative.append("equipment_or_rebate_text")
        b = poly.bounds
        if (b[0] < content_rect.x0 - 1 or b[1] < content_rect.y0 - 1
                or b[2] > content_rect.x1 + 1 or b[3] > content_rect.y1 + 1):
            negative.append("outside_drawing_content")
        if (b[0] > page.rect.width * 0.74
                or b[1] > page.rect.height * 0.86):
            negative.append("title_legend_schedule_zone")

        opening_hits = []
        for i, opening in enumerate(openings):
            if poly.intersection(opening.polygon).area > 0:
                opening_hits.append(f"opening_{i + 1:02d}")
        if opening_hits:
            negative.append("confirmed_opening_overlap")

        score = (1.5 * len(positive) - 1.5 * len(negative)
                 + min(poly.intersection(gross_geometry).area / gross_area, 0.5))
        candidates.append(SlabFaceCandidate(
            id=f"face_{face.id:04d}", polygon=poly,
            area_pt2=poly.area, source=face.source,
            fill_style={"protected": face.source == "fill"},
            boundary_style_ids=sorted(face.style_ids),
            parent_id=(f"face_{face.parent_id:04d}"
                       if face.parent_id is not None else None),
            depth=face.depth, nearby_text=texts,
            contained_columns=ncols, contained_walls=nwalls,
            intersects_openings=opening_hits,
            positive_evidence=positive, negative_evidence=negative,
            deterministic_score=round(score, 3),
        ))
    return candidates


def _public(c: SlabFaceCandidate) -> dict:
    return {
        "id": c.id, "area_pt2": round(c.area_pt2, 1),
        "bbox": [round(x, 1) for x in c.polygon.bounds],
        "source": c.source, "fill_style": c.fill_style,
        "boundary_style_ids": c.boundary_style_ids,
        "parent_id": c.parent_id, "depth": c.depth,
        "nearby_text": c.nearby_text,
        "contained_columns": c.contained_columns,
        "contained_walls": c.contained_walls,
        "intersects_openings": c.intersects_openings,
        "positive_evidence": c.positive_evidence,
        "negative_evidence": c.negative_evidence,
        "deterministic_score": c.deterministic_score,
    }


def _has_explicit_negative(c: SlabFaceCandidate) -> bool:
    return "explicit_negative_text" in c.negative_evidence


def _protected_fill(c: SlabFaceCandidate) -> bool:
    return bool(c.fill_style.get("protected")) and not _has_explicit_negative(c)


def _judge(page, candidates, cfg, renderer) -> dict:
    out_dir = Path(renderer.out_dir)
    overlay = renderer.step08_slab_candidates(
        candidates, "step_08a_slab_face_candidates.png")
    image = renderer.render_for_prompt(overlay)
    from src.vision_refiner import find_legend_rect, render_crop
    _img, legend = render_crop(page, find_legend_rect(page), cfg.prompt_dpi)
    rows = [_public(c) for c in candidates]
    text = page.get_text("text")[:5000]
    prompt = f"""You are the semantic JUDGE for one structural slab plan.

Code already generated every face. Return IDs only; never invent coordinates.
Classify candidates into selected structural slab, appendage, full-depth
opening, non-slab, or review. A dashed line alone is not a slab boundary.
A filled floor-structure face is protected and must stay slab unless explicit
visible text says NO SLAB, VOID, OPENING, or PENETRATION. FB/FLOOR BOX,
SETDOWN and REBATE are not full-depth openings. Title, legend, schedule,
revision and keyplan regions are non-slab. Be conservative: uncertain faces
go to review, not non_slab.

CANDIDATES:
{json.dumps(rows, ensure_ascii=False)}

PAGE TEXT:
{text}
"""
    (out_dir / "step_08b_slab_judge_prompt.txt").write_text(
        prompt, encoding="utf-8")
    data = gemini_client.call_gemini_json(
        prompt, [image, legend], _SCHEMA, cfg.gemini_model,
        log_path=str(out_dir / "prompts.log"), tag="slab_face_judge",
        raw_path=str(out_dir / "step_08b_slab_judge_raw.txt"))
    (out_dir / "step_08b_slab_judge.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def resolve_slab_faces(page, face_graph, gross_slabs, opening_candidates,
                       content_rect, cfg, renderer, columns=None, walls=None,
                       resolved_openings=None, use_ai=True) -> SlabResolution:
    gross = unary_union([s["polygon_pdf"] for s in gross_slabs])
    openings = resolved_openings or []
    candidates = build_candidate_registry(
        page, face_graph, gross, content_rect, columns, walls, openings)
    by_id = {c.id: c for c in candidates}
    warnings = []
    decision = None
    if use_ai and cfg.enable_slab_face_judge and candidates:
        try:
            decision = _judge(page, candidates, cfg, renderer)
        except Exception as exc:
            warnings.append(f"slab face judge failed: {exc}")

    if not decision:
        resolution = SlabResolution(
            gross_geometry=gross, net_geometry=gross,
            review_ids=[c.id for c in candidates],
            status="deterministic_fallback", confidence=0.0,
            reason="No valid semantic decision; gross slab preserved.",
            warnings=warnings)
        return resolution, candidates

    valid = set(by_id)
    lists = {}
    for key in ("selected_slab_ids", "appendage_ids", "opening_ids",
                "non_slab_ids", "review_ids"):
        lists[key] = list(dict.fromkeys(
            x for x in decision.get(key, []) if x in valid))
    confidence = max(0.0, min(1.0, float(decision.get("confidence") or 0)))
    if confidence < cfg.slab_judge_min_confidence:
        warnings.append(
            f"slab judge confidence {confidence:.2f} below "
            f"{cfg.slab_judge_min_confidence:.2f}; gross slab preserved")
        resolution = SlabResolution(
            **lists, gross_geometry=gross, net_geometry=gross,
            confidence=confidence, status="deterministic_fallback",
            reason=str(decision.get("reason") or ""), warnings=warnings)
        return resolution, candidates

    removable = []
    blocked = []
    for cid in lists["non_slab_ids"]:
        c = by_id[cid]
        explicit = _has_explicit_negative(c)
        enough_negative = len(set(c.negative_evidence)) >= 2
        if _protected_fill(c):
            blocked.append(cid)
            continue
        if confidence < cfg.slab_subtract_min_confidence:
            blocked.append(cid)
            continue
        if not (explicit or enough_negative):
            blocked.append(cid)
            continue
        clipped = c.polygon.intersection(gross)
        if not clipped.is_empty and clipped.area > 0:
            removable.append(clipped)

    # Openings do not alter gross slab. Existing stair/core resolver geometry
    # is strong corroboration; otherwise the same conservative evidence gate
    # applies before a face may enter the net-slab opening set.
    accepted_opening_ids = []
    for cid in lists["opening_ids"]:
        c = by_id[cid]
        overlap = max((c.polygon.intersection(e.polygon).area
                       / max(c.polygon.area, 1.0) for e in openings),
                      default=0.0)
        explicit = _has_explicit_negative(c)
        enough_negative = len(set(c.negative_evidence)) >= 2
        if overlap >= 0.50:
            accepted_opening_ids.append(cid)
        elif (confidence >= cfg.slab_subtract_min_confidence
              and not _protected_fill(c)
              and (explicit or enough_negative)):
            accepted_opening_ids.append(cid)
        else:
            blocked.append(cid)
    lists["opening_ids"] = accepted_opening_ids

    if blocked:
        lists["review_ids"] = sorted(set(lists["review_ids"]) | set(blocked))
        lists["non_slab_ids"] = [x for x in lists["non_slab_ids"]
                                       if x not in blocked]
        lists["opening_ids"] = [x for x in lists["opening_ids"]
                                if x not in blocked]
        warnings.append("destructive decisions blocked by fill/evidence guard: "
                        + ", ".join(blocked))

    subtract = unary_union(removable) if removable else None
    resolved_gross = gross.difference(subtract) if subtract else gross
    loss = 1.0 - resolved_gross.area / max(gross.area, 1.0)
    all_explicit = all(_has_explicit_negative(by_id[cid])
                       for cid in lists["non_slab_ids"])
    if loss > cfg.slab_max_net_area_loss_frac and not all_explicit:
        warnings.append(f"judge removal {loss:.0%} exceeds safety limit; "
                        "gross slab preserved")
        resolved_gross = gross
        lists["review_ids"] = sorted(set(lists["review_ids"])
                                     | set(lists["non_slab_ids"]))
        lists["non_slab_ids"] = []

    opening_geoms = [e.polygon for e in openings
                     if e.polygon.intersects(resolved_gross)]
    opening_geoms += [by_id[cid].polygon for cid in lists["opening_ids"]]
    net = resolved_gross.difference(unary_union(opening_geoms)) \
        if opening_geoms else resolved_gross
    status = "verified" if not lists["review_ids"] else "review"
    resolution = SlabResolution(
        **lists, gross_geometry=resolved_gross, net_geometry=net,
        confidence=confidence, status=status,
        reason=str(decision.get("reason") or ""), warnings=warnings)
    return resolution, candidates


def candidate_payload(candidates: list[SlabFaceCandidate]) -> list[dict]:
    return [_public(c) for c in candidates]
