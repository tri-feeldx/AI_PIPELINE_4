"""Gemini semantic judge for code-generated slab/opening candidates."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.slab_v2 import gemini_client


_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "main_slab_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "appendage_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "opening_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "exclude_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "thickness_mm": {"type": "NUMBER"},
        "thickness_source_text": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "reason": {"type": "STRING"},
    },
    "required": ["main_slab_ids", "opening_ids", "exclude_ids",
                 "confidence", "reason"],
}


def _public(candidate: dict) -> dict:
    return {k: v for k, v in candidate.items() if k != "polygon"}


def judge_candidates(page, candidates, slabs, cfg, renderer,
                     content_area_pt2: float) -> dict:
    if not candidates or renderer is None:
        return {"status": "skipped", "reason": "no candidates/renderer"}

    candidate_path = renderer.step09_candidates(
        candidates, "step_09c_opening_candidates.png")
    candidate_png = renderer.render_for_prompt(candidate_path)
    from src.vision_refiner import find_legend_rect, render_crop
    _legend_img, legend_png = render_crop(
        page, find_legend_rect(page), cfg.prompt_dpi)

    slab_rows = []
    for i, slab in enumerate(slabs, 1):
        poly = slab["polygon_pdf"]
        slab_rows.append({
            "id": f"slab_{i:02d}", "area_pt2": round(poly.area, 1),
            "bbox": [round(x, 1) for x in poly.bounds],
            "label": slab.get("label", "SLAB"),
        })
    rows = [_public(c) for c in candidates]
    title = " ".join(
        w[4] for w in page.get_text("words")
        if w[0] > page.rect.width * 0.75 or w[1] > page.rect.height * 0.84
    )[:1800]
    prompt = f"""You are the semantic JUDGE for one structural slab plan.

Code already generated every polygon. You MUST return only candidate IDs;
never invent geometry or coordinates.

IMAGE 1 labels every opening candidate by ID.
IMAGE 2 is the drawing legend.

Decide which CUT-ELIGIBLE candidates are real slab openings. Object identity
and opening intent are independent. STAIR graphics, blue fill, tread lines,
STAIR labels and stair X graphics are context only and NEVER authorize a slab
cut. A stair may coexist with a real penetration, but that candidate is
eligible only when code supplies an independent opening_intent of
SLAB_PENETRATION, VOID, or LIFT_SHAFT plus opening_evidence_ids. Never promote
a candidate whose cut_eligible field is false. Candidates already verified as
penetration/void/lift-shaft should remain openings unless the image shows a
direct structural conflict. A floor box (FB), setdown, step or shallow rebate
is not a full-depth opening. Do not select wall or column footprints.

SLABS:
{json.dumps(slab_rows, ensure_ascii=False)}

OPENING CANDIDATES:
{json.dumps(rows, ensure_ascii=False)}

TITLE/LEGEND TEXT CONTEXT:
{title}

Return main_slab_ids from the supplied slab IDs, opening_ids, exclude_ids,
confidence 0..1, and a concise evidence-based reason. thickness_mm is allowed
only when thickness_source_text quotes visible title/legend evidence.
"""
    out_dir = Path(renderer.out_dir)
    (out_dir / "step_09d_llm_judge_prompt.txt").write_text(
        prompt, encoding="utf-8")
    data = gemini_client.call_gemini_json(
        prompt, [candidate_png, legend_png], _SCHEMA, cfg.gemini_model,
        log_path=str(out_dir / "prompts.log"), tag="opening_candidate_judge")

    valid = {c["id"] for c in candidates}
    slab_ids = {s["id"] for s in slab_rows}
    opening_ids = [x for x in data.get("opening_ids", []) if x in valid]
    exclude_ids = [x for x in data.get("exclude_ids", []) if x in valid]
    main_slab_ids = [x for x in data.get("main_slab_ids", []) if x in slab_ids]
    confidence = float(data.get("confidence") or 0.0)

    # Hard guards: equipment/rebate candidates cannot become openings merely
    # because the model selected them; explicit geometry remains review-only.
    blocked = {c["id"] for c in candidates
               if (c.get("default_action") == "exclude"
                   or not c.get("cut_eligible", False))}
    opening_ids = [x for x in opening_ids if x not in blocked]
    exclude_ids = sorted(set(exclude_ids) | blocked)
    status = "accepted" if (
        confidence >= cfg.opening_judge_min_confidence and opening_ids
    ) else "rejected"

    result = {
        **data,
        "status": status,
        "main_slab_ids": main_slab_ids,
        "opening_ids": opening_ids,
        "exclude_ids": exclude_ids,
        "confidence": confidence,
    }
    source = str(result.get("thickness_source_text") or "")
    if result.get("thickness_mm") and not re.search(r"\b\d{2,4}\b", source):
        result["thickness_mm"] = None
        result["thickness_source_text"] = ""
    (out_dir / "step_09d_llm_judge.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    renderer.step09_judgement(
        candidates, result, "step_09d_llm_judge.png")
    return result
