"""Evidence-driven floor-system partitioning.

The PDF code creates every polygon and separator. Gemini may classify stable
candidate IDs, but it cannot create coordinates or override geometry guards.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path

from shapely.geometry import LineString, Point
from shapely.ops import split, unary_union

from src.slab_v2 import gemini_client
from src.slab_v2.models import (
    FloorSystemCandidate,
    FloorSystemResolution,
    FloorSystemSemanticProfile,
)


_CONCRETE_RE = re.compile(
    r"\b(POST[ -]?TENSION(?:ED)?\s+SLAB|PT\s+SLAB|CONCRETE\s+SLAB|"
    r"SLAB\s+ON\s+GRADE|S\.?O\.?G\.?)\b", re.I)
_FLOOR_EXTENT_RE = re.compile(r"\b(FLOOR\s+STRUCTURE|FLOOR\s+EXTENT)\b", re.I)
_STEEL_FLOOR_RE = re.compile(
    r"\b(STEELWORK|STEEL\s+FLOOR|COMPOSITE\s+DECK|METAL\s+DECK|"
    r"BONDEK|KINGFLOR)\b", re.I)
_STEEL_SYMBOL_RE = re.compile(r"^(?:SH|UB|UC|CH|SHS|RHS|CHS)\w*$", re.I)
_CONCRETE_SYMBOL_RE = re.compile(r"^(?:C\d+|RC\d+|LW\d+)$", re.I)


_PROFILE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "concrete_slab_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
        "floor_extent_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
        "steel_floor_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
        "opening_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
        "fill_rules": {"type": "ARRAY", "items": {"type": "OBJECT"}},
        "boundary_rules": {"type": "ARRAY", "items": {"type": "OBJECT"}},
        "symbol_families": {"type": "ARRAY", "items": {"type": "OBJECT"}},
        "confidence": {"type": "NUMBER"},
        "warnings": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["concrete_slab_terms", "floor_extent_terms",
                 "steel_floor_terms", "opening_terms", "fill_rules",
                 "boundary_rules", "symbol_families", "confidence", "warnings"],
}

_JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "pt_concrete_slab_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "other_floor_system_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "opening_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "non_floor_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "unknown_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence_by_id": {"type": "OBJECT"},
        "reason_by_id": {"type": "OBJECT"},
    },
    "required": ["pt_concrete_slab_ids", "other_floor_system_ids",
                 "opening_ids", "non_floor_ids", "unknown_ids",
                 "confidence_by_id", "reason_by_id"],
}


def _unique_matches(pattern, text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0).strip() for m in pattern.finditer(text)))


def _deterministic_profile(page) -> FloorSystemSemanticProfile:
    text = page.get_text("text")
    concrete = _unique_matches(_CONCRETE_RE, text)
    extent = _unique_matches(_FLOOR_EXTENT_RE, text)
    steel = _unique_matches(_STEEL_FLOOR_RE, text)
    warnings = []
    if extent and not concrete:
        warnings.append(
            "FLOOR STRUCTURE describes extent only; concrete material is unproven.")
    return FloorSystemSemanticProfile(
        concrete_slab_terms=concrete,
        floor_extent_terms=extent,
        steel_floor_terms=steel,
        opening_terms=_unique_matches(
            re.compile(r"\b(STAIR\s*\w+|VOID|OPENING|SHAFT|LIFT)\b", re.I), text),
        confidence=0.75 if concrete else 0.45,
        warnings=warnings,
    )


def _profile_with_gemini(page, paths, classes, cfg, renderer, use_ai: bool):
    profile = _deterministic_profile(page)
    out_dir = Path(renderer.out_dir)
    if not use_ai or not cfg.enable_floor_system_judge:
        (out_dir / "step_08b_floor_system_profile.json").write_text(
            json.dumps(asdict(profile), indent=2, ensure_ascii=False), encoding="utf-8")
        return profile

    style_rows = []
    for c in classes:
        if c.key.fill is None:
            continue
        style_rows.append({
            "style_id": c.id,
            "description": c.key.describe(),
            "path_count": c.n_paths,
            "bbox": [round(v, 1) for v in c.bbox],
        })
    prompt = f"""You are the semantic floor-system analyst for a structural PDF.

Return evidence rules only. Do not draw polygons and do not infer RGB rules
from appearance alone. Generic FLOOR STRUCTURE means FLOOR_EXTENT_ONLY; it is
not proof of concrete. Only explicit PT SLAB, POST TENSIONED SLAB, CONCRETE
SLAB, SLAB ON GRADE or S.O.G text proves concrete material. Distinguish
PT_CONCRETE_SLAB, CONCRETE_SLAB, FLOOR_EXTENT_ONLY,
STEEL_OR_COMPOSITE_FLOOR, OPENING, ANNOTATION and UNKNOWN.

VISIBLE PAGE TEXT:
{page.get_text('text')[:10000]}

FILLED STYLE CATALOG:
{json.dumps(style_rows, ensure_ascii=False)}
"""
    (out_dir / "step_08b_floor_system_profile_prompt.txt").write_text(
        prompt, encoding="utf-8")
    try:
        from src.vision_refiner import find_legend_rect, render_crop
        _full, legend = render_crop(page, find_legend_rect(page), cfg.prompt_dpi)
        pix = page.get_pixmap(matrix=__import__("fitz").Matrix(
            cfg.prompt_dpi / 72.0, cfg.prompt_dpi / 72.0), alpha=False)
        data = gemini_client.call_gemini_json(
            prompt, [pix.tobytes("png"), legend], _PROFILE_SCHEMA,
            cfg.gemini_model, log_path=str(out_dir / "prompts.log"),
            tag="floor_system_profile",
            raw_path=str(out_dir / "step_08b_floor_system_profile_raw.txt"))
        if isinstance(data, dict):
            # Deterministic explicit terms cannot be erased by the model.
            profile.concrete_slab_terms = list(dict.fromkeys(
                profile.concrete_slab_terms + data.get("concrete_slab_terms", [])))
            profile.floor_extent_terms = list(dict.fromkeys(
                profile.floor_extent_terms + data.get("floor_extent_terms", [])))
            profile.steel_floor_terms = list(dict.fromkeys(
                profile.steel_floor_terms + data.get("steel_floor_terms", [])))
            profile.opening_terms = list(dict.fromkeys(
                profile.opening_terms + data.get("opening_terms", [])))
            profile.fill_rules = data.get("fill_rules", [])
            profile.boundary_rules = data.get("boundary_rules", [])
            profile.symbol_families = data.get("symbol_families", [])
            profile.confidence = max(profile.confidence,
                                     float(data.get("confidence") or 0.0))
            profile.warnings.extend(data.get("warnings", []))
    except Exception as exc:
        profile.warnings.append(f"floor-system profile Gemini failed: {exc}")
    (out_dir / "step_08b_floor_system_profile.json").write_text(
        json.dumps(asdict(profile), indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def _words_for_polygon(words, polygon, buffer_pt: float = 3.0) -> list[str]:
    region = polygon.buffer(buffer_pt)
    found = []
    for w in words:
        p = Point((w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0)
        if region.contains(p):
            found.append(str(w[4]))
    return found


def _iter_edges(poly):
    coords = list(poly.exterior.coords)
    for a, b in zip(coords, coords[1:]):
        line = LineString([a, b])
        if line.length > 1e-6:
            yield line


def _extended_split_line(edge, gross):
    (x1, y1), (x2, y2) = edge.coords
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    span = math.hypot(gross.bounds[2] - gross.bounds[0],
                     gross.bounds[3] - gross.bounds[1]) * 3.0
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return LineString([(mx - ux * span, my - uy * span),
                       (mx + ux * span, my + uy * span)])


def _projected_span(gross, edge) -> float:
    (x1, y1), (x2, y2) = edge.coords
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    coords = list(gross.minimum_rotated_rectangle.exterior.coords)
    vals = [x * ux + y * uy for x, y in coords]
    return max(vals) - min(vals)


def _projection_range(geom, ux: float, uy: float) -> tuple[float, float]:
    coords = list(geom.minimum_rotated_rectangle.exterior.coords)
    values = [x * ux + y * uy for x, y in coords]
    return min(values), max(values)


def _terminal_tolerance_pt(scale, cfg) -> float:
    try:
        denominator = float(scale or 0)
    except (TypeError, ValueError):
        denominator = 0.0
    if denominator <= 0:
        return min(cfg.floor_system_terminal_tolerance_max_pt, 3.0)
    paper_mm_per_pt = 25.4 / 72.0
    tolerance = (cfg.floor_system_terminal_tolerance_mm /
                 (paper_mm_per_pt * denominator))
    return max(0.5, min(cfg.floor_system_terminal_tolerance_max_pt,
                        tolerance))


def _endpoint_aware_outer(gross, full_outer, edge, stairs, scale, cfg):
    """Cap an outer-side split at a stair-confirmed separator endpoint.

    Returns geometry plus auditable terminal evidence. No endpoint match means
    no destructive geometry: callers may expose the full strip for review,
    but it must remain part of the PT slab.
    """
    (x1, y1), (x2, y2) = list(edge.coords)
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    edge_points = [(x1, y1), (x2, y2)]
    edge_t = [x * ux + y * uy for x, y in edge_points]
    tolerance = _terminal_tolerance_pt(scale, cfg)
    matches = []
    for stair in stairs:
        stair_min, stair_max = _projection_range(stair.polygon, ux, uy)
        for edge_index, et in enumerate(edge_t):
            for stair_side, st in (("min", stair_min), ("max", stair_max)):
                matches.append((abs(et - st), stair, edge_index,
                                stair_side, st))
    if not matches:
        return None
    error, stair, cap_index, stair_side, stair_t = min(
        matches, key=lambda row: row[0])
    if error > tolerance:
        return {
            "bounded": None, "rejected": full_outer,
            "stair": stair, "error": error, "tolerance": tolerance,
            "cap": None, "separator": edge,
            "direction": "unresolved", "status": "unknown_review",
        }

    cap_edge_point = edge_points[cap_index]
    cap_edge_t = edge_t[cap_index]
    cap_point = (cap_edge_point[0] + ux * (stair_t - cap_edge_t),
                 cap_edge_point[1] + uy * (stair_t - cap_edge_t))
    other_t = edge_t[1 - cap_index]
    keep_sign = -1.0 if other_t < stair_t else 1.0
    px, py = -uy, ux
    span = math.hypot(gross.bounds[2] - gross.bounds[0],
                     gross.bounds[3] - gross.bounds[1]) * 3.0
    cap_line = LineString([
        (cap_point[0] - px * span, cap_point[1] - py * span),
        (cap_point[0] + px * span, cap_point[1] + py * span),
    ])
    try:
        pieces = list(split(full_outer, cap_line).geoms)
    except Exception:
        pieces = []
    eligible = []
    for piece in pieces:
        rp = piece.representative_point()
        value = rp.x * ux + rp.y * uy
        if (value - stair_t) * keep_sign >= -0.05:
            eligible.append(piece)
    if not eligible:
        return {
            "bounded": None, "rejected": full_outer,
            "stair": stair, "error": error, "tolerance": tolerance,
            "cap": cap_line.intersection(gross), "separator": edge,
            "direction": "unresolved", "status": "unknown_review",
        }
    bounded = unary_union(eligible).intersection(full_outer)
    rejected = full_outer.difference(bounded)
    direction = ("toward_separator_start" if 1 - cap_index == 0
                 else "toward_separator_end")
    return {
        "bounded": bounded, "rejected": rejected,
        "stair": stair, "error": error, "tolerance": tolerance,
        "cap": cap_line.intersection(gross), "separator": edge,
        "direction": direction, "status": "bounded_verified",
        "stair_side": stair_side,
    }


def _candidate_public(c: FloorSystemCandidate) -> dict:
    def line_coords(line):
        if line is None or getattr(line, "is_empty", True):
            return None
        geoms = getattr(line, "geoms", [line])
        return [[[round(x, 3), round(y, 3)] for x, y in g.coords]
                for g in geoms if hasattr(g, "coords")]

    return {
        "id": c.id,
        "bbox": [round(v, 2) for v in c.polygon.bounds],
        "area_pt2": round(c.polygon.area, 2),
        "source_face_ids": c.source_face_ids,
        "fill_role": c.fill_role,
        "boundary_styles": c.boundary_styles,
        "nearby_text": c.nearby_text[:40],
        "steel_symbols": c.steel_symbols,
        "concrete_symbols": c.concrete_symbols,
        "adjacent_openings": c.adjacent_openings,
        "touches_outer_edge": c.touches_outer_edge,
        "separator_evidence": c.separator_evidence,
        "positive_pt_evidence": c.positive_pt_evidence,
        "negative_pt_evidence": c.negative_pt_evidence,
        "deterministic_score": c.deterministic_score,
        "separator_segment": line_coords(c.separator_segment),
        "terminal_cap_segment": line_coords(c.terminal_cap_segment),
        "terminal_source": c.terminal_source,
        "terminal_alignment_error_pt": (
            round(c.terminal_alignment_error_pt, 3)
            if c.terminal_alignment_error_pt is not None else None),
        "extension_direction": c.extension_direction,
        "bounded_cut_area_pt2": round(c.bounded_cut_area_pt2, 2),
        "rejected_extension_area_pt2": round(
            c.rejected_extension_area_pt2, 2),
        "rejected_extension_bbox": (
            [round(v, 2) for v in c.rejected_extension_geometry.bounds]
            if c.rejected_extension_geometry is not None
            and not c.rejected_extension_geometry.is_empty else None),
        "cut_status": c.cut_status,
    }


def build_floor_system_candidates(page, paths, gross, openings, profile, cfg,
                                  scale=None):
    """Split floor extent only where vector topology and context corroborate."""
    words = page.get_text("words")
    gross_area = max(gross.area, 1.0)
    raw_other = []
    raw_review = []

    # Nested filled regions provide real vector system-boundary candidates.
    nested = []
    for path in paths:
        poly = path.fill_polygon
        if not path.is_filled or poly is None or poly.is_empty:
            continue
        clipped = poly.intersection(gross)
        ratio = clipped.area / gross_area
        if 0.20 <= ratio <= 0.90 and gross.buffer(0.5).contains(
                clipped.representative_point()):
            nested.append((path, clipped))

    stairs = [o for o in openings if str(o.type).upper() == "STAIR"]
    for path, inner in nested:
        inner_point = inner.representative_point()
        for edge in _iter_edges(inner.minimum_rotated_rectangle):
            coverage = edge.length / max(_projected_span(gross, edge), 1.0)
            if coverage < cfg.floor_system_separator_min_coverage:
                continue
            try:
                pieces = list(split(gross, _extended_split_line(edge, gross)).geoms)
            except Exception:
                continue
            if len(pieces) != 2:
                continue
            outer = next((p for p in pieces
                          if not p.buffer(0.1).contains(inner_point)), None)
            if outer is None or not (0.01 <= outer.area / gross_area <= 0.22):
                continue
            adjacent = [o for o in stairs if outer.buffer(
                cfg.floor_system_stair_proximity_pt).intersects(o.polygon)]
            if not adjacent:
                continue
            terminal = _endpoint_aware_outer(
                gross, outer, edge, adjacent, scale, cfg)
            if terminal is None:
                continue
            bounded = terminal.get("bounded")
            candidate_geom = bounded if bounded is not None else outer
            texts = _words_for_polygon(words, candidate_geom, 5.0)
            steel = sorted({t for t in texts if _STEEL_SYMBOL_RE.match(t)})
            concrete = sorted({t for t in texts if _CONCRETE_SYMBOL_RE.match(t)})
            separator = [
                "nested_floor_extent_edge",
                f"separator_coverage={coverage:.2f}",
                f"vector_style={path.style_id}",
            ]
            if bounded is not None:
                separator.extend([
                    "stair_confirmed_terminal_cap",
                    f"terminal_error_pt={terminal['error']:.3f}",
                ])
            negative = ["external_stair_interface"]
            if steel:
                negative.append("steel_symbol_family_inside_region")
            if profile.steel_floor_terms:
                negative.append("steel_floor_context")
            if profile.floor_extent_terms:
                negative.append("fill_is_floor_extent_only")
            positive = []
            if profile.concrete_slab_terms:
                positive.append("page_has_explicit_pt_concrete_title")
            score = len(positive) * 1.0 - len(negative) * 1.5 - 1.0
            candidate = FloorSystemCandidate(
                id="", polygon=candidate_geom,
                source_face_ids=[f"fill_path_{path.id}"],
                fill_role="FLOOR_EXTENT_ONLY", boundary_styles=[path.style_id],
                nearby_text=texts, steel_symbols=steel,
                concrete_symbols=concrete,
                adjacent_openings=[terminal["stair"].label],
                touches_outer_edge=candidate_geom.boundary.intersection(
                    gross.boundary.buffer(0.5)).length > 0,
                separator_evidence=separator,
                positive_pt_evidence=positive,
                negative_pt_evidence=negative,
                deterministic_score=round(score, 3),
                separator_segment=terminal.get("separator"),
                terminal_cap_segment=terminal.get("cap"),
                terminal_source=terminal["stair"].label,
                terminal_alignment_error_pt=terminal.get("error"),
                extension_direction=terminal.get("direction", ""),
                bounded_cut_area_pt2=(candidate_geom.area
                                      if bounded is not None else 0.0),
                rejected_extension_area_pt2=(
                    terminal["rejected"].area
                    if terminal.get("rejected") is not None else 0.0),
                rejected_extension_geometry=terminal.get("rejected"),
                cut_status=terminal.get("status", "unknown_review"),
            )
            if bounded is not None:
                raw_other.append(candidate)
            else:
                candidate.fill_role = "UNKNOWN_REVIEW"
                candidate.negative_pt_evidence.append(
                    "terminal_endpoint_unresolved")
                raw_review.append(candidate)

    # Same separator can appear in multiple filled paths; retain one geometry.
    other = []
    for candidate in sorted(raw_other, key=lambda c: -c.polygon.area):
        if any(candidate.polygon.intersection(x.polygon).area /
               max(candidate.polygon.union(x.polygon).area, 1.0) > 0.85
               for x in other):
            continue
        other.append(candidate)

    review = []
    for candidate in sorted(raw_review, key=lambda c: -c.polygon.area):
        if any(candidate.polygon.intersection(x.polygon).area /
               max(candidate.polygon.union(x.polygon).area, 1.0) > 0.85
               for x in review):
            continue
        review.append(candidate)

    # Main PT candidate remains the full gross floor. Destructive subtraction
    # happens only after a bounded OTHER candidate passes every guard.
    main = gross
    candidates = [FloorSystemCandidate(
        id="floor_pt_001", polygon=main,
        fill_role=("PT_CONCRETE_SLAB" if profile.concrete_slab_terms
                   else "UNKNOWN"),
        nearby_text=_words_for_polygon(words, main, 2.0),
        positive_pt_evidence=(
            ["explicit_pt_or_concrete_slab_text"]
            if profile.concrete_slab_terms else []),
        negative_pt_evidence=[],
        deterministic_score=4.0 if profile.concrete_slab_terms else 0.0,
    )]
    for i, candidate in enumerate(other, 1):
        candidate.id = f"floor_other_{i:03d}"
        candidates.append(candidate)
    for i, candidate in enumerate(review, 1):
        candidate.id = f"floor_review_{i:03d}"
        candidates.append(candidate)
    for i, opening in enumerate(openings, 1):
        candidates.append(FloorSystemCandidate(
            id=f"floor_opening_{i:03d}", polygon=opening.polygon,
            fill_role="OPENING", nearby_text=[opening.label],
            adjacent_openings=[opening.label],
            negative_pt_evidence=["verified_opening_resolver"],
            deterministic_score=-5.0,
        ))
    return candidates


def _judge(page, candidates, profile, cfg, renderer):
    out_dir = Path(renderer.out_dir)
    overlay = renderer.step08_floor_system_candidates(
        candidates, "step_08b_bounded_cut_candidates.png")
    image = renderer.render_for_prompt(overlay)
    from src.vision_refiner import find_legend_rect, render_crop
    _img, legend = render_crop(page, find_legend_rect(page), cfg.prompt_dpi)
    rows = [_candidate_public(c) for c in candidates]
    prompt = f"""You are the floor-system JUDGE for one structural plan.

Code generated stable candidate IDs and all coordinates. Return IDs only.
Never invent geometry. Generic FLOOR STRUCTURE means floor extent, not proof
of concrete. Classify PT/concrete slab separately from steel/composite/other
floor systems. OTHER FLOOR is not an opening. Verified stair/core candidates
remain openings. One signal alone (steel symbol, fill, stair adjacency) is not
enough to remove concrete; require the supplied multi-signal evidence.

SEMANTIC PROFILE:
{json.dumps(asdict(profile), ensure_ascii=False)}

CANDIDATES:
{json.dumps(rows, ensure_ascii=False)}

PAGE TEXT:
{page.get_text('text')[:7000]}
"""
    (out_dir / "step_08c_floor_system_judge_prompt.txt").write_text(
        prompt, encoding="utf-8")
    data = gemini_client.call_gemini_json(
        prompt, [image, legend], _JUDGE_SCHEMA, cfg.gemini_model,
        log_path=str(out_dir / "prompts.log"), tag="floor_system_judge",
        raw_path=str(out_dir / "step_08c_floor_system_judge_raw.txt"))
    (out_dir / "step_08c_floor_system_judge.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def resolve_floor_systems(page, paths, classes, gross_slabs, openings, cfg,
                          renderer, use_ai=True, scale=None):
    gross = unary_union([s["polygon_pdf"] for s in gross_slabs])
    profile = _profile_with_gemini(page, paths, classes, cfg, renderer, use_ai)
    candidates = build_floor_system_candidates(
        page, paths, gross, openings, profile, cfg, scale=scale)
    by_id = {c.id: c for c in candidates}
    warnings = list(profile.warnings)
    decision = None
    if use_ai and cfg.enable_floor_system_judge:
        try:
            decision = _judge(page, candidates, profile, cfg, renderer)
        except Exception as exc:
            warnings.append(f"floor-system judge failed: {exc}")

    pt_ids = []
    other_ids = []
    opening_ids = [c.id for c in candidates if c.fill_role == "OPENING"]
    non_floor_ids = []
    unknown_ids = []
    confidence_by_id = (decision or {}).get("confidence_by_id", {})

    for c in candidates:
        if c.fill_role == "OPENING":
            continue
        judge_pt = c.id in (decision or {}).get("pt_concrete_slab_ids", [])
        judge_other = c.id in (decision or {}).get("other_floor_system_ids", [])
        judge_non_floor = c.id in (decision or {}).get("non_floor_ids", [])
        try:
            conf = float(confidence_by_id.get(c.id, 0.0))
        except (TypeError, ValueError):
            conf = 0.0

        if (c.fill_role == "PT_CONCRETE_SLAB"
                and "explicit_pt_or_concrete_slab_text" in c.positive_pt_evidence):
            pt_ids.append(c.id)
            continue

        independent = set(c.negative_pt_evidence)
        strong_other = (
            len(c.separator_evidence) >= 2
            and c.cut_status == "bounded_verified"
            and "stair_confirmed_terminal_cap" in c.separator_evidence
            and "external_stair_interface" in independent
            and len(independent - {"external_stair_interface"}) >= 1)
        # Vertex response schemas may preserve the selected ID arrays while
        # omitting arbitrary-key confidence maps. Membership is semantic
        # evidence, but it is accepted only behind the same deterministic
        # multi-signal geometry gate.
        effective_conf = conf
        if judge_other and strong_other and effective_conf <= 0:
            effective_conf = max(profile.confidence,
                                 cfg.floor_system_other_min_confidence)
        if strong_other and (not decision or
                             (judge_other and effective_conf >=
                              cfg.floor_system_other_min_confidence)):
            other_ids.append(c.id)
        elif judge_non_floor and conf >= cfg.floor_system_other_min_confidence:
            non_floor_ids.append(c.id)
        elif judge_pt and conf >= cfg.floor_system_judge_min_confidence:
            pt_ids.append(c.id)
        else:
            unknown_ids.append(c.id)

    # Unknown/review regions remain in PT geometry. Only verified bounded
    # OTHER/NON_FLOOR candidates may destructively remove material.
    pt_geom = unary_union([by_id[x].polygon for x in pt_ids]) if pt_ids else gross
    other_geom = unary_union([by_id[x].polygon for x in other_ids]) \
        if other_ids else None
    destructive_ids = other_ids + non_floor_ids
    if destructive_ids:
        pt_geom = pt_geom.difference(unary_union(
            [by_id[x].polygon for x in destructive_ids]))
    opening_geom = unary_union([o.polygon for o in openings]) if openings else None
    net = pt_geom.difference(opening_geom) if opening_geom else pt_geom
    if unknown_ids:
        warnings.append("Unknown floor-system regions preserved in PT slab; debug review required: "
                        + ", ".join(unknown_ids))
    status = "verified" if decision and not unknown_ids else "review"
    confidences = []
    for cid in pt_ids + other_ids:
        try:
            value = float(confidence_by_id.get(cid, 0.0))
            if value <= 0 and decision:
                value = profile.confidence
            confidences.append(value)
        except (TypeError, ValueError):
            pass
    confidence = min(confidences) if confidences else (
        0.75 if other_ids and not unknown_ids else 0.0)
    resolution = FloorSystemResolution(
        pt_slab_ids=pt_ids, other_floor_ids=other_ids,
        opening_ids=opening_ids, non_floor_ids=non_floor_ids,
        unknown_ids=unknown_ids, pt_gross_geometry=pt_geom,
        other_floor_geometry=other_geom, pt_net_geometry=net,
        status=status, confidence=confidence, warnings=warnings,
        reason="Floor extent partitioned by vector separators and multi-signal evidence.")
    return resolution, candidates, profile


def candidate_payload(candidates):
    return [_candidate_public(c) for c in candidates]
