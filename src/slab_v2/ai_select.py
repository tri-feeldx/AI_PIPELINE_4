"""
Stage C — AI semantic selection (closed vocabulary).

Round 1: Gemini elects which STYLE CLASSES bound the slab (sees the page
         with classes color-coded, a synthesized swatch sheet, and the
         drawing's own legend crop).
Round 2: re-polygonize only the elected classes, Gemini elects which FACE
         ids form the slab and which are voids (sees numbered faces).

The model only ever returns ids. Coordinates come from the PDF vector data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import (StyleClass, VectorPath, FaceGraph,
                                ClassElection, FaceSelection)
from src.slab_v2 import gemini_client
from src.slab_v2.debug_render import PageRenderer


class AIError(Exception):
    """Raised when Gemini fails or exhausts the call budget."""


ROLES = ["SLAB_EDGE", "WALL", "COLUMN", "GRID", "DIMENSION", "HATCH",
         "VOID_EDGE", "ANNOTATION", "FRAME", "OTHER"]


@dataclass
class SelectionContext:
    page: fitz.Page
    paths: list
    classes: list
    cfg: SlabV2Config
    content_rect: fitz.Rect
    content_area_pt2: float
    renderer: PageRenderer
    fg_all: FaceGraph
    scale: int | None
    calls_used: int = 0
    title_text: str = ""
    _legend_png: bytes | None = None

    def log_path(self) -> str:
        return str(Path(self.renderer.out_dir) / "prompts.log")

    def legend_png(self) -> bytes:
        if self._legend_png is None:
            from src.vision_refiner import find_legend_rect, render_crop
            rect = find_legend_rect(self.page)
            _, png = render_crop(self.page, rect, self.cfg.prompt_dpi)
            self._legend_png = png
        return self._legend_png

    def call(self, prompt, images, schema, tag) -> dict:
        if self.calls_used >= self.cfg.max_total_calls:
            raise AIError("Gemini call budget exhausted")
        self.calls_used += 1
        try:
            return gemini_client.call_gemini_json(
                prompt, images, schema, self.cfg.gemini_model,
                log_path=self.log_path(), tag=tag)
        except RuntimeError as e:
            raise AIError(str(e)) from e


# ── prefilter ──────────────────────────────────────────────────────────────────

def _prefilter_classes(ctx: SelectionContext) -> list[StyleClass]:
    """Deterministic fingerprints: hatch/tick classes (many tiny segments)
    and FRAME classes are excluded from the Round-1 prompt."""
    classes = ctx.classes
    if not classes:
        return []
    max_len = max(c.total_length_pt for c in classes)
    kept = []
    for c in classes:
        if c.role == "FRAME":
            c.prefiltered = True
            continue
        if c.total_length_pt < 0.01 * max_len:
            c.prefiltered = True
            c.role = "OTHER"
            continue
        if c.median_seg_len_pt < 6.0 and c.n_segments > 200:
            c.prefiltered = True
            c.role = "HATCH"
            continue
        kept.append(c)
    return kept[: ctx.cfg.max_classes_in_prompt]


# ── Round 1: class election ───────────────────────────────────────────────────

_ROUND1_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "roles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "class_id": {"type": "INTEGER"},
                    "role": {"type": "STRING", "enum": ROLES},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["class_id", "role"],
            },
        },
        "slab_edge_classes": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "supporting_classes": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "reasoning": {"type": "STRING"},
    },
    "required": ["roles", "slab_edge_classes", "reasoning"],
}


def _round1_prompt(ctx: SelectionContext, candidates: list[StyleClass],
                   feedback: list[str]) -> str:
    rows = []
    for c in candidates:
        rows.append(
            f"CLASS {c.id}: stroke={c.key.stroke} fill={c.key.fill} "
            f"width={c.key.width}pt "
            f"{'dashed ' + c.key.dashes if c.key.dashes else 'solid'} | "
            f"{c.n_paths} paths, {c.n_segments} segments, "
            f"total {int(c.total_length_pt)}pt, "
            f"median seg {c.median_seg_len_pt:.1f}pt")
    excluded = [c for c in ctx.classes if c.prefiltered]
    excl_line = (f"(Pre-excluded as hatch/frame/noise: "
                 f"{[c.id for c in excluded]} — say so in reasoning if you "
                 f"believe the slab edge is among them.)" if excluded else "")
    fb = ""
    if feedback:
        fb = ("\nPREVIOUS ATTEMPT FAILED VALIDATION:\n- "
              + "\n- ".join(feedback) + "\nRe-answer accordingly.\n")
    return f"""You are analysing ONE structural engineering drawing (a concrete slab plan).

IMAGE 1: the drawing with every line STYLE CLASS over-drawn in a distinct color.
IMAGE 2: a swatch sheet showing each class's true line style (width/dash/color) and stats.
IMAGE 3: the drawing's own legend, cropped from the sheet.

The structural slab face is bounded by SOME of these line classes. Different
drawings use different conventions: a thick black outline, a colored line,
or the faces of concrete walls. Use the drawing's legend (IMAGE 3) to decide
what each line style means on THIS drawing.

Style classes:
{chr(10).join(rows)}
{excl_line}

TASK:
1. Assign each CLASS id exactly one role from: {", ".join(ROLES)}.
2. List slab_edge_classes — the class ids whose lines form the OUTER BOUNDARY
   of the structural slab on this drawing.
3. List supporting_classes — class ids that complete the boundary where the
   slab-edge line stops: wall faces, and especially MATCH LINES (a dashed
   line marked "REFER TO DRAWING ..." where the plan continues on another
   sheet — the slab region on this page is bounded by that match line).
   Empty list if none.
{fb}
Title-block text for context:
{ctx.title_text[:1200]}
"""


def elect_classes(ctx: SelectionContext,
                  feedback: list[str]) -> ClassElection:
    candidates = _prefilter_classes(ctx)
    if not candidates:
        raise AIError("No style classes left after prefilter")

    img1 = ctx.renderer.render_for_prompt(
        str(Path(ctx.renderer.out_dir) / "step_01_paths_by_style.png"))
    img2 = ctx.renderer.render_for_prompt(
        str(Path(ctx.renderer.out_dir) / "step_02_style_legend_sheet.png"))
    images = [img1, img2, ctx.legend_png()]
    if ctx.cfg.save_prompt_images:
        (Path(ctx.renderer.out_dir) / "step_05_prompt_round1_legend.png"
         ).write_bytes(images[2])

    # capture fingerprinted SLAB_EDGE ids before Gemini overwrites roles
    fingerprinted_slab_ids = {
        c.id for c in candidates
        if c.role == "SLAB_EDGE" and c.role_confidence >= 0.70
    }

    data = ctx.call(_round1_prompt(ctx, candidates, feedback), images,
                    _ROUND1_SCHEMA, tag="round1_class_election")

    valid_ids = {c.id for c in candidates}
    slab_ids = [i for i in data.get("slab_edge_classes", []) if i in valid_ids]
    supp_ids = [i for i in data.get("supporting_classes", [])
                if i in valid_ids and i not in slab_ids]

    # extract roles BEFORE checking slab_ids — roles (WALL, COLUMN, etc.)
    # are valuable even on pages where slab edges aren't found
    by_id = {c.id: c for c in ctx.classes}
    roles = {}
    for r in data.get("roles", []):
        cid = r.get("class_id")
        if cid in by_id:
            roles[cid] = r.get("role", "OTHER")
            by_id[cid].role = r.get("role", "OTHER")
            by_id[cid].role_confidence = float(r.get("confidence") or 0.0)

    # deterministic anchor: fingerprinted SLAB_EDGE always included
    for fid in fingerprinted_slab_ids:
        if fid not in slab_ids:
            slab_ids.insert(0, fid)

    if not slab_ids:
        raise AIError("Round 1: model returned no valid slab_edge_classes")

    # deterministic fallback: too many slab_edge classes → filter by style
    max_slab = getattr(ctx.cfg, "max_slab_edge_classes", 2)
    if len(slab_ids) > max_slab:
        filtered = [sid for sid in slab_ids
                    if by_id[sid].key.stroke is not None
                    and max(by_id[sid].key.stroke) <= 0.3
                    and not by_id[sid].key.dashes
                    and by_id[sid].key.width >= 0.5]
        if filtered:
            if len(filtered) > max_slab:
                filtered.sort(key=lambda sid: -by_id[sid].total_length_pt)
                filtered = filtered[:max_slab]
            dropped = [s for s in slab_ids if s not in filtered]
            supp_ids = dropped + supp_ids
            slab_ids = filtered

    # coverage check: low coverage is a WARNING, not a failure — the greedy
    # class augmentation in the pipeline can still close the real boundary
    xs, ys = [], []
    for i in slab_ids:
        b = by_id[i].bbox
        xs.extend((b[0], b[2]))
        ys.extend((b[1], b[3]))
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    cov = (bw * bh) / max(ctx.content_area_pt2, 1.0)
    coverage_warning = None
    if cov < ctx.cfg.min_class_coverage_frac:
        coverage_warning = (
            f"elected classes {slab_ids} cover only {cov:.0%} of the "
            f"drawing area — relying on class augmentation to close the "
            f"boundary (check step_06)")

    return ClassElection(
        slab_edge_classes=slab_ids,
        supporting_classes=supp_ids,
        roles=roles,
        reasoning=data.get("reasoning", ""),
        raw_response=str(data),
        warning=coverage_warning or "",
    )


# ── Round 2: face election ────────────────────────────────────────────────────

_ROUND2_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "slabs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "face_ids": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "void_face_ids": {"type": "ARRAY",
                                      "items": {"type": "INTEGER"}},
                    "label": {"type": "STRING"},
                },
                "required": ["face_ids"],
            },
        },
        "confidence": {"type": "NUMBER"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["slabs", "reasoning"],
}


def _face_table(ctx: SelectionContext, fg: FaceGraph,
                shown_ids: set[int]) -> str:
    rows = []
    scale = ctx.scale or 100
    pt_to_m = (25.4 / 72.0) * scale / 1000.0
    for f in sorted(fg.faces, key=lambda f: -f.area_pt2):
        if f.id not in shown_ids:
            continue
        area_m2 = f.area_pt2 * pt_to_m * pt_to_m
        rows.append(
            f"FACE {f.id}: area {area_m2:.1f} m2 (at 1:{scale}), "
            f"centroid ({f.label_anchor[0]:.0f},{f.label_anchor[1]:.0f})pt, "
            f"depth {f.depth}, bounded by classes {sorted(f.style_ids)}"
            + (", from filled region" if f.source == "fill" else ""))
    return "\n".join(rows)


def _texts_inside_faces(ctx: SelectionContext, fg: FaceGraph,
                        shown_ids: set[int]) -> str:
    """Words inside each candidate face (void/stair keywords surface here)."""
    from shapely.geometry import Point
    words = ctx.page.get_text("words")
    out = []
    for f in fg.faces:
        if f.id not in shown_ids or f.depth == 0:
            continue
        hits = []
        for w in words:
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            if f.polygon.contains(Point(cx, cy)):
                hits.append(w[4])
            if len(hits) >= 8:
                break
        if hits:
            out.append(f"FACE {f.id} contains text: {' '.join(hits)}")
    return "\n".join(out[:60])


def _round2_prompt(ctx: SelectionContext, fg: FaceGraph,
                   shown_ids: set[int], feedback: list[str]) -> str:
    fb = ""
    if feedback:
        fb = ("\nPREVIOUS ATTEMPT FAILED VALIDATION:\n- "
              + "\n- ".join(feedback) + "\nRe-answer accordingly.\n")
    return f"""You are analysing ONE structural slab plan. The drawing area was
polygonized into closed FACES using only the line classes you elected as the
slab boundary. IMAGE 1 shows every face filled with a pastel color and its
FACE id printed at its center.

TASK:
1. slabs: group the face ids that together constitute each structural slab
   (gross outline). Most pages have ONE slab; if the page clearly shows
   multiple separate slabs, return one group per slab with a short label.
   Faces in one group must be adjacent (their union must be one region).
2. void_face_ids: faces INSIDE a slab that are openings — stairs, lifts,
   shafts, penetrations. Look for hatch crosses and the text evidence below.
   Do NOT mark column/wall outlines as voids.

Face table:
{_face_table(ctx, fg, shown_ids)}

Text found inside nested faces:
{_texts_inside_faces(ctx, fg, shown_ids)}
{fb}"""


def select_faces(ctx: SelectionContext, election: ClassElection,
                 fg: FaceGraph, feedback: list[str]) -> FaceSelection:
    """DEPRECATED — Round 2 was replaced by deterministic assembly in
    pipeline.py (union of all significant faces; architect's rule "better
    too much than too little"). Kept for experiments only; not called by
    the default pipeline."""
    cfg = ctx.cfg
    faces_sorted = sorted(fg.faces, key=lambda f: -f.area_pt2)
    shown = faces_sorted[: cfg.max_faces_in_prompt]
    shown_ids = {f.id for f in shown}

    img_path = ctx.renderer.faces_numbered(
        fg, "step_07_faces_candidates.png",
        content_area_pt2=ctx.content_area_pt2)
    images = [ctx.renderer.render_for_prompt(img_path)]

    data = ctx.call(_round2_prompt(ctx, fg, shown_ids, feedback), images,
                    _ROUND2_SCHEMA, tag="round2_face_election")

    by_id = {f.id: f for f in fg.faces}
    slabs = []
    for s in data.get("slabs", []):
        f_ids = [i for i in s.get("face_ids", []) if i in by_id]
        v_ids = [i for i in s.get("void_face_ids", []) if i in by_id
                 and i not in f_ids]
        if f_ids:
            slabs.append({"face_ids": f_ids, "void_face_ids": v_ids,
                          "label": s.get("label") or ""})
    if not slabs:
        raise AIError("Round 2: model selected no valid faces")

    # geometric validation (tolerant — assembly drops stray faces itself)
    from shapely.ops import unary_union
    total_area = 0.0
    for s in slabs:
        union = unary_union([by_id[i].polygon for i in s["face_ids"]])
        # voids must touch the slab body; silently drop the rest
        s["void_face_ids"] = [
            v for v in s["void_face_ids"]
            if by_id[v].polygon.buffer(1.0).intersects(union)]
        total_area += union.area
    frac = total_area / max(ctx.content_area_pt2, 1.0)
    if frac < cfg.min_area_frac:
        raise AIError(
            f"Round 2: selected slab area is only {frac:.0%} of the drawing "
            f"area — too small to be the structural slab")
    if frac > cfg.max_area_frac:
        raise AIError(
            f"Round 2: selected slab area is {frac:.0%} of the drawing "
            f"area — too large, likely includes the sheet border")

    return FaceSelection(
        slabs=slabs,
        confidence=float(data.get("confidence") or 0.0),
        reasoning=data.get("reasoning", ""),
        raw_response=str(data),
    )
