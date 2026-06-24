"""
Stage E — debug rendering. One PNG per pipeline step, written to
debug_slab_v2/<pdf_stem>/page_<n>/step_XX_*.png.

All overlays share the same transform (PDF pt -> px at cfg.debug_dpi) and a
faded page raster underlay, so every step is visually comparable.
Images sent to Gemini are produced by these same functions and saved
byte-identical (render once, reuse bytes).
"""

from __future__ import annotations

import colorsys
import io
import math
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import StyleClass, VectorPath, Face, FaceGraph


# fixed 12-color palette for style classes (cycled with numbered tags beyond 12)
PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (128, 0, 0),
]


def class_color(cid: int) -> tuple:
    return PALETTE[cid % len(PALETTE)]


def face_color(fid: int) -> tuple:
    """Pastel distinct colors via golden-ratio hue stepping."""
    h = (fid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.45, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _safe_text(text) -> str:
    """Keep debug labels drawable when PIL falls back to latin-1 fonts."""
    return str(text).encode("ascii", "replace").decode("ascii")


class PageRenderer:
    """Shared transform + raster underlay for one page."""

    def __init__(self, page: fitz.Page, cfg: SlabV2Config, out_dir: str):
        self.page = page
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.scale = cfg.debug_dpi / 72.0
        self._raster: Image.Image | None = None
        self._faded: Image.Image | None = None

    def _ensure_raster(self):
        if self._raster is not None:
            return
        mat = fitz.Matrix(self.scale, self.scale)
        pix = self.page.get_pixmap(matrix=mat, alpha=False)
        self._raster = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        arr = np.asarray(self._raster, dtype=np.float32)
        faded = (arr * 0.35 + 255 * 0.65).astype(np.uint8)
        self._faded = Image.fromarray(faded)

    @property
    def raster(self) -> Image.Image:
        self._ensure_raster()
        return self._raster

    @property
    def faded(self) -> Image.Image:
        self._ensure_raster()
        return self._faded

    # ── coordinate transform ──────────────────────────────────────────────
    def tx(self, p) -> tuple:
        return (p[0] * self.scale, p[1] * self.scale)

    def _save(self, img: Image.Image, name: str) -> str:
        path = self.out_dir / name
        img.save(path)
        return str(path)

    def _base(self) -> Image.Image:
        return self.faded.copy()

    # ── step 00 ───────────────────────────────────────────────────────────
    def step00_page_raster(self) -> str:
        return self._save(self.raster, "step_00_page_raster.png")

    # ── step 01: all paths colored by style class ─────────────────────────
    def step01_paths_by_style(self, paths: list[VectorPath],
                              classes: list[StyleClass],
                              name: str = "step_01_paths_by_style.png") -> str:
        img = self._base()
        dr = ImageDraw.Draw(img)
        for p in paths:
            color = class_color(p.style_id)
            w = 2 if not p.outside_content else 1
            for (a, b) in p.segments:
                dr.line([self.tx(a), self.tx(b)], fill=color, width=w)
        self._legend_inset(dr, classes, img.width)
        return self._save(img, name)

    def _legend_inset(self, dr: ImageDraw.ImageDraw,
                      classes: list[StyleClass], img_w: int) -> None:
        font = _font(16)
        x0, y0 = 10, 10
        row_h = 22
        shown = classes[:20]
        dr.rectangle([x0 - 4, y0 - 4, x0 + 360, y0 + row_h * len(shown) + 4],
                     fill=(255, 255, 255), outline=(0, 0, 0))
        for i, c in enumerate(shown):
            y = y0 + i * row_h
            dr.line([(x0, y + row_h // 2), (x0 + 40, y + row_h // 2)],
                    fill=class_color(c.id), width=3)
            label = (f"C{c.id} w={c.key.width} "
                     f"{'dash' if c.key.dashes else 'solid'} "
                     f"n={c.n_segments} L={int(c.total_length_pt)} {c.role}")
            dr.text((x0 + 48, y + 2), label, fill=(0, 0, 0), font=font)

    # ── step 02: synthesized style legend sheet ───────────────────────────
    def step02_style_legend_sheet(self, classes: list[StyleClass],
                                  name: str = "step_02_style_legend_sheet.png") -> str:
        row_h, sw_w, width = 56, 220, 1000
        shown = [c for c in classes if not c.prefiltered][:self.cfg.max_classes_in_prompt]
        img = Image.new("RGB", (width, row_h * max(len(shown), 1) + 20),
                        (255, 255, 255))
        dr = ImageDraw.Draw(img)
        font = _font(18)
        for i, c in enumerate(shown):
            y = 10 + i * row_h
            cy = y + row_h // 2 - 6
            color = c.key.stroke or c.key.fill or (0, 0, 0)
            rgb = tuple(int(v * 255) for v in color) if max(color) <= 1 else color
            wpx = max(1, int(round(c.key.width * self.scale)))
            if c.key.dashes:
                x = 20
                while x < 20 + sw_w:
                    dr.line([(x, cy), (min(x + 12, 20 + sw_w), cy)],
                            fill=rgb, width=wpx)
                    x += 20
            else:
                dr.line([(20, cy), (20 + sw_w, cy)], fill=rgb, width=wpx)
            if c.key.fill is not None:
                frgb = tuple(int(v * 255) for v in c.key.fill)
                dr.rectangle([20, cy + 10, 80, cy + 22], fill=frgb,
                             outline=(120, 120, 120))
            desc = (f"CLASS {c.id} - width {c.key.width}pt, "
                    f"{'dashed ' + c.key.dashes if c.key.dashes else 'solid'}, "
                    f"stroke {c.key.stroke}, fill {c.key.fill}, "
                    f"{c.n_segments} segs, total {int(c.total_length_pt)}pt")
            dr.text((20 + sw_w + 16, y + 8), _safe_text(desc),
                    fill=(0, 0, 0), font=font)
        return self._save(img, name)

    # ── step 03: planarized graph ─────────────────────────────────────────
    def step03_planarized(self, fg: FaceGraph,
                          name: str = "step_03_planarized.png") -> str:
        img = self._base()
        dr = ImageDraw.Draw(img)
        for f in fg.faces:
            ext = [self.tx(p) for p in f.polygon.exterior.coords]
            dr.line(ext, fill=(110, 110, 110), width=1)
        for d in fg.dangles:
            pts = [self.tx(p) for p in d.coords]
            dr.line(pts, fill=(230, 25, 75), width=3)
        for c in fg.cut_edges:
            pts = [self.tx(p) for p in c.coords]
            dr.line(pts, fill=(245, 130, 48), width=2)
        font = _font(20)
        dr.text((10, 10),
                f"faces={len(fg.faces)} dangles={len(fg.dangles)} "
                f"cuts={len(fg.cut_edges)} snap={fg.snap_used_pt}pt "
                f"segs_in={fg.n_segments_in}",
                fill=(0, 0, 0), font=font)
        return self._save(img, name)

    # ── step 04 / 07: numbered faces ──────────────────────────────────────
    def faces_numbered(self, fg: FaceGraph, name: str,
                       min_label_frac: float = 0.002,
                       content_area_pt2: float | None = None) -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        area_ref = content_area_pt2 or (self.page.rect.width *
                                        self.page.rect.height)
        for f in fg.faces:
            col = face_color(f.id) + (110,)
            ext = [self.tx(p) for p in f.polygon.exterior.coords]
            dr.polygon(ext, fill=col, outline=face_color(f.id) + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        for f in fg.faces:
            if f.area_pt2 < min_label_frac * area_ref:
                continue
            size = int(max(8, min(28, math.sqrt(f.area_pt2) * self.scale * 0.08)))
            font = _font(size + 8)
            x, y = self.tx(f.label_anchor)
            txt = str(f.id)
            bbox = dr.textbbox((x, y), txt, font=font, anchor="mm")
            dr.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2],
                         fill=(255, 255, 255))
            dr.text((x, y), txt, fill=(0, 0, 0), font=font, anchor="mm")
        return self._save(img, name)

    # ── step 06: elected classes ──────────────────────────────────────────
    def step06_elected_classes(self, paths: list[VectorPath],
                               classes: list[StyleClass],
                               slab_classes: list[int],
                               support_classes: list[int],
                               name: str = "step_06_elected_classes.png") -> str:
        img = self._base()
        dr = ImageDraw.Draw(img)
        slab_set, supp_set = set(slab_classes), set(support_classes)
        for p in paths:
            if p.outside_content:
                continue
            if p.style_id in slab_set:
                color, w = (230, 25, 75), 4
            elif p.style_id in supp_set:
                color, w = (0, 130, 200), 2
            else:
                color, w = (200, 200, 200), 1
            for (a, b) in p.segments:
                dr.line([self.tx(a), self.tx(b)], fill=color, width=w)
        font = _font(22)
        dr.text((10, 10),
                f"SLAB_EDGE classes (red): {sorted(slab_set)}   "
                f"supporting (blue): {sorted(supp_set)}",
                fill=(0, 0, 0), font=font)
        return self._save(img, name)

    # ── step 08: AI selection ─────────────────────────────────────────────
    def step08_ai_selection(self, fg: FaceGraph, slabs: list,
                            name: str = "step_08_ai_selection.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        by_id = {f.id: f for f in fg.faces}
        selected, voids = set(), set()
        for s in slabs:
            selected.update(s.get("face_ids", []))
            voids.update(s.get("void_face_ids", []))
        for f in fg.faces:
            if f.id in voids:
                col = (230, 25, 75, 140)
            elif f.id in selected:
                col = (60, 180, 75, 120)
            else:
                col = (160, 160, 160, 40)
            ext = [self.tx(p) for p in f.polygon.exterior.coords]
            dr.polygon(ext, fill=col)
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        font = _font(22)
        dr.text((10, 10),
                f"AI selection - slab faces (green): {sorted(selected)}, "
                f"voids (red): {sorted(voids)}",
                fill=(0, 0, 0), font=font)
        return self._save(img, name)

    # ── step 08 (v2): deterministic assembly ──────────────────────────────
    def step08_assembled_slab(self, slabs: list, kept_face_ids: list,
                              name: str = "step_08_assembled_slab.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for s in slabs:
            geom = s["polygon_pdf"]
            for g in getattr(geom, "geoms", [geom]):
                ext = [self.tx(p) for p in g.exterior.coords]
                dr.polygon(ext, fill=(60, 180, 75, 110),
                           outline=(0, 100, 0, 255))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.text((10, 10),
                f"deterministic assembly - union of {len(kept_face_ids)} "
                f"faces, {len(slabs)} slab(s)",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step08_slab_candidates(self, candidates: list,
                               name: str = "step_08a_slab_face_candidates.png") -> str:
        """Atomic slab faces with stable IDs for semantic judging."""
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for c in candidates:
            col = ((60, 180, 75, 85) if c.deterministic_score >= 0
                   else (230, 160, 30, 85))
            for g in getattr(c.polygon, "geoms", [c.polygon]):
                if not hasattr(g, "exterior"):
                    continue
                ext = [self.tx(p) for p in g.exterior.coords]
                dr.polygon(ext, fill=col, outline=col[:3] + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        font = _font(13)
        for c in candidates:
            rp = c.polygon.representative_point()
            x, y = self.tx((rp.x, rp.y))
            bbox = dr.textbbox((x, y), c.id, font=font, anchor="mm")
            dr.rectangle([bbox[0] - 2, bbox[1] - 1,
                          bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255))
            dr.text((x, y), c.id, fill=(0, 0, 0), font=font, anchor="mm")
        dr.text((10, 10), f"slab face candidates: {len(candidates)}",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step08_slab_decision(self, candidates: list, resolution,
                             name: str = "step_08c_slab_decision.png") -> str:
        """Green slab, lime appendage, red removal, orange review."""
        selected = set(resolution.selected_slab_ids)
        appendages = set(resolution.appendage_ids)
        removed = set(resolution.non_slab_ids) | set(resolution.opening_ids)
        review = set(resolution.review_ids)
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for c in candidates:
            if c.id in removed:
                col = (230, 25, 75, 150)
            elif c.id in review:
                col = (245, 130, 48, 130)
            elif c.id in appendages:
                col = (110, 230, 40, 120)
            elif c.id in selected:
                col = (60, 180, 75, 120)
            else:
                col = (150, 150, 150, 35)
            for g in getattr(c.polygon, "geoms", [c.polygon]):
                if hasattr(g, "exterior"):
                    dr.polygon([self.tx(p) for p in g.exterior.coords],
                               fill=col, outline=col[:3] + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), _safe_text(
            f"slab judge={resolution.status} confidence="
            f"{resolution.confidence:.2f} review={len(review)}"),
            fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step08_floor_system_candidates(
            self, candidates: list,
            name: str = "step_08a_floor_system_candidates.png") -> str:
        """Candidate partitions offered to the floor-system judge."""
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for c in candidates:
            if c.fill_role == "OPENING":
                col = (235, 40, 55, 115)
            elif c.id.startswith("floor_other"):
                col = (30, 125, 210, 105)
            else:
                col = (60, 180, 75, 90)
            for g in getattr(c.polygon, "geoms", [c.polygon]):
                if hasattr(g, "exterior"):
                    dr.polygon([self.tx(p) for p in g.exterior.coords],
                               fill=col, outline=col[:3] + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        font = _font(13)
        for c in candidates:
            rp = c.polygon.representative_point()
            x, y = self.tx((rp.x, rp.y))
            bbox = dr.textbbox((x, y), c.id, font=font, anchor="mm")
            dr.rectangle([bbox[0] - 2, bbox[1] - 1,
                          bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255))
            dr.text((x, y), c.id, fill=(0, 0, 0), font=font, anchor="mm")
        dr.text((10, 10), f"floor-system candidates: {len(candidates)}",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step08_separator_endpoints(
            self, candidates: list,
            name: str = "step_08a_separator_endpoints.png") -> str:
        """Vector separators and stair-confirmed terminal caps."""
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for c in candidates:
            sep = getattr(c, "separator_segment", None)
            cap = getattr(c, "terminal_cap_segment", None)
            if sep is not None and not sep.is_empty:
                for line in getattr(sep, "geoms", [sep]):
                    pts = [self.tx(p) for p in line.coords]
                    dr.line(pts, fill=(0, 180, 210, 255), width=5)
                    for p in (pts[0], pts[-1]):
                        dr.ellipse([p[0] - 6, p[1] - 6,
                                    p[0] + 6, p[1] + 6],
                                   fill=(0, 180, 210, 255))
            if cap is not None and not cap.is_empty:
                for line in getattr(cap, "geoms", [cap]):
                    if not hasattr(line, "coords"):
                        continue
                    dr.line([self.tx(p) for p in line.coords],
                            fill=(245, 130, 48, 255), width=5)
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), "cyan=separator, orange=stair terminal cap",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step08_floor_system_decision(
            self, candidates: list, resolution,
            name: str = "step_08d_floor_system_decision.png") -> str:
        """Green PT slab, blue other floor, red opening, orange unknown."""
        pt = set(resolution.pt_slab_ids)
        other = set(resolution.other_floor_ids)
        openings = set(resolution.opening_ids)
        non_floor = set(resolution.non_floor_ids)
        unknown = set(resolution.unknown_ids)
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for c in candidates:
            if c.id in openings:
                col = (245, 245, 245, 220)
                outline = (230, 25, 75, 255)
            elif c.id in other:
                col, outline = (25, 120, 210, 120), (10, 80, 180, 255)
            elif c.id in non_floor:
                col = outline = (220, 35, 45, 150)
            elif c.id in unknown:
                col, outline = (245, 130, 48, 130), (210, 90, 20, 255)
            elif c.id in pt:
                col, outline = (60, 180, 75, 120), (0, 110, 0, 255)
            else:
                col = outline = (140, 140, 140, 35)
            for g in getattr(c.polygon, "geoms", [c.polygon]):
                if hasattr(g, "exterior"):
                    dr.polygon([self.tx(p) for p in g.exterior.coords],
                               fill=col, outline=outline)
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), _safe_text(
            f"floor systems | {resolution.status} | PT={len(pt)} "
            f"other={len(other)} unknown={len(unknown)}"),
            fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step08_overcut_guard(
            self, candidates: list, resolution,
            name: str = "step_08e_overcut_guard.png") -> str:
        """Show accepted cuts and the infinite extension rejected by guard."""
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        pt = resolution.pt_gross_geometry
        for g in getattr(pt, "geoms", [pt]):
            if hasattr(g, "exterior"):
                dr.polygon([self.tx(p) for p in g.exterior.coords],
                           fill=(60, 180, 75, 80),
                           outline=(0, 110, 0, 255))
        accepted = set(resolution.other_floor_ids)
        for c in candidates:
            rejected = getattr(c, "rejected_extension_geometry", None)
            if rejected is not None and not rejected.is_empty:
                for g in getattr(rejected, "geoms", [rejected]):
                    if hasattr(g, "exterior"):
                        dr.polygon([self.tx(p) for p in g.exterior.coords],
                                   fill=(205, 40, 210, 125),
                                   outline=(150, 20, 160, 255))
            if c.id in accepted:
                for g in getattr(c.polygon, "geoms", [c.polygon]):
                    if hasattr(g, "exterior"):
                        dr.polygon([self.tx(p) for p in g.exterior.coords],
                                   fill=(25, 120, 210, 115),
                                   outline=(10, 80, 180, 255))
        img = Image.alpha_composite(img, ov).convert("RGB")
        ImageDraw.Draw(img).text(
            (10, 10), "blue=bounded cut, magenta=prevented overcut",
            fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step10_floor_system_geometry(self, geometry, title: str,
                                     name: str) -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for g in getattr(geometry, "geoms", [geometry]):
            if not hasattr(g, "exterior"):
                continue
            dr.polygon([self.tx(p) for p in g.exterior.coords],
                       fill=(60, 180, 75, 115), outline=(0, 110, 0, 255))
            for ring in g.interiors:
                dr.polygon([self.tx(p) for p in ring.coords],
                           fill=(255, 255, 255, 220),
                           outline=(230, 25, 75, 255))
        img = Image.alpha_composite(img, ov).convert("RGB")
        ImageDraw.Draw(img).text((10, 10), _safe_text(title),
                                 fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step10_net_slab(self, resolution,
                        name: str = "step_10_final_net_slab.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for g in getattr(resolution.net_geometry, "geoms",
                         [resolution.net_geometry]):
            if not hasattr(g, "exterior"):
                continue
            ext = [self.tx(p) for p in g.exterior.coords]
            dr.polygon(ext, fill=(60, 180, 75, 115),
                       outline=(0, 110, 0, 255))
            for ring in g.interiors:
                dr.polygon([self.tx(p) for p in ring.coords],
                           fill=(255, 255, 255, 220),
                           outline=(230, 25, 75, 255))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.text((10, 10), _safe_text(
            f"final net slab | {resolution.status} | "
            f"confidence={resolution.confidence:.2f}"),
            fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    # ── step 09 (v2): element footprints ──────────────────────────────────
    ELEMENT_COLORS = {
        "STAIR": (230, 25, 75), "LIFT": (245, 130, 48),
        "SHAFT": (145, 30, 180), "VOID": (0, 130, 200),
        "DUCT": (170, 110, 40),
    }

    def step09_elements(self, elements: list,
                        name: str = "step_09_elements.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for e in elements:
            col = self.ELEMENT_COLORS.get(e.type, (128, 128, 128))
            ext = [self.tx(p) for p in e.polygon.exterior.coords]
            dr.polygon(ext, fill=col + (130,), outline=col + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        font = _font(16)
        for e in elements:
            rp = e.polygon.representative_point()
            x, y = self.tx((rp.x, rp.y))
            txt = f"{e.type}: {e.label}"
            bbox = dr.textbbox((x, y), txt, font=font, anchor="mm")
            dr.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
                         fill=(255, 255, 255))
            dr.text((x, y), txt, fill=(0, 0, 0), font=font, anchor="mm")
        dr.text((10, 10), f"elements found: {len(elements)} "
                          f"(openings cut at Ruby export, not in 2D)",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step09_candidates(self, candidates: list,
                          name: str = "step_09c_opening_candidates.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for candidate in candidates:
            kind = candidate.get("kind_hint", "")
            if kind == "EQUIPMENT_REBATE":
                col = (120, 120, 120)
            elif kind.startswith("STAIR"):
                col = (230, 25, 75)
            elif kind == "SHAFT":
                col = (145, 30, 180)
            else:
                col = (0, 130, 200)
            poly = candidate["polygon"]
            for geom in getattr(poly, "geoms", [poly]):
                ext = [self.tx(p) for p in geom.exterior.coords]
                dr.polygon(ext, fill=col + (90,), outline=col + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        font = _font(13)
        for candidate in candidates:
            rp = candidate["polygon"].representative_point()
            x, y = self.tx((rp.x, rp.y))
            txt = _safe_text(candidate["id"])
            bbox = dr.textbbox((x, y), txt, font=font, anchor="mm")
            dr.rectangle([bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1],
                         fill=(255, 255, 255))
            dr.text((x, y), txt, fill=(0, 0, 0), font=font, anchor="mm")
        dr.text((10, 10), f"opening candidates: {len(candidates)}",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step09_opening_guards(
        self, candidates: list, walls: list, selected_ids: set[str],
        name: str = "step_09e_opening_geometry_guards.png",
    ) -> str:
        """Show fail-closed opening decisions and protected LW geometry."""
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for wall in walls:
            if not str(wall.label).upper().startswith("LW"):
                continue
            for geom in getattr(wall.polygon, "geoms", [wall.polygon]):
                ext = [self.tx(point) for point in geom.exterior.coords]
                dr.polygon(ext, fill=(130, 35, 170, 75),
                           outline=(130, 35, 170, 255))
        for candidate in candidates:
            cid = candidate.get("id", "")
            snap = candidate.get("geometry_audit", {}).get("boundary_snap", {})
            if not candidate.get("destructive_allowed", False):
                fill, outline = (220, 20, 180, 60), (220, 20, 180, 255)
            elif cid in selected_ids and snap.get("status") == "verified_snap":
                fill, outline = (255, 255, 255, 210), (0, 180, 210, 255)
            elif cid in selected_ids:
                fill, outline = (255, 255, 255, 210), (220, 30, 45, 255)
            else:
                fill, outline = (230, 150, 20, 70), (230, 150, 20, 255)
            polygon = candidate.get("polygon")
            if polygon is None:
                continue
            for geom in getattr(polygon, "geoms", [polygon]):
                ext = [self.tx(point) for point in geom.exterior.coords]
                dr.polygon(ext, fill=fill, outline=outline)
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.text((10, 10),
                f"opening guards: verified={len(selected_ids)} "
                f"prevented={sum(not c.get('destructive_allowed', False) for c in candidates)}",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    def step09_judgement(self, candidates: list, judgement: dict,
                         name: str = "step_09d_llm_judge.png") -> str:
        accepted = set(judgement.get("opening_ids", []))
        excluded = set(judgement.get("exclude_ids", []))
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for candidate in candidates:
            cid = candidate["id"]
            col = ((20, 170, 70) if cid in accepted else
                   (210, 40, 40) if cid in excluded else (230, 160, 30))
            poly = candidate["polygon"]
            for geom in getattr(poly, "geoms", [poly]):
                ext = [self.tx(p) for p in geom.exterior.coords]
                dr.polygon(ext, fill=col + (100,), outline=col + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        dr.text((10, 10),
                _safe_text(f"judge={judgement.get('status')} "
                           f"confidence={judgement.get('confidence', 0):.2f} "
                           f"accepted={len(accepted)} excluded={len(excluded)}"),
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    # ── step 10 (v2): final = gross slab + element openings preview ───────
    def step10_final(self, slabs: list, elements: list,
                     name: str = "step_10_final.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for s in slabs:
            geom = s["polygon_pdf"]
            for g in getattr(geom, "geoms", [geom]):
                ext = [self.tx(p) for p in g.exterior.coords]
                dr.polygon(ext, fill=(60, 180, 75, 110),
                           outline=(0, 100, 0, 255))
        for e in elements:
            col = self.ELEMENT_COLORS.get(e.type, (128, 128, 128))
            ext = [self.tx(p) for p in e.polygon.exterior.coords]
            dr.polygon(ext, fill=(255, 255, 255, 210), outline=col + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        y = 10
        for s in slabs:
            area = s.get("area_m2")
            label = (f"{s['label']}: {area:.1f} m2 (gross)"
                     if area else s["label"])
            dr.text((10, y), label, fill=(0, 0, 0), font=_font(22))
            y += 28
        dr.text((10, y), f"openings (white) = {len(elements)} elements, "
                         f"cut at export", fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    # ── step 10c: detected walls ──────────────────────────────────────────
    WALL_COLOR = (142, 36, 170)  # #8E24AA purple

    def step10c_walls(self, walls: list,
                      name: str = "step_10c_walls.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        wc = self.WALL_COLOR
        for w in walls:
            geoms = list(w.polygon.geoms) \
                if w.polygon.geom_type == "MultiPolygon" else [w.polygon]
            for g in geoms:
                ext = [self.tx(p) for p in g.exterior.coords]
                dr.polygon(ext, fill=wc + (130,), outline=wc + (255,))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        font = _font(14)
        for w in walls:
            rp = w.polygon.representative_point()
            x, y = self.tx((rp.x, rp.y))
            txt = f"{w.label} ({w.w_mm:.0f}x{w.l_mm:.0f})"
            bbox = dr.textbbox((x, y), txt, font=font, anchor="mm")
            dr.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
                         fill=(255, 255, 255))
            dr.text((x, y), txt, fill=(100, 0, 120), font=font, anchor="mm")
        types_count = {}
        for w in walls:
            types_count[w.wall_type] = types_count.get(w.wall_type, 0) + 1
        summary = "  ".join(f"{t}:{n}" for t, n in sorted(types_count.items()))
        dr.text((10, 10), f"walls: {len(walls)}   {summary}",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    # ── step 11: detected columns ──────────────────────────────────────────
    def step11_columns(self, columns: list,
                       name: str = "step_11_columns.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for c in columns:
            ext = [self.tx(p) for p in c.polygon.exterior.coords]
            dr.polygon(ext, fill=(255, 225, 25, 150),
                       outline=(200, 0, 0, 255))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        font = _font(14)
        for c in columns:
            rp = c.polygon.representative_point()
            x, y = self.tx((rp.x, rp.y))
            txt = c.symbol + ("" if c.labeled else "*")
            bbox = dr.textbbox((x, y), txt, font=font, anchor="mm")
            dr.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
                         fill=(255, 255, 255))
            dr.text((x, y), txt, fill=(160, 0, 0), font=font, anchor="mm")
        counts = {}
        for c in columns:
            counts[c.symbol] = counts.get(c.symbol, 0) + 1
        summary = "  ".join(f"{s}:{n}" for s, n in sorted(counts.items()))
        dr.text((10, 10), f"columns: {len(columns)}   {summary}   "
                          f"(* = size-matched, no text mark)",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    # ── dimensions (informational verify) ─────────────────────────────────
    def step_dimensions(self, dims: list, name: str) -> str:
        return self.step09_dimensions(dims, name)

    # ── legacy step 09: dimensions ────────────────────────────────────────
    def step09_dimensions(self, dims: list,
                          name: str = "step_09_dimensions.png") -> str:
        img = self._base()
        dr = ImageDraw.Draw(img)
        font = _font(16)
        for d in dims:
            x0, y0, x1, y1 = d.bbox
            ok = d.dim_line is not None
            color = (60, 180, 75) if ok else (150, 150, 150)
            dr.rectangle([self.tx((x0, y0)), self.tx((x1, y1))],
                         outline=color, width=2)
            if ok:
                a, b = d.dim_line
                dr.line([self.tx(a), self.tx(b)], fill=(0, 130, 200), width=3)
        n_ok = sum(1 for d in dims if d.dim_line is not None)
        dr.text((10, 10), f"dimensions: {len(dims)} parsed, {n_ok} associated",
                fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    # ── step 10: verification ─────────────────────────────────────────────
    def step10_verification(self, slab_geometry, report,
                            name: str = "step_10_verification.png") -> str:
        img = self._base()
        dr = ImageDraw.Draw(img)
        font = _font(18)
        if slab_geometry is not None and not slab_geometry.is_empty:
            geoms = getattr(slab_geometry, "geoms", [slab_geometry])
            for g in geoms:
                ext = [self.tx(p) for p in g.exterior.coords]
                dr.line(ext, fill=(0, 130, 200), width=3)
        for m in report.edge_matches:
            (a, b) = m["edge"]
            ok = m["rel_err"] <= self.cfg.dim_rel_tol
            dr.line([self.tx(a), self.tx(b)],
                    fill=(60, 180, 75) if ok else (230, 25, 75), width=5)
            mid = self.tx(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            dr.text(mid, f"{m['dim_value_mm']:.0f}mm "
                         f"(err {m['rel_err'] * 100:.1f}%)",
                    fill=(0, 0, 0), font=font)
        lines = [
            f"PASSED={report.passed}  scale=1:{report.scale_used}",
            f"scale_consistency={report.scale_consistency:.2f} "
            f"({report.n_dims_associated} dims)",
            f"edge_matches={len(report.edge_matches)}",
        ] + [f"FAIL: {f}" for f in report.failures]
        y = 10
        for ln in lines:
            dr.text((10, y), ln, fill=(0, 0, 0), font=_font(20))
            y += 26
        return self._save(img, name)

    # ── step 11: final ────────────────────────────────────────────────────
    def step11_final(self, slabs: list, attempts: int,
                     name: str = "step_11_final.png") -> str:
        img = self._base().convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        for s in slabs:
            geom = s["polygon_pdf"]
            geoms = getattr(geom, "geoms", [geom])
            for g in geoms:
                ext = [self.tx(p) for p in g.exterior.coords]
                dr.polygon(ext, fill=(60, 180, 75, 110),
                           outline=(0, 100, 0, 255))
                for hole in g.interiors:
                    pts = [self.tx(p) for p in hole.coords]
                    dr.polygon(pts, fill=(255, 255, 255, 200),
                               outline=(230, 25, 75, 255))
        img = Image.alpha_composite(img, ov).convert("RGB")
        dr = ImageDraw.Draw(img)
        y = 10
        for s in slabs:
            area = s.get("area_m2")
            dr.text((10, y),
                    f"{s['label']}: area={area:.1f} m2" if area else s["label"],
                    fill=(0, 0, 0), font=_font(22))
            y += 28
        dr.text((10, y), f"attempts={attempts}", fill=(0, 0, 0), font=_font(22))
        return self._save(img, name)

    # ── composite prompt image (Round 1) ──────────────────────────────────
    def render_for_prompt(self, img_path: str) -> bytes:
        """Load a previously saved debug image as PNG bytes (byte-identical)."""
        return Path(img_path).read_bytes()
