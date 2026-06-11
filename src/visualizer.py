"""
Debug image generation at each processing step.
All images saved to debug_images/{session_id}/step_N_desc.png
"""

import io
import random
import math
from pathlib import Path
from typing import Optional

import fitz
from shapely.geometry import MultiPolygon as _MultiPolygon
from shapely.geometry import box as _box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np
from PIL import Image


STEP_COLORS = {
    "raw": "cyan",
    "polygon": "yellow",
    "filtered": "lime",
    "labeled": "orange",
    "final": "deepskyblue",
}

SLAB_PALETTE = [
    (0.2, 0.6, 1.0, 0.45),
    (0.2, 0.9, 0.5, 0.45),
    (1.0, 0.6, 0.2, 0.45),
    (0.9, 0.2, 0.9, 0.45),
    (0.2, 0.9, 0.9, 0.45),
    (1.0, 0.9, 0.2, 0.45),
    (0.8, 0.2, 0.2, 0.45),
    (0.5, 0.5, 1.0, 0.45),
]


def _page_to_image(page: fitz.Page, dpi: int = 120) -> np.ndarray:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)


def _get_exterior(poly):
    """Return exterior coords, handling MultiPolygon by taking largest part."""
    if isinstance(poly, _MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return list(poly.exterior.coords)


def _pdf_coords_to_img(pts, page: fitz.Page, dpi: int = 120):
    """Scale PDF point coordinates to image pixel coordinates."""
    scale = dpi / 72.0
    return [(x * scale, y * scale) for x, y in pts]


def save_step1_raw_paths(
    page: fitz.Page,
    drawings: list[dict],
    save_path: str,
    dpi: int = 120,
) -> str:
    """Step 1: Visualize all raw vector paths from the PDF page."""
    img = _page_to_image(page, dpi)
    h, w = img.shape[:2]
    scale = dpi / 72.0

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, origin="upper")

    for d in drawings:
        color = (random.random(), random.random(), random.random(), 0.7)
        for item in d.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                ax.plot(
                    [p1.x * scale, p2.x * scale],
                    [p1.y * scale, p2.y * scale],
                    color=color, linewidth=0.5
                )
            elif item[0] == "re":
                r = item[1]
                rect = plt.Rectangle(
                    (r.x0 * scale, r.y0 * scale),
                    (r.x1 - r.x0) * scale, (r.y1 - r.y0) * scale,
                    fill=False, edgecolor=color, linewidth=0.5
                )
                ax.add_patch(rect)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"Step 1: Raw Paths — {len(drawings)} drawing items", fontsize=8, color="white",
                 backgroundcolor="black")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return save_path


def save_step2_polygons(
    page: fitz.Page,
    polygons: list,
    save_path: str,
    title: str = "Step 2: All Closed Polygons",
    dpi: int = 120,
) -> str:
    """Step 2: Show all reconstructed closed polygons."""
    img = _page_to_image(page, dpi)
    h, w = img.shape[:2]
    scale = dpi / 72.0

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, origin="upper")

    for i, poly in enumerate(polygons):
        color = SLAB_PALETTE[i % len(SLAB_PALETTE)]
        pts = _pdf_coords_to_img(_get_exterior(poly), page, dpi)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.fill(xs, ys, alpha=color[3], color=color[:3])
        ax.plot(xs, ys, color="white", linewidth=0.6, alpha=0.8)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"{title} — {len(polygons)} found", fontsize=8,
                 color="white", backgroundcolor="black")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return save_path


def save_step3_filtered(
    page: fitz.Page,
    candidates: list,
    save_path: str,
    dpi: int = 120,
) -> str:
    """Step 3: Filtered slab candidate polygons."""
    return save_step2_polygons(
        page, candidates, save_path,
        title="Step 3: Filtered Slab Candidates", dpi=dpi
    )


def _draw_poly(ax, poly, page: fitz.Page, dpi: int, facecolor=None, edgecolor="white",
               alpha: float = 0.35, linewidth: float = 1.0, linestyle: str = "-"):
    for part in ([poly] if not isinstance(poly, _MultiPolygon) else list(poly.geoms)):
        pts = _pdf_coords_to_img(list(part.exterior.coords), page, dpi)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if facecolor is not None:
            ax.fill(xs, ys, alpha=alpha, color=facecolor)
        ax.plot(xs, ys, color=edgecolor, linewidth=linewidth, linestyle=linestyle)
        for interior in getattr(part, "interiors", []):
            hole_pts = _pdf_coords_to_img(list(interior.coords), page, dpi)
            hx = [p[0] for p in hole_pts]
            hy = [p[1] for p in hole_pts]
            ax.plot(hx, hy, color=edgecolor, linewidth=linewidth, linestyle=linestyle)


def save_gross_net_slab_debug(
    page: fitz.Page,
    extraction_result,
    save_path: str,
    dpi: int = 150,
) -> str:
    """Overlay gross slab, recovered appendages, void candidates, and ignored regions."""
    img = _page_to_image(page, dpi)
    h, w = img.shape[:2]

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, origin="upper")

    for item in getattr(extraction_result, "ignored_regions", [])[:80]:
        poly = item.get("polygon")
        if poly is not None and not poly.is_empty:
            _draw_poly(ax, poly, page, dpi, facecolor=None, edgecolor="#78909C",
                       alpha=0.12, linewidth=0.5, linestyle=":")

    for poly in getattr(extraction_result, "gross_slabs", []):
        _draw_poly(ax, poly, page, dpi, facecolor="#7ED957", edgecolor="#00C853",
                   alpha=0.22, linewidth=1.4)

    for poly in getattr(extraction_result, "appendages", []):
        _draw_poly(ax, poly, page, dpi, facecolor=None, edgecolor="#B2FF59",
                   alpha=0.4, linewidth=2.2)

    for candidate in getattr(extraction_result, "void_candidates", []):
        poly = candidate.get("polygon")
        if poly is None or poly.is_empty:
            continue
        auto_cut = candidate.get("auto_cut")
        color = "#FF1744" if auto_cut else "#FF9100"
        _draw_poly(ax, poly, page, dpi, facecolor=color, edgecolor=color,
                   alpha=0.28 if auto_cut else 0.20, linewidth=1.4)
        try:
            cx = poly.centroid.x * dpi / 72.0
            cy = poly.centroid.y * dpi / 72.0
            label = "CUT" if auto_cut else "REVIEW"
            reason = candidate.get("reason", "void")
            conf = candidate.get("confidence", 0)
            ax.text(
                cx, cy, f"{label}\n{reason}\n{conf:.2f}",
                ha="center", va="center", fontsize=5, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.65, edgecolor=color),
            )
        except Exception:
            pass

    handles = [
        mpatches.Patch(color="#7ED957", alpha=0.35, label="gross slab"),
        mpatches.Patch(color="#B2FF59", alpha=0.65, label="recovered appendage"),
        mpatches.Patch(color="#FF1744", alpha=0.45, label="auto subtract"),
        mpatches.Patch(color="#FF9100", alpha=0.45, label="review candidate"),
        mpatches.Patch(color="#78909C", alpha=0.35, label="ignored"),
    ]

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    debug = getattr(extraction_result, "debug", {}) or {}
    title = (
        "Gross -> Net Slab | "
        f"gross={len(getattr(extraction_result, 'gross_slabs', []))}, "
        f"appendages={debug.get('appendages', 0)}, "
        f"voids={len(getattr(extraction_result, 'void_candidates', []))}"
    )
    ax.set_title(title, fontsize=8, color="white", backgroundcolor="black")
    ax.legend(handles=handles, loc="lower right", fontsize=5, framealpha=0.75)
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return save_path


def save_step4_labeled(
    page: fitz.Page,
    slab_regions: list,
    save_path: str,
    dpi: int = 120,
) -> str:
    """Step 4: Labeled slabs with label + FFL overlay."""
    img = _page_to_image(page, dpi)
    h, w = img.shape[:2]
    scale = dpi / 72.0

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, origin="upper")

    for i, slab in enumerate(slab_regions):
        color = SLAB_PALETTE[i % len(SLAB_PALETTE)]
        pts = _pdf_coords_to_img(_get_exterior(slab.polygon), page, dpi)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.fill(xs, ys, alpha=0.4, color=color[:3])
        ax.plot(xs, ys, color="white", linewidth=1.0)

        # Label at centroid
        cx, cy = slab.polygon.centroid.x * scale, slab.polygon.centroid.y * scale
        label_text = slab.label
        if slab.ffl_m is not None:
            label_text += f"\nFFL {slab.ffl_m:.3f}m"
        ax.text(
            cx, cy, label_text,
            ha="center", va="center", fontsize=6, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"),
        )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"Step 4: Labeled Slabs — {len(slab_regions)} detected",
                 fontsize=8, color="white", backgroundcolor="black")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return save_path


def save_step5_final(
    page: fitz.Page,
    slab_regions: list,
    save_path: str,
    dpi: int = 150,
) -> str:
    """Step 5: Final review image with full annotations."""
    img = _page_to_image(page, dpi)
    h, w = img.shape[:2]
    scale = dpi / 72.0

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, origin="upper")

    legend_handles = []
    for i, slab in enumerate(slab_regions):
        color = SLAB_PALETTE[i % len(SLAB_PALETTE)]
        pts = _pdf_coords_to_img(_get_exterior(slab.polygon), page, dpi)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        ax.fill(xs, ys, alpha=0.35, color=color[:3])
        ax.plot(xs, ys, color=color[:3], linewidth=1.5)

        cx, cy = slab.polygon.centroid.x * scale, slab.polygon.centroid.y * scale
        area_txt = f"{slab.area_m2:.1f}m²" if slab.area_m2 > 0 else ""
        ffl_txt = f"FFL {slab.ffl_m:.3f}m" if slab.ffl_m is not None else "FFL ?"
        full_label = f"{slab.label}\n{ffl_txt}\nt=200mm\n{area_txt}"

        ax.text(
            cx, cy, full_label,
            ha="center", va="center", fontsize=5.5, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#003366", alpha=0.75, edgecolor=color[:3], linewidth=0.8),
        )
        legend_handles.append(mpatches.Patch(color=color[:3], label=f"{slab.label} ({area_txt})"))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"Final: {len(slab_regions)} Slabs Detected | Thickness: 200mm",
                 fontsize=9, color="white", backgroundcolor="#003366", pad=6)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=5,
            framealpha=0.7,
            facecolor="#001133",
            labelcolor="white",
            ncol=min(4, len(legend_handles)),
        )

    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="#001133")
    plt.close(fig)
    return save_path


def get_session_dir(base_dir: str, session_id: str) -> Path:
    d = Path(base_dir) / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def image_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _save_element_polygons(page: fitz.Page, elements: list, save_path: str, title: str,
                           color: str, dpi: int = 150) -> str:
    img = _page_to_image(page, dpi)
    h, w = img.shape[:2]
    scale = dpi / 72.0
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, origin="upper")

    for elem in elements:
        poly = getattr(elem, "polygon", None)
        if poly is None or poly.is_empty:
            continue
        _draw_poly(ax, poly, page, dpi, facecolor=color, edgecolor=color, alpha=0.25, linewidth=1.4)
        try:
            cx, cy = poly.centroid.x * scale, poly.centroid.y * scale
            label = getattr(elem, "symbol", "?")
            conf = getattr(elem, "detection_confidence", 0.0)
            ax.text(
                cx, cy, f"{label}\n{conf:.2f}",
                ha="center", va="center", fontsize=5.5, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.65, edgecolor=color),
            )
        except Exception:
            pass

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"{title}: {len(elements)} detected", fontsize=8, color="white", backgroundcolor="black")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return save_path


def save_column_polygons(page: fitz.Page, columns: list, save_path: str, dpi: int = 150) -> str:
    return _save_element_polygons(page, columns, save_path, "Columns", "#D500F9", dpi=dpi)


def save_foundation_polygons(page: fitz.Page, foundations: list, save_path: str, dpi: int = 150) -> str:
    return _save_element_polygons(page, foundations, save_path, "Foundations", "#795548", dpi=dpi)


def save_building_footprints(registry: dict, save_path: str, dpi: int = 140) -> str:
    """Save a native-coordinate building footprint preview in real-world mm."""
    buildings = list((registry or {}).get("buildings", {}).values())
    polys = [
        p
        for b in buildings
        for p in (b.get("footprint_parts") or [])
        if p is not None and not p.is_empty
    ]
    if not polys:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.text(0.5, 0.5, "No building footprint polygons", ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return save_path

    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)
    width = max(maxx - minx, 1.0)
    depth = max(maxy - miny, 1.0)
    fig_w = min(max(width / 2500.0, 7), 16)
    fig_h = min(max(depth / 2500.0, 5), 12)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    colors = [
        "#00C853", "#2962FF", "#FF6D00", "#D500F9",
        "#00B8D4", "#FFD600", "#C51162", "#64DD17",
    ]
    for idx, b in enumerate(buildings):
        color = colors[idx % len(colors)]
        for poly in b.get("footprint_parts") or []:
            if poly is None or poly.is_empty:
                continue
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, facecolor=color, edgecolor=color, alpha=0.22, linewidth=2.0)
            for interior in getattr(poly, "interiors", []):
                hx, hy = interior.xy
                ax.fill(hx, hy, facecolor="white", edgecolor=color, alpha=1.0, linewidth=1.0)
        fp = b.get("footprint_polygon")
        if fp is not None and not fp.is_empty:
            c = fp.centroid
            ax.text(
                c.x, c.y,
                f"{b.get('name')}\n{len(b.get('levels', {}))} floors\n{b.get('area_m2', 0):.1f} m2",
                ha="center", va="center", fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.72, edgecolor=color),
            )

    ax.set_aspect("equal", adjustable="box")
    pad_x = width * 0.04
    pad_y = depth * 0.04
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(maxy + pad_y, miny - pad_y)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_title("Building Footprints (native PDF coordinates, mm)", fontsize=10)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


def save_boundary_first_debug(page: fitz.Page, boundary_result, save_path: str, dpi: int = 150) -> str:
    """Overlay wall-guided boundary evidence on the PDF page."""
    img = _page_to_image(page, dpi)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, origin="upper")

    for poly in getattr(boundary_result, "gross_regions", []) or []:
        _draw_poly(ax, poly, page, dpi, facecolor="#00C853", edgecolor="#00C853", alpha=0.22, linewidth=2.0)
    for poly in getattr(boundary_result, "wall_core_candidates", []) or []:
        _draw_poly(ax, poly, page, dpi, facecolor="#00BCD4", edgecolor="#00BCD4", alpha=0.16, linewidth=0.9)
    for poly in getattr(boundary_result, "boundary_evidence", []) or []:
        _draw_poly(ax, poly, page, dpi, facecolor="none", edgecolor="#2962FF", alpha=0.95, linewidth=1.4)
    structural = getattr(boundary_result, "structural_objects", None)
    if structural:
        for obj in getattr(structural, "walls", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="#00BCD4", edgecolor="#00BCD4", alpha=0.18, linewidth=1.0)
        for obj in getattr(structural, "cut_candidates", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="#F44336", edgecolor="#F44336", alpha=0.20, linewidth=1.4)
            if obj.label:
                c = obj.polygon.centroid
                ax.text(c.x * dpi / 72, c.y * dpi / 72, obj.kind, color="white", fontsize=6,
                        bbox=dict(facecolor="#B71C1C", alpha=0.75, pad=1, edgecolor="none"))
        for obj in getattr(structural, "uncertain_regions", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="#FF9800", edgecolor="#FF9800", alpha=0.14, linewidth=0.9)
        for obj in getattr(structural, "ignored_regions", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="#9E9E9E", edgecolor="#9E9E9E", alpha=0.10, linewidth=0.6)
    for poly in getattr(boundary_result, "grid_column_anchors", []) or []:
        _draw_poly(ax, poly, page, dpi, facecolor="#2962FF", edgecolor="#2962FF", alpha=0.18, linewidth=0.8)
    for poly in getattr(boundary_result, "uncertain_candidates", []) or []:
        _draw_poly(ax, poly, page, dpi, facecolor="#FFAB00", edgecolor="#FFAB00", alpha=0.13, linewidth=0.8)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(
        f"Wall/Boundary: {len(getattr(boundary_result, 'final_regions', []) or [])} slab candidates | "
        f"signatures={len(getattr(boundary_result, 'boundary_signatures', []) or [])} | "
        f"confidence={getattr(boundary_result, 'confidence', 0.0):.2f}",
        fontsize=8,
        color="white",
        backgroundcolor="black",
    )
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return save_path


def _label_poly(ax, poly, page: fitz.Page, dpi: int, text: str, color: str) -> None:
    try:
        c = poly.centroid
        ax.text(
            c.x * dpi / 72,
            c.y * dpi / 72,
            text,
            color="white",
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(facecolor=color, alpha=0.78, pad=1, edgecolor="none"),
        )
    except Exception:
        pass


def _base_wall_fig(page: fitz.Page, dpi: int):
    img = _page_to_image(page, dpi)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img, origin="upper")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def _finish_wall_fig(fig, ax, title: str, save_path: str) -> str:
    ax.set_title(title, fontsize=8, color="white", backgroundcolor="black")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return save_path


def save_wall_evidence_only(page: fitz.Page, boundary_result, save_path: str, dpi: int = 170) -> str:
    """Show only structural boundary evidence, without slab fills hiding it."""
    fig, ax = _base_wall_fig(page, dpi)
    structural = getattr(boundary_result, "structural_objects", None)
    wall_count = 0
    if structural:
        for obj in getattr(structural, "ignored_regions", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="#9E9E9E", edgecolor="#616161", alpha=0.18, linewidth=1.0)
            _label_poly(ax, obj.polygon, page, dpi, "ignored", "#616161")
        for obj in getattr(structural, "walls", []) or []:
            wall_count += 1
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#00E5FF", alpha=1.0, linewidth=2.0)
            _label_poly(ax, obj.polygon, page, dpi, "wall", "#00838F")
        for obj in getattr(structural, "load_bearing_elements", []) or []:
            wall_count += 1
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#00BFA5", alpha=1.0, linewidth=2.2)
            _label_poly(ax, obj.polygon, page, dpi, "load", "#00796B")
        for obj in getattr(structural, "cores", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#AA00FF", alpha=1.0, linewidth=2.2)
            _label_poly(ax, obj.polygon, page, dpi, "core", "#6A1B9A")
        for obj in getattr(structural, "columns_or_piles", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#607D8B", alpha=0.9, linewidth=1.6)
            _label_poly(ax, obj.polygon, page, dpi, "exclude", "#455A64")
        for obj in getattr(structural, "footings", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#795548", alpha=0.9, linewidth=1.6)
            _label_poly(ax, obj.polygon, page, dpi, "footing", "#5D4037")
        for obj in getattr(structural, "stairs", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#F44336", alpha=1.0, linewidth=2.2)
            _label_poly(ax, obj.polygon, page, dpi, "stair", "#B71C1C")
        for obj in getattr(structural, "openings", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#FF5722", alpha=1.0, linewidth=2.2)
            _label_poly(ax, obj.polygon, page, dpi, "opening", "#BF360C")
        for obj in getattr(structural, "penetrations", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#FF1744", alpha=1.0, linewidth=2.2)
            _label_poly(ax, obj.polygon, page, dpi, "pen", "#B71C1C")
    boundary_count = 0
    for poly in getattr(boundary_result, "boundary_evidence", []) or []:
        boundary_count += 1
        _draw_poly(ax, poly, page, dpi, facecolor="none", edgecolor="#2962FF", alpha=0.95, linewidth=1.8)
    sig_count = len(getattr(boundary_result, "boundary_signatures", []) or [])
    return _finish_wall_fig(
        fig,
        ax,
        f"Wall/Core First | walls+load={wall_count} | signatures={sig_count} | evidence={boundary_count}",
        save_path,
    )


def save_slab_candidates_only(page: fitz.Page, boundary_result, save_path: str, dpi: int = 150) -> str:
    """Show gross/uncertain polygon candidates separately from wall evidence."""
    fig, ax = _base_wall_fig(page, dpi)
    for i, poly in enumerate(getattr(boundary_result, "gross_regions", []) or [], start=1):
        color = SLAB_PALETTE[(i - 1) % len(SLAB_PALETTE)]
        _draw_poly(ax, poly, page, dpi, facecolor=color, edgecolor="#00C853", alpha=0.28, linewidth=2.0)
        _label_poly(ax, poly, page, dpi, f"keep {i}", "#00A152")
    for poly in getattr(boundary_result, "uncertain_candidates", []) or []:
        _draw_poly(ax, poly, page, dpi, facecolor="#FFAB00", edgecolor="#FFAB00", alpha=0.12, linewidth=1.2)
    return _finish_wall_fig(
        fig,
        ax,
        f"Slab Candidates Only | kept={len(getattr(boundary_result, 'gross_regions', []) or [])}",
        save_path,
    )


def save_wall_guided_final(page: fitz.Page, boundary_result, save_path: str, dpi: int = 150) -> str:
    """Show final wall-guided slab result with cuts, but lighter than combined debug."""
    fig, ax = _base_wall_fig(page, dpi)
    for i, poly in enumerate(getattr(boundary_result, "final_regions", []) or [], start=1):
        _draw_poly(ax, poly, page, dpi, facecolor="#00C853", edgecolor="#00C853", alpha=0.20, linewidth=2.4)
        _label_poly(ax, poly, page, dpi, f"final {i}", "#00A152")
    structural = getattr(boundary_result, "structural_objects", None)
    if structural:
        for obj in getattr(structural, "walls", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#00E5FF", alpha=0.95, linewidth=1.8)
        for obj in getattr(structural, "load_bearing_elements", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#00BFA5", alpha=0.95, linewidth=1.8)
        for obj in getattr(structural, "columns_or_piles", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#607D8B", alpha=0.65, linewidth=1.0)
        for obj in getattr(structural, "footings", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#795548", alpha=0.65, linewidth=1.0)
        for obj in getattr(structural, "cut_candidates", []) or []:
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="#F44336", edgecolor="#F44336", alpha=0.18, linewidth=2.0)
            _label_poly(ax, obj.polygon, page, dpi, obj.kind, "#B71C1C")
    return _finish_wall_fig(
        fig,
        ax,
        f"Final Wall-Guided Result | final={len(getattr(boundary_result, 'final_regions', []) or [])}",
        save_path,
    )


def save_interior_resolver_debug(page: fitz.Page, boundary_result, save_path: str, dpi: int = 150) -> str:
    """Show inside/outside evidence used to choose no-fill slab interiors."""
    fig, ax = _base_wall_fig(page, dpi)
    resolution = getattr(boundary_result, "interior_resolution", None)
    if resolution:
        for mask in getattr(resolution, "outside_masks", []) or []:
            poly = getattr(mask, "polygon", None)
            if poly is not None and not poly.is_empty:
                _draw_poly(ax, poly, page, dpi, facecolor="#F44336", edgecolor="#B71C1C", alpha=0.16, linewidth=1.4)
                label = getattr(mask, "kind", "outside")
                _label_poly(ax, poly, page, dpi, label[:16], "#B71C1C")
        for decision in getattr(resolution, "rejected_candidates", []) or []:
            poly = getattr(decision, "polygon", None)
            if poly is not None and not poly.is_empty:
                _draw_poly(ax, poly, page, dpi, facecolor="#FFD600", edgecolor="#FF8F00", alpha=0.10, linewidth=1.4)
        for poly in getattr(resolution, "selected_inside_slabs", []) or []:
            if poly is not None and not poly.is_empty:
                _draw_poly(ax, poly, page, dpi, facecolor="#00C853", edgecolor="#00C853", alpha=0.24, linewidth=2.8)
                _label_poly(ax, poly, page, dpi, "inside slab", "#00A152")
        for seed in getattr(resolution, "inside_seeds", []) or []:
            poly = getattr(seed, "polygon", None)
            if poly is not None and not poly.is_empty:
                _draw_poly(ax, poly, page, dpi, facecolor="#2962FF", edgecolor="#0039CB", alpha=0.22, linewidth=1.2)
                label = getattr(seed, "kind", "seed")
                _label_poly(ax, poly, page, dpi, label[:14], "#0039CB")
    for poly in getattr(boundary_result, "boundary_evidence", []) or []:
        _draw_poly(ax, poly, page, dpi, facecolor="none", edgecolor="#00E5FF", alpha=0.75, linewidth=1.1)
    debug = getattr(boundary_result, "debug", {}) or {}
    return _finish_wall_fig(
        fig,
        ax,
        "Interior Resolver | "
        f"selected={debug.get('interior_selected_count', 0)} "
        f"rejected={debug.get('interior_rejected_count', 0)} "
        f"seeds={debug.get('interior_seed_count', 0)} "
        f"masks={debug.get('interior_outside_mask_count', 0)} "
        f"conf={debug.get('interior_confidence', 0):.2f}",
        save_path,
    )


def _legend_bbox(candidate) -> fitz.Rect:
    if hasattr(candidate, "bbox"):
        bbox = candidate.bbox
    else:
        bbox = candidate.get("bbox", [])
    return fitz.Rect(*bbox)


def save_legend_overlay(page: fitz.Page, candidates: list, save_path: str, dpi: int = 140) -> str:
    """Show detected legend crop regions on the full page."""
    fig, ax = _base_wall_fig(page, dpi)
    scale = dpi / 72.0
    colors = {"left": "#00B0FF", "right": "#FF6D00"}
    for cand in candidates or []:
        rect = _legend_bbox(cand)
        side = getattr(cand, "side", None) or cand.get("side", "legend")
        conf = getattr(cand, "confidence", None) or cand.get("confidence", 0)
        color = colors.get(side, "#00E676")
        patch = plt.Rectangle(
            (rect.x0 * scale, rect.y0 * scale),
            rect.width * scale,
            rect.height * scale,
            fill=False,
            edgecolor=color,
            linewidth=3.0,
        )
        ax.add_patch(patch)
        ax.text(
            rect.x0 * scale,
            max(0, rect.y0 * scale - 4),
            f"legend {side} {float(conf):.2f}",
            color="white",
            fontsize=7,
            bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"),
        )
    return _finish_wall_fig(fig, ax, f"Legend Crop Overlay | candidates={len(candidates or [])}", save_path)


def save_legend_crop(page: fitz.Page, bbox, save_path: str, dpi: int = 220) -> str:
    """Save a high-resolution crop of the detected legend region."""
    rect = fitz.Rect(*bbox)
    rect = fitz.Rect(
        max(page.rect.x0, rect.x0),
        max(page.rect.y0, rect.y0),
        min(page.rect.x1, rect.x1),
        min(page.rect.y1, rect.y1),
    )
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)
    return save_path


def save_semantic_wall_overlay(page: fitz.Page, structural_result, save_path: str, dpi: int = 170) -> str:
    """Show wall evidence after applying Gemini legend semantic rules."""
    fig, ax = _base_wall_fig(page, dpi)
    semantic_walls = 0
    geometry_walls = 0
    for obj in getattr(structural_result, "ignored_regions", []) or []:
        _draw_poly(ax, obj.polygon, page, dpi, facecolor="#9E9E9E", edgecolor="#616161", alpha=0.12, linewidth=0.8)
    for obj in getattr(structural_result, "walls", []) or []:
        source = str(getattr(obj, "source", ""))
        if source.startswith("legend_semantic") or source == "wall_label":
            semantic_walls += 1
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="#FF00FF", edgecolor="#FF00FF", alpha=0.18, linewidth=2.4)
            _label_poly(ax, obj.polygon, page, dpi, "wall label", "#AD1457")
        else:
            geometry_walls += 1
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="none", edgecolor="#00E5FF", alpha=0.95, linewidth=1.5)
    for obj in getattr(structural_result, "uncertain_regions", []) or []:
        if getattr(obj, "kind", "") == "slab_boundary_evidence":
            _draw_poly(ax, obj.polygon, page, dpi, facecolor="#FFAB00", edgecolor="#FFAB00", alpha=0.18, linewidth=2.0)
            _label_poly(ax, obj.polygon, page, dpi, "slab cue", "#E65100")
    for obj in getattr(structural_result, "cut_candidates", []) or []:
        _draw_poly(ax, obj.polygon, page, dpi, facecolor="#F44336", edgecolor="#F44336", alpha=0.20, linewidth=2.2)
        _label_poly(ax, obj.polygon, page, dpi, obj.kind, "#B71C1C")
    return _finish_wall_fig(
        fig,
        ax,
        f"Semantic Wall Overlay | semantic={semantic_walls} geometry={geometry_walls}",
        save_path,
    )


def save_wall_polygons(page: fitz.Page, walls: list, save_path: str, dpi: int = 170) -> str:
    """Show only wall polygons that will be exported to the model."""
    fig, ax = _base_wall_fig(page, dpi)
    for i, wall in enumerate(walls or [], start=1):
        poly = getattr(wall, "polygon", None)
        if poly is None or poly.is_empty:
            continue
        _draw_poly(ax, poly, page, dpi, facecolor="#8E24AA", edgecolor="#D500F9", alpha=0.22, linewidth=2.6)
        label = getattr(wall, "label", "") or f"wall {i}"
        _label_poly(ax, poly, page, dpi, label[:18], "#6A1B9A")
    return _finish_wall_fig(fig, ax, f"Wall Model Polygons | walls={len(walls or [])}", save_path)


def _draw_semantic_objects(ax, objects: list, page: fitz.Page, dpi: int,
                           facecolor: str, edgecolor: str, label_prefix: str) -> int:
    count = 0
    for obj in objects or []:
        poly = getattr(obj, "polygon", None)
        if poly is None or poly.is_empty:
            continue
        count += 1
        _draw_poly(ax, poly, page, dpi, facecolor=facecolor, edgecolor=edgecolor, alpha=0.22, linewidth=2.2)
        label = getattr(obj, "label", "") or label_prefix
        _label_poly(ax, poly, page, dpi, label[:18], edgecolor)
    return count


def save_slab_semantic_surface(page: fitz.Page, preview, save_path: str, dpi: int = 150) -> str:
    fig, ax = _base_wall_fig(page, dpi)
    n = _draw_semantic_objects(
        ax,
        getattr(preview, "surface_regions", []) or [],
        page,
        dpi,
        "#00C853",
        "#00A152",
        "slab surface",
    )
    fallback = getattr(preview, "fallback_policy", "unknown")
    source = getattr(preview, "effective_surface_source", "unknown")
    return _finish_wall_fig(
        fig,
        ax,
        f"Slab Surface Evidence | regions={n} | source={source} | fallback={fallback}",
        save_path,
    )


def save_slab_semantic_boundary_cues(page: fitz.Page, preview, save_path: str, dpi: int = 150) -> str:
    fig, ax = _base_wall_fig(page, dpi)
    n = _draw_semantic_objects(
        ax,
        getattr(preview, "boundary_cues", []) or [],
        page,
        dpi,
        "#FFAB00",
        "#E65100",
        "slab cue",
    )
    return _finish_wall_fig(fig, ax, f"Slab Boundary Cues | cues={n} | no auto-cut", save_path)


def save_slab_semantic_cut_candidates(page: fitz.Page, preview, save_path: str, dpi: int = 150) -> str:
    fig, ax = _base_wall_fig(page, dpi)
    n = _draw_semantic_objects(
        ax,
        getattr(preview, "cut_candidates", []) or [],
        page,
        dpi,
        "#F44336",
        "#B71C1C",
        "slab cut",
    )
    return _finish_wall_fig(fig, ax, f"Slab Cut Candidates | cuts={n}", save_path)


def save_line_semantic_policy_overlay(
    page: fitz.Page,
    page_catalog: dict,
    style_rules: list[dict],
    save_path: str,
    dpi: int = 150,
) -> str:
    """Show Gemini line-style policies using compact catalog sample bboxes."""
    fig, ax = _base_wall_fig(page, dpi)
    rule_by_style = {
        str(r.get("style_id") or r.get("style_key")): r
        for r in (style_rules or [])
        if isinstance(r, dict) and (r.get("style_id") or r.get("style_key"))
    }
    colors = {
        "building_boundary": "#00C853",
        "slab_edge": "#64DD17",
        "wall": "#00BCD4",
        "site_boundary": "#D50000",
        "grid": "#9E9E9E",
        "dimension": "#757575",
        "annotation": "#607D8B",
        "joint": "#FF9800",
        "reference": "#795548",
        "unknown": "#FFD600",
    }
    counts: dict[str, int] = {}
    for style in page_catalog.get("line_styles", []) or []:
        style_id = str(style.get("style_id"))
        rule = rule_by_style.get(style_id, {})
        semantic = str(rule.get("semantic") or "unknown")
        color = colors.get(semantic, colors["unknown"])
        counts[semantic] = counts.get(semantic, 0) + 1
        for bbox in style.get("sample_bboxes", [])[:5]:
            try:
                poly = _box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])).buffer(4)
                _draw_poly(ax, poly, page, dpi, facecolor=None, edgecolor=color, alpha=0.20, linewidth=2.0)
            except Exception:
                continue
        if style.get("sample_bboxes"):
            try:
                bbox = style["sample_bboxes"][0]
                poly = _box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                cx = poly.centroid.x * dpi / 72.0
                cy = poly.centroid.y * dpi / 72.0
                use_for = ",".join(rule.get("use_for", []) or [])
                conf = float(rule.get("confidence") or 0)
                ax.text(
                    cx, cy, f"{semantic}\n{use_for}\n{conf:.2f}",
                    ha="center", va="center", fontsize=5, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.65, edgecolor=color),
                )
            except Exception:
                pass
    handles = [
        mpatches.Patch(color=color, alpha=0.55, label=f"{semantic} ({counts.get(semantic, 0)})")
        for semantic, color in colors.items()
        if counts.get(semantic, 0)
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=5, framealpha=0.75)
    return _finish_wall_fig(
        fig,
        ax,
        f"Line Semantic Policy | styles={len(page_catalog.get('line_styles', []) or [])}",
        save_path,
    )


def save_floor_alignment_preview(rows: list[dict], save_path: str, dpi: int = 140) -> str:
    """Create a compact text/table preview for floor alignment offsets."""
    fig, ax = plt.subplots(figsize=(10, max(3, 0.38 * max(len(rows), 1) + 1.5)), dpi=dpi)
    ax.axis("off")
    ax.set_title("Floor Alignment Report", fontsize=12, weight="bold")
    if not rows:
        ax.text(0.5, 0.5, "No floor alignment rows", ha="center", va="center")
    else:
        cols = ["Building", "Page", "Reference", "dx_mm", "dy_mm", "Confidence", "Applied", "Warning"]
        table_data = [[str(r.get(c, "")) for c in cols] for r in rows]
        table = ax.table(cellText=table_data, colLabels=cols, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path
