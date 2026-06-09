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
