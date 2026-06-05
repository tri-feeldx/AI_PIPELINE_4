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
