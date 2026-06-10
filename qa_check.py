"""
Quick QA slab extraction preview — no Streamlit needed.

Usage:
  python qa_check.py <pdf_path> <page> [scale] [output.png]

  pdf_path : path to the structural PDF
  page     : 1-indexed page number to inspect
  scale    : drawing scale integer (e.g. 100 for 1:100). Omit to auto-detect.
  output   : output PNG path. Default: <pdf_stem>_qa_p<page>.png next to the PDF

Examples:
  python qa_check.py structural.pdf 8
  python qa_check.py structural.pdf 8 100
  python qa_check.py "D:/drawings/Level01.pdf" 1 100 output/qa_level01.png
"""

import os
import sys
import tempfile
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.pdf_processor import extract_text_blocks, extract_ffl_values, extract_slab_labels, detect_scale_from_blocks
from src.slab_extractor import extract_slabs_from_page
from src.coordinate_mapper import transform_all_slabs
from src.visualizer import _page_to_image, save_step5_final
from src.pipeline_logger import get_logger


def _build_output_path(pdf_path: str, page_num: int, given_output: str | None) -> str:
    if given_output:
        Path(given_output).parent.mkdir(parents=True, exist_ok=True)
        return given_output
    stem = Path(pdf_path).stem
    out_dir = Path(pdf_path).parent
    return str(out_dir / f"{stem}_qa_p{page_num}.png")


def qa_check(pdf_path: str, page_num: int, scale: int | None = None, output_path: str | None = None) -> str:
    """
    Run slab extraction on one page and save a side-by-side QA image.

    Returns the path to the saved PNG.
    """
    # ── 1. Open PDF ────────────────────────────────────────────────────────────
    if not Path(pdf_path).exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    if page_num < 1 or page_num > doc.page_count:
        print(f"ERROR: Page {page_num} out of range (PDF has {doc.page_count} pages)")
        doc.close()
        sys.exit(1)

    page = doc[page_num - 1]

    # ── 2. Extract text signals ────────────────────────────────────────────────
    text_blocks = extract_text_blocks(page)
    ffl_values  = extract_ffl_values(text_blocks)
    slab_labels = extract_slab_labels(text_blocks)

    # Scale: use provided, else auto-detect, else default 100
    if scale is None:
        detected = detect_scale_from_blocks(text_blocks)
        if detected:
            scale = detected
            print(f"  Scale auto-detected: 1:{scale}")
        else:
            scale = 100
            print(f"  Scale not found in text — using default 1:{scale}")
    else:
        print(f"  Scale: 1:{scale}")

    # ── 3. Extract + transform slabs ──────────────────────────────────────────
    slab_regions, _ = extract_slabs_from_page(page, text_blocks, ffl_values, slab_labels)
    slab_regions     = transform_all_slabs(slab_regions, page, scale)

    # ── 4. Render original PDF page ───────────────────────────────────────────
    dpi = 150
    img_orig = _page_to_image(page, dpi=dpi)

    # ── 5. Generate overlay (step5) to temp file ──────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    save_step5_final(page, slab_regions, tmp_path, dpi=dpi)
    _overlay_pil = Image.open(tmp_path).convert("RGB")
    img_overlay = np.array(_overlay_pil)
    os.unlink(tmp_path)

    # ── 6. Combine side by side ───────────────────────────────────────────────
    # Ensure both images have the same height for clean concat
    h1, w1 = img_orig.shape[:2]
    h2, w2 = img_overlay.shape[:2]
    target_h = max(h1, h2)

    def _pad_height(img: np.ndarray, target: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h >= target:
            return img
        pad = np.full((target - h, w, 3), 20, dtype=np.uint8)
        return np.vstack([img, pad])

    img_orig    = _pad_height(img_orig, target_h)
    img_overlay = _pad_height(img_overlay, target_h)

    # Separator bar (4px dark)
    sep = np.full((target_h, 4, 3), 40, dtype=np.uint8)
    combined = np.hstack([img_orig, sep, img_overlay])

    # ── 7. Annotate with matplotlib ───────────────────────────────────────────
    fig_w = combined.shape[1] / 100
    fig_h = combined.shape[0] / 100 + 0.6  # extra for suptitle

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(combined, origin="upper")
    ax.axis("off")

    # Column headers
    mid_orig    = w1 / 2
    mid_overlay = w1 + 4 + w2 / 2
    ax.text(mid_orig,    -12, "ORIGINAL PDF",       ha="center", va="bottom", fontsize=8,
            color="white", fontweight="bold", transform=ax.transData)
    ax.text(mid_overlay, -12, "DETECTED SLABS",     ha="center", va="bottom", fontsize=8,
            color="#4FC3F7", fontweight="bold", transform=ax.transData)

    # Page title from PDF
    from src.pdf_processor import _extract_page_title
    page_title = _extract_page_title(text_blocks) or f"Page {page_num}"
    n_ffl = len({s.ffl_m for s in slab_regions if s.ffl_m is not None})

    fig.suptitle(
        f"QA CHECK  |  {page_title}  |  Page {page_num}/{doc.page_count}"
        f"  |  Scale 1:{scale}  |  {len(slab_regions)} slab(s)  |  {n_ffl} FFL level(s)",
        fontsize=8, color="white", y=0.995,
        fontweight="bold", backgroundcolor="#001133",
    )
    fig.patch.set_facecolor("#001133")
    fig.tight_layout(pad=0.2)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    out = _build_output_path(pdf_path, page_num, output_path)
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor="#001133")
    plt.close(fig)
    doc.close()

    # ── 9. Console summary ────────────────────────────────────────────────────
    sep_line = "-" * 60
    print(f"\n{sep_line}")
    print(f"  Page {page_num} | Scale 1:{scale} | {len(slab_regions)} slab(s) detected")
    for s in slab_regions:
        ffl_str  = f"FFL={s.ffl_m:.3f}m" if s.ffl_m is not None else "FFL=?"
        area_str = f"{s.area_m2:.1f}m2"  if s.area_m2 > 0 else "area=?"
        print(f"    {s.label:8s}  {ffl_str:16s}  {area_str}")
    print(sep_line)
    print(f"  Saved: {out}")

    # ── 10. Auto-open on Windows ──────────────────────────────────────────────
    try:
        os.startfile(out)
    except Exception:
        pass

    return out


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    pdf_path   = args[0]
    page_num   = int(args[1]) if len(args) > 1 else 1
    scale      = int(args[2]) if len(args) > 2 and args[2].isdigit() else None
    output     = args[3] if len(args) > 3 else (args[2] if len(args) > 2 and not args[2].isdigit() else None)

    qa_check(pdf_path, page_num, scale, output)


if __name__ == "__main__":
    main()
