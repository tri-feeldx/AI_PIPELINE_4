import re
import fitz  # pymupdf
from pathlib import Path
from PIL import Image
import io


FLOOR_PLAN_KEYWORDS = [
    "FLOOR PLAN", "SLAB PLAN", "LEVEL PLAN", "OUTLINE PLAN",
    "SUSPENDED SLAB", "DIAPHRAGM", "FRAMING PLAN", "STRUCTURAL PLAN",
    "LEVEL 0", "LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5",
    "GROUND FLOOR", "FIRST FLOOR", "SECOND FLOOR", "ROOF", "PODIUM",
]

FFL_PATTERN = re.compile(
    r"(?:FFL|RL|EL|FL|AHD|NGL)\s*[=:+]?\s*([+-]?\d{1,4}(?:\.\d{1,3})?)\s*(?:m|M|mAHD)?",
    re.IGNORECASE,
)
SCALE_PATTERN = re.compile(r"1\s*[:/]\s*(\d+)", re.IGNORECASE)
SLAB_LABEL_PATTERN = re.compile(
    r"\b([A-Z]?S\d+[A-Z]?)\b|\b(SL[\s-]?\d+)\b|\b([A-Z]-S\d+-[A-Z]{2}\d+)\b",
    re.IGNORECASE,
)


def load_pdf(path: str) -> fitz.Document:
    return fitz.open(path)


def get_pdf_metadata(doc: fitz.Document) -> dict:
    meta = doc.metadata or {}
    return {
        "page_count": doc.page_count,
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "page_size_pts": (doc[0].rect.width, doc[0].rect.height) if doc.page_count > 0 else (0, 0),
    }


def get_page_thumbnail(page: fitz.Page, dpi: int = 72) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def extract_text_blocks(page: fitz.Page) -> list[dict]:
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    result = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    result.append({
                        "text": text,
                        "bbox": span["bbox"],  # (x0, y0, x1, y1) in PDF points
                        "size": span.get("size", 0),
                    })
    return result


def classify_pages(doc: fitz.Document) -> list[dict]:
    """Return list of page info dicts with is_floor_plan flag."""
    pages = []
    for i, page in enumerate(doc):
        text_blocks = extract_text_blocks(page)
        full_text = " ".join(b["text"] for b in text_blocks).upper()

        is_floor_plan = any(kw in full_text for kw in FLOOR_PLAN_KEYWORDS)
        title = _extract_page_title(text_blocks)
        ffl_values = extract_ffl_values(text_blocks)
        scale = detect_scale_from_blocks(text_blocks)

        pages.append({
            "index": i,
            "label": f"Page {i + 1}: {title}" if title else f"Page {i + 1}",
            "title": title,
            "is_floor_plan": is_floor_plan,
            "ffl_values": ffl_values,
            "scale": scale,
        })
    return pages


def _extract_page_title(text_blocks: list[dict]) -> str:
    """Heuristic: largest font text near bottom of page = drawing title."""
    if not text_blocks:
        return ""
    # Sort by font size descending, take top candidates
    sorted_blocks = sorted(text_blocks, key=lambda b: b["size"], reverse=True)
    for b in sorted_blocks[:5]:
        t = b["text"].strip()
        if len(t) > 4 and not t.replace(".", "").replace("-", "").replace(" ", "").isdigit():
            return t[:80]
    return ""


def extract_ffl_values(text_blocks: list[dict]) -> list[dict]:
    """Parse FFL annotations like 'FFL 44.000' or 'FFL=40.700'."""
    results = []
    for block in text_blocks:
        matches = FFL_PATTERN.findall(block["text"])
        for match in matches:
            try:
                ffl_m = float(match)
                results.append({
                    "ffl_m": ffl_m,
                    "ffl_mm": ffl_m * 1000,
                    "bbox": block["bbox"],
                    "source_text": block["text"],
                })
            except ValueError:
                pass
    return results


def detect_scale_from_blocks(text_blocks: list[dict]) -> int | None:
    """Find scale ratio like '1:100' or '1:200' from text blocks."""
    for block in text_blocks:
        m = SCALE_PATTERN.search(block["text"])
        if m:
            try:
                scale = int(m.group(1))
                if 10 <= scale <= 2000:
                    return scale
            except ValueError:
                pass
    return None


def extract_slab_labels(text_blocks: list[dict]) -> list[dict]:
    """Find slab label annotations like S1, S2, B-S1-CW01."""
    results = []
    for block in text_blocks:
        matches = SLAB_LABEL_PATTERN.findall(block["text"])
        for match_groups in matches:
            label = next((g for g in match_groups if g), None)
            if label:
                results.append({
                    "label": label.strip(),
                    "bbox": block["bbox"],
                })
    return results
