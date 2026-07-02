"""
Vision Refiner — Stage 4 of the slab extraction pipeline.

Sends 2 cropped images (drawing content + legend) to a Vision LLM
(Gemini via Vertex AI or GPT-4o via OpenAI) to trace the slab exterior
boundary and identify no-go zones. The result is snapped to PDF vector
endpoints and the no-go zones are subtracted via Shapely.

Public API:
    get_vision_client(backend)        -> (client, model_name)
    refine_page_slabs(slabs, page, client, model, backend, snap_r) -> list[SlabRegion]
"""

import base64
import io
import json
import math
import os
import re

import fitz
from PIL import Image
from shapely.geometry import Polygon as SPoly
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

DPI_CONTENT  = 200
DPI_LEGEND   = 150
SNAP_R       = 40

PROMPT_VISION = """You are a structural engineer analysing a floor plan.
You are given 2 images:
- Image 1: A structural floor plan (drawing content only, legend stripped)
- Image 2: The legend/key for this drawing (use it to understand line styles)

━━━ TASK 1 — NO-GO ZONES ━━━
Identify rectangular regions that must NOT be included in the slab polygon.
Including them would make the slab shape WRONG for THIS drawing.

Include ONLY these types:
  a) "REFER TO DRAWING XXXX" boxes — areas belonging to a DIFFERENT structural drawing.
     These have a dashed or chain-dot border and text "REFER TO DRAWING ST-XXX-XX"
     written horizontally or vertically inside.
  b) Dense text-only note blocks (no structural lines, just specification text paragraphs).
  c) Schedule/table grids (e.g. PILE CAP SCHEDULE, PAD FOOTING SCHEDULE).

Do NOT include: column grid lines, dimension strings, pile symbols, parking bay markings,
hatch patterns, or anything that is part of the slab drawing itself.

Return each as [x_min, y_min, x_max, y_max] normalised to Image 1.

━━━ TASK 2 — SLAB EXTERIOR BOUNDARY ━━━
Trace the outer perimeter of the main structural concrete slab on this drawing.
Rules:
  • Follow the THICK or CONTINUOUS boundary lines that define where the slab EDGE is.
  • The slab edge is usually a solid or thick polyline forming a closed shape.
  • Where a no-go zone touches the boundary, trace ALONG THE EDGE of that zone.
  • Do NOT trace the drawing frame / sheet border — that is the paper edge, not the slab.
  • Do NOT snap to column grid lines or dimension lines.
  • Include diagonal cuts, curved edges, and notched corners if present.
  • Return 8–30 corner points (more if the shape is complex).

━━━ OUTPUT ━━━
Return ONLY valid JSON (no markdown fences, no explanation):
{
  "no_go_zones": [[x1,y1,x2,y2], ...],
  "exterior": [[x,y], ...]
}

All coordinates normalised 0.0–1.0 from TOP-LEFT of Image 1.
"""


# ── Client factories ──────────────────────────────────────────────────────────

def get_vision_client(backend: str = "gemini") -> tuple:
    """Return (client, model_name) for the given backend."""
    if backend == "openai":
        return _get_openai_client()
    return _get_gemini_client()


def _get_gemini_client() -> tuple:
    from google import genai
    from google.oauth2 import service_account

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project    = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location   = os.environ.get("VERTEX_LOCATION", "us-central1")

    if not creds_path or not project:
        raise EnvironmentError(
            "Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT in .env"
        )
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(
        vertexai=True, project=project, location=location, credentials=creds,
    )
    return client, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _get_openai_client() -> tuple:
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in .env")
    return openai.OpenAI(api_key=api_key), os.environ.get("OPENAI_MODEL", "gpt-4o")


# ── Crop helpers ──────────────────────────────────────────────────────────────

def find_legend_rect(page: fitz.Page) -> fitz.Rect:
    pw, ph = page.rect.width, page.rect.height
    for kw in ["LEGEND", "LEGEND:", "KEY PLAN", "KEYNOTES"]:
        hits = page.search_for(kw)
        if hits:
            r    = hits[0]
            rect = fitz.Rect(r.x0 - 200, r.y0 - 20, r.x0 + 400, r.y0 + 400) & page.rect
            return rect
    return fitz.Rect(pw * 0.75, 0, pw, ph)


def _title_block_divider_x(page: fitz.Page) -> float | None:
    """Leftmost full-height vertical line in the right strip of the sheet —
    the divider between the plan area and the title block."""
    pw, ph = page.rect.width, page.rect.height
    best = None
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] != "l":
                continue
            a, b = item[1], item[2]
            if abs(a.x - b.x) > 1.0:            # not vertical
                continue
            if abs(a.y - b.y) < ph * 0.7:       # not full height
                continue
            x = (a.x + b.x) / 2.0
            if pw * 0.70 <= x <= pw * 0.98:
                best = x if best is None else min(best, x)
    return best


def find_drawing_content_rect(page: fitz.Page, legend_rect: fitz.Rect) -> fitz.Rect:
    pw, ph   = page.rect.width, page.rect.height
    border_x = pw * 0.03
    border_y = ph * 0.03
    right    = pw - border_x
    # the legend only bounds the content when it really is a right-side
    # panel; a LEGEND block at the bottom-left must not cut the x-axis
    # (that produced an inverted rect on GA sheets and fail-closed pages)
    if legend_rect is not None and legend_rect.x0 > pw * 0.55 \
            and legend_rect.x1 >= pw * 0.9:
        right = legend_rect.x0 - 4
    else:
        divider = _title_block_divider_x(page)
        if divider is not None:
            right = divider - 4
    rect = fitz.Rect(border_x, border_y, right, ph - border_y)
    if rect.x1 <= rect.x0 or rect.y1 <= rect.y0:
        rect = fitz.Rect(border_x, border_y, pw - border_x, ph - border_y)
    return rect


def render_crop(page: fitz.Page, clip_rect: fitz.Rect, dpi: int) -> tuple:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img, buf.getvalue()


# ── Vision API calls ──────────────────────────────────────────────────────────

def call_gemini_2images(client, model: str,
                        content_bytes: bytes, legend_bytes: bytes) -> tuple:
    """Returns (exterior_verts, no_go_zones). Falls back to ([], []) on error."""
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=content_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=legend_bytes,  mime_type="image/png"),
            PROMPT_VISION,
        ],
    )
    return _parse_vision_response(response.text)


def call_openai_2images(client, model: str,
                        content_bytes: bytes, legend_bytes: bytes) -> tuple:
    """Returns (exterior_verts, no_go_zones). Falls back to ([], []) on error."""
    b64_content = base64.b64encode(content_bytes).decode()
    b64_legend  = base64.b64encode(legend_bytes).decode()

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64_content}"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64_legend}"}},
                {"type": "text", "text": PROMPT_VISION},
            ],
        }],
        max_tokens=4096,
    )
    return _parse_vision_response(response.choices[0].message.content)


def _parse_vision_response(raw: str) -> tuple:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```$",           "", raw, flags=re.MULTILINE).strip()

    for attempt in range(2):
        try:
            parsed = json.loads(raw)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                raw = re.sub(r'(?<=[,\[{]\s*)\b[a-zA-Z_][a-zA-Z_0-9 ]*\b(?=\s*[,\]\}])', '', raw)
                raw = re.sub(r',\s*,', ',', raw)
                raw = re.sub(r',\s*\]', ']', raw)
                raw = re.sub(r',\s*\}', '}', raw)
            else:
                print(f"[VisionRefiner] JSON parse failed. Raw: {raw[:200]}")
                return [], []

    exterior    = parsed.get("exterior") or parsed.get("vertices", [])
    no_go_zones = parsed.get("no_go_zones", [])
    print(f"[VisionRefiner] exterior={len(exterior)}v  no_go={len(no_go_zones)}")
    return exterior, no_go_zones


# ── Vector snap ───────────────────────────────────────────────────────────────

def _bezier_sample(p1, p2, p3, p4, n: int = 8) -> list:
    pts = []
    for i in range(n + 1):
        t = i / n; u = 1 - t
        x = u**3*p1.x + 3*u**2*t*p2.x + 3*u*t**2*p3.x + t**3*p4.x
        y = u**3*p1.y + 3*u**2*t*p2.y + 3*u*t**2*p3.y + t**3*p4.y
        pts.append(fitz.Point(x, y))
    return pts


def collect_endpoints(page: fitz.Page) -> list:
    pts = []
    for path in page.get_drawings():
        for item in path["items"]:
            kind = item[0]
            if kind == "l":
                pts.append(item[1]); pts.append(item[2])
            elif kind == "c":
                pts.append(item[1]); pts.append(item[4])
                pts.extend(_bezier_sample(item[1], item[2], item[3], item[4]))
            elif kind == "re":
                r2 = item[1]
                pts += [fitz.Point(r2.x0, r2.y0), fitz.Point(r2.x1, r2.y0),
                        fitz.Point(r2.x1, r2.y1), fitz.Point(r2.x0, r2.y1)]
            elif kind == "qu":
                q = item[1]
                pts += [q.ul, q.ur, q.lr, q.ll]
    return pts


def _content_norm_to_pdf(xn: float, yn: float, cr: fitz.Rect) -> tuple:
    return cr.x0 + xn * cr.width, cr.y0 + yn * cr.height


def snap_all(exterior: list, content_rect: fitz.Rect,
             page: fitz.Page, snap_r: float) -> tuple:
    """Snap normalized exterior coords to nearest PDF vector endpoint."""
    endpoints = collect_endpoints(page)
    snapped, info = [], []
    for i, (xn, yn) in enumerate(exterior):
        x_pdf, y_pdf = _content_norm_to_pdf(xn, yn, content_rect)
        if endpoints:
            nearest = min(endpoints, key=lambda p: (p.x - x_pdf)**2 + (p.y - y_pdf)**2)
            dist    = math.hypot(nearest.x - x_pdf, nearest.y - y_pdf)
        else:
            nearest, dist = fitz.Point(x_pdf, y_pdf), 999.0

        if dist <= snap_r:
            snapped.append((nearest.x, nearest.y))
            info.append({"snapped": True,  "dist": round(dist, 1)})
        else:
            snapped.append((x_pdf, y_pdf))
            info.append({"snapped": False, "dist": round(dist, 1)})

    n_snap = sum(1 for x in info if x["snapped"])
    print(f"[VisionRefiner] {n_snap}/{len(info)} vertices snapped (snap_r={snap_r}pt)")
    return snapped, info


# ── Shapely polygon builder ───────────────────────────────────────────────────

def build_final_polygon(snapped_ext: list, no_go_zones: list,
                        content_rect: fitz.Rect):
    """Return Shapely Polygon after subtracting no-go zones. None on failure."""
    if len(snapped_ext) < 3:
        return None

    slab = SPoly(snapped_ext)

    if no_go_zones:
        excl = []
        for (nx0, ny0, nx1, ny1) in no_go_zones:
            x0 = content_rect.x0 + nx0 * content_rect.width
            y0 = content_rect.y0 + ny0 * content_rect.height
            x1 = content_rect.x0 + nx1 * content_rect.width
            y1 = content_rect.y0 + ny1 * content_rect.height
            excl.append(shapely_box(x0, y0, x1, y1))
        slab = slab.difference(unary_union(excl))

    if slab.is_empty:
        return None

    from shapely.geometry import MultiPolygon, GeometryCollection
    if isinstance(slab, (MultiPolygon, GeometryCollection)):
        polys = [g for g in slab.geoms if isinstance(g, SPoly) and not g.is_empty]
        if not polys:
            return None
        slab = max(polys, key=lambda g: g.area)

    return slab if slab.is_valid else slab.buffer(0)


# ── Public entry point ────────────────────────────────────────────────────────

def refine_page_slabs(slabs: list, page: fitz.Page, client, model: str,
                      backend: str = "gemini", snap_r: float = SNAP_R) -> list:
    """
    Call Vision API once for the whole page, snap to vectors, subtract no-go zones.
    Replaces the largest SlabRegion's polygon with the vision result.
    Falls back to original slabs if Vision fails.
    """
    if not slabs:
        return slabs

    try:
        legend_rect  = find_legend_rect(page)
        content_rect = find_drawing_content_rect(page, legend_rect)

        _, bytes_content = render_crop(page, content_rect, DPI_CONTENT)
        _, bytes_legend  = render_crop(page, legend_rect,  DPI_LEGEND)

        if backend == "openai":
            exterior, no_go_zones = call_openai_2images(
                client, model, bytes_content, bytes_legend)
        else:
            exterior, no_go_zones = call_gemini_2images(
                client, model, bytes_content, bytes_legend)

        if len(exterior) < 3:
            print("[VisionRefiner] Not enough exterior vertices — keeping original polygon")
            return slabs

        snapped_ext, _ = snap_all(exterior, content_rect, page, snap_r)
        vision_poly    = build_final_polygon(snapped_ext, no_go_zones, content_rect)

        if vision_poly is None or vision_poly.is_empty:
            print("[VisionRefiner] Vision polygon empty — keeping original")
            return slabs

        # Replace the largest slab's polygon with the vision result
        largest_idx = max(range(len(slabs)), key=lambda i: slabs[i].polygon.area)
        slabs[largest_idx].polygon = vision_poly
        slabs[largest_idx].source  = "vision"
        print(f"[VisionRefiner] Replaced slab[{largest_idx}].polygon with vision result "
              f"({vision_poly.area:.0f} pt²)")
        if len(slabs) > 1:
            others = [s for i, s in enumerate(slabs) if i != largest_idx and getattr(s, "polygon", None)]
            max_other = max((s.polygon.area for s in others), default=0)
            if max_other > 0 and vision_poly.area >= max_other * 2.5:
                cleaned = [slabs[largest_idx]]
                for s in others:
                    keep_large_external = (
                        s.polygon.area >= vision_poly.area * 0.20
                        and not vision_poly.buffer(8).contains(s.polygon.centroid)
                    )
                    if keep_large_external:
                        cleaned.append(s)
                dropped = len(slabs) - len(cleaned)
                if dropped:
                    print(f"[VisionRefiner] Dropped {dropped} small boundary fragments after vision cleanup")
                    slabs = cleaned

    except Exception as exc:
        print(f"[VisionRefiner] Error: {exc} — keeping original polygons")

    return slabs
