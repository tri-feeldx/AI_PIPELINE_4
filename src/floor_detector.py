"""
Smart floor detection across multi-page structural PDFs.

Algorithm:
  1. Quick parallel text-scan of all selected pages (no polygon extraction)
  2. Filter globally-appearing FFL values (noise from title blocks / general notes)
  3. Group pages by dominant FFL — the single most-frequently-mentioned FFL on each page
  4. Per group: pick canonical page (most info), identify supplements (different area),
     mark true duplicates as skipped
  5. Return pages_to_process (canonical + supplements) — skip the rest

Works for any PDF: 3 floors, 5 floors, 10 floors — fully adaptive.

Key insight: on a Level N slab plan, "FFL N.000" is annotated on every slab (many occurrences),
while other floors' FFLs appear once in cross-reference notes. The dominant (most frequent) FFL
is the reliable floor identifier, not the full set of unique FFLs on the page.
"""

import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PageProfile:
    page_idx: int
    title: str
    ffl_set: frozenset           # frozenset[float] — unique FFL values (for display)
    ffl_raw_list: list           # list[float] — all occurrences (for counting dominant)
    floor_keyword: str           # "level_1", "ground", "basement", "roof", "unknown"
    floor_key: str               # grouping key — derived from dominant FFL
    dominant_ffl: Optional[float] = None   # most-mentioned FFL on this page


@dataclass
class FloorGroup:
    floor_key: str
    floor_label: str             # human-readable, e.g. "Level 1 — FFL 1.000m"
    canonical_page: int          # best page for this floor
    supplemental_pages: list = field(default_factory=list)   # different area, same floor
    skipped_pages: list = field(default_factory=list)        # true duplicates


@dataclass
class FloorDetectResult:
    groups: list                          # list[FloorGroup]
    pages_to_process: list                # canonical + supplemental, sorted
    skipped_pages: list                   # truly redundant pages
    detection_basis: str                  # "ffl" | "title" | "all_pages"
    floor_count: int
    warnings: list = field(default_factory=list)


# ── Floor keyword normalisation ────────────────────────────────────────────────

_LEVEL_RE    = re.compile(r"(?:LEVEL|LVL|L)\s*0*(\d+)", re.IGNORECASE)
_GROUND_KW   = {"GROUND", "GF", "G/F", "GRADE", "GRD"}
_BASEMENT_KW = {"BASEMENT", "BSMT", "B1", "B2", "B3", "SUB"}
_ROOF_KW     = {"ROOF", "ROOFTOP", "TOP", "UPPERMOST"}
_SUPPLEMENT_KW = {"AREA", "ZONE", "SECTION", "PART", "WING", "BLOCK", "GRID", "PRECINCT"}

_TITLE_BLOCK_SKIP = frozenset([
    "FOR TENDER", "ISSUED FOR TENDER", "ISSUED FOR", "NOT TO BE USED",
    "TO BE PRINTED IN COLOUR", "TO BE PRINTED IN COLOR",
    "TENDER NOTE", "TENDER NOTES", "COPYRIGHT", "SCALE AT A1", "SCALE AT A3",
    "DRAWN", "CHECKED", "APPROVED", "REVISION", "DO NOT SCALE",
])

_FLOOR_KWS = {"LEVEL", "FLOOR", "SLAB", "PLAN", "GROUND", "BASEMENT", "ROOF", "PODIUM"}


def _normalize_floor_keyword(title: str) -> str:
    t = title.upper()
    m = _LEVEL_RE.search(t)
    if m:
        return f"level_{int(m.group(1))}"
    if any(k in t for k in _GROUND_KW):
        return "ground"
    if any(k in t for k in _BASEMENT_KW):
        return "basement"
    if any(k in t for k in _ROOF_KW):
        return "roof"
    return "unknown"


def _make_floor_key(dominant_ffl: Optional[float], floor_keyword: str, page_idx: int) -> str:
    """
    Build grouping key using dominant FFL (most-mentioned on this page), not the full FFL set.

    Fallback chain:
      dominant_ffl → floor keyword in title → page index (safest: every page = unique)
    """
    if dominant_ffl is not None:
        return f"ffl__{dominant_ffl:.2f}"
    if floor_keyword != "unknown":
        return f"kw__{floor_keyword}"
    return f"page__{page_idx}"


def _is_supplement(title: str, canonical_title: str) -> bool:
    """True if page covers a different area of the same floor (e.g. 'Area B' vs 'Area A')."""
    t  = title.upper()
    ct = canonical_title.upper()
    for kw in _SUPPLEMENT_KW:
        if kw in t and kw not in ct:
            return True
    return False


def _make_floor_label(profile: "PageProfile") -> str:
    parts = []
    if profile.floor_keyword != "unknown":
        parts.append(profile.floor_keyword.replace("_", " ").title())
    if profile.dominant_ffl is not None:
        parts.append(f"FFL {profile.dominant_ffl:.3f}m")
    elif profile.ffl_set:
        ffls = ", ".join(f"{v:.3f}m" for v in sorted(profile.ffl_set))
        parts.append(f"FFL {ffls}")
    if not parts:
        parts.append(f"Page {profile.page_idx + 1}")
    return " — ".join(parts)


# ── Quick parallel scan (text-only, cheap) ─────────────────────────────────────

def _extract_title(text_blocks: list) -> str:
    """
    Extract the drawing title, skipping title block boilerplate.
    Priority: text containing floor plan keywords → any non-trivial non-boilerplate text.
    """
    candidates = sorted(text_blocks, key=lambda b: b.get("size", 0), reverse=True)

    # Priority 1: contains floor plan keywords and is not boilerplate
    for b in candidates[:20]:
        t = b.get("text", "").strip()
        if not t:
            continue
        t_up = t.upper()
        if t_up in _TITLE_BLOCK_SKIP:
            continue
        if any(kw in t_up for kw in _FLOOR_KWS):
            return t[:80]

    # Priority 2: any non-numeric, non-boilerplate text (4+ chars)
    for b in candidates[:12]:
        t = b.get("text", "").strip()
        if not t or re.fullmatch(r"[\d\s.,:;+\-/()°%]+", t):
            continue
        if t.upper() in _TITLE_BLOCK_SKIP or len(t) <= 4:
            continue
        return t[:80]

    return ""


def _scan_one_page(args: tuple) -> "PageProfile":
    """Worker: open own fitz doc, extract text signals only — no polygon work."""
    pdf_path, page_idx = args
    import fitz
    from src.pdf_processor import extract_text_blocks, extract_ffl_values

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        text_blocks = extract_text_blocks(page)
        ffl_values  = extract_ffl_values(text_blocks)
        title       = _extract_title(text_blocks)

        # Count ALL occurrences (not just unique) — needed for dominant FFL
        ffl_raw_list = [round(f["ffl_m"], 2) for f in ffl_values]
        ffl_set      = frozenset(ffl_raw_list)
        ffl_counts   = Counter(ffl_raw_list)
        dominant_ffl = ffl_counts.most_common(1)[0][0] if ffl_counts else None

        kw  = _normalize_floor_keyword(title)
        key = _make_floor_key(dominant_ffl, kw, page_idx)

        return PageProfile(
            page_idx=page_idx,
            title=title,
            ffl_set=ffl_set,
            ffl_raw_list=ffl_raw_list,
            floor_keyword=kw,
            floor_key=key,
            dominant_ffl=dominant_ffl,
        )
    finally:
        doc.close()


def quick_scan_pages(pdf_path: str, page_indices: list) -> list:
    """Parallel text-only scan. Returns list[PageProfile] sorted by page_idx."""
    args = [(pdf_path, idx) for idx in page_indices]
    workers = min(len(args), 8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_scan_one_page, args))
    return sorted(results, key=lambda p: p.page_idx)


# ── Global noise filter ────────────────────────────────────────────────────────

def _filter_noise_ffls(profiles: list) -> set:
    """
    FFL values appearing on >60% of pages are global references (title block, general notes,
    ridge heights) — they are not useful for identifying individual floors.
    """
    n = len(profiles)
    if n == 0:
        return set()
    ffl_page_count: Counter = Counter()
    for p in profiles:
        for v in p.ffl_set:
            ffl_page_count[v] += 1
    return {v for v, cnt in ffl_page_count.items() if cnt > n * 0.6}


def _recompute_dominant_ffls(profiles: list, noise_ffls: set) -> None:
    """
    After identifying noise FFLs, recompute dominant_ffl and floor_key for each profile
    using only clean (non-noise) FFL values.
    Mutates profiles in-place.
    """
    for p in profiles:
        clean_counts = Counter(v for v in p.ffl_raw_list if v not in noise_ffls)
        p.dominant_ffl = clean_counts.most_common(1)[0][0] if clean_counts else None
        p.floor_key = _make_floor_key(p.dominant_ffl, p.floor_keyword, p.page_idx)


# ── Floor grouping ─────────────────────────────────────────────────────────────

def _build_floor_groups(profiles: list) -> list:
    """Group PageProfiles by floor_key, pick canonical, identify supplements/duplicates."""
    buckets: dict = defaultdict(list)
    for p in profiles:
        buckets[p.floor_key].append(p)

    groups = []
    for floor_key, members in buckets.items():
        # Canonical = page with most FFL occurrences; tie-break → earliest page
        canonical = max(members, key=lambda p: (len(p.ffl_raw_list), -p.page_idx))

        supplements, skipped = [], []
        for m in members:
            if m.page_idx == canonical.page_idx:
                continue
            if _is_supplement(m.title, canonical.title):
                supplements.append(m.page_idx)
            else:
                skipped.append(m.page_idx)

        groups.append(FloorGroup(
            floor_key=floor_key,
            floor_label=_make_floor_label(canonical),
            canonical_page=canonical.page_idx,
            supplemental_pages=sorted(supplements),
            skipped_pages=sorted(skipped),
        ))

    return sorted(groups, key=lambda g: g.canonical_page)


# ── Main entry point ───────────────────────────────────────────────────────────

def detect_unique_floors(pdf_path: str, page_indices: list) -> FloorDetectResult:
    """
    Scan all selected pages, detect unique floors, return pages_to_process.

    Detection basis (adaptive):
      "ffl"       — >50% pages have FFL annotations (most reliable)
      "title"     — >30% pages have floor keywords in title
      "all_pages" — not enough signals; safe fallback: process everything
    """
    if not page_indices:
        return FloorDetectResult([], [], [], "all_pages", 0, ["No pages selected"])

    profiles = quick_scan_pages(pdf_path, page_indices)

    # Filter globally-appearing FFL noise before using FFLs for grouping
    noise_ffls = _filter_noise_ffls(profiles)
    if noise_ffls:
        _recompute_dominant_ffls(profiles, noise_ffls)

    ffl_count = sum(1 for p in profiles if p.dominant_ffl is not None)
    kw_count  = sum(1 for p in profiles if p.floor_keyword != "unknown")
    n = len(profiles)

    warnings = []
    if noise_ffls:
        warnings.append(
            f"Lọc {len(noise_ffls)} giá trị FFL toàn cục (xuất hiện >60% pages): "
            + ", ".join(f"{v:.3f}m" for v in sorted(noise_ffls))
        )

    if ffl_count >= n * 0.5:
        basis = "ffl"
    elif kw_count >= n * 0.3:
        basis = "title"
        if ffl_count > 0:
            warnings.append(
                f"Chỉ {ffl_count}/{n} trang có FFL sau khi lọc — dùng title keyword để nhóm tầng"
            )
    else:
        return FloorDetectResult(
            groups=[], pages_to_process=list(page_indices),
            skipped_pages=[], detection_basis="all_pages",
            floor_count=len(page_indices),
            warnings=warnings + [
                f"Không đủ tín hiệu FFL/title ({ffl_count} FFL, {kw_count} keyword) "
                f"— xử lý tất cả {n} pages"
            ],
        )

    groups = _build_floor_groups(profiles)

    pages_to_process = sorted({
        g.canonical_page for g in groups
    } | {
        p for g in groups for p in g.supplemental_pages
    })
    skipped = sorted({p for g in groups for p in g.skipped_pages})

    return FloorDetectResult(
        groups=groups,
        pages_to_process=pages_to_process,
        skipped_pages=skipped,
        detection_basis=basis,
        floor_count=len(groups),
        warnings=warnings,
    )
