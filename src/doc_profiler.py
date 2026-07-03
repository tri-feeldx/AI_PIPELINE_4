"""Document profiler: page-kind classification + STR/ARCH pairing (Bước 2).

Every signal is a Revit-export invariant, none is office-specific:
sheet titles, level datums (sections), grid bubbles, on-page schedules.
The profiler only OBSERVES — extraction routing consumes its output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from src.slab_v2.pipeline import _page_text_audits

_STEEL_TITLE_RE = re.compile(r"STEEL\s+(FRAMING|CANOPY|MARKING)", re.I)
_LOADING_TITLE_RE = re.compile(r"\bLOADING\s+PLAN\b", re.I)
_SITE_TITLE_RE = re.compile(r"\bSITE\b", re.I)
_SECTION_MIN_LEVELS = 5
_PROJECT_CODE_RE = re.compile(r"^\D{0,4}(\d{3,5})")
_ARCH_NAME_RE = re.compile(r"(?:^|[_\s-])(ARCH|ARCHITECT\w*)(?:[_\s.-]|$)", re.I)
_STR_NAME_RE = re.compile(
    r"(?:^|[_\s-])(STR|STRUCT\w*|STRUCTURAL)(?:[_\s.&-]|$)", re.I)


@dataclass
class DocProfile:
    pdf: str
    pages: list = field(default_factory=list)


def _classify_page(doc, page_index: int) -> dict:
    from src.arch_ref.levels import extract_levels
    from src.slab_v2.schedule_parser import parse_schedules

    page = doc[page_index]
    _, scale_audit, role = _page_text_audits(doc, page_index)
    title = role.get("title", "") or ""

    sched = parse_schedules(page)
    levels = extract_levels(page)

    if _STEEL_TITLE_RE.search(title):
        kind = "steel_framing"
    elif role.get("role") == "foundation_plan":
        kind = "foundation_plan"
    elif _LOADING_TITLE_RE.search(title):
        kind = "loading_plan"
    elif role.get("role") == "geometry_plan":
        kind = "ga_plan"
    elif len(levels.elevations_m) >= _SECTION_MIN_LEVELS:
        kind = "section"
    elif _SITE_TITLE_RE.search(title):
        kind = "site"
    elif sched.columns or sched.walls:
        kind = "schedule_detail"
    else:
        kind = "notes_other"

    return {
        "page_no": page_index + 1,
        "kind": kind,
        "title": title,
        "role": role.get("role"),
        "scale": scale_audit.get("chosen_scale"),
        "has_schedule": bool(sched.columns or sched.walls),
        "n_level_datums": len(levels.elevations_m),
    }


def profile_document(pdf_path: str) -> DocProfile:
    doc = fitz.open(pdf_path)
    prof = DocProfile(pdf=pdf_path)
    for i in range(len(doc)):
        try:
            prof.pages.append(_classify_page(doc, i))
        except Exception as exc:               # noqa: BLE001
            prof.pages.append({"page_no": i + 1, "kind": "error",
                               "error": str(exc)[:120]})
    return prof


def pair_documents(pdf_paths: list) -> dict:
    """Group a folder's PDFs by project code; tag STR/ARCH discipline.

    Project code = leading digit run of the file name (company convention:
    '2381_MSCP_STR_...', '2402. South Melbourne ...').
    """
    groups: dict[str, dict] = {}
    for p in pdf_paths:
        name = Path(p).name
        m = _PROJECT_CODE_RE.match(name)
        if not m:
            groups.setdefault("_unmatched", {}).setdefault("files", []).append(p)
            continue
        code = m.group(1)
        g = groups.setdefault(code, {})
        if _ARCH_NAME_RE.search(name):
            g["arch"] = p
        elif _STR_NAME_RE.search(name):
            g["str"] = p
        else:
            g.setdefault("other", []).append(p)
    return groups
