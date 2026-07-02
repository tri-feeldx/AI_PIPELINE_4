"""Page-role classification must recognise GA titles without the word PLAN.

Bug (pre-existing at checkpoint f63e0db): _GEOMETRY_TITLE_RE only accepts
"GENERAL ARRANGEMENT PLAN" / "GA PLAN" style titles.  Combined Structural
p17/p18 are titled "GENERAL ARRANGEMENT GROUND FLOOR" and
"GENERAL ARRANGEMENT LEVEL - 1" — no "PLAN" — so the incidental words
SCHEDULE/DETAIL/LEGEND on the sheet pushed them to evidence_only and
extract_slabs_v2 exits early with EVIDENCE_ONLY_PAGE (this is what broke
tests/test_assembly_background_fix.py::TestP18SlabAssembly).
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.pdf_processor import classify_page_role_from_blocks

PDF_COMBINED = Path(r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")
PDF_2381 = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")


def _blocks(title: str, *body: str) -> list[dict]:
    """Synthetic text blocks: big title + small body text."""
    out = [{"text": title, "size": 24.0, "bbox": [2000, 1600, 2300, 1640]}]
    out += [{"text": t, "size": 8.0, "bbox": [100, 100 + 20 * i, 400, 118 + 20 * i]}
            for i, t in enumerate(body)]
    return out


BODY_NOISE = ("CONCRETE COLUMN SCHEDULE", "TYPICAL DETAIL", "LEGEND:",
              "NOTES: DO NOT SCALE", "REINFORCEMENT TO AS3600")


@pytest.mark.parametrize("title", [
    "GENERAL ARRANGEMENT LEVEL - 1",          # Combined Structural p18
    "GENERAL ARRANGEMENT GROUND FLOOR",       # Combined Structural p17
    "GENERAL ARRANGEMENT PLAN - LEVEL 01",    # 2381 p17
    "ROOF - GENERAL ARRANGEMENT PLAN",        # 2381 p27
    "GA PLAN LEVEL 03",
])
def test_ga_titles_are_geometry_even_with_schedule_noise(title):
    role = classify_page_role_from_blocks(_blocks(title, *BODY_NOISE))
    assert role["role"] == "geometry_plan", role


def test_schedule_sheet_stays_evidence_only():
    role = classify_page_role_from_blocks(
        _blocks("COLUMN SCHEDULE AND DETAILS", *BODY_NOISE))
    assert role["role"] == "evidence_only", role


def test_foundation_plan_stays_foundation():
    role = classify_page_role_from_blocks(
        _blocks("FOUNDATION PLAN", *BODY_NOISE))
    assert role["role"] == "foundation_plan", role


def test_loading_plan_stays_evidence_only():
    role = classify_page_role_from_blocks(
        _blocks("LEVEL 1 LOADING PLAN", *BODY_NOISE))
    assert role["role"] == "evidence_only", role


@pytest.mark.skipif(not PDF_COMBINED.exists(), reason="Combined PDF not present")
@pytest.mark.parametrize("page_no", [17, 18])
def test_combined_ga_pages_geometry(page_no):
    from src.slab_v2.pipeline import _page_text_audits
    doc = fitz.open(str(PDF_COMBINED))
    _, _, role = _page_text_audits(doc, page_no - 1)
    assert role["role"] == "geometry_plan", role


@pytest.mark.skipif(not PDF_2381.exists(), reason="2381 PDF not present")
def test_2381_roles_unchanged():
    from src.slab_v2.pipeline import _page_text_audits
    doc = fitz.open(str(PDF_2381))
    expected = {15: "foundation_plan", 16: "geometry_plan",
                17: "geometry_plan", 27: "geometry_plan"}
    for page_no, want in expected.items():
        _, _, role = _page_text_audits(doc, page_no - 1)
        assert role["role"] == want, (page_no, role)
