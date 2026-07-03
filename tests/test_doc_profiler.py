"""Document profiler: self-detection and routing (robustness Bước 2).

profile_document classifies every page from invariants (title + content
signals), pair_documents groups a folder's PDFs by project code and
discipline — the entry point for '2k PDFs with no manual page picking'.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.doc_profiler import profile_document, pair_documents

PDF_2381_STR = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")
PDF_2381_ARCH = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_ARCH_Combine.pdf")
PDF_SMPS = Path(r"C:\Users\LENOVO\Downloads"
                r"\2402. South Melbourne Primary School - CIVIL & STR - 260610.pdf")


@pytest.mark.skipif(not PDF_2381_STR.exists(), reason="2381 STR not present")
class TestProfile2381Str:

    @pytest.fixture(scope="class")
    def prof(self):
        return profile_document(str(PDF_2381_STR))

    def _kind(self, prof, page_no):
        return next(p for p in prof.pages if p["page_no"] == page_no)["kind"]

    def test_ga_plans(self, prof):
        for pno in (16, 17, 18, 24, 27):
            assert self._kind(prof, pno) == "ga_plan", pno

    def test_loading_plans(self, prof):
        for pno in (5, 8, 13):
            assert self._kind(prof, pno) == "loading_plan", pno

    def test_foundation(self, prof):
        assert self._kind(prof, 15) == "foundation_plan"

    def test_steel_framing(self, prof):
        assert self._kind(prof, 28) == "steel_framing"

    def test_schedule_flag_on_ga(self, prof):
        p17 = next(p for p in prof.pages if p["page_no"] == 17)
        assert p17["has_schedule"] is True


@pytest.mark.skipif(not PDF_2381_ARCH.exists(), reason="2381 ARCH not present")
def test_arch_section_pages_detected():
    prof = profile_document(str(PDF_2381_ARCH))
    kinds = {p["page_no"]: p["kind"] for p in prof.pages}
    assert kinds[16] == "section"
    assert kinds[17] == "section"


@pytest.mark.skipif(
    not (PDF_2381_STR.exists() and PDF_2381_ARCH.exists()
         and PDF_SMPS.exists()),
    reason="all three PDFs needed")
def test_pairing_by_project_code():
    groups = pair_documents([str(PDF_2381_STR), str(PDF_2381_ARCH),
                             str(PDF_SMPS)])
    assert set(groups) == {"2381", "2402"}
    g = groups["2381"]
    assert Path(g["str"]).name == PDF_2381_STR.name
    assert Path(g["arch"]).name == PDF_2381_ARCH.name
    assert "arch" not in groups["2402"]
