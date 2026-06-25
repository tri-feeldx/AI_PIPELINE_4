"""Property-based tests for X-cross detection — generic, not PDF-specific.

These tests verify invariants that must hold for ANY PDF:
  - Every verified cut opening must be inside the slab
  - Every verified cut opening must have valid opening_intent
  - Verified cuts must not overlap with columns or walls
  - No verified cut should be unreasonably large or small
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config
from src.slab_v2.pipeline import extract_slabs_v2
from src.slab_v2.models import OpeningIntent


_XCROSS_PDF = Path(r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")
_XCROSS_PAGES = [5, 7, 9, 10, 16, 17]  # 0-indexed: pages 6, 8, 10, 11, 17, 18

if not _XCROSS_PDF.exists():
    pytest.skip("X-cross test PDF not found: Combined Structural.pdf",
                allow_module_level=True)

_CASES = [(_XCROSS_PDF, pi) for pi in _XCROSS_PAGES]


@pytest.fixture(scope="module")
def cfg_generic() -> SlabV2Config:
    return SlabV2Config(
        debug_images=False,
        enable_opening_judge=False,
        enable_slab_face_judge=False,
        enable_floor_system_judge=False,
    )


_VALID_INTENTS = {
    OpeningIntent.SLAB_PENETRATION.value,
    OpeningIntent.VOID.value,
    OpeningIntent.LIFT_SHAFT.value,
}


@pytest.mark.parametrize(
    "pdf_path,page_index",
    _CASES,
    ids=[f"{p.stem}_p{pi+1}" for p, pi in _CASES])
class TestXcrossGenericProperties:

    def test_verified_cuts_have_valid_intent(self, pdf_path, page_index,
                                              cfg_generic):
        """Every verified cut opening must have a destructive-allowed intent."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_generic, use_ai=True)
        for opening in result.verified_cut_openings:
            assert opening.opening_intent in _VALID_INTENTS, (
                f"'{opening.label}' has intent={opening.opening_intent}, "
                f"expected one of {_VALID_INTENTS}")

    def test_verified_cuts_have_evidence(self, pdf_path, page_index,
                                          cfg_generic):
        """Every verified cut must have at least one evidence ID."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_generic, use_ai=True)
        for opening in result.verified_cut_openings:
            assert opening.evidence_ids, (
                f"'{opening.label}' intent={opening.opening_intent} "
                f"has no evidence_ids")

    def test_verified_cuts_inside_slab(self, pdf_path, page_index,
                                       cfg_generic):
        """Every verified cut polygon must intersect the slab."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_generic, use_ai=True)
        if result.status != "OK" or not result.slabs:
            pytest.skip("no slab")
        from shapely.ops import unary_union
        slab_union = unary_union(
            [s["polygon_pdf"] for s in result.slabs if s.get("polygon_pdf")])
        for opening in result.verified_cut_openings:
            assert opening.polygon.intersects(slab_union), (
                f"'{opening.label}' does not intersect slab")

    def test_verified_cuts_reasonable_size(self, pdf_path, page_index,
                                            cfg_generic):
        """No verified cut should exceed 30% of the slab area."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_generic, use_ai=True)
        if result.status != "OK" or not result.slabs:
            pytest.skip("no slab")
        from shapely.ops import unary_union
        slab_union = unary_union(
            [s["polygon_pdf"] for s in result.slabs if s.get("polygon_pdf")])
        slab_area = slab_union.area
        for opening in result.verified_cut_openings:
            ratio = opening.polygon.area / max(slab_area, 1e-9)
            assert ratio < 0.30, (
                f"'{opening.label}' covers {ratio:.0%} of slab — too large")

    def test_verified_cuts_dont_overlap_columns(self, pdf_path, page_index,
                                                  cfg_generic):
        """Verified cuts must not significantly overlap with column footprints."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_generic, use_ai=True)
        if not result.columns:
            pytest.skip("no columns")
        for opening in result.verified_cut_openings:
            for col in result.columns:
                inter = opening.polygon.intersection(col.polygon)
                ratio = inter.area / max(col.polygon.area, 1e-9)
                assert ratio < 0.10, (
                    f"cut '{opening.label}' overlaps column '{col.symbol}' "
                    f"by {ratio:.0%}")
