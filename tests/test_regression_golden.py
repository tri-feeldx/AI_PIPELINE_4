"""Golden regression tests — lock metrics from the 10:45 26/6/26 golden version.

These tests protect the quality achieved by the golden version on Combined
Structural (Edwins Street) while ensuring the Phase 3.6 improvements for
Cairns Hospital and South Melbourne are preserved.

Tests run single-page pipeline extraction and assert per-page metrics.
Column-level tests require column_types census data; element/opening tests
run without census.

PDF locations are auto-discovered; tests skip if a PDF is not found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.slab_v2.config import SlabV2Config
from src.slab_v2.pipeline import extract_slabs_v2

# ── PDF discovery ────────────────────────────────────────────────────────────

_COMBINED_STRUCTURAL = Path(
    r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")
_CAIRNS_HOSPITAL = Path(
    r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")
_SOUTH_MELBOURNE = Path(
    r"C:\Users\LENOVO\Downloads"
    r"\2402. South Melbourne Primary School - CIVIL & STR - 260610.pdf")

has_combined = _COMBINED_STRUCTURAL.exists()
has_cairns = _CAIRNS_HOSPITAL.exists()
has_south_melb = _SOUTH_MELBOURNE.exists()


# ── Shared config ────────────────────────────────────────────────────────────

@pytest.fixture
def cfg_golden() -> SlabV2Config:
    """Config matching golden version behavior — no AI judges for speed."""
    return SlabV2Config(
        debug_images=False,
        enable_opening_judge=False,
        enable_slab_face_judge=False,
        enable_floor_system_judge=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Structural (Edwins Street) — GOLDEN REFERENCE
#
# Golden version (10:45 26/6/26) metrics across building export:
#   276 RC columns, 46 walls, 2 verified cut openings, 0 C? ambiguous
#
# Representative test pages:
#   Page 6 (index 5): Foundation — has 14 slabs, many columns
#   Page 17 (index 16): Ground floor — has columns + walls
#   Page 18 (index 17): Level 1 — typical floor with columns
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not has_combined, reason="Combined Structural PDF not found")
class TestCombinedStructuralGolden:
    """Protect golden-version quality on Combined Structural."""

    def test_page_produces_slab(self, cfg_golden):
        """Page 18 (Level 1) must produce at least one slab polygon."""
        result = extract_slabs_v2(
            str(_COMBINED_STRUCTURAL), 17, cfg_golden, use_ai=False)
        assert result.status == "OK"
        assert len(result.slabs) >= 1

    def test_element_count_not_excessive(self, cfg_golden):
        """Golden version had few raw elements per page. Phase 3.6 inflated
        this by lowering thresholds. Elements should stay conservative."""
        result = extract_slabs_v2(
            str(_COMBINED_STRUCTURAL), 17, cfg_golden, use_ai=False)
        n_elements = len(result.elements)
        assert n_elements <= 20, (
            f"Page 18 has {n_elements} raw elements — golden had few. "
            f"Over-detection suggests xcross thresholds are too loose.")

    def test_verified_openings_conservative(self, cfg_golden):
        """Golden version had only 2 verified openings across ALL pages.
        A single page should have at most 3."""
        result = extract_slabs_v2(
            str(_COMBINED_STRUCTURAL), 17, cfg_golden, use_ai=False)
        n_openings = len(result.verified_cut_openings)
        assert n_openings <= 3, (
            f"Page 18 has {n_openings} verified openings — golden had ≤2 "
            f"total across all pages. Over-detection likely.")

    def test_no_ambiguous_columns_without_census(self, cfg_golden):
        """Even without census, shape-fallback should not produce C? on
        a clean Revit-exported page."""
        result = extract_slabs_v2(
            str(_COMBINED_STRUCTURAL), 17, cfg_golden, use_ai=False)
        c_unknown = sum(1 for c in result.columns if c.symbol == "C?")
        assert c_unknown == 0, (
            f"Page 18 has {c_unknown} ambiguous C? columns — golden had 0")

    def test_scale_detected(self, cfg_golden):
        """Scale must be detected on structural plan pages."""
        result = extract_slabs_v2(
            str(_COMBINED_STRUCTURAL), 17, cfg_golden, use_ai=False)
        assert result.scale is not None
        assert 50 <= result.scale <= 200, f"Unexpected scale: {result.scale}"

    def test_foundation_page_produces_slab(self, cfg_golden):
        """Page 6 (foundation) must produce slabs."""
        result = extract_slabs_v2(
            str(_COMBINED_STRUCTURAL), 5, cfg_golden, use_ai=False)
        assert result.status == "OK"
        assert len(result.slabs) >= 1

    def test_multiple_pages_total_openings_low(self, cfg_golden):
        """Golden had 2 verified openings across ALL pages. Test 3
        representative pages — total should stay ≤ 5."""
        total_openings = 0
        for page_idx in [5, 16, 17]:  # pages 6, 17, 18
            result = extract_slabs_v2(
                str(_COMBINED_STRUCTURAL), page_idx, cfg_golden, use_ai=False)
            total_openings += len(result.verified_cut_openings)
        assert total_openings <= 5, (
            f"3 representative pages have {total_openings} total verified "
            f"openings — golden had 2 across ALL pages")


# ═══════════════════════════════════════════════════════════════════════════════
# Cairns Hospital (2381_MSCP) — PHASE 3.6 IMPROVEMENTS TO PRESERVE
#
# Phase 3.6 enabled CW+LW wall clustering which correctly detects core
# shafts in this PDF. These tests ensure we don't regress on that.
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not has_cairns, reason="Cairns Hospital PDF not found")
class TestCairnsHospitalPreserved:
    """Ensure Phase 3.6 improvements for Cairns Hospital are preserved."""

    def test_page_produces_slab(self, cfg_golden):
        """A structural plan page must produce slabs."""
        result = extract_slabs_v2(
            str(_CAIRNS_HOSPITAL), 3, cfg_golden, use_ai=False)
        assert result.status == "OK"
        assert len(result.slabs) >= 1

    def test_elements_detected(self, cfg_golden):
        """Cairns Hospital has stair/lift/shaft elements that must be found."""
        result = extract_slabs_v2(
            str(_CAIRNS_HOSPITAL), 3, cfg_golden, use_ai=False)
        assert len(result.elements) >= 1, (
            "Cairns Hospital should have detectable X-cross elements")

    def test_scale_detected(self, cfg_golden):
        """Scale must be detected."""
        result = extract_slabs_v2(
            str(_CAIRNS_HOSPITAL), 3, cfg_golden, use_ai=False)
        assert result.scale is not None
        assert result.scale > 0


# ═══════════════════════════════════════════════════════════════════════════════
# South Melbourne — BASIC PIPELINE HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not has_south_melb, reason="South Melbourne PDF not found")
class TestSouthMelbourneBasic:
    """Basic pipeline health for South Melbourne PDF."""

    def test_pipeline_does_not_crash(self, cfg_golden):
        """Pipeline must complete without exception."""
        doc = fitz.open(str(_SOUTH_MELBOURNE))
        mid_page = min(5, doc.page_count - 1)
        doc.close()
        result = extract_slabs_v2(
            str(_SOUTH_MELBOURNE), mid_page, cfg_golden, use_ai=False)
        assert result is not None
        # graceful fail-closed statuses count as 'did not crash' — SMPS p6
        # is a notes/site sheet and correctly refuses a tiny slab export
        assert result.status in {"OK", "NO_FACES", "NO_EXPORT_TINY_SLAB"}

    def test_structural_page_has_scale(self, cfg_golden):
        """At least one page in the first 15 should have a detectable scale."""
        doc = fitz.open(str(_SOUTH_MELBOURNE))
        n_pages = min(15, doc.page_count)
        doc.close()
        found_scale = False
        for page_idx in range(n_pages):
            result = extract_slabs_v2(
                str(_SOUTH_MELBOURNE), page_idx, cfg_golden, use_ai=False)
            if result.status == "OK" and result.scale is not None:
                found_scale = True
                break
        assert found_scale, (
            "No structural page with detectable scale in first 15 pages")
