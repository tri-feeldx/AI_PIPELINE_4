"""X-cross end-to-end diagnostic — traces the FULL detection chain.

For every test PDF page that produces elements, this test audits:
  1. How many X-crosses were detected by elements.py?
  2. How many became opening candidates in opening_resolver?
  3. How many were verified and approved for slab cutting?
  4. What happened to the ones that were NOT cut?

Output: diagnostic JSON + audit PNG per page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config
from src.slab_v2.pipeline import extract_slabs_v2


_XCROSS_PDF = Path(r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")
_XCROSS_PAGES = [5, 7, 9, 10, 16, 17]  # 0-indexed: pages 6, 8, 10, 11, 17, 18

if not _XCROSS_PDF.exists():
    pytest.skip("X-cross test PDF not found: Combined Structural.pdf",
                allow_module_level=True)

_CASES = [(_XCROSS_PDF, pi) for pi in _XCROSS_PAGES]


@pytest.fixture(scope="module")
def diagnostic_dir() -> Path:
    d = Path(__file__).resolve().parent / "xcross_diagnostic"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def cfg_diag() -> SlabV2Config:
    return SlabV2Config(
        debug_images=True,
        enable_opening_judge=False,
        enable_slab_face_judge=False,
        enable_floor_system_judge=False,
    )


def _build_diagnostic(result) -> dict:
    """Build diagnostic report for X-cross detection chain."""
    detected_elements = [
        e for e in result.elements
        if e.type in {"VOID", "STAIR", "LIFT", "SHAFT", "DUCT"}
    ]

    opening_candidates = [
        c for c in result.opening_candidates
        if c.get("kind_hint") in {
            "SLAB_PENETRATION", "VOID", "SHAFT", "LIFT",
            "STAIR_OPENING", "STAIRWELL"}
    ]

    verified_cuts = list(result.verified_cut_openings)

    uncut_elements = []
    cut_labels = {e.label for e in verified_cuts}
    for elem in detected_elements:
        if elem.label not in cut_labels:
            uncut_elements.append({
                "type": elem.type,
                "label": elem.label,
                "area_pt2": elem.area_pt2,
                "opening_intent": elem.opening_intent,
                "bbox": list(elem.polygon.bounds),
            })

    stuck_review = [
        {
            "id": c.get("id"),
            "kind": c.get("kind_hint"),
            "action": c.get("default_action"),
            "confidence": c.get("confidence"),
            "verification": c.get("verification_status"),
            "reject_reason": c.get("reject_reason", ""),
            "intent": c.get("opening_intent"),
        }
        for c in opening_candidates
        if c.get("default_action") == "review"
    ]

    return {
        "page_index": result.page_index,
        "status": result.status,
        "scale": result.scale,
        "n_detected_elements": len(detected_elements),
        "n_opening_candidates": len(opening_candidates),
        "n_verified_cuts": len(verified_cuts),
        "n_uncut_elements": len(uncut_elements),
        "n_stuck_review": len(stuck_review),
        "detected_elements": [
            {
                "type": e.type, "label": e.label,
                "area_pt2": e.area_pt2,
                "intent": e.opening_intent,
                "bbox": list(e.polygon.bounds),
            }
            for e in detected_elements
        ],
        "verified_cuts": [
            {
                "type": e.type, "label": e.label,
                "intent": e.opening_intent,
                "evidence": e.evidence_ids,
                "bbox": list(e.polygon.bounds),
            }
            for e in verified_cuts
        ],
        "uncut_elements": uncut_elements,
        "stuck_review_candidates": stuck_review,
        "warnings": [w for w in result.warnings if "opening" in w.lower()
                      or "penetration" in w.lower() or "x-cross" in w.lower()
                      or "xcross" in w.lower()],
    }


@pytest.mark.parametrize(
    "pdf_path,page_index",
    _CASES,
    ids=[f"{p.stem}_p{pi+1}" for p, pi in _CASES])
class TestXcrossE2E:

    def test_xcross_chain_diagnostic(self, pdf_path, page_index,
                                     cfg_diag, diagnostic_dir):
        """Run pipeline and save full X-cross diagnostic report."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_diag, use_ai=True)

        diag = _build_diagnostic(result)

        out_dir = diagnostic_dir / pdf_path.stem.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"p{page_index + 1}_xcross_report.json"
        report_path.write_text(
            json.dumps(diag, indent=2, ensure_ascii=False),
            encoding="utf-8")
        assert report_path.exists()

    def test_detected_elements_inside_slab(self, pdf_path, page_index,
                                            cfg_diag):
        """Every detected X-cross element must intersect the slab polygon."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_diag, use_ai=True)
        if result.status != "OK" or not result.slabs:
            pytest.skip("no slab on this page")

        from shapely.ops import unary_union
        slab_union = unary_union(
            [s["polygon_pdf"] for s in result.slabs if s.get("polygon_pdf")])

        for elem in result.elements:
            assert elem.polygon.intersects(slab_union), (
                f"{elem.type} '{elem.label}' at {elem.polygon.bounds} "
                f"does not intersect slab")

    def test_no_xcross_has_zero_area(self, pdf_path, page_index, cfg_diag):
        """No detected element should have zero or negative area."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_diag, use_ai=True)
        for elem in result.elements:
            assert elem.area_pt2 > 0, (
                f"{elem.type} '{elem.label}' has area_pt2={elem.area_pt2}")

    def test_no_duplicate_xcross(self, pdf_path, page_index, cfg_diag):
        """No two detected elements should overlap > 50% (dedup should work)."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_diag, use_ai=True)
        elems = result.elements
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                inter = elems[i].polygon.intersection(elems[j].polygon)
                overlap = inter.area / max(
                    min(elems[i].area_pt2, elems[j].area_pt2), 1e-9)
                assert overlap < 0.50, (
                    f"elements {i} and {j} overlap {overlap:.0%}: "
                    f"{elems[i].label} vs {elems[j].label}")
