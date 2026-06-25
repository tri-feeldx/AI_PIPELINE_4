"""Regression tests: run slab_v2 pipeline on real PDFs, assert basic invariants.

These tests protect against regressions — if the pipeline crashes or produces
garbage after a code change, these tests catch it BEFORE shipping.

Tests are parametric: they run on every available test PDF.  If no PDFs are
found, they are skipped (CI without test data still passes).

Each test run saves a snapshot JSON for checkpoint verification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config
from src.slab_v2.pipeline import extract_slabs_v2
from tests.conftest import AVAILABLE_PDFS


def _slab_pages(pdf_path: Path) -> list[int]:
    """Heuristic: pick up to 4 pages likely to be slab plans (skip first few)."""
    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    doc.close()
    if n <= 5:
        return list(range(n))
    candidates = []
    for i in range(4, min(n, 15)):
        candidates.append(i)
        if len(candidates) >= 4:
            break
    return candidates or [0]


def _snapshot_path(snapshot_dir: Path, pdf_stem: str, page_idx: int) -> Path:
    d = snapshot_dir / pdf_stem.replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"p{page_idx + 1}.json"


def _result_to_dict(result) -> dict:
    """Serialize SlabV2Result to a JSON-safe dict for snapshot."""
    return {
        "page_index": result.page_index,
        "status": result.status,
        "scale": result.scale,
        "n_slabs": len(result.slabs) if result.slabs else 0,
        "n_elements": len(result.elements),
        "n_columns": len(result.columns),
        "n_walls": len(result.walls),
        "n_verified_openings": len(result.verified_cut_openings),
        "n_steel_members": len(result.steel_members),
        "n_warnings": len(result.warnings),
        "timings": {k: round(v, 3) for k, v in result.timings.items()},
        "column_status": result.column_detection_report.get("status", ""),
        "wall_status": result.wall_readiness.get("status", ""),
        "steel_status": result.steel_readiness.get("status", ""),
        "slab_readiness": result.slab_readiness.get("status", ""),
        "warnings": result.warnings[:20],
    }


# Skip entire module if no PDFs
if not AVAILABLE_PDFS:
    pytest.skip("No test PDFs found in workspace", allow_module_level=True)

# Build parametrize list: (pdf_path, page_index)
_TEST_CASES = []
for pdf in AVAILABLE_PDFS:
    for pi in _slab_pages(pdf):
        _TEST_CASES.append((pdf, pi))


@pytest.fixture(scope="module")
def cfg_regression() -> SlabV2Config:
    return SlabV2Config(
        debug_images=True,
        enable_opening_judge=False,
        enable_slab_face_judge=False,
        enable_floor_system_judge=False,
    )


@pytest.fixture(scope="module")
def snapshot_base() -> Path:
    d = Path(__file__).resolve().parent / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.parametrize(
    "pdf_path,page_index",
    _TEST_CASES,
    ids=[f"{p.stem}_p{pi+1}" for p, pi in _TEST_CASES])
class TestRegressionPipeline:

    def test_pipeline_does_not_crash(self, pdf_path, page_index, cfg_regression):
        """The pipeline must complete without unhandled exceptions."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_regression, use_ai=True)
        assert result is not None
        assert result.status in {"OK", "NO_FACES", "NO_AI", "VERIFY_FAILED"}

    def test_slab_produced_when_ok(self, pdf_path, page_index, cfg_regression):
        """When status=OK, there must be at least one slab polygon."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_regression, use_ai=True)
        if result.status == "OK":
            assert result.slabs, f"status=OK but no slabs on page {page_index}"
            for slab in result.slabs:
                poly = slab.get("polygon_pdf") or slab.get("polygon_mm")
                assert poly is not None, f"slab has no polygon"
                assert poly.area > 0, f"slab polygon has zero area"

    def test_scale_detected(self, pdf_path, page_index, cfg_regression):
        """Scale must be detected (text or dimension-calibrated)."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_regression, use_ai=True)
        if result.status == "OK":
            assert result.scale is not None, "scale is None on OK page"
            assert result.scale > 0, f"invalid scale: {result.scale}"

    def test_all_pipeline_stages_ran(self, pdf_path, page_index, cfg_regression):
        """Timings dict must contain keys for all major stages."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_regression, use_ai=True)
        if result.status == "OK":
            expected_stages = {"stage_a", "stage_b", "assembly", "verify"}
            actual = set(result.timings.keys())
            missing = expected_stages - actual
            assert not missing, f"pipeline stages missing: {missing}"

    def test_snapshot_saved(self, pdf_path, page_index,
                            cfg_regression, snapshot_base):
        """Save result snapshot for checkpoint verification."""
        result = extract_slabs_v2(
            str(pdf_path), page_index, cfg_regression, use_ai=True)
        snap = _snapshot_path(snapshot_base, pdf_path.stem, page_index)
        snap.write_text(
            json.dumps(_result_to_dict(result), indent=2, ensure_ascii=False),
            encoding="utf-8")
        assert snap.exists()
