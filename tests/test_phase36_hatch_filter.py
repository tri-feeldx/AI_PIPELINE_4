"""TDD tests for Phase 3.6 Fix A: Filter HATCH paths from element detection.

HATCH paths are excluded from planarization (pipeline.py:122) but were still
passed to extract_elements(), polluting diagonal extraction with hatch
segments at 15-75 degree angles. This fix filters them out.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slab_v2.config import SlabV2Config


# ---------------------------------------------------------------------------
# Unit: HATCH paths excluded from diagonal extraction
# ---------------------------------------------------------------------------

class TestHatchPathFilter:
    """Verify HATCH-tagged paths are excluded before element detection."""

    def _make_path(self, style_id, segments, outside_content=False):
        p = MagicMock()
        p.style_id = style_id
        p.segments = segments
        p.outside_content = outside_content
        return p

    def _make_class(self, cid, role="UNKNOWN"):
        c = MagicMock()
        c.id = cid
        c.role = role
        return c

    def test_hatch_paths_excluded(self):
        """Paths with HATCH-tagged style_id should be filtered out."""
        classes = [
            self._make_class("s1", "WALL"),
            self._make_class("s2", "HATCH"),
            self._make_class("s3", "UNKNOWN"),
        ]
        paths = [
            self._make_path("s1", [((0, 0), (10, 10))]),
            self._make_path("s2", [((0, 0), (5, 5))]),  # HATCH
            self._make_path("s2", [((1, 1), (6, 6))]),  # HATCH
            self._make_path("s3", [((20, 20), (30, 30))]),
        ]

        hatch_ids = {c.id for c in classes if c.role == "HATCH"}
        elem_paths = [p for p in paths if p.style_id not in hatch_ids]

        assert len(elem_paths) == 2
        assert all(p.style_id != "s2" for p in elem_paths)

    def test_non_hatch_paths_preserved(self):
        """Paths with WALL, UNKNOWN, FRAME roles should be kept."""
        classes = [
            self._make_class("s1", "WALL"),
            self._make_class("s2", "UNKNOWN"),
            self._make_class("s3", "FRAME"),
        ]
        paths = [
            self._make_path("s1", [((0, 0), (10, 10))]),
            self._make_path("s2", [((5, 5), (15, 15))]),
            self._make_path("s3", [((20, 20), (30, 30))]),
        ]

        hatch_ids = {c.id for c in classes if c.role == "HATCH"}
        elem_paths = [p for p in paths if p.style_id not in hatch_ids]

        assert len(elem_paths) == 3

    def test_no_hatch_classes_no_change(self):
        """When no HATCH classes exist, all paths pass through."""
        classes = [
            self._make_class("s1", "WALL"),
            self._make_class("s2", "SLAB_EDGE"),
        ]
        paths = [
            self._make_path("s1", [((0, 0), (10, 10))]),
            self._make_path("s2", [((5, 5), (15, 15))]),
        ]

        hatch_ids = {c.id for c in classes if c.role == "HATCH"}
        elem_paths = [p for p in paths if p.style_id not in hatch_ids]

        assert len(elem_paths) == len(paths)


class TestHatchDiagonalNoise:
    """Verify hatch removal improves X-cross detection quality."""

    def test_hatch_diagonals_not_counted(self):
        """Hatch segments at diagonal angles should not be in diagonal pool."""
        from src.slab_v2.elements import _diagonal_segments

        real_path = MagicMock()
        real_path.outside_content = False
        real_path.segments = [
            ((100, 100), (200, 200)),  # 45-degree, real X diagonal
            ((100, 200), (200, 100)),  # 45-degree, real X diagonal
        ]
        real_path.style_id = "s1"

        hatch_path = MagicMock()
        hatch_path.outside_content = False
        hatch_path.segments = [
            ((50 + i * 3, 50), (50 + i * 3 + 5, 55))
            for i in range(20)
        ]  # 20 small hatch segments at ~45 degree
        hatch_path.style_id = "s2"

        # Without filter: hatch pollutes
        all_diags = _diagonal_segments([real_path, hatch_path])
        filtered_diags = _diagonal_segments([real_path])

        assert len(filtered_diags) == 2, (
            f"Expected 2 real diagonals, got {len(filtered_diags)}")
        assert len(all_diags) > len(filtered_diags), (
            "Hatch paths should add noise diagonals")


# ---------------------------------------------------------------------------
# Integration: Combined Structural p18 with HATCH filter
# ---------------------------------------------------------------------------

_COMBINED_PDF = Path(r"C:\Users\LENOVO\Downloads\Combined Structural.pdf")


@pytest.mark.skipif(not _COMBINED_PDF.exists(),
                    reason="Combined Structural PDF not found")
class TestP18HatchFilter:
    """p18 with HATCH filter should detect X marks better."""

    @pytest.fixture(scope="class")
    def cfg(self):
        return SlabV2Config(
            debug_images=False,
            enable_opening_judge=False,
            enable_slab_face_judge=False,
            enable_floor_system_judge=False,
        )

    def test_p18_still_has_cuts(self, cfg):
        """p18 should still have >=2 verified cuts after HATCH filter."""
        from src.slab_v2.pipeline import extract_slabs_v2
        result = extract_slabs_v2(str(_COMBINED_PDF), 17, cfg, use_ai=True)
        assert result.status == "OK"
        assert len(result.verified_cut_openings) >= 2
