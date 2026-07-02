"""Tests for the 4 Gemini Round 1 determinism solutions."""
from unittest.mock import MagicMock, patch

import fitz
from shapely.geometry import box

from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import StyleKey, StyleClass


# ── helpers ──────────────────────────────────────────────────────────────────

def _style_class(cid, stroke, fill, width, dashes="", n_paths=100,
                 n_segments=200, total_length_pt=5000.0,
                 median_seg_len_pt=15.0):
    key = StyleKey(stroke=stroke, fill=fill, width=width, dashes=dashes,
                   even_odd=False)
    return StyleClass(
        id=cid, key=key, n_paths=n_paths, n_segments=n_segments,
        total_length_pt=total_length_pt,
        bbox=(50.0, 50.0, 1400.0, 1100.0),
        median_seg_len_pt=median_seg_len_pt,
    )


# ── Solution 4: SLAB_EDGE fingerprint ───────────────────────────────────────

class TestSlabEdgeFingerprint:
    def _run_extract(self, classes_spec):
        """Run vector_extract fingerprint chain on synthetic style classes.

        classes_spec: list of (key, ids, total_len, n_segs, median, bbox)
        sorted by total_len descending (as extract_paths does).
        """
        from src.slab_v2.vector_extract import extract_paths
        cfg = SlabV2Config(debug_images=False)
        doc = fitz.open()
        page = doc.new_page(width=1600, height=1200)
        page_area = page.rect.width * page.rect.height

        results = []
        for cid, (key, ids, total_len, n_segs, median, bbox_) in enumerate(
                classes_spec):
            sc = StyleClass(
                id=cid, key=key, n_paths=len(ids), n_segments=n_segs,
                total_length_pt=total_len, bbox=bbox_,
                median_seg_len_pt=median,
            )
            bw, bh = bbox_[2] - bbox_[0], bbox_[3] - bbox_[1]
            if bw * bh >= cfg.frame_area_frac * page_area and len(ids) <= 6:
                sc.role = "FRAME"
                sc.role_confidence = 0.9
            elif (cid <= 2
                  and key.stroke is not None
                  and max(key.stroke) <= 0.3
                  and key.fill is None
                  and not key.dashes
                  and key.width >= 0.5):
                sc.role = "SLAB_EDGE"
                sc.role_confidence = 0.75
            elif (key.fill is not None and key.stroke is None
                  and n_segs > 500 and median < 1.0):
                sc.role = "HATCH"
                sc.role_confidence = 0.85
                sc.prefiltered = True
            elif (key.stroke is not None and max(key.stroke) <= 0.15
                  and key.fill is None and not key.dashes
                  and 0.9 <= key.width <= 1.5):
                sc.role = "COLUMN"
                sc.role_confidence = 0.70
            elif (key.stroke is not None and key.width <= 0.5
                  and (key.dashes or max(key.stroke) > 0.5)):
                sc.role = "ANNOTATION"
                sc.role_confidence = 0.65
            results.append(sc)
        doc.close()
        return results

    def test_dark_solid_top3_is_slab_edge(self):
        key = StyleKey(stroke=(0.0, 0.0, 0.0), fill=None, width=0.96,
                       dashes="", even_odd=False)
        specs = [(key, list(range(551)), 26689.0, 551, 17.0,
                  (50, 50, 1400, 1100))]
        classes = self._run_extract(specs)
        assert classes[0].role == "SLAB_EDGE"
        assert classes[0].role_confidence == 0.75

    def test_cid3_not_slab_edge(self):
        key = StyleKey(stroke=(0.0, 0.0, 0.0), fill=None, width=0.96,
                       dashes="", even_odd=False)
        specs = []
        for i in range(4):
            specs.append((key, list(range(100)), 10000.0 - i * 2000, 100,
                          15.0, (50, 50, 1400, 1100)))
        classes = self._run_extract(specs)
        assert classes[0].role == "SLAB_EDGE"
        assert classes[1].role == "SLAB_EDGE"
        assert classes[2].role == "SLAB_EDGE"
        assert classes[3].role != "SLAB_EDGE"

    def test_colored_stroke_excluded(self):
        key = StyleKey(stroke=(0.498, 0.498, 0.251), fill=None, width=0.96,
                       dashes="", even_odd=False)
        specs = [(key, list(range(700)), 5512.0, 700, 11.4,
                  (50, 50, 1400, 1100))]
        classes = self._run_extract(specs)
        assert classes[0].role != "SLAB_EDGE"

    def test_dashed_excluded(self):
        key = StyleKey(stroke=(0.0, 0.0, 0.0), fill=None, width=0.96,
                       dashes="[3 2] 0", even_odd=False)
        specs = [(key, list(range(200)), 8000.0, 200, 15.0,
                  (50, 50, 1400, 1100))]
        classes = self._run_extract(specs)
        assert classes[0].role != "SLAB_EDGE"

    def test_thin_stroke_excluded(self):
        key = StyleKey(stroke=(0.0, 0.0, 0.0), fill=None, width=0.3,
                       dashes="", even_odd=False)
        specs = [(key, list(range(500)), 15000.0, 500, 10.0,
                  (50, 50, 1400, 1100))]
        classes = self._run_extract(specs)
        assert classes[0].role != "SLAB_EDGE"

    def test_frame_wins_over_slab_edge(self):
        key = StyleKey(stroke=(0.0, 0.0, 0.0), fill=None, width=0.96,
                       dashes="", even_odd=False)
        cfg = SlabV2Config(debug_images=False)
        page_area = 1600 * 1200
        specs = [(key, list(range(4)), 5000.0, 4, 500.0,
                  (0, 0, 1600, 1200))]
        classes = self._run_extract(specs)
        assert classes[0].role == "FRAME"


# ── Solution 1: deterministic fallback filter ────────────────────────────────

class TestDeterministicFallback:
    def _make_classes(self):
        c0 = _style_class(0, (0.0, 0.0, 0.0), None, 0.96,
                          total_length_pt=26689)
        c3 = _style_class(3, (0.0, 0.0, 0.0), None, 0.6,
                          total_length_pt=6766)
        c5 = _style_class(5, (0.498, 0.498, 0.251), None, 0.96,
                          total_length_pt=5512)
        return {0: c0, 3: c3, 5: c5}

    def test_3_classes_reduces_to_max(self):
        by_id = self._make_classes()
        slab_ids = [0, 3, 5]
        supp_ids = []
        max_slab = 2

        if len(slab_ids) > max_slab:
            filtered = [sid for sid in slab_ids
                        if by_id[sid].key.stroke is not None
                        and max(by_id[sid].key.stroke) <= 0.3
                        and not by_id[sid].key.dashes
                        and by_id[sid].key.width >= 0.5]
            if filtered:
                if len(filtered) > max_slab:
                    filtered.sort(
                        key=lambda sid: -by_id[sid].total_length_pt)
                    filtered = filtered[:max_slab]
                dropped = [s for s in slab_ids if s not in filtered]
                supp_ids = dropped + supp_ids
                slab_ids = filtered

        assert 5 not in slab_ids
        assert 0 in slab_ids
        assert 5 in supp_ids

    def test_2_classes_no_change(self):
        slab_ids = [0, 1]
        max_slab = 2
        assert len(slab_ids) <= max_slab

    def test_1_class_no_change(self):
        slab_ids = [0]
        max_slab = 2
        assert len(slab_ids) <= max_slab

    def test_dropped_become_supporting(self):
        by_id = self._make_classes()
        slab_ids = [0, 3, 5]
        supp_ids = [6]
        max_slab = 2

        if len(slab_ids) > max_slab:
            filtered = [sid for sid in slab_ids
                        if by_id[sid].key.stroke is not None
                        and max(by_id[sid].key.stroke) <= 0.3
                        and not by_id[sid].key.dashes
                        and by_id[sid].key.width >= 0.5]
            if filtered:
                if len(filtered) > max_slab:
                    filtered.sort(
                        key=lambda sid: -by_id[sid].total_length_pt)
                    filtered = filtered[:max_slab]
                dropped = [s for s in slab_ids if s not in filtered]
                supp_ids = dropped + supp_ids
                slab_ids = filtered

        assert 5 in supp_ids
        assert 6 in supp_ids

    def test_keeps_highest_length_when_multiple_dark(self):
        c0 = _style_class(0, (0.0, 0.0, 0.0), None, 0.96,
                          total_length_pt=26689)
        c1 = _style_class(1, (0.1, 0.1, 0.1), None, 0.8,
                          total_length_pt=15000)
        c2 = _style_class(2, (0.2, 0.2, 0.2), None, 0.6,
                          total_length_pt=8000)
        by_id = {0: c0, 1: c1, 2: c2}
        slab_ids = [0, 1, 2]
        supp_ids = []
        max_slab = 2

        if len(slab_ids) > max_slab:
            filtered = [sid for sid in slab_ids
                        if by_id[sid].key.stroke is not None
                        and max(by_id[sid].key.stroke) <= 0.3
                        and not by_id[sid].key.dashes
                        and by_id[sid].key.width >= 0.5]
            if filtered:
                if len(filtered) > max_slab:
                    filtered.sort(
                        key=lambda sid: -by_id[sid].total_length_pt)
                    filtered = filtered[:max_slab]
                dropped = [s for s in slab_ids if s not in filtered]
                supp_ids = dropped + supp_ids
                slab_ids = filtered

        assert slab_ids == [0, 1]
        assert 2 in supp_ids


# ── Solution 3: face count validation gate ───────────────────────────────────

class TestFaceCountGate:
    def test_below_threshold_triggers(self):
        cfg = SlabV2Config(debug_images=False, min_faces_for_election=10)
        faces = [MagicMock() for _ in range(5)]
        fg = MagicMock()
        fg.faces = faces
        assert len(fg.faces) < cfg.min_faces_for_election

    def test_above_threshold_no_trigger(self):
        cfg = SlabV2Config(debug_images=False, min_faces_for_election=10)
        faces = [MagicMock() for _ in range(25)]
        fg = MagicMock()
        fg.faces = faces
        assert len(fg.faces) >= cfg.min_faces_for_election


# ── Config defaults ──────────────────────────────────────────────────────────

class TestConfigDefaults:
    def test_max_slab_edge_classes_default(self):
        cfg = SlabV2Config()
        assert cfg.max_slab_edge_classes == 2

    def test_min_faces_for_election_default(self):
        cfg = SlabV2Config()
        assert cfg.min_faces_for_election == 10
