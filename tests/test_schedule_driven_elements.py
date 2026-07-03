"""Integration: columns/walls driven by the sheet's OWN schedule (Phase 2.2).

No Gemini census, no per-file flags: extract_slabs_v2 with use_ai=False
must pull element types from the on-page schedules and detect instances.
Baselines measured 2026-07-02 (floors, not exact counts, so richer future
detection cannot fail these).
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from src.slab_v2.config import SlabV2Config

PDF_2381 = Path(r"C:\Users\LENOVO\Downloads\2381_MSCP_STR_Combine.pdf")
PDF_SMPS = Path(r"C:\Users\LENOVO\Downloads"
                r"\2402. South Melbourne Primary School - CIVIL & STR - 260610.pdf")


@pytest.fixture(scope="module")
def cfg():
    return SlabV2Config(debug_images=False, enable_opening_judge=False,
                        enable_slab_face_judge=False,
                        enable_floor_system_judge=False)


@pytest.mark.skipif(not PDF_2381.exists(), reason="2381 PDF not present")
def test_2381_p17_columns_and_walls_from_schedule(cfg):
    from src.slab_v2.pipeline import extract_slabs_v2
    r = extract_slabs_v2(str(PDF_2381), 16, cfg, use_ai=False)
    assert r.status == "OK"
    # baseline: 20 columns (15 text-anchored), symbols from the schedule
    assert len(r.columns) >= 15
    syms = {c.symbol for c in r.columns} - {"C?"}
    assert syms <= {"C-A1", "C-A2", "C-A3", "C-B", "C-C", "C-D", "C-E", "C-F"}
    # baseline: 12 walls, labels from the wall schedule
    assert len(r.walls) >= 10
    assert {w.label for w in r.walls} <= {"BW1", "IW20", "IW25", "IW30",
                                          "IW35", "NLB1"}


@pytest.mark.skipif(not PDF_SMPS.exists(), reason="SMPS PDF not present")
def test_smps_p9_rc_and_steel_from_schedule(cfg):
    from src.slab_v2.pipeline import extract_slabs_v2
    r = extract_slabs_v2(str(PDF_SMPS), 8, cfg, use_ai=False)
    assert r.status == "OK"
    # concrete: CC1 circular columns detected
    assert any(c.symbol == "CC1" for c in r.columns)
    # steel: the STEEL COLUMN SCHEDULE marks reach the steel subsystem
    steel_syms = {m.symbol for m in r.steel_members}
    assert len(r.steel_members) >= 15
    assert {"C2", "C3", "C5"} <= steel_syms
