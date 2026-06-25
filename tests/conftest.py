"""Shared fixtures for the slab_v2 test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.slab_v2.config import SlabV2Config

# ---------------------------------------------------------------------------
# Known test PDF locations (scan common workspace paths)
# ---------------------------------------------------------------------------
_SEARCH_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT.parent,
    Path(r"D:\FeelDX_Workspace\AI-PIPELINE\data\jobs"),
    Path(r"D:\FeelDX_Workspace\sketchup_auto_project\data\input_pdf"),
]

_PDF_NAMES = [
    "combine strc.pdf",
    "Structural.pdf",
    "2019-St Carloa Moama-STRUC-Com.pdf",
]


def _find_pdfs() -> list[Path]:
    """Find unique test PDFs across workspace (one copy per name)."""
    found: dict[str, Path] = {}
    for search_dir in _SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for name in _PDF_NAMES:
            if name in found:
                continue
            for match in search_dir.rglob(name):
                found[name] = match
                break
    return list(found.values())


AVAILABLE_PDFS = _find_pdfs()


@pytest.fixture
def slabv2_config() -> SlabV2Config:
    """Standard config with debug images enabled for checkpoint verification."""
    return SlabV2Config(
        debug_images=True,
        enable_opening_judge=False,
        enable_slab_face_judge=False,
        enable_floor_system_judge=False,
    )


@pytest.fixture
def slabv2_config_fast() -> SlabV2Config:
    """Fast config without AI judges — for unit tests that need speed."""
    return SlabV2Config(
        debug_images=False,
        enable_opening_judge=False,
        enable_slab_face_judge=False,
        enable_floor_system_judge=False,
    )


@pytest.fixture
def snapshot_dir() -> Path:
    """Directory for storing test snapshots."""
    d = PROJECT_ROOT / "tests" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pdf_parametrize():
    """Decorator: skip test if no PDFs available."""
    if not AVAILABLE_PDFS:
        return pytest.mark.skip(reason="No test PDFs found")
    return pytest.mark.parametrize(
        "pdf_path", AVAILABLE_PDFS, ids=[p.stem for p in AVAILABLE_PDFS])
