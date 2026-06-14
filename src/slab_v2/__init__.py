"""
slab_v2 — Deterministic-geometry slab polygon extraction.

Architecture: deterministic vector kernel + AI as a SELECTOR.
All polygon coordinates come straight from the PDF vector data;
Gemini only picks style-class IDs and face IDs, never coordinates.
"""

from src.slab_v2.pipeline import extract_slabs_v2
from src.slab_v2.config import SlabV2Config
from src.slab_v2.models import SlabV2Result

__all__ = ["extract_slabs_v2", "SlabV2Config", "SlabV2Result"]
