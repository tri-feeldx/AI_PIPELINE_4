from src.pdf_processor import (
    classify_page_role_from_blocks,
    collect_scale_candidates_from_blocks,
    detect_scale_from_blocks,
)


def _block(text, y=100.0, size=10.0):
    return {"text": text, "bbox": (0.0, y, 100.0, y + 10.0), "size": size}


def test_scale_parser_rejects_timestamp_and_chooses_explicit_scale():
    blocks = [
        _block("12/09/2024 1:15 PM"),
        _block("GENERAL ARRANGEMENT PLAN - LEVEL 07", y=2200, size=18),
        _block("SCALE - 1 : 100", y=2250),
    ]

    audit = collect_scale_candidates_from_blocks(blocks)

    assert detect_scale_from_blocks(blocks) == 100
    assert audit["chosen_scale"] == 100
    rejected = [c for c in audit["candidates"] if c["denominator"] == 15]
    assert rejected
    assert rejected[0]["reject_reason"] == "timestamp_or_time_ratio"


def test_scale_parser_does_not_turn_do_not_scale_timestamp_into_scale():
    blocks = [
        _block("SCALE @ A0 DO NOT SCALE As indicated"),
        _block("Printed 1:15 PM"),
    ]

    audit = collect_scale_candidates_from_blocks(blocks)

    assert audit["chosen_scale"] is None
    assert detect_scale_from_blocks(blocks) is None


def test_page_role_general_arrangement_is_geometry_plan():
    role = classify_page_role_from_blocks([
        _block("GENERAL ARRANGEMENT PLAN - LEVEL 07", size=18),
        _block("SCALE - 1 : 100"),
    ])

    assert role["role"] == "geometry_plan"


def test_page_role_foundation_title_wins_over_incidental_floor_words():
    role = classify_page_role_from_blocks([
        _block("FOUNDATION PLAN", size=18),
        _block("REFER ROOF STEEL SCHEDULE AND LEVEL 05 NOTES"),
        _block("SCALE - 1 : 100"),
    ])

    assert role["role"] == "foundation_plan"


def test_page_role_loading_plan_is_evidence_not_slab_authority():
    role = classify_page_role_from_blocks([
        _block("LEVEL 04 - LOADING PLAN", size=18),
        _block("GENERAL FLOOR LOADS AND ROOF LOADS"),
        _block("SCALE - 1 : 100"),
    ])

    assert role["role"] == "evidence_only"
