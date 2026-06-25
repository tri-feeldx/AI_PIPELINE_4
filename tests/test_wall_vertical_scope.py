import fitz

from src.slab_v2.doc_analyze import collect_page_wall_scope_evidence


def _word(x0, y0, x1, y1, text, block, line):
    return (x0, y0, x1, y1, text, block, line, 0)


def test_wall_under_reference_is_not_current_wall():
    words = [
        _word(100, 100, 120, 110, "W1", 1, 0),
        _word(122, 100, 140, 110, "(U)", 1, 0),
        _word(200, 100, 220, 110, "LW1", 2, 0),
    ]
    evidence = collect_page_wall_scope_evidence(
        words, fitz.Rect(0, 0, 500, 500), {"W1", "LW1"})
    assert evidence["W1"] == {
        "current": 0, "under_only": 1, "over_only": 0}
    assert evidence["LW1"]["current"] == 1


def test_wall_legend_outside_content_is_ignored():
    words = [
        _word(900, 100, 920, 110, "W1", 3, 0),
        _word(100, 100, 120, 110, "W2", 4, 0),
    ]
    evidence = collect_page_wall_scope_evidence(
        words, fitz.Rect(0, 0, 800, 500), {"W1", "W2"})
    assert "W1" not in evidence
    assert evidence["W2"]["current"] == 1


def test_vertical_wall_suffix_may_precede_symbol_in_pdf_word_order():
    words = [
        _word(100, 80, 110, 90, "(U)", 5, 0),
        _word(100, 92, 110, 110, "W2", 5, 0),
    ]
    evidence = collect_page_wall_scope_evidence(
        words, fitz.Rect(0, 0, 500, 500), {"W2"})
    assert evidence["W2"] == {
        "current": 0, "under_only": 1, "over_only": 0}
