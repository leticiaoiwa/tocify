"""
Characterization tests for digest.keyword_prefilter().

Function signature:
    keyword_prefilter(items: list[dict], keywords: list[str], keep_top: int) -> list[dict]

Key behaviours captured here:
- Matches on title substring (lowercased)
- Matches on summary substring (lowercased)
- Items with zero hits are excluded when enough items match (non-fallback path)
- Matching is case-insensitive (both keyword and text are lowercased)
- Empty items list returns []

Fallback note: when fewer than min(50, keep_top) items match, the function returns
items[:keep_top] (all items, not just matched ones). Tests that verify exclusion of
non-matching items are written to keep_top=2 with exactly 2 matching items so that
len(matched) == min(50, keep_top) == 2, which does NOT trigger the fallback.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from digest import keyword_prefilter


def item(title: str = "", summary: str = "") -> dict:
    """Minimal paper dict with only the fields keyword_prefilter inspects."""
    return {"title": title, "summary": summary}


# ---------------------------------------------------------------------------
# Case 5: empty input
# ---------------------------------------------------------------------------

def test_empty_items_returns_empty_list():
    result = keyword_prefilter([], ["machine learning", "NLP"], keep_top=10)
    assert result == []


# ---------------------------------------------------------------------------
# Case 1: title match passes the filter
# ---------------------------------------------------------------------------

def test_title_match_included_in_result():
    paper = item(title="Advances in transformer architectures", summary="No relevant text here.")
    result = keyword_prefilter([paper], ["transformer"], keep_top=10)
    assert paper in result


# ---------------------------------------------------------------------------
# Case 2: abstract / summary match passes the filter
# ---------------------------------------------------------------------------

def test_summary_match_included_in_result():
    paper = item(title="An unrelated title", summary="We apply diffusion models to image synthesis.")
    result = keyword_prefilter([paper], ["diffusion models"], keep_top=10)
    assert paper in result


# ---------------------------------------------------------------------------
# Case 3: paper matching no keyword is excluded
#
# Setup that avoids the fallback path:
#   keep_top=2, 2 matching items, 1 non-matching item
#   → min(50, 2) = 2; len(matched) = 2; 2 < 2 is False → no fallback
#   → function returns matched[:2], which does not contain the non-matching item
# ---------------------------------------------------------------------------

def test_no_keyword_match_excluded():
    match_a = item(title="Deep learning for protein folding")
    match_b = item(title="Deep learning in drug discovery")
    no_match = item(title="A cookbook for beginners", summary="Recipes and cooking tips.")

    result = keyword_prefilter(
        [match_a, match_b, no_match],
        keywords=["deep learning"],
        keep_top=2,
    )

    assert no_match not in result
    assert match_a in result
    assert match_b in result


# ---------------------------------------------------------------------------
# Case 4: matching is case-insensitive
# ---------------------------------------------------------------------------

def test_uppercase_title_matched_by_lowercase_keyword():
    paper = item(title="REINFORCEMENT LEARNING FROM HUMAN FEEDBACK")
    result = keyword_prefilter([paper], ["reinforcement learning"], keep_top=10)
    assert paper in result


def test_mixed_case_keyword_matched_against_lowercase_title():
    paper = item(title="reinforcement learning from human feedback")
    result = keyword_prefilter([paper], ["Reinforcement Learning"], keep_top=10)
    assert paper in result


def test_mixed_case_summary_matched_by_mixed_case_keyword():
    paper = item(title="Unrelated title", summary="We explore Graph Neural Networks for molecule property prediction.")
    result = keyword_prefilter([paper], ["GRAPH NEURAL NETWORKS"], keep_top=10)
    assert paper in result
