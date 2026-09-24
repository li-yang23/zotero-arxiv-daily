import pytest

from zotero_arxiv_daily.request_parser import parse_paper_list


def test_parse_paper_list_supports_real_world_heading_link_and_bare_title_format():
    markdown = """
# CCS 2026

Private Direct Preference Optimization for LLM Alignment

# NDSS 2026

[Attention is All You Need to Defend](https://www.ndss-symposium.org/ndss-paper/example/)

1. Numbered Paper
- Bulleted Paper
"""

    groups = parse_paper_list(markdown)

    assert [group.label for group in groups] == ["CCS 2026", "NDSS 2026"]
    assert [paper.title for paper in groups[0].papers] == [
        "Private Direct Preference Optimization for LLM Alignment"
    ]
    assert [paper.title for paper in groups[1].papers] == [
        "Attention is All You Need to Defend",
        "Numbered Paper",
        "Bulleted Paper",
    ]
    assert groups[1].papers[0].url == "https://www.ndss-symposium.org/ndss-paper/example/"
    assert [paper.input_index for group in groups for paper in group.papers] == [0, 1, 2, 3]


def test_parse_paper_list_deduplicates_exact_repeated_entries():
    groups = parse_paper_list("# Venue\nPaper One\nPaper One\n")

    assert len(groups) == 1
    assert [paper.title for paper in groups[0].papers] == ["Paper One"]


def test_parse_paper_list_rejects_too_many_papers():
    with pytest.raises(ValueError, match="more than 1 papers"):
        parse_paper_list("Paper One\nPaper Two", max_papers=1)


def test_parse_paper_list_rejects_unsafe_link_scheme():
    with pytest.raises(ValueError, match="Unsupported paper URL"):
        parse_paper_list("[Paper](file:///etc/passwd)")
