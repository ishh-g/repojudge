"""Tests for the shared keyword-matching and claim-similarity helpers."""
from __future__ import annotations

from app import keywords


def test_match_terms_tries_simple_variants():
    assert keywords.match_terms("Chart.js") == ["chart.js", "chart", "chartjs"]
    # A short leading part is dropped, but the compact form is kept.
    assert keywords.match_terms("my-lib") == ["my-lib", "mylib"]


def test_match_terms_drops_terms_that_are_too_short():
    assert keywords.match_terms("a") == []
    assert keywords.match_terms("") == []


def test_match_terms_keeps_a_plain_name():
    assert keywords.match_terms("requests") == ["requests"]


def test_searchable_keywords_ignores_stopwords_and_short_words():
    found = keywords.searchable_keywords(["the", "a", "requests", "FLASK"])
    assert [keyword for keyword, _terms in found] == ["requests", "flask"]


def test_score_line_prefers_whole_words_and_code():
    plain = keywords.score_line("# a note about requests", ["requests"])
    code = keywords.score_line("import requests", ["requests"])
    assert plain > 0
    assert code > plain


def test_score_line_returns_zero_without_a_match():
    assert keywords.score_line("def main():", ["requests"]) == 0


def test_score_line_is_case_insensitive():
    assert keywords.score_line("NEW CHART(ctx)", keywords.match_terms("chart.js")) > 0


def test_significant_tokens_ignores_broad_words():
    tokens = keywords.significant_tokens("Uses the requests library for the web")
    assert tokens == {"requests"}


def test_is_duplicate_needs_a_distinctive_shared_term():
    shared_term = {"frobnicator", "motion"}
    other = {"frobnicator"}
    assert keywords.is_duplicate(shared_term, other, {"frobnicator": 2})

    short_terms = {"req"}
    assert not keywords.is_duplicate(short_terms, {"req"}, {"req": 2})

    common = {"requests", "json"}
    assert not keywords.is_duplicate(common, {"requests", "yaml"},
                                     {"requests": 5, "json": 1})


def test_normalise_text_ignores_punctuation_and_case():
    assert keywords.normalise_text("Has a CLI!") == "hasacli"
    assert keywords.normalise_text("  has-a  cli ") == "hasacli"
    assert keywords.normalise_text("") == ""


def test_merge_keywords_dedupes_and_caps():
    assert keywords.merge_keywords(["a", "b"], ["B", "c"]) == ["a", "b", "c"]
    assert len(keywords.merge_keywords([f"k{i}" for i in range(10)], [])) == 6
    assert keywords.merge_keywords(None, ["x"]) == ["x"]


# ---------------------------------------------------------------------------
# Versions and separator-insensitive name matching
# ---------------------------------------------------------------------------


def test_versions_are_removed_before_matching():
    assert keywords.strip_version("Chart.js 4.4") == "Chart.js"
    assert keywords.strip_version("FontAwesome v3") == "FontAwesome"
    assert keywords.strip_version("Vue 3.x") == "Vue"
    assert keywords.match_terms("Chart.js 4.4") == ["chart.js", "chart", "chartjs"]
    assert "fontawesome" in keywords.strong_match_terms("font-awesome 6.4")


def test_extract_version_reads_claim_and_cdn_forms():
    assert keywords.extract_version("Uses Chart.js 4.4 for charts") == (4, 4)
    assert keywords.extract_version("Uses FontAwesome 6.4") == (6, 4)
    assert keywords.extract_version("chart.js@4.4.1") == (4, 4)
    assert keywords.extract_version(
        "https://cdn.example.com/font-awesome/6.4.0/css/all.min.css") == (6, 4)
    assert keywords.extract_version("v3") == (3, 0)
    assert keywords.extract_version("3.x") == (3, 0)
    assert keywords.extract_version("Reads 21 hand landmarks") is None


def test_separator_insensitive_matching_covers_every_spelling():
    """font awesome / font-awesome / font_awesome / fontawesome are one name."""
    terms = keywords.strong_match_terms("fontawesome")
    for spelling in ("font awesome", "font-awesome", "font_awesome",
                     "FontAwesome", "fontawesome-free"):
        assert keywords.score_line(spelling, terms) > 0, spelling


def test_a_two_word_tool_matches_either_word_and_prefers_both():
    terms = keywords.strong_match_terms("font awesome")
    both = keywords.score_line("load font-awesome icons", terms)
    one = keywords.score_line("load the awesome icons", terms)
    assert both > 0 and one > 0
    assert both > one


def test_cdn_file_name_styles_match_the_library():
    terms = keywords.strong_match_terms("chart.js")
    assert keywords.score_line("chart.umd.js", terms) > 0
    assert keywords.score_line("chartjs", terms) > 0
