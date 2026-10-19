"""Milestone 2 tests: the file cache and Gemma claim extraction."""
from __future__ import annotations

import json

import pytest

from app import cache, claims, config
from app.errors import GemmaResponseError
from app.github_client import fetch_repo
from fakes import FakeGitHub


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    """Point the cache at a throwaway directory for every test."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    yield


@pytest.fixture
def repo():
    fake = FakeGitHub(
        files={
            "main.py": "import requests\n",
            "requirements.txt": "requests\n",
        },
        readme=(
            "# Widget\n\n"
            "Widget uses the requests library and has a CLI. The CLI reads item "
            "data and writes JSON files. It reads the local storage history and "
            "uses a router for navigation. It uses Frobnicator for motion "
            "tracking and to track motion. It reads a plain string and uses a "
            "typed field.\n"
        ),
    )
    return fetch_repo("https://github.com/acme/widget", session=fake)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_roundtrip():
    key = cache.make_key("a", "b")
    assert cache.get(key) is None
    cache.set(key, {"hello": "world"})
    got = cache.get(key)
    assert got["hello"] == "world"
    assert got["_cache_hit"] is True


def test_cache_key_depends_on_all_parts():
    assert cache.make_key("a", "b") != cache.make_key("a", "c")
    assert cache.make_key("url", "sha", "") == cache.make_key("url", "sha", "")


def test_cache_survives_bad_json(tmp_path, monkeypatch):
    key = cache.make_key("broken")
    path = config.CACHE_DIR / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def test_extract_claims_basic(repo, monkeypatch):
    calls = []

    def fake_generate_json(prompt, **kwargs):
        calls.append(prompt)
        return {"claims": [
            {"claim": "Uses the requests library", "type": "tech_used",
             "keywords": ["Requests", "requests"]},
            {"claim": "Has a CLI", "type": "feature", "keywords": ["cli", "argparse"]},
        ]}

    monkeypatch.setattr(claims, "generate_json", fake_generate_json)
    result = claims.extract_claims(repo)

    assert len(result) == 2
    assert result[0]["type"] == "tech_used"
    assert result[0]["keywords"] == ["requests"]  # de-duped and lowercased
    assert len(calls) == 1
    # The prompt must include the README and the file list.
    assert "acme/widget" in calls[0]
    assert "main.py" in calls[0]


def test_extract_claims_caps_at_max(repo, monkeypatch):
    many = {"claims": [{"claim": f"Reads item {i}", "type": "feature",
                        "keywords": [f"kw{i}"]} for i in range(40)]}
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: many)
    result = claims.extract_claims(repo)
    assert len(result) == config.MAX_CLAIMS


def test_requirements_become_their_own_claims(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json",
                        lambda *a, **k: {"claims": [
                            {"claim": "a", "type": "feature", "keywords": ["a"]},
                        ]})
    result = claims.extract_claims(repo, ["Must support Docker deployment"])
    types = [c["type"] for c in result]
    assert "requirement" in types
    req = next(c for c in result if c["type"] == "requirement")
    assert "Docker" in req["claim"]
    assert "docker" in req["keywords"]


def test_total_claims_never_exceed_max(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json",
                        lambda *a, **k: {"claims": [
                            {"claim": f"c{i}", "type": "feature", "keywords": ["x"]}
                            for i in range(40)]})
    result = claims.extract_claims(repo, ["req one", "req two"])
    assert len(result) <= config.MAX_CLAIMS
    assert sum(1 for c in result if c["type"] == "requirement") == 2


def test_duplicate_requirement_merges_with_readme_claim(repo, monkeypatch):
    """A requirement about something the README already claims is merged."""
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "Uses Frobnicator for motion tracking", "type": "tech_used",
         "keywords": ["frobnicator", "motion", "tracking"]},
    ]})
    result = claims.extract_claims(repo, ["Check that Frobnicator is used"])
    assert len(result) == 1
    assert result[0]["type"] == "tech_used"
    assert "frobnicator" in result[0]["keywords"]


def test_unrelated_claims_are_not_merged(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "Uses the requests library", "type": "tech_used",
         "keywords": ["requests"]},
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ]})
    result = claims.extract_claims(repo, ["Uses Docker deployment"])
    assert len(result) == 3


def test_similar_but_different_claims_are_kept_apart(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "Reads JSON files", "type": "feature", "keywords": ["json"]},
        {"claim": "Writes JSON files", "type": "feature", "keywords": ["json"]},
    ]})
    result = claims.extract_claims(repo)
    assert len(result) == 2


def test_claims_are_cached(repo, monkeypatch):
    calls = {"n": 0}

    def fake(prompt, **kwargs):
        calls["n"] += 1
        return {"claims": [{"claim": "c", "type": "feature", "keywords": ["x"]}]}

    monkeypatch.setattr(claims, "generate_json", fake)
    claims.extract_claims(repo)
    claims.extract_claims(repo)  # identical repo + commit + requirements
    assert calls["n"] == 1


def test_cache_key_changes_with_requirements(repo, monkeypatch):
    calls = {"n": 0}

    def fake(prompt, **kwargs):
        calls["n"] += 1
        return {"claims": [{"claim": "c", "type": "feature", "keywords": ["x"]}]}

    monkeypatch.setattr(claims, "generate_json", fake)
    claims.extract_claims(repo, ["one"])
    claims.extract_claims(repo, ["two"])
    assert calls["n"] == 2


def test_bad_gemma_shape_raises(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"nope": 1})
    with pytest.raises(GemmaResponseError):
        claims.extract_claims(repo)


def test_claim_normalisation_handles_strings_and_junk(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        "Reads a plain string",
        {"claim": "", "type": "feature", "keywords": ["x"]},
        {"claim": "Uses a typed field", "type": "nonsense",
         "keywords": "single"},
        {"not_a_claim": True},
        42,
    ]})
    result = claims.extract_claims(repo)
    texts = [c["claim"] for c in result]
    assert "Reads a plain string" in texts
    assert "Uses a typed field" in texts
    assert all(t for t in texts)
    typed = next(c for c in result if c["claim"] == "Uses a typed field")
    assert typed["type"] == "feature"          # invalid type falls back
    assert typed["keywords"] == ["single"]     # bare string becomes a list


def test_claims_that_are_not_sentences_are_dropped(repo, monkeypatch):
    """A claim must be a plain sentence, not a label or a bare noun phrase."""
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "Local storage history", "type": "feature",
         "keywords": ["history"]},
        {"claim": "Reads the local storage history", "type": "feature",
         "keywords": ["history"]},
    ]})
    result = claims.extract_claims(repo)
    texts = [c["claim"] for c in result]
    assert "Reads the local storage history" in texts
    assert "Local storage history" not in texts


def test_layout_claims_are_dropped(repo, monkeypatch):
    """A claim about where files live is not a feature claim."""
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "router-based SPA entry point in index.html",
         "type": "feature", "keywords": ["router"]},
        {"claim": "Uses a router for navigation", "type": "feature",
         "keywords": ["router"]},
    ]})
    result = claims.extract_claims(repo)
    texts = [c["claim"] for c in result]
    assert "Uses a router for navigation" in texts
    assert "router-based SPA entry point in index.html" not in texts


def test_long_strong_keywords_are_dropped():
    """A strong keyword must be short, or it is discarded."""
    item = claims._normalise_claim({
        "claim": "Turns signs into speech", "type": "feature",
        "keywords": ["speech"],
        "strong_keywords": ["speech", "web speech recognition synthesis engine"],
    })
    assert item["strong_keywords"] == ["speech"]


def test_clean_keywords_trims_and_bounds():
    assert claims._clean_keywords(["  Foo  ", "foo", "b" * 100]) == ["foo"]
    assert claims._clean_keywords(None) == []
    assert len(claims._clean_keywords([f"k{i}" for i in range(20)])) == 6


def test_strong_keywords_and_evidence_kind_are_kept():
    item = claims._normalise_claim({
        "claim": "Stores users in a database", "type": "feature",
        "keywords": ["users"], "strong_keywords": ["sqlite3", "sqlite3", "DB"],
        "evidence_kind": "presentation",
    })
    assert item["strong_keywords"] == ["sqlite3", "db"]
    assert item["evidence_kind"] == "presentation"


def test_a_library_claim_is_always_functional():
    """A library, framework or service can never be proved by a stylesheet."""
    item = claims._normalise_claim({
        "claim": "Uses Chart.js", "type": "tech_used", "keywords": ["chart"],
        "strong_keywords": ["chart.js"], "evidence_kind": "presentation",
    })
    assert item["evidence_kind"] == "functional"


def test_evidence_kind_defaults_to_functional():
    item = claims._normalise_claim({"claim": "Has tests", "type": "feature",
                                    "keywords": []})
    assert item["evidence_kind"] == "functional"
    assert item["strong_keywords"] == []


def test_requirements_get_strong_keywords(repo, monkeypatch):
    """A requirement the model does not return still needs concrete proof."""
    monkeypatch.setattr(claims, "generate_json",
                        lambda *a, **k: {"claims": []})
    result = claims.extract_claims(repo, ["Stores users in a database"])
    req = next(c for c in result if c["type"] == "requirement")
    assert req["strong_keywords"] == ["stores", "database"]


def test_a_mobile_app_requirement_is_gated_on_the_platform(repo, monkeypatch):
    """The broad word "mobile" is not specific proof of a mobile app.

    A responsive UI is full of the word "mobile" (a drawer, a toggle button), so
    it must not be a strong keyword. The platform name is what has to be found.
    """
    from app import keywords as keywords_module

    terms = keywords_module.specific_terms("Has a mobile app for Android")
    assert "mobile" not in terms
    assert "android" in terms
    # "mobile" expands to no searchable strong term at all.
    assert keywords_module.strong_match_terms("mobile") == []


def test_prompt_lists_the_extra_requirements(repo, monkeypatch):
    captured = {}

    def fake(prompt, **kwargs):
        captured["prompt"] = prompt
        return {"claims": []}

    monkeypatch.setattr(claims, "generate_json", fake)
    claims.extract_claims(repo, ["Stores users in a database"])
    assert "Stores users in a database" in captured["prompt"]
    assert "requirement" in captured["prompt"]


def test_strong_keywords_are_merged_with_duplicates(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "Uses Frobnicator for motion tracking", "type": "tech_used",
         "keywords": ["frobnicator"], "strong_keywords": ["frobnicator"]},
        {"claim": "Uses Frobnicator to track motion", "type": "tech_used",
         "keywords": ["motion"], "strong_keywords": ["frobnicator-js"]},
    ]})
    result = claims.extract_claims(repo)
    assert len(result) == 1
    assert set(result[0]["strong_keywords"]) == {"frobnicator", "frobnicator-js"}


def test_normalise_claim_drops_junk(repo, monkeypatch):
    """Broken claim items are dropped, not reported as invented claims."""
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "- Uses the requests library", "type": "tech_used",
         "keywords": ["requests"]},
        {"claim": "{", "type": "feature", "keywords": ["docker"]},
        {"claim": "   ", "type": "feature", "keywords": []},
        {"claim": "42", "type": "feature", "keywords": []},
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ]})
    result = claims.extract_claims(repo)
    texts = [c["claim"] for c in result]
    # The list marker is stripped, and the three junk items are dropped.
    assert texts == ["Uses the requests library", "Has a CLI"]


def test_long_claim_text_is_capped():
    item = claims._normalise_claim({"claim": "Reads " + "word " * 200,
                                    "type": "feature", "keywords": []})
    assert len(item["claim"]) <= claims.MAX_CLAIM_CHARS


def test_prompt_asks_for_short_compact_json(repo, monkeypatch):
    """The claim prompt must keep replies short to make Gemma faster."""
    captured = {}

    def fake(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {"claims": []}

    monkeypatch.setattr(claims, "generate_json", fake)
    claims.extract_claims(repo)

    assert captured["kwargs"]["max_output_tokens"] == config.GEMMA_MAX_OUTPUT_TOKENS
    assert captured["kwargs"]["response_schema"] is claims.CLAIMS_SCHEMA
    prompt = captured["prompt"]
    assert "COMPACT JSON ONLY" in prompt
    assert "12 words" in prompt
    assert "nothing outside" in prompt.lower()


# ---------------------------------------------------------------------------
# Claims must come from the README or the extra requirements
# ---------------------------------------------------------------------------


def test_invented_claims_are_dropped(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "Uses the Kubernetes Operator for deployment",
         "type": "tech_used", "keywords": ["kubernetes"]},
        {"claim": "Uses Frobnicator for motion tracking",
         "type": "tech_used", "keywords": ["frobnicator"]},
    ]})
    result = claims.extract_claims(repo)
    texts = [c["claim"] for c in result]
    assert "Uses Frobnicator for motion tracking" in texts
    assert "Uses the Kubernetes Operator for deployment" not in texts


def test_claims_with_odd_openings_are_dropped(repo, monkeypatch):
    monkeypatch.setattr(claims, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "Claims to be a warm responsive application",
         "type": "feature", "keywords": ["warm"]},
        {"claim": "Classically built with Frobnicator",
         "type": "tech_used", "keywords": ["frobnicator"]},
    ]})
    assert claims.extract_claims(repo) == []


def test_claims_call_runs_cold(repo, monkeypatch):
    captured = {}

    def fake(prompt, **kwargs):
        captured.update(kwargs)
        return {"claims": []}

    monkeypatch.setattr(claims, "generate_json", fake)
    claims.extract_claims(repo)
    assert captured["temperature"] == config.GEMMA_CLAIMS_TEMPERATURE
    assert captured["temperature"] == 0.0
