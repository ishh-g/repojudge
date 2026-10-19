"""Milestone 3 tests: evidence search and evidence-only judging."""
from __future__ import annotations

import pytest

from app import config, verifier
from app import claims as claims_module
from app.errors import GemmaResponseError
from app.github_client import fetch_repo
from fakes import FakeGitHub

FILES = {
    "main.py": (
        "import argparse\n"
        "\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--name')\n"
        "    print('hello')\n"
    ),
    "requirements.txt": "flask==3.0.0\nrequests==2.31.0\n",
    "src/api.py": "from flask import Flask\napp = Flask(__name__)\n",
}


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    yield


@pytest.fixture
def repo():
    return fetch_repo("https://github.com/acme/widget",
                      session=FakeGitHub(files=FILES))


# ---------------------------------------------------------------------------
# Evidence search
# ---------------------------------------------------------------------------


def test_find_evidence_locates_file_and_line(repo):
    hits = verifier.find_evidence(repo, ["argparse"])
    assert hits, "argparse should be found"
    assert hits[0]["file"] == "main.py"
    assert hits[0]["line"] == 1
    assert "argparse" in hits[0]["snippet"]


def test_find_evidence_multiple_files(repo):
    hits = verifier.find_evidence(repo, ["flask"])
    files = {h["file"] for h in hits}
    assert "requirements.txt" in files
    assert "src/api.py" in files


def test_find_evidence_is_case_insensitive(repo):
    assert verifier.find_evidence(repo, ["FLASK"])
    assert verifier.find_evidence(repo, ["Flask"])


def test_find_evidence_empty_for_unknown_keyword(repo):
    assert verifier.find_evidence(repo, ["totally_missing_symbol"]) == []
    assert verifier.find_evidence(repo, []) == []


def test_find_evidence_respects_limit(repo):
    hits = verifier.find_evidence(repo, ["a"], limit=2)
    assert len(hits) <= 2


def test_find_evidence_matches_keyword_variants():
    """A keyword with a file extension also matches the plain name in code."""
    files = {
        "index.html": '<script src="https://cdn.example.com/chart.umd.js"></script>\n',
        "js/app.js": (
            "const ctx = document.getElementById('c');\n"
            "new Chart(ctx, { type: 'bar' });\n"
        ),
    }
    repo = fetch_repo(
        "https://github.com/acme/widget",
        session=FakeGitHub(files=files, readme="# Widget\n\nA tiny chart page.\n",
                           languages={"JavaScript": 100}),
    )
    hits = verifier.find_evidence(repo, ["Chart.js"])
    assert hits
    assert any(hit["file"] == "js/app.js" for hit in hits)
    assert any("Chart(" in hit["snippet"] for hit in hits)


def test_find_evidence_prefers_the_file_named_by_a_keyword():
    """When a keyword names a file in the repo, that file's lines win."""
    files = {
        "index.html": '<script src="js/theme.js"></script>\n',
        "js/theme.js": "localStorage.setItem('theme', 'dark');\n",
    }
    repo = fetch_repo(
        "https://github.com/acme/widget",
        session=FakeGitHub(files=files, readme="# Widget\n\nDoes things.\n",
                           languages={"JavaScript": 100}),
    )
    hits = verifier.find_evidence(repo, ["theme.js", "localstorage"])
    assert hits
    assert hits[0]["file"] == "js/theme.js"


def test_find_evidence_prefers_a_line_matching_more_keywords():
    files = {
        "js/store.js": (
            "localStorage.setItem('a', 'b');\n"
            "const history = localStorage.getItem('history');\n"
        ),
    }
    repo = fetch_repo(
        "https://github.com/acme/widget",
        session=FakeGitHub(files=files, readme="# Widget\n\nDoes things.\n",
                           languages={"JavaScript": 100}),
    )
    hits = verifier.find_evidence(repo, ["localstorage", "history"])
    assert hits[0]["line"] == 2


def test_find_evidence_keeps_the_best_three():
    files = {
        "src/app.py": (
            "".join(f"# note about requests number {i}\n" for i in range(20))
            + "import requests  # the real use\n"
        ),
    }
    repo = fetch_repo(
        "https://github.com/acme/widget",
        session=FakeGitHub(files=files, readme="# Widget\n\nDoes things.\n"),
    )
    hits = verifier.find_evidence(repo, ["requests"])
    assert len(hits) == 3
    # The real import outranks the prose mentions.
    assert hits[0]["line"] == 21
    assert "import requests" in hits[0]["snippet"]


def test_find_evidence_ranks_code_above_prose():
    files = {"a.py": "requests is mentioned here\nimport requests\n"}
    repo = fetch_repo(
        "https://github.com/acme/widget",
        session=FakeGitHub(files=files, readme="# Widget\n\nDoes things.\n"),
    )
    hits = verifier.find_evidence(repo, ["requests"])
    assert hits[0]["line"] == 2


def test_find_evidence_uses_keyword_lines_from_a_cut_file():
    """A large file is cut down, but keyword lines from its tail are kept."""
    from app.github_client import add_keyword_lines

    big = "x = 1\n" * 20000 + "value = special_marker_call()\n"
    fake = FakeGitHub(files={"big.py": big, "other.py": "y = 2\n"})
    repo = fetch_repo("https://github.com/acme/widget", session=fake)
    assert repo.files["big.py"].truncated is True

    # The tail line is outside the part we kept, so it is not found yet.
    assert verifier.find_evidence(repo, ["special_marker_call"]) == []

    add_keyword_lines(repo, ["special_marker_call"], session=fake)
    hits = verifier.find_evidence(repo, ["special_marker_call"])
    assert hits
    assert hits[0]["file"] == "big.py"
    assert hits[0]["line"] == 20001


def test_find_evidence_skips_readme():
    from app.github_client import RepoFile

    fake = FakeGitHub(files={"main.py": "print('hi')\n"})
    repo = fetch_repo("https://github.com/acme/widget", session=fake)
    # Inject README as a fetched file to prove it is excluded from evidence.
    repo.files["README.md"] = RepoFile("README.md", "# Widget\nmagic engine\n", 20, 2)
    hits = verifier.find_evidence(repo, ["magic engine"])
    assert hits == []


def test_attach_evidence_adds_to_every_claim(repo):
    enriched = verifier.attach_evidence(repo, [
        {"claim": "uses argparse", "type": "feature", "keywords": ["argparse"]},
        {"claim": "uses zzz", "type": "feature", "keywords": ["zzz"]},
    ])
    assert enriched[0]["evidence"]
    assert enriched[1]["evidence"] == []


# ---------------------------------------------------------------------------
# File kinds and comments
# ---------------------------------------------------------------------------


def test_file_kind_classifies_by_path_only():
    assert verifier.file_kind("js/app.js") == "code"
    assert verifier.file_kind("styles/main.css") == "style"
    assert verifier.file_kind("docs/guide.md") == "docs"
    assert verifier.file_kind("index.html") == "markup"
    assert verifier.file_kind("package.json") == "config"
    assert verifier.file_kind("requirements.txt") == "config"
    assert verifier.file_kind("tests/test_app.py") == "test"
    assert verifier.file_kind("app.spec.ts") == "test"


def test_is_comment_only_detects_comments():
    assert verifier.is_comment_only("// a note")
    assert verifier.is_comment_only("# a note")
    assert verifier.is_comment_only("* a block comment line")
    assert verifier.is_comment_only("<!-- a note -->")
    assert not verifier.is_comment_only("#include <stdio.h>")
    assert not verifier.is_comment_only("import requests")
    assert not verifier.is_comment_only("")


def test_comments_are_never_evidence():
    files = {"js/app.js": "// uses the requests library\nconst x = 1;\n"}
    repo = fetch_repo(
        "https://github.com/acme/widget",
        session=FakeGitHub(files=files, readme="# Widget\n\nDoes things.\n",
                           languages={"JavaScript": 100}),
    )
    assert verifier.find_evidence(repo, ["requests"], allow_comments=True)
    assert verifier.find_evidence(repo, ["requests"], allow_comments=False) == []


def test_a_style_file_cannot_prove_a_technical_claim():
    files = {
        "static/style.css": ".foowidget-icon { color: red; }\n",
        "app.py": "print('hi')\n",
    }
    repo = fetch_repo(
        "https://github.com/acme/widget",
        session=FakeGitHub(files=files, readme="# Widget\n\nDoes things.\n",
                           languages={"JavaScript": 100}),
    )
    functional = verifier.find_evidence(
        repo, ["foowidget"], allowed_kinds=verifier.FUNCTIONAL_KINDS
    )
    assert functional == []
    # A presentation claim may still be proved by the stylesheet.
    everything = verifier.find_evidence(repo, ["foowidget"], allowed_kinds=None)
    assert everything and everything[0]["file"] == "static/style.css"


# ---------------------------------------------------------------------------
# Judging call
# ---------------------------------------------------------------------------


def _judgement(index, supports, reason="because", proof=None):
    return {"id": index, "supports": supports, "proof": proof, "reason": reason}


def _judge(items):
    """A fake Gemma that always answers with the given judgement items."""
    return lambda *a, **k: {"judgements": items}


def test_verify_applies_gemma_supports(repo, monkeypatch):
    monkeypatch.setattr(verifier, "generate_json", _judge([
        _judgement(0, "full", "main.py imports argparse", "main.py:1"),
        _judgement(1, "part", "declared but not used"),
    ]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
        {"claim": "Uses flask", "type": "tech_used", "keywords": ["flask"]},
    ])
    assert result[0]["status"] == "Verified"
    assert result[1]["status"] == "Partial"
    assert result[0]["reason"] == "main.py imports argparse"


def test_legacy_status_word_is_still_understood(repo, monkeypatch):
    """An older reply shape is read, using its reason as the proof."""
    monkeypatch.setattr(verifier, "generate_json", _judge([
        {"id": 0, "status": "Verified", "reason": "main.py:1 imports argparse"},
    ]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ])
    assert result[0]["status"] == "Verified"


def test_no_evidence_forces_unverified(repo, monkeypatch):
    """Even if Gemma says full, a claim with no evidence stays Unverified."""
    monkeypatch.setattr(verifier, "generate_json", _judge([
        _judgement(0, "full", "definitely true", "main.py:1"),
    ]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a quantum engine", "type": "feature",
         "keywords": ["quantum_engine_xyz"]},
    ])
    assert result[0]["status"] == "Unverified"
    assert result[0]["reason"] == verifier.NO_PROOF_REASON
    assert result[0]["evidence"] == []


def test_supports_none_is_unverified_whatever_else(repo, monkeypatch):
    """The model saying "none" can never become Verified or Partial."""
    monkeypatch.setattr(verifier, "generate_json", _judge([
        _judgement(0, "none", "the code only prints a message", "main.py:1"),
    ]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ])
    assert result[0]["status"] == "Unverified"
    assert result[0]["evidence"] == []


def test_proof_outside_the_evidence_is_unverified(repo, monkeypatch):
    monkeypatch.setattr(verifier, "generate_json", _judge([
        _judgement(0, "full", "proved elsewhere", "src/nothing.py:99"),
    ]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ])
    assert result[0]["status"] == "Unverified"
    assert result[0]["reason"] == verifier.NO_PROOF_REASON


def test_missing_judgement_becomes_not_judged(repo, monkeypatch):
    """A claim the model never judges is Not judged, never guessed."""
    monkeypatch.setattr(verifier, "generate_json", _judge([]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ])
    assert result[0]["status"] == "Not judged"
    assert result[0]["evidence"] == []


def test_one_extra_call_is_made_for_missing_claims(repo, monkeypatch):
    calls = {"n": 0}

    def fake(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"judgements": []}          # the model skipped every claim
        return {"judgements": [
            _judgement(0, "full", "main.py imports argparse", "main.py:1"),
            _judgement(1, "none", "there is no flask call"),
        ]}

    monkeypatch.setattr(verifier, "generate_json", fake)
    result = verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
        {"claim": "Uses flask", "type": "tech_used", "keywords": ["flask"]},
    ])
    assert calls["n"] == 2                    # one batch + one retry
    assert result[0]["status"] == "Verified"
    assert result[1]["status"] == "Unverified"


def test_claims_are_judged_in_small_batches(repo, monkeypatch):
    """A long reply is split so it can never be cut off half way."""
    prompts: list[str] = []

    def fake(prompt, **kwargs):
        prompts.append(prompt)
        indexes = [int(line.split()[1]) for line in prompt.splitlines()
                   if line.startswith("CLAIM ")]
        return {"judgements": [
            _judgement(index, "none", "nothing proves it") for index in indexes
        ]}

    monkeypatch.setattr(verifier, "generate_json", fake)
    monkeypatch.setattr(config, "MAX_JUDGE_BATCH", 3)
    claim_list = [
        {"claim": f"claim {i}", "type": "feature", "keywords": ["argparse"]}
        for i in range(7)
    ]
    verifier.verify_claims(repo, claim_list)
    assert len(prompts[0].split("CLAIM ")) - 1 == 3
    assert len(prompts) >= 3


def test_reason_naming_a_file_outside_evidence_is_corrected(repo, monkeypatch):
    """A reason may only name files that appear in the claim's evidence."""
    monkeypatch.setattr(verifier, "generate_json", _judge([
        _judgement(0, "full", "proved by src/api.py:1", "main.py:1"),
    ]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ])
    assert result[0]["status"] == "Verified"
    assert "src/api.py" not in result[0]["reason"]
    assert result[0]["reason"] == "Proved by main.py:1."


def test_reason_naming_a_file_in_evidence_is_kept(repo, monkeypatch):
    monkeypatch.setattr(verifier, "generate_json", _judge([
        _judgement(0, "full", "main.py imports argparse", "main.py:1"),
    ]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ])
    assert result[0]["reason"] == "main.py imports argparse"


def test_unverified_claim_never_shows_evidence(repo, monkeypatch):
    monkeypatch.setattr(verifier, "generate_json", _judge([
        _judgement(0, "none", "no proof"),
    ]))
    result = verifier.verify_claims(repo, [
        {"claim": "Has a quantum engine", "type": "feature",
         "keywords": ["quantum_engine_xyz"]},
    ])
    assert result[0]["status"] == "Unverified"
    assert result[0]["evidence"] == []


def test_bad_judgement_shape_raises(repo, monkeypatch):
    monkeypatch.setattr(verifier, "generate_json", lambda *a, **k: {"oops": True})
    with pytest.raises(GemmaResponseError):
        verifier.verify_claims(repo, [
            {"claim": "x", "type": "feature", "keywords": ["argparse"]},
        ])


def test_verify_results_are_cached(repo, monkeypatch):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return {"judgements": [_judgement(0, "full", "ok", "main.py:1")]}

    monkeypatch.setattr(verifier, "generate_json", fake)
    claim_list = [{"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]}]
    verifier.verify_claims(repo, claim_list)
    verifier.verify_claims(repo, claim_list)
    assert calls["n"] == 1


def test_empty_claims_returns_empty(repo, monkeypatch):
    monkeypatch.setattr(verifier, "generate_json",
                        lambda *a, **k: pytest.fail("should not call Gemma"))
    assert verifier.verify_claims(repo, []) == []


def test_prompt_asks_for_short_compact_json(repo, monkeypatch):
    """The judge prompt must keep replies short to make Gemma faster."""
    captured = {}

    def fake(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return {"judgements": []}

    monkeypatch.setattr(verifier, "generate_json", fake)
    verifier.verify_claims(repo, [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]},
    ])

    assert captured["kwargs"]["max_output_tokens"] == config.GEMMA_MAX_OUTPUT_TOKENS
    assert captured["kwargs"]["response_schema"] is verifier.JUDGEMENTS_SCHEMA
    prompt = captured["prompt"]
    assert "COMPACT JSON ONLY" in prompt
    assert "15 words" in prompt
    assert "nothing outside" in prompt.lower()


# ---------------------------------------------------------------------------
# Strong keywords: proving that a claim really is implemented
# ---------------------------------------------------------------------------


def _repo(files: dict[str, str], readme: str = "# Widget\n\nA small project.\n"):
    return fetch_repo(
        "https://github.com/acme/widget",
        session=FakeGitHub(files=files, readme=readme,
                           languages={"JavaScript": 500}),
    )


def _run(repo, claim, judgements, monkeypatch):
    monkeypatch.setattr(verifier, "generate_json", _judge(judgements))
    return verifier.verify_claims(repo, [claim])


def test_a_mock_list_is_not_a_database(monkeypatch):
    """A hard-coded array is not a database, however the model answers."""
    repo = _repo({"js/data.js": "const users = [{ id: 1 }];\nconst defaultUsers = users;\n"})
    result = _run(repo, {
        "claim": "Stores users in a database", "type": "feature",
        "keywords": ["users", "database"],
        "strong_keywords": ["sqlite3", "psycopg2", "database", "mongoose"],
    }, [_judgement(0, "full", "the users are stored", "js/data.js:1")], monkeypatch)
    assert result[0]["status"] == "Unverified"
    assert result[0]["evidence"] == []
    assert result[0]["reason"] == verifier.NO_PROOF_REASON


def test_a_real_database_call_can_be_verified(monkeypatch):
    repo = _repo({"db.py": "import sqlite3\n\nconn = sqlite3.connect('app.db')\n"})
    result = _run(repo, {
        "claim": "Stores users in a database", "type": "feature",
        "keywords": ["sqlite3"],
        "strong_keywords": ["sqlite3", "database"],
    }, [_judgement(0, "full", "db.py opens a real database", "db.py:1")], monkeypatch)
    assert result[0]["status"] == "Verified"


def test_a_tests_claim_needs_a_real_test_file(monkeypatch):
    repo = _repo({"main.py": "print('hi')\n"})
    result = _run(repo, {
        "claim": "Has automated tests", "type": "feature",
        "keywords": ["tests"],
        "strong_keywords": ["pytest", "unittest", "jest", "test_*.py"],
    }, [_judgement(0, "full", "tests are included", "main.py:1")], monkeypatch)
    assert result[0]["status"] == "Unverified"


def test_a_tests_claim_with_a_real_test_file_can_be_verified(monkeypatch):
    repo = _repo({"tests/test_app.py": "import pytest\n\ndef test_x():\n    assert True\n"})
    result = _run(repo, {
        "claim": "Has automated tests", "type": "feature",
        "keywords": ["pytest"],
        "strong_keywords": ["pytest", "test_*.py"],
    }, [_judgement(0, "full", "tests/test_app.py runs pytest",
                   "tests/test_app.py:1")], monkeypatch)
    assert result[0]["status"] == "Verified"


def test_a_mobile_app_claim_needs_platform_files(monkeypatch):
    repo = _repo({"app.py": "def start():\n    pass\n"})
    result = _run(repo, {
        "claim": "Has a mobile app for Android", "type": "feature",
        "keywords": ["android", "mobile"],
        "strong_keywords": ["android", "AndroidManifest.xml", "react-native",
                            "build.gradle"],
    }, [_judgement(0, "full", "there is an app", "app.py:1")], monkeypatch)
    assert result[0]["status"] == "Unverified"


def test_a_mobile_app_claim_with_platform_files_can_be_verified(monkeypatch):
    files = {
        "android/app/src/main/AndroidManifest.xml":
            '<manifest xmlns:android="http://schemas.android.com/apk/res/android"\n'
            '          package="com.example.app">\n',
        "android/app/build.gradle": "apply plugin: 'com.android.application'\n",
    }
    repo = _repo(files)
    result = _run(repo, {
        "claim": "Has a mobile app for Android", "type": "feature",
        "keywords": ["android"],
        "strong_keywords": ["android", "AndroidManifest.xml"],
    }, [_judgement(0, "full", "the android build files are present",
                   "android/app/build.gradle:1")], monkeypatch)
    assert result[0]["status"] == "Verified"


def test_a_library_included_from_a_cdn_script_tag_is_verified(monkeypatch):
    files = {
        "index.html": '<script src="https://cdn.example.com/chart.umd.js"></script>\n',
    }
    repo = _repo(files)
    result = _run(repo, {
        "claim": "Uses Chart.js for the charts", "type": "tech_used",
        "keywords": ["chart"],
        "strong_keywords": ["chart.js", "chart"],
    }, [_judgement(0, "full", "index.html loads the library", "index.html:1")],
        monkeypatch)
    assert result[0]["status"] == "Verified"


def test_a_stylesheet_alone_cannot_prove_a_library(monkeypatch):
    files = {
        "static/style.css": ".foowidget-icon { color: red; }\n",
        "app.py": "print('hi')\n",
    }
    repo = _repo(files)
    result = _run(repo, {
        "claim": "Uses the FooWidget library", "type": "tech_used",
        "keywords": ["foowidget"],
        "strong_keywords": ["foowidget"],
    }, [_judgement(0, "full", "the stylesheet uses it", "static/style.css:1")],
        monkeypatch)
    assert result[0]["status"] == "Unverified"
    assert result[0]["evidence"] == []


def test_a_stylesheet_can_prove_a_presentation_claim(monkeypatch):
    files = {"static/style.css": ".gallery { display: grid; }\n"}
    repo = _repo(files)
    result = _run(repo, {
        "claim": "Lays the gallery out with a grid", "type": "feature",
        "evidence_kind": "presentation",
        "keywords": ["grid"],
        "strong_keywords": ["display grid"],
    }, [_judgement(0, "full", "the stylesheet uses a grid", "static/style.css:1")],
        monkeypatch)
    assert result[0]["status"] == "Verified"


# ---------------------------------------------------------------------------
# Strong-keyword variants: a joined or oddly spelled term must still be found
# ---------------------------------------------------------------------------


def test_a_joined_strong_keyword_matches_the_real_object(monkeypatch):
    """A model may write ``webspeechapi``; the code says ``speechSynthesis``."""
    repo = _repo({
        "js/app.js": (
            "const utter = new SpeechSynthesisUtterance(text);\n"
            "window.speechSynthesis.speak(utter);\n"
        ),
    })
    result = _run(repo, {
        "claim": "Converts signs into spoken speech", "type": "feature",
        "keywords": ["speech"],
        "strong_keywords": ["webspeechapi"],
    }, [_judgement(0, "full", "js/app.js speaks the text", "js/app.js:2")],
        monkeypatch)
    assert result[0]["status"] == "Verified"
    assert any(hit["file"] == "js/app.js" for hit in result[0]["evidence"])


def test_a_strong_keyword_object_name_is_found_in_one_js_file(monkeypatch):
    """A shared object name written as one lowercase word still matches."""
    repo = _repo({
        "js/audio.js": (
            "function playTone(freq) {\n"
            "  const ctx = new AudioContext();\n"
            "  const osc = ctx.createOscillator();\n"
            "}\n"
        ),
    })
    enriched = verifier.attach_evidence(repo, [{
        "claim": "Plays a tone with the audio context", "type": "feature",
        "keywords": ["audio"],
        "strong_keywords": ["audiocontext"],
    }])
    assert enriched[0]["strong_proof"] is True
    assert enriched[0]["evidence"]
    assert any(hit["file"] == "js/audio.js"
               and "audiocontext" in hit["snippet"].lower()
               for hit in enriched[0]["evidence"])


def test_a_long_joined_strong_keyword_is_rescued_by_the_claim_terms(monkeypatch):
    """A strong keyword that matches nothing must not sink a real claim."""
    repo = _repo({
        "js/speech.js": (
            "const utter = new SpeechSynthesisUtterance('hi');\n"
            "speechSynthesis.speak(utter);\n"
        ),
    })
    result = _run(repo, {
        "claim": "Reads text aloud with speech", "type": "feature",
        "keywords": ["speechsynthesis"],
        "strong_keywords": ["webspeechsynthesisengineering"],
    }, [_judgement(0, "full", "js/speech.js speaks the text", "js/speech.js:2")],
        monkeypatch)
    assert result[0]["status"] == "Verified"


def test_a_real_constructor_outranks_a_line_that_only_mentions_the_word():
    """``new SpeechSynthesisUtterance(...)`` is stronger than "Speech Output"."""
    repo = _repo({
        "js/app.js": (
            "showToast('Speech Output', spoken);\n"
            "const utterance = new SpeechSynthesisUtterance(text);\n"
        ),
    })
    hits = verifier.find_evidence(repo, ["speech"],
                                  strong_keywords=["webspeechapi"])
    assert hits
    assert hits[0]["line"] == 2
    assert "SpeechSynthesisUtterance" in hits[0]["snippet"]


# ---------------------------------------------------------------------------
# Evidence ranking: the best proof first, and no unrelated lines
# ---------------------------------------------------------------------------


def test_a_library_script_tag_outranks_the_code_that_calls_it(monkeypatch):
    """A ``<script>`` tag that loads a library is the best possible proof."""
    repo = _repo({
        "index.html":
            '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n',
        "js/dashboard.js": "const chart = new Chart(ctx, { type: 'bar' });\n",
    })
    hits = verifier.find_evidence(repo, ["chart"], strong_keywords=["chart.js"])
    assert hits
    assert hits[0]["file"] == "index.html"
    assert "script" in hits[0]["snippet"].lower()

    result = _run(repo, {
        "claim": "Uses Chart.js for the charts", "type": "tech_used",
        "keywords": ["chart"],
        "strong_keywords": ["chart.js"],
    }, [_judgement(0, "full", "index.html loads the library", "index.html:1")],
        monkeypatch)
    assert result[0]["status"] == "Verified"


def test_unrelated_lines_are_not_shown_when_a_strong_hit_exists(monkeypatch):
    """Only lines matching a strong term are evidence, not every shared word."""
    repo = _repo({
        "js/notes.js": (
            "function recordNote() {\n"
            "  return 'a note';\n"
            "}\n"
            "const rec = new MediaRecorder(stream);\n"
        ),
    })
    hits = verifier.find_evidence(repo, ["audio", "record"],
                                  strong_keywords=["mediarecorder"])
    assert hits
    assert all("mediarecorder" in hit["snippet"].lower() for hit in hits)
    assert all("recordnote" not in hit["snippet"].lower() for hit in hits)


def test_extra_requirements_stay_unverified_on_a_js_app(monkeypatch):
    """A JS/HTML app proves no database, no test suite and no Android app.

    The model is deliberately overconfident: it says "full" for every claim and
    points at a real line. The evidence rules still refuse all three.
    """
    repo = _repo({
        "index.html":
            '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n',
        "js/app.js": "const users = [{ id: 1 }];\n",
    })
    extras = [
        {"claim": "Stores users in a database", "type": "requirement",
         "keywords": ["users", "database"],
         "strong_keywords": ["sqlite3", "database"]},
        {"claim": "Has automated tests", "type": "requirement",
         "keywords": ["tests"],
         "strong_keywords": ["pytest", "jest"]},
        {"claim": "Has a mobile app for Android", "type": "requirement",
         "keywords": ["android", "mobile"],
         "strong_keywords": ["android", "AndroidManifest.xml"]},
    ]
    for claim in extras:
        result = _run(repo, claim, [
            _judgement(0, "full", "the code shows it", "js/app.js:1")
        ], monkeypatch)
        assert result[0]["status"] == "Unverified", claim["claim"]
        assert result[0]["evidence"] == [], claim["claim"]
        assert result[0]["reason"] == verifier.NO_PROOF_REASON, claim["claim"]


# ---------------------------------------------------------------------------
# Versions and consistency between claims
# ---------------------------------------------------------------------------


def test_a_matching_version_is_verified(monkeypatch):
    repo = _repo({
        "index.html": ('<script src="https://cdn.jsdelivr.net/npm/'
                       'chart.js@4.4.1/dist/chart.umd.js"></script>\n'),
    })
    result = _run(repo, {
        "claim": "Uses Chart.js 4.4 for the charts", "type": "tech_used",
        "keywords": ["chart"], "strong_keywords": ["chart.js"],
    }, [_judgement(0, "full", "index.html loads the library", "index.html:1")],
        monkeypatch)
    assert result[0]["status"] == "Verified"


def test_a_different_version_is_partial_and_names_the_found_version(monkeypatch):
    repo = _repo({
        "index.html": ('<link rel="stylesheet" href="https://cdn.example.com/'
                       'font-awesome/6.5.1/css/all.min.css">\n'),
    })
    result = _run(repo, {
        "claim": "Uses FontAwesome 6.4 for icons", "type": "tech_used",
        "keywords": ["fontawesome"], "strong_keywords": ["fontawesome"],
    }, [_judgement(0, "full", "index.html loads the icons", "index.html:1")],
        monkeypatch)
    assert result[0]["status"] == "Partial"
    assert "6.5" in result[0]["reason"]


def test_no_visible_version_is_a_plain_name_match(monkeypatch):
    repo = _repo({
        "index.html": '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>\n',
    })
    result = _run(repo, {
        "claim": "Uses Chart.js 4.4 for the charts", "type": "tech_used",
        "keywords": ["chart"], "strong_keywords": ["chart.js"],
    }, [_judgement(0, "full", "index.html loads the library", "index.html:1")],
        monkeypatch)
    assert result[0]["status"] == "Verified"


def test_font_awesome_link_tag_is_found_and_ranked_first(monkeypatch):
    """A hyphenated CDN path proves a joined strong keyword like fontawesome."""
    repo = _repo({
        "index.html": ('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/'
                       'ajax/libs/font-awesome/6.4.0/css/all.min.css">\n'),
        "js/icons.js": "// font awesome icons are used here\n",
    })
    hits = verifier.find_evidence(repo, ["fontawesome"],
                                  strong_keywords=["fontawesome"])
    assert hits
    assert hits[0]["file"] == "index.html"
    assert "font-awesome" in hits[0]["snippet"]


def test_an_unverified_claim_reuses_another_claims_proof(monkeypatch):
    """A line that proved one claim is proof for a claim with the same term.

    The model answers "none" for the second claim even though the same proof
    line already proves the first. The consistency pass attaches that line and
    asks the judge once more; the judge then decides on the real evidence.
    """
    repo = _repo({
        "js/app.js": (
            "const vision = require('@mediapipe/tasks-vision');\n"
            "const landmarker = new HandLandmarker();\n"
        ),
    })
    calls = {"n": 0}

    def fake(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"judgements": [
                _judgement(0, "full", "js/app.js imports mediapipe", "js/app.js:1"),
                _judgement(1, "none", "no clear proof", None),
            ]}
        return {"judgements": [
            _judgement(1, "full", "js/app.js uses mediapipe", "js/app.js:1"),
        ]}

    monkeypatch.setattr(verifier, "generate_json", fake)
    result = verifier.verify_claims(repo, [
        {"claim": "Uses MediaPipe for hand tracking", "type": "tech_used",
         "keywords": ["mediapipe"], "strong_keywords": ["mediapipe"]},
        {"claim": "Uses MediaPipe HandLandmarker for hands", "type": "tech_used",
         "keywords": ["handlandmarker"],
         "strong_keywords": ["mediapipe", "handlandmarker"]},
    ])
    assert result[0]["status"] == "Verified"
    assert result[1]["status"] == "Verified"
    assert calls["n"] == 2


def test_a_repo_name_strong_keyword_does_not_hide_the_real_evidence():
    """A strong keyword equal to the repo name must not suppress real proof.

    The repo name appears in imports all over the project, so if it forced the
    strong-only filter it would hide the files that actually implement the
    claim. Here ``parse`` lives in ``src/widget.py``, which never says "widget".
    """
    repo = _repo(
        {
            "src/widget.py": "def parse(value):\n    return value.split(',')\n",
            "tests/test_widget.py": "from widget import parse\n",
        },
        readme="# Widget\n\nParses comma separated values.\n",
    )
    hits = verifier.find_evidence(repo, ["parse"], strong_keywords=["widget"])
    assert hits
    assert any(hit["file"] == "src/widget.py" for hit in hits)
