"""Tests for the full report pipeline and the Markdown export."""
from __future__ import annotations

import pytest

from app import claims as claims_module
from app import config, github_client, report, verifier
from fakes import FakeGitHub

FILES = {
    "main.py": (
        "import argparse\n"
        "import requests\n"
        "\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.parse_args()\n"
    ),
    "requirements.txt": "requests==2.31.0\n",
    "src/api.py": "from flask import Flask\napp = Flask(__name__)\n",
    "tests/test_main.py": "def test_main():\n    assert True\n",
    "requirements-dev.txt": "pytest\n",
}

README = """\
# Widget

Widget is a small tool that reads numbers and prints friendly words for them.
It has a command line interface and uses the requests library. The web API uses
flask, and it achieves 99.9% accuracy.

## Installation

pip install -r requirements.txt

## Usage

python main.py
"""

CLAIMS = [
    {"claim": "Has a command line interface", "type": "feature",
     "keywords": ["argparse"]},
    {"claim": "Uses the requests library", "type": "tech_used",
     "keywords": ["requests"]},
    {"claim": "Uses flask", "type": "tech_used", "keywords": ["flask"]},
    {"claim": "Achieves 99.9% accuracy", "type": "result",
     "keywords": ["accuracy_metric_xyz"]},
]

JUDGEMENTS = [
    {"id": 0, "supports": "full", "proof": "main.py:1",
     "reason": "main.py imports argparse."},
    {"id": 1, "supports": "full", "proof": "main.py:2",
     "reason": "requests is imported in main.py."},
    {"id": 2, "supports": "full", "proof": "src/api.py:1",
     "reason": "src/api.py imports flask."},
    {"id": 3, "supports": "full", "proof": None, "reason": "trust me"},
]


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    yield


def patch_pipeline(monkeypatch, fake_gh, claims=CLAIMS, judgements=JUDGEMENTS):
    real_fetch = github_client.fetch_repo
    calls = {"claims": 0, "judge": 0}

    def fake_fetch(url, **kwargs):
        return real_fetch(url, session=fake_gh)

    def fake_claims(prompt, **kwargs):
        calls["claims"] += 1
        return {"claims": claims}

    def fake_judge(prompt, **kwargs):
        calls["judge"] += 1
        return {"judgements": judgements}

    monkeypatch.setattr(report, "fetch_repo", fake_fetch)
    monkeypatch.setattr(claims_module, "generate_json", fake_claims)
    monkeypatch.setattr(verifier, "generate_json", fake_judge)
    return calls


def make_fake(**kwargs):
    kwargs.setdefault("files", FILES)
    kwargs.setdefault("readme", README)
    kwargs.setdefault("directories", [])
    return FakeGitHub(**kwargs)


# ---------------------------------------------------------------------------


def test_full_pipeline_scores_and_evidence(monkeypatch):
    calls = patch_pipeline(monkeypatch, make_fake())
    result = report.analyze_repo("https://github.com/acme/widget")

    assert result["repo"]["full_name"] == "acme/widget"
    assert result["analysis_type"] == "full"
    assert result["summary"]["verified"] == 3
    assert result["summary"]["unverified"] == 1
    assert result["score"]["max"] == 100
    assert 0 <= result["score"]["total"] <= 100
    # Exactly two Gemma calls.
    assert calls["claims"] == 1
    assert calls["judge"] == 1


def test_unprovable_claim_is_unverified_despite_gemma(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    result = report.analyze_repo("https://github.com/acme/widget")
    inaccurate = next(c for c in result["claims"]
                      if "99.9%" in c["claim"])
    assert inaccurate["status"] == "Unverified"
    assert inaccurate["evidence"] == []


def test_evidence_has_file_and_line(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    result = report.analyze_repo("https://github.com/acme/widget")
    cli = next(c for c in result["claims"] if "command line" in c["claim"])
    assert cli["evidence"]
    first = cli["evidence"][0]
    assert first["file"] == "main.py"
    assert first["line"] == 1
    assert "argparse" in first["snippet"]


def test_report_is_cached(monkeypatch):
    calls = patch_pipeline(monkeypatch, make_fake())
    report.analyze_repo("https://github.com/acme/widget")
    second = report.analyze_repo("https://github.com/acme/widget")
    assert second["cache_hit"] is True
    assert calls["claims"] == 1     # Gemma was not called again


def test_extra_requirements_are_scored(monkeypatch):
    # The model answers for the extra requirement too (index 4) and says it is
    # not proved: nothing about a database exists in this repository.
    judgements = JUDGEMENTS + [
        {"id": 4, "supports": "none", "proof": None, "reason": "no database code"},
    ]
    patch_pipeline(monkeypatch, make_fake(), judgements=judgements)
    result = report.analyze_repo("https://github.com/acme/widget",
                                 ["Uses a database"])
    reqs = [c for c in result["claims"] if c["type"] == "requirement"]
    assert len(reqs) == 1
    assert reqs[0]["status"] == "Unverified"   # nothing about a database exists
    assert reqs[0]["evidence"] == []


def test_a_claim_the_model_skips_is_not_judged(monkeypatch):
    """A claim with no judgement is reported as Not judged, never guessed."""
    # The model only answers for the first two claims.
    patch_pipeline(monkeypatch, make_fake(), judgements=JUDGEMENTS[:2])
    result = report.analyze_repo("https://github.com/acme/widget")

    skipped = [c for c in result["claims"] if c["status"] == "Not judged"]
    assert skipped, "the unjudged claims must be reported"
    assert result["summary"]["not_judged"] == len(skipped)
    # Not judged claims are left out of the claims score entirely.
    assert result["score"]["breakdown"]["claims"]["judged_claims"] == 2
    assert any("could not be judged" in b for b in result["banners"])


def test_limited_analysis_banner_and_not_checked(monkeypatch):
    fake = make_fake(languages={"Rust": 5000},
                     files={"src/main.rs": "fn main() {}\n"})
    patch_pipeline(monkeypatch, fake)
    result = report.analyze_repo("https://github.com/acme/widget")

    assert result["analysis_type"] == "limited"
    assert any("Limited analysis" in b for b in result["banners"])
    checks = {c["id"]: c["status"] for c in result["quality_checks"]}
    assert checks["structure"] == "not_checked"
    assert checks["secrets"] == "not_checked"
    assert checks["error_handling"] == "not_checked"


def test_files_skipped_banner(monkeypatch):
    chunk = "x = 1\n" * 5000  # about 30 KB per file
    files = {f"src/mod_{i}.py": chunk for i in range(20)}
    fake = make_fake(files=files, readme=README)
    patch_pipeline(monkeypatch, fake)
    result = report.analyze_repo("https://github.com/acme/widget")
    assert any("skipped" in b.lower() for b in result["banners"])
    assert result["files"]["skipped_count"] > 0
    assert result["files"]["skipped"]


def test_report_lists_files_read_and_skipped(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    result = report.analyze_repo("https://github.com/acme/widget")
    files = result["files"]
    assert files["read_count"] == len(files["read"])
    read_paths = {entry["path"] for entry in files["read"]}
    assert "main.py" in read_paths
    assert all(entry.get("reason") for entry in files["read"])
    assert all(entry.get("reason") for entry in files["skipped"])


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_export_sections(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    result = report.analyze_repo("https://github.com/acme/widget")
    md = report.to_markdown(result)

    assert "# RepoJudge report: acme/widget" in md
    assert "## Summary" in md
    assert "## Score breakdown" in md
    assert "## Claims" in md
    assert "## Engineering checks" in md
    assert "## Files read" in md
    assert "## Files skipped" in md
    assert "## Repository health" in md
    assert "Unverified" in md
    assert "main.py" in md
    assert "never guesses" in md


def test_markdown_shows_not_applicable_and_bonus_checks(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    result = report.analyze_repo("https://github.com/acme/widget")
    md = report.to_markdown(result)
    # CI is missing in this repository, so it must read as a bonus, not a fail.
    assert "not checked" in md


def test_markdown_never_prints_a_question_mark_evidence():
    fake_report = {
        "repo": {"full_name": "a/b", "url": "u", "language": "Python"},
        "analysis_type": "full", "model": "gemma-4-26b-a4b-it",
        "generated_at": "now", "banners": [], "warnings": [], "files": {},
        "summary": {"verified": 0, "partial": 0, "unverified": 1, "total": 1},
        "score": {"total": 0, "grade": "F", "breakdown": {}},
        "claims": [],
        "quality_checks": [{
            "id": "gitignore", "name": "Has .gitignore", "status": "fail",
            "reason": "No .gitignore was found.",
            "evidence": [{"file": None, "line": None, "snippet": "missing"}],
        }],
    }
    md = report.to_markdown(fake_report)
    assert "?" not in md
    assert "none" in md


def test_markdown_pipes_are_escaped():
    fake_report = {
        "repo": {"full_name": "a/b", "url": "u", "language": "Python"},
        "analysis_type": "full", "model": "gemma-4-26b-a4b-it",
        "generated_at": "now", "banners": [], "warnings": [],
        "summary": {"verified": 1, "partial": 0, "unverified": 0, "total": 1},
        "score": {"total": 80, "grade": "B",
                  "breakdown": {"claims": {"score": 50, "max": 50, "reason": "ok"},
                                "quality": {"score": 30, "max": 30, "reason": "ok"},
                                "health": {"score": 10, "max": 20, "items": []}}},
        "claims": [{"claim": "has a|b feature", "type": "feature",
                    "status": "Verified", "reason": "r|e",
                    "evidence": [{"file": "f.py", "line": 2}]}],
        "quality_checks": [],
    }
    md = report.to_markdown(fake_report)
    assert "\\|" in md


def test_to_json_is_valid(monkeypatch):
    import json
    patch_pipeline(monkeypatch, make_fake())
    result = report.analyze_repo("https://github.com/acme/widget")
    assert json.loads(report.to_json(result))["repo"]["name"] == "widget"
