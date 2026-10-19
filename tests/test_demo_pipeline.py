"""Milestone 8 tests: the demo repository and the end-to-end pipeline.

The demo README makes 5 claims. Three are true and provable from the code, two
are false. This module proves that RepoJudge reports the two false claims as
"Unverified" even when Gemma incorrectly insists they are all "Verified".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import claims as claims_module
from app import config, quality_checks, report, verifier
from fakes import local_repo

DEMO_REPO = Path(__file__).resolve().parent.parent / "demo" / "fake_repo"

# The claims the demo README makes, as Gemma would extract them.
DEMO_CLAIMS = [
    {"claim": "Provides a command line interface for adding two numbers",
     "type": "feature", "keywords": ["argparse"]},
    {"claim": "Uses the requests library to fetch a number from a URL",
     "type": "tech_used", "keywords": ["requests"]},
    {"claim": "Includes a unit test suite that runs with pytest",
     "type": "feature", "keywords": ["pytest", "test_"]},
    {"claim": "Stores every result in a PostgreSQL database",
     "type": "feature", "keywords": ["postgres", "psycopg2", "database"]},
    {"claim": "Supports Docker deployment with a one-command build",
     "type": "feature", "keywords": ["docker", "dockerfile"]},
]

TRUE_INDEXES = {0, 1, 2}
FALSE_INDEXES = {3, 4}


def overconfident_judgements(repo):
    """A deliberately overconfident model: it calls every claim "full".

    It still has to point its proof at one of the evidence lines it was given
    (or null when it was given none), because the code checks that the proof is
    real before it accepts a verdict.
    """
    judgements = []
    for index, claim in enumerate(verifier.attach_evidence(repo, DEMO_CLAIMS)):
        evidence = claim["evidence"]
        proof = None
        if evidence:
            proof = f"{evidence[0]['file']}:{evidence[0]['line']}"
        judgements.append({
            "id": index, "supports": "full", "proof": proof,
            "reason": "Looks fine to me.",
        })
    return judgements


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    yield


@pytest.fixture
def demo_repo():
    return local_repo(DEMO_REPO)


# ---------------------------------------------------------------------------
# The demo repo itself
# ---------------------------------------------------------------------------


def test_demo_repo_exists_with_expected_files():
    assert (DEMO_REPO / "README.md").exists()
    assert (DEMO_REPO / "main.py").exists()
    assert (DEMO_REPO / "requirements.txt").exists()
    assert (DEMO_REPO / "tests" / "test_calculator.py").exists()


def test_demo_repo_reads_cleanly(demo_repo):
    assert demo_repo.readme and "Number Buddy" in demo_repo.readme
    assert "main.py" in demo_repo.files
    assert "requirements.txt" in demo_repo.files


# ---------------------------------------------------------------------------
# The core promise: false claims are Unverified
# ---------------------------------------------------------------------------


def test_false_claims_are_unverified_even_when_gemma_says_verified(
    demo_repo, monkeypatch
):
    monkeypatch.setattr(verifier, "generate_json", lambda *a, **k: {
        "judgements": overconfident_judgements(demo_repo),
    })
    results = verifier.verify_claims(demo_repo, DEMO_CLAIMS)

    for index in TRUE_INDEXES:
        assert results[index]["status"] == "Verified", results[index]["claim"]
        assert results[index]["evidence"], "true claims must have evidence"

    for index in FALSE_INDEXES:
        assert results[index]["status"] == "Unverified", results[index]["claim"]
        assert results[index]["evidence"] == []
        assert verifier.NO_PROOF_REASON == results[index]["reason"]


def test_evidence_points_at_real_files(demo_repo, monkeypatch):
    monkeypatch.setattr(verifier, "generate_json", lambda *a, **k: {
        "judgements": overconfident_judgements(demo_repo),
    })
    results = verifier.verify_claims(demo_repo, DEMO_CLAIMS)
    cli = results[0]
    locations = {(hit["file"], hit["line"]) for hit in cli["evidence"]}
    assert any(path == "main.py" for path, _ in locations)
    assert all(isinstance(line, int) and line > 0 for _, line in locations)


# ---------------------------------------------------------------------------
# Whole pipeline on the demo repo
# ---------------------------------------------------------------------------


def test_full_pipeline_on_demo_repo(monkeypatch, demo_repo):
    calls = {"claims": 0, "judge": 0}

    monkeypatch.setattr(report, "fetch_repo", lambda url, **kw: demo_repo)

    def fake_claims(prompt, **kwargs):
        calls["claims"] += 1
        return {"claims": DEMO_CLAIMS}

    def fake_judge(prompt, **kwargs):
        calls["judge"] += 1
        return {"judgements": overconfident_judgements(demo_repo)}

    monkeypatch.setattr(claims_module, "generate_json", fake_claims)
    monkeypatch.setattr(verifier, "generate_json", fake_judge)

    result = report.analyze_repo("https://github.com/demo/number-buddy")

    assert calls == {"claims": 1, "judge": 1}     # one batch, nothing missing
    assert result["analysis_type"] == "full"
    assert result["summary"]["total"] == 5
    assert result["summary"]["verified"] == 3
    assert result["summary"]["unverified"] == 2
    assert result["summary"]["not_judged"] == 0

    statuses = {c["claim"]: c["status"] for c in result["claims"]}
    assert statuses["Stores every result in a PostgreSQL database"] == "Unverified"
    assert statuses["Supports Docker deployment with a one-command build"] == "Unverified"

    # The label the report shows makes it clear that "Unverified" means "no
    # proof was found", not "false".
    markdown = report.to_markdown(result)
    assert "Unverified (no proof found in the code read)" in markdown

    # The score must reflect the two unverified claims.
    assert 0 < result["score"]["total"] < 100
    assert result["score"]["breakdown"]["claims"]["unverified"] == 2

    assert "PostgreSQL" in markdown
    assert "Unverified" in markdown


def test_quality_checks_on_demo_repo(demo_repo):
    results = {c["id"]: c for c in quality_checks.run_quality_checks(demo_repo)}
    assert results["tests"]["status"] == "pass"
    assert results["dependencies"]["status"] == "pass"
    assert results["documentation"]["status"] == "pass"
    assert results["setup"]["status"] == "pass"
    assert results["error_handling"]["status"] == "pass"
    assert results["secrets"]["status"] == "pass"
    assert results["gitignore"]["status"] == "pass"
    # CI is a bonus: a missing workflow is not checked, never a failure.
    assert results["ci"]["status"] == "not_checked"
    assert results["ci"]["bonus"] is True
