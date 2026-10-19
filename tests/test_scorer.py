"""Milestone 5 tests: scoring, including "not checked" never lowering the score."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.github_client import RepoData
from app.scorer import score_claims, score_health, score_quality, score_report


def claim(status):
    return {"claim": "x", "type": "feature", "status": status, "reason": ""}


def check(check_id, status, weight):
    return {"id": check_id, "name": check_id, "status": status, "weight": weight,
            "reason": "", "evidence": [], "bonus": False}


def make_repo(**overrides) -> RepoData:
    now = datetime.now(timezone.utc)
    defaults = dict(
        url="https://github.com/acme/widget", owner="acme", name="widget",
        default_branch="main", private=False, description="", homepage="",
        stars=10, forks=1, open_issues=0, language="Python", languages={"Python": 1},
        license_name="MIT License", license_spdx="MIT", readme="# Widget\nstuff\n",
        readme_path="README.md", tree=["main.py"], tree_truncated=False, files={},
        files_considered=1, files_skipped=0, commit_count=25,
        last_commit_date=now.isoformat(), last_commit_sha="abc",
        created_at=None, pushed_at=None, analysis_type="full",
    )
    defaults.update(overrides)
    return RepoData(**defaults)


# ---------------------------------------------------------------------------
# Claims scoring
# ---------------------------------------------------------------------------


def test_claims_all_verified_is_full_marks():
    result = score_claims([claim("Verified")] * 4)
    assert result["score"] == 50.0


def test_claims_partial_is_half():
    result = score_claims([claim("Partial")] * 4)
    assert result["score"] == 25.0


def test_claims_spread_evenly_mixed():
    # 1 verified (1.0) + 1 partial (0.5) + 2 unverified (0) = 1.5/4 of 50
    result = score_claims([
        claim("Verified"), claim("Partial"),
        claim("Unverified"), claim("Unverified"),
    ])
    assert result["score"] == 18.8
    assert result["total_claims"] == 4
    assert result["verified"] == 1


def test_no_claims_scores_zero_with_reason():
    result = score_claims([])
    assert result["score"] == 0.0
    assert "No claims" in result["reason"]


def test_not_judged_claims_are_left_out_of_the_score():
    """A claim the model never judged must not count for or against anything."""
    result = score_claims([
        claim("Verified"), claim("Not judged"), claim("Not judged"),
    ])
    # Only the single judged claim counts, and it is fully verified.
    assert result["score"] == 50.0
    assert result["judged_claims"] == 1
    assert result["not_judged"] == 2
    assert result["total_claims"] == 3
    assert "not judged" in result["reason"]


def test_nothing_judged_does_not_punish_the_project():
    result = score_claims([claim("Not judged"), claim("Not judged")])
    assert result["score"] == 50.0
    assert result["judged_claims"] == 0
    assert result["not_judged"] == 2


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


def test_quality_all_pass_is_full():
    checks = [check("tests", "pass", 5), check("setup", "pass", 5)]
    assert score_quality(checks)["score"] == 30.0


def test_quality_half_pass():
    checks = [check("tests", "pass", 5), check("setup", "fail", 5)]
    assert score_quality(checks)["score"] == 15.0


def test_not_checked_does_not_lower_score():
    """A project should not be punished for checks we could not run."""
    checks = [
        check("tests", "pass", 5),
        check("setup", "pass", 5),
        check("secrets", "not_checked", 6),
        check("structure", "not_checked", 3),
    ]
    result = score_quality(checks)
    assert result["score"] == 30.0   # scaled to full, not 10/19
    assert result["not_checked"] == 2


def test_all_not_checked_is_full_marks():
    checks = [check("secrets", "not_checked", 6)]
    assert score_quality(checks)["score"] == 30.0


def test_quality_reports_failed_ids():
    checks = [check("tests", "fail", 5), check("ci", "fail", 2)]
    result = score_quality(checks)
    assert set(result["failed_checks"]) == {"tests", "ci"}


def test_not_applicable_is_excluded_like_not_checked():
    """A check that does not apply must not lower the score."""
    checks = [check("tests", "pass", 5),
              check("dependencies", "not_applicable", 4)]
    result = score_quality(checks)
    assert result["not_checked"] == 1
    assert result["score"] == 30.0
    assert result["failed_checks"] == []


# ---------------------------------------------------------------------------
# Health scoring
# ---------------------------------------------------------------------------


def test_health_fully_healthy():
    result = score_health(make_repo())
    assert result["score"] == 20.0
    assert all(i["status"] == "pass" for i in result["items"])


def test_health_no_license():
    repo = make_repo(license_name=None, license_spdx="NOASSERTION")
    result = score_health(repo)
    assert result["score"] == 15.0
    license_item = next(i for i in result["items"] if i["id"] == "license")
    assert license_item["status"] == "fail"


def test_health_stale_repo():
    old = (datetime.now(timezone.utc) - timedelta(days=800)).isoformat()
    repo = make_repo(last_commit_date=old)
    result = score_health(repo)
    recent = next(i for i in result["items"] if i["id"] == "recent_activity")
    assert recent["earned"] < recent["max"]


def test_health_few_commits():
    repo = make_repo(commit_count=1)
    commits = next(i for i in score_health(repo)["items"]
                   if i["id"] == "commit_history")
    assert commits["status"] == "partial"


def test_health_many_commits_passes():
    repo = make_repo(commit_count=25)
    commits = next(i for i in score_health(repo)["items"]
                   if i["id"] == "commit_history")
    assert commits["status"] == "pass"


# ---------------------------------------------------------------------------
# Total
# ---------------------------------------------------------------------------


def test_total_is_sum_of_parts():
    repo = make_repo()
    claims = [claim("Verified"), claim("Partial")]
    checks = [
        check("tests", "pass", 5), check("setup", "pass", 5),
        check("dependencies", "pass", 4), check("secrets", "pass", 6),
        check("structure", "pass", 3), check("documentation", "pass", 3),
        check("error_handling", "pass", 2), check("ci", "pass", 2),
    ]
    report = score_report(repo, claims, checks)
    assert report["max"] == 100
    assert report["breakdown"]["claims"]["score"] == 37.5
    assert report["breakdown"]["quality"]["score"] == 30.0
    assert report["breakdown"]["health"]["score"] == 20.0
    assert report["total"] == 87.5
    assert report["grade"] == "B"


def test_limited_analysis_never_lowers_quality_score():
    """LIMITED: code-based checks are not_checked, so quality stays fair."""
    repo = make_repo(analysis_type="limited", language="Rust", files={})
    checks = [
        check("tests", "pass", 5),
        check("setup", "pass", 5),
        check("dependencies", "pass", 4),
        check("secrets", "not_checked", 6),
        check("structure", "not_checked", 3),
        check("documentation", "pass", 3),
        check("error_handling", "not_checked", 2),
        check("ci", "fail", 2),
    ]
    result = score_quality(checks)
    # Checked weight = 5+5+4+3+2 = 19, earned = 17 of 19 -> 26.8 of 30
    assert result["score"] == 26.8
    assert result["not_checked"] == 3
