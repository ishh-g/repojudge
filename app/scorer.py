"""Milestone 5: turn claims, quality checks and repo health into a score.

Total is out of 100:
  * claims   50 points - Verified = full, Partial = half, Unverified = 0,
              spread evenly across every claim (including extra requirements)
  * quality  30 points - split across the engineering checks. Checks that were
              "not checked" or "not applicable" are removed from the total, so
              they never lower the score.
  * health   20 points - license, commit history, recent activity, README.

Every part returns a reason, not just a number.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from . import config
from .github_client import RepoData

# Points awarded per claim status.
STATUS_FACTOR = {"Verified": 1.0, "Partial": 0.5, "Unverified": 0.0}

# Claim statuses that are removed from the score entirely. A claim the model
# never judged is not evidence against the project, so it is left out of the
# total (just like a "not checked" engineering check).
CLAIM_EXCLUDED_STATUSES = {"Not judged"}

# Quality-check statuses that are removed from the score entirely.
EXCLUDED_CHECK_STATUSES = {"not_checked", "not_applicable"}

HEALTH_MAX = {
    "license": 5,
    "commit_history": 5,
    "recent_activity": 5,
    "readme": 5,
}

# A commit is "recent" if it landed within this many days.
RECENT_DAYS = 365
VERY_RECENT_DAYS = 90


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _round(value: float) -> float:
    return round(value + 1e-9, 1)


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def score_claims(claims: list[dict]) -> dict:
    max_points = config.CLAIM_POINTS
    counts = {"Verified": 0, "Partial": 0, "Unverified": 0, "Not judged": 0}
    for claim in claims:
        status = claim.get("status", "Unverified")
        counts[status if status in counts else "Unverified"] += 1

    total = len(claims)
    not_judged = counts["Not judged"]
    judged = [c for c in claims if c.get("status") not in CLAIM_EXCLUDED_STATUSES]

    if total == 0:
        return {
            "score": 0.0,
            "max": max_points,
            "total_claims": 0,
            "judged_claims": 0,
            "verified": 0,
            "partial": 0,
            "unverified": 0,
            "not_judged": 0,
            "reason": "No claims about the project were found in the README.",
        }

    if not judged:
        # Nothing could be judged: do not punish the project for that.
        return {
            "score": float(max_points),
            "max": max_points,
            "total_claims": total,
            "judged_claims": 0,
            "verified": 0,
            "partial": 0,
            "unverified": 0,
            "not_judged": not_judged,
            "reason": (
                f"None of the {total} claim(s) could be judged, so this part "
                "was not scored down."
            ),
        }

    per_claim = max_points / len(judged)
    earned = sum(per_claim * STATUS_FACTOR.get(c.get("status"), 0.0)
                 for c in judged)

    reason = (
        f"{counts['Verified']} verified, {counts['Partial']} partial and "
        f"{counts['Unverified']} unverified out of {len(judged)} judged claims"
    )
    reason += (f" ({not_judged} not judged)." if not_judged
               else f" (of {total} in total).")

    return {
        "score": _round(earned),
        "max": max_points,
        "total_claims": total,
        "judged_claims": len(judged),
        "verified": counts["Verified"],
        "partial": counts["Partial"],
        "unverified": counts["Unverified"],
        "not_judged": not_judged,
        "per_claim": _round(per_claim),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Engineering quality
# ---------------------------------------------------------------------------


def score_quality(checks: list[dict]) -> dict:
    max_points = config.QUALITY_POINTS
    checked = [c for c in checks if c.get("status") not in EXCLUDED_CHECK_STATUSES]
    excluded = [c for c in checks if c.get("status") in EXCLUDED_CHECK_STATUSES]
    passed = [c for c in checked if c.get("status") == "pass"]
    failed = [c for c in checked if c.get("status") == "fail"]

    total_weight = sum(c.get("weight", 0) for c in checked)
    earned_weight = sum(c.get("weight", 0) for c in passed)

    if total_weight <= 0:
        # Nothing could be checked: do not punish the project.
        score = float(max_points)
        reason = "No engineering checks could be run, so this part was not scored down."
    else:
        score = max_points * (earned_weight / total_weight)
        reason = (
            f"{len(passed)} of {len(checked)} engineering checks passed"
            + (f" ({len(excluded)} not counted)." if excluded else ".")
        )

    return {
        "score": _round(score),
        "max": max_points,
        "passed": len(passed),
        "failed": len(failed),
        "not_checked": len(excluded),
        "reason": reason,
        "failed_checks": [c.get("id") for c in failed],
    }


# ---------------------------------------------------------------------------
# Repository health
# ---------------------------------------------------------------------------


def score_health(repo: RepoData) -> dict:
    items: list[dict] = []

    # License ("NOASSERTION" means GitHub could not identify one).
    spdx = repo.license_spdx if repo.license_spdx not in (None, "", "NOASSERTION") else None
    if repo.license_name or spdx:
        label = spdx or repo.license_name
        items.append({
            "id": "license", "name": "License",
            "status": "pass", "earned": HEALTH_MAX["license"],
            "max": HEALTH_MAX["license"],
            "reason": f"A license is present ({label}).",
        })
    else:
        items.append({
            "id": "license", "name": "License",
            "status": "fail", "earned": 0.0, "max": HEALTH_MAX["license"],
            "reason": "No license file was detected.",
        })

    # Commit history
    commits = repo.commit_count
    if commits >= 10:
        earned, status, note = (HEALTH_MAX["commit_history"], "pass",
                                f"{commits} commits.")
    elif commits >= 1:
        factor = 0.6 if commits >= 2 else 0.3
        earned = HEALTH_MAX["commit_history"] * factor
        status = "partial"
        note = (f"Only {commits} commits." if commits >= 2
                else "Only a single commit.")
    else:
        earned, status, note = 0.0, "fail", "No commit history was found."
    items.append({
        "id": "commit_history", "name": "Commit history",
        "status": status,
        "earned": _round(earned), "max": HEALTH_MAX["commit_history"],
        "reason": note,
    })

    # Recent activity
    last = _parse_date(repo.last_commit_date)
    max_recent = HEALTH_MAX["recent_activity"]
    if last is None:
        items.append({
            "id": "recent_activity", "name": "Recent activity",
            "status": "fail", "earned": 0.0, "max": max_recent,
            "reason": "The date of the last commit is unknown.",
        })
    else:
        now = datetime.now(timezone.utc)
        age_days = (now - last).days
        if age_days <= VERY_RECENT_DAYS:
            earned, status = max_recent, "pass"
        elif age_days <= RECENT_DAYS:
            earned, status = max_recent * 0.6, "pass"
        else:
            earned, status = max_recent * 0.3, "fail"
        items.append({
            "id": "recent_activity", "name": "Recent activity",
            "status": status, "earned": _round(earned), "max": max_recent,
            "reason": f"The last commit was about {age_days} day(s) ago.",
        })

    # README (required to analyse at all, so normally present)
    if repo.readme and repo.readme.strip():
        items.append({
            "id": "readme", "name": "README",
            "status": "pass", "earned": HEALTH_MAX["readme"],
            "max": HEALTH_MAX["readme"],
            "reason": "A README is present.",
        })
    else:
        items.append({
            "id": "readme", "name": "README",
            "status": "fail", "earned": 0.0, "max": HEALTH_MAX["readme"],
            "reason": "No README was found.",
        })

    earned_total = sum(i["earned"] for i in items)
    passed = sum(1 for i in items if i["status"] == "pass")
    return {
        "score": _round(earned_total),
        "max": config.HEALTH_POINTS,
        "items": items,
        "reason": f"{passed} of {len(items)} repository health checks passed.",
    }


# ---------------------------------------------------------------------------
# Total
# ---------------------------------------------------------------------------


def _grade(total: float) -> str:
    if total >= 90:
        return "A"
    if total >= 80:
        return "B"
    if total >= 70:
        return "C"
    if total >= 60:
        return "D"
    if total >= 50:
        return "E"
    return "F"


def score_report(repo: RepoData, claims: list[dict],
                 checks: list[dict]) -> dict:
    """Combine the three parts into the final 100-point breakdown."""
    claims_part = score_claims(claims)
    quality_part = score_quality(checks)
    health_part = score_health(repo)

    total = claims_part["score"] + quality_part["score"] + health_part["score"]
    total = _round(total)

    return {
        "total": total,
        "max": 100,
        "grade": _grade(total),
        "breakdown": {
            "claims": claims_part,
            "quality": quality_part,
            "health": health_part,
        },
    }
