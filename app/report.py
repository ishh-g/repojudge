"""Build the final report object and export it as JSON or Markdown.

:func:`analyze_repo` is the whole pipeline in one place:

    fetch repo -> extract claims (Gemma 1) -> find evidence + judge (Gemma 2)
    -> engineering checks -> score -> report

The finished report is cached by repo URL + latest commit + requirements text.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from . import cache, claims as claims_module, config, quality_checks, scorer, verifier
from .github_client import RepoData, add_keyword_lines, fetch_repo

LIMITED_BANNER = "Limited analysis: code checks are not available for this language"

# Readable names for the check statuses used in Markdown.
STATUS_LABELS = {
    "pass": "pass",
    "fail": "fail",
    "not_checked": "not checked",
    "not_applicable": "not applicable",
}

# Readable names for the claim statuses. "Unverified" always says what it
# means: RepoJudge found no proof, which is not the same as "false".
CLAIM_STATUS_LABELS = {
    "Verified": "Verified",
    "Partial": "Partial",
    "Unverified": "Unverified (no proof found in the code read)",
    "Not judged": "Not judged",
}


def _status_label(status) -> str:
    status = str(status or "")
    return STATUS_LABELS.get(status, status)


def _claim_status_label(status) -> str:
    status = str(status or "")
    return CLAIM_STATUS_LABELS.get(status, status)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _report_id(repo: RepoData, requirements: list[str]) -> str:
    return cache.make_key("report", repo.url, repo.last_commit_sha or "",
                          "\n".join(requirements))


def _banners(repo: RepoData) -> list[str]:
    banners: list[str] = []
    if repo.analysis_type == "limited":
        banners.append(LIMITED_BANNER)
    if repo.files_skipped > 0:
        banners.append(
            f"{repo.files_skipped} file(s) were skipped to keep the analysis fast. "
            "The result may not cover the whole codebase."
        )
    if repo.tree_truncated:
        banners.append(
            "GitHub truncated the file listing, so very large repositories may be "
            "partially analysed."
        )
    return banners


def _repo_summary(repo: RepoData) -> dict:
    return {
        "url": repo.url,
        "owner": repo.owner,
        "name": repo.name,
        "full_name": repo.full_name,
        "description": repo.description,
        "homepage": repo.homepage,
        "language": repo.language,
        "languages": repo.languages,
        "stars": repo.stars,
        "forks": repo.forks,
        "open_issues": repo.open_issues,
        "license": repo.license_name,
        "license_spdx": repo.license_spdx,
        "default_branch": repo.default_branch,
        "commit_count": repo.commit_count,
        "last_commit_date": repo.last_commit_date,
        "last_commit_sha": repo.last_commit_sha,
        "readme_path": repo.readme_path,
        "files_read": len(repo.files),
        "files_considered": repo.files_considered,
        "files_skipped": repo.files_skipped,
    }


def _files_summary(repo: RepoData) -> dict:
    """Which files were read and which were skipped, each with a reason.

    The fetcher stores this directly; for in-memory repositories (such as the
    test fake) it is rebuilt from the files we hold.
    """
    stored = repo.files_report or {}
    read = [dict(entry) for entry in (stored.get("read") or [])]
    skipped = [dict(entry) for entry in (stored.get("skipped") or [])]

    known = {entry.get("path") for entry in read}
    for path, repo_file in sorted(repo.files.items()):
        if path in known:
            continue
        read.append({
            "path": path,
            "size": repo_file.size,
            "lines": repo_file.lines,
            "truncated": bool(repo_file.truncated),
            "reason": repo_file.reason or "Read as source text.",
        })

    return {
        "read": read,
        "skipped": skipped,
        "read_count": len(read),
        "skipped_count": len(skipped),
        "truncated_count": sum(1 for entry in read if entry.get("truncated")),
    }


def build_report(repo: RepoData, claim_list: list[dict], checks: list[dict],
                 requirements: list[str], *, cache_hit: bool = False) -> dict:
    """Assemble the report dict from already-computed parts."""
    score = scorer.score_report(repo, claim_list, checks)
    summary = {
        "verified": sum(1 for c in claim_list if c.get("status") == "Verified"),
        "partial": sum(1 for c in claim_list if c.get("status") == "Partial"),
        "unverified": sum(1 for c in claim_list if c.get("status") == "Unverified"),
        "not_judged": sum(1 for c in claim_list
                          if c.get("status") == "Not judged"),
        "total": len(claim_list),
    }
    banners = _banners(repo)
    if summary["not_judged"]:
        banners.append(
            f"{summary['not_judged']} claim(s) could not be judged. They are "
            "reported as \"Not judged\" and left out of the claims score."
        )
    return {
        "id": _report_id(repo, requirements),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.GEMMA_MODEL,
        "repo": _repo_summary(repo),
        "files": _files_summary(repo),
        "analysis_type": repo.analysis_type,
        "banners": banners,
        "requirements": requirements,
        "claims": claim_list,
        "quality_checks": checks,
        "summary": summary,
        "score": score,
        "warnings": repo.warnings,
        "cache_hit": cache_hit,
    }


def analyze_repo(
    repo_url: str,
    requirements: Optional[list[str]] = None,
    *,
    force_refresh: bool = False,
    use_cache: bool = True,
    gh_session=None,
    gemma_session=None,
) -> dict:
    """Run (or load from cache) the full RepoJudge analysis.

    Raises :class:`app.errors.RepoJudgeError` subclasses for clean API errors.
    """
    requirements = [str(r).strip() for r in (requirements or []) if str(r).strip()]

    # We must always read the repo first: the cache key includes the latest commit.
    repo = fetch_repo(repo_url, session=gh_session)
    report_id = _report_id(repo, requirements)

    if use_cache and not force_refresh:
        cached = cache.get(report_id)
        if cached and "score" in cached:
            cached["cache_hit"] = True
            cached["id"] = report_id
            return cached

    # A re-run without cache ignores every cache, including the claims and
    # judging sub-caches, so the whole analysis really runs again.
    sub_cache = use_cache and not force_refresh

    # Gemma call 1 - claims.
    claim_list = claims_module.extract_claims(
        repo, requirements, session=gemma_session, use_cache=sub_cache
    )
    # A large file is only read in part. Now that we know the claim keywords,
    # keep the later lines of those files that match a keyword, so evidence can
    # still point at the right line.
    add_keyword_lines(
        repo,
        [keyword for claim in claim_list for keyword in (claim.get("keywords") or [])],
        session=gh_session,
    )
    # Gemma call 2 - evidence + judging.
    judged = verifier.verify_claims(
        repo, claim_list, session=gemma_session, use_cache=sub_cache
    )
    # Rule-based engineering checks (no LLM).
    checks = quality_checks.run_quality_checks(repo)

    report = build_report(repo, judged, checks, requirements, cache_hit=False)
    if sub_cache:
        cache.set(report_id, report)
    return report


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def _md_cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _evidence_text(evidence: list[dict]) -> str:
    parts = []
    for hit in evidence[:3]:
        path = hit.get("file")
        if not path:
            # A finding without a file (for example "there is no .gitignore")
            # has no location to point at, so it is reported as "none".
            continue
        location = str(path)
        if hit.get("line"):
            location += f":{hit['line']}"
        parts.append(location)
    return ", ".join(parts) if parts else "none"


def to_markdown(report: dict) -> str:
    """Render a report dict as Markdown."""
    repo = report.get("repo", {})
    score = report.get("score", {})
    breakdown = score.get("breakdown", {})
    summary = report.get("summary", {})

    lines: list[str] = []
    lines.append(f"# RepoJudge report: {repo.get('full_name', 'unknown')}")
    lines.append("")
    lines.append(f"**Score:** {score.get('total', 0)} / 100 "
                 f"(grade {score.get('grade', '-')})  ")
    lines.append(f"**Repository:** {repo.get('url', '')}  ")
    lines.append(f"**Main language:** {repo.get('language', 'unknown')}  ")
    lines.append(f"**Analysis:** {report.get('analysis_type', 'full')}  ")
    lines.append(f"**Model:** {report.get('model', '')}  ")
    lines.append(f"**Generated:** {report.get('generated_at', '')}")
    lines.append("")

    for banner in report.get("banners", []):
        lines.append(f"> {banner}")
    if report.get("banners"):
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Verified: {summary.get('verified', 0)}")
    lines.append(f"- Partial: {summary.get('partial', 0)}")
    lines.append(f"- Unverified: {summary.get('unverified', 0)}")
    if summary.get("not_judged"):
        lines.append(f"- Not judged: {summary.get('not_judged', 0)}")
    lines.append(f"- Total claims: {summary.get('total', 0)}")
    lines.append("")

    lines.append("## Score breakdown")
    lines.append("")
    lines.append("| Part | Score | Max | Notes |")
    lines.append("| --- | --- | --- | --- |")
    for key, label in (("claims", "Claims"), ("quality", "Engineering quality"),
                       ("health", "Repository health")):
        part = breakdown.get(key, {})
        lines.append(
            f"| {label} | {part.get('score', 0)} | {part.get('max', 0)} | "
            f"{_md_cell(part.get('reason', ''))} |"
        )
    lines.append("")

    lines.append("## Claims")
    lines.append("")
    lines.append("| # | Claim | Type | Status | Reason | Evidence |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for index, claim in enumerate(report.get("claims", []), start=1):
        lines.append(
            f"| {index} | {_md_cell(claim.get('claim'))} | "
            f"{_md_cell(claim.get('type'))} | "
            f"{_md_cell(_claim_status_label(claim.get('status')))} | "
            f"{_md_cell(claim.get('reason'))} | "
            f"{_md_cell(_evidence_text(claim.get('evidence', [])))} |"
        )
    if not report.get("claims"):
        lines.append("| - | No claims were found. | | | | |")
    lines.append("")

    lines.append("## Engineering checks")
    lines.append("")
    lines.append("| Check | Status | Reason | Evidence |")
    lines.append("| --- | --- | --- | --- |")
    for check in report.get("quality_checks", []):
        lines.append(
            f"| {_md_cell(check.get('name'))} | "
            f"{_md_cell(_status_label(check.get('status')))} | "
            f"{_md_cell(check.get('reason'))} | "
            f"{_md_cell(_evidence_text(check.get('evidence', [])))} |"
        )
    lines.append("")

    files = report.get("files", {}) or {}
    read_files = files.get("read") or []
    skipped_files = files.get("skipped") or []
    lines.append("## Files read")
    lines.append("")
    lines.append(
        f"RepoJudge read {len(read_files)} file(s) and skipped "
        f"{len(skipped_files)}. Every skipped file is listed with its reason."
    )
    lines.append("")
    lines.append("| File | Size | Lines | Note |")
    lines.append("| --- | --- | --- | --- |")
    for entry in read_files:
        note = entry.get("reason") or "Read as source text."
        lines.append(
            f"| {_md_cell(entry.get('path'))} | {entry.get('size', 0)} bytes | "
            f"{entry.get('lines', 0)} | {_md_cell(note)} |"
        )
    if not read_files:
        lines.append("| - | No files were read. | | |")
    lines.append("")
    lines.append("## Files skipped")
    lines.append("")
    lines.append("| File | Reason |")
    lines.append("| --- | --- |")
    for entry in skipped_files:
        lines.append(
            f"| {_md_cell(entry.get('path'))} | {_md_cell(entry.get('reason'))} |"
        )
    if not skipped_files:
        lines.append("| - | No files were skipped. |")
    lines.append("")

    lines.append("## Repository health")
    lines.append("")
    lines.append("| Item | Status | Points | Reason |")
    lines.append("| --- | --- | --- | --- |")
    for item in breakdown.get("health", {}).get("items", []):
        lines.append(
            f"| {_md_cell(item.get('name'))} | {_md_cell(item.get('status'))} | "
            f"{item.get('earned', 0)} / {item.get('max', 0)} | "
            f"{_md_cell(item.get('reason'))} |"
        )
    lines.append("")

    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Every score and status above is backed by evidence, or reported as "
        "Unverified. RepoJudge searches the code it reads for evidence: "
        "Unverified means no proof was found, not that the claim is false. "
        "Check important results by hand. RepoJudge never guesses."
    )
    lines.append("")
    return "\n".join(lines)


def to_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
