"""Shared error types.

All of these map to a clean JSON error response from the API:
``{"error": {"code": "...", "message": "..."}}``.
"""
from __future__ import annotations


class RepoJudgeError(Exception):
    """Base class for expected, user-facing failures."""

    code: str = "error"
    status_code: int = 400

    def __init__(self, message: str, code: str | None = None,
                 status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class InvalidRepoUrlError(RepoJudgeError):
    code = "invalid_url"
    status_code = 400


class RepoNotFoundError(RepoJudgeError):
    code = "repo_not_found"
    status_code = 404


class PrivateRepoError(RepoJudgeError):
    code = "private_repo"
    status_code = 403


class NoReadmeError(RepoJudgeError):
    code = "no_readme"
    status_code = 422


class GitHubRateLimitError(RepoJudgeError):
    code = "github_rate_limit"
    status_code = 429


class GemmaUnavailableError(RepoJudgeError):
    code = "gemma_unavailable"
    status_code = 503


class GemmaRateLimitError(RepoJudgeError):
    code = "gemma_rate_limit"
    status_code = 429


class GemmaTimeoutError(RepoJudgeError):
    code = "gemma_timeout"
    status_code = 504


class GemmaResponseError(RepoJudgeError):
    code = "gemma_bad_response"
    status_code = 502
