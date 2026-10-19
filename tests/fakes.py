"""Shared test fakes: a tiny in-memory GitHub API and Gemma stub.

Nothing here touches the network. Import these helpers from tests.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Optional


class FakeResponse:
    """Just enough of ``requests.Response`` for our code and tests."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        headers: Optional[dict] = None,
        content: Optional[bytes] = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        if text:
            self.text = text
        elif payload is not None:
            self.text = json.dumps(payload)
        else:
            self.text = ""
        self.headers = headers or {}
        self.content = content if content is not None else self.text.encode("utf-8")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("response has no JSON body")
        return self._payload


class FakeGitHub:
    """A minimal, deterministic stand-in for the GitHub REST API.

    Build one with ``FakeGitHub(files={"main.py": "..."}, readme="...")`` and
    pass it to :func:`fetch_repo(..., session=fake)`.
    """

    def __init__(
        self,
        *,
        owner: str = "acme",
        name: str = "widget",
        files: Optional[dict[str, str]] = None,
        directories: Optional[list[str]] = None,
        readme: Optional[str] = "# Widget\nA tiny widget library.\n",
        languages: Optional[dict[str, int]] = None,
        license_name: Optional[str] = "MIT License",
        license_spdx: Optional[str] = "MIT",
        private: bool = False,
        commit_count: int = 7,
        default_branch: str = "main",
        description: str = "A widget",
        stars: int = 12,
        last_commit_date: str = "2026-09-01T10:00:00Z",
        sha: str = "deadbeefcafe",
        truncate_tree: bool = False,
        repo_status: int = 200,
        commits_status: int = 200,
        readme_status: int = 200,
    ) -> None:
        self.owner = owner
        self.name = name
        self.files = dict(files or {})
        self.directories = list(directories or [])
        self.readme = readme
        self.languages = languages if languages is not None else {"Python": 5000, "Shell": 100}
        self.license_name = license_name
        self.license_spdx = license_spdx
        self.private = private
        self.commit_count = commit_count
        self.default_branch = default_branch
        self.description = description
        self.stars = stars
        self.last_commit_date = last_commit_date
        self.sha = sha
        self.truncate_tree = truncate_tree
        self.repo_status = repo_status
        self.commits_status = commits_status
        self.readme_status = readme_status
        self.requests: list[str] = []

    # -- helpers -----------------------------------------------------------
    @property
    def base(self) -> str:
        return f"/repos/{self.owner}/{self.name}"

    def _tree(self) -> list[dict]:
        entries: list[dict] = []
        for path, text in self.files.items():
            entries.append({
                "path": path,
                "type": "blob",
                "size": len(text.encode("utf-8")),
                "sha": "f" + str(abs(hash(path)))[:10],
            })
        for d in self.directories:
            entries.append({"path": d, "type": "tree", "size": 0})
        return entries

    # -- request handling --------------------------------------------------
    def get(self, url: str, params: Optional[dict] = None,
            headers: Optional[dict] = None, timeout: int | None = None) -> FakeResponse:
        self.requests.append(url)
        params = params or {}

        # Raw file content (raw.githubusercontent.com)
        raw_marker = f"/{self.owner}/{self.name}/"
        if "raw.githubusercontent.com" in url:
            _, _, rest = url.partition("raw.githubusercontent.com")
            parts = rest.strip("/").split("/", 3)
            path = parts[3] if len(parts) > 3 else ""
            if path in self.files:
                return FakeResponse(text=self.files[path])
            return FakeResponse(status_code=404, text="not found")

        if url.endswith(f"{self.base}/readme"):
            if self.readme_status != 200 or self.readme is None:
                return FakeResponse(status_code=404, payload={"message": "Not Found"})
            encoded = base64.b64encode(self.readme.encode("utf-8")).decode("ascii")
            return FakeResponse(payload={
                "path": "README.md",
                "content": encoded,
                "encoding": "base64",
                "download_url": f"https://raw.githubusercontent.com/{self.owner}/{self.name}/{self.sha}/README.md",
            })

        if url.endswith(f"{self.base}/languages"):
            return FakeResponse(payload=self.languages)

        if f"{self.base}/git/trees/" in url:
            return FakeResponse(payload={
                "tree": self._tree(),
                "truncated": self.truncate_tree,
            })

        if url.endswith(f"{self.base}/commits"):
            if self.commits_status != 200:
                return FakeResponse(status_code=self.commits_status,
                                    payload={"message": "rate limited"})
            per_page = int(params.get("per_page", 30))
            payload = [{
                "sha": self.sha,
                "commit": {"committer": {"date": self.last_commit_date}},
            }]
            headers: dict[str, str] = {}
            pages = max(1, -(-self.commit_count // per_page))
            if pages > 1:
                headers["Link"] = (
                    f'<https://api.github.com/commits?per_page={per_page}&page={pages}>; '
                    'rel="last"'
                )
            return FakeResponse(payload=payload, headers=headers)

        if url.endswith(self.base):
            if self.repo_status != 200:
                return FakeResponse(status_code=self.repo_status,
                                    payload={"message": "Not Found"})
            payload = {
                "full_name": f"{self.owner}/{self.name}",
                "private": self.private,
                "visibility": "private" if self.private else "public",
                "default_branch": self.default_branch,
                "description": self.description,
                "homepage": "",
                "stargazers_count": self.stars,
                "forks_count": 2,
                "open_issues_count": 1,
                "language": next(iter(self.languages), None),
                "created_at": "2024-01-01T00:00:00Z",
                "pushed_at": self.last_commit_date,
            }
            if self.license_name:
                payload["license"] = {
                    "key": (self.license_spdx or "mit").lower(),
                    "name": self.license_name,
                    "spdx_id": self.license_spdx,
                }
            else:
                payload["license"] = None
            return FakeResponse(payload=payload)

        return FakeResponse(status_code=404, payload={"message": "Not Found"})


def fake_gemma_response(claims: list[dict]) -> str:
    """Serialise a claims list the way Gemma would return it."""
    return json.dumps({"claims": claims})


def local_repo(directory, *, analysis_type: str = "full", language: str = "Python"):
    """Build a RepoData from a local folder (used for the demo repo).

    Every readable text file becomes a fetched file, except the README, which
    is stored as the README. Imported lazily so this module stays dependency
    free at import time.
    """
    from pathlib import Path

    from app.github_client import RepoData, RepoFile

    root = Path(directory)
    files: dict = {}
    tree: list[str] = []
    readme = None
    readme_path = None

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in {"__pycache__", ".git", ".venv", "node_modules"}
               for part in rel.split("/")):
            continue
        tree.append(rel)
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if rel.rsplit("/", 1)[-1].lower().startswith("readme"):
            readme = text
            readme_path = rel
            continue
        files[rel] = RepoFile(rel, text, len(text.encode("utf-8")),
                              text.count("\n") + 1)

    return RepoData(
        url="https://github.com/demo/number-buddy",
        owner="demo",
        name="number-buddy",
        default_branch="main",
        private=False,
        description="Tiny demo project for RepoJudge.",
        homepage="",
        stars=3,
        forks=0,
        open_issues=0,
        language=language,
        languages={language: 1200},
        license_name="MIT License",
        license_spdx="MIT",
        readme=readme,
        readme_path=readme_path or "README.md",
        tree=tree,
        tree_truncated=False,
        files=files,
        files_considered=len(files),
        files_skipped=0,
        commit_count=4,
        last_commit_date="2026-09-01T00:00:00Z",
        last_commit_sha="demo-sha",
        created_at="2026-01-01T00:00:00Z",
        pushed_at="2026-09-01T00:00:00Z",
        analysis_type=analysis_type,
    )
