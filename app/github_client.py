"""GitHub REST API client.

RepoJudge only *reads* repository data. It never clones, installs or executes
code from a judged repository - every file is fetched as text and treated as
an untrusted string.

Public entry point: :func:`fetch_repo`, which returns one clean ``RepoData``
object with everything the rest of the pipeline needs.
"""
from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

import requests

from . import config
from . import keywords as keywords_module
from .errors import (
    GitHubRateLimitError,
    InvalidRepoUrlError,
    NoReadmeError,
    PrivateRepoError,
    RepoNotFoundError,
)

# ---------------------------------------------------------------------------
# File filtering rules
# ---------------------------------------------------------------------------

# Directories we never look inside (dependency dumps, build output, VCS data).
SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", "dist", "build", ".git", ".next", "vendor", "__pycache__",
    ".venv", "venv", "env", "target", "coverage", ".idea", ".vscode", "out",
    "bin", "obj", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    ".gradle", ".tox", ".cache", "bower_components", "jspm_packages",
})

# Lockfiles and generated manifests carry no signal for claim checking.
SKIP_FILE_NAMES: frozenset[str] = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "pipfile.lock", "cargo.lock", "composer.lock", "gemfile.lock",
    "npm-shrinkwrap.json", "go.sum",
})

# Binary / asset extensions we cannot read as source.
SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp", ".tiff",
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".bz2", ".xz",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp4", ".mp3", ".mov", ".avi",
    ".webm", ".wav", ".ogg", ".map", ".lock", ".pyc", ".pyo", ".class", ".jar",
    ".so", ".dll", ".exe", ".dylib", ".o", ".a", ".bin", ".dat", ".db",
    ".sqlite", ".ipynb", ".min.js", ".min.css", ".snap", ".patch", ".diff",
})

SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".h", ".cpp", ".cc", ".hpp", ".cxx", ".cs", ".kt",
    ".kts", ".swift", ".scala", ".sh", ".bash", ".zsh", ".sql", ".vue",
    ".svelte", ".html", ".css", ".scss", ".sass", ".less", ".dart", ".lua",
    ".pl", ".r", ".jl", ".ex", ".exs", ".clj", ".hs", ".elm", ".asm",
})

# Dependency / build declarations. Useful for "tech_used" claims.
DEP_FILES: frozenset[str] = frozenset({
    "requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py",
    "setup.cfg", "pipfile", "environment.yml", "package.json", "go.mod",
    "cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts", "composer.json",
    "gemfile", "pubspec.yaml", "mix.exs", "cmakelists.txt", "makefile",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
})

# Classic entry-point filenames. These are the best places to find claims.
ENTRY_NAMES: frozenset[str] = frozenset({
    "main.py", "app.py", "__main__.py", "manage.py", "wsgi.py", "asgi.py",
    "run.py", "cli.py", "server.py", "index.js", "index.ts", "index.jsx",
    "index.tsx", "main.js", "main.ts", "main.jsx", "main.tsx", "server.js",
    "server.ts", "app.js", "app.ts", "app.jsx", "app.tsx", "index.php",
    "main.go", "main.rs", "main.java", "main.c", "main.cpp",
})

CONFIG_EXTENSIONS: frozenset[str] = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env", ".properties",
    ".conf", ".xml",
})

# Entry pages that are always read when they exist, whatever the budget says.
ALWAYS_INCLUDE_NAMES: frozenset[str] = frozenset({"index.html", "index.htm"})

# File-looking tokens inside a README (and its project-structure listing).
PATH_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_./\\-]+\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|vue|svelte|html|htm|"
    r"css|scss|sass|less|json|ya?ml|toml|cfg|ini|md|txt|java|kt|kts|go|rs|rb|"
    r"php|c|h|cpp|cc|hpp|cs|swift|scala|sh|bash|sql|dart|lua|r|jl|ex|exs|clj|"
    r"hs|gradle|cmake|lock)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RepoFile:
    """One file RepoJudge read, stored as text.

    ``content`` is the part we analyse. For a very large file it holds the first
    part only (``truncated`` is then True) and ``extra_lines`` holds later lines
    that matched a claim keyword, keyed by their real line number so evidence
    can still point at the right place.
    """

    path: str
    content: str
    size: int
    lines: int
    truncated: bool = False
    extra_lines: dict[int, str] = field(default_factory=dict)
    head_bytes: int = 0
    reason: str = ""


@dataclass
class FilePlan:
    """A decision to read ``keep_bytes`` of a file."""

    path: str
    size: int
    keep_bytes: int
    truncated: bool
    reason: str


@dataclass
class RepoData:
    """Everything RepoJudge knows about a repository, as plain Python."""

    url: str
    owner: str
    name: str
    default_branch: str
    private: bool
    description: str
    homepage: str
    stars: int
    forks: int
    open_issues: int
    language: str
    languages: dict[str, int]
    license_name: Optional[str]
    license_spdx: Optional[str]
    readme: Optional[str]
    readme_path: Optional[str]
    tree: list[str]                    # every file path in the default branch
    tree_truncated: bool
    files: dict[str, RepoFile]         # path -> content of the important files
    files_considered: int              # candidate source files found
    files_skipped: int                 # candidates we did not read
    commit_count: int
    last_commit_date: Optional[str]
    last_commit_sha: Optional[str]
    created_at: Optional[str]
    pushed_at: Optional[str]
    analysis_type: str = "full"        # "full" or "limited"
    warnings: list[str] = field(default_factory=list)
    # Which files were read and which were skipped, with the reason for each.
    files_report: dict = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def file_paths(self) -> list[str]:
        return list(self.files.keys())


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

_REPO_PATH_RE = re.compile(r"^/([^/]+)/([^/]+?)(?:\.git)?/?$")


def parse_repo_url(url: str) -> tuple[str, str]:
    """Return ``(owner, name)`` for a public GitHub repo URL.

    Accepts ``https://github.com/owner/repo`` with an optional ``.git`` suffix,
    trailing slash and query string. Raises :class:`InvalidRepoUrlError`
    otherwise.
    """
    if not url or not isinstance(url, str):
        raise InvalidRepoUrlError("Please provide a GitHub repository URL.")

    raw = url.strip()
    if not raw:
        raise InvalidRepoUrlError("Please provide a GitHub repository URL.")

    if "://" not in raw:
        raw = "https://" + raw

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host not in {"github.com", "www.github.com"}:
        raise InvalidRepoUrlError(
            "Only public github.com repository URLs are supported "
            "(for example https://github.com/owner/repo)."
        )

    match = _REPO_PATH_RE.match(parsed.path or "")
    if not match:
        raise InvalidRepoUrlError(
            "That does not look like a repository URL. "
            "Expected https://github.com/owner/repo."
        )

    owner, name = match.group(1), match.group(2)
    if not owner or not name or name in {".", ".."}:
        raise InvalidRepoUrlError(
            "That does not look like a repository URL. "
            "Expected https://github.com/owner/repo."
        )
    return owner, name


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _make_session() -> requests.Session:
    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "RepoJudge/1.0",
    }
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    session.headers.update(headers)
    return session


def _raise_for_status(status: int, url: str, body: str = "") -> None:
    """Translate GitHub HTTP errors into RepoJudge errors."""
    if status == 404:
        raise RepoNotFoundError(
            "Repository not found. Check the URL, or it may be private.",
            code="repo_not_found",
            status_code=404,
        )
    if status in (401, 403):
        lowered = (body or "").lower()
        if "rate limit" in lowered or "rate-limit" in lowered:
            raise GitHubRateLimitError(
                "GitHub API rate limit reached. Add a GITHUB_TOKEN to raise the "
                "limit, then try again."
            )
        raise PrivateRepoError(
            "This repository cannot be accessed. It may be private.",
            code="private_repo",
            status_code=403,
        )
    if status == 429:
        raise GitHubRateLimitError(
            "GitHub is rate limiting requests right now. Please wait and retry."
        )
    if status >= 400:
        raise RepoNotFoundError(
            f"GitHub returned an unexpected error ({status}).",
            code="github_error",
            status_code=502,
        )


def _get(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict] = None,
    accept: Optional[str] = None,
) -> requests.Response:
    headers = {"Accept": accept} if accept else None
    try:
        resp = session.get(url, params=params, headers=headers,
                           timeout=config.GITHUB_TIMEOUT)
    except requests.RequestException as exc:  # network failure
        raise RepoNotFoundError(
            f"Could not reach GitHub: {exc}",
            code="github_unreachable",
            status_code=502,
        ) from exc
    if resp.status_code >= 400:
        _raise_for_status(resp.status_code, url, resp.text[:500])
    return resp


def _get_json(session: requests.Session, url: str, *, params: Optional[dict] = None) -> Any:
    resp = _get(session, url, params=params)
    try:
        return resp.json()
    except ValueError as exc:
        raise RepoNotFoundError(
            "GitHub returned a malformed response.",
            code="github_bad_response",
            status_code=502,
        ) from exc


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------


def _extension(path: str) -> str:
    return os.path.splitext(path.rsplit("/", 1)[-1])[1].lower()


def _is_minified(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".min.js") or lowered.endswith(".min.css")


def _should_skip(path: str) -> bool:
    """Return True for paths we never read (assets, deps, build output)."""
    parts = path.split("/")
    lower_parts = [p.lower() for p in parts]
    if any(part in SKIP_DIRS for part in lower_parts[:-1]):
        return True
    if lower_parts[-1] in SKIP_DIRS:
        return True

    name = lower_parts[-1]
    if name in SKIP_FILE_NAMES:
        return True
    if _is_minified(name):
        return True

    if _extension(path) in SKIP_EXTENSIONS:
        return True
    return False


def _skip_reason(path: str) -> str:
    """A human-readable reason why ``path`` will not be read."""
    parts = path.split("/")
    lower_parts = [p.lower() for p in parts]
    name = lower_parts[-1]
    if any(part in SKIP_DIRS for part in lower_parts[:-1]) or name in SKIP_DIRS:
        return "It lives in a dependency, build or cache folder."
    if name in SKIP_FILE_NAMES:
        return "Lockfiles and generated manifests carry no signal for claim checking."
    if _is_minified(name):
        return "It is a minified build artefact, not readable source."
    if _extension(path) in SKIP_EXTENSIONS:
        return "It is a binary, asset or generated file, so it cannot be read as text."
    return "It is not a readable source file."


def _file_priority(path: str) -> int:
    """Higher score = more likely to contain proof of a README claim."""
    parts = path.split("/")
    lower_parts = [p.lower() for p in parts]
    name = lower_parts[-1]
    ext = os.path.splitext(name)[1]
    depth = len(parts)
    score = 0

    if name in ENTRY_NAMES:
        score += 100
    if name in DEP_FILES:
        score += 90
    if name.startswith("readme") or name in {"license", "licence", "license.md"}:
        score += 80
    if name == ".gitignore":
        score += 60
    if name.startswith("dockerfile") or name.startswith("makefile"):
        score += 70
    if name.startswith(".env"):
        score += 65  # we want to see whether secrets are committed
    if any(p in {"src", "app", "lib", "core", "api", "server", "backend",
                 "pkg", "internal", "cmd", "components", "services"} for p in lower_parts[:-1]):
        score += 40
    if "test" in name or "spec" in name or any(
        p in {"tests", "test", "__tests__", "spec", "specs"} for p in lower_parts[:-1]
    ):
        score += 30
    if ext in SOURCE_EXTENSIONS:
        score += 20
    elif ext in CONFIG_EXTENSIONS:
        score += 10

    score -= depth * 2  # prefer shallow, top-level files
    return score


def _readme_referenced_paths(readme: str, tree_paths: list[str]) -> set[str]:
    """Tree paths that the README names or lists in a project structure block."""
    if not readme:
        return set()
    text = readme.replace("\\", "/")
    tree_set = set(tree_paths)
    by_basename: dict[str, list[str]] = {}
    for path in tree_paths:
        by_basename.setdefault(path.rsplit("/", 1)[-1].lower(), []).append(path)

    found: set[str] = set()
    for token in PATH_TOKEN_RE.findall(text):
        token = token.strip("`'\",;:()[]<>*").lstrip("./")
        if not token:
            continue
        if token in tree_set:
            found.add(token)
            continue
        matches = by_basename.get(token.rsplit("/", 1)[-1].lower(), [])
        if len(matches) == 1:
            found.add(matches[0])
            continue
        tail = token.lower()
        for path in tree_paths:
            if path.lower().endswith("/" + tail):
                found.add(path)
                break
    return found


def plan_files(
    tree: Iterable[dict],
    *,
    readme: Optional[str] = None,
    max_total: Optional[int] = None,
    max_files: Optional[int] = None,
) -> tuple[list[FilePlan], list[dict], int]:
    """Decide which files to read, and how much of each.

    Returns ``(plans, skipped, considered)``. ``skipped`` is a list of
    ``{"path", "reason"}`` so the report can always explain itself.

    Every readable source file is included until the total size budget runs out.
    A file larger than the per-file cap is not skipped: it is read down to the
    cap (its keyword-matching lines are added later).
    """
    cap = max_files or config.MAX_FILES
    budget = max_total or config.MAX_TOTAL_FILE_BYTES

    blobs: list[tuple[str, int]] = []
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path") or ""
        if not path:
            continue
        blobs.append((path, int(entry.get("size") or 0)))

    referenced = _readme_referenced_paths(readme or "", [p for p, _ in blobs])

    candidates: list[tuple[str, int, str, bool]] = []
    skipped: list[dict] = []
    for path, size in blobs:
        name = path.rsplit("/", 1)[-1].lower()
        if _should_skip(path):
            skipped.append({"path": path, "reason": _skip_reason(path)})
            continue
        is_always = (path in referenced or name in ALWAYS_INCLUDE_NAMES
                     or name.startswith("readme"))
        if name in ALWAYS_INCLUDE_NAMES:
            reason = "It is an entry page, which RepoJudge always reads."
        elif name.startswith("readme"):
            reason = "It is the project README, which RepoJudge always reads."
        elif name in DEP_FILES:
            reason = "It is a dependency manifest, which RepoJudge always reads."
        elif path in referenced:
            reason = "The README names it or lists it in its project structure."
        else:
            reason = "Source or configuration file."
        candidates.append((path, size, reason, is_always))

    candidates.sort(key=lambda item: (
        0 if item[3] else 1, -_file_priority(item[0]), item[0]
    ))

    plans: list[FilePlan] = []
    used = 0
    for path, size, reason, _is_always in candidates:
        if len(plans) >= cap:
            skipped.append({
                "path": path,
                "reason": f"RepoJudge reads at most {cap} files per repository.",
            })
            continue
        remaining = budget - used
        if remaining <= 0:
            skipped.append({
                "path": path,
                "reason": "The total size budget was reached before this file.",
            })
            continue
        keep = min(size or remaining, config.MAX_SINGLE_FILE_BYTES, remaining)
        if keep <= 0:
            skipped.append({
                "path": path,
                "reason": "The total size budget was reached before this file.",
            })
            continue
        plans.append(FilePlan(path=path, size=size, keep_bytes=keep,
                              truncated=bool(size) and size > keep,
                              reason=reason))
        used += keep
    return plans, skipped, len(candidates)


def select_important_files(
    tree: Iterable[dict],
    *,
    readme: Optional[str] = None,
    max_total: Optional[int] = None,
    max_files: Optional[int] = None,
) -> tuple[list[str], int, int]:
    """Pick the files to read. Returns ``(paths, considered, skipped_count)``."""
    plans, skipped, considered = plan_files(
        tree, readme=readme, max_total=max_total, max_files=max_files
    )
    return [plan.path for plan in plans], considered, len(skipped)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _decode_readme(payload: dict) -> Optional[str]:
    """Decode a README from the contents API response."""
    content = payload.get("content")
    if content:
        try:
            data = base64.b64decode(content)
            return data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - fall through to download_url
            pass
    download_url = payload.get("download_url")
    if download_url:
        try:
            resp = requests.get(download_url, timeout=config.GITHUB_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
        except requests.RequestException:
            pass
    return None


def _main_language(languages: dict[str, int]) -> str:
    if not languages:
        return "Unknown"
    return max(languages.items(), key=lambda kv: kv[1])[0]


def _commit_count(session: requests.Session, owner: str, name: str) -> int:
    """Total commits, read from the Link header of a 1-per-page request."""
    url = f"{config.GITHUB_API_BASE}/repos/{owner}/{name}/commits"
    resp = _get(session, url, params={"per_page": 1})
    link = resp.headers.get("Link") or ""
    match = re.search(r'[?&]page=(\d+)>;\s*rel="last"', link)
    if match:
        return int(match.group(1))
    # No Link header means a single page: either 0 or 1 commit.
    try:
        return len(resp.json())
    except ValueError:
        return 0


def _fetch_tree(session: requests.Session, owner: str, name: str,
                branch: str) -> tuple[list[str], bool, list[dict]]:
    """Return ``(all_paths, truncated, raw_entries)`` for the default branch."""
    url = f"{config.GITHUB_API_BASE}/repos/{owner}/{name}/git/trees/{branch}"
    data = _get_json(session, url, params={"recursive": "1"})
    entries = data.get("tree", []) or []
    paths = [e.get("path") for e in entries if e.get("path")]
    return paths, bool(data.get("truncated")), entries


def _head_of(text: str, limit: int) -> str:
    """First ``limit`` characters of ``text``, cut back to a whole line."""
    head = text[:limit]
    cut = head.rfind("\n")
    if cut != -1:
        head = head[:cut + 1]
    return head


def _fetch_file_contents(
    session: requests.Session,
    owner: str,
    name: str,
    sha: str,
    plans: list[FilePlan],
) -> tuple[dict[str, RepoFile], list[dict]]:
    """Download the planned part of every file.

    Returns ``(files, read_report)``. A file larger than the per-file cap is read
    only as far as the cap allows; the later lines that match claim keywords are
    added afterwards by :func:`add_keyword_lines`.
    """
    files: dict[str, RepoFile] = {}
    read: list[dict] = []

    for plan in plans:
        raw_url = f"{config.GITHUB_RAW_BASE}/{owner}/{name}/{sha}/{plan.path}"
        # Ask for one byte more than we intend to keep. If we get it back, the
        # file is bigger than the cap and was cut down. (A range request can
        # legitimately answer 206 for a small file, so the status code alone
        # does not tell us whether anything was left out.)
        headers = {"Range": f"bytes=0-{max(plan.keep_bytes, 0)}"}
        entry = {"path": plan.path, "size": plan.size, "lines": 0,
                 "truncated": False, "reason": plan.reason}
        try:
            resp = session.get(raw_url, headers=headers,
                               timeout=config.GITHUB_TIMEOUT)
        except requests.RequestException as exc:
            entry["reason"] = f"Could not be downloaded ({type(exc).__name__})."
            read.append(entry)
            continue
        if resp.status_code not in (200, 206):
            entry["reason"] = f"GitHub returned HTTP {resp.status_code} for it."
            read.append(entry)
            continue

        content = resp.content or b""
        if b"\x00" in content[:1024]:
            entry["reason"] = "It looks like a binary file, not text."
            read.append(entry)
            continue

        text = content.decode("utf-8", errors="replace")
        truncated = len(content) > plan.keep_bytes
        if truncated:
            text = _head_of(text, plan.keep_bytes)

        repo_file = RepoFile(
            path=plan.path,
            content=text,
            size=plan.size or len(content),
            lines=text.count("\n") + 1,
            truncated=truncated,
            head_bytes=len(text.encode("utf-8")),
            reason=plan.reason,
        )
        files[plan.path] = repo_file
        entry["size"] = repo_file.size
        entry["lines"] = repo_file.lines
        entry["truncated"] = truncated
        if truncated:
            entry["reason"] = (
                plan.reason + " The file is large, so only the first part is "
                "kept here (plus any lines matching a claim keyword)."
            )
        read.append(entry)

    return files, read


def add_keyword_lines(
    repo: RepoData,
    keywords: list[str],
    *,
    session: Optional[requests.Session] = None,
) -> int:
    """Keep the later lines of a truncated big file that match a claim keyword.

    Only the files that were actually cut down are revisited, so small
    repositories never make an extra request. Returns the number of lines kept.
    """
    cut_down = {p: f for p, f in repo.files.items() if f.truncated}
    if not cut_down or not keywords:
        return 0

    terms: list[list[str]] = []
    for keyword in keywords:
        matched = keywords_module.match_terms(str(keyword))
        if matched:
            terms.append(matched)
    if not terms:
        return 0

    sess = session or _make_session()
    kept = 0
    for path, repo_file in cut_down.items():
        start = repo_file.head_bytes
        raw_url = (
            f"{config.GITHUB_RAW_BASE}/{repo.owner}/{repo.name}/"
            f"{repo.last_commit_sha or ''}/{path}"
        )
        try:
            resp = sess.get(raw_url, headers={"Range": f"bytes={start}-"},
                            timeout=config.GITHUB_TIMEOUT)
        except requests.RequestException:
            continue
        if resp.status_code not in (200, 206):
            continue
        body = resp.content or b""
        if resp.status_code == 200:
            body = body[start:]
        tail = body.decode("utf-8", errors="replace")

        first_line = repo_file.content.count("\n") + 1
        for offset, line in enumerate(tail.splitlines()):
            if len(repo_file.extra_lines) >= config.MAX_KEYWORD_LINES_PER_FILE:
                break
            snippet = line.strip()
            if not snippet:
                continue
            if any(keywords_module.score_line(snippet, group) for group in terms):
                repo_file.extra_lines[first_line + offset] = snippet[:200]
                kept += 1
    return kept


def fetch_repo(
    repo_url: str,
    *,
    include_files: Optional[bool] = None,
    session: Optional[requests.Session] = None,
) -> RepoData:
    """Fetch everything RepoJudge needs about a public GitHub repo.

    ``include_files=None`` auto-detects whether the main language supports a
    full code analysis. Pass ``True``/``False`` to force it.
    """
    owner, name = parse_repo_url(repo_url)
    sess = session or _make_session()

    meta = _get_json(sess, f"{config.GITHUB_API_BASE}/repos/{owner}/{name}")

    if meta.get("private") or meta.get("visibility") == "private":
        raise PrivateRepoError(
            f"{owner}/{name} is a private repository. Only public repositories "
            "can be judged."
        )

    default_branch = meta.get("default_branch") or "main"

    # README (optional at the API level; RepoJudge requires one).
    readme: Optional[str] = None
    readme_path: Optional[str] = None
    try:
        readme_payload = _get_json(
            sess, f"{config.GITHUB_API_BASE}/repos/{owner}/{name}/readme"
        )
        readme = _decode_readme(readme_payload)
        readme_path = readme_payload.get("path") or "README.md"
    except RepoNotFoundError:
        readme = None
        readme_path = None

    if not readme or not readme.strip():
        raise NoReadmeError(
            f"{owner}/{name} has no README, so there is nothing to check. "
            "Add a README that describes the project and try again."
        )

    languages_raw = _get_json(
        sess, f"{config.GITHUB_API_BASE}/repos/{owner}/{name}/languages"
    ) or {}
    languages = {k: int(v) for k, v in languages_raw.items()}
    language = _main_language(languages) if languages else (meta.get("language") or "Unknown")

    tree_paths: list[str] = []
    tree_truncated = False
    tree_entries: list[dict] = []
    try:
        tree_paths, tree_truncated, tree_entries = _fetch_tree(
            sess, owner, name, default_branch
        )
    except RepoNotFoundError:
        tree_paths, tree_entries = [], []

    # Commit history.
    commit_count = 0
    last_commit_date: Optional[str] = None
    last_commit_sha: Optional[str] = None
    try:
        commits = _get_json(
            sess,
            f"{config.GITHUB_API_BASE}/repos/{owner}/{name}/commits",
            params={"per_page": 1},
        )
        if commits:
            last_commit_sha = commits[0].get("sha")
            last_commit_date = (
                commits[0].get("commit", {}).get("committer", {}).get("date")
            )
        commit_count = _commit_count(sess, owner, name)
    except RepoNotFoundError:
        pass

    license_meta = meta.get("license") or {}

    warnings: list[str] = []

    # Decide analysis type from the main language.
    if include_files is None:
        include_files = language in config.FULL_ANALYSIS_LANGUAGES
    analysis_type = "full" if include_files else "limited"

    files: dict[str, RepoFile] = {}
    considered = 0
    skipped_entries: list[dict] = []
    read_entries: list[dict] = []

    if include_files:
        if tree_entries:
            plans, skipped_entries, considered = plan_files(
                tree_entries, readme=readme
            )
            if last_commit_sha:
                files, read_entries = _fetch_file_contents(
                    sess, owner, name, last_commit_sha, plans
                )
            skipped = len(skipped_entries)
            if skipped:
                warnings.append(
                    f"{skipped} file(s) were skipped. The report lists every "
                    "one of them and the reason."
                )
            truncated = [e for e in read_entries if e.get("truncated")]
            if truncated:
                warnings.append(
                    f"{len(truncated)} large file(s) were read only in part "
                    "(the size budget is "
                    f"{config.MAX_TOTAL_FILE_BYTES // 1024} KB in total)."
                )
        else:
            warnings.append("No file tree could be read for this repository.")

        if tree_truncated:
            warnings.append(
                "The repository file tree was truncated by GitHub, so very large "
                "repositories may be only partially analysed."
            )
    else:
        warnings.append(
            f"Limited analysis: code checks are not available for {language}."
        )

    skipped = len(skipped_entries)

    return RepoData(
        url=f"https://github.com/{owner}/{name}",
        owner=owner,
        name=name,
        default_branch=default_branch,
        private=False,
        description=meta.get("description") or "",
        homepage=meta.get("homepage") or "",
        stars=int(meta.get("stargazers_count") or 0),
        forks=int(meta.get("forks_count") or 0),
        open_issues=int(meta.get("open_issues_count") or 0),
        language=language,
        languages=languages,
        license_name=license_meta.get("name"),
        license_spdx=license_meta.get("spdx_id"),
        readme=readme,
        readme_path=readme_path,
        tree=tree_paths,
        tree_truncated=tree_truncated,
        files=files,
        files_considered=considered,
        files_skipped=skipped,
        commit_count=commit_count,
        last_commit_date=last_commit_date,
        last_commit_sha=last_commit_sha,
        created_at=meta.get("created_at"),
        pushed_at=meta.get("pushed_at"),
        analysis_type=analysis_type,
        warnings=warnings,
        files_report={"read": read_entries, "skipped": skipped_entries},
    )
