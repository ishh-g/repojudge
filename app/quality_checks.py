"""Milestone 4: rule-based engineering quality checks (no LLM).

Each check returns a plain dict with a status and, wherever possible, evidence
pointing at a file and line. The statuses are:

``pass``            the check found what it was looking for;
``fail``            the check found a real problem;
``not_checked``     RepoJudge could not look (for example a LIMITED analysis, or
                    a bonus check such as CI that is missing);
``not_applicable``  the check does not apply to this repository (for example a
                    plain HTML/JS site with no external packages).

Only ``pass`` and ``fail`` count towards the score. ``not_checked`` and
``not_applicable`` are removed from the total, so they never lower a score.
"""
from __future__ import annotations

import re
import sys
from typing import Optional

from . import config
from .github_client import SOURCE_EXTENSIONS, RepoData

# Which checks need the file *contents* (and are therefore skipped in LIMITED mode).
CODE_BASED_CHECKS = {"secrets", "structure", "error_handling"}

# Points each check contributes to the 30-point engineering score.
CHECK_WEIGHTS: dict[str, int] = {
    "tests": 5,
    "setup": 5,
    "dependencies": 4,
    "secrets": 4,
    "gitignore": 2,
    "structure": 3,
    "documentation": 3,
    "error_handling": 2,
    "ci": 2,
}

DEPENDENCY_FILES = {
    "requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.py",
    "setup.cfg", "pipfile", "environment.yml", "package.json", "go.mod",
    "cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts", "composer.json",
    "gemfile", "pubspec.yaml", "mix.exs",
}

TEST_DIR_NAMES = {"tests", "test", "__tests__", "spec", "specs"}
TEST_FILE_RE = re.compile(
    r"(^|/)(test_.*\.(py|js|ts|jsx|tsx)|.*_test\.(py|js|ts|jsx|tsx)|"
    r".*\.(test|spec)\.(js|ts|jsx|tsx))$",
    re.IGNORECASE,
)
TEST_FRAMEWORKS = ("pytest", "unittest", "jest", "mocha", "vitest", "jasmine",
                   "nose", "tox", "testing-library", "playwright", "cypress")

INSTALL_HINTS = (
    "pip install", "pip3 install", "npm install", "npm i ", "yarn add",
    "yarn install", "pnpm install", "poetry install", "pipenv install",
    "go get", "go install", "cargo build", "cargo install", "composer install",
    "bundle install", "make install", "docker build", "docker compose up",
    "conda install", "python -m pip install",
)
RUN_HINTS = (
    "npm start", "npm run", "yarn start", "yarn run", "pnpm start", "pnpm run",
    "uvicorn", "gunicorn", "flask run", "python main.py", "python app.py",
    "python -m", "node ", "go run", "cargo run", "docker run", "make run",
    "streamlit run", "manage.py runserver",
)
INSTALL_HEADINGS = ("install", "installation", "setup", "getting started",
                    "quick start", "quickstart")
RUN_HEADINGS = ("usage", "run", "running", "quick start", "quickstart",
                "example", "how to use", "getting started")
# Any of these headings may hold usable setup instructions.
SETUP_HEADINGS = INSTALL_HEADINGS + RUN_HEADINGS + ("set up", "how to run",
                                                    "build", "development")
# Lines that look like a command the reader is meant to type.
COMMAND_START_RE = re.compile(
    r"(?i)^(?:sudo\s+)?(?:python\d?|py|pip\d?|npm|npx|yarn|pnpm|node|deno|bun|"
    r"go|cargo|rustc|docker|docker-compose|make|cmake|git|bash|sh|zsh|uvicorn|"
    r"gunicorn|flask|streamlit|django-admin|manage\.py|composer|bundle|gem|php|"
    r"ruby|rails|mvn|gradlew|gradle|dotnet|conda|poetry|pipenv|uv|rake|swift|"
    r"kotlinc|javac|java|psql|sqlite3|cd|export|source|chmod|\./)\b"
)

# Hard-coded secrets: key-ish names assigned a long literal value.
ASSIGN_RE = re.compile(
    r"""(?ix)
    (?P<key>
        api[_-]?key | apikey | secret[_-]?key | client[_-]?secret | secret |
        access[_-]?key | auth[_-]?token | access[_-]?token | token |
        password | passwd | pwd | private[_-]?key
    )
    \s*[:=]\s*
    (?P<quote>['"])(?P<value>[^'"]{8,})\2
    """
)
# Well-known token shapes that are secrets wherever they appear.
KNOWN_SECRET_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),          # OpenAI / Anthropic style
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),            # GitHub personal token
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),         # Google API key
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),              # AWS access key id
    re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}"),   # Slack token
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
# Values that are clearly placeholders, not real secrets.
PLACEHOLDER_RE = re.compile(
    r"(?i)^(your[_-]?|my[_-]?|<|\.\.\.|\$\{|%\(|placeholder|changeme|change[_-]?me|"
    r"example|dummy|fake|test|xxx|todo|none|null|secret|password|token|"
    r"os\.environ|process\.env|getenv|settings\.|config\.|env\.)"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _line_of(text: str, needle: str) -> Optional[int]:
    """1-based line number of the first line containing ``needle``."""
    lowered = needle.lower()
    for number, line in enumerate(text.splitlines(), start=1):
        if lowered in line.lower():
            return number
    return None


def _mask(value: str) -> str:
    """Show only the first 4 characters of a secret value."""
    value = value.strip()
    if len(value) <= 4:
        return "****"
    return value[:4] + "*" * min(len(value) - 4, 8)


def _result(check_id: str, name: str, status: str, reason: str,
            evidence: Optional[list] = None, *, bonus: bool = False) -> dict:
    return {
        "id": check_id,
        "name": name,
        "status": status,
        "weight": CHECK_WEIGHTS.get(check_id, 0),
        "reason": reason,
        "evidence": evidence or [],
        "bonus": bonus,
    }


def _source_files(repo: RepoData) -> list[tuple[str, int]]:
    """The read files that look like source code, with their line counts."""
    out = []
    for path, repo_file in repo.files.items():
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in SOURCE_EXTENSIONS:
            out.append((path, repo_file.lines))
    return out


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def check_tests(repo: RepoData) -> dict:
    for path in repo.tree:
        parts = path.lower().split("/")
        if any(p in TEST_DIR_NAMES for p in parts[:-1]):
            return _result("tests", "Tests", "pass",
                           f"Test directory found ({path}).",
                           [{"file": path, "line": None, "snippet": ""}])
        if TEST_FILE_RE.search(path):
            return _result("tests", "Tests", "pass",
                           f"Test file found ({path}).",
                           [{"file": path, "line": None, "snippet": ""}])
    # Also accept a declared test framework in a dependency file.
    for path, repo_file in repo.files.items():
        name = path.rsplit("/", 1)[-1].lower()
        if name in DEPENDENCY_FILES:
            for framework in TEST_FRAMEWORKS:
                line = _line_of(repo_file.content, framework)
                if line:
                    return _result(
                        "tests", "Tests", "pass",
                        f"A test framework ({framework}) is declared.",
                        [{"file": path, "line": line, "snippet": framework}],
                    )
    return _result("tests", "Tests", "fail",
                   "No tests folder, test file or test framework was found.")


def _readme_sections(readme: str):
    """Yield ``(heading, heading_line_number, body_lines)`` for each heading.

    A section runs from its heading until the next heading of the same or a
    higher level, so a "How to run" heading keeps the instructions that live in
    its ``###`` subsections.
    """
    lines = readme.splitlines()
    headings: list[tuple[int, int, str]] = []
    for number, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            headings.append((number, len(match.group(1)), match.group(2).strip()))

    for index, (number, level, text) in enumerate(headings):
        end = len(lines) + 1
        for next_number, next_level, _next_text in headings[index + 1:]:
            if next_level <= level:
                end = next_number
                break
        body = [(n, lines[n - 1]) for n in range(number + 1, end)]
        yield text, number, body


def _section_command(body: list[tuple[int, str]]) -> Optional[tuple[int, str]]:
    """First line in a section that looks like a command to type."""
    in_fence = False
    for number, line in body:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not stripped:
            continue
        if in_fence:
            return number, stripped
        lowered = stripped.lower()
        if any(hint in lowered for hint in INSTALL_HINTS + RUN_HINTS):
            return number, stripped
        if stripped.startswith(("$", ">", "PS>", "C:\\")):
            return number, stripped
        if COMMAND_START_RE.match(stripped):
            return number, stripped
    return None


def check_setup(repo: RepoData) -> dict:
    """Pass when the README has a setup section (install/usage/...) with commands."""
    readme = repo.readme or ""
    readme_name = repo.readme_path or "README.md"
    lowered = readme.lower()

    # Best case: a heading such as "Install", "Usage" or "Getting started" that
    # actually contains a command the reader can run.
    for heading, heading_line, body in _readme_sections(readme):
        if not any(word in heading.lower() for word in SETUP_HEADINGS):
            continue
        command = _section_command(body)
        if command:
            command_line, command_text = command
            return _result(
                "setup", "Setup instructions", "pass",
                f'The README has a "{heading}" section with commands.',
                [
                    {"file": readme_name, "line": heading_line,
                     "snippet": f"## {heading}"},
                    {"file": readme_name, "line": command_line,
                     "snippet": command_text[:160]},
                ],
            )

    # Fall back to the older rule: install hints and run hints anywhere.
    install_hit = next((h for h in INSTALL_HINTS if h in lowered), None)
    run_hit = next((h for h in RUN_HINTS if h in lowered), None)
    if install_hit and run_hit:
        evidence = []
        for hint in (install_hit, run_hit):
            evidence.append({"file": readme_name,
                             "line": _line_of(readme, hint),
                             "snippet": hint})
        return _result("setup", "Setup instructions", "pass",
                       "The README explains how to install and how to run the "
                       "project.", evidence)

    missing = []
    if not install_hit:
        missing.append("how to install")
    if not run_hit:
        missing.append("how to run")
    return _result("setup", "Setup instructions", "fail",
                   "The README is missing " + " and ".join(missing) + ".")


def check_dependencies(repo: RepoData) -> dict:
    for path in repo.tree:
        if path.rsplit("/", 1)[-1].lower() in DEPENDENCY_FILES:
            return _result("dependencies", "Dependency file", "pass",
                           f"A dependency file is present ({path}).",
                           [{"file": path, "line": None, "snippet": ""}])

    if repo.files and not _has_external_package_imports(repo):
        return _result(
            "dependencies", "Dependency file", "not_applicable",
            "Not applicable: the code does not import any external packages, so "
            "no dependency file is needed. A plain HTML/JS site that loads its "
            "libraries from a CDN works this way.",
        )

    if not repo.files:
        return _result("dependencies", "Dependency file", "fail",
                       "No dependency file (requirements.txt, package.json, "
                       "pyproject.toml, ...) was found.")
    return _result("dependencies", "Dependency file", "fail",
                   "No dependency file was found even though the code imports "
                   "external packages.")


# ---------------------------------------------------------------------------
# external-package detection (used by the dependency check)
# ---------------------------------------------------------------------------

_JS_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
                            ".vue", ".svelte"})
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ()))
_PY_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+([A-Za-z_][\w]*)|from\s+([A-Za-z_][\w]*))"
)
_JS_FROM_RE = re.compile(r"""\bfrom\s*['"](?P<spec>[^'"]+)['"]""")
_JS_REQUIRE_RE = re.compile(r"""\brequire\s*\(\s*['"](?P<spec>[^'"]+)['"]\s*\)""")
_JS_BARE_IMPORT_RE = re.compile(r"""\bimport\s+['"](?P<spec>[^'"]+)['"]""")


def _is_bare_specifier(spec: str) -> bool:
    """True for an npm-style package name, False for a relative/builtin path."""
    if not spec:
        return False
    if spec.startswith((".", "/", "\\", "#")):
        return False
    if spec.startswith(("http://", "https://", "//", "data:", "node:", "bun:")):
        return False
    return "${" not in spec


def _has_external_package_imports(repo: RepoData) -> bool:
    """True when any file we read imports a package from outside the repo."""
    local_modules = set()
    for path in repo.tree:
        name = path.rsplit("/", 1)[-1]
        stem = name.rsplit(".", 1)[0] if "." in name else name
        local_modules.add(stem.lower())

    for path, repo_file in repo.files.items():
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext == ".py":
            for line in repo_file.content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                match = _PY_IMPORT_RE.match(stripped)
                if not match:
                    continue
                module = (match.group(1) or match.group(2) or "").lower()
                if not module or module in _STDLIB or module in local_modules:
                    continue
                return True
        elif ext in _JS_EXTENSIONS:
            for pattern in (_JS_FROM_RE, _JS_REQUIRE_RE, _JS_BARE_IMPORT_RE):
                for match in pattern.finditer(repo_file.content):
                    if _is_bare_specifier(match.group("spec").strip()):
                        return True
    return False


def check_secrets(repo: RepoData) -> dict:
    """Look only for hard-coded secrets in the files RepoJudge read."""
    findings: list[dict] = []

    for path, repo_file in repo.files.items():
        # A committed .env file (but not .env.example) is a red flag.
        name = path.rsplit("/", 1)[-1].lower()
        if name == ".env":
            findings.append({
                "file": path, "line": 1,
                "snippet": "A .env file is committed to the repository.",
            })

        for number, line in enumerate(repo_file.content.splitlines(), start=1):
            if _looks_like_comment(line):
                continue
            masked = _scan_line_for_secret(line)
            if masked:
                findings.append({"file": path, "line": number, "snippet": masked})

    if findings:
        return _result(
            "secrets", "No hard-coded secrets", "fail",
            f"{len(findings)} possible hard-coded secret(s) were found in the "
            "files RepoJudge read.",
            findings[:5],
        )

    return _result("secrets", "No hard-coded secrets", "pass",
                   "No hard-coded secrets were found in the files RepoJudge read.")


def check_gitignore(repo: RepoData) -> dict:
    """Report the presence of a .gitignore as its own check."""
    path = next((p for p in repo.tree if p.rsplit("/", 1)[-1] == ".gitignore"), None)
    if not path and ".gitignore" in repo.files:
        path = ".gitignore"
    if path:
        return _result("gitignore", "Has .gitignore", "pass",
                       "A .gitignore file is present, which helps keep secrets "
                       "and build output out of the repository.",
                       [{"file": path, "line": None,
                         "snippet": ".gitignore present"}])
    return _result("gitignore", "Has .gitignore", "fail",
                   "No .gitignore was found, so secrets or build output may be "
                   "committed by accident.",
                   [{"file": None, "line": None,
                     "snippet": "No .gitignore in the repository."}])


def _looks_like_comment(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith(("#", "//", "*", "/*", "--"))


def _scan_line_for_secret(line: str) -> Optional[str]:
    """Return a masked description of a secret found on this line, else None."""
    match = ASSIGN_RE.search(line)
    if match:
        value = match.group("value")
        if not PLACEHOLDER_RE.match(value) and not _is_variable_reference(value):
            masked = line.replace(value, _mask(value))
            return masked.strip()[:200]

    for pattern in KNOWN_SECRET_RES:
        found = pattern.search(line)
        if found and not PLACEHOLDER_RE.match(found.group(0)):
            masked = line.replace(found.group(0), _mask(found.group(0)))
            return masked.strip()[:200]
    return None


def _is_variable_reference(value: str) -> bool:
    """True for values like ``os.environ['X']`` or ``{{ secret }}`` - not secrets."""
    lowered = value.lower()
    return any(token in lowered for token in (
        "environ", "process.env", "getenv", "{{", "}}", "%(", "settings.",
        "config.", "env.", "import.meta.env", "${",
    ))


def check_structure(repo: RepoData) -> dict:
    sources = _source_files(repo)
    if not sources:
        return _result("structure", "Code structure", "fail",
                       "No source code files were read, so the structure could "
                       "not be assessed.")

    giant = [(p, n) for p, n in sources if n > config.MAX_FILE_LINES]
    if giant:
        path, lines = max(giant, key=lambda item: item[1])
        evidence = [{"file": p, "line": None, "snippet": f"{n} lines"} for p, n in giant]
        return _result("structure", "Code structure", "fail",
                       f"{path} has {lines} lines. Very large single files are hard "
                       "to read and maintain.", evidence)

    in_folder = any("/" in path for path, _ in sources)
    if in_folder or len(sources) >= 3:
        return _result("structure", "Code structure", "pass",
                       f"Code is organised across {len(sources)} file(s), none "
                       f"above {config.MAX_FILE_LINES} lines.",
                       [{"file": p, "line": None, "snippet": f"{n} lines"}
                        for p, n in sources[:3]])

    return _result("structure", "Code structure", "fail",
                   "All code sits in a single flat file. Consider splitting it "
                   "into modules inside folders.",
                   [{"file": sources[0][0], "line": None,
                     "snippet": f"{sources[0][1]} lines"}])


def check_documentation(repo: RepoData) -> dict:
    readme = (repo.readme or "").strip()
    if not readme:
        return _result("documentation", "Documentation", "fail",
                       "There is no README content.")
    # Ignore the title line(s) and look for real prose.
    prose_line = None
    for number, line in enumerate(readme.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("="):
            continue
        if stripped.startswith(("!", "[", "<", "|", "-", "*")) and len(stripped) < 40:
            continue
        if len(stripped.split()) >= 8:
            prose_line = (number, stripped[:160])
            break

    if len(readme) > 150 and prose_line:
        return _result("documentation", "Documentation", "pass",
                       "The README explains what the project does.",
                       [{"file": repo.readme_path or "README.md",
                         "line": prose_line[0], "snippet": prose_line[1]}])
    return _result("documentation", "Documentation", "fail",
                   "The README has little more than a title; it does not explain "
                   "what the project does.",
                   [{"file": repo.readme_path or "README.md", "line": 1,
                     "snippet": readme.splitlines()[0][:120] if readme else ""}])


def check_error_handling(repo: RepoData) -> dict:
    pattern = re.compile(
        r"(\btry\s*:)|(\btry\s*\{)|(\bexcept\b)|(\bcatch\s*\()|(\brescue\b)",
    )
    for path, repo_file in repo.files.items():
        for number, line in enumerate(repo_file.content.splitlines(), start=1):
            if _looks_like_comment(line):
                continue
            if pattern.search(line):
                return _result("error_handling", "Error handling", "pass",
                               "Error handling was found in the code.",
                               [{"file": path, "line": number,
                                 "snippet": line.strip()[:160]}])
    return _result("error_handling", "Error handling", "fail",
                   "No try/except or try/catch blocks were found in the files "
                   "RepoJudge read.")


def check_ci(repo: RepoData) -> dict:
    for path in repo.tree:
        lowered = path.lower()
        if lowered.startswith(".github/workflows/") and lowered.endswith(
            (".yml", ".yaml")
        ):
            return _result("ci", "Continuous integration", "pass",
                           f"A CI workflow is configured ({path}).",
                           [{"file": path, "line": None, "snippet": ""}],
                           bonus=True)
        if lowered in {".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml",
                       ".circleci/config.yml"}:
            return _result("ci", "Continuous integration", "pass",
                           f"A CI configuration is present ({path}).",
                           [{"file": path, "line": None, "snippet": ""}],
                           bonus=True)
    return _result("ci", "Continuous integration", "not_checked",
                   "Not checked (bonus): no CI configuration was found. "
                   "Continuous integration is a bonus, so it never lowers the "
                   "score.",
                   bonus=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

CHECK_ORDER = ["tests", "setup", "dependencies", "secrets", "gitignore",
               "structure", "documentation", "error_handling", "ci"]


def run_quality_checks(repo: RepoData) -> list[dict]:
    """Run every engineering check and return them in a stable order."""
    results = [
        check_tests(repo),
        check_setup(repo),
        check_dependencies(repo),
        check_secrets(repo),
        check_gitignore(repo),
        check_structure(repo),
        check_documentation(repo),
        check_error_handling(repo),
        check_ci(repo),
    ]

    if repo.analysis_type == "limited":
        for item in results:
            if item["id"] in CODE_BASED_CHECKS:
                item["status"] = "not_checked"
                item["reason"] = (
                    "Not checked: this needs a code analysis, which is not "
                    f"available for {repo.language}."
                )
                item["evidence"] = []

    return results
