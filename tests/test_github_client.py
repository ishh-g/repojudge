"""Milestone 1 tests: the GitHub client and its file-selection rules."""
from __future__ import annotations

import pytest

from app import config, github_client
from app.errors import (
    InvalidRepoUrlError,
    NoReadmeError,
    PrivateRepoError,
    RepoNotFoundError,
)
from app.github_client import fetch_repo, parse_repo_url, select_important_files
from fakes import FakeGitHub

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/psf/requests", ("psf", "requests")),
    ("https://github.com/psf/requests/", ("psf", "requests")),
    ("https://github.com/psf/requests.git", ("psf", "requests")),
    ("http://www.github.com/psf/requests", ("psf", "requests")),
    ("github.com/psf/requests", ("psf", "requests")),
])
def test_parse_repo_url_ok(url, expected):
    assert parse_repo_url(url) == expected


@pytest.mark.parametrize("url", [
    "",
    "   ",
    "https://gitlab.com/psf/requests",
    "https://github.com/psf",
    "https://github.com/",
    "not a url at all",
    "https://github.com/psf/requests/issues/1",
])
def test_parse_repo_url_rejects(url):
    with pytest.raises(InvalidRepoUrlError):
        parse_repo_url(url)


# ---------------------------------------------------------------------------
# Full fetch
# ---------------------------------------------------------------------------

SAMPLE_FILES = {
    "README.md": "# Widget\n",
    "main.py": "import requests\n\n\ndef main():\n    print('hi')\n",
    "src/core.py": "def add(a, b):\n    return a + b\n",
    "requirements.txt": "requests==2.31.0\n",
    "tests/test_core.py": "def test_add():\n    assert True\n",
    "assets/logo.png": "\x00binary",
    "package-lock.json": "{}",
    "node_modules/left-pad/index.js": "module.exports = 1;",
    "big.py": "x = 1\n" * 20000,  # larger than the per-file cap
}


def test_fetch_repo_full_analysis():
    fake = FakeGitHub(files=SAMPLE_FILES)
    data = fetch_repo("https://github.com/acme/widget", session=fake)

    assert data.analysis_type == "full"
    assert data.owner == "acme" and data.name == "widget"
    assert data.language == "Python"
    assert data.license_spdx == "MIT"
    assert data.license_name == "MIT License"
    assert data.commit_count == 7
    assert data.last_commit_sha == "deadbeefcafe"
    assert data.last_commit_date == "2026-09-01T10:00:00Z"
    assert data.readme.startswith("# Widget")
    assert data.default_branch == "main"

    # Every real source file should be read.
    assert "main.py" in data.files
    assert "requirements.txt" in data.files
    assert "src/core.py" in data.files
    assert "tests/test_core.py" in data.files

    # Junk files must never be fetched.
    assert "assets/logo.png" not in data.files
    assert "package-lock.json" not in data.files
    assert "node_modules/left-pad/index.js" not in data.files

    # A big file is no longer skipped: it is read down to the per-file cap.
    assert "big.py" in data.files
    assert data.files["big.py"].truncated is True
    assert data.files["big.py"].content.startswith("x = 1")

    # Line counts are computed for quality checks.
    assert data.files["src/core.py"].lines == 3

    # The report always explains what was read and what was skipped.
    read_paths = {entry["path"] for entry in data.files_report["read"]}
    assert {"main.py", "src/core.py"} <= read_paths
    skipped_paths = {entry["path"]: entry["reason"]
                     for entry in data.files_report["skipped"]}
    assert "node_modules/left-pad/index.js" in skipped_paths
    assert "package-lock.json" in skipped_paths
    assert "assets/logo.png" in skipped_paths
    assert all(reason for reason in skipped_paths.values())


def test_fetch_repo_limited_for_other_languages():
    fake = FakeGitHub(
        name="rusty",
        files={"src/main.rs": "fn main() {}\n"},
        languages={"Rust": 9000},
    )
    data = fetch_repo("https://github.com/acme/rusty", session=fake)

    assert data.analysis_type == "limited"
    assert data.language == "Rust"
    # Limited analysis must not read code files.
    assert data.files == {}
    assert any("Limited analysis" in w for w in data.warnings)


def test_fetch_repo_javascript_is_full():
    fake = FakeGitHub(name="js", files={"index.js": "console.log('x')\n"},
                      languages={"JavaScript": 1000})
    data = fetch_repo("https://github.com/acme/js", session=fake)
    assert data.analysis_type == "full"
    assert "index.js" in data.files


def test_fetch_repo_typescript_is_full():
    fake = FakeGitHub(name="ts", files={"src/index.ts": "export const x = 1\n"},
                      languages={"TypeScript": 1000})
    data = fetch_repo("https://github.com/acme/ts", session=fake)
    assert data.analysis_type == "full"


def test_private_repo_raises():
    fake = FakeGitHub(private=True)
    with pytest.raises(PrivateRepoError):
        fetch_repo("https://github.com/acme/widget", session=fake)


def test_missing_repo_raises():
    fake = FakeGitHub(repo_status=404)
    with pytest.raises(RepoNotFoundError):
        fetch_repo("https://github.com/acme/widget", session=fake)


def test_no_readme_raises():
    fake = FakeGitHub(readme=None)
    with pytest.raises(NoReadmeError):
        fetch_repo("https://github.com/acme/widget", session=fake)


def test_empty_readme_raises():
    fake = FakeGitHub(readme="   \n  ")
    with pytest.raises(NoReadmeError):
        fetch_repo("https://github.com/acme/widget", session=fake)


def test_truncated_tree_adds_warning():
    fake = FakeGitHub(files=SAMPLE_FILES, truncate_tree=True)
    data = fetch_repo("https://github.com/acme/widget", session=fake)
    assert any("truncated" in w.lower() for w in data.warnings)


def test_warning_when_files_skipped():
    # Each file is modest, but together they exceed the total size budget.
    chunk = "x = 1\n" * 5000  # about 30 KB per file
    files = {f"src/mod_{i}.py": chunk for i in range(20)}
    fake = FakeGitHub(files=files)
    data = fetch_repo("https://github.com/acme/widget", session=fake)
    assert 0 < len(data.files) < 20
    assert data.files_skipped > 0
    assert any("skipped" in w.lower() for w in data.warnings)
    assert data.files_report["skipped"]
    assert all(entry.get("reason") for entry in data.files_report["skipped"])


def test_index_html_is_always_read():
    files = {f"src/mod_{i}.py": "x = 1\n" for i in range(250)}
    files["nested/deep/index.html"] = "<!doctype html><html></html>\n"
    fake = FakeGitHub(files=files, languages={"JavaScript": 100})
    data = fetch_repo("https://github.com/acme/widget", session=fake)
    assert "nested/deep/index.html" in data.files


def test_readme_referenced_files_are_always_read():
    files = {f"src/mod_{i}.py": "x = 1\n" for i in range(250)}
    files["tools/helper.py"] = "def helper():\n    return 1\n"
    readme = (
        "# Widget\n\n"
        "The project structure:\n\n"
        "```\n"
        "src/\n"
        "tools/helper.py\n"
        "```\n"
    )
    fake = FakeGitHub(files=files, readme=readme)
    data = fetch_repo("https://github.com/acme/widget", session=fake)
    assert "tools/helper.py" in data.files
    reason = next(entry["reason"] for entry in data.files_report["read"]
                  if entry["path"] == "tools/helper.py")
    assert "README" in reason


# ---------------------------------------------------------------------------
# File selection rules (unit level)
# ---------------------------------------------------------------------------


def test_select_important_files_priority():
    entries = [
        {"path": "docs/guide.md", "type": "blob", "size": 100},
        {"path": "src/service/auth.py", "type": "blob", "size": 100},
        {"path": "main.py", "type": "blob", "size": 100},
        {"path": "requirements.txt", "type": "blob", "size": 100},
        {"path": "nested/deep/deeper/file.py", "type": "blob", "size": 100},
    ]
    selected, considered, skipped = select_important_files(entries)
    assert considered == 5
    assert skipped == 0
    # Entry points and dependency files rank above nested files.
    assert selected[0] in {"main.py", "requirements.txt"}
    assert selected.index("main.py") < selected.index("nested/deep/deeper/file.py")


def test_select_important_files_skips_generated_and_binary():
    entries = [
        {"path": "dist/bundle.js", "type": "blob", "size": 10},
        {"path": "node_modules/a/b.js", "type": "blob", "size": 10},
        {"path": "yarn.lock", "type": "blob", "size": 10},
        {"path": "logo.svg", "type": "blob", "size": 10},
        {"path": "huge.py", "type": "blob", "size": 10},
        {"path": "keep.py", "type": "blob", "size": 10},
    ]
    selected, considered, skipped = select_important_files(entries)
    assert set(selected) == {"huge.py", "keep.py"}
    assert considered == 2
    assert skipped == 4


def test_big_file_is_truncated_not_skipped():
    entries = [{"path": "huge.py", "type": "blob",
                "size": config.MAX_SINGLE_FILE_BYTES * 3}]
    plans, skipped, considered = github_client.plan_files(entries)
    assert considered == 1
    assert skipped == []
    assert plans[0].truncated is True
    assert plans[0].keep_bytes == config.MAX_SINGLE_FILE_BYTES


def test_total_budget_skips_files_with_a_reason():
    entries = [{"path": f"mod_{i}.py", "type": "blob", "size": 100_000}
               for i in range(6)]
    plans, skipped, _considered = github_client.plan_files(entries)
    assert len(plans) == 5
    assert len(skipped) == 1
    assert "budget" in skipped[0]["reason"].lower()


def test_directory_entries_are_ignored():
    entries = [
        {"path": "src", "type": "tree", "size": 0},
        {"path": "src/main.py", "type": "blob", "size": 10},
    ]
    plans, skipped, considered = github_client.plan_files(entries)
    assert [p.path for p in plans] == ["src/main.py"]
    assert skipped == []
    assert considered == 1


def test_main_language_picks_largest():
    assert github_client._main_language({"Python": 10, "Go": 90}) == "Go"
    assert github_client._main_language({}) == "Unknown"
