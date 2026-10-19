"""Milestone 4 tests: rule-based engineering quality checks."""
from __future__ import annotations

from app.github_client import RepoData, RepoFile
from app.quality_checks import (
    check_ci,
    check_dependencies,
    check_documentation,
    check_error_handling,
    check_gitignore,
    check_secrets,
    check_setup,
    check_structure,
    check_tests,
    run_quality_checks,
)

GOOD_README = """\
# Widget

Widget is a small library that turns numbers into friendly words. It exists
so that other programs do not have to repeat the same boring formatting code.

## Installation

pip install widget

## Usage

python main.py
"""


def make_repo(*, files=None, tree=None, readme=GOOD_README,
              analysis_type="full", language="Python") -> RepoData:
    files = dict(files or {})
    if tree is None:
        tree = list(files.keys())
    repo_files = {
        path: RepoFile(path, text, len(text.encode()), text.count("\n") + 1)
        for path, text in files.items()
    }
    return RepoData(
        url="https://github.com/acme/widget", owner="acme", name="widget",
        default_branch="main", private=False, description="", homepage="",
        stars=0, forks=0, open_issues=0, language=language,
        languages={language: 1}, license_name="MIT License", license_spdx="MIT",
        readme=readme, readme_path="README.md", tree=list(tree),
        tree_truncated=False, files=repo_files,
        files_considered=len(tree), files_skipped=0, commit_count=3,
        last_commit_date="2026-01-01T00:00:00Z", last_commit_sha="abc",
        created_at="2025-01-01T00:00:00Z", pushed_at="2026-01-01T00:00:00Z",
        analysis_type=analysis_type,
    )


# ---------------------------------------------------------------------------
# Tests check
# ---------------------------------------------------------------------------


def test_tests_pass_with_tests_directory():
    repo = make_repo(tree=["main.py", "tests/test_main.py"])
    assert check_tests(repo)["status"] == "pass"


def test_tests_pass_with_declared_framework():
    repo = make_repo(files={"pyproject.toml": "[tool.pytest]\n"},
                     tree=["pyproject.toml"])
    result = check_tests(repo)
    assert result["status"] == "pass"
    assert "pytest" in result["reason"]


def test_tests_fail_when_absent():
    repo = make_repo(tree=["main.py"])
    assert check_tests(repo)["status"] == "fail"


# ---------------------------------------------------------------------------
# Setup check
# ---------------------------------------------------------------------------


def test_setup_pass_with_install_and_run():
    repo = make_repo()
    result = check_setup(repo)
    assert result["status"] == "pass"
    assert result["evidence"]


def test_setup_passes_with_a_usage_section_that_has_a_command():
    repo = make_repo(readme="# Widget\n\n## Usage\n\n```\npython app.py --help\n```\n")
    result = check_setup(repo)
    assert result["status"] == "pass"
    assert any("python app.py" in (e["snippet"] or "") for e in result["evidence"])


def test_setup_passes_with_a_getting_started_section():
    repo = make_repo(readme="# Widget\n\n## Getting started\n\nnpm install\n")
    assert check_setup(repo)["status"] == "pass"


def test_setup_passes_when_instructions_are_in_a_subsection():
    readme = (
        "# Widget\n\n"
        "## How to run\n\n"
        "### Option 1: local server\n\n"
        "```bash\n"
        "python server.py\n"
        "```\n\n"
        "### Option 2: open the file\n"
    )
    result = check_setup(make_repo(readme=readme))
    assert result["status"] == "pass"
    assert any("python server.py" in (e["snippet"] or "")
               for e in result["evidence"])


def test_setup_fails_when_instructions_have_no_commands():
    repo = make_repo(readme="# Widget\n\n## Installation\n\nJust download it.\n")
    assert check_setup(repo)["status"] == "fail"


def test_setup_fail_when_missing_instructions():
    repo = make_repo(readme="# Widget\n\nJust a thing.\n")
    result = check_setup(repo)
    assert result["status"] == "fail"
    assert "install" in result["reason"]


# ---------------------------------------------------------------------------
# Dependencies check
# ---------------------------------------------------------------------------


def test_dependencies_pass():
    repo = make_repo(tree=["main.py", "requirements.txt"])
    assert check_dependencies(repo)["status"] == "pass"


def test_dependencies_fail():
    repo = make_repo(tree=["main.py"])
    assert check_dependencies(repo)["status"] == "fail"


def test_dependencies_not_applicable_for_plain_html_site():
    repo = make_repo(
        files={"app.js": "console.log('hi');\n",
               "index.html": '<script src="https://cdn.example.com/lib.js"></script>\n'},
        tree=["index.html", "app.js"],
    )
    result = check_dependencies(repo)
    assert result["status"] == "not_applicable"
    assert "not applicable" in result["reason"].lower()


def test_dependencies_not_applicable_for_stdlib_only_python():
    repo = make_repo(files={"main.py": "import os\nimport sys\n"}, tree=["main.py"])
    assert check_dependencies(repo)["status"] == "not_applicable"


def test_dependencies_fail_when_an_external_package_is_imported():
    repo = make_repo(files={"main.py": "import requests\n"}, tree=["main.py"])
    assert check_dependencies(repo)["status"] == "fail"


def test_dependencies_fail_when_javascript_imports_a_package():
    repo = make_repo(files={"app.js": "import lodash from 'lodash';\n"},
                     tree=["app.js"])
    assert check_dependencies(repo)["status"] == "fail"


def test_dependencies_ignores_relative_imports():
    repo = make_repo(
        files={"app.js": "import './local.js';\nimport helper from './helper';\n"},
        tree=["app.js", "local.js", "helper.js"],
    )
    assert check_dependencies(repo)["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# Secrets check
# ---------------------------------------------------------------------------


def test_secrets_pass_for_clean_repo():
    repo = make_repo(
        files={"main.py": "import os\n\nkey = os.environ['API_KEY']\n"},
        tree=["main.py", ".gitignore"],
    )
    assert check_secrets(repo)["status"] == "pass"


def test_secrets_detects_hardcoded_key_and_masks_it():
    secret = "sk-live-abcd1234efgh5678ijkl"
    repo = make_repo(
        files={"config.py": f'API_KEY = "{secret}"\n'},
        tree=["config.py", ".gitignore"],
    )
    result = check_secrets(repo)
    assert result["status"] == "fail"
    assert result["evidence"]
    snippet = result["evidence"][0]["snippet"]
    # The full secret value must never be shown.
    assert secret not in snippet
    assert "sk-l***" in snippet
    assert result["evidence"][0]["line"] == 1


def test_secrets_detects_aws_and_github_tokens():
    repo = make_repo(
        files={"a.py": 'x = "AKIAIOSFODNN7EXAMPLE"\n',
               "b.py": 'y = "ghp_0123456789abcdefghijABCDEFGHIJ0123"\n'},
        tree=["a.py", "b.py", ".gitignore"],
    )
    result = check_secrets(repo)
    assert result["status"] == "fail"
    assert len(result["evidence"]) == 2


def test_secrets_flags_committed_env_file():
    repo = make_repo(files={".env": "API_KEY=whatever\n"},
                     tree=[".env", ".gitignore"])
    result = check_secrets(repo)
    assert result["status"] == "fail"
    assert any("env" in (e["snippet"] or "").lower() for e in result["evidence"])


def test_secrets_check_does_not_cover_gitignore():
    repo = make_repo(files={"main.py": "print(1)\n"}, tree=["main.py"])
    assert check_secrets(repo)["status"] == "pass"


def test_secrets_pass_has_no_dangling_evidence():
    repo = make_repo(files={"main.py": "print(1)\n"}, tree=["main.py"])
    assert check_secrets(repo)["evidence"] == []


def test_gitignore_check_passes():
    repo = make_repo(tree=["main.py", ".gitignore"])
    result = check_gitignore(repo)
    assert result["status"] == "pass"
    assert result["evidence"][0]["file"] == ".gitignore"


def test_gitignore_check_fails_without_a_file_location():
    repo = make_repo(tree=["main.py"])
    result = check_gitignore(repo)
    assert result["status"] == "fail"
    assert "gitignore" in result["reason"].lower()
    # The finding has no file to point at, so it must not invent one.
    assert result["evidence"][0]["file"] is None


def test_secrets_ignores_placeholder_and_env_refs():
    repo = make_repo(
        files={"config.py": (
            'api_key = "your-api-key-here"\n'
            'token = os.environ.get("TOKEN")\n'
            'secret = "${SECRET_VALUE}"\n'
        )},
        tree=["config.py", ".gitignore"],
    )
    assert check_secrets(repo)["status"] == "pass"


def test_secrets_ignores_comments():
    repo = make_repo(
        files={"config.py": '# api_key = "sk-live-abcdefgh12345678"\n'},
        tree=["config.py", ".gitignore"],
    )
    assert check_secrets(repo)["status"] == "pass"


# ---------------------------------------------------------------------------
# Structure check
# ---------------------------------------------------------------------------


def test_structure_pass_for_folders():
    repo = make_repo(files={"src/a.py": "x=1\n", "src/b.py": "y=2\n"})
    assert check_structure(repo)["status"] == "pass"


def test_structure_fails_for_giant_file():
    repo = make_repo(files={"big.py": "x = 1\n" * 1200})
    result = check_structure(repo)
    assert result["status"] == "fail"
    assert "1201 lines" in result["reason"]


def test_structure_fails_for_single_flat_file():
    repo = make_repo(files={"solo.py": "x = 1\n"})
    result = check_structure(repo)
    assert result["status"] == "fail"
    assert "flat" in result["reason"]


# ---------------------------------------------------------------------------
# Documentation check
# ---------------------------------------------------------------------------


def test_documentation_pass():
    assert check_documentation(make_repo())["status"] == "pass"


def test_documentation_fail_for_title_only():
    repo = make_repo(readme="# Widget\n")
    result = check_documentation(repo)
    assert result["status"] == "fail"


# ---------------------------------------------------------------------------
# Error handling check
# ---------------------------------------------------------------------------


def test_error_handling_pass_python():
    repo = make_repo(files={"main.py": "try:\n    x()\nexcept ValueError:\n    pass\n"})
    result = check_error_handling(repo)
    assert result["status"] == "pass"
    assert result["evidence"][0]["line"] == 1


def test_error_handling_pass_javascript():
    repo = make_repo(files={"app.js": "try {\n  go();\n} catch (e) {}\n"})
    assert check_error_handling(repo)["status"] == "pass"


def test_error_handling_fail():
    repo = make_repo(files={"main.py": "print('hi')\n"})
    assert check_error_handling(repo)["status"] == "fail"


# ---------------------------------------------------------------------------
# CI check
# ---------------------------------------------------------------------------


def test_ci_pass():
    repo = make_repo(tree=["main.py", ".github/workflows/ci.yml"])
    assert check_ci(repo)["status"] == "pass"


def test_ci_missing_is_not_checked_bonus():
    repo = make_repo(tree=["main.py"])
    result = check_ci(repo)
    assert result["status"] == "not_checked"
    assert result["bonus"] is True
    assert "bonus" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Limited analysis
# ---------------------------------------------------------------------------


def test_limited_marks_only_code_checks_not_checked():
    repo = make_repo(
        files={"main.py": "print(1)\n"},
        tree=["main.py", "requirements.txt", ".gitignore"],
        analysis_type="limited", language="Rust",
    )
    results = {c["id"]: c for c in run_quality_checks(repo)}
    assert results["secrets"]["status"] == "not_checked"
    assert results["structure"]["status"] == "not_checked"
    assert results["error_handling"]["status"] == "not_checked"
    # Tree / README based checks still run.
    assert results["dependencies"]["status"] == "pass"
    assert results["documentation"]["status"] == "pass"


def test_run_quality_checks_returns_all_in_order():
    repo = make_repo(files={"src/main.py": "print(1)\n"},
                     tree=["src/main.py", "requirements.txt", ".gitignore",
                           "tests/test_main.py", ".github/workflows/ci.yml"])
    results = run_quality_checks(repo)
    assert [c["id"] for c in results] == [
        "tests", "setup", "dependencies", "secrets", "gitignore", "structure",
        "documentation", "error_handling", "ci",
    ]
    assert all("status" in c and "weight" in c for c in results)


def test_check_weights_still_total_thirty():
    from app.quality_checks import CHECK_WEIGHTS
    assert sum(CHECK_WEIGHTS.values()) == 30
    assert CHECK_WEIGHTS["secrets"] + CHECK_WEIGHTS["gitignore"] == 6
