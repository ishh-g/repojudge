"""API tests: routes, JSON errors and the Markdown endpoint."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import claims as claims_module
from app import config, github_client, main, verifier
from app.main import app
from fakes import FakeGitHub

client = TestClient(app)

FILES = {
    "main.py": "import argparse\n\ndef main():\n    pass\n",
    "requirements.txt": "requests\n",
    "tests/test_main.py": "def test_x():\n    assert True\n",
}
README = ("# Widget\n\nWidget turns numbers into friendly words for people.\n\n"
          "It has a CLI. It uses the requests library.\n\n"
          "## Installation\n\npip install -r requirements.txt\n\n"
          "## Usage\n\npython main.py\n")


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    yield


def patch_pipeline(monkeypatch, fake_gh):
    real_fetch = github_client.fetch_repo
    monkeypatch.setattr(main.report, "fetch_repo",
                        lambda url, **kw: real_fetch(url, session=fake_gh))
    monkeypatch.setattr(claims_module, "generate_json", lambda *a, **k: {"claims": [
        {"claim": "Has a CLI", "type": "feature", "keywords": ["argparse"]}]})
    monkeypatch.setattr(verifier, "generate_json", lambda *a, **k: {"judgements": [
        {"index": 0, "status": "Verified", "reason": "main.py imports argparse."}]})


def make_fake(**kwargs):
    kwargs.setdefault("files", FILES)
    kwargs.setdefault("readme", README)
    return FakeGitHub(**kwargs)


# ---------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_serves_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert "RepoJudge" in response.text


def test_static_css_is_served():
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_analyze_returns_full_report(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    response = client.post("/analyze",
                           json={"repo_url": "https://github.com/acme/widget"})
    assert response.status_code == 200
    body = response.json()
    assert body["repo"]["full_name"] == "acme/widget"
    assert body["score"]["max"] == 100
    assert body["claims"][0]["status"] == "Verified"
    assert body["id"]


def test_analyze_accepts_requirements_as_string(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    response = client.post("/analyze", json={
        "repo_url": "https://github.com/acme/widget",
        "requirements": "Uses Docker\nHas a database\n",
    })
    assert response.status_code == 200
    types = [c["type"] for c in response.json()["claims"]]
    assert types.count("requirement") == 2


def test_analyze_merges_a_requirement_that_repeats_a_claim(monkeypatch):
    """A requirement that repeats a README claim is judged once, not twice."""
    patch_pipeline(monkeypatch, make_fake())
    response = client.post("/analyze", json={
        "repo_url": "https://github.com/acme/widget",
        "requirements": "Has a CLI\n",
    })
    assert response.status_code == 200
    body = response.json()
    assert [c["claim"] for c in body["claims"]].count("Has a CLI") == 1
    assert body["summary"]["total"] == 1


def test_analyze_rejects_bad_url(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    response = client.post("/analyze", json={"repo_url": "https://gitlab.com/a/b"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_url"


def test_analyze_reports_private_repo(monkeypatch):
    patch_pipeline(monkeypatch, make_fake(private=True))
    response = client.post("/analyze",
                           json={"repo_url": "https://github.com/acme/widget"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "private_repo"


def test_analyze_reports_missing_readme(monkeypatch):
    patch_pipeline(monkeypatch, make_fake(readme=None))
    response = client.post("/analyze",
                           json={"repo_url": "https://github.com/acme/widget"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_readme"


def test_analyze_reports_gemma_timeout(monkeypatch):
    """A Gemma timeout must reach the client as a clear, specific error."""
    from app.errors import GemmaTimeoutError

    real_fetch = github_client.fetch_repo
    fake = make_fake()
    monkeypatch.setattr(main.report, "fetch_repo",
                        lambda url, **kw: real_fetch(url, session=fake))

    def boom(*args, **kwargs):
        raise GemmaTimeoutError(
            "Gemma is being slow and did not answer in time. Please try again."
        )

    monkeypatch.setattr(claims_module, "generate_json", boom)

    response = client.post("/analyze",
                           json={"repo_url": "https://github.com/acme/widget"})
    assert response.status_code == 504
    body = response.json()
    assert body["error"]["code"] == "gemma_timeout"
    assert "try again" in body["error"]["message"].lower()
    assert "slow" in body["error"]["message"].lower()


def test_analyze_requires_repo_url():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_analyze_nocache_is_passed_through(monkeypatch):
    captured = {}

    def fake_analyze(repo_url, requirements, **kwargs):
        captured["repo_url"] = repo_url
        captured["requirements"] = requirements
        captured["force_refresh"] = kwargs.get("force_refresh")
        return {"ok": True}

    monkeypatch.setattr(main.report, "analyze_repo", fake_analyze)

    response = client.post("/analyze", json={
        "repo_url": "https://github.com/acme/widget",
        "nocache": True,
    })
    assert response.status_code == 200
    assert captured["force_refresh"] is True

    response = client.post("/analyze", json={
        "repo_url": "https://github.com/acme/widget",
    })
    assert response.status_code == 200
    assert captured["force_refresh"] is False


def test_index_has_a_rerun_without_cache_checkbox():
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="nocache"' in response.text
    assert "Re-run without cache" in response.text


def test_markdown_endpoint(monkeypatch):
    patch_pipeline(monkeypatch, make_fake())
    report_id = client.post(
        "/analyze", json={"repo_url": "https://github.com/acme/widget"}
    ).json()["id"]

    response = client.get(f"/report/{report_id}/markdown")
    assert response.status_code == 200
    assert "RepoJudge report" in response.text
    assert "text/markdown" in response.headers["content-type"]


def test_markdown_endpoint_unknown_id():
    response = client.get("/report/doesnotexist/markdown")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "report_not_found"
