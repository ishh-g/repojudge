"""Tests for the tiny .env loader in config.py."""
from __future__ import annotations

import os

from app.config import _load_dotenv


def test_loads_simple_values(tmp_path, monkeypatch):
    monkeypatch.delenv("REPOJUDGE_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "REPOJUDGE_TEST_KEY=hello\n"
        'QUOTED_KEY="quoted value"\n'
        "export EXPORTED_KEY=from-export\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("QUOTED_KEY", raising=False)
    monkeypatch.delenv("EXPORTED_KEY", raising=False)

    _load_dotenv(env)

    assert os.environ["REPOJUDGE_TEST_KEY"] == "hello"
    assert os.environ["QUOTED_KEY"] == "quoted value"
    assert os.environ["EXPORTED_KEY"] == "from-export"


def test_real_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOJUDGE_TEST_KEY", "real")
    env = tmp_path / ".env"
    env.write_text("REPOJUDGE_TEST_KEY=fromfile\n", encoding="utf-8")

    _load_dotenv(env)

    assert os.environ["REPOJUDGE_TEST_KEY"] == "real"


def test_missing_file_is_silent(tmp_path):
    _load_dotenv(tmp_path / "does-not-exist.env")  # must not raise


def test_ignores_malformed_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("NOEQUALS", raising=False)
    env = tmp_path / ".env"
    env.write_text("this line has no equals sign\n=novalue\n", encoding="utf-8")
    _load_dotenv(env)  # must not raise
    assert "NOEQUALS" not in os.environ
