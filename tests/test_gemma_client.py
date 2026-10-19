"""Tests for the Gemma HTTP client: parsing, retries, timeouts and request limits."""
from __future__ import annotations

import importlib
import os

import pytest
import requests

from app import config, gemma_client
from app.errors import (
    GemmaRateLimitError,
    GemmaResponseError,
    GemmaTimeoutError,
    GemmaUnavailableError,
)
from fakes import FakeResponse


class FakePost:
    """A stand-in requests session that replays a list of responses.

    A list item that is an exception is raised instead of returned. The last
    item repeats once the list is exhausted.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.bodies = []
        self.timeouts = []
        self.urls = []

    def post(self, url, params=None, json=None, timeout=None):
        self.calls += 1
        self.urls.append(url)
        self.bodies.append(json)
        self.timeouts.append(timeout)
        index = min(self.calls - 1, len(self.responses) - 1)
        item = self.responses[index]
        if isinstance(item, BaseException):
            raise item
        return item


def gemini_text(payload_text: str) -> FakeResponse:
    return FakeResponse(payload={
        "candidates": [{"content": {"parts": [{"text": payload_text}]}}],
    })


@pytest.fixture(autouse=True)
def keyed_and_fast(monkeypatch):
    """Provide a fake key, make retry sleeps instant, reset optional settings."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "GEMMA_THINKING_LEVEL", "low")
    monkeypatch.setattr(gemma_client.time, "sleep", lambda seconds: None)
    gemma_client.reset_optional_settings()
    yield
    gemma_client.reset_optional_settings()


# ---------------------------------------------------------------------------
# Configuration guard
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with pytest.raises(GemmaUnavailableError):
        gemma_client.generate("hello")


def test_gemma_configured(monkeypatch):
    assert gemma_client.gemma_configured() is True
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    assert gemma_client.gemma_configured() is False


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_generate_returns_text():
    session = FakePost([gemini_text("hello world")])
    assert gemma_client.generate("prompt", session=session) == "hello world"
    assert session.calls == 1


def test_request_uses_configured_model():
    session = FakePost([gemini_text("ok")])
    gemma_client.generate("prompt", session=session)
    assert f"/models/{config.GEMMA_MODEL}:generateContent" in session.urls[0]


def test_request_uses_configured_timeout():
    session = FakePost([gemini_text("ok")])
    gemma_client.generate("prompt", session=session)
    assert session.timeouts == [config.GEMMA_TIMEOUT_SECONDS]


# --- rate limits (unchanged behaviour) -------------------------------------


def test_generate_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "GEMMA_MAX_RETRIES", 3)
    session = FakePost([
        FakeResponse(status_code=429, payload={"error": {"message": "rate"}}),
        FakeResponse(status_code=429, payload={"error": {"message": "rate"}}),
        gemini_text("finally"),
    ])
    assert gemma_client.generate("prompt", session=session) == "finally"
    assert session.calls == 3


def test_generate_gives_up_after_max_429(monkeypatch):
    monkeypatch.setattr(config, "GEMMA_MAX_RETRIES", 2)
    session = FakePost([
        FakeResponse(status_code=429, payload={"error": {"message": "rate"}}),
    ])
    with pytest.raises(GemmaRateLimitError):
        gemma_client.generate("prompt", session=session)
    assert session.calls == 2


# --- timeouts (new behaviour) ----------------------------------------------


def test_read_timeout_retries_only_once_then_raises(monkeypatch):
    monkeypatch.setattr(config, "GEMMA_TIMEOUT_RETRIES", 1)
    session = FakePost([requests.exceptions.ReadTimeout("Read timed out")])
    with pytest.raises(GemmaTimeoutError) as exc:
        gemma_client.generate("prompt", session=session)
    # Original attempt plus exactly one retry - not three.
    assert session.calls == 2
    assert "slow" in str(exc.value).lower()
    assert "try again" in str(exc.value).lower()


def test_read_timeout_then_success():
    session = FakePost([
        requests.exceptions.ReadTimeout("Read timed out"),
        gemini_text("eventually"),
    ])
    assert gemma_client.generate("prompt", session=session) == "eventually"
    assert session.calls == 2


def test_connect_timeout_is_also_retried_once():
    session = FakePost([requests.exceptions.ConnectTimeout("connect timed out")])
    with pytest.raises(GemmaTimeoutError):
        gemma_client.generate("prompt", session=session)
    assert session.calls == 2


def test_timeout_error_message_mentions_the_wait():
    session = FakePost([requests.exceptions.ReadTimeout("x")])
    with pytest.raises(GemmaTimeoutError) as exc:
        gemma_client.generate("prompt", session=session)
    assert str(config.GEMMA_TIMEOUT_SECONDS) in str(exc.value)


# --- other failures ---------------------------------------------------------


def test_generate_handles_server_error(monkeypatch):
    monkeypatch.setattr(config, "GEMMA_MAX_RETRIES", 2)
    session = FakePost([FakeResponse(status_code=503, payload={})])
    with pytest.raises(GemmaUnavailableError):
        gemma_client.generate("prompt", session=session)


def test_generate_handles_client_error():
    session = FakePost([FakeResponse(
        status_code=400, payload={"error": {"message": "bad model name"}})])
    with pytest.raises(GemmaUnavailableError):
        gemma_client.generate("prompt", session=session)


def test_generate_handles_blocked_prompt():
    session = FakePost([FakeResponse(payload={"promptFeedback": {"blockReason": "SAFETY"}})])
    with pytest.raises(GemmaResponseError):
        gemma_client.generate("prompt", session=session)


def test_generate_handles_empty_candidates():
    session = FakePost([FakeResponse(payload={"candidates": []})])
    with pytest.raises(GemmaResponseError):
        gemma_client.generate("prompt", session=session)


def test_generate_handles_blank_text():
    session = FakePost([gemini_text("   ")])
    with pytest.raises(GemmaResponseError):
        gemma_client.generate("prompt", session=session)


# ---------------------------------------------------------------------------
# Short answers: output token limit and thinking level
# ---------------------------------------------------------------------------


def test_request_sets_max_output_tokens(monkeypatch):
    monkeypatch.setattr(config, "GEMMA_MAX_OUTPUT_TOKENS", 1024)
    session = FakePost([gemini_text("ok")])
    gemma_client.generate("prompt", session=session)
    assert session.bodies[0]["generationConfig"]["maxOutputTokens"] == 1024


def test_request_sets_sampling_params(monkeypatch):
    """A little temperature plus top-k/top-p stops repetition loops."""
    monkeypatch.setattr(config, "GEMMA_TEMPERATURE", 0.3)
    monkeypatch.setattr(config, "GEMMA_TOP_K", 40)
    monkeypatch.setattr(config, "GEMMA_TOP_P", 0.95)
    session = FakePost([gemini_text("ok")])
    gemma_client.generate("prompt", session=session)
    gen = session.bodies[0]["generationConfig"]
    assert gen["temperature"] == 0.3
    assert gen["topK"] == 40
    assert gen["topP"] == 0.95


def test_explicit_temperature_overrides_config(monkeypatch):
    monkeypatch.setattr(config, "GEMMA_TEMPERATURE", 0.3)
    session = FakePost([gemini_text("ok")])
    gemma_client.generate("prompt", session=session, temperature=0.0)
    assert session.bodies[0]["generationConfig"]["temperature"] == 0.0


def test_explicit_max_output_tokens_overrides(monkeypatch):
    session = FakePost([gemini_text("ok")])
    gemma_client.generate("prompt", session=session, max_output_tokens=256)
    assert session.bodies[0]["generationConfig"]["maxOutputTokens"] == 256


def test_request_asks_for_low_thinking_level():
    session = FakePost([gemini_text("ok")])
    gemma_client.generate("prompt", session=session)
    assert session.bodies[0]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "low"
    }


def test_thinking_omitted_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "GEMMA_THINKING_LEVEL", "")
    session = FakePost([gemini_text("ok")])
    gemma_client.generate("prompt", session=session)
    assert "thinkingConfig" not in session.bodies[0]["generationConfig"]
    assert gemma_client.thinking_enabled() is False


def test_thinking_rejected_is_dropped_and_retried():
    """If the model does not support thinkingConfig, drop it and try again."""
    session = FakePost([
        FakeResponse(status_code=400,
                     payload={"error": {"message": "Invalid thinkingConfig field"}}),
        gemini_text("ok"),
    ])
    assert gemma_client.generate("prompt", session=session) == "ok"
    assert session.calls == 2
    assert "thinkingConfig" in session.bodies[0]["generationConfig"]
    assert "thinkingConfig" not in session.bodies[1]["generationConfig"]
    # It stays off for the rest of the process.
    assert gemma_client.thinking_enabled() is False


# ---------------------------------------------------------------------------
# Structured JSON output (responseSchema)
# ---------------------------------------------------------------------------


SCHEMA = {"type": "OBJECT", "properties": {"ok": {"type": "BOOLEAN"}}}


def test_schema_sent_when_given():
    session = FakePost([gemini_text('{"ok":true}')])
    gemma_client.generate("prompt", session=session, response_schema=SCHEMA)
    gen = session.bodies[0]["generationConfig"]
    assert gen["responseSchema"] == SCHEMA
    assert gen["responseMimeType"] == "application/json"


def test_schema_omitted_when_not_given():
    session = FakePost([gemini_text('{"ok":true}')])
    gemma_client.generate("prompt", session=session)
    assert "responseSchema" not in session.bodies[0]["generationConfig"]


def test_schema_rejected_is_dropped_and_retried():
    """If the model does not support responseSchema, drop it and try again."""
    session = FakePost([
        FakeResponse(status_code=400,
                     payload={"error": {"message": "responseSchema is not supported"}}),
        gemini_text('{"ok":true}'),
    ])
    assert gemma_client.generate("prompt", session=session,
                                 response_schema=SCHEMA) == '{"ok":true}'
    assert session.calls == 2
    assert "responseSchema" in session.bodies[0]["generationConfig"]
    assert "responseSchema" not in session.bodies[1]["generationConfig"]
    assert gemma_client.schema_enabled() is False


def test_generate_json_forwards_schema():
    session = FakePost([gemini_text('{"ok":true}')])
    assert gemma_client.generate_json("prompt", session=session,
                                      response_schema=SCHEMA) == {"ok": True}
    assert session.bodies[0]["generationConfig"]["responseSchema"] == SCHEMA


def test_generate_json_retry_keeps_schema():
    session = FakePost([gemini_text("not json"), gemini_text('{"ok":true}')])
    gemma_client.generate_json("prompt", session=session, response_schema=SCHEMA)
    assert all("responseSchema" in b["generationConfig"] for b in session.bodies)


def test_claims_and_judgement_schemas_are_well_formed():
    for schema in (gemma_client.CLAIMS_SCHEMA, gemma_client.JUDGEMENTS_SCHEMA):
        assert schema["type"] == "OBJECT"
        assert schema.get("required")
        for value in schema["properties"].values():
            assert value["type"] == "ARRAY"
            assert value["items"]["required"]


# ---------------------------------------------------------------------------
# parse_json()
# ---------------------------------------------------------------------------


def test_parse_json_bare_object():
    assert gemma_client.parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_code_fences():
    assert gemma_client.parse_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_parse_json_ignores_surrounding_prose():
    text = 'Sure! Here you go:\n{"claims": []}\nHope that helps.'
    assert gemma_client.parse_json(text) == {"claims": []}


def test_parse_json_handles_arrays():
    assert gemma_client.parse_json("[1, 2, 3]") == [1, 2, 3]


def test_parse_json_rejects_garbage():
    with pytest.raises(ValueError):
        gemma_client.parse_json("no json here at all")


# ---------------------------------------------------------------------------
# generate_json()
# ---------------------------------------------------------------------------


def test_generate_json_parses_reply():
    session = FakePost([gemini_text('{"claims": [{"claim": "x"}]}')])
    assert gemma_client.generate_json("prompt", session=session) == {
        "claims": [{"claim": "x"}]
    }


def test_generate_json_retries_once_on_bad_json():
    session = FakePost([
        gemini_text("not json"),
        gemini_text('{"ok": true}'),
    ])
    assert gemma_client.generate_json("prompt", session=session) == {"ok": True}
    assert session.calls == 2


def test_generate_json_passes_max_output_tokens():
    session = FakePost([gemini_text('{"ok": true}')])
    gemma_client.generate_json("prompt", session=session, max_output_tokens=512)
    assert session.bodies[0]["generationConfig"]["maxOutputTokens"] == 512


def test_generate_json_raises_after_second_bad_reply():
    session = FakePost([gemini_text("nope"), gemini_text("still nope")])
    with pytest.raises(GemmaResponseError):
        gemma_client.generate_json("prompt", session=session)
    assert session.calls == 2


def test_generate_json_without_retry_raises_immediately():
    session = FakePost([gemini_text("nope")])
    with pytest.raises(GemmaResponseError):
        gemma_client.generate_json("prompt", session=session,
                                   retry_on_bad_json=False)
    assert session.calls == 1


# ---------------------------------------------------------------------------
# Defaults required by the spec
# ---------------------------------------------------------------------------


def test_defaults_match_the_spec(monkeypatch):
    """With no environment overrides, the new defaults apply."""
    names = ("GEMMA_MODEL", "GEMMA_TIMEOUT_SECONDS", "GEMMA_MAX_OUTPUT_TOKENS",
             "GEMMA_THINKING_LEVEL", "GEMMA_TEMPERATURE", "GEMMA_TOP_K",
             "GEMMA_TOP_P")
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        monkeypatch.delenv(name, raising=False)
    try:
        reloaded = importlib.reload(config)
        assert reloaded.GEMMA_MODEL == "gemma-4-26b-a4b-it"
        assert reloaded.GEMMA_TIMEOUT_SECONDS == 300
        assert reloaded.GEMMA_TIMEOUT_RETRIES == 1
        assert reloaded.GEMMA_MAX_OUTPUT_TOKENS == 2048
        assert reloaded.GEMMA_THINKING_LEVEL == "low"
        assert reloaded.GEMMA_TEMPERATURE == 0.3
        assert reloaded.GEMMA_TOP_K == 40
        assert reloaded.GEMMA_TOP_P == 0.95
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        importlib.reload(config)
