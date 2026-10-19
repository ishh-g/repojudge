"""Gemma client (Google Gemma served through the Gemini API).

RepoJudge uses Gemma for exactly two things per analysis:
  1. extracting claims from the README (see ``claims.py``)
  2. judging all claims together against the evidence (see ``verifier.py``)

This module owns the HTTP details: the API key, the JSON-only prompt contract,
safe JSON parsing, short replies, and retry behaviour.

Retries:
  * HTTP 429 (rate limit): wait and retry up to ``config.GEMMA_MAX_RETRIES``.
  * Read timeout: retry ``config.GEMMA_TIMEOUT_RETRIES`` time(s), then raise
    :class:`GemmaTimeoutError` with a clear message for the user.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import requests

from . import config
from .errors import (
    GemmaRateLimitError,
    GemmaResponseError,
    GemmaTimeoutError,
    GemmaUnavailableError,
)

logger = logging.getLogger("repojudge")

# Optional request settings. Each is dropped automatically (and retried) if the
# API or model rejects it.
_THINKING_SUPPORTED = True
_SCHEMA_SUPPORTED = True

# Safety net so the retry loop can never spin forever.
_MAX_TOTAL_ATTEMPTS = 12

# Structured-output schemas. Plain "reply with JSON" prompting is not reliable
# with every Gemma variant (some narrate their reasoning first), so we also ask
# the API to constrain the reply to a schema.
CLAIMS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "claim": {"type": "STRING"},
                    "type": {"type": "STRING",
                             "enum": ["feature", "tech_used", "result",
                                      "requirement"]},
                    "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "strong_keywords": {"type": "ARRAY",
                                        "items": {"type": "STRING"}},
                    "evidence_kind": {
                        "type": "STRING",
                        "enum": ["functional", "presentation"],
                    },
                },
                "required": ["claim", "type", "keywords", "strong_keywords",
                             "evidence_kind"],
            },
        }
    },
    "required": ["claims"],
}

JUDGEMENTS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "judgements": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "supports": {
                        "type": "STRING",
                        "enum": ["full", "part", "none"],
                    },
                    "proof": {"type": "STRING", "nullable": True},
                    "reason": {"type": "STRING"},
                },
                "required": ["id", "supports", "reason"],
            },
        }
    },
    "required": ["judgements"],
}


def gemma_configured() -> bool:
    """True when a Gemini API key is present in the environment."""
    return bool(config.GEMINI_API_KEY)


def thinking_enabled() -> bool:
    """True when RepoJudge is asking the model to think less (faster answers)."""
    return bool(config.GEMMA_THINKING_LEVEL) and _THINKING_SUPPORTED


def schema_enabled() -> bool:
    """True when RepoJudge is asking the API to constrain the reply to a schema."""
    return _SCHEMA_SUPPORTED


def reset_optional_settings() -> None:
    """Restore the thinking/schema support flags. Used by tests."""
    global _THINKING_SUPPORTED, _SCHEMA_SUPPORTED
    _THINKING_SUPPORTED = True
    _SCHEMA_SUPPORTED = True


def _extract_text(payload: dict) -> str:
    """Pull the text out of a Gemini ``generateContent`` response."""
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback") or {}
        reason = feedback.get("blockReason")
        if reason:
            raise GemmaResponseError(
                f"Gemma refused to answer (blocked: {reason}). "
                "Try a different repository or README."
            )
        raise GemmaResponseError("Gemma returned an empty response.")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        finish = candidates[0].get("finishReason")
        raise GemmaResponseError(
            f"Gemma returned no text (finishReason={finish})."
        )
    return text


def _build_body(prompt: str, temperature: float,
                max_output_tokens: Optional[int],
                response_schema: Optional[dict] = None) -> dict:
    """Build the ``generateContent`` request body.

    Replies are kept short (RepoJudge only ever wants compact JSON), a low
    thinking level is requested when supported, and the reply is constrained to
    ``response_schema`` when one is given.
    """
    generation_config: dict = {
        "temperature": temperature,
        "topK": config.GEMMA_TOP_K,
        "topP": config.GEMMA_TOP_P,
        "maxOutputTokens": max_output_tokens or config.GEMMA_MAX_OUTPUT_TOKENS,
    }
    if thinking_enabled():
        generation_config["thinkingConfig"] = {
            "thinkingLevel": config.GEMMA_THINKING_LEVEL
        }
    if response_schema is not None and schema_enabled():
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }


def _error_detail(resp: requests.Response) -> str:
    """Best-effort human-readable detail from an error response."""
    try:
        return (resp.json().get("error") or {}).get("message", "") or ""
    except ValueError:
        return resp.text[:200]


def generate(
    prompt: str,
    *,
    session: Optional[requests.Session] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    response_schema: Optional[dict] = None,
) -> str:
    """Send one prompt to Gemma and return the raw text reply.

    Retry behaviour:
      * HTTP 429 (rate limit): wait and retry up to ``GEMMA_MAX_RETRIES`` times.
      * Read timeout: retry only ``GEMMA_TIMEOUT_RETRIES`` time(s), then give up
        with a clear :class:`GemmaTimeoutError`.
      * HTTP 5xx: retry with backoff, then give up.
      * HTTP 400 caused by an optional setting (thinkingConfig / responseSchema):
        drop that setting and retry immediately.
    """
    global _THINKING_SUPPORTED, _SCHEMA_SUPPORTED

    if not gemma_configured():
        raise GemmaUnavailableError(
            "GEMINI_API_KEY is not set, so Gemma cannot be called. "
            "Set GEMINI_API_KEY in the environment (see .env.example) and retry."
        )

    url = f"{config.GEMINI_API_BASE}/models/{config.GEMMA_MODEL}:generateContent"
    sess = session or requests.Session()
    if temperature is None:
        temperature = config.GEMMA_TEMPERATURE
    body = _build_body(prompt, temperature, max_output_tokens, response_schema)

    timeout_retries = 0
    rate_limit_attempts = 0
    server_attempts = 0
    connection_attempts = 0
    total_attempts = 0

    while total_attempts < _MAX_TOTAL_ATTEMPTS:
        total_attempts += 1

        try:
            resp = sess.post(
                url,
                params={"key": config.GEMINI_API_KEY},
                json=body,
                timeout=config.GEMMA_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            # A slow model, not a transient blip: retry once only.
            if timeout_retries < config.GEMMA_TIMEOUT_RETRIES:
                timeout_retries += 1
                logger.warning(
                    "Gemma read timeout after %ss; retrying once.",
                    config.GEMMA_TIMEOUT_SECONDS,
                )
                continue
            raise GemmaTimeoutError(
                "Gemma is being slow and did not answer in time "
                f"(waited {config.GEMMA_TIMEOUT_SECONDS} seconds and retried "
                "once). This is usually temporary - please try again."
            ) from exc
        except requests.RequestException as exc:
            connection_attempts += 1
            if connection_attempts < 3:
                time.sleep(min(2 ** connection_attempts, 8))
                continue
            raise GemmaUnavailableError(
                f"Could not reach Gemma: {exc}"
            ) from exc

        if resp.status_code == 429:
            rate_limit_attempts += 1
            if rate_limit_attempts < config.GEMMA_MAX_RETRIES:
                time.sleep(config.GEMMA_RETRY_WAIT_SECONDS)
                continue
            raise GemmaRateLimitError(
                "Gemma is rate limited (about 15 requests per minute on the free "
                "tier). RepoJudge retried a few times without success. Please wait "
                "a moment and try again."
            )

        # If the model rejects an optional setting, drop it and retry once.
        if resp.status_code == 400:
            detail = _error_detail(resp)
            lowered = detail.lower()
            changed = False
            if thinking_enabled() and "thinking" in lowered:
                _THINKING_SUPPORTED = False
                changed = True
            if (response_schema is not None and schema_enabled()
                    and any(word in lowered for word in (
                        "schema", "nullable", "required", "enum", "property",
                        "response", "generationconfig"))):
                _SCHEMA_SUPPORTED = False
                changed = True
            if changed:
                logger.info(
                    "Gemma rejected an optional setting (%s); retrying without it.",
                    detail[:120],
                )
                body = _build_body(prompt, temperature, max_output_tokens,
                                   response_schema)
                continue
            raise GemmaUnavailableError(
                f"Gemma request failed (HTTP 400). {detail}".strip()
            )

        if resp.status_code >= 500:
            server_attempts += 1
            if server_attempts < config.GEMMA_MAX_RETRIES:
                time.sleep(min(2 ** server_attempts, 8))
                continue
            raise GemmaUnavailableError(
                f"Gemma is unavailable right now (HTTP {resp.status_code}). "
                "Please try again shortly."
            )

        if resp.status_code >= 400:
            raise GemmaUnavailableError(
                f"Gemma request failed (HTTP {resp.status_code}). "
                f"{_error_detail(resp)}".strip()
            )

        try:
            return _extract_text(resp.json())
        except ValueError as exc:
            raise GemmaResponseError("Gemma returned a malformed response.") from exc

    raise GemmaUnavailableError(
        "Gemma did not respond after several attempts. Please try again shortly."
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> Any:
    """Best-effort JSON parse of a model reply.

    Handles bare JSON, markdown fences, and leading/trailing prose. Raises
    ``ValueError`` when nothing parseable is found.
    """
    if text is None:
        raise ValueError("empty response")

    cleaned = text.strip()

    # 1. Direct parse.
    try:
        return json.loads(cleaned)
    except ValueError:
        pass

    # 2. JSON inside ``` fences.
    fence = _JSON_FENCE_RE.search(cleaned)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except ValueError:
            cleaned = fence.group(1).strip()

    # 3. Slice from the first brace/bracket to the last one.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except ValueError:
                continue

    raise ValueError("no JSON object found in reply")


def generate_json(
    prompt: str,
    *,
    session: Optional[requests.Session] = None,
    retry_on_bad_json: bool = True,
    max_output_tokens: Optional[int] = None,
    response_schema: Optional[dict] = None,
    temperature: Optional[float] = None,
) -> Any:
    """Ask Gemma for JSON and return the parsed object.

    If the reply is not valid JSON we retry exactly once with a stricter
    instruction, then raise :class:`GemmaResponseError`. Passing a
    ``response_schema`` asks the API to constrain the reply to that shape, and
    ``temperature`` overrides the configured default (the claims call runs cold).
    """
    text = generate(prompt, session=session, max_output_tokens=max_output_tokens,
                    response_schema=response_schema, temperature=temperature)
    try:
        return parse_json(text)
    except ValueError:
        if not retry_on_bad_json:
            raise GemmaResponseError(
                "Gemma did not return valid JSON. Please try again."
            )

    stricter = (
        prompt
        + "\n\nIMPORTANT: Reply with ONE compact JSON object and nothing else. "
        "No prose, no explanations, no markdown code fences, no text before or "
        "after the JSON."
    )
    text = generate(stricter, session=session, max_output_tokens=max_output_tokens,
                    response_schema=response_schema, temperature=temperature)
    try:
        return parse_json(text)
    except ValueError as exc:
        raise GemmaResponseError(
            "Gemma returned a reply that could not be parsed as JSON after a "
            "retry. Please try again."
        ) from exc
