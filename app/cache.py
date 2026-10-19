"""A dead-simple file cache.

Keys are hashed so they are safe as filenames. Values are JSON. The cache is
keyed by repo URL + latest commit SHA + the extra requirements text, so a
changed repository or changed requirements always produces a fresh analysis.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from . import config


def make_key(*parts: str) -> str:
    """Build a stable cache id from arbitrary string parts."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def _path(key: str) -> Path:
    return config.CACHE_DIR / f"{key}.json"


def get(key: str) -> Optional[dict]:
    """Return a cached value, or ``None`` on a miss."""
    path = _path(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    payload["_cache_hit"] = True
    return payload


def set(key: str, value: dict) -> None:  # noqa: A001 - matching the spec name
    """Write a value to the cache, best effort."""
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stored = dict(value)
        stored.pop("_cache_hit", None)
        stored["_cached_at"] = time.time()
        tmp = _path(key).with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(stored, fh, ensure_ascii=False, indent=2)
        tmp.replace(_path(key))
    except OSError:
        # A cache write failure must never break an analysis.
        pass


def clear() -> int:
    """Delete every cached entry. Returns how many files were removed."""
    if not config.CACHE_DIR.exists():
        return 0
    count = 0
    for item in config.CACHE_DIR.glob("*.json"):
        try:
            item.unlink()
            count += 1
        except OSError:
            pass
    return count
