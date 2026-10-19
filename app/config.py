"""Central configuration for RepoJudge.

Every tunable lives here so behaviour is easy to change in one place.
Secrets are read from the environment only - never hardcode them.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal, dependency-free `.env` loader.

    Reads simple ``KEY=VALUE`` lines and only sets variables that are not
    already present in the environment, so real environment variables always
    win. Comments (``#``) and blank lines are ignored. Never raises.
    """
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


# Load .env next to the project root, if one exists.
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Gemma / Gemini ---------------------------------------------------------
# RepoJudge uses Google's Gemma model, served through the Gemini API.
# Override the model with the GEMMA_MODEL environment variable.
GEMMA_MODEL: str = os.environ.get("GEMMA_MODEL", "gemma-4-26b-a4b-it")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_BASE: str = os.environ.get(
    "REPOJUDGE_GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta"
)

# Hard cap on Gemma calls per analysis.
# 1 call extracts the claims from the README. The rest are used to judge those
# claims in small batches (see MAX_JUDGE_BATCH), plus one retry for any claim
# the model left out of its reply, plus one call to re-judge claims whose
# evidence was filled in from another claim's proof. A claim that is still
# missing after that is reported as "Not judged" - RepoJudge never guesses.
MAX_GEMMA_CALLS: int = 6
# Claims are judged in batches of at most this many per Gemma call, so a long
# reply can never be cut off half way through the JSON.
MAX_JUDGE_BATCH: int = 8
# The claims call runs cold (temperature 0) so the same repository gives the
# same claims every time it is analysed.
GEMMA_CLAIMS_TEMPERATURE: float = float(
    os.environ.get("GEMMA_CLAIMS_TEMPERATURE", "0.0")
)
# On HTTP 429 (rate limit) we wait and retry this many times.
GEMMA_MAX_RETRIES: int = 3
GEMMA_RETRY_WAIT_SECONDS: int = 20
# How long to wait for a reply. Gemma can be slow, so this is generous.
GEMMA_TIMEOUT_SECONDS: int = int(os.environ.get("GEMMA_TIMEOUT_SECONDS", "300"))
# A read timeout is retried only this many times (it is usually a slow model,
# not a transient blip, so we do not want to wait for it repeatedly).
GEMMA_TIMEOUT_RETRIES: int = 1
# Keep replies short: RepoJudge only ever wants compact JSON back.
# Note: too small a budget truncates the JSON mid-object, while too large a
# budget lets slower models ramble until the limit is hit. 2048 is the tested
# sweet spot for the default model.
GEMMA_MAX_OUTPUT_TOKENS: int = int(os.environ.get("GEMMA_MAX_OUTPUT_TOKENS", "2048"))
# Sampling. A little temperature (not 0) plus top-k/top-p keeps this model from
# falling into repetition loops, which used to burn the whole token budget.
GEMMA_TEMPERATURE: float = float(os.environ.get("GEMMA_TEMPERATURE", "0.3"))
GEMMA_TOP_K: int = int(os.environ.get("GEMMA_TOP_K", "40"))
GEMMA_TOP_P: float = float(os.environ.get("GEMMA_TOP_P", "0.95"))
# Ask the model to think less, which makes answers faster. If the model or API
# rejects this setting, RepoJudge drops it automatically and retries.
GEMMA_THINKING_LEVEL: str = os.environ.get("GEMMA_THINKING_LEVEL", "low")

# --- GitHub -----------------------------------------------------------------
GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")
GITHUB_API_BASE: str = os.environ.get(
    "REPOJUDGE_GITHUB_API", "https://api.github.com"
)
GITHUB_RAW_BASE: str = os.environ.get(
    "REPOJUDGE_GITHUB_RAW", "https://raw.githubusercontent.com"
)
GITHUB_TIMEOUT: int = 30

# --- Analysis limits --------------------------------------------------------
MAX_CLAIMS: int = 15
# Safety cap on the number of files we read. The real limit is the size budget
# below: RepoJudge reads every source file until the budget runs out.
MAX_FILES: int = 200
# Total amount of source text RepoJudge reads per repository (about 300 KB).
MAX_TOTAL_FILE_BYTES: int = int(
    os.environ.get("REPOJUDGE_MAX_TOTAL_FILE_BYTES", str(300 * 1024))
)
# A single file bigger than this is cut down to its first part (plus any lines
# that match claim keywords) instead of being skipped.
MAX_SINGLE_FILE_BYTES: int = int(
    os.environ.get("REPOJUDGE_MAX_SINGLE_FILE_BYTES", str(64 * 1024))
)
# How many keyword-matching lines we keep from a truncated large file.
MAX_KEYWORD_LINES_PER_FILE: int = int(
    os.environ.get("REPOJUDGE_MAX_KEYWORD_LINES_PER_FILE", "200")
)
MAX_FILE_LINES: int = 1000                # a single source file above this is "giant"
MAX_README_CHARS: int = 16000             # keep Gemma prompts small
MAX_EVIDENCE_PER_CLAIM: int = 3           # best evidence lines kept per claim

# Languages that get a FULL (code) analysis. Anything else is LIMITED.
FULL_ANALYSIS_LANGUAGES: frozenset[str] = frozenset(
    {"Python", "JavaScript", "TypeScript"}
)

# --- Cache ------------------------------------------------------------------
CACHE_DIR: Path = Path(os.environ.get("REPOJUDGE_CACHE_DIR", "./cache"))

# --- Paths ------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
STATIC_DIR: Path = BASE_DIR / "static"

# --- Report / scoring weights ----------------------------------------------
CLAIM_POINTS: int = 50
QUALITY_POINTS: int = 30
HEALTH_POINTS: int = 20
