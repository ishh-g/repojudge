"""Gemma call 1: extract the claims a README makes.

One Gemma call per analysis. The README plus a short list of the important
files is sent, and Gemma returns JSON: a list of claims, each with a type and
a set of keywords that can be searched for in the code.

Any "extra things to check" the user typed are appended as their own claims
with type ``requirement``.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from . import cache, config
from . import keywords as keywords_module
from .errors import GemmaResponseError
from .gemma_client import CLAIMS_SCHEMA, generate_json
from .github_client import RepoData

VALID_TYPES = {"feature", "tech_used", "result", "requirement"}

# Leading list markers ("- ", "* ", "1. ") that a model may echo from a README.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+")
_TEXT_RE = re.compile(r"[A-Za-z]{3,}")
MAX_CLAIM_CHARS = 240
MAX_STRONG_KEYWORD_WORDS = 3

# Field names a model may leak from the JSON schema into a keyword list. They
# describe the reply, not the project, so they are never useful search terms.
_SCHEMA_WORDS = frozenset({"keywords", "strong_keywords", "evidence_kind"})

# Words and phrases that describe where files live rather than a feature. A
# claim that names a file *and* one of these is about the layout, not the code.
_LAYOUT_PHRASES = (
    "entry point", "entry-point", "entrypoint", "entry", "file layout",
    "file structure", "project structure", "folder structure",
    "directory structure", "code structure", "file tree", "folder tree",
    "router-based", "routing table", "site map", "sitemap", "scaffold",
    "boilerplate", "folder layout",
)
_FILE_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9]{1,8}\b"
)

# Common claim verbs. Used only to tell a real sentence from a bare noun phrase;
# the suffix check below catches most verb forms too.
_COMMON_VERBS = frozenset({
    "is", "are", "was", "were", "has", "have", "had", "can", "will",
    "use", "uses", "used", "read", "reads", "write", "writes", "store",
    "stores", "save", "saves", "load", "loads", "offer", "offers", "provide",
    "provides", "support", "supports", "include", "includes", "render",
    "renders", "display", "displays", "show", "shows", "allow", "allows",
    "enable", "enables", "turn", "turns", "convert", "converts", "generate",
    "generates", "create", "creates", "track", "tracks", "recognize",
    "recognizes", "recognise", "translates", "translate", "download",
    "downloads", "export", "exports", "upload", "uploads", "send", "sends",
    "receive", "receives", "manage", "manages", "handle", "handles",
    "implement", "implements", "connect", "connects", "integrate",
    "integrates", "run", "runs", "build", "builds", "add", "adds", "give",
    "gives", "keep", "keeps", "open", "opens", "close", "closes", "edit",
    "edits", "delete", "deletes", "search", "searches", "filter", "filters",
    "sort", "sorts", "calculate", "calculates", "score", "scores", "train",
    "trains", "test", "tests", "detect", "detects", "capture", "captures",
    "play", "plays", "record", "records", "draw", "draws", "print", "prints",
    "log", "logs", "serve", "serves", "host", "hosts", "deploy", "deploys",
    "install", "installs", "configure", "configures", "authenticate",
    "authenticates", "extract", "extracts", "process", "processes", "analyse",
    "analyses", "analyze", "analyzes", "validate", "validates", "visualize",
    "visualizes", "visualise", "visualises", "predict", "predicts",
})


def _looks_like_sentence(text: str) -> bool:
    """True when ``text`` reads as a plain sentence rather than a label.

    A claim is a sentence when it carries a verb and is more than a single word.
    Bare noun phrases such as "Local storage history" are dropped.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text or "")
    if len(words) < 2:
        return False
    lowered = [word.lower() for word in words]
    if any(word in _COMMON_VERBS for word in lowered):
        return True
    return any(len(word) >= 4 and word.endswith(("s", "es", "ed", "ing"))
               for word in lowered)


def _is_layout_claim(text: str) -> bool:
    """True when a claim is about where files live, not about a feature."""
    if not _FILE_TOKEN_RE.search(text or ""):
        return False
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in _LAYOUT_PHRASES)


def _acceptable_claim(text: str) -> bool:
    """True when a claim is a sentence about the project, not about its files."""
    return _looks_like_sentence(text) and not _is_layout_claim(text)

PROMPT_TEMPLATE = """\
You are a strict technical code auditor. Read the README of a software \
repository and extract the concrete, checkable claims it makes about the project.

Rules:
- Only extract claims the README actually states. Never invent claims. Use only \
words that appear in the README. Each claim must be a short, plain sentence, \
not a heading, a file path or a project-structure note.
- Do not extract a claim about the file layout itself (for example "entry point \
in the main HTML file" or "router-based SPA entry point"): that describes how \
files are arranged, not a feature.
- type "feature": something the project says it can do.
- type "tech_used": a library, framework, service, model or language the project \
says it uses.
- type "result": a measurable outcome or benchmark the project claims (accuracy, \
speed, score, size, and so on).
- type "requirement": one of the extra things the user asked to check (see \
"Extra things to check" below). Return exactly one requirement claim per item.
- "keywords": 1 to 6 short lowercase words or identifiers that would appear in \
the source code or config if the claim were true. Prefer names of functions, \
modules, packages and config keys.
- "strong_keywords": 0 to 6 SHORT, specific technical terms that would have to \
appear in the code if the claim were true. Prefer one or two words that would \
literally appear in the code: a library, package, service or driver name, an \
import, a function, class or API object name, or a file name the claim needs (a \
manifest, config, test or platform file). Never write a long phrase: any term of \
more than three words is discarded, so keep each term short. Use the exact \
spelling the code would use - never join words together or invent a name. Never \
use generic English words such as "test", "data", "api", "server", "user", "app" \
or "feature". If you cannot name a short, specific term, return an empty list.
- "evidence_kind": "presentation" only for claims about styling, layout or \
documentation. Use "functional" for everything else (features, libraries, APIs, \
data storage, tests, platforms, results).
- Extract at most {max_claims} claims. Prefer the most specific and testable ones.
- Keep every "claim" to at most 12 words.
- Write "claim" as plain prose. Never start it with a bullet, dash or number, \
and never leave it empty or truncated.
- Reply with COMPACT JSON ONLY, in exactly this shape:
{{"claims":[{{"claim":"short claim text","type":"feature","keywords":["word"],"strong_keywords":["package-name"],"evidence_kind":"functional"}}]}}
- Output nothing outside that JSON object: no prose, no explanation, no markdown, \
no code fences, no trailing text. Keep the whole reply as short as possible.

Repository: {full_name}
Main language: {language}
Important files:
{file_list}
Extra things to check (return one "requirement" claim per line):
{requirements}

README:
<<<README
{readme}
README
"""


def _clean_keywords(raw: Any, limit: int = 6,
                    max_words: Optional[int] = None) -> list[str]:
    """Normalise keywords into short, searchable, lowercase strings."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if item is None:
            continue
        word = re.sub(r"\s+", " ", str(item)).strip().lower()
        word = word.strip(".,;:!?`'\"()[]{}")
        if not word or len(word) > 60 or word in seen:
            continue
        if word in _SCHEMA_WORDS:
            continue
        if max_words is not None and len(word.split()) > max_words:
            continue
        seen.add(word)
        out.append(word)
        if len(out) >= limit:
            break
    return out


# Values a model may use for "evidence_kind" that all mean "this claim is about
# how the project looks or what it documents, not a technical mechanism".
_PRESENTATION_KINDS = frozenset({
    "presentation", "ui", "ux", "styling", "style", "layout", "visual",
    "design", "docs", "documentation",
})


def _clean_evidence_kind(raw: Any) -> str:
    """What kind of proof a claim needs: "functional" or "presentation"."""
    value = str(raw or "").strip().lower()
    return "presentation" if value in _PRESENTATION_KINDS else "functional"


# --- Grounding: a claim must come from the README or the extra requirements ---

# Opening words that mark a claim as meta-commentary or a paraphrase, not a
# statement the README makes about the project.
_ODD_CLAIM_STARTS = (
    "claims to", "claim to", "claims ", "claim ", "classically",
    "we ", "our ", "this project", "the project", "note that", "note:",
    "here ",
)
_COVERAGE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# A claim must share this fraction of its meaningful words with the README or
# the extra requirements, otherwise it was invented by the model and dropped.
MIN_CLAIM_COVERAGE = 0.6


def _coverage_words(text: str) -> list[str]:
    """Meaningful words of a claim: length >= 3 and not a plain function word."""
    return [word for word in _COVERAGE_WORD_RE.findall((text or "").lower())
            if len(word) >= 3 and word not in keywords_module.STOPWORDS]


def _starts_oddly(text: str) -> bool:
    """True when a claim opens with a word a real README claim would not."""
    lowered = (text or "").strip().lower()
    for start in _ODD_CLAIM_STARTS:
        if lowered.startswith(start):
            return True
    return False


def _claim_is_grounded(claim_text: str, source_text: str) -> bool:
    """True when the README / extra requirements really contain the claim.

    Words are compared on a short stem (their first three letters) so that small
    inflections still count ("adding" matches "adds"), and a claim is kept only
    when at least ``MIN_CLAIM_COVERAGE`` of its meaningful words are found.
    """
    if _starts_oddly(claim_text):
        return False
    words = _coverage_words(claim_text)
    if not words:
        return True
    source_words = _coverage_words(source_text)
    if not source_words:
        return False
    stems = {word[:3] for word in source_words}
    present = sum(1 for word in words if word[:3] in stems)
    return present / len(words) >= MIN_CLAIM_COVERAGE



def _clean_claim_text(raw: Any) -> str:
    """Tidy a claim string: drop list markers, collapse spaces, cap the length."""
    text = _LIST_MARKER_RE.sub("", str(raw).strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CLAIM_CHARS]


def _normalise_claim(item: Any, default_type: str = "feature") -> Optional[dict]:
    """Turn one raw Gemma claim into a clean claim dict, or ``None``.

    Obviously broken items (empty strings, stray punctuation, wrong types) are
    dropped rather than reported as claims the README never made.
    """
    if isinstance(item, str):
        text = _clean_claim_text(item)
        if not _TEXT_RE.search(text) or not _acceptable_claim(text):
            return None
        return {"claim": text, "type": default_type, "keywords": [],
                "strong_keywords": [], "evidence_kind": "functional"}
    if not isinstance(item, dict):
        return None
    text = _clean_claim_text(item.get("claim") or item.get("text") or "")
    if not _TEXT_RE.search(text) or not _acceptable_claim(text):
        return None
    ctype = str(item.get("type") or default_type).strip().lower()
    if ctype not in VALID_TYPES:
        ctype = default_type
    kind = _clean_evidence_kind(item.get("evidence_kind"))
    if ctype == "tech_used":
        # A library, framework or service can never be proved by a stylesheet.
        kind = "functional"
    return {
        "claim": text,
        "type": ctype,
        "keywords": _clean_keywords(item.get("keywords")),
        "strong_keywords": _clean_keywords(
            item.get("strong_keywords"), limit=5,
            max_words=MAX_STRONG_KEYWORD_WORDS,
        ),
        "evidence_kind": kind,
    }


def _requirements_claims(requirements: list[str]) -> list[dict]:
    """Turn the user's extra requirements into claims of their own.

    Used for any requirement the model did not return itself. The keywords are
    the meaningful words of the requirement, and the strong keywords are the
    specific technical words among them, so a requirement still has to be
    proved by something concrete in the code.
    """
    claims: list[dict] = []
    for req in requirements:
        text = str(req).strip()
        if not text:
            continue
        # Keywords are the meaningful words of the requirement itself.
        words = re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        stop = {"the", "and", "for", "with", "that", "this", "are", "was",
                "has", "have", "does", "can", "should", "must", "from", "into",
                "check", "make", "sure", "there", "which", "when", "your"}
        keywords = [w for w in dict.fromkeys(words) if w not in stop][:6]
        claims.append({"claim": text[:300], "type": "requirement",
                       "keywords": keywords,
                       "strong_keywords": keywords_module.specific_terms(text)[:5],
                       "evidence_kind": "functional"})
    return claims


def _covers_requirement(claim_text: str, requirement: str) -> bool:
    """True when a claim the model returned is really the user's requirement."""
    text = keywords_module.normalise_text(claim_text)
    wanted = keywords_module.normalise_text(requirement)
    if not wanted:
        return False
    if wanted in text or text in wanted:
        return True
    claim_tokens = keywords_module.significant_tokens(claim_text)
    wanted_tokens = keywords_module.significant_tokens(requirement)
    if not wanted_tokens:
        return False
    return len(claim_tokens & wanted_tokens) / len(wanted_tokens) >= 0.5


def _merge_duplicate_claims(claim_list: list[dict]) -> list[dict]:
    """Collapse claims that mean the same thing into a single claim.

    A README claim and an extra requirement about the same subject (for example
    the same library or service) should be judged once, not twice. Keywords are
    merged so the combined claim gets the best chance of finding evidence.
    """
    if len(claim_list) < 2:
        return claim_list

    token_sets = [
        keywords_module.significant_tokens(claim.get("claim", ""),
                                          claim.get("keywords") or [])
        for claim in claim_list
    ]
    normal_texts = [keywords_module.normalise_text(claim.get("claim", ""))
                    for claim in claim_list]

    document_frequency: dict[str, int] = {}
    for tokens in token_sets:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    kept_texts: list[str] = []

    for claim, tokens, text in zip(claim_list, token_sets, normal_texts):
        target = None
        for index in range(len(kept)):
            if keywords_module.is_duplicate(tokens, kept_tokens[index],
                                           document_frequency):
                target = index
                break
            # Two claims worded identically are the same claim too.
            if text and text == kept_texts[index]:
                target = index
                break
        if target is None:
            kept.append(dict(claim))
            kept_tokens.append(set(tokens))
            kept_texts.append(text)
            continue
        existing = kept[target]
        existing["keywords"] = keywords_module.merge_keywords(
            existing.get("keywords"), claim.get("keywords")
        )
        existing["strong_keywords"] = keywords_module.merge_keywords(
            existing.get("strong_keywords"), claim.get("strong_keywords"),
            limit=5,
        )
        if (existing.get("evidence_kind") != "presentation"
                or claim.get("evidence_kind") != "presentation"):
            existing["evidence_kind"] = "functional"
        kept_tokens[target] |= tokens
    return kept


def build_prompt(repo: RepoData, max_claims: int,
                 requirements: Optional[list[str]] = None) -> str:
    """Assemble the claim-extraction prompt."""
    file_list = "\n".join(f"- {p}" for p in repo.file_paths) or "- (no source files read)"
    readme = (repo.readme or "")[: config.MAX_README_CHARS]
    requested = [str(r).strip() for r in (requirements or []) if str(r).strip()]
    requirements_block = "\n".join(f"- {item}" for item in requested) or "- (none)"
    return PROMPT_TEMPLATE.format(
        full_name=repo.full_name,
        language=repo.language,
        file_list=file_list,
        readme=readme,
        max_claims=max_claims,
        requirements=requirements_block,
    )


def extract_claims(
    repo: RepoData,
    requirements: Optional[list[str]] = None,
    *,
    session=None,
    use_cache: bool = True,
) -> list[dict]:
    """Return the list of claims for a repository.

    Cached under repo URL + latest commit + requirements text.
    """
    requirements = [r for r in (requirements or []) if str(r).strip()]
    key = cache.make_key("claims", repo.url, repo.last_commit_sha or "",
                         "\n".join(requirements))

    if use_cache:
        cached = cache.get(key)
        if cached and isinstance(cached.get("claims"), list):
            return cached["claims"]

    # Reserve room so the total never exceeds MAX_CLAIMS.
    extract_limit = max(0, config.MAX_CLAIMS - len(requirements))
    claims: list[dict] = []

    if extract_limit > 0:
        prompt = build_prompt(repo, extract_limit, requirements)
        payload = generate_json(
            prompt, session=session,
            max_output_tokens=config.GEMMA_MAX_OUTPUT_TOKENS,
            response_schema=CLAIMS_SCHEMA,
            temperature=config.GEMMA_CLAIMS_TEMPERATURE,
        )
        raw_claims = payload.get("claims") if isinstance(payload, dict) else payload
        if not isinstance(raw_claims, list):
            raise GemmaResponseError(
                "Gemma did not return a list of claims. Please try again."
            )
        for item in raw_claims[:extract_limit]:
            claim = _normalise_claim(item)
            if claim:
                claims.append(claim)

    # The user's extra requirements are always checked, even if the model did
    # not return a claim for them: one claim per requirement, never guessed.
    covered = [c.get("claim", "") for c in claims]
    missing = [r for r in requirements
               if not any(_covers_requirement(text, r) for text in covered)]
    claims.extend(_requirements_claims(missing))
    claims = _merge_duplicate_claims(claims)

    # A claim must come from the README or the user's extra requirements. Drop
    # anything the model invented (too few of its words are in the source) and
    # any claim that does not read as a normal statement about the project.
    source_text = (repo.readme or "") + "\n" + "\n".join(requirements)
    claims = [claim for claim in claims
              if _claim_is_grounded(claim.get("claim", ""), source_text)]

    if use_cache:
        cache.set(key, {"claims": claims})

    return claims
