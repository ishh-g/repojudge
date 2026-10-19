"""Evidence search plus Gemma call 2: judge the claims against the evidence.

Step 1 (no LLM): for every claim, search the files we read for its keywords and
its *strong* keywords (the specific technical terms that would have to appear in
the code if the claim were true) and collect the best evidence lines as
``file:line`` with a short snippet. Keywords are matched case-insensitively and
with simple variants (a library written ``foo.js`` is also matched as ``foo``,
which covers ``new Foo(``). Hits are ranked by relevance and only the best few
are kept, so unrelated files are not listed.

Step 2 (Gemma): the claims are judged in small batches (``MAX_JUDGE_BATCH`` per
call) against the evidence we found. For every claim the model must answer with
``supports`` ("full", "part" or "none"), the exact ``proof`` (a ``file:line``
from that claim's evidence, or null) and a short ``reason``.

Hard rules enforced in code - the model can never talk RepoJudge into a status
the evidence does not support:

* no qualifying evidence, or none of the claim's strong keywords anywhere in the
  code, means ``Unverified``;
* a claim with no usable model judgement becomes ``Not judged`` after one extra
  call, is never guessed, and does not count towards the score;
* ``supports`` is "none" (or the proof does not point at the evidence) means
  ``Unverified``;
* ``supports`` is "part" means ``Partial``; "full" with a real proof means
  ``Verified``;
* a reason must never name a file that is not in the claim's evidence.

RepoJudge never guesses.
"""
from __future__ import annotations

import fnmatch
import re
from typing import Any, Iterator, Optional

from . import cache, config
from . import keywords as keywords_module
from .errors import GemmaResponseError
from .gemma_client import JUDGEMENTS_SCHEMA, generate_json
from .github_client import RepoData

VALID_STATUSES = {"Verified", "Partial", "Unverified", "Not judged"}
UNJUDGED_STATUS = "Not judged"
NO_PROOF_REASON = "No proof found in the code read."
UNJUDGED_REASON = "No judgement was returned for this claim."
SNIPPET_CHARS = 180
MAX_HITS_PER_FILE = 2

# What the model's "supports" value means.
SUPPORTS_FULL = "full"
SUPPORTS_PART = "part"
SUPPORTS_NONE = "none"

# A strong-keyword hit is far better evidence than a plain word hit, so it is
# ranked first and the model is asked to point its proof at one of those lines.
STRONG_BONUS = 8

# A line that includes a library (an HTML script/link tag or an import/require)
# is the best possible proof that the library is used, so it outranks a line
# that merely calls it. A line that actually constructs or calls the library is
# next best.
INCLUDE_BONUS = 24
USAGE_BONUS = 6

# Lines that load a library or import a module. The line must still match one of
# the claim's keywords to be considered at all (see :func:`find_evidence`).
_INCLUDE_RE = re.compile(
    r"(?i)(<\s*(?:script|link)\b|\bimport\b|\brequire\s*\(|@import\b|"
    r"^\s*from\s+[\w.]+\s+import\b|\busing\s+[\w.]+)"
)

# Real source and test files are stronger evidence than markup or configuration,
# so they are ranked first. Markup still counts (a script or link tag that loads
# a library really is proof that the library is used), it just never outranks
# code that mentions the same thing.
_KIND_BONUS = {"code": 6, "test": 6, "config": 3}

# ---------------------------------------------------------------------------
# File kinds
#
# A stylesheet, a document or a lockfile can never prove that a library, API,
# data store, test suite or platform is really implemented. Only source, test,
# markup and configuration files can. Markup counts because including a library
# with a script or link tag is real proof (for example a CDN tag in an entry
# page).
# ---------------------------------------------------------------------------

STYLE_EXTENSIONS = frozenset({".css", ".scss", ".sass", ".less", ".styl",
                              ".stylus", ".pcss"})
DOCS_EXTENSIONS = frozenset({".md", ".markdown", ".mdx", ".rst", ".txt",
                             ".adoc", ".asciidoc", ".textile", ".org", ".man"})
MARKUP_EXTENSIONS = frozenset({".html", ".htm", ".xhtml", ".xml", ".svg",
                               ".vue", ".svelte", ".astro", ".hbs", ".ejs",
                               ".j2", ".jinja", ".jinja2", ".erb", ".twig"})
# Kinds that may prove a technical ("functional") claim.
FUNCTIONAL_KINDS = frozenset({"code", "test", "markup", "config"})

_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs",
                        "testing", "testdata"})

# Standard build and dependency files. Their names matter even when the
# extension is normally documentation-like (``requirements.txt``), because they
# are configuration, not prose.
_CONFIG_NAMES = frozenset({
    "requirements.txt", "requirements-dev.txt", "requirements-test.txt",
    "requirements-prod.txt", "constraints.txt", "pipfile", "lock.txt",
    "dockerfile", "makefile", "gnumakefile", "cmakelists.txt", "procfile",
    "gemfile", "rakefile", "justfile", "vagrantfile", "brewfile", "cargo.txt",
})

# A line that is only a comment proves nothing: it is documentation, not code.
_COMMENT_PREFIXES = ("//", "--", "/*", "*/", "*", "<!--", "-->", '"""', "'''",
                     ";;")
_C_HASH_DIRECTIVES = ("#include", "#define", "#pragma", "#if", "#else",
                      "#elif", "#endif", "#undef", "#error", "#warning", "#line",
                      "#!/")

# File names a reason or proof might mention (only real source/config
# extensions).
_REASON_PATH_RE = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_./\\-]*\.(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|vue|svelte|"
    r"html|htm|css|scss|json|ya?ml|toml|cfg|ini|md|txt|rs|go|java|kt|rb|php|"
    r"c|cpp|h|hpp|cs|swift|sh|sql|lock|xml|gradle)\b",
    re.IGNORECASE,
)

# The "file:line" form of a proof, anywhere in a string.
_PROOF_RE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./\\-]*)\s*[:#]\s*(\d+)")
_PATH_LIKE_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")


PROMPT_TEMPLATE = """\
You are a strict, evidence-only code auditor. For each claim about a software \
repository you are given the claim text, its strong terms and the evidence lines \
(file:line plus the exact code) that a search of the repository found.

Decide how well the evidence supports each claim:
- "full": the evidence shows the claim really is implemented - the library is \
imported or included, the function is called, the config or test really exists.
- "part": the evidence is related but only partly shows the claim (declared but \
never used, only a small part present, a stand-in instead of the real thing).
- "none": the evidence does not show the claim at all.

Rules:
- Judge ONLY from the evidence shown. Never use outside knowledge or assumptions.
- A mock, fake, demo, sample, default or hard-coded list is NOT a database, an \
API or a real feature.
- An in-memory list or browser localStorage is not a database or server storage.
- A function that only prints, logs, fakes or simulates something does not \
implement the real feature.
- A comment, a CSS rule, a README sentence or a lockfile is NOT proof of a \
library, API, database, test suite or platform. Such a thing may only prove a \
claim about styling, layout or documentation.
- A tag that loads a library (a script or link tag, for example from a CDN) IS \
proof that the library is used. Other words or a comment in an HTML file are not.
- A claim about tests needs a real test file, test folder or test framework in \
the evidence. A claim about a database needs a real driver, query or connection \
in the evidence. A claim about a mobile or desktop app needs the platform files \
or framework in the evidence.
- A shared word is NOT proof. The word "mobile" in a responsive web UI, "test" \
in a toast or "database" in a variable name does not partly prove a mobile app, \
a test suite or a database. For a claim about a mobile or desktop app, answer \
"part" only when real platform code or a framework is present; when only web \
markup or UI is present, answer "none".
- If the claim names a version and the evidence shows a different version, \
answer "part" and say which version the code uses.
- If the evidence does not show the claim, answer "none". Do not be generous.
- "proof" must be the exact file:line of the ONE evidence line that proves the \
claim, copied from that claim's evidence list. If nothing proves it, use null.
- "reason" is ONE short sentence of at most 15 words. It must never contradict \
the decision. Only ever name a file that appears in that claim's evidence.
- Every claim id must appear exactly once in your answer.

Reply with COMPACT JSON ONLY, in exactly this shape:
{{"judgements":[{{"id":0,"supports":"full","proof":"path/to/file.js:12","reason":"short reason"}}]}}
Output nothing outside that JSON object: no prose, no explanation, no markdown, \
no code fences, no trailing text. Keep the whole reply as short as possible.

Claims and evidence:
{claims_block}
"""


# ---------------------------------------------------------------------------
# File kinds and comments
# ---------------------------------------------------------------------------


def _extension(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return ("." + base.rsplit(".", 1)[1].lower()) if "." in base else ""


def file_kind(path: str) -> str:
    """Classify a file as code, test, markup, config, style or docs.

    The decision only uses the path itself (extension, file name and folders),
    so it works for any repository.
    """
    parts = [part.lower() for part in (path or "").split("/")]
    name = parts[-1] if parts else ""
    ext = _extension(name)
    if ext in STYLE_EXTENSIONS:
        return "style"
    if name in _CONFIG_NAMES or name.startswith(("requirements", "dockerfile",
                                                 "makefile")):
        return "config"
    if ext in DOCS_EXTENSIONS:
        return "docs"
    if any(part in _TEST_DIRS for part in parts[:-1]):
        return "test"
    if name.startswith(("test_", "test.")) or name.endswith(("_test.py",
                                                            "_test.go",
                                                            "_test.js",
                                                            "_test.ts")):
        return "test"
    if ".test." in name or ".spec." in name or "__tests__" in parts:
        return "test"
    if ext in MARKUP_EXTENSIONS:
        return "markup"
    if ext in {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env",
               ".properties", ".conf", ".lock"}:
        return "config"
    return "code"


def is_comment_only(line: str) -> bool:
    """True when a line is nothing but a comment.

    A comment is documentation: it can describe a feature that was never built,
    so it is never treated as proof.
    """
    stripped = (line or "").strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return not stripped.lower().startswith(_C_HASH_DIRECTIVES)
    return stripped.startswith(_COMMENT_PREFIXES)


def _is_readme(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return name.startswith("readme") or name.startswith("changelog")


def _allowed_kinds(claim: dict) -> Optional[frozenset[str]]:
    """Which file kinds may prove this claim, or ``None`` for any of them.

    Styling, layout and documentation claims may be proved by a stylesheet or a
    document. Every other claim must be proved by code, a test, markup or
    configuration.
    """
    if str(claim.get("type") or "").lower() == "tech_used":
        return FUNCTIONAL_KINDS
    if str(claim.get("evidence_kind") or "").lower() == "presentation":
        return None
    return FUNCTIONAL_KINDS


# ---------------------------------------------------------------------------
# Step 1: evidence search (rule based, no LLM)
# ---------------------------------------------------------------------------


def _iter_lines(repo_file) -> Iterator[tuple[int, str]]:
    """Every line we hold for a file, with its real line number.

    For a large file whose middle was not read, ``extra_lines`` holds the later
    lines that matched a claim keyword, keyed by their real line number.
    """
    for number, line in enumerate(repo_file.content.splitlines(), start=1):
        yield number, line
    for number in sorted(repo_file.extra_lines or {}):
        yield number, repo_file.extra_lines[number]


def _preferred_files(repo: RepoData, keywords: list[str]) -> set[str]:
    """Files whose own name appears among the claim's keywords.

    The model often lists the file a claim is about (for example
    ``translator.js``). When it does, lines from that file are stronger
    evidence than a line that merely shares a word.
    """
    by_name: dict[str, str] = {}
    for path in repo.files:
        name = path.rsplit("/", 1)[-1].lower()
        by_name.setdefault(name, path)

    preferred: set[str] = set()
    for keyword in keywords:
        candidate = (keyword or "").strip().lower().replace("\\", "/")
        if not candidate:
            continue
        for form in (candidate, candidate.rsplit("/", 1)[-1]):
            path = by_name.get(form)
            if path:
                preferred.add(path)
    return preferred


def _repo_name_keys(repo: RepoData) -> set[str]:
    """The repository's own name(s), normalised for comparison.

    A strong keyword that is just the project's own name is not discriminating:
    it appears in almost every import while proving nothing about a specific
    claim. It is still used for the gate, but it must not hide the claim's other
    evidence.
    """
    keys: set[str] = set()
    for value in (repo.full_name or "", repo.name or ""):
        text = str(value).replace("/", "")
        key = keywords_module.normalise_text(text)
        if key:
            keys.add(key)
    return keys


def _is_repo_name_keyword(keyword: str, repo_names: set[str]) -> bool:
    key = keywords_module.normalise_text(
        keywords_module.strip_version(str(keyword or ""))
    )
    return bool(key) and key in repo_names


def find_evidence(
    repo: RepoData,
    keywords: list[str],
    *,
    limit: int | None = None,
    strong_keywords: list[str] | None = None,
    allowed_kinds: Optional[frozenset[str]] = None,
    allow_comments: bool = True,
) -> list[dict]:
    """Search every file we read for keywords and return the best evidence hits.

    Each hit is ``{"file": ..., "line": ..., "snippet": ...}``. Matching is
    case-insensitive and also tries simple variants of each keyword. Hits are
    ranked by relevance and only the best few are kept per claim, so unrelated
    files are never listed. The README itself is skipped so that a claim is
    never "proved" by the sentence that made it.

    ``allowed_kinds`` restricts the search to certain kinds of file (see
    :func:`file_kind`) and ``allow_comments=False`` drops lines that are only a
    comment, so a stylesheet rule or a TODO can never be passed off as proof.

    When any strong-keyword hit exists, only strong-keyword lines are kept: the
    ordinary keywords would otherwise pad the list with lines that merely share
    a common word. A ``<script>``/``<link>`` tag or an import for the library is
    ranked above a line that calls it.
    """
    cap = limit or config.MAX_EVIDENCE_PER_CLAIM
    searchable = keywords_module.searchable_keywords(keywords)
    strong_searchable = keywords_module.searchable_strong_keywords(
        strong_keywords or []
    )
    # A strong keyword that is only the repo's own name proves nothing specific,
    # so it must not force the strong-only filter (which would hide the claim's
    # real evidence in files that never repeat the project name).
    repo_names = _repo_name_keys(repo)
    if repo_names:
        strong_searchable = [
            (keyword, terms) for keyword, terms in strong_searchable
            if not _is_repo_name_keyword(keyword, repo_names)
        ]
    if not searchable and not strong_searchable:
        return []
    preferred = _preferred_files(
        repo, list(keywords or []) + list(strong_keywords or [])
    )

    scored: list[tuple[int, str, int, str, bool]] = []
    for path, repo_file in repo.files.items():
        if _is_readme(path):
            continue
        kind = file_kind(path)
        if allowed_kinds is not None and kind not in allowed_kinds:
            continue
        kind_bonus = _KIND_BONUS.get(kind, 0)
        for number, line in _iter_lines(repo_file):
            snippet = line.strip()
            if not snippet:
                continue
            if not allow_comments and is_comment_only(snippet):
                continue
            best = 0
            best_strong = 0
            matched_keywords = 0
            matched_strong_terms: set[str] = set()
            lowered = snippet.lower()
            for _keyword, terms in searchable:
                score = keywords_module.score_line(snippet, terms)
                if score:
                    matched_keywords += 1
                    if score > best:
                        best = score
            for _keyword, terms in strong_searchable:
                score = keywords_module.score_line(snippet, terms)
                if score:
                    matched_keywords += 1
                    if score > best_strong:
                        best_strong = score
                    for term in terms:
                        if term in lowered:
                            matched_strong_terms.add(term)
            if not best and not best_strong:
                continue
            # A line that matches several of the claim's keywords, or that lives
            # in a file the claim names, is better evidence.
            total = (max(best, best_strong) + 4 * (matched_keywords - 1)
                     + kind_bonus)
            if best_strong:
                total += STRONG_BONUS
                if _is_include_line(snippet):
                    total += INCLUDE_BONUS
                if _uses_term(snippet, matched_strong_terms):
                    total += USAGE_BONUS
            if path in preferred:
                total += 5
            scored.append((total, path, number, snippet, bool(best_strong)))

    # When a claim has any strong-keyword proof, show only those lines: the
    # plain keyword matches are how a common word drags in unrelated lines.
    strong_hits = [hit for hit in scored if hit[4]]
    candidates = strong_hits or scored

    # Most relevant first; ties broken by file then line so results are stable.
    candidates.sort(key=lambda hit: (-hit[0], hit[1], hit[2]))

    hits: list[dict] = []
    per_file: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    picked: set[tuple[str, int]] = set()

    def take(path: str, number: int, snippet: str) -> None:
        fingerprint = (path, re.sub(r"\s+", " ", snippet.lower()))
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        picked.add((path, number))
        per_file[path] = per_file.get(path, 0) + 1
        text = snippet if len(snippet) <= SNIPPET_CHARS else snippet[:SNIPPET_CHARS] + "…"
        hits.append({"file": path, "line": number, "snippet": text})

    # Fill the list in widening passes: first one line per file, so a single
    # noisy file cannot fill the whole list, then a second line, then any line.
    for max_per_file in (1, MAX_HITS_PER_FILE, cap):
        for _score, path, number, snippet, _strong in candidates:
            if len(hits) >= cap:
                break
            if per_file.get(path, 0) >= max_per_file:
                continue
            if (path, number) in picked:
                continue
            take(path, number, snippet)
        if len(hits) >= cap:
            break

    return hits


def _is_include_line(line: str) -> bool:
    """True when a line loads a library or imports a module.

    A ``<script>`` / ``<link>`` tag or an import/require statement is the best
    possible proof that a library is really used.
    """
    return bool(_INCLUDE_RE.search(line or ""))


def _uses_term(line: str, terms: set[str]) -> bool:
    """True when a line really calls, constructs or namespaces one of ``terms``.

    A constructor (``new Chart(``), a call (``chart(``) or a method call
    (``canvas.toDataURL(``) counts. A plain property read (``canvas.width``) does
    not, so the line that actually *does* something ranks above a line that only
    mentions the object.

    A term may also start a longer identifier that is constructed or called: a
    strong term like ``speech`` is really used on a line such as
    ``new SpeechSynthesisUtterance(text)``, where the object name continues after
    the term.
    """
    lowered = (line or "").lower()
    tail = r"[A-Za-z0-9_$]*"
    for term in terms:
        if not term:
            continue
        escaped = re.escape(term)
        if re.search(rf"\bnew\s+{escaped}{tail}\b", lowered):
            return True
        if re.search(rf"\b{escaped}{tail}\s*\(", lowered):
            return True
        if re.search(rf"\b{escaped}{tail}\s*\.\s*[A-Za-z_$][\w$]*\s*\(", lowered):
            return True
    return False


def _looks_like_path(value: str) -> bool:
    """True when a strong keyword names a file, folder or glob."""
    text = (value or "").strip().strip("`'\"")
    if not text:
        return False
    if text.endswith("/") or "/" in text or "\\" in text:
        return True
    if "*" in text or "?" in text:
        return True
    return bool(_PATH_LIKE_RE.search(text))


def _tree_has(repo: RepoData, pattern: str) -> bool:
    """True when the repository really contains a file matching ``pattern``."""
    text = (pattern or "").strip().strip("`'\"").lstrip("./").replace("\\", "/")
    text = text.lower()
    if not text:
        return False
    for path in repo.tree or []:
        lowered = path.lower()
        if text.endswith("/"):
            if lowered.startswith(text) or ("/" + text) in lowered:
                return True
            continue
        if lowered == text or lowered.endswith("/" + text):
            return True
        if fnmatch.fnmatch(lowered, text) or fnmatch.fnmatch(
            lowered.rsplit("/", 1)[-1], text
        ):
            return True
    return False


def strong_proof_found(
    repo: RepoData,
    strong_keywords: list[str] | None,
    *,
    allowed_kinds: Optional[frozenset[str]] = None,
    fallback_keywords: list[str] | None = None,
) -> bool:
    """True when a claim's specific terms appear somewhere they can prove it.

    The first pass searches the claim's strong keywords, expanded with tolerant
    variants (case, spaces, dots, dashes, underscores, camelCase, plurals and a
    redundant ``js``/``api`` suffix), and also checks whether a strong keyword
    names a file the repository really contains.

    Only if that finds nothing is the claim given up on: a second pass then
    searches the claim's ordinary keywords and the main nouns of the claim
    sentence, in case the model chose a long, joined or misspelled strong term.
    A claim with no usable strong keyword has nothing to gate on, so this
    returns True and the model's judgement decides on its own.
    """
    raw = [str(item).strip() for item in (strong_keywords or [])
           if str(item).strip()]
    if not raw:
        return True

    usable = keywords_module.searchable_strong_keywords(raw)
    if usable and find_evidence(repo, [], limit=1, strong_keywords=raw,
                                allowed_kinds=allowed_kinds,
                                allow_comments=False):
        return True

    for keyword in raw:
        if _looks_like_path(keyword) and _tree_has(repo, keyword):
            return True

    fallback = keywords_module.specific_terms(fallback_keywords or [])
    if fallback and find_evidence(repo, fallback, limit=1,
                                  allowed_kinds=allowed_kinds,
                                  allow_comments=False):
        return True

    # Nothing specific enough to gate on: let the model decide.
    if not usable and not any(_looks_like_path(item) for item in raw):
        return True
    return False


def attach_evidence(repo: RepoData, claim_list: list[dict]) -> list[dict]:
    """Return copies of the claims with their evidence attached.

    Also records whether the claim's specific terms were found at all
    (``strong_proof``): if they were not, the claim cannot be proved, however
    generous the model is.
    """
    enriched: list[dict] = []
    for claim in claim_list:
        item = dict(claim)
        keywords = claim.get("keywords") or []
        strong = claim.get("strong_keywords") or []
        allowed = _allowed_kinds(claim)
        # The fallback pass may also use the claim's own words and its ordinary
        # keywords, so a long or joined strong keyword cannot sink a real claim.
        fallback = list(keywords) + [claim.get("claim") or ""]
        item["strong_proof"] = strong_proof_found(
            repo, strong, allowed_kinds=allowed, fallback_keywords=fallback
        )
        item["evidence"] = find_evidence(
            repo, keywords, strong_keywords=strong, allowed_kinds=allowed,
            allow_comments=False,
        )
        enriched.append(item)
    return enriched


# ---------------------------------------------------------------------------
# Step 2: judging (batched Gemma calls)
# ---------------------------------------------------------------------------


def _format_claims_block(claim_list: list[dict], indexes: list[int]) -> str:
    blocks: list[str] = []
    for index in indexes:
        claim = claim_list[index]
        lines = [f"CLAIM {index} [{claim.get('type', 'feature')}]: {claim.get('claim')}"]
        strong = claim.get("strong_keywords") or []
        if strong:
            lines.append("STRONG TERMS: " + ", ".join(str(s) for s in strong))
        evidence = claim.get("evidence") or []
        if evidence:
            lines.append("EVIDENCE:")
            for hit in evidence:
                lines.append(f"- {hit['file']}:{hit['line']}: {hit['snippet']}")
        else:
            lines.append("EVIDENCE: (none found)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _normalise_supports(value: Any) -> Optional[str]:
    """Map a model's word for "how much does this prove" onto full/part/none."""
    text = str(value or "").strip().lower()
    if text in {"full", "verified", "verify", "yes", "true", "complete",
                "supports"}:
        return SUPPORTS_FULL
    if text in {"part", "partial", "partly", "some", "weak", "unsure"}:
        return SUPPORTS_PART
    if text in {"none", "no", "false", "unverified", "unsupported", "missing",
                "absent"}:
        return SUPPORTS_NONE
    return None


def _parse_judgements(payload: Any, valid_ids: set[int]) -> dict[int, dict]:
    """Map Gemma's judgements to ``{index: {"supports", "proof", "reason"}}``.

    An item without a usable ``supports`` value is dropped, so the claim counts
    as unjudged instead of being guessed at.
    """
    if isinstance(payload, dict):
        raw = None
        for key in ("judgements", "judgments", "results", "claims", "items"):
            if key in payload:          # explicit check so `[]` is honoured
                raw = payload[key]
                break
    else:
        raw = payload
    if not isinstance(raw, list):
        raise GemmaResponseError(
            "Gemma did not return a list of judgements. Please try again."
        )

    out: dict[int, dict] = {}
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("id", item.get("index", position)))
        except (TypeError, ValueError):
            index = position
        supports = _normalise_supports(item.get("supports") or item.get("status"))
        if supports is None or index not in valid_ids or index in out:
            continue
        proof = item.get("proof")
        if proof is None:
            proof = item.get("evidence")
        reason = str(item.get("reason") or "").strip()
        out[index] = {"supports": supports, "proof": proof,
                      "reason": reason[:300]}
    return out


def _same_file(left: str, right: str) -> bool:
    a = (left or "").replace("\\", "/").strip().lower()
    b = (right or "").replace("\\", "/").strip().lower()
    if not a or not b:
        return False
    return a == b or a.rsplit("/", 1)[-1] == b.rsplit("/", 1)[-1]


def _proof_in_evidence(proof: Any, evidence: list[dict],
                       *, allow_bare: bool = True) -> bool:
    """True when ``proof`` really points at one of the evidence lines.

    Accepts ``file:line`` (the requested form), a bare file name that is in the
    evidence, and a little surrounding wording, because a model may phrase it
    slightly differently.
    """
    text = str(proof or "").strip().strip("`\"'").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"none", "null", "no", "n/a", "nil", "-", "unknown"}:
        return False

    pairs = [(match.group(1), int(match.group(2)))
             for match in _PROOF_RE.finditer(text)]
    for wanted_file, wanted_line in pairs:
        for hit in evidence:
            if not _same_file(wanted_file, str(hit.get("file") or "")):
                continue
            if int(hit.get("line") or -1) == wanted_line:
                return True
    if pairs or not allow_bare:
        return False
    for hit in evidence:
        name = str(hit.get("file") or "")
        if not name:
            continue
        if name.lower() in lowered or name.rsplit("/", 1)[-1].lower() in lowered:
            return True
    return False


def _evidence_hit_for(proof: Any, evidence: list[dict]) -> Optional[dict]:
    """The evidence hit a ``file:line`` proof points at, or the best match."""
    text = str(proof or "").strip().strip("`\"'").strip()
    if not text or not evidence:
        return None
    lowered = text.lower()
    for match in _PROOF_RE.finditer(text):
        wanted_file, wanted_line = match.group(1), int(match.group(2))
        for hit in evidence:
            if (_same_file(wanted_file, str(hit.get("file") or ""))
                    and int(hit.get("line") or -1) == wanted_line):
                return hit
    for hit in evidence:
        name = str(hit.get("file") or "")
        if name and (name.lower() in lowered
                     or name.rsplit("/", 1)[-1].lower() in lowered):
            return hit
    return None


def _version_downgrade(
    claim: dict,
    evidence: list[dict],
    explicit: Any,
    judgement: dict,
) -> Optional[tuple[str, str, list[dict]]]:
    """Downgrade "full" to "Partial" when the code shows another version.

    Only applies when the claim names a version *and* the proof line is about the
    claim's own strong term (so an unrelated number in the code never triggers
    this). A different major.minor is reported in the reason; the same one, or no
    visible version at all, is a plain name match.
    """
    strong = claim.get("strong_keywords") or []
    if not strong:
        return None
    wanted = keywords_module.extract_version(claim.get("claim") or "")
    if not wanted:
        return None
    proof = explicit if explicit is not None else judgement.get("reason")
    hit = _evidence_hit_for(proof, evidence)
    if hit is None:
        return None
    snippet = str(hit.get("snippet") or "")
    terms = keywords_module.searchable_strong_keywords(strong)
    if terms and not any(keywords_module.score_line(snippet, t) for _kw, t in terms):
        return None
    found = keywords_module.extract_version(snippet)
    if not found or found == wanted:
        return None
    reason = (f"The code uses version {found[0]}.{found[1]}, not "
              f"{wanted[0]}.{wanted[1]}.")
    return "Partial", reason, evidence


def _default_reason(status: str, evidence: list[dict]) -> str:
    if status == "Unverified" or not evidence:
        return NO_PROOF_REASON
    best = evidence[0]
    verb = "Proved by" if status == "Verified" else "Supported by"
    return f"{verb} {best['file']}:{best['line']}."


def _consistent_reason(reason: str, evidence: list[dict], status: str) -> str:
    """A reason may only ever name files that appear in the claim's evidence."""
    text = (reason or "").strip()
    if not text:
        return _default_reason(status, evidence)
    if status == "Unverified" and _proof_in_evidence(text, evidence,
                                                     allow_bare=False):
        # A "nothing proves this" verdict may not quote a proof.
        return NO_PROOF_REASON
    named = _REASON_PATH_RE.findall(text)
    if not named:
        return text
    known = {str(hit.get("file") or "") for hit in evidence}
    known_names = {name.rsplit("/", 1)[-1] for name in known if name}
    for path in named:
        if path in known or path.rsplit("/", 1)[-1] in known_names:
            continue
        # The reason points at a file we have no evidence from: replace it with
        # a reason built only from evidence we actually have.
        return _default_reason(status, evidence)
    return text


def _decide(claim: dict, judgement: dict) -> tuple[str, str, list[dict]]:
    """Final status, reason and evidence for one claim.

    The model proposes; the evidence rules decide.
    """
    evidence = claim.get("evidence") or []
    if not claim.get("strong_proof", True):
        return "Unverified", NO_PROOF_REASON, []
    if not evidence:
        return "Unverified", NO_PROOF_REASON, []

    supports = _normalise_supports(judgement.get("supports"))

    # The explicit proof, when the model gave one.
    explicit = judgement.get("proof")
    if explicit is not None and str(explicit).strip().lower() in {"", "none",
                                                                  "null"}:
        explicit = None
    explicit_ok: Optional[bool] = None
    if explicit is not None:
        explicit_ok = _proof_in_evidence(explicit, evidence)

    # An older style of reply put the location in the reason instead.
    location_ok = explicit_ok
    if location_ok is None:
        location_ok = _proof_in_evidence(judgement.get("reason"), evidence)

    if supports == SUPPORTS_NONE:
        return "Unverified", _consistent_reason(
            judgement.get("reason", ""), evidence, "Unverified"), []
    if supports == SUPPORTS_FULL:
        if location_ok:
            downgraded = _version_downgrade(claim, evidence, explicit, judgement)
            if downgraded is not None:
                return downgraded
            return "Verified", _consistent_reason(
                judgement.get("reason", ""), evidence, "Verified"), evidence
        # "full" with a proof that does not point at the evidence is no proof.
        return "Unverified", NO_PROOF_REASON, []
    if supports == SUPPORTS_PART:
        if explicit_ok is False:
            # The model pointed at a location that is not in the evidence.
            return "Unverified", NO_PROOF_REASON, []
        return "Partial", _consistent_reason(
            judgement.get("reason", ""), evidence, "Partial"), evidence
    return "Unverified", NO_PROOF_REASON, []


def _judge_batch(
    claim_list: list[dict],
    indexes: list[int],
    *,
    session=None,
) -> dict[int, dict]:
    """Ask Gemma to judge one small batch of claims."""
    prompt = PROMPT_TEMPLATE.format(
        claims_block=_format_claims_block(claim_list, indexes)
    )
    # Small batches keep the reply short; the token budget is raised only if a
    # batch would ever need more room, so a reply can never be cut off.
    max_output_tokens = min(4096, max(config.GEMMA_MAX_OUTPUT_TOKENS,
                                      256 * max(1, len(indexes))))
    payload = generate_json(
        prompt, session=session,
        max_output_tokens=max_output_tokens,
        response_schema=JUDGEMENTS_SCHEMA,
    )
    return _parse_judgements(payload, set(indexes))


def _judge_all(claim_list: list[dict], *, session=None) -> dict[int, dict]:
    """Judge every claim, in batches, plus one retry for missing claims.

    Claims are sent ``config.MAX_JUDGE_BATCH`` at a time so a long reply can
    never be truncated. Afterwards, one extra call is made for the claims the
    model left out. Anything still missing is reported as ``Not judged`` rather
    than guessed.
    """
    budget = max(1, config.MAX_GEMMA_CALLS - 1)
    indexes = list(range(len(claim_list)))
    judgements: dict[int, dict] = {}
    calls = 0

    for start in range(0, len(indexes), config.MAX_JUDGE_BATCH):
        if calls >= budget:
            break
        batch = indexes[start:start + config.MAX_JUDGE_BATCH]
        calls += 1
        judgements.update(_judge_batch(claim_list, batch, session=session))

    missing = [index for index in indexes if index not in judgements]
    if missing and calls < budget:
        # ONE extra call, with only the claims the model left out.
        calls += 1
        judgements.update(
            _judge_batch(claim_list, missing[:config.MAX_JUDGE_BATCH],
                         session=session)
        )
    return judgements


def _matches_strong(hit: dict, strong: list[str]) -> bool:
    """True when an evidence line contains one of a claim's strong terms."""
    terms = keywords_module.searchable_strong_keywords(strong)
    if not terms:
        terms = keywords_module.searchable_keywords(strong)
    snippet = str(hit.get("snippet") or "")
    return any(keywords_module.score_line(snippet, t) for _kw, t in terms)


def _repair_consistent_evidence(
    claims: list[dict],
    judgements: dict[int, dict],
    *,
    session=None,
) -> dict[int, dict]:
    """Attach proof lines that another claim already used, then re-judge.

    If a claim came back ``Unverified`` with no evidence, but a line that is
    already proof for *another* claim contains one of this claim's strong
    keywords, that line is real evidence for this claim too. It is attached and
    the affected claims are sent to the judge once more; the judge still decides
    full, part or none.
    """
    if not judgements:
        return judgements

    pool: list[dict] = []
    for index, claim in enumerate(claims):
        judgement = judgements.get(index)
        if judgement is None:
            continue
        status, _reason, evidence = _decide(claim, judgement)
        if status in ("Verified", "Partial") and evidence:
            pool.extend(evidence)
    if not pool:
        return judgements

    rescue: list[int] = []
    for index, claim in enumerate(claims):
        judgement = judgements.get(index)
        if judgement is None:
            continue
        status, _reason, evidence = _decide(claim, judgement)
        if status != "Unverified" or evidence:
            continue
        strong = claim.get("strong_keywords") or []
        if not strong:
            continue
        matched: list[dict] = []
        seen: set[tuple] = set()
        for hit in pool:
            key = (hit.get("file"), hit.get("line"))
            if key in seen:
                continue
            if _matches_strong(hit, strong):
                seen.add(key)
                matched.append(dict(hit))
            if len(matched) >= config.MAX_EVIDENCE_PER_CLAIM:
                break
        if matched:
            claim["evidence"] = matched
            claim["strong_proof"] = True
            rescue.append(index)

    if not rescue:
        return judgements

    batch = rescue[:config.MAX_JUDGE_BATCH]
    judgements.update(_judge_batch(claims, batch, session=session))
    return judgements


def verify_claims(
    repo: RepoData,
    claim_list: list[dict],
    *,
    session=None,
    use_cache: bool = True,
) -> list[dict]:
    """Return the claims enriched with evidence, status and reason."""
    claims_with_evidence = attach_evidence(repo, claim_list)

    if not claims_with_evidence:
        return []

    key = cache.make_key(
        "verify",
        repo.url,
        repo.last_commit_sha or "",
        "|".join(c.get("claim", "") for c in claims_with_evidence),
    )
    if use_cache:
        cached = cache.get(key)
        if cached and isinstance(cached.get("claims"), list):
            return cached["claims"]

    judgements = _judge_all(claims_with_evidence, session=session)
    judgements = _repair_consistent_evidence(
        claims_with_evidence, judgements, session=session
    )

    results: list[dict] = []
    for index, claim in enumerate(claims_with_evidence):
        judgement = judgements.get(index)
        if judgement is None:
            status, reason, evidence = UNJUDGED_STATUS, UNJUDGED_REASON, []
        else:
            status, reason, evidence = _decide(claim, judgement)
        item = {**claim, "status": status, "reason": reason, "evidence": evidence}
        item.pop("strong_proof", None)
        results.append(item)

    if use_cache:
        cache.set(key, {"claims": results})

    return results
