"""Generic keyword matching and claim-similarity helpers.

This module is deliberately repo-agnostic: nothing in here knows about any
particular project, file name or library. It works for any public Python or
JavaScript/TypeScript repository.

Two jobs:

* :func:`match_terms` / :func:`score_line` - search a line of source text for a
  claim keyword, trying simple variants (for example a library written with a
  file extension such as ``foo.js`` is also matched as ``foo``, which covers
  ``new Foo(`` and ``foo(``).
* :func:`significant_tokens` / :func:`is_duplicate` - decide whether two claims
  mean the same thing, so the same claim is not reported twice.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

# Plain English function words: no signal at all.
STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "these", "those", "are",
    "was", "were", "has", "have", "had", "does", "did", "can", "could",
    "should", "must", "from", "into", "onto", "out", "over", "under", "your",
    "you", "its", "it", "is", "be", "been", "being", "of", "on", "in", "at",
    "to", "by", "as", "an", "or", "not", "no", "but", "if", "then", "than",
    "so", "such", "when", "which", "who", "whom", "whose", "will", "would",
    "may", "might", "shall", "about", "across", "after", "before", "between",
    "during", "through", "up", "down", "more", "most", "other", "some", "any",
    "all", "each", "both", "few", "many", "much", "very", "also", "only",
    "just", "own", "same", "too", "via", "per", "and/or", "etc", "eg", "ie",
})

# Broad software words. They are real words, but too common to prove that two
# claims are about the same thing.
GENERIC_TERMS = frozenset({
    "project", "code", "repo", "repository", "app", "application", "web",
    "website", "site", "tool", "library", "package", "module", "file", "files",
    "folder", "directory", "data", "api", "apis", "http", "https", "url",
    "json", "xml", "html", "css", "js", "ts", "py", "ui", "ux", "cli",
    "command", "line", "interface", "feature", "features", "support",
    "supports", "supported", "use", "uses", "used", "using", "make", "makes",
    "made", "build", "builds", "built", "add", "adds", "added", "run", "runs",
    "running", "get", "gets", "set", "sets", "new", "list", "lists", "main",
    "core", "page", "pages", "screen", "screens", "user", "users", "client",
    "server", "backend", "frontend", "full", "stack", "test", "tests",
    "mobile", "mobiles", "desktop", "desktops",
    "testing", "version", "update", "updates", "create", "creates", "show",
    "shows", "display", "displays", "allow", "allows", "provide", "provides",
    "enable", "enables", "include", "includes", "work", "works", "working",
    "simple", "small", "tiny", "easy", "fast", "quick", "basic", "general",
    "custom", "config", "settings", "environment", "error", "errors", "handle",
    "handles", "handling", "process", "processes", "output", "input", "text",
    "string", "number", "numbers", "value", "values", "size", "type", "types",
    "name", "names", "path", "paths", "time", "start", "stop", "end", "open",
    "close", "read", "reads", "write", "writes", "load", "loads", "save",
    "saves", "send", "sends", "receive", "return", "call", "calls", "function",
    "method", "class", "object", "array", "claim", "claims", "thing", "things",
    "based", "into", "first", "second", "third", "one", "two", "three",
    "check", "checks", "verify", "verifies", "ensure", "ensures", "confirm",
    "confirms", "please", "using",
})

# Shortest keyword we will search for on its own.
MIN_TERM_LEN = 3

# Only a reasonably long shared word is distinctive enough to prove that two
# claims are about the same thing.
MIN_DISTINCTIVE_LEN = 5

# A line that looks like code is stronger evidence than prose.
CODE_HINT_RE = re.compile(
    r"(?i)"
    r"(^\s*(?:import|from|require|new|const|let|var|using|\#include|\$|npm|pip)\b)"
    r"|(<script\b)"
    r"|(src\s*=\s*['\"])"
    r"|(=\s*[^=])"
    r"|(\w+\s*\([^)]*\))"
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.\-]*")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS_RE = re.compile(r"[._\-/\\\s:]+")
# Split a joined name into its parts: on camelCase, on separators and on a
# digit/letter change. ``"html5canvas"`` -> ``html``, ``5``, ``canvas`` and
# ``"webSpeech"`` -> ``web``, ``Speech``.
_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|"
                          r"(?<=\d)(?=[A-Za-z])")
# Short suffixes a library or API name commonly carries, which the code may
# leave out (``chart.js`` is often just ``Chart`` in the code).
_NAME_SUFFIXES = ("js", "api")
# Build tags a CDN or bundle file name puts after the library name
# (``chart.umd``, ``font-awesome.min``). They are not part of the name.
_BUILD_TAGS = frozenset({"umd", "esm", "cjs", "iife", "min", "bundle", "dist",
                         "module", "map"})
# How many generic words may be peeled off the ends of a joined strong keyword
# while looking for the meaningful part (``webspeechapi`` -> ``speech``).
_MAX_PEELS = 4

# A version number attached to a name: ``4.4``, ``4.4.1``, ``v3``, ``3.x``.
# A plain trailing number is deliberately left alone (``html5`` is a name).
_VERSION_RE = re.compile(r"(?i)(?:\bv?\d+(?:\.(?:\d+|x))+|\bv\d+\b)")
# Two strings match when their letters-and-digits form is identical: this one
# rule covers "font awesome", "font-awesome", "font_awesome" and "fontawesome".
_COMPACT_RE = re.compile(r"[^a-z0-9]+")
# Shortest compacted (letters+digits only) term worth matching without word
# boundaries, so the rule cannot fire on a tiny fragment.
MIN_COMPACT_LEN = 5

# Ordered patterns for reading the first version out of a claim or a line. The
# CDN spellings (``@4.4.1``, ``/6.4.0/``) come first so a URL is read correctly.
_VERSION_CAPTURE = (
    re.compile(r"@v?(\d+)\.(\d+)"),
    re.compile(r"\bv(\d+)\.(\d+)", re.IGNORECASE),
    re.compile(r"\b(\d+)\.(\d+)\b"),
    re.compile(r"\bv(\d+)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\.x\b", re.IGNORECASE),
)


def _compact(text: str) -> str:
    """Lowercase letters and digits only, so separators stop mattering."""
    return _COMPACT_RE.sub("", (text or "").lower())


def strip_version(text: str) -> str:
    """Return ``text`` with any version numbers removed.

    ``"Chart.js 4.4"`` -> ``"Chart.js"``, ``"FontAwesome v3"`` -> ``"FontAwesome"``.
    """
    if not text:
        return text or ""
    cleaned = _VERSION_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    if cleaned.endswith("@"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def extract_version(text: str):
    """Return ``(major, minor)`` for the first version in ``text``, or ``None``.

    Handles plain versions (``6.4``), ``v``-prefixed ones (``v3``) and the
    ``x`` form (``3.x``), as well as CDN spellings (``@4.4.1``, ``/6.4.0/``).
    """
    for pattern in _VERSION_CAPTURE:
        match = pattern.search(text or "")
        if match:
            minor = int(match.group(2)) if (match.lastindex or 0) >= 2 else 0
            return (int(match.group(1)), minor)
    return None


def _words(text: str) -> list[str]:
    """Lowercase word tokens, splitting camelCase at the same time."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        for part in _CAMEL_SPLIT_RE.split(raw):
            part = part.strip("._-+")
            if part:
                out.append(part.lower())
    return out


def match_terms(keyword: str) -> list[str]:
    """Lowercase substrings to search for, most specific first.

    ``"Chart.js"`` becomes ``["chart.js", "chart", "chartjs"]`` so that a file
    that writes ``new Chart(...)`` is still found. Any version number is removed
    first (``"Chart.js 4.4"`` -> ``"chart.js"``). Very short terms are dropped so
    we never match on noise.
    """
    keyword = strip_version((keyword or "").strip().lower())
    if not keyword:
        return []
    terms: list[str] = []

    def add(term: str) -> None:
        if term and term not in terms and len(term) >= MIN_TERM_LEN:
            terms.append(term)

    add(keyword)
    # The leading part of a dotted / dashed / underscored name.
    base = _SEPARATORS_RE.split(keyword)[0]
    add(base)
    # The name with separators removed entirely.
    add(_SEPARATORS_RE.sub("", keyword))
    return terms


def _is_searchable(keyword: str) -> bool:
    keyword = (keyword or "").strip().lower()
    if len(keyword) < MIN_TERM_LEN:
        return False
    return keyword not in STOPWORDS


def searchable_keywords(keywords: list[str]) -> list[tuple[str, list[str]]]:
    """``(keyword, terms)`` pairs for the keywords worth searching for."""
    out: list[tuple[str, list[str]]] = []
    for keyword in keywords or []:
        if not _is_searchable(keyword):
            continue
        terms = match_terms(keyword)
        if terms:
            out.append((keyword.strip().lower(), terms))
    return out


def split_compound(name: str) -> list[str]:
    """Split a joined name into lowercase parts.

    Handles camelCase, spaces, dots, dashes, underscores, slashes and a
    letter/digit boundary, so ``"Chart.js"`` -> ``["chart", "js"]`` and
    ``"html5canvas"`` -> ``["html", "5", "canvas"]``.
    """
    parts: list[str] = []
    for chunk in _SEPARATORS_RE.split(name or ""):
        for piece in _BOUNDARY_RE.split(chunk):
            piece = piece.strip("._-+")
            if piece:
                parts.append(piece.lower())
    return parts


def strong_match_terms(keyword: str) -> list[str]:
    """Search terms for a *strong* keyword, with many tolerant variants.

    A strong keyword is the specific thing that would have to appear in the code
    if the claim were true, but models often write it as a joined blob, with a
    file extension, with a plural, or with a redundant ``js``/``api`` suffix.
    This expands one keyword into the literal forms worth searching for, most
    specific first: the whole name, the name with separators removed, each part
    (and its plural/singular), the name without a ``js``/``api`` suffix, and the
    name with generic words peeled off its ends.
    """
    kw = strip_version(re.sub(r"\s+", " ", (keyword or "").strip())).lower()
    kw = kw.strip("`'\"()[]{}.,;:!?")
    if not kw:
        return []

    terms: list[str] = []

    def add(term: str) -> None:
        term = (term or "").strip("._-+ ").lower()
        if not term or len(term) < MIN_TERM_LEN:
            return
        if term in STOPWORDS or term in GENERIC_TERMS:
            return
        if term not in terms:
            terms.append(term)

    add(kw)
    compact = _compact(kw)
    add(compact)

    parts = [part for part in split_compound(kw) if part]
    # Build tags (``umd``, ``min``) and redundant suffixes (``js``) are not part
    # of the name, so they never become match terms on their own.
    name_parts = [part for part in parts
                  if part not in _BUILD_TAGS and part not in _NAME_SUFFIXES]
    if not name_parts:
        name_parts = parts

    # Every joined spelling of a two-word name ("font awesome" ->
    # "fontawesome", "font-awesome", "font_awesome", "font awesome"). These come
    # before the single words so a line with both words ranks first; a line with
    # either word alone still matches.
    if len(name_parts) >= 2:
        add("".join(name_parts))
        add("-".join(name_parts))
        add("_".join(name_parts))
        add(" ".join(name_parts))

    for part in name_parts:
        add(part)
        if part.endswith("s") and len(part) > MIN_TERM_LEN:
            add(part[:-1])
        else:
            add(part + "s")

    for suffix in _NAME_SUFFIXES:
        if kw.endswith(suffix) and len(kw) > len(suffix) + 1:
            add(kw[:-len(suffix)])
            add(compact[:-len(suffix)])

    # Peel generic words off the ends (``webspeechapi`` -> ``speech``).
    peeled = compact
    for _ in range(_MAX_PEELS):
        changed = False
        for generic in GENERIC_TERMS:
            if len(peeled) > len(generic) + 1 and peeled.startswith(generic):
                peeled = peeled[len(generic):]
                changed = True
            if len(peeled) > len(generic) + 1 and peeled.endswith(generic):
                peeled = peeled[:-len(generic)]
                changed = True
        if not changed:
            break
    if peeled and peeled != compact:
        add(peeled)

    return terms


def searchable_strong_keywords(
    keywords: list[str],
) -> list[tuple[str, list[str]]]:
    """``(keyword, terms)`` pairs for the strong keywords worth searching for."""
    out: list[tuple[str, list[str]]] = []
    for keyword in keywords or []:
        terms = strong_match_terms(keyword)
        if terms:
            out.append((str(keyword).strip().lower(), terms))
    return out


def score_line(line: str, terms: list[str]) -> int:
    """Relevance of one line for a keyword's terms. ``0`` means "no match"."""
    lowered = (line or "").lower()
    if not lowered:
        return 0
    best = 0
    for position, term in enumerate(terms):
        if term not in lowered:
            continue
        score = 10 if position == 0 else 6
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered):
            score += 3
        score -= min(position, 3)
        best = max(best, score)
    if best == 0:
        # Separator-insensitive match: "font awesome", "font-awesome" and
        # "fontawesome" all reduce to the same letters-and-digits form.
        compact_line = _compact(lowered)
        if compact_line:
            for term in terms:
                compact_term = _compact(term)
                if len(compact_term) >= MIN_COMPACT_LEN and compact_term in compact_line:
                    best = 8
                    break
    if best and CODE_HINT_RE.search(line):
        best += 2
    return best


# ---------------------------------------------------------------------------
# Claim similarity
# ---------------------------------------------------------------------------


def significant_tokens(*texts) -> set[str]:
    """Tokens that carry meaning, ignoring stopwords and broad software words."""
    out: set[str] = set()
    for text in texts:
        if isinstance(text, str):
            words = _words(text)
        else:
            words = []
            for item in text or []:
                words.extend(_words(str(item)))
        for word in words:
            if len(word) < MIN_TERM_LEN:
                continue
            if word in STOPWORDS or word in GENERIC_TERMS:
                continue
            out.add(word)
    return out


def normalise_text(text: str) -> str:
    """Lowercase letters and digits only, for comparing two texts exactly."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def specific_terms(values) -> list[str]:
    """The terms in ``values`` that are specific enough to prove a claim.

    Stop words and broad software words (``test``, ``data``, ``api``, ``server``
    and friends) are dropped: they are far too common to show that a claim about
    a library, service, data store or platform is really implemented. Short or
    empty terms are dropped too, so the result only holds words that a specific
    claim would have to put in the code somewhere.
    """
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for value in values or []:
        for word in _words(str(value)):
            if len(word) < MIN_DISTINCTIVE_LEN:
                continue
            if word in STOPWORDS or word in GENERIC_TERMS:
                continue
            if word not in out:
                out.append(word)
    return out


def is_duplicate(left: set[str], right: set[str],
                 document_frequency: dict[str, int]) -> bool:
    """True when two claims are clearly about the same subject.

    ``left``/``right`` are :func:`significant_tokens` sets and
    ``document_frequency`` counts how many claims each token appears in, so a
    term shared by only these claims counts as distinctive. Only reasonably
    long terms (``MIN_DISTINCTIVE_LEN`` or more) may trigger a merge, so short
    filler words can never collapse unrelated claims into one.
    """
    if not left or not right:
        return False
    shared = left & right
    if not shared:
        return False
    strong = {term for term in shared
              if len(term) >= MIN_DISTINCTIVE_LEN
              and document_frequency.get(term, 0) <= 2}
    if not strong:
        return False
    smaller = min(len(left), len(right))
    return len(shared) / smaller >= 0.5


def merge_keywords(first: list[str], second: list[str], limit: int = 6) -> list[str]:
    """Union of two keyword lists, keeping order and dropping duplicates."""
    out: list[str] = []
    for keyword in list(first or []) + list(second or []):
        keyword = str(keyword).strip().lower()
        if keyword and keyword not in out:
            out.append(keyword)
    return out[:limit]
