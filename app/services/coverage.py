"""Query-term coverage: which of the requested characteristics does a product contain?

Vector search always returns the *nearest* products, even when nothing in the catalog
satisfies every requested characteristic (query "мат плата з hdmi на 1200" against a
catalog that has LGA 1200 boards but none with HDMI). Coverage makes that distinction
explicit: a hit is *confident* only when every significant query term is present in the
product's own fields; everything else is an *alternative* with `missing_terms` naming
exactly what the product lacks. The `strict` search mode splits the response on this.

Matching is a deliberate heuristic (unit-tested, no morphology dependency):
* numeric tokens match on digit boundaries — "1200" hits "LGA 1200"/"LGA1200", not "12000";
* alphabetic tokens match any word they prefix — "мат" covers "материнська";
* tokens of ≥5 chars also match with the last char dropped — "плата" covers "плати" (inflection).
"""

from __future__ import annotations

import re
from typing import Any

# Query words that carry no product semantics (uk / ru / en prepositions & conjunctions).
STOPWORDS = frozenset(
    {
        # uk
        "з",
        "зі",
        "із",
        "на",
        "для",
        "та",
        "і",
        "й",
        "до",
        "у",
        "в",
        "по",
        "без",
        "при",
        "або",
        "чи",
        "не",
        "є",
        "як",
        "що",
        # ru
        "с",
        "со",
        "и",
        "к",
        "от",
        "о",
        "об",
        "под",
        "над",
        "или",
        "есть",
        # en
        "with",
        "for",
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "by",
    }
)

MIN_STEM_LEN = 5  # tokens at least this long also match with the final char dropped

_QUERY_TOKEN_RE = re.compile(r"[a-zа-яёіїєґ0-9]+", re.IGNORECASE)
_HAYSTACK_WORD_RE = re.compile(r"[a-zа-яёіїєґ0-9]+", re.IGNORECASE)

# Payload fields whose text represents the product for coverage purposes.
_TEXT_FIELDS = ("name", "brand", "category", "description", "article", "product_code", "ean13")


def significant_tokens(query: str) -> list[str]:
    """Lowercased query tokens that carry meaning: stopwords and 1-char scraps dropped."""
    tokens = _QUERY_TOKEN_RE.findall(query.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def _build_haystack(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for fld in _TEXT_FIELDS:
        value = payload.get(fld)
        if value:
            parts.append(str(value))
    attrs = payload.get("attributes") or {}
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            parts.append(str(key))
            parts.append(str(value))
    return " ".join(parts).lower()


def _token_covered(token: str, haystack: str, words: list[str]) -> bool:
    if token.isdigit():
        # digit-boundary match: "1200" must not be satisfied by "12000"
        return re.search(rf"(?<!\d){re.escape(token)}(?!\d)", haystack) is not None
    if any(w.startswith(token) for w in words):
        return True
    if len(token) >= MIN_STEM_LEN:
        stem = token[:-1]
        return any(w.startswith(stem) for w in words)
    return False


def coverage(tokens: list[str], payload: dict[str, Any]) -> tuple[float, list[str]]:
    """(covered_ratio, missing_tokens) of the significant query tokens in the product.

    An empty token list is vacuously fully covered (ratio 1.0).
    """
    if not tokens:
        return 1.0, []
    haystack = _build_haystack(payload)
    words = _HAYSTACK_WORD_RE.findall(haystack)
    missing = [t for t in tokens if not _token_covered(t, haystack, words)]
    return (len(tokens) - len(missing)) / len(tokens), missing
