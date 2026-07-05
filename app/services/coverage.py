"""Requirement coverage: which of the requested characteristics does a product contain?

Vector search always returns the *nearest* products, even when nothing in the catalog
satisfies every requested characteristic (query "мат плата з hdmi на 1200" against a
catalog that has LGA 1200 boards but none with HDMI). Coverage makes that distinction
explicit: a hit is *confident* only when every requirement of the query is present in
the product's own fields; everything else is an *alternative* with `missing_terms`
naming exactly what the product lacks. The `strict` search mode splits the response.

A :class:`Requirement` is one requested characteristic with its lexical *variants* —
alternative surface forms a product card could contain (synonyms, uk/ru/en translations,
abbreviations). Variants come from the LLM query-understanding service (dictionary-free,
see `query_understanding.py`); without it every significant query token becomes its own
single-variant requirement (:func:`fallback_requirements`).

Lexical matching of a single variant token is a deliberate heuristic (unit-tested,
no morphology dependency):
* numeric tokens match on digit boundaries — "1200" hits "LGA 1200"/"LGA1200", not "12000";
* alphabetic tokens match any word they prefix — "мат" covers "материнська";
* tokens of ≥5 chars also match with the last char dropped — "плата" covers "плати" (inflection).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Requirement:
    """One requested characteristic; covered when ANY variant is found in the product."""

    name: str  # human-readable label, reported in match.missing_terms
    variants: tuple[str, ...]  # lexical surface forms, each may be a multi-word phrase


def significant_tokens(query: str) -> list[str]:
    """Lowercased query tokens that carry meaning: stopwords and 1-char scraps dropped."""
    tokens = _QUERY_TOKEN_RE.findall(query.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def fallback_requirements(query: str) -> list[Requirement]:
    """Dictionary-free degradation: each significant token is its own requirement."""
    return [Requirement(name=t, variants=(t,)) for t in significant_tokens(query)]


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
    # ingest-time LLM enrichment (synonyms, spec summary, ...) counts as product text:
    # an alias generated from the card covers a query term with no query-time LLM cost
    enrichment = payload.get("enrichment") or {}
    if isinstance(enrichment, dict):
        for value in enrichment.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(v) for v in value)
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


def token_in_texts(token: str, texts: list[str]) -> bool:
    """Is the token lexically represented in any of the given texts? Used by the
    query-understanding guardrail to detect constraints the LLM dropped."""
    haystack = " ".join(texts).lower()
    words = _HAYSTACK_WORD_RE.findall(haystack)
    return _token_covered(token, haystack, words)


def _phrase_covered(phrase: str, haystack: str, words: list[str]) -> bool:
    """A variant phrase is covered when every one of its tokens is covered."""
    tokens = _QUERY_TOKEN_RE.findall(phrase.lower())
    return bool(tokens) and all(_token_covered(t, haystack, words) for t in tokens)


def requirements_coverage(
    requirements: list[Requirement], payload: dict[str, Any]
) -> tuple[float, list[str]]:
    """(covered_ratio, missing_requirement_names) for the product.

    An empty requirement list is vacuously fully covered (ratio 1.0).
    """
    if not requirements:
        return 1.0, []
    haystack = _build_haystack(payload)
    words = _HAYSTACK_WORD_RE.findall(haystack)
    missing = [
        r.name for r in requirements if not any(_phrase_covered(v, haystack, words) for v in r.variants)
    ]
    return (len(requirements) - len(missing)) / len(requirements), missing


def coverage(tokens: list[str], payload: dict[str, Any]) -> tuple[float, list[str]]:
    """Token-level coverage: each token is a single-variant requirement."""
    return requirements_coverage([Requirement(name=t, variants=(t,)) for t in tokens], payload)
