"""Code normalization, query-token classification and embedding-text composition.

Two canonical forms for every code (article / SKU / product code / barcode):

* ``norm_code`` — reversible-ish canonical form: NFKC, uppercase, separators stripped,
  Cyrillic homoglyphs mapped to Latin. Stored in Qdrant payload (`*_norm`) and used as
  the RapidFuzz corpus.
* ``skeleton`` — lossy OCR/visual folding (O→0, I→1, …) applied on top of ``norm_code``.
  Never stored; lives only in the in-memory CodeIndex as a secondary exact-match key.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.product import ProductIn

# Uppercase Cyrillic letters visually identical to Latin ones (uk/ru keyboards & OCR).
CYR_TO_LAT = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "Ё": "E",
        "І": "I",
        "Ї": "I",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)

# Lossy folding of characters commonly confused visually / in OCR.
OCR_FOLD = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "З": "3",  # Cyrillic Ze survives norm_code untranslated
        "S": "5",
        "G": "6",
        "B": "8",
    }
)

_SEPARATORS_RE = re.compile(r"[\s\-_./\\,:;#()\[\]]+")

# Measurement tokens ("18V", "5Ah", "500ГБ") must never be treated as product codes.
_UNIT_RE = re.compile(
    r"^\d+([.,]\d+)?("
    r"V|W|KW|A|AH|MAH|WH|KWH|HZ|KHZ|MHZ|GHZ|RPM|"
    r"MM|CM|M|KM|ML|L|G|KG|T|GB|TB|MB|PCS|"
    r"В|ВТ|КВТ|А|АЧ|МАЧ|ГЦ|КГЦ|МГЦ|ГГЦ|ОБ|"
    r"ММ|СМ|М|КМ|МЛ|Л|Г|КГ|Т|ГБ|ТБ|МБ|ШТ"
    r")$"
)

_CODE_CHARS_RE = re.compile(r"[A-Z0-9]+")
_LETTERS_THEN_DIGITS_RE = re.compile(r"[A-Z]{1,4}\d{3,}[A-Z0-9]*")


def norm_code(value: str) -> str:
    s = unicodedata.normalize("NFKC", value).upper()
    s = _SEPARATORS_RE.sub("", s)
    return s.translate(CYR_TO_LAT)


def skeleton(normed: str) -> str:
    return normed.translate(OCR_FOLD)


def _compact(token: str) -> str:
    return _SEPARATORS_RE.sub("", unicodedata.normalize("NFKC", token).upper())


def is_ean_like(token: str) -> bool:
    compact = _compact(token)
    return compact.isdigit() and 12 <= len(compact) <= 14


def is_code_like(token: str) -> bool:
    """Heuristic: does this query token look like an article / product code / barcode?"""
    compact = _compact(token)
    if len(compact) < 4:
        return False
    if _UNIT_RE.match(compact):
        return False
    latin = compact.translate(CYR_TO_LAT)
    if not _CODE_CHARS_RE.fullmatch(latin):
        return False
    digits = sum(c.isdigit() for c in latin)
    if digits == 0:
        return False
    if digits / len(latin) >= 0.4:
        return True
    # letter<->digit alternations catch codes like "GSB13RE" with low digit ratio
    transitions = sum(
        1 for a, b in zip(latin, latin[1:], strict=False) if a.isdigit() != b.isdigit()
    )
    if transitions >= 2:
        return True
    return bool(_LETTERS_THEN_DIGITS_RE.fullmatch(latin))


def tokenize_query(query: str) -> list[str]:
    return [t for t in query.split() if t]


def compose_dense_text(p: ProductIn) -> str:
    """Semantic embedding text. Codes/SKU/EAN are deliberately excluded so that
    alphanumeric noise does not pollute the dense vector."""
    parts: list[str] = [p.name.strip()]
    if p.brand:
        parts.append(f"Бренд: {p.brand.strip()}.")
    if p.category:
        parts.append(f"Категорія: {p.category.strip()}.")
    if p.attributes:
        attrs = "; ".join(f"{k}: {v}" for k, v in p.attributes.items())
        parts.append(attrs + ".")
    if p.description:
        parts.append(" ".join(p.description.split())[:800])
    return " ".join(parts)


def compose_sparse_text(p: ProductIn) -> str:
    """BM25 text: semantic text PLUS all code fields (raw and normalized) so the
    hybrid branch gets lexical exact-match power on codes for free."""
    codes: list[str] = []
    for value in (p.article, p.product_code, p.ean13):
        if value:
            codes.append(str(value))
            normed = norm_code(str(value))
            if normed and normed != str(value):
                codes.append(normed)
    return " ".join([compose_dense_text(p), *codes]).strip()


def compose_sparse_query(query: str, code_tokens: list[str]) -> str:
    """Query-side sparse text: raw query plus normalized variants of detected codes."""
    extra = [norm_code(t) for t in code_tokens]
    return " ".join([query, *[e for e in extra if e]])
