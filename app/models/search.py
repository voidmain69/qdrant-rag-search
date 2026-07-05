from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.product import AttrValue


class QueryKind(StrEnum):
    CODE_ONLY = "code_only"
    MIXED = "mixed"
    TEXT = "text"


class MatchBranch(StrEnum):
    EXACT = "exact"
    EXACT_NORMALIZED = "exact_normalized"
    EAN_CORRECTED = "ean_corrected"
    FUZZY = "fuzzy"
    HYBRID = "hybrid"


class SearchMode(StrEnum):
    RELAXED = "relaxed"  # current behavior: one ranked list, nearest-first
    STRICT = "strict"  # items = every query term covered; the rest go to `alternatives`


class SearchFilters(BaseModel):
    brand: str | None = None
    category: str | None = None
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    in_stock: bool | None = None
    attributes: dict[str, AttrValue] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    # a whitespace-only query must fail the min_length check, not reach the embedder
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    rerank: bool = False
    mode: SearchMode = SearchMode.RELAXED
    include_archived: bool = False  # archived products are hidden from search by default
    filters: SearchFilters | None = None


class MatchExplanation(BaseModel):
    branch: MatchBranch
    matched_field: str | None = None
    code_score: float | None = None
    reranked: bool = False
    # share of significant query terms found in the product (hybrid branch only)
    query_coverage: float | None = None
    # the query terms this product does NOT contain — why it is only an alternative
    missing_terms: list[str] | None = None


class SearchHit(BaseModel):
    product: dict[str, Any]
    score: float
    match: MatchExplanation


class SearchResponse(BaseModel):
    query_kind: QueryKind
    took_ms: float
    total: int
    items: list[SearchHit]
    # strict mode: near-miss products (ranked, capped at `limit`) with match.missing_terms
    # explaining what each one lacks; always [] in relaxed mode
    alternatives: list[SearchHit] = Field(default_factory=list)
