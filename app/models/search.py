from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

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


class SearchFilters(BaseModel):
    brand: str | None = None
    category: str | None = None
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    in_stock: bool | None = None
    attributes: dict[str, AttrValue] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    rerank: bool = False
    filters: SearchFilters | None = None


class MatchExplanation(BaseModel):
    branch: MatchBranch
    matched_field: str | None = None
    code_score: float | None = None
    reranked: bool = False


class SearchHit(BaseModel):
    product: dict[str, Any]
    score: float
    match: MatchExplanation


class SearchResponse(BaseModel):
    query_kind: QueryKind
    took_ms: float
    total: int
    items: list[SearchHit]
