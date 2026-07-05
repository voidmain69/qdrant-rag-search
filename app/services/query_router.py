"""Query classification and search orchestration.

Branches:
* CODE_ONLY — every token looks like an article/code/EAN. Exact-tier hits short-circuit;
  weak/fuzzy hits fall through to hybrid search with code hits pinned on top.
* TEXT — dense+sparse hybrid with server-side RRF fusion (+ optional cross-encoder rerank).
* MIXED — both branches run; exact-tier code hits are pinned first, the rest interleaved.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from fastapi import HTTPException

from app.core.config import Settings
from app.models.search import (
    MatchBranch,
    MatchExplanation,
    QueryKind,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services.code_index import CodeHit, CodeIndex
from app.services.embedding import EmbeddingService
from app.services.normalization import compose_sparse_query, is_code_like, tokenize_query
from app.services.qdrant import QdrantService, build_filter, payload_matches_filters
from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)

STRONG_CODE_SCORE = 0.95


@dataclass(frozen=True)
class QueryClassification:
    kind: QueryKind
    code_tokens: list[str]
    text: str


def classify(query: str) -> QueryClassification:
    tokens = tokenize_query(query)
    code_idx = {i for i, t in enumerate(tokens) if is_code_like(t)}
    code_tokens = [t for i, t in enumerate(tokens) if i in code_idx]
    text_tokens = [t for i, t in enumerate(tokens) if i not in code_idx]
    if not code_tokens:
        return QueryClassification(QueryKind.TEXT, [], query)
    if not text_tokens:
        return QueryClassification(QueryKind.CODE_ONLY, code_tokens, "")
    return QueryClassification(QueryKind.MIXED, code_tokens, " ".join(text_tokens))


class SearchService:
    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantService,
        embedder: EmbeddingService,
        code_index: CodeIndex,
        reranker: RerankerService | None,
    ):
        self.settings = settings
        self.qdrant = qdrant
        self.embedder = embedder
        self.code_index = code_index
        self.reranker = reranker

    async def search(self, req: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        cls = classify(req.query)

        code_hits = await self._match_codes(cls.code_tokens) if cls.code_tokens else []
        strong = [h for h in code_hits if h.score >= STRONG_CODE_SCORE]

        if cls.kind == QueryKind.CODE_ONLY and strong:
            # unambiguous code lookup — no vector search needed
            items = await self._code_hits_to_items(code_hits, req)
        else:
            hybrid_items = await self._hybrid_search(req, cls)
            code_items = await self._code_hits_to_items(code_hits, req)
            items = _merge(code_items, hybrid_items)

        total = len(items)
        items = items[req.offset : req.offset + req.limit]
        took_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(query_kind=cls.kind, took_ms=round(took_ms, 1), total=total, items=items)

    # --- code branch ---

    async def _match_codes(self, tokens: list[str]) -> list[CodeHit]:
        hits: list[CodeHit] = []
        for token in tokens:
            hits.extend(await asyncio.to_thread(self.code_index.match, token))
        # dedup by point, keep the best score
        best: dict[str, CodeHit] = {}
        for hit in hits:
            current = best.get(hit.point_id)
            if current is None or hit.score > current.score:
                best[hit.point_id] = hit
        return sorted(best.values(), key=lambda h: h.score, reverse=True)

    async def _code_hits_to_items(self, hits: list[CodeHit], req: SearchRequest) -> list[SearchHit]:
        if not hits:
            return []
        payloads = await self.qdrant.retrieve_payloads([h.point_id for h in hits])
        items: list[SearchHit] = []
        for hit in hits:
            payload = payloads.get(hit.point_id)
            if payload is None or not payload_matches_filters(payload, req.filters):
                continue
            items.append(
                SearchHit(
                    product=payload,
                    score=hit.score,
                    match=MatchExplanation(
                        branch=hit.branch, matched_field=hit.field, code_score=hit.score
                    ),
                )
            )
        return items

    # --- hybrid branch ---

    async def _hybrid_search(self, req: SearchRequest, cls: QueryClassification) -> list[SearchHit]:
        sparse_text = compose_sparse_query(req.query, cls.code_tokens)
        dense_vec, sparse_vec = await self.embedder.aembed_query(req.query, sparse_text)
        flt = build_filter(req.filters)
        fetch = max(self.settings.prefetch_limit, req.offset + req.limit)
        points = await self.qdrant.hybrid_query(dense_vec, sparse_vec, flt, limit=fetch)
        items = [
            SearchHit(
                product=p.payload or {},
                score=p.score,
                match=MatchExplanation(branch=MatchBranch.HYBRID),
            )
            for p in points
        ]
        if req.rerank:
            items = await self._rerank(req.query, items)
        return items

    async def _rerank(self, query: str, items: list[SearchHit]) -> list[SearchHit]:
        if self.reranker is None:
            raise HTTPException(
                status_code=400,
                detail="Reranking is disabled on this instance (set RERANK_ENABLED=true "
                "and install the 'rerank' dependency group).",
            )
        if not items:
            return items
        top = items[: self.settings.rerank_top_k]
        docs = [_rerank_text(hit.product) for hit in top]
        scores = await self.reranker.rerank(query, docs)
        for hit, score in zip(top, scores, strict=True):
            hit.score = float(score)
            hit.match.reranked = True
        top.sort(key=lambda h: h.score, reverse=True)
        return top + items[self.settings.rerank_top_k :]


def _rerank_text(payload: dict) -> str:
    parts = [str(payload.get("name") or "")]
    if payload.get("brand"):
        parts.append(str(payload["brand"]))
    if payload.get("category"):
        parts.append(str(payload["category"]))
    attrs = payload.get("attributes") or {}
    if attrs:
        parts.append("; ".join(f"{k}: {v}" for k, v in attrs.items()))
    if payload.get("description"):
        parts.append(str(payload["description"])[:400])
    return ". ".join(p for p in parts if p)


def _merge(code_items: list[SearchHit], hybrid_items: list[SearchHit]) -> list[SearchHit]:
    """Exact-tier code hits pinned first; fuzzy code hits interleaved with hybrid results."""
    seen: set[str] = set()
    out: list[SearchHit] = []

    def push(item: SearchHit) -> None:
        key = str(item.product.get("external_id") or id(item))
        if key not in seen:
            seen.add(key)
            out.append(item)

    pinned = [i for i in code_items if (i.match.code_score or 0) >= STRONG_CODE_SCORE]
    fuzzy = [i for i in code_items if (i.match.code_score or 0) < STRONG_CODE_SCORE]
    for item in pinned:
        push(item)
    fi, hi = 0, 0
    while fi < len(fuzzy) or hi < len(hybrid_items):
        if fi < len(fuzzy):
            push(fuzzy[fi])
            fi += 1
        if hi < len(hybrid_items):
            push(hybrid_items[hi])
            hi += 1
    return out
