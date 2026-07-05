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
from prometheus_client import Counter, Histogram

from app.core.config import Settings
from app.models.search import (
    MatchBranch,
    MatchExplanation,
    QueryKind,
    SearchHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
)
from app.services.code_index import STRONG_CODE_SCORE, CodeHit, CodeIndex
from app.services.coverage import coverage, significant_tokens
from app.services.embedding import EmbeddingService
from app.services.normalization import compose_sparse_query, is_code_like, tokenize_query
from app.services.qdrant import QdrantService, build_filter, payload_matches_filters
from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)

# strict mode: a near-miss must still cover at least this share of the query terms
# to qualify as an alternative — vector neighbours below it are just noise
MIN_ALTERNATIVE_COVERAGE = 0.5

SEARCH_REQUESTS_TOTAL = Counter(
    "search_requests_total", "Search requests by query classification", ["query_kind"]
)
SEARCH_LATENCY_SECONDS = Histogram(
    "search_latency_seconds", "End-to-end search latency by query classification", ["query_kind"]
)


@dataclass(frozen=True)
class QueryClassification:
    kind: QueryKind
    code_tokens: list[str]
    text: str


def classify(query: str) -> QueryClassification:
    code_tokens: list[str] = []
    text_tokens: list[str] = []
    for token in tokenize_query(query):
        (code_tokens if is_code_like(token) else text_tokens).append(token)
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
        if req.rerank and self.reranker is None:
            # validated up-front so every query kind gets the same contract
            raise HTTPException(
                status_code=400,
                detail="Reranking is disabled on this instance (set RERANK_ENABLED=true "
                "and install the 'rerank' dependency group).",
            )
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

        _annotate_coverage(req.query, items)

        alternatives: list[SearchHit] = []
        if req.mode == SearchMode.STRICT:
            # confident hits keep the ranked order; near-misses become alternatives
            confident = [i for i in items if _is_confident(i)]
            alternatives = [i for i in items if not _is_confident(i) and _is_alternative(i)][: req.limit]
            items = confident

        total = len(items)
        items = items[req.offset : req.offset + req.limit]
        took_ms = round((time.perf_counter() - started) * 1000, 1)
        SEARCH_REQUESTS_TOTAL.labels(query_kind=cls.kind.value).inc()
        SEARCH_LATENCY_SECONDS.labels(query_kind=cls.kind.value).observe(took_ms / 1000)
        logger.info(
            "search completed",
            extra={
                "query_kind": cls.kind.value,
                "mode": req.mode.value,
                "took_ms": took_ms,
                "total": total,
                "alternatives": len(alternatives),
                "rerank": req.rerank,
            },
        )
        return SearchResponse(
            query_kind=cls.kind, took_ms=took_ms, total=total, items=items, alternatives=alternatives
        )

    # --- code branch ---

    async def _match_codes(self, tokens: list[str]) -> list[CodeHit]:
        # one thread hop for all tokens; CodeIndex.match is synchronous CPU-bound code
        def _match_all() -> list[CodeHit]:
            return [hit for token in tokens for hit in self.code_index.match(token)]

        hits = await asyncio.to_thread(_match_all)
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
                    match=MatchExplanation(branch=hit.branch, matched_field=hit.field, code_score=hit.score),
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
        assert self.reranker is not None  # guaranteed by the up-front check in search()
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


def _annotate_coverage(query: str, items: list[SearchHit]) -> None:
    """Stamp query_coverage / missing_terms on hybrid-branch hits (both modes:
    even in relaxed mode clients see how well each hit matches the request)."""
    tokens = significant_tokens(query)
    for item in items:
        if item.match.branch is MatchBranch.HYBRID:
            ratio, missing = coverage(tokens, item.product)
            item.match.query_coverage = round(ratio, 3)
            item.match.missing_terms = missing or None


def _is_confident(item: SearchHit) -> bool:
    """Strict-mode gate: hybrid hits must cover every significant query term;
    code hits must come from a strong tier (exact / skeleton / EAN-corrected) —
    a fuzzy code hit is by definition a different code, i.e. an alternative."""
    if item.match.branch is MatchBranch.HYBRID:
        return item.match.query_coverage == 1.0
    return (item.match.code_score or 0.0) >= STRONG_CODE_SCORE


def _is_alternative(item: SearchHit) -> bool:
    """A useful near-miss covers a meaningful share of the request; fuzzy code hits
    (a close but different code) always qualify."""
    if item.match.branch is MatchBranch.HYBRID:
        return (item.match.query_coverage or 0.0) >= MIN_ALTERNATIVE_COVERAGE
    return True


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
