"""Optional cross-encoder reranking (BAAI/bge-reranker-v2-m3 via sentence-transformers).

Lazy-loaded on the first rerank=true request: the torch stack is heavy and most
deployments never enable it. Requires `uv sync --group rerank`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankerService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> CrossEncoder:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise RuntimeError(
                        "sentence-transformers is not installed; run `uv sync --group rerank`"
                    ) from exc
                logger.info("Loading reranker %s (first rerank request) ...", self.model_name)
                self._model = await asyncio.to_thread(CrossEncoder, self.model_name)
                logger.info("Reranker ready")
        return self._model

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        model = await self._ensure_loaded()
        pairs = [(query, doc) for doc in docs]
        scores = await asyncio.to_thread(model.predict, pairs)
        return [float(s) for s in scores]
