"""In-memory exact/fuzzy index over product codes (article, product_code, ean13).

Chosen over Qdrant-side trigram sparse vectors deliberately: at ≤100k products ×3 code
fields RapidFuzz's C++ batch ``process.extract`` with a score cutoff completes in tens of
milliseconds, models edit distance directly (trigram overlap does not), and keeps the
collection schema simple. Cost: each API replica rebuilds the index at startup from a
payload-only scroll and applies incremental updates on upsert/delete.

Matching cascade per token (first non-empty tier wins):
  1. exact  — normalized form equality                          → score 1.00
  2. exact_normalized — lossy OCR-skeleton equality             → score 0.97
  3. ean_corrected — checksum-aware EAN-13 error candidates     → score 0.95
  4. fuzzy  — OSA + Jaro-Winkler blend over the whole corpus    → score ∈ [0.80..1)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from rapidfuzz import process
from rapidfuzz.distance import OSA, JaroWinkler

from app.models.search import MatchBranch
from app.services.ean import ean13_variants
from app.services.normalization import norm_code, skeleton

# Score tiers of the matching cascade. Fuzzy scores are capped strictly below the
# strong tiers so that ordering by score always reproduces the tier order.
EXACT_SCORE = 1.0
SKELETON_SCORE = 0.97
EAN_SCORE = 0.95
FUZZY_MAX_SCORE = 0.949

# Hits at/above this score are unambiguous enough to short-circuit vector search.
STRONG_CODE_SCORE = EAN_SCORE

FUZZY_CUTOFF = 0.72
FUZZY_THRESHOLD = 0.80
FUZZY_CANDIDATES = 20
FUZZY_OSA_WEIGHT = 0.6
FUZZY_JW_WEIGHT = 0.4
FUZZY_LENGTH_PENALTY_STEP = 0.02
FUZZY_LENGTH_PENALTY_CAP = 0.10


@dataclass(frozen=True)
class CodeRef:
    point_id: str
    field: str  # "article" | "product_code" | "ean13"
    raw: str


@dataclass(frozen=True)
class CodeHit:
    point_id: str
    field: str
    score: float
    branch: MatchBranch


@dataclass
class _PointKeys:
    entries: list[tuple[str, str, CodeRef]] = field(default_factory=list)  # (norm, skel, ref)


class CodeIndex:
    def __init__(self) -> None:
        self._exact: dict[str, list[CodeRef]] = {}
        self._skeleton: dict[str, list[CodeRef]] = {}
        self._by_point: dict[str, _PointKeys] = {}
        self._corpus: list[str] | None = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._by_point)

    # --- mutations ---

    def add_product(self, point_id: str, codes: dict[str, str | None]) -> None:
        with self._lock:
            self._remove_locked(point_id)
            keys = _PointKeys()
            for fld, raw in codes.items():
                if not raw:
                    continue
                normed = norm_code(str(raw))
                if not normed:
                    continue
                skel = skeleton(normed)
                ref = CodeRef(point_id=point_id, field=fld, raw=str(raw))
                self._exact.setdefault(normed, []).append(ref)
                self._skeleton.setdefault(skel, []).append(ref)
                keys.entries.append((normed, skel, ref))
            if keys.entries:
                self._by_point[point_id] = keys
            self._corpus = None

    def remove_product(self, point_id: str) -> None:
        with self._lock:
            self._remove_locked(point_id)
            self._corpus = None

    def _remove_locked(self, point_id: str) -> None:
        keys = self._by_point.pop(point_id, None)
        if keys is None:
            return
        for normed, skel, ref in keys.entries:
            for mapping, key in ((self._exact, normed), (self._skeleton, skel)):
                refs = mapping.get(key)
                if refs is None:
                    continue
                refs[:] = [r for r in refs if r != ref]
                if not refs:
                    del mapping[key]

    # --- matching ---

    def _ensure_corpus(self) -> list[str]:
        with self._lock:
            if self._corpus is None:
                self._corpus = list(self._exact.keys())
            return self._corpus

    def _lookup(self, table: dict[str, list[CodeRef]], key: str) -> list[CodeRef]:
        """Snapshot of refs for a key; the lock guards against concurrent incremental updates."""
        with self._lock:
            return list(table.get(key, ()))

    def match(self, token: str) -> list[CodeHit]:
        normed = norm_code(token)
        if not normed:
            return []

        refs = self._lookup(self._exact, normed)
        if refs:
            return [CodeHit(r.point_id, r.field, EXACT_SCORE, MatchBranch.EXACT) for r in refs]

        refs = self._lookup(self._skeleton, skeleton(normed))
        if refs:
            return [CodeHit(r.point_id, r.field, SKELETON_SCORE, MatchBranch.EXACT_NORMALIZED) for r in refs]

        if normed.isdigit() and 12 <= len(normed) <= 14:
            candidates, _ = ean13_variants(normed)
            ean_hits = [
                CodeHit(r.point_id, r.field, EAN_SCORE, MatchBranch.EAN_CORRECTED)
                for cand in candidates
                for r in self._lookup(self._exact, cand)
            ]
            if ean_hits:
                return ean_hits

        return self._fuzzy(normed)

    def _fuzzy(self, normed: str) -> list[CodeHit]:
        corpus = self._ensure_corpus()
        if not corpus:
            return []
        matches = process.extract(
            normed,
            corpus,
            scorer=OSA.normalized_similarity,
            score_cutoff=FUZZY_CUTOFF,
            limit=FUZZY_CANDIDATES,
        )
        hits: list[CodeHit] = []
        for candidate, osa_score, _idx in matches:
            jw = JaroWinkler.normalized_similarity(normed, candidate)
            length_penalty = min(
                abs(len(normed) - len(candidate)) * FUZZY_LENGTH_PENALTY_STEP,
                FUZZY_LENGTH_PENALTY_CAP,
            )
            score = FUZZY_OSA_WEIGHT * osa_score + FUZZY_JW_WEIGHT * jw - length_penalty
            if score < FUZZY_THRESHOLD:
                continue
            for ref in self._lookup(self._exact, candidate):
                hits.append(
                    CodeHit(ref.point_id, ref.field, round(min(score, FUZZY_MAX_SCORE), 4), MatchBranch.FUZZY)
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits
