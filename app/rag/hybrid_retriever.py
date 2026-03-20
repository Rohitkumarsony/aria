from __future__ import annotations
from collections import defaultdict
from rank_bm25 import BM25Okapi
from app.core.config import get_settings
from app.rag.vector_store import VectorStore

settings = get_settings()
_RRF_K = 60


class HybridRetriever:
    def __init__(self, vs: VectorStore) -> None:
        self._vs = vs
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []

    def rebuild(self) -> None:
        self._corpus = self._vs.get_all()
        if self._corpus:
            self._bm25 = BM25Okapi([d["content"].lower().split() for d in self._corpus])

    def retrieve(self, query: str) -> list[dict]:
        dense = self._vs.dense_search(query, top_k=settings.retrieval_top_k)
        if not self._bm25: return dense[:settings.hybrid_final_k]
        scores = self._bm25.get_scores(query.lower().split())
        sparse = [{"content": self._corpus[i]["content"], "metadata": self._corpus[i]["metadata"]}
                  for i in sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:settings.bm25_top_k] if scores[i] > 0]
        return self._rrf(dense, sparse)[:settings.hybrid_final_k]

    @staticmethod
    def _rrf(dense, sparse):
        scores: dict[str, float] = defaultdict(float)
        store: dict[str, dict] = {}
        for rank, h in enumerate(dense, 1):
            scores[h["content"]] += 1 / (_RRF_K + rank); store[h["content"]] = h
        for rank, h in enumerate(sparse, 1):
            scores[h["content"]] += 1 / (_RRF_K + rank); store.setdefault(h["content"], h)
        return [store[k] for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    @property
    def has_documents(self) -> bool: return self._vs.total_chunks > 0
