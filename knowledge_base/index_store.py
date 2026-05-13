from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from knowledge_base.bm25_store import BM25Store
from knowledge_base.vector_store import VectorStore
from schemas import Evidence


@dataclass
class KnowledgeBase:
    vector_store: VectorStore
    bm25_store: BM25Store

    @classmethod
    def load(cls, settings: Settings) -> "KnowledgeBase":
        return cls(
            vector_store=VectorStore.load(settings),
            bm25_store=BM25Store.load(settings.bm25_dir),
        )

    def hybrid_search(self, query: str, *, vector_k: int = 12, bm25_k: int = 12) -> list[Evidence]:
        vector_results = self.vector_store.search(query, top_k=vector_k)
        bm25_results = self.bm25_store.search(query, top_k=bm25_k)
        return merge_evidence(vector_results, bm25_results)


def merge_evidence(*groups: list[Evidence]) -> list[Evidence]:
    merged: dict[str, Evidence] = {}
    for group in groups:
        for item in group:
            if item.id in merged:
                previous = merged[item.id]
                score_details = {**previous.score_details, **item.score_details}
                merged[item.id] = Evidence(
                    id=previous.id,
                    content=previous.content,
                    source=previous.source,
                    page=previous.page,
                    title_path=previous.title_path,
                    score=max(previous.score, item.score),
                    score_details=score_details,
                )
            else:
                merged[item.id] = item
    return sorted(merged.values(), key=lambda evidence: evidence.score, reverse=True)
