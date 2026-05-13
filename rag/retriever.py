from __future__ import annotations

from dataclasses import dataclass

from knowledge_base.index_store import KnowledgeBase
from rag.reranker import BGEReranker
from schemas import Evidence


@dataclass
class HybridRetriever:
    knowledge_base: KnowledgeBase
    reranker: BGEReranker

    def retrieve(self, query: str, *, top_k: int = 6, recall_k: int = 16) -> list[Evidence]:
        candidates = self.knowledge_base.hybrid_search(query, vector_k=recall_k, bm25_k=recall_k)
        return self.reranker.rerank(query, candidates, top_k=top_k)
