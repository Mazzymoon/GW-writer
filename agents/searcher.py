from __future__ import annotations

import logging
from dataclasses import dataclass

from llm_client import LLMClient
from rag.query_rewrite import rewrite_query
from rag.retriever import HybridRetriever
from schemas import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Searcher:
    llm: LLMClient
    retriever: HybridRetriever

    def search(self, user_query: str, *, top_k: int = 6, recall_k: int = 16) -> SearchResult:
        rewrite = rewrite_query(user_query, self.llm)
        logger.info("Query rewrite: %s", rewrite.rewritten_query)
        evidence = self.retriever.retrieve(rewrite.rewritten_query, top_k=top_k, recall_k=recall_k)
        logger.info("Retrieved and reranked %s evidence chunks", len(evidence))
        return SearchResult(
            original_query=user_query,
            rewritten_query=rewrite.rewritten_query,
            keywords=rewrite.keywords,
            evidence=evidence,
        )
