from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_client import LLMClient
from memory.case_memory import CaseMemory
from rag.query_rewrite import rewrite_query
from rag.retriever import HybridRetriever
from schemas import Evidence


@dataclass
class AgentToolProvider:
    llm: LLMClient
    retriever: HybridRetriever
    case_memory: CaseMemory | None = None

    def official_document_search(self, query: str, top_k: int = 6, recall_k: int = 16) -> dict[str, Any]:
        rewrite = rewrite_query(query, self.llm)
        evidence = self.retriever.retrieve(rewrite.rewritten_query, top_k=top_k, recall_k=recall_k)
        return {
            "original_query": query,
            "rewritten_query": rewrite.rewritten_query,
            "keywords": rewrite.keywords,
            "evidence": [evidence_to_dict(item) for item in evidence],
        }

    def case_memory_search(self, query: str, top_k: int = 3) -> dict[str, Any]:
        if self.case_memory is None:
            return {"query": query, "cases": [], "status": "disabled"}
        cases = self.case_memory.retrieve_similar_cases(query, top_k=top_k)
        return {"query": query, "cases": cases, "status": "ok"}

    def case_memory_save(
        self,
        query: str,
        final_document: str,
        review_summary: str,
        evidence: list[Evidence] | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.case_memory is None:
            return {"status": "disabled", "message": "Long-term case memory is disabled."}
        case_id = self.case_memory.add_success_case(
            query=query,
            final_document=final_document,
            review_summary=review_summary,
            evidence=evidence or [],
            keywords=keywords,
        )
        return {"status": "ok", "case_id": case_id}


def evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "content": evidence.content,
        "source": evidence.source,
        "page": evidence.page,
        "title_path": evidence.title_path,
        "score": evidence.score,
        "score_details": evidence.score_details,
    }
