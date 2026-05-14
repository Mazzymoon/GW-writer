from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from config import Settings
from knowledge_base.index_store import KnowledgeBase
from llm_client import LLMClient
from memory.case_memory import CaseMemory
from rag.reranker import BGEReranker
from rag.retriever import HybridRetriever
from tools.agent_tools import AgentToolProvider


logging.basicConfig(level=logging.WARNING)
mcp = FastMCP("gw-writer-agent-tools")
_provider: AgentToolProvider | None = None


def get_provider() -> AgentToolProvider:
    global _provider
    if _provider is not None:
        return _provider

    settings = Settings.load()
    settings.require_llm()
    knowledge_base = KnowledgeBase.load(settings)
    retriever = HybridRetriever(knowledge_base, BGEReranker(settings))
    case_memory = CaseMemory(settings) if settings.enable_long_term_memory else None
    _provider = AgentToolProvider(
        llm=LLMClient(settings),
        retriever=retriever,
        case_memory=case_memory,
    )
    return _provider


@mcp.tool()
def official_document_search(query: str, top_k: int = 6, recall_k: int = 16) -> dict[str, Any]:
    """Search official document knowledge and return reranked evidence chunks."""
    try:
        return get_provider().official_document_search(query=query, top_k=top_k, recall_k=recall_k)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "original_query": query, "evidence": []}


@mcp.tool()
def case_memory_search(query: str, top_k: int = 3) -> dict[str, Any]:
    """Search similar successful writing cases from long-term case memory."""
    try:
        return get_provider().case_memory_search(query=query, top_k=top_k)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "query": query, "cases": []}


@mcp.tool()
def case_memory_save(
    query: str,
    final_document: str,
    review_summary: str,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Save a successful writing case. Only call this after Reviewer passes the draft."""
    try:
        return get_provider().case_memory_save(
            query=query,
            final_document=final_document,
            review_summary=review_summary,
            evidence=[],
            keywords=keywords,
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
