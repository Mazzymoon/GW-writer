from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from agents.reviewer import Reviewer
from agents.searcher import Searcher
from agents.writer import Writer
from memory.case_memory import CaseMemory, CaseMemoryError
from memory.short_term import ShortTermMemory
from schemas import Evidence, ReviewResult, WorkflowEvent, WorkflowResult


INSUFFICIENT_EVIDENCE_MESSAGE = "知识库依据不足，无法生成合规版本。"

logger = logging.getLogger(__name__)


@dataclass
class AgenticWorkflow:
    searcher: Searcher
    writer: Writer
    reviewer: Reviewer
    case_memory: CaseMemory | None = None

    def run(self, user_query: str, *, max_rounds: int = 2, top_k: int = 6) -> WorkflowResult:
        started = time.perf_counter()
        trace: list[WorkflowEvent] = []
        memory = ShortTermMemory(user_query=user_query)
        evidence: list[Evidence] = []
        draft = ""
        review = ReviewResult(action="revise", issues=[], missing_evidence=[], suggestions=[])
        similar_cases: list[str] = []
        latest_keywords: list[str] = []

        if self.case_memory is not None:
            try:
                similar_cases = self.case_memory.retrieve_similar_cases(user_query, top_k=3)
                trace.append(
                    WorkflowEvent(
                        "memory",
                        "Retrieved similar cases",
                        {"case_count": len(similar_cases)},
                    )
                )
                memory.record_tool_call(
                    "case_memory_search",
                    {"query": user_query, "top_k": 3},
                    {"case_count": len(similar_cases)},
                )
            except CaseMemoryError as exc:
                logger.warning("Long-term memory retrieval failed: %s", exc)
                trace.append(WorkflowEvent("memory", "Long-term memory unavailable", {"error": str(exc)}))
        else:
            trace.append(WorkflowEvent("memory", "Long-term memory disabled", {}))

        search_query = user_query
        for round_index in range(max_rounds + 1):
            trace.append(WorkflowEvent("round", f"Start round {round_index + 1}", {"round": round_index + 1}))

            if round_index == 0 or review.action == "retrieve_again":
                if review.missing_evidence:
                    search_query = user_query + "\n需要补充检索：" + "；".join(review.missing_evidence)
                search_result = self.searcher.search(search_query, top_k=top_k)
                evidence = search_result.evidence
                latest_keywords = search_result.keywords
                memory.add_evidence(evidence)
                memory.record_tool_call(
                    "official_document_search",
                    {"query": search_query, "top_k": top_k, "recall_k": 16},
                    {
                        "rewritten_query": search_result.rewritten_query,
                        "keywords": search_result.keywords,
                        "evidence_count": len(evidence),
                    },
                )
                trace.append(
                    WorkflowEvent(
                        "searcher",
                        "Retrieved evidence",
                        {
                            "rewritten_query": search_result.rewritten_query,
                            "keywords": search_result.keywords,
                            "evidence_count": len(evidence),
                        },
                    )
                )

            draft = self.writer.draft(
                user_query,
                evidence,
                previous_draft=draft or None,
                review_suggestions=review.suggestions,
                memory_cases=similar_cases,
            )
            memory.add_draft(draft)
            trace.append(WorkflowEvent("writer", "Draft generated", {"chars": len(draft)}))

            review = self.reviewer.review(user_query, draft, evidence)
            memory.add_review(review)
            if review.missing_evidence:
                memory.add_missing_evidence(review.missing_evidence)
            trace.append(
                WorkflowEvent(
                    "reviewer",
                    f"Reviewer action: {review.action}",
                    {
                        "issues": review.issues,
                        "missing_evidence": review.missing_evidence,
                        "suggestions": review.suggestions,
                        "summary": review.summary,
                    },
                )
            )
            logger.info("Reviewer action=%s round=%s", review.action, round_index + 1)

            if review.action == "pass":
                if self.case_memory is not None:
                    try:
                        case_id = self.case_memory.add_success_case(
                            query=user_query,
                            final_document=draft,
                            review_summary=review.summary,
                            evidence=evidence,
                            keywords=latest_keywords,
                        )
                        trace.append(WorkflowEvent("memory", "Saved success case", {"case_id": case_id}))
                        memory.record_tool_call(
                            "case_memory_save",
                            {"query": user_query, "keywords": latest_keywords},
                            {"case_id": case_id},
                        )
                    except CaseMemoryError as exc:
                        logger.warning("Long-term memory save failed: %s", exc)
                        trace.append(WorkflowEvent("memory", "Failed to save success case", {"error": str(exc)}))
                break
            if round_index >= max_rounds:
                break

        elapsed = time.perf_counter() - started
        status = "needs_more_evidence" if review.action == "retrieve_again" else "passed"
        final_message = INSUFFICIENT_EVIDENCE_MESSAGE if status == "needs_more_evidence" else ""
        trace.append(
            WorkflowEvent(
                "workflow",
                "Finished",
                {
                    "elapsed_seconds": round(elapsed, 2),
                    "status": status,
                    "final_message": final_message,
                },
            )
        )
        return WorkflowResult(
            final_document=draft,
            review=review,
            evidence=evidence,
            trace=trace,
            rounds_used=len([event for event in trace if event.step == "round"]),
            status=status,  # type: ignore[arg-type]
            final_message=final_message,
            memory_cases=similar_cases,
        )
