from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from agents.reviewer import Reviewer
from agents.searcher import Searcher
from agents.writer import Writer
from schemas import Evidence, ReviewResult, WorkflowEvent, WorkflowResult


INSUFFICIENT_EVIDENCE_MESSAGE = "知识库依据不足，无法生成合规版本。"

logger = logging.getLogger(__name__)


@dataclass
class AgenticWorkflow:
    searcher: Searcher
    writer: Writer
    reviewer: Reviewer

    def run(self, user_query: str, *, max_rounds: int = 2, top_k: int = 6) -> WorkflowResult:
        started = time.perf_counter()
        trace: list[WorkflowEvent] = []
        evidence: list[Evidence] = []
        draft = ""
        review = ReviewResult(action="revise", issues=[], missing_evidence=[], suggestions=[])

        search_query = user_query
        for round_index in range(max_rounds + 1):
            trace.append(WorkflowEvent("round", f"Start round {round_index + 1}", {"round": round_index + 1}))

            if round_index == 0 or review.action == "retrieve_again":
                if review.missing_evidence:
                    search_query = user_query + "\n需要补充检索：" + "；".join(review.missing_evidence)
                search_result = self.searcher.search(search_query, top_k=top_k)
                evidence = search_result.evidence
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
            )
            trace.append(WorkflowEvent("writer", "Draft generated", {"chars": len(draft)}))

            review = self.reviewer.review(user_query, draft, evidence)
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
        )
