from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemas import Evidence, ReviewResult, ToolCallRecord, WorkflowEvent


@dataclass
class ShortTermMemory:
    user_query: str
    task_plan: dict[str, Any] | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    drafts: list[str] = field(default_factory=list)
    reviews: list[ReviewResult] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    trace: list[WorkflowEvent] = field(default_factory=list)

    def record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        self.tool_calls.append(
            ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                result_summary=result_summary or {},
            )
        )

    def add_evidence(self, evidence: list[Evidence]) -> None:
        self.evidence.extend(evidence)

    def add_draft(self, draft: str) -> None:
        self.drafts.append(draft)

    def add_review(self, review: ReviewResult) -> None:
        self.reviews.append(review)

    def add_missing_evidence(self, items: list[str]) -> None:
        for item in items:
            if item and item not in self.missing_evidence:
                self.missing_evidence.append(item)
