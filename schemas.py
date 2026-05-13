from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ReviewAction = Literal["pass", "revise", "retrieve_again"]
WorkflowStatus = Literal["passed", "needs_more_evidence"]


@dataclass(frozen=True)
class DocumentPage:
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Evidence:
    id: str
    content: str
    source: str
    page: int | None = None
    title_path: str = ""
    score: float = 0.0
    score_details: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    original_query: str
    rewritten_query: str
    keywords: list[str]
    evidence: list[Evidence]


@dataclass(frozen=True)
class ReviewResult:
    action: ReviewAction
    issues: list[str]
    missing_evidence: list[str]
    suggestions: list[str]
    summary: str = ""


@dataclass(frozen=True)
class WorkflowEvent:
    step: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowResult:
    final_document: str
    review: ReviewResult
    evidence: list[Evidence]
    trace: list[WorkflowEvent]
    rounds_used: int
    status: WorkflowStatus = "passed"
    final_message: str = ""
