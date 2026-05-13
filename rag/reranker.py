from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from schemas import Evidence


class RerankerDependencyError(RuntimeError):
    pass


@dataclass
class BGEReranker:
    settings: Settings

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerDependencyError(
                "Missing dependency: sentence-transformers. Install requirements.txt first."
            ) from exc
        try:
            self.model = CrossEncoder(self.settings.reranker_model)
        except Exception as exc:
            raise RerankerDependencyError(
                f"Unable to load BGE reranker model '{self.settings.reranker_model}': {exc}"
            ) from exc

    def rerank(self, query: str, evidence: list[Evidence], *, top_k: int) -> list[Evidence]:
        if not evidence:
            return []
        pairs = [(query, item.content) for item in evidence]
        scores = self.model.predict(pairs)
        reranked = []
        for item, score in zip(evidence, scores):
            score_float = float(score)
            reranked.append(
                Evidence(
                    id=item.id,
                    content=item.content,
                    source=item.source,
                    page=item.page,
                    title_path=item.title_path,
                    score=score_float,
                    score_details={**item.score_details, "rerank": score_float},
                )
            )
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
