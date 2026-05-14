from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from config import Settings
from schemas import Evidence


class CaseMemoryError(RuntimeError):
    pass


@dataclass
class CaseMemory:
    settings: Settings

    def __post_init__(self) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise CaseMemoryError(
                "Missing dependency for long-term memory: chromadb and sentence-transformers are required."
            ) from exc

        self.settings.memory_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer(self.settings.embedding_model)
        self.client = chromadb.PersistentClient(path=str(self.settings.memory_dir))
        self.collection = self.client.get_or_create_collection(name=self.settings.memory_collection)

    def retrieve_similar_cases(self, query: str, top_k: int = 3) -> list[str]:
        if self.collection.count() == 0:
            return []
        embedding = self._embed(query)
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        cases: list[str] = []
        for document, metadata in zip(documents, metadatas):
            review_summary = str((metadata or {}).get("review_summary", ""))
            evidence_titles = safe_json_loads(str((metadata or {}).get("evidence_titles", "[]")), [])
            cases.append(
                "历史成功案例：\n"
                f"审查摘要：{review_summary}\n"
                f"参考依据标题：{'；'.join(evidence_titles)}\n"
                f"公文片段：{document}"
            )
        return cases

    def add_success_case(
        self,
        query: str,
        final_document: str,
        review_summary: str,
        evidence: list[Evidence],
        keywords: list[str] | None = None,
    ) -> str:
        case_id = f"case_{uuid.uuid4().hex[:16]}"
        created_at = datetime.now(timezone.utc).isoformat()
        evidence_titles = [item.title_path or item.source for item in evidence[:8]]
        document = format_case_document(query, final_document)
        metadata: dict[str, Any] = {
            "query": query,
            "review_summary": review_summary,
            "evidence_titles": json.dumps(evidence_titles, ensure_ascii=False),
            "keywords": json.dumps(keywords or [], ensure_ascii=False),
            "created_at": created_at,
        }
        self.collection.add(
            ids=[case_id],
            documents=[document],
            embeddings=[self._embed(query + "\n" + final_document[:1200])],
            metadatas=[metadata],
        )
        return case_id

    def _embed(self, text: str) -> list[float]:
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return [float(value) for value in embedding.tolist()]


def format_case_document(query: str, final_document: str) -> str:
    clipped = final_document.strip()
    if len(clipped) > 1800:
        clipped = clipped[:1800] + "..."
    return f"用户需求：{query}\n\n通过草稿：\n{clipped}"


def safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default
