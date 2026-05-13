from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import jieba

from schemas import Chunk, Evidence


class BM25DependencyError(RuntimeError):
    pass


@dataclass
class BM25Store:
    bm25: object
    chunks: list[Chunk]

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Store":
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise BM25DependencyError("Missing dependency: rank-bm25. Install requirements.txt first.") from exc
        corpus = [tokenize(chunk.content + " " + str(chunk.metadata.get("title_path", ""))) for chunk in chunks]
        return cls(bm25=BM25Okapi(corpus), chunks=chunks)

    @classmethod
    def load(cls, index_dir: str | Path) -> "BM25Store":
        root = Path(index_dir)
        with (root / "bm25.pkl").open("rb") as file:
            bm25 = pickle.load(file)
        raw_chunks = json.loads((root / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(id=item["id"], content=item["content"], metadata=item["metadata"]) for item in raw_chunks]
        return cls(bm25=bm25, chunks=chunks)

    def persist(self, index_dir: str | Path) -> None:
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "bm25.pkl").open("wb") as file:
            pickle.dump(self.bm25, file)
        raw_chunks = [
            {"id": chunk.id, "content": chunk.content, "metadata": chunk.metadata}
            for chunk in self.chunks
        ]
        (root / "chunks.json").write_text(json.dumps(raw_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query: str, *, top_k: int) -> list[Evidence]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for index, score in ranked:
            if score <= 0:
                continue
            chunk = self.chunks[index]
            results.append(chunk_to_evidence(chunk, score=float(score), score_key="bm25"))
        return results


def tokenize(text: str) -> list[str]:
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


def chunk_to_evidence(chunk: Chunk, *, score: float, score_key: str) -> Evidence:
    metadata = chunk.metadata
    return Evidence(
        id=chunk.id,
        content=chunk.content,
        source=str(metadata.get("source", "")),
        page=int(metadata["page"]) if metadata.get("page") is not None else None,
        title_path=str(metadata.get("title_path", "")),
        score=score,
        score_details={score_key: score},
    )
