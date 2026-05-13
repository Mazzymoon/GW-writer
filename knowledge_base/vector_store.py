from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import Settings
from knowledge_base.bm25_store import chunk_to_evidence
from schemas import Chunk, Evidence


class VectorDependencyError(RuntimeError):
    pass


@dataclass
class VectorStore:
    db: object

    @classmethod
    def build(cls, chunks: list[Chunk], settings: Settings) -> "VectorStore":
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise VectorDependencyError("Missing dependency: chromadb. Install requirements.txt first.") from exc

        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
        except Exception as exc:
            raise VectorDependencyError(
                f"Unable to load embedding model '{settings.embedding_model}'. "
                "Install sentence-transformers and ensure the model is available."
            ) from exc
        collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        existing = collection.count()
        if existing:
            ids = collection.get(include=[])["ids"]
            if ids:
                collection.delete(ids=ids)
        collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[sanitize_metadata(chunk.metadata) for chunk in chunks],
        )
        return cls(db=collection)

    @classmethod
    def load(cls, settings: Settings) -> "VectorStore":
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise VectorDependencyError("Missing dependency: chromadb. Install requirements.txt first.") from exc

        if not Path(settings.chroma_dir).exists():
            raise FileNotFoundError(f"Chroma directory not found: {settings.chroma_dir}")
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        try:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
        except Exception as exc:
            raise VectorDependencyError(
                f"Unable to load embedding model '{settings.embedding_model}'. "
                "Install sentence-transformers and ensure the model is available."
            ) from exc
        collection = client.get_collection(
            name=settings.chroma_collection,
            embedding_function=embedding_function,
        )
        return cls(db=collection)

    def search(self, query: str, *, top_k: int) -> list[Evidence]:
        result = self.db.query(query_texts=[query], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(ids)

        evidence = []
        for chunk_id, content, metadata, distance in zip(ids, docs, metadatas, distances):
            score = 1.0 - float(distance)
            chunk = Chunk(id=chunk_id, content=content, metadata=metadata or {})
            evidence.append(chunk_to_evidence(chunk, score=score, score_key="vector"))
        return evidence


def sanitize_metadata(metadata: dict) -> dict:
    clean = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
