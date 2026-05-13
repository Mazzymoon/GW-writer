from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing."""


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str | None
    llm_model: str | None
    embedding_model: str
    reranker_model: str
    chroma_dir: Path
    chroma_collection: str
    bm25_dir: Path
    log_level: str
    redis_url: str

    @classmethod
    def load(cls, env_file: str | Path = ".env") -> "Settings":
        load_env_file(env_file)
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_model=os.getenv("LLM_MODEL"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
            chroma_dir=Path(os.getenv("CHROMA_DIR", "./chroma_db")),
            chroma_collection=os.getenv("CHROMA_COLLECTION", "official_documents_agentic"),
            bm25_dir=Path(os.getenv("BM25_DIR", "./bm25_index")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        )

    def require_llm(self) -> None:
        missing = []
        if not self.llm_api_key:
            missing.append("LLM_API_KEY")
        if not self.llm_model:
            missing.append("LLM_MODEL")
        if missing:
            joined = ", ".join(missing)
            raise ConfigurationError(
                f"Missing {joined}. Copy .env.example to .env and fill OpenAI-compatible LLM settings."
            )


def load_env_file(env_file: str | Path) -> None:
    path = Path(env_file)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
