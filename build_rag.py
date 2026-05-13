from __future__ import annotations

import argparse

from config import Settings
from knowledge_base.builder import build_knowledge_base
from logging_utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="兼容入口：构建结构化公文 RAG 知识库")
    parser.add_argument("--docs", default="./docs", help="PDF 文档目录")
    args = parser.parse_args()

    settings = Settings.load()
    configure_logging(settings.log_level)
    kb = build_knowledge_base(args.docs, settings)
    print("知识库构建完成")
    print(f"- Chroma chunks: {kb.vector_store.db.count()}")
    print(f"- BM25 chunks: {len(kb.bm25_store.chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
