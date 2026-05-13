from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from document_parser.document_loader import load_document_pages
from document_parser.title_tree import parse_sections
from knowledge_base.bm25_store import BM25Store
from knowledge_base.chunker import chunk_sections
from knowledge_base.index_store import KnowledgeBase
from knowledge_base.vector_store import VectorStore

logger = logging.getLogger(__name__)


def build_knowledge_base(docs_dir: str | Path, settings: Settings) -> KnowledgeBase:
    logger.info("Loading documents from %s", docs_dir)
    pages = load_document_pages(docs_dir)
    logger.info("Loaded %s document pages/units", len(pages))

    sections = parse_sections(pages)
    logger.info("Parsed %s structural sections", len(sections))

    chunks = chunk_sections(sections)
    logger.info("Produced %s title-tree chunks", len(chunks))

    vector_store = VectorStore.build(chunks, settings)
    bm25_store = BM25Store.build(chunks)
    bm25_store.persist(settings.bm25_dir)
    logger.info("Persisted Chroma index to %s and BM25 index to %s", settings.chroma_dir, settings.bm25_dir)
    return KnowledgeBase(vector_store=vector_store, bm25_store=bm25_store)
