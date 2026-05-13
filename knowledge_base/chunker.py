from __future__ import annotations

import hashlib

from document_parser.title_tree import Section
from schemas import Chunk


def chunk_sections(
    sections: list[Section],
    *,
    max_chars: int = 1200,
    min_chars: int = 80,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        content = section.content.strip()
        if not content:
            continue
        parts = split_by_paragraph(content, max_chars=max_chars)
        for part_index, part in enumerate(parts):
            if len(part.strip()) < min_chars and chunks:
                previous = chunks[-1]
                if previous.metadata.get("source") == section.source and len(previous.content) + len(part) < max_chars:
                    merged = previous.content.rstrip() + "\n" + part.strip()
                    chunks[-1] = Chunk(id=previous.id, content=merged, metadata=previous.metadata)
                    continue
            metadata = {
                "source": section.source,
                "page": section.page,
                "title_path": section.full_path,
                "level": section.level,
                "chunk_type": section.section_type,
                "chunk_index": len(chunks),
                "part_index": part_index,
            }
            chunk_id = make_chunk_id(section.source, section.page, section.full_path, part_index)
            chunks.append(Chunk(id=chunk_id, content=part.strip(), metadata=metadata))
    if not chunks:
        raise RuntimeError("No chunks produced. Check PDF extraction and title parsing.")
    return chunks


def split_by_paragraph(text: str, *, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                parts.append("\n".join(current))
                current = []
                current_len = 0
            parts.extend(split_long_text(paragraph, max_chars=max_chars))
            continue
        next_len = current_len + len(paragraph) + (1 if current else 0)
        if current and next_len > max_chars:
            parts.append("\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = next_len
    if current:
        parts.append("\n".join(current))
    return parts


def split_long_text(text: str, *, max_chars: int) -> list[str]:
    delimiters = ["。", "；", "！", "？"]
    sentences: list[str] = []
    buffer = ""
    for char in text:
        buffer += char
        if char in delimiters:
            sentences.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())

    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            parts.append(current.strip())
            current = sentence
        else:
            current += sentence
    if current.strip():
        parts.append(current.strip())
    sliced: list[str] = []
    for part in parts or [text]:
        if len(part) <= max_chars:
            sliced.append(part)
        else:
            sliced.extend(part[index : index + max_chars] for index in range(0, len(part), max_chars))
    return sliced


def make_chunk_id(source: str, page: int, title_path: str, part_index: int) -> str:
    payload = f"{source}|{page}|{title_path}|{part_index}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]
