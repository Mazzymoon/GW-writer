from __future__ import annotations

from pathlib import Path

from document_parser.pdf_parser import load_pdf_file, normalize_text
from schemas import DocumentPage


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def load_document_pages(docs_dir: str | Path) -> list[DocumentPage]:
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {root}")

    pages: list[DocumentPage] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".pdf":
            pages.extend(load_pdf_file(path))
        else:
            text = read_text_file(path)
            if text:
                pages.append(DocumentPage(source=path.name, page=1, text=text))

    if not pages:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RuntimeError(f"No readable documents found under {root}. Supported extensions: {supported}")
    return pages


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return normalize_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode text file: {path}")
