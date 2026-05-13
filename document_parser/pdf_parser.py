from __future__ import annotations

from pathlib import Path

from schemas import DocumentPage


def load_pdf_pages(docs_dir: str | Path) -> list[DocumentPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pypdf. Install requirements.txt first.") from exc

    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {root}")

    pages: list[DocumentPage] = []
    for pdf_path in sorted(root.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = normalize_text(text)
            if text:
                pages.append(DocumentPage(source=pdf_path.name, page=index, text=text))
    if not pages:
        raise RuntimeError(f"No readable PDF pages found under {root}")
    return pages


def normalize_text(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = " ".join(raw_line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)
