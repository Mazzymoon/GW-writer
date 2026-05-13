from __future__ import annotations

import re
from dataclasses import dataclass, field

from schemas import DocumentPage


HEADING_PATTERNS: list[tuple[int, str, re.Pattern[str]]] = [
    (1, "chapter", re.compile(r"^第[一二三四五六七八九十百千万\d]+章[\s、：:.-]*(.+)?$")),
    (2, "section", re.compile(r"^第[一二三四五六七八九十百千万\d]+节[\s、：:.-]*(.+)?$")),
    (3, "article", re.compile(r"^第[一二三四五六七八九十百千万\d]+条[\s、：:.-]*(.+)?$")),
    (4, "cn_number", re.compile(r"^[一二三四五六七八九十]+[、.．]\s*(.+)$")),
    (5, "cn_parenthesis", re.compile(r"^（[一二三四五六七八九十]+）\s*(.+)$")),
    (6, "number", re.compile(r"^\d+[.．、]\s*(.+)$")),
]


@dataclass
class Section:
    title: str
    level: int
    section_type: str
    source: str
    page: int
    content_lines: list[str] = field(default_factory=list)
    title_path: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n".join(self.content_lines).strip()

    @property
    def full_path(self) -> str:
        return " > ".join(self.title_path) if self.title_path else self.title


def parse_sections(pages: list[DocumentPage]) -> list[Section]:
    sections: list[Section] = []
    stack: list[Section] = []
    current_source: str | None = None
    root_section: Section | None = None

    for page in pages:
        if page.source != current_source:
            current_source = page.source
            root_title = infer_document_title(page.source)
            root_section = Section(
                title=root_title,
                level=0,
                section_type="document",
                source=page.source,
                page=page.page,
                title_path=[root_title],
            )
            stack = [root_section]

        for line in page.text.splitlines():
            heading = detect_heading(line)
            if heading:
                level, section_type, title = heading
                while stack and stack[-1].level >= level:
                    stack.pop()
                path = [node.title for node in stack] + [title]
                section = Section(
                    title=title,
                    level=level,
                    section_type=section_type,
                    source=page.source,
                    page=page.page,
                    title_path=path,
                )
                sections.append(section)
                stack.append(section)
            else:
                if stack:
                    stack[-1].content_lines.append(line)
                    if stack[-1].section_type == "document" and stack[-1] not in sections:
                        sections.append(stack[-1])
                elif root_section is not None:
                    root_section.content_lines.append(line)
                    if root_section not in sections:
                        sections.append(root_section)

    return [section for section in sections if section.content or section.section_type != "document"]


def detect_heading(line: str) -> tuple[int, str, str] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    for level, section_type, pattern in HEADING_PATTERNS:
        if pattern.match(stripped):
            return level, section_type, stripped
    return None


def infer_document_title(source: str) -> str:
    return re.sub(r"\.pdf$", "", source, flags=re.IGNORECASE)
