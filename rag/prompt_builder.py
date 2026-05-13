from __future__ import annotations

from schemas import Evidence


def format_evidence(evidence: list[Evidence], *, max_chars: int = 6000) -> str:
    blocks = []
    total = 0
    for index, item in enumerate(evidence, start=1):
        header = f"[{index}] 来源：{item.source} 第{item.page or '?'}页 | 路径：{item.title_path}"
        body = item.content.strip()
        block = f"{header}\n{body}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)
