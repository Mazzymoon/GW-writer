from __future__ import annotations

import json
import re
from dataclasses import dataclass

from llm_client import LLMClient
from rag.prompt_builder import format_evidence
from schemas import Evidence, ReviewResult


@dataclass
class Reviewer:
    llm: LLMClient

    def review(self, user_query: str, draft: str, evidence: list[Evidence]) -> ReviewResult:
        evidence_text = format_evidence(evidence, max_chars=5000)
        system_prompt = """你是严格的央企公文合规审查员。你只输出 JSON，不要输出 Markdown。
action 只能是 pass、revise、retrieve_again：
- pass：草稿要素基本完整，依据与主题匹配，可以输出。
- revise：问题主要是措辞、结构、要素轻微缺失，可直接让 Writer 修改。
- retrieve_again：现有依据明显不足、模板不匹配、主题依据缺失，需要重新检索。
JSON 格式：
{"action":"pass|revise|retrieve_again","issues":["..."],"missing_evidence":["..."],"suggestions":["..."],"summary":"..."}"""
        user_prompt = f"""
用户需求：
{user_query}

检索依据：
{evidence_text}

待审查草稿：
{draft}

请检查：公文要素完整性、是否口语化、是否编造依据、是否和检索依据冲突、是否需要重新检索。"""
        raw = self.llm.complete(system_prompt, user_prompt, temperature=0.1)
        data = parse_json_object(raw)
        action = str(data.get("action", "revise")).strip()
        if action not in {"pass", "revise", "retrieve_again"}:
            action = "revise"
        return ReviewResult(
            action=action,  # type: ignore[arg-type]
            issues=to_string_list(data.get("issues")),
            missing_evidence=to_string_list(data.get("missing_evidence")),
            suggestions=to_string_list(data.get("suggestions")),
            summary=str(data.get("summary", "")).strip(),
        )


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {
                "action": "revise",
                "issues": ["Reviewer did not return valid JSON."],
                "missing_evidence": [],
                "suggestions": [text[:500]],
                "summary": "JSON parse failed.",
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "action": "revise",
                "issues": ["Reviewer returned malformed JSON."],
                "missing_evidence": [],
                "suggestions": [text[:500]],
                "summary": "JSON parse failed.",
            }


def to_string_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]
