from __future__ import annotations

import json
import re
from dataclasses import dataclass

from llm_client import LLMClient


@dataclass(frozen=True)
class JudgeResult:
    fact_score: float
    style_score: float
    completeness_score: float
    comments: str


def llm_judge(llm: LLMClient, user_query: str, draft: str) -> JudgeResult:
    system_prompt = """你是严格的央企公文评估专家。只输出 JSON：
{"fact_score":0-10,"style_score":0-10,"completeness_score":0-10,"comments":"..."}"""
    user_prompt = f"""用户需求：
{user_query}

候选公文：
{draft}

请从事实合规、行文风格、内容完整度三个维度评分。"""
    raw = llm.complete(system_prompt, user_prompt, temperature=0.1)
    data = parse_json_object(raw)
    return JudgeResult(
        fact_score=float(data.get("fact_score", 0)),
        style_score=float(data.get("style_score", 0)),
        completeness_score=float(data.get("completeness_score", 0)),
        comments=str(data.get("comments", "")),
    )


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"Judge did not return JSON: {text[:200]}")
        return json.loads(match.group(0))
