from __future__ import annotations

import json
import re
from dataclasses import dataclass

import jieba

from llm_client import LLMClient


@dataclass(frozen=True)
class QueryRewriteResult:
    rewritten_query: str
    keywords: list[str]


def rewrite_query(user_query: str, llm: LLMClient) -> QueryRewriteResult:
    system_prompt = """你是国企公文 RAG 检索专家。你的任务是把用户需求改写成适合检索模板、制度依据、格式要求、合规条款的查询语句。
只输出 JSON，格式为：
{"rewritten_query": "...", "keywords": ["...", "..."]}"""
    user_prompt = f"""用户需求：
{user_query}

请补充会议/通知/请示/报告等公文要素、格式规范、依据条款、参会人员、材料准备、职责分工、保密要求等检索词。"""
    raw = llm.complete(system_prompt, user_prompt, temperature=0.1)
    data = parse_json_object(raw)
    rewritten = str(data.get("rewritten_query") or user_query).strip()
    keywords = [str(item).strip() for item in data.get("keywords", []) if str(item).strip()]
    if not keywords:
        keywords = extract_keywords(rewritten)
    return QueryRewriteResult(rewritten_query=rewritten, keywords=keywords[:12])


def extract_keywords(text: str, *, max_keywords: int = 12) -> list[str]:
    stopwords = {"请", "帮", "我", "一份", "关于", "进行", "需要", "起草", "正文", "要求"}
    candidates = [token for token in jieba.lcut(text) if len(token.strip()) >= 2 and token not in stopwords]
    seen = set()
    keywords = []
    for token in candidates:
        if token not in seen:
            seen.add(token)
            keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"LLM did not return JSON: {text[:200]}")
        return json.loads(match.group(0))
