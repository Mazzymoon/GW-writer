from __future__ import annotations

from dataclasses import dataclass

from llm_client import LLMClient
from rag.prompt_builder import format_evidence
from schemas import Evidence


@dataclass
class Writer:
    llm: LLMClient

    def draft(
        self,
        user_query: str,
        evidence: list[Evidence],
        *,
        previous_draft: str | None = None,
        review_suggestions: list[str] | None = None,
    ) -> str:
        evidence_text = format_evidence(evidence)
        system_prompt = """你是资深央企公文写作专家。你必须基于参考依据起草正式、克制、结构完整的公文草稿。
要求：
1. 不编造不存在的政策文件号、数据、会议地点、联系人或日期。
2. 用户未给出的关键要素，用【待补充：要素名】标注。
3. 语言符合国企/机关公文风格，避免口语化表达。
4. 文末列出“参考依据”，用编号引用检索片段。"""

        revision_block = ""
        if previous_draft:
            revision_block = f"""
上一版草稿：
{previous_draft}

Reviewer 修改建议：
{chr(10).join(f"- {item}" for item in (review_suggestions or []))}

请在上一版基础上修订，不要无关扩写。"""

        user_prompt = f"""
用户需求：
{user_query}

参考依据：
{evidence_text}

{revision_block}

请输出一份可供业务人员继续修改的公文草稿。"""
        return self.llm.complete(system_prompt, user_prompt, temperature=0.2)
