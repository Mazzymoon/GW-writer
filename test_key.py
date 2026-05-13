from __future__ import annotations

from config import Settings
from llm_client import LLMClient


def main() -> int:
    settings = Settings.load()
    settings.require_llm()
    llm = LLMClient(settings)
    response = llm.complete(
        "你是一个接口连通性测试助手。",
        "请回复：LLM 配置可用。",
        temperature=0.0,
        max_tokens=64,
    )
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
