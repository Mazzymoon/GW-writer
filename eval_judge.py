from __future__ import annotations

import argparse
import json
import random

from config import Settings
from evaluation.judge import llm_judge
from llm_client import LLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge blind evaluation helper")
    parser.add_argument("query", help="原始起草需求")
    parser.add_argument("candidate_a", help="候选 A 文本文件路径")
    parser.add_argument("candidate_b", help="候选 B 文本文件路径")
    args = parser.parse_args()

    settings = Settings.load()
    settings.require_llm()
    llm = LLMClient(settings)

    candidates = [
        ("candidate_a", read_text(args.candidate_a)),
        ("candidate_b", read_text(args.candidate_b)),
    ]
    random.shuffle(candidates)

    first_name, first_text = candidates[0]
    second_name, second_text = candidates[1]
    first_report = llm_judge(llm, args.query, first_text)
    second_report = llm_judge(llm, args.query, second_text)

    print(
        json.dumps(
            {
                "blind_order": {"A": first_name, "B": second_name},
                "A": first_report.__dict__,
                "B": second_report.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


if __name__ == "__main__":
    raise SystemExit(main())
