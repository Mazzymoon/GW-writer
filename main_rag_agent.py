from __future__ import annotations

import argparse
import json
import sys

from agents.workflow import AgenticWorkflow
from agents.reviewer import Reviewer
from agents.searcher import Searcher
from agents.writer import Writer
from config import ConfigurationError, Settings
from knowledge_base.builder import build_knowledge_base
from knowledge_base.index_store import KnowledgeBase
from llm_client import LLMClient
from logging_utils import configure_logging
from rag.reranker import BGEReranker
from rag.retriever import HybridRetriever
from schemas import Evidence, WorkflowResult


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = Settings.load()
    configure_logging(settings.log_level)

    try:
        if args.command == "build":
            return command_build(args, settings)
        if args.command == "inspect":
            return command_inspect(args, settings)
        if args.command == "draft":
            return command_draft(args, settings)
    except (ConfigurationError, RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="国企公文 Agentic Workflow CLI")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build", help="构建结构化 RAG 知识库")
    build.add_argument("--docs", default="./docs", help="文档目录，支持 .pdf/.md/.txt，默认 ./docs")

    inspect = subparsers.add_parser("inspect", help="查看混合检索和 BGE rerank 结果")
    inspect.add_argument("query", help="检索问题")
    inspect.add_argument("--top-k", type=int, default=5, help="返回证据数量")
    inspect.add_argument("--recall-k", type=int, default=16, help="每路召回候选数量")

    draft = subparsers.add_parser("draft", help="运行 Searcher-Writer-Reviewer 工作流")
    draft.add_argument("query", help="公文起草需求")
    draft.add_argument("--max-rounds", type=int, default=2, help="最多返工轮数")
    draft.add_argument("--top-k", type=int, default=6, help="每轮输入 Writer 的证据数量")
    draft.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")
    return parser


def command_build(args: argparse.Namespace, settings: Settings) -> int:
    kb = build_knowledge_base(args.docs, settings)
    vector_count = kb.vector_store.db.count()
    bm25_count = len(kb.bm25_store.chunks)
    print("知识库构建完成")
    print(f"- Chroma collection: {settings.chroma_collection}")
    print(f"- Chroma chunks: {vector_count}")
    print(f"- BM25 chunks: {bm25_count}")
    print(f"- Chroma dir: {settings.chroma_dir}")
    print(f"- BM25 dir: {settings.bm25_dir}")
    return 0


def command_inspect(args: argparse.Namespace, settings: Settings) -> int:
    kb = KnowledgeBase.load(settings)
    reranker = BGEReranker(settings)
    retriever = HybridRetriever(kb, reranker)
    evidence = retriever.retrieve(args.query, top_k=args.top_k, recall_k=args.recall_k)
    print_evidence(evidence)
    return 0


def command_draft(args: argparse.Namespace, settings: Settings) -> int:
    settings.require_llm()
    kb = KnowledgeBase.load(settings)
    llm = LLMClient(settings)
    reranker = BGEReranker(settings)
    retriever = HybridRetriever(kb, reranker)
    workflow = AgenticWorkflow(
        searcher=Searcher(llm, retriever),
        writer=Writer(llm),
        reviewer=Reviewer(llm),
    )
    result = workflow.run(args.query, max_rounds=args.max_rounds, top_k=args.top_k)
    if args.json:
        print(json.dumps(workflow_to_dict(result), ensure_ascii=False, indent=2))
    else:
        print_workflow_result(result)
    return 0


def print_evidence(evidence: list[Evidence]) -> None:
    if not evidence:
        print("未检索到结果。")
        return
    for index, item in enumerate(evidence, start=1):
        print(f"\n[{index}] score={item.score:.4f} source={item.source} page={item.page}")
        print(f"路径：{item.title_path}")
        print(f"分数：{item.score_details}")
        preview = item.content.replace("\n", " ")
        print(preview[:500] + ("..." if len(preview) > 500 else ""))


def print_workflow_result(result: WorkflowResult) -> None:
    if result.status == "needs_more_evidence":
        print("\n=== 生成状态 ===\n")
        print(result.final_message)
        print("以下草稿仅为失败轮次的中间产物，不应作为合规版本使用。")

    print("\n=== 最终公文草稿 ===\n")
    print(result.final_document)
    print("\n=== Reviewer 结果 ===")
    print(f"action: {result.review.action}")
    if result.review.summary:
        print(f"summary: {result.review.summary}")
    if result.review.issues:
        print("issues:")
        for item in result.review.issues:
            print(f"- {item}")
    if result.review.suggestions:
        print("suggestions:")
        for item in result.review.suggestions:
            print(f"- {item}")
    print("\n=== 参考依据 ===")
    print_evidence(result.evidence)
    print("\n=== 工作流轨迹 ===")
    for event in result.trace:
        print(f"- [{event.step}] {event.message} {event.metadata}")


def workflow_to_dict(result: WorkflowResult) -> dict:
    return {
        "final_document": result.final_document,
        "review": {
            "action": result.review.action,
            "issues": result.review.issues,
            "missing_evidence": result.review.missing_evidence,
            "suggestions": result.review.suggestions,
            "summary": result.review.summary,
        },
        "evidence": [
            {
                "id": item.id,
                "content": item.content,
                "source": item.source,
                "page": item.page,
                "title_path": item.title_path,
                "score": item.score,
                "score_details": item.score_details,
            }
            for item in result.evidence
        ],
        "trace": [
            {"step": event.step, "message": event.message, "metadata": event.metadata}
            for event in result.trace
        ],
        "rounds_used": result.rounds_used,
        "status": result.status,
        "final_message": result.final_message,
    }


if __name__ == "__main__":
    raise SystemExit(main())
