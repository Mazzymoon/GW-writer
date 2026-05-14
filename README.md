# GW-writer：面向国企公文写作的多 Agent 智能写作系统

GW-writer 是一个面向国企公文写作场景的多 Agent 智能写作系统。系统由 Planner/Searcher Agent、Writer Agent、Reviewer Agent 和 Memory Manager 组成，通过工具调用和 MCP Server 接入公文知识检索、模板检索、历史案例检索和成功案例保存能力。

## Motivation

传统公文写作通常存在这些问题：

- 依赖人工查制度、找模板，检索成本高。
- 格式和语言风格不统一，容易出现口语化或要素缺失。
- 大模型直接生成容易编造政策文件号、日期、地点、联系人或数据。
- 缺少审查和返工机制，无法稳定发现依据不足或内容跑题。
- 历史成功经验分散在个人文档里，难以复用结构、语气和要素组织方式。

## Method

系统构建了多 Agent 协作架构：

- Planner/Searcher Agent：理解用户需求、拆解写作任务，并调用知识检索工具。
- Writer Agent：基于工具返回的制度依据、模板片段和历史成功案例生成公文草稿。
- Reviewer Agent：审查要素完整性、语言规范性、依据充分性和是否存在编造。
- Memory Manager：维护短期任务状态，并保存 Reviewer 判定通过的长期成功案例。
- MCP Tools：标准化暴露知识检索和案例记忆能力，便于外部 Agent 或 IDE 接入。

内部的知识检索实现层包含向量检索、关键词检索和重排，但系统对外定位是多 Agent 智能写作系统。

## Architecture

```text
用户需求
  ↓
Planner/Searcher Agent
  ├── official_document_search
  ├── case_memory_search
  ↓
Writer Agent
  ↓
Reviewer Agent
  ├── pass：输出并写入长期记忆
  ├── revise：返回 Writer 修改
  └── retrieve_again：重新调用知识检索工具
```

## Case：评审会会议通知

用户输入：

```text
请帮我写一份关于召开 XX 项目评审会的会议通知。
```

系统流程：

- Planner/Searcher Agent 识别为会议通知类公文。
- 拆解出会议时间、地点、参会人员、会议内容、材料要求、联系人、落款等要素。
- 调用 `official_document_search` 检索会议通知模板、评审会组织要求、材料报送要求。
- 调用 `case_memory_search` 检索历史相似成功案例。
- Writer Agent 生成草稿。
- Reviewer Agent 检查是否存在编造、要素缺失、依据不足。
- 未提供的会议时间、地点、联系人必须用 `〖待补充〗` 标注，不得编造。
- Reviewer 判定通过后，Memory Manager 将最终草稿写入长期成功案例记忆。

## Run

安装依赖：

```bash
python -m pip install -r requirements.txt
```

配置 `.env`：

```env
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your-key
LLM_MODEL=your-model
```

构建公文知识检索索引：

```bash
python main_rag_agent.py build --docs ./docs
```

查看知识检索工具召回结果：

```bash
python main_rag_agent.py inspect "中央企业合规管理要求" --top-k 5
```

运行多 Agent 公文写作流程：

```bash
python main_rag_agent.py draft "帮我写一个评审会会议通知" --max-rounds 1 --top-k 3
```

本次运行禁用长期记忆：

```bash
python main_rag_agent.py draft "帮我写一个评审会会议通知" --disable-memory
```

启动 MCP Server：

```bash
python mcp_agent_tools_server.py
```

## MCP 配置示例

```json
{
  "mcpServers": {
    "gw-writer-agent-tools": {
      "command": "python",
      "args": ["mcp_agent_tools_server.py"],
      "cwd": "/path/to/GW-writer",
      "env": {
        "LLM_API_KEY": "your-key",
        "LLM_MODEL": "your-model",
        "LLM_BASE_URL": "your-base-url"
      }
    }
  }
}
```

## Notes

- 长期记忆库不是制度依据库，只保存 Reviewer 判定通过的成功案例。
- 历史成功案例只能作为结构、语气和要素组织方式参考，不能替代知识检索工具返回的制度依据。
- 如果 Reviewer 在最大轮数后仍要求重新检索，系统会明确提示“知识库依据不足，无法生成合规版本”。
