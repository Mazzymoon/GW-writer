import random
from openai import OpenAI
# 假设你已经把之前的纯模型调用和 RAG 调用封装好了，这里直接导入
# from doubao_agent import call_llm
# from main_rag_agent import generate_official_document

# ==========================================
# 1. 准备大模型客户端 (裁判模型)
# ==========================================
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="993af805-a176-4a1a-9980-b79d3ba17f7f",
)
MODEL_ENDPOINT = "ep-m-20260320165100-f55rq" # 接入点 ID

def llm_judge(user_query, doc_a, doc_b):
    """
    裁判 Agent：对两份公文进行盲评打分
    """
    judge_system_prompt = """你是一个极其严格的央企公文审核专家。
你需要对两篇针对同一需求起草的公文（候选A 和 候选B）进行盲评。
请从以下三个维度进行 1-10 分的打分，并给出简要的评价理由：
1. 事实与规范性 (Fact & Compliance)：是否包含编造的政策文件号、是否符合国资委或央企的宏观合规要求？
2. 行文风格 (Tone & Style)：是否具备体制内公文的严肃性、专业性和特定的称谓格式？
3. 内容详实度 (Comprehensiveness)：内容是否空洞，是否有具体的指导意见或操作抓手？

最后，请明确宣判哪一篇胜出。
请严格按照以下格式输出：
【候选A得分】事实: x/10, 风格: x/10, 详实度: x/10。总分: x/30
【候选B得分】事实: x/10, 风格: x/10, 详实度: x/10。总分: x/30
【评价理由】...
【最终胜出】候选X
"""

    prompt = f"""
起草需求：{user_query}

--- 候选公文 A ---
{doc_a}

--- 候选公文 B ---
{doc_b}
"""
    print("⚖️ 裁判 Agent 正在进行深度双盲评估...")
    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1 # 裁判必须极其客观稳定
    )
    return response.choices[0].message.content

# ==========================================
# 2. 自动化测试主流程
# ==========================================
if __name__ == "__main__":
    test_query = "请撰写一份关于加强中央企业合规管理与风险排查的通知正文。"
    
    print("🚀 开始生成对照组 (纯大模型)...")
    # baseline_doc = call_llm(test_query) # 调用纯大模型
    baseline_doc = "（这里替换为纯大模型生成的文本）" # 演示用
    
    print("🚀 开始生成实验组 (RAG 增强大模型)...")
    # rag_doc = generate_official_document(test_query) # 调用 RAG 流程
    rag_doc = "（这里替换为 RAG 生成的文本）" # 演示用
    
    # 核心：打乱顺序（双盲化处理）
    docs = [("纯大模型(Baseline)", baseline_doc), ("RAG增强(Experimental)", rag_doc)]
    random.shuffle(docs)
    
    doc_a_name, doc_a_content = docs[0]
    doc_b_name, doc_b_content = docs[1]
    
    # 交给裁判打分
    evaluation_report = llm_judge(test_query, doc_a_content, doc_b_content)
    
    # 揭晓盲测结果
    print("\n" + "="*50)
    print("📊 自动化评估报告揭晓")
    print("="*50)
    print(f"🕵️‍♂️ 身份揭秘：\n候选A的真实身份是: {doc_a_name}\n候选B的真实身份是: {doc_b_name}\n")
    print("📝 裁判详细评语：")
    print(evaluation_report)