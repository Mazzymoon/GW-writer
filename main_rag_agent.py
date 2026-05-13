from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# ==========================================
# 1. 初始化豆包大模型客户端
# 大模型API：Doubao-Seed-2.0-lite
# ==========================================
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="993af805-a176-4a1a-9980-b79d3ba17f7f",
)
MODEL_ENDPOINT = "ep-m-20260320165100-f55rq" # 接入点 ID

# ==========================================
# 2. 连接本地知识库 (免去重新解析文档的时间)
# ==========================================
print("🧠 正在唤醒本地公文知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# 直接读取你刚才生成的 chroma_db 文件夹
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
print("✅ 知识库连接成功！")

# ==========================================
# 3. 核心 RAG 生成函数
# ==========================================
def generate_official_document(user_query):
    print(f"\n🔍 正在检索与【{user_query}】相关的历史公文规范...")
    
    # 步骤 A：去数据库里找最相关的 3 个文本块
    retrieved_docs = vector_db.similarity_search(user_query, k=3)
    
    # 步骤 B：把找到的文本块拼接成一段长文本，作为“参考背景”
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    print("✅ 检索完毕，已获取相关参考规范。")
    
    # 步骤 C：构建极其重要的 RAG Prompt（提示词工程）
    system_prompt = """你是一个资深的央企公文写作专家。
请【严格】基于我提供的<参考文件>中的行文规范、特定称谓和政策精神，来撰写公文。
如果参考文件中没有包含你需要的信息，你可以基于你的专业知识补充，但绝不能捏造政策文件号或虚假数据。
"""
    
    user_prompt = f"""
                <参考文件>
                {context_text}
                </参考文件>

                用户需求：{user_query}

                请根据上述需求和参考文件，起草一份结构完整、用词严谨的公文。
                """
    
    print("🔄 正在深度思考并起草公文中，请稍候...")
    
    # 步骤 D：调用大模型
    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2 # 极低的温度，保证遵循参考文件，拒绝幻觉
    )
    
    return response.choices[0].message.content

# ==========================================
# 4. 运行测试
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 公文书写智能体 (RAG 增强版) 已启动")
    print("=" * 50)
    
    # 你可以把这里换成你想测试的任何需求
    test_query = "我需要一份4.23开评审会的会议通知：会议名称是“政企行业销售能力培训”，会议密级，会议地点，分享人，会议内容，请帮我起草正文。"
    
    final_result = generate_official_document(test_query)
    
    print("\n🎉 === 最终生成的专业公文 === 🎉\n")
    print(final_result) 