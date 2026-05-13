"""
文档加载 (Loader)：自动把你文件夹里的所有 PDF 读进来。
切块策略 (Chunking)：大模型的上下文是有限的，不能一次性塞入几十本 PDF。所以我们设置了 chunk_size=500，把文章切成小段。设置 chunk_overlap=50 是为了防止一句话正好被从中间切断，保留上下文语义。
向量化 (Embedding)：这里配置了 BAAI/bge-small-zh-v1.5，这是国内非常优秀的开源中文 Embedding 模型，它能把你的公文文本变成多维向量，专门用来做相似度匹配。（注意：第一次运行这段代码时，后台会自动从 HuggingFace 下载这个模型，大概一百多兆，需要稍微等一两分钟）。
持久化存储：数据被存到了项目里的 chroma_db 文件夹。以后你就不需要每次都重新解析 PDF 了，直接读取这个文件夹就行。
"""
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# ==========================================
# 1. 读取并解析 PDF 文档
# ==========================================
print("📄 正在读取 docs 目录下的 PDF 文件...")
# 这个工具会自动遍历 docs 文件夹下所有的 pdf
loader = PyPDFDirectoryLoader("./docs") 
documents = loader.load()
print(f"✅ 成功读取，共解析出 {len(documents)} 页文档。")

# ==========================================
# 2. 文档切块 (Chunking)
# ==========================================
# 公文很长，我们需要把它切成小块，大模型才能“消化”
print("✂️ 正在对文档进行智能切块...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # 每个文本块大约 500 个字符
    chunk_overlap=50,   # 块与块之间重叠 50 个字符，防止上下文断裂
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"✅ 切块完成，共生成 {len(chunks)} 个文本块。")

# ==========================================
# 3. 向量化并存入 ChromaDB 数据库
# ==========================================
print("🧠 正在加载开源中文 Embedding 模型 (首次运行会自动下载，请耐心等待)...")
# 我们使用 BAAI（北京智源）开源的优秀中文向量模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

print("💾 正在构建本地向量数据库...")
# 把切好的文本块存到本地的 chroma_db 文件夹中
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)
print("🎉 知识库构建成功！数据已保存在 ./chroma_db 目录下。")

# ==========================================
# 4. 测试一下检索效果
# ==========================================
print("\n" + "="*50)
query = "中央企业在合规管理方面有哪些要求？"
print(f"🔍 模拟大模型检索问题：{query}")

# 检索最相关的 3 个文本块
results = vector_db.similarity_search(query, k=3)

for i, res in enumerate(results):
    print(f"\n--- 💡 检索结果 {i+1} (来源: {res.metadata['source']} 第 {res.metadata['page']} 页) ---")
    print(res.page_content)