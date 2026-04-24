from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "qwen2.5:7b"
RETRIEVE_TOP_K = 20
RERANK_TOP_K = 4

SYSTEM_PROMPT = """你是一个工厂内部知识库助手，专门回答关于设备操作、工艺规程、故障处理的问题。
回答要求：
1. 只根据提供的参考资料回答，不要编造内容
2. 如果参考资料中没有相关信息，直接说"文档中未找到相关内容"
3. 回答要简洁准确，必要时分点说明
4. 在回答末尾注明信息来源的文件名和页码"""

# ★ 新增：查询改写的 prompt
REWRITE_PROMPT = """你是一个搜索优化助手。请将用户的问题改写为3个不同角度的搜索查询，用于在工厂技术文档中检索信息。

要求：
1. 每行输出一个查询，不要编号，不要多余说明
2. 换不同的表达方式，覆盖可能的同义词（如"缺点"→"限制"、"不支持"、"注意事项"）
3. 保持简洁，每个查询不超过20字

用户问题：{question}"""


def load_reranker(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model


def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})


# ★ 新增：查询改写函数
def rewrite_query(question, llm):
    """用 LLM 把原始问题改写成3个不同角度的查询"""
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    # 加上原始问题，一共最多4个查询
    all_queries = [question] + queries[:3]
    print(f"  改写查询：{all_queries[1:]}")  # 打印改写结果，方便调试
    return all_queries


# ★ 新增：多路检索并去重
def multi_retrieve(queries, retriever):
    """对每个查询分别检索，按 page_content 去重合并"""
    seen = set()
    all_docs = []
    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)
    return all_docs


def rerank(question, docs, reranker, top_k=RERANK_TOP_K):
    tokenizer, model = reranker
    device = next(model.parameters()).device
    pairs = [[question, doc.page_content] for doc in docs]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        scores = model(**inputs).logits.squeeze(-1).tolist()

    if isinstance(scores, float):
        scores = [scores]

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def ask(question, retriever, reranker, llm):
    # 第一阶段：查询改写
    queries = rewrite_query(question, llm)

    # 第二阶段：多路检索 + 去重
    docs = multi_retrieve(queries, retriever)
    if not docs:
        return "未检索到相关文档内容。", []
    print(f"  检索到 {len(docs)} 个候选片段（去重后）")

    # 第三阶段：reranker 重排，用原始问题打分
    docs = rerank(question, docs, reranker)

    context = ""
    sources = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        filename = source.split("\\")[-1]
        context += f"\n[片段{i+1} - {filename} 第{page}页]\n{doc.page_content}\n"
        sources.append(f"{filename} 第{page}页")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"参考资料：\n{context}\n\n问题：{question}")
    ]

    response = llm.invoke(messages)
    return response.content, list(set(sources))


def main():
    print("加载模型和数据库...")
    retriever = load_retriever()

    print("加载 Reranker 模型...")
    reranker = load_reranker(RERANKER_MODEL)

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    print("就绪，输入问题开始提问（输入 exit 退出）\n")

    while True:
        question = input("问题：").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("检索中...")
        answer, sources = ask(question, retriever, reranker, llm)
        print(f"\n回答：\n{answer}")
        if sources:
            print(f"\n来源：{', '.join(sources)}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()