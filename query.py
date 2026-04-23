from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """你是一个工厂内部知识库助手，专门回答关于设备操作、工艺规程、故障处理的问题。
回答要求：
1. 只根据提供的参考资料回答，不要编造内容
2. 如果参考资料中没有相关信息，直接说"文档中未找到相关内容"
3. 回答要简洁准确，必要时分点说明
4. 在回答末尾注明信息来源的文件名和页码"""

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
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def ask(question, retriever, llm):
    docs = retriever.invoke(question)

    if not docs:
        return "未检索到相关文档内容。", []

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
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    print("就绪，输入问题开始提问（输入 exit 退出）\n")

    while True:
        question = input("问题：").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("检索中...")
        answer, sources = ask(question, retriever, llm)
        print(f"\n回答：\n{answer}")
        if sources:
            print(f"\n来源：{', '.join(sources)}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()