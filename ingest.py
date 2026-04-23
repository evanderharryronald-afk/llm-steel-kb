import os
import fitz
import docx
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "BAAI/bge-m3"

def load_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "source": str(path), "page": i + 1})
    return pages

def load_docx(path):
    doc = docx.Document(path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [{"text": full_text, "source": str(path), "page": 1}]

def load_all_docs(docs_dir):
    all_pages = []
    for path in Path(docs_dir).rglob("*"):
        if path.suffix.lower() == ".pdf":
            print(f"读取 PDF: {path.name}")
            all_pages.extend(load_pdf(path))
        elif path.suffix.lower() == ".docx":
            print(f"读取 Word: {path.name}")
            all_pages.extend(load_docx(path))
    return all_pages

def main():
    print("加载文档...")
    pages = load_all_docs(DOCS_DIR)
    print(f"共读取 {len(pages)} 个页面/段落")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )

    texts = []
    metadatas = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            if len(chunk.strip()) > 20:
                texts.append(chunk)
                metadatas.append({"source": page["source"], "page": page["page"]})

    print(f"切片完成，共 {len(texts)} 个片段")

    print("加载 Embedding 模型（首次运行会下载约 1GB，请耐心等待）...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("向量化并存入数据库...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR
    )

    print(f"完成！共存入 {len(texts)} 个片段，数据库保存在 {CHROMA_DIR}/")

if __name__ == "__main__":
    main()