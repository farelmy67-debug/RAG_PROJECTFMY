"""
RAG Chatbot sederhana pakai LangChain + Groq + ChromaDB (lokal)
Disesuaikan untuk Knowledge Base PPKD Jakarta Barat
"""

import os
import glob

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. KONFIGURASI OPTIMAL UNTUK KNOWLEDGE BASE PPKD
# ============================================================

CHAT_MODEL = "openai/gpt-oss-120b"
COLLECTION_NAME = "ppkd_jakbar"

# Settingan chunk & retrieval yang dinamis dan pas untuk dokumen Bahasa Indonesia
CHUNK_SIZE = 1200       
CHUNK_OVERLAP = 250     
TOP_K = 15

KNOWLEDGE_DIR = "./knowledge_docs"
SYSTEM_PROMPT_PATH = "./system_prompt.md"

# ============================================================
# 2. SETUP MODEL
# ============================================================

def buat_model() -> ChatGroq:
    """Siapkan koneksi ke model chat lewat Groq."""
    return ChatGroq(
        model=CHAT_MODEL,
        temperature=0,
        reasoning_effort="low",
    )


# ============================================================
# 3. DATA INGESTION (Load -> Split -> Embed -> Store)
# ============================================================

def muat_dokumen(folder: str) -> list[Document]:
    daftar_dokumen = []
    path_file = sorted(glob.glob(os.path.join(folder, "*.pdf")))

    if not path_file:
        raise FileNotFoundError(
            f"Tidak ada file .pdf ditemukan di folder '{folder}'. "
            "Pastikan folder knowledge_docs/ berisi file sumber."
        )

    for path in path_file:
        # Mengubah mode="page" agar PDF dibaca per halaman dan metadata halaman terjaga
        loader = PyMuPDF4LLMLoader(file_path=path, mode="page", use_layout=False)
        daftar_dokumen.extend(loader.load())

    return daftar_dokumen


def bangun_vectorstore(dokumen: list) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n10.", "\n9.", "\n8.", "\n7.", "\n6.", "\n5.", "\n4.", "\n3.", "\n2.", "\n1.",
            "\nA.", "\nB.", "\nC.", "\nD.", "\nE.",
            "\n\n", "\n", " ", ""
        ]
    )
    potongan = splitter.split_documents(dokumen)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    vectorstore.reset_collection()
    vectorstore.add_documents(potongan)

    return vectorstore


# ============================================================
# 4. RAG CHAIN (Context Injection -> Prompt -> Model -> Parser)
# ============================================================

def format_docs(daftar_dokumen: list[Document]) -> str:
    return "\n\n".join(dok.page_content for dok in daftar_dokumen)


def muat_system_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def buat_rag_chain(retriever, model: ChatGroq, system_prompt: str):
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Konteks:\n{context}\n\nPertanyaan: {question}"),
    ])

    generation_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )

    full_chain = RunnableParallel(
        answer=generation_chain,
        sources=retriever
    )
    return full_chain


# ============================================================
# 5. PROGRAM UTAMA
# ============================================================

def main():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY tidak ditemukan. Pastikan file .env ada di folder "
            "yang sama dengan script ini dan berisi GROQ_API_KEY=..."
        )

    print("Menyiapkan model...")
    model = buat_model()

    print(f"Memuat dokumen dari folder '{KNOWLEDGE_DIR}'...")
    dokumen = muat_dokumen(KNOWLEDGE_DIR)
    print(f"  -> {len(dokumen)} halaman/dokumen berhasil dimuat.")

    print("Membangun vector store dari dokumen (menggunakan Multilingual Embedding)...")
    vectorstore = bangun_vectorstore(dokumen)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print(f"Memuat system prompt dari '{SYSTEM_PROMPT_PATH}'...")
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)
    
    print("Merakit RAG chain...")
    rag_chain = buat_rag_chain(retriever, model, system_prompt)

    print("\nRAG chatbot siap. Ketik pertanyaan, atau 'keluar' untuk berhenti.\n")

    while True:
        pertanyaan = input("Pertanyaan: ").strip()
        if pertanyaan.lower() in {"keluar", "exit", "quit"}:
            print("Sampai jumpa.")
            break
        if not pertanyaan:
            continue

        # Eksekusi RAG chain
        hasil = rag_chain.invoke(pertanyaan)
        
        # Format cetak output agar bersih & profesional
        print("\n" + "="*50)
        print("JAWABAN:")
        print("="*50)
        print(hasil["answer"])
        
        print("\n" + "-"*50)
        print("SUMBER DOKUMEN RELEVAN:")
        print("-"*50)
        
        sumber_unik = list(set([doc.metadata.get("source", "Dokumen Tanpa Nama") for doc in hasil["sources"]]))
        for i, src in enumerate(sumber_unik, 1):
            nama_file = os.path.basename(src)
            print(f"{i}. {nama_file}")
            
        print("="*50 + "\n")


if __name__ == "__main__":
    main()