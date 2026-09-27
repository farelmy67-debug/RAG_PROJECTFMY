import os
import glob
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. KONFIGURASI HALAMAN STREAMLIT
st.set_page_config(
    page_title="Asisten AI PPKD Jakarta Barat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS UNTUK TAMPILAN MODERN
st.markdown("""
<style>
    .main-title {
        color: #1E3A8A;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1rem;
        margin-bottom: 20px;
    }
    .info-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        margin-bottom: 15px;
    }
    div.stButton > button {
        border-radius: 20px;
        border: 1px solid #2563EB;
        color: #2563EB;
        background-color: #EFF6FF;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #2563EB;
        color: white;
        border-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# 3. KONFIGURASI RAG & MODEL
CHAT_MODEL = "openai/gpt-oss-120b"
COLLECTION_NAME = "ppkd_jakbar"
KNOWLEDGE_DIR = "./knowledge_docs"
SYSTEM_PROMPT_PATH = "./system_prompt.md"

@st.cache_resource
def inisialisasi_rag():
    load_dotenv()
    
    model = ChatGroq(
        model=CHAT_MODEL, 
        temperature=0, 
        max_tokens=4096
    )
    
    path_file = sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf")))
    daftar_dokumen = []
    for path in path_file:
        loader = PyMuPDF4LLMLoader(file_path=path, mode="page", use_layout=False)
        daftar_dokumen.extend(loader.load())
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
        separators=[
            "\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. ", "\n6. ", "\n7. ", "\n8. ",
            "\nA. ", "\nB. ", "\nC. ", "\nD. ",
            "\n\n", "\n", " ", ""
        ]
    )
    potongan = splitter.split_documents(daftar_dokumen)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings)
    vectorstore.reset_collection()
    vectorstore.add_documents(potongan)
    
    # K dinaikkan ke 20 agar semua potongan dokumen langsung terbawa
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    base_system_prompt = ""
    if os.path.exists(SYSTEM_PROMPT_PATH):
        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            base_system_prompt = f.read()
        
    system_prompt_lengkap = base_system_prompt + """

---
ATURAN KETAT KELENGKAPAN PROGRAM & KEJURUAN:
1. **Dilarang Menyimpulkan Status Pendaftaran**:
   - JANGAN PERNAH menyimpulkan bahwa program "saat ini sedang dibuka" atau "sedang buka pendaftaran".
   - Gunakan kalimat pembuka netral: *"Berikut adalah daftar kejuruan dan program pelatihan yang diselenggarakan oleh PPKD Jakarta Barat berdasarkan dokumen resmi:"*.

2. **WAJIB Menyebutkan SELURUH Program (Tanpa Terlewat/Terpotong)**:
   Jika pertanyaan berkaitan dengan kejuruan, program, atau pelatihan, Anda WAJIB menyajikan seluruh struktur Bagian 4 secara utuh:
   - **A. Pelatihan Reguler / Kejuruan Utama (4 Rumpun Lengkap)**:
     1. Otomotif & Teknik Industri
     2. Teknologi Informasi, Digital & Komunikasi
     3. Tata Boga, Jasa & Keamanan
     4. Kecantikan, Fashion & Kesehatan
   - **B. Program Kerja Sama / Vokasi Khusus / Green Jobs**
   - **C. Program Mobile Training Unit (MTU) / Pelatihan Tingkat Kelurahan** (Sebutkan seluruh 8 kejuruan MTU).

3. Jangan memotong daftar hanya pada Rumpun 1. Jika konteks memuat Rumpun 2, 3, 4, Vokasi Khusus, dan MTU, semuanya WAJIB ditampilkan.
"""
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_lengkap),
        ("human", "Konteks Dokumen:\n{context}\n\nPertanyaan: {question}"),
    ])
    
    # Fungsi pembantu untuk menggabungkan konteks
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    generation_chain = (
        {
            "context": base_retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )
    return RunnableParallel(answer=generation_chain, sources=base_retriever)

rag_chain = inisialisasi_rag()

# 4. SIDEBAR INFOGRAFIS
with st.sidebar:
    # Ganti baris st.image dengan nama file gambar kamu
    st.image("assets/ppkd_logo.png", width=100)
    st.title("PPKD Jakarta Barat")
    st.caption("Pusat Pelatihan Kerja Daerah Dinas Tenaga Kerja, Transmigrasi dan Energi Provinsi DKI Jakarta")
    
    
    st.markdown("### 📌 Informasi Penting")
    st.info("💡 **100% Gratis** untuk warga DKI Jakarta (ber-KTP DKI / Domisili DKI).")
    
    st.markdown("### 📍 Alamat Kantor")
    st.write("Jl. Kamal Raya No. 2, Kel. Tegal Alur, Kec. Kalideres, Jakarta Barat.")
    st.write("⏰ **Jam Operasional:** Senin - Jumat, 08.00 - 15.00 WIB")
    
    st.divider()
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. HEADER UTAMA & METRICS
st.markdown('<p class="main-title">🤖 Asisten AI PPKD Jakarta Barat</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Dapatkan informasi lengkap seputar program pelatihan, syarat pendaftaran, dan jadwal operasional secara cepat.</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Program Pelatihan", value="20+ Kejuruan", delta="Reguler & MTU")
with col2:
    st.metric(label="Biaya Pelatihan", value="Rp 0 (Gratis)", delta="Biaya APBD/APBN")
with col3:
    st.metric(label="Sertifikasi", value="BNSP & PPKD", delta="Standar Industri")

st.divider()

# ============================================================
# 6. INISIALISASI SESSION STATE & QUICK SUGGESTIONS
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya Asisten AI Resmi PPKD Jakarta Barat. Ada yang bisa saya bantu terkait program pelatihan atau pendaftaran?"}
    ]

prompt_input = None

if len(st.session_state.messages) <= 1:
    st.write("👉 **Pertanyaan yang sering ditanyakan:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    
    with q_col1:
        if st.button("📋 Apa saja syarat pendaftarannya?", use_container_width=True):
            prompt_input = "Apa saja syarat pendaftaran pelatihan di PPKD Jakarta Barat?"
    with q_col2:
        if st.button("🎓 Apa saja daftar kejuruan di PPKD JB?", use_container_width=True):
            prompt_input = "Apa saja daftar kejuruan dan program pelatihan yang diselenggarakan oleh PPKD Jakarta Barat?"
    with q_col3:
        if st.button("🗣️ Apakah ada program pelatihan bahasa asing?", use_container_width=True):
            prompt_input = "Apakah ada program pelatihan bahasa asing di PPKD Jakarta Barat?"

if not prompt_input:
    prompt_input = st.chat_input("Ketik pertanyaan Anda di sini...")

# ============================================================
# 7. TAMPILKAN RIWAYAT CHAT LAMA
# ============================================================
for message in st.session_state.messages:
    avatar = "assets/logo_assisten.png" if message["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ============================================================
# 8. EKSEKUSI PEMROSESAN PESAN BARU (RAG CHAIN)
# ============================================================
if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="assets/logo_assisten.png"):
        with st.spinner("Mencari data dari dokumen resmi PPKD Jakbar..."):
            res = rag_chain.invoke(prompt_input)
            jawaban = res["answer"]
            
            sumber_list = list(set([os.path.basename(doc.metadata.get("source", "")) for doc in res["sources"]]))
            
            st.markdown(jawaban)
            
            if sumber_list:
                with st.expander("📚 Lihat Sumber Dokumen Rujukan"):
                    for src in sumber_list:
                        st.write(f"- `{src}`")
                        
    st.session_state.messages.append({"role": "assistant", "content": jawaban})
    st.rerun()