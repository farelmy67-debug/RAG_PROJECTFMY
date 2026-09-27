# 🤖 Asisten AI PPKD Jakarta Barat — RAG Chatbot

Chatbot berbasis **Retrieval-Augmented Generation (RAG)** yang menjawab pertanyaan seputar program pelatihan, syarat pendaftaran, dan informasi operasional **Pusat Pelatihan Kerja Daerah (PPKD) Jakarta Barat**, berdasarkan dokumen resmi lembaga — bukan hasil karangan model.

> Mini Project 2 — AI Bootcamp KODE by Hacktiv8

---

## 📌 Daftar Isi

- [Use Case & Latar Belakang](#-use-case--latar-belakang)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur & Cara Kerja](#-arsitektur--cara-kerja)
- [Justifikasi Pemilihan Model](#-justifikasi-pemilihan-model)
- [Structured Output & Validasi](#-structured-output--validasi)
- [Struktur Prompt / System Instruction](#-struktur-prompt--system-instruction)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Konfigurasi RAG](#-konfigurasi-rag)
- [Pengujian (Test Set)](#-pengujian-test-set)
- [Estimasi Biaya API](#-estimasi-biaya-api)
- [Batasan Sistem](#-batasan-sistem)
- [Struktur Folder](#-struktur-folder)

---

## 🎯 Use Case & Latar Belakang

PPKD Jakarta Barat menyediakan puluhan program pelatihan kerja gratis untuk warga DKI Jakarta, namun informasinya tersebar di berbagai dokumen resmi (profil lembaga, daftar kejuruan, syarat pendaftaran, jadwal operasional). Warga yang ingin bertanya sering kesulitan menemukan jawaban cepat dan akurat.

Chatbot ini dibangun untuk menjawab pertanyaan seputar:
- Daftar kejuruan/program pelatihan (Reguler, Vokasi Khusus/Green Jobs, Mobile Training Unit)
- Syarat dan cara pendaftaran
- Biaya (gratis untuk warga ber-KTP/domisili DKI Jakarta)
- Jam operasional dan lokasi kantor

Sumber jawaban **100% berbasis dokumen resmi** yang di-*load* ke sistem — chatbot dilarang mengarang atau berasumsi status pendaftaran yang tidak tertulis eksplisit di dokumen.

---

## ✨ Fitur Utama

- **RAG Pipeline lengkap**: ingestion → chunking → embedding → retrieval → generation
- **Structured Output** dengan skema Pydantic (jawaban, status, tingkat keyakinan)
- **Anti-halusinasi**: jawaban jujur menyatakan "tidak ditemukan" jika info tidak ada di dokumen
- **Sumber rujukan**: setiap jawaban menyertakan nama file dokumen yang jadi dasar jawaban
- **Antarmuka Streamlit** interaktif dengan quick-suggestion buttons dan riwayat chat
- **Mode CLI** (`rag_chatbot.py`) untuk debugging/pengujian cepat dengan log chunk yang diambil retriever

---

## 🏗️ Arsitektur & Cara Kerja

```
┌─────────────┐    ┌───────────┐    ┌────────────────┐    ┌─────────────┐
│  Dokumen PDF │ →  │  Chunking  │ →  │   Embedding    │ →  │  ChromaDB    │
│ (knowledge_  │    │(Recursive- │    │ (Multilingual  │    │ (Vector      │
│  docs/)      │    │ Splitter)  │    │  MiniLM-L12)   │    │  Store)      │
└─────────────┘    └───────────┘    └────────────────┘    └──────┬──────┘
                                                                    │
                                                             Retrieval (top-k)
                                                                    │
┌──────────────┐    ┌────────────────┐    ┌───────────────┐        │
│  Jawaban      │ ←  │  LLM (Groq +   │ ←  │ System Prompt  │ ←─────┘
│  Terstruktur  │    │  gpt-oss-120b) │    │ + Konteks      │
│  (Pydantic)   │    │                │    │ + Pertanyaan   │
└──────────────┘    └────────────────┘    └───────────────┘
```

**Alur teknis:**
1. **Ingestion** — seluruh file PDF di `knowledge_docs/` dimuat per halaman via `PyMuPDF4LLMLoader` (metadata sumber & halaman tetap terjaga)
2. **Chunking** — dokumen dipecah dengan `RecursiveCharacterTextSplitter`, menggunakan separator kustom yang menghormati struktur penomoran (`1.`, `2.`, `A.`, `B.`, dst) dan poin bullet (`-`) supaya daftar kejuruan tidak terpotong di tengah
3. **Embedding** — setiap chunk diubah jadi vector pakai model multilingual, disimpan di **ChromaDB** (in-memory/local)
4. **Retrieval** — saat ada pertanyaan, sistem mengambil chunk paling relevan (top-k)
5. **Generation** — chunk + system prompt + pertanyaan dikirim ke LLM Groq, dipaksa menghasilkan output sesuai skema Pydantic
6. **Validasi & Grounding** — jawaban divalidasi, dan daftar nama file sumber ditampilkan ke pengguna sebagai bukti rujukan

---

## 🧠 Justifikasi Pemilihan Model

| Komponen | Model / Teknologi | Alasan |
|---|---|---|
| **LLM Engine** | `openai/gpt-oss-120b` via **Groq API** | Inferensi sangat cepat (Groq LPU, 500+ token/s), context window besar (131K token), harga rendah, dan mendukung *structured output* native |
| **Embedding Model** | `paraphrase-multilingual-MiniLM-L12-v2` | Dilatih khusus untuk pemahaman semantik multi-bahasa, akurat untuk dokumen Bahasa Indonesia, ringan untuk dijalankan lokal |
| **Vector Store** | ChromaDB | Ringan, jalan lokal (tidak perlu server terpisah), cepat untuk pencarian cosine similarity pada skala dokumen kecil-menengah |

**Batasan sistem yang diketahui:**
- Rate limit Groq (tier gratis): sekitar **8.000 TPM (Token Per Minute)** — dikelola dengan mengatur ukuran chunk dan jumlah `k` retrieval agar tidak melebihi kuota
- Context window model: 131.072 token (jauh di atas kebutuhan chunk yang di-retrieve)

---

## 📐 Structured Output & Validasi

Chatbot **tidak** mengembalikan teks bebas. Setiap jawaban dipaksa mengikuti skema Pydantic berikut (lewat `model.with_structured_output(...)`):

```python
class JawabanChatbot(BaseModel):
    jawaban: str                                              # jawaban lengkap ke pengguna
    status: Literal["terjawab", "tidak_ditemukan", "di_luar_topik"]
    tingkat_keyakinan: Literal["tinggi", "sedang", "rendah"]
```

**Lapisan validasi tambahan** (`validasi_jawaban()`):
- Jika status bukan `"terjawab"` dan field jawaban kosong → diisi otomatis dengan pesan "informasi tidak ditemukan"
- Jika status `"terjawab"` tapi keyakinan `"rendah"` → sistem menambahkan catatan otomatis agar pengguna memverifikasi ke sumber resmi

**Penanganan error**: seluruh pemanggilan RAG chain dibungkus `try-except (ValidationError, Exception)`. Jika terjadi kegagalan parsing/validasi, chatbot menampilkan pesan fallback yang aman ("Maaf, terjadi kendala...") alih-alih crash atau menampilkan jawaban tidak valid ke pengguna.

---

## 📝 Struktur Prompt / System Instruction

System prompt (`system_prompt.md`) mendefinisikan aturan ketat berbasis grounding:

1. Jawaban **wajib** didasarkan pada konteks yang diberikan, tidak boleh mengarang
2. Jika diminta daftar (kejuruan/program/syarat), **seluruh** poin di konteks harus disebutkan — tidak boleh terpotong
3. **Larangan menyimpulkan status pendaftaran** — chatbot tidak boleh bilang "sedang dibuka" kecuali tanggalnya eksplisit tertulis di dokumen
4. Jika info tidak tersedia di dokumen, chatbot **wajib jujur** menyampaikan hal itu (prinsip anti-halusinasi)
5. Pertanyaan di luar topik PPKD Jakarta Barat dijawab dengan sopan bahwa fokus chatbot hanya seputar PPKD Jakarta Barat
6. Untuk status pendaftaran terkini, pengguna diarahkan ke kanal resmi (`linktr.ee/klikppkdjb`, IG `@ppkd_jakarta_barat`)

Selain itu, `app.py` menambahkan **instruksi tambahan khusus** di runtime untuk memastikan chatbot menampilkan **seluruh struktur kejuruan** (4 rumpun pelatihan reguler, program vokasi khusus/*green jobs*, dan 8 kejuruan Mobile Training Unit) tanpa terpotong hanya di rumpun pertama.

---

## ⚙️ Instalasi

**Prasyarat:** Python 3.10+ dan API key dari [Groq Console](https://console.groq.com)

```bash
# 1. Clone repository
git clone https://github.com/farelmy67-debug/RAG_PROJECTFMY.git
cd RAG_PROJECTFMY

# 2. Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Siapkan environment variable
cp .env.example .env
# lalu edit .env dan isi GROQ_API_KEY=<api-key-kamu>
```

---

## ▶️ Cara Menjalankan

### Mode Streamlit (Web UI) — direkomendasikan untuk demo

```bash
streamlit run app.py
```
Buka browser ke `http://localhost:8501`. Vector store dibangun otomatis saat pertama kali dijalankan (`@st.cache_resource`), sehingga *reload* berikutnya lebih cepat.

### Mode CLI (untuk debugging/pengujian cepat)

```bash
python rag_chatbot.py
```
Mode ini menampilkan log chunk yang diambil retriever untuk setiap pertanyaan (berguna untuk memverifikasi kualitas retrieval), lalu jawaban terstruktur beserta status dan tingkat keyakinan. Ketik `keluar`, `exit`, atau `quit` untuk berhenti.

---

## 🔧 Konfigurasi RAG

| Parameter | `app.py` (Streamlit) | `rag_chatbot.py` (CLI) |
|---|---|---|
| Chunk size | 900 | 1200 |
| Chunk overlap | 150 | 250 |
| Top-k retrieval | 30 | 30 |
| Separator kustom | Penomoran (`1.`–`8.`, `A.`–`D.`), baris baru ganda, baris baru, poin `-` | Penomoran (`1.`–`10.`, `A.`–`E.`), baris baru ganda, baris baru, poin `-` |

> ⚠️ **Catatan:** nilai `chunk_size`/`chunk_overlap` antara mode Streamlit dan CLI saat ini **belum identik** (lihat bagian [Batasan Sistem](#-batasan-sistem)). Konfigurasi di `app.py` adalah yang digunakan pada deliverable utama (Web UI).

---

## ✅ Pengujian (Test Set)

Dilakukan pengujian terhadap **8 pertanyaan uji**, mencakup kasus normal, edge case, dan out-of-scope (di luar topik PPKD).

| # | Kategori | Contoh Pertanyaan | Hasil |
|---|---|---|---|
| 1–5 | Normal | Syarat pendaftaran, daftar kejuruan, biaya, jam operasional, lokasi | ✅ Terjawab akurat, sumber ditampilkan |
| 6 | Edge case | Pertanyaan tentang informasi yang tidak ada di dokumen | ✅ Status `tidak_ditemukan`, jujur tanpa halusinasi |
| 7–8 | Out-of-scope | Pertanyaan tak berkaitan (mis. resep makanan, jadwal piket) | ✅ Status `di_luar_topik`, menolak dengan sopan |

**Hasil keseluruhan: 100% lulus evaluasi** — chatbot tidak berhalusinasi pada seluruh kasus uji, termasuk saat informasi tidak tersedia atau pertanyaan di luar cakupan topik.

> 📎 Detail lengkap 8 pertanyaan uji beserta jawaban aktual disarankan dilampirkan sebagai file terpisah (`test_set.md` atau di slide presentasi) untuk keperluan penilaian.

---

## 💰 Estimasi Biaya API

Model `openai/gpt-oss-120b` di Groq per Agustus 2026 diberi harga:

| | Harga |
|---|---|
| Input | $0.15 / 1 juta token |
| Output | $0.75 / 1 juta token |

**Estimasi biaya per interaksi** (asumsi ~1.500 token konteks + prompt, ~300 token output):
- Input: ~1.500 token × $0.15/1M ≈ **$0.000225**
- Output: ~300 token × $0.75/1M ≈ **$0.000225**
- **Total ≈ $0.00045 per pertanyaan** (sangat murah, cocok untuk skala penggunaan publik)

Untuk 1.000 pertanyaan/hari, estimasi biaya harian sekitar **$0.45**, dengan catatan tier gratis Groq memiliki rate limit ±8.000 TPM yang perlu dipantau pada traffic tinggi.

---

## ⚠️ Batasan Sistem

- **Rate limit Groq**: tier gratis dibatasi ±8.000 token per menit; traffic tinggi bisa memicu *throttling*
- **Mismatch parameter chunking**: `app.py` menggunakan `chunk_size=900/overlap=150`, sedangkan `rag_chatbot.py` menggunakan `chunk_size=1200/overlap=250` — belum disamakan sepenuhnya antara mode Web dan CLI
- **Vector store tidak persisten**: ChromaDB dibangun ulang (`reset_collection()`) setiap kali aplikasi restart, sehingga tidak ada penyimpanan index permanen di disk
- **Cakupan topik terbatas**: chatbot hanya menjawab hal-hal yang tercantum secara eksplisit di `knowledge_docs/`; tidak bisa memberi info real-time (misal status gelombang pendaftaran aktif) — untuk itu pengguna diarahkan ke kanal resmi PPKD

---

## 📁 Struktur Folder

```
RAG_PROJECTFMY/
├── app.py                  # Aplikasi utama (Streamlit Web UI)
├── rag_chatbot.py           # Versi CLI (untuk debugging & testing pipeline)
├── requirements.txt         # Daftar dependencies Python
├── system_prompt.md         # System instruction untuk LLM
├── .env.example              # Contoh format environment variable
├── .streamlit/
│   └── config.toml          # Tema tampilan Streamlit (dark navy)
├── knowledge_docs/          # Dokumen resmi PPKD Jakarta Barat (sumber RAG)
└── assets/                  # Logo & aset visual
```

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `LangChain` · `Groq API (openai/gpt-oss-120b)` · `ChromaDB` · `HuggingFace Embeddings` · `Pydantic` · `PyMuPDF4LLM`

---

*Dibuat sebagai bagian dari Mini Project 2 — AI Bootcamp KODE by Hacktiv8.*
