# Multimodal RAG — Talk to your PDFs and Images, with Brains

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)  
[![Flask](https://img.shields.io/badge/Web-Flask-%23000?logo=flask)](https://flask.palletsprojects.com/) [![LangChain](https://img.shields.io/badge/AI-LangChain-1f6feb?logo=chainlink&logoColor=white)](https://python.langchain.com/) [![FAISS](https://img.shields.io/badge/VectorDB-FAISS-ff69b4)](https://github.com/facebookresearch/faiss) [![HuggingFace](https://img.shields.io/badge/Embeddings-Sentence--Transformers-f9a03c?logo=huggingface&logoColor=white)](https://www.sbert.net/) [![Google Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?logo=google)](https://ai.google.dev/) [![DuckDuckGo](https://img.shields.io/badge/Search-DuckDuckGo-FC4C02?logo=duckduckgo&logoColor=white)](https://duckduckgo.com/)

> If it hallucinates, that’s just its creative side showing. We still retrieve receipts. 📎

---

## ⚡ Project Title & Tagline

Multimodal Retrieval‑Augmented Generation (RAG) — a fast, flexible system that grounds LLM answers in your PDFs and images, with optional web search for extra context.

## 🧩 Overview

This app demonstrates a complete, production‑leaning Multimodal RAG stack:

- Ingest PDFs → clean → chunk → embed → index in a vector DB (FAISS)
- Retrieve relevant chunks for your question (MMR/similarity)
- Generate grounded answers with an LLM (Gemini via LangChain)
- Image uploads are embedded with CLIP (sentence‑transformers) and auto‑labeled to guide retrieval and web search
- Clean Flask web UI with chat history, per‑chat PDF/image attachments, and simple session management

What you get: concise answers with receipts (source snippets), and a path to grow into enterprise‑grade RAG.

---

## 🏗️ Architecture

```
User → Flask UI → (Chat Controller) → RAG Pipeline
                          │
                          ├─ PDF Ingestion → Clean → Chunk (LangChain) → Embeddings (Google/HF) → FAISS Index
                          │                                                   │
                          │                                                   └─ Retriever (MMR / similarity)
                          │
                          ├─ Image Upload → CLIP Embeddings + Zero‑shot Labels → Query Augmentation
                          │
                          └─ Optional Web Search (DuckDuckGo) → Extra Context → LLM (Gemini) Answer + Sources
```


---

## 💾 Supported Modalities

- Text (chat) — normal LLM chat with optional web‑augmented context
- PDF — fully grounded RAG over uploaded documents
- Image — embedded via CLIP; predicts coarse labels to steer retrieval/search

---

## 🔧 Setup & Installation

### 1) Clone and configure

Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set environment variables (put these in your shell or a local .env):

- GOOGLE_API_KEY: required for Gemini LLM and Google embeddings
- FLASK_SECRET_KEY: any random string for session security
- Optional tuning:
  - USE_GOOGLE_EMBEDDINGS=1 to prefer Google Generative AI embeddings (falls back to HF if quota issues)
  - IMAGE_EMBEDDER_MODEL=clip-ViT-B-32 (default)
  - IMAGE_EMBED_WARMUP=1 to pre‑download the image model on boot

Example .env:

```ini
GOOGLE_API_KEY="your_google_api_key"
FLASK_SECRET_KEY="replace_me"
USE_GOOGLE_EMBEDDINGS=1
IMAGE_EMBEDDER_MODEL=clip-ViT-B-32
IMAGE_EMBED_WARMUP=1
```

### 2) Run the app

```bash
python app.py
```

Open http://localhost:5000 and start chatting. Upload a PDF to enable RAG. Upload an image to get smart labels that guide retrieval.

---

## 🚀 Usage Instructions

### Web UI

1) Ask a question (normal chat)
2) Upload a PDF to ground answers in your document
3) Upload images (JPG/PNG) — the app predicts labels (e.g., “cat”, “invoice”, “diagram”) and uses them to augment queries/web search

### REST API (handy for testing/automation)

- Chat

```bash
curl -s -X POST http://localhost:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What does the document say about attention mechanisms?"}'
```

- Upload PDF

```bash
curl -s -X POST http://localhost:5000/api/upload_pdf \
  -F file=@/path/to/your.pdf
```

- Upload Image

```bash
curl -s -X POST http://localhost:5000/api/upload_image \
  -F file=@/path/to/image.jpg
```

- Remove an image

```bash
curl -s -X POST http://localhost:5000/api/remove_image \
  -H 'Content-Type: application/json' \
  -d '{"name":"image.jpg"}'
```

Other routes (used by the UI): new/delete/rename chat, set active chat, reset active, remove pdf.

---

## 💡 Key Features

- PDF RAG, end‑to‑end:
  - PyPDFLoader → text cleaning → recursive chunking
  - Google Generative AI embeddings with HF fallback → FAISS index
  - Retriever (MMR by default) → Gemini LLM generation with cited snippets
- Multimodal twist:
  - Image embeddings and zero‑shot labels via sentence‑transformers (CLIP)
  - Labels automatically steer web search and context stuffing
- Web augmentation:
  - DuckDuckGo search results converted to LangChain Documents for extra context
- Resilient by default:
  - Google quota triggers automatic fallback to HuggingFace embeddings
  - Defensive error handling and friendly user messages
- Clean UX:
  - Flask + Jinja templates + vanilla JS; chat history is per‑session with multiple chats

---

## 📁 Repository Structure

```
.
├── app.py                        # Flask app & API routes (chat, upload PDF/image, sessions)
├── History/
│   └── history.py                # In‑memory chat sessions + vector/embedding introspection
├── Utils/
│   ├── pdf_utils.py              # PDF load + cleaning
│   ├── chunking.py               # Recursive chunking
│   ├── embedding.py              # Google/HF embeddings with smart fallback
│   ├── vector_db.py              # FAISS index wrapper
│   ├── retriever.py              # Retriever wrapper (MMR/similarity)
│   ├── rag.py                    # RetrievalQA chain + augmented answering
│   ├── image_embedding.py        # CLIP image embeddings + zero‑shot labels
│   └── web_search.py             # DuckDuckGo → LangChain Documents
├── templates/                    # Jinja templates (chat UI)
├── static/                       # CSS/JS
├── requirements.txt
├── LANGCHAIN_MIGRATION.md        # Notes on the recent migration to real LC components
└── README.md                     # You are here
```

---

## 🧠 Tech Stack

- Backend/UI: Flask, Jinja, vanilla JS/CSS
- LLM: Google Gemini (langchain-google-genai)
- Embeddings: Google Generative AI (preferred) with HuggingFace sentence‑transformers fallback
- Vector DB: FAISS (langchain_community.vectorstores)
- PDF: PyPDFLoader + text cleaning and recursive chunking
- Images: sentence‑transformers CLIP ViT‑B/32 for embeddings and zero‑shot labels
- Web Search: DuckDuckGo (duckduckgo-search)

---

## 🛣️ Roadmap / Future Work

- Persist FAISS index to disk (save/load per chat)
- Hybrid retrieval (BM25 + dense) and/or reranking
- Streaming responses and tool‑use telemetry
- GPU acceleration (CUDA) for faster embeddings
- Support audio/video modalities (ASR, frame sampling)
- Multi‑tenant persistence (DB for sessions and attachments)
- Add authentication and rate limiting

---

## 🔍 How It Works (TL;DR)

- Document path: PDF → clean text → chunk → embed → FAISS → retriever → LLM
- Image path: image → CLIP embedding → predict labels → augment query/web search → LLM
- Answering: retrieve top chunks (and optional web snippets), stuff into a structured prompt, and ask Gemini to synthesize a clear, source‑aware answer. Top sources are appended for transparency.

---

## 🤝 Contributing

PRs welcome! Please:
- Keep code modular and typed where feasible
- Prefer small, focused changes with clear logs
- Add brief docs/tests when changing public behavior

---

## 📜 License

No explicit license is included yet. If you plan to use or extend this project, consider adding a LICENSE file (MIT/Apache‑2.0 recommended).

---

## 🔎 Behind the Scenes

In plain terms: we take your documents, chop them into bite‑sized chunks, and turn each chunk into a number soup (embeddings). When you ask a question, we search which soups taste most like your question, serve those to the LLM as context, and let it cook up an answer. If you toss in an image, we use CLIP to guess labels (like “invoice” or “diagram”), which helps the system decide what to search and which text chunks to bring to the party. The result: fewer creative tangents, more grounded answers.
