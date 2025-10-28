# Flask Chat + RAG Interface

This directory now includes a Flask-based replacement for the original Streamlit prototype. The app provides:

- Modern single-chat UI (sidebar collapsible, single active chat now)
- Normal chat responses until a PDF is uploaded
- PDF upload builds an in-memory (stub) RAG pipeline and switches answers to context-based responses
- Stateless browser; state is kept per user via Flask session + in-memory server dictionary

## File Map

- `app.py` – Flask application entrypoint & API routes
- `templates/base.html` – Base template with shared head/scripts
- `templates/chat.html` – Chat UI template
- `static/css/style.css` – Extracted CSS from original design
- `static/js/chat.js` – Frontend logic (sidebar toggle, chat send, PDF upload)
- `Utils/` – Existing utility modules reused (stub implementations)

## Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Chat UI page |
| `/api/chat` | POST | JSON: `{message}` -> returns answer & messages |
| `/api/upload_pdf` | POST | multipart/form-data `file` (PDF) builds RAG pipeline |
| `/api/reset` | POST | Reset current chat (messages + PDF context) |

## Behavior

- Before any PDF upload: responses are plain echo-style fallback: `(Normal Chat) You asked: ...`.
- After PDF upload: answers use stub RAG pipeline (retrieves naive chunks & responds).
- Only one chat session shown (requirement). Multi-session can be restored later by expanding server store.

## Run Locally

Create / activate an environment and install dependencies:

```bash
pip install -r requirements.txt
python app.py
```

Then open: http://127.0.0.1:5000

## Notes & Next Steps

- Current vector DB & embeddings are placeholder; integrate real embeddings & Qdrant / FAISS for production.
- Add persistence (e.g., Redis) if you deploy behind multiple workers; current in-memory store will reset on restart.
- Add multi-chat sessions by storing a dict of sessions inside each user session and exposing minimal session CRUD APIs.
- Replace fallback normal chat with a real LLM call (OpenAI, Gemini, etc.).
- Harden PDF parsing (`pdf_utils.convert_pdf_to_markdown`) with an actual extractor library (`pypdf`, `unstructured`, etc.).
- Add streaming responses (Server-Sent Events or WebSockets) for better UX.

## Security

- `FLASK_SECRET_KEY` should be set in environment for production.
- Validate and size-limit PDF uploads.

---
Generated migration from Streamlit to Flask while preserving UI aesthetics and requested simplifications.
