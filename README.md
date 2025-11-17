# Voice-heavy RAG agent (Streamlit + LangChain)

This repository is a scaffold for a voice-first agent using Streamlit, LangChain, OpenAI (LLM + embeddings + Whisper), and Postgres + pgvector.

Overview
- Streamlit app (`app.py`) provides a UI for login (placeholder), audio upload/record, transcription, transcript view, and admin pages for RAG documents and Telegram groups.
- `db.py` contains Postgres connection helpers and schema init (pgvector extension, documents, telegram_groups, users).
- `transcribe.py` is a transcription wrapper that uses OpenAI (preferred) and falls back to local Whisper if available.
- `rag.py` contains chunking and embedding helpers and a function to add documents to the vector DB.
- `agent_router.py` is a small LangChain-style router selecting among three tools: RAG, DB CRUD, and Telegram group control.

Setup (Windows PowerShell)

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your secrets (OpenAI key and Postgres connection). Do NOT commit `.env`.

4. Initialize DB and run Streamlit app:

```powershell
python -c "from db import init_db; init_db()"
streamlit run app.py
```

Notes
- The scaffold includes TODOs and placeholders for Telegram integration and full LangChain routing. The DB module creates base tables and extension but advanced similarity queries and embedding dims should be adapted to the embedding model you use.
- I recommend rotating any secrets you shared here.
# Hack-a-thon_ftr