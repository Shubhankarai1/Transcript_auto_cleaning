# Architecture: System Boundaries & Module Structure

## Layer Map

```
┌──────────────────────────────────────────────────────────────────┐
│                    FRONTEND (app.py)                             │
│  Streamlit SPA  ──HTTP──>  api.py  (FastAPI)                    │
│                                                                  │
│  Responsibilities: layout, scope state, chat UI, source display │
│  No direct DB / Pinecone / OpenAI calls                         │
└──────────────────────────┬───────────────────────────────────────┘
                           │  HTTP + JSON
┌──────────────────────────▼───────────────────────────────────────┐
│              APPLICATION BACKEND (api.py → to be split)          │
│                                                                  │
│  FastAPI         → routing, request/response, CORS              │
│  RAG pipeline    → rewrite, expand, HyDE, embed, search, filter │
│  Profile CRUD    → Supabase user profiles                       │
│  Content catalog → reads input/ filesystem for level/module list │
│                                                                  │
│  Currently monolithic — next step is to split into modules.     │
└───────┬─────────────────────┬─────────────────────┬──────────────┘
        │                     │                     │
        │  OpenAI API         │  Pinecone gRPC      │  Supabase REST
┌───────▼──────┐   ┌─────────▼──────────┐   ┌──────▼───────────┐
│ AI KNOWLEDGE │   │  VECTOR STORE      │   │  USER DATA       │
│ LAYER        │   │  (Pinecone)        │   │  (Supabase)      │
│              │   │                    │   │                  │
│ Embeddings   │   │  iitm-modules-rag  │   │  profiles table  │
│ GPT-4o-mini  │   │  namespace: none   │   │                  │
│ text-embed-3 │   │  metadata: level,  │   │                  │
│              │   │  category, module, │   │                  │
│              │   │  session, chunk,   │   │                  │
│              │   │  topic, keywords,  │   │                  │
│              │   │  content_id,       │   │                  │
│              │   │  module_path       │   │                  │
└──────────────┘   └────────────────────┘   └──────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│            INGESTION PIPELINE (offline / local)                  │
│                                                                  │
│  main.py + utils.py  ──>  output/*_final_cleaned.txt            │
│                        ──>  rag_chunks/<module>/*.txt           │
│  upload_to_pinecone.py ──>  Pinecone (embed + upsert)           │
│                                                                  │
│  External call: Ollama (localhost:11434) — no cloud dependencies │
│  No connection to api.py, app.py, or Supabase                   │
└──────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### AI Knowledge Layer
- **OpenAI** — embeddings and LLM inference (chat, metadata extraction)
- **Pinecone** — vector index serving RAG retrieval
- **Owned by:** `api.py` (currently), `upload_to_pinecone.py`, `query_rag.py`

### Application Backend (`api.py`)
- HTTP API surface for the frontend
- Orchestrates RAG pipeline (rewrite → expand → HyDE → embed → search → filter → answer)
- Content catalog (reads `input/` filesystem for levels/modules/sessions)
- Supabase profile CRUD
- Currently contains all the above in one file. Next phase should split it.

### Frontend (`app.py`)
- Pure Streamlit UI. No business logic.
- Calls backend API only. Does not call OpenAI/Supabase/Pinecone directly.
- Scope state management (level/module selection, chat history reset on scope change).

### Supabase
- User profile persistence only (no other tables used yet).
- `supabase_client.py` is a factory — no query logic lives there.

### Ingestion Pipeline (`main.py`, `utils.py`)
- Fully offline. Local Ollama for cleaning.
- Purely CLI. No HTTP API, no frontend.
- Produces `rag_chunks/` which `upload_to_pinecone.py` reads and pushes to Pinecone.

## Current File -> Layer Mapping

| File | Layer | Responsibilities |
|------|-------|-----------------|
| `main.py` | Ingestion | Orchestrates transcript cleaning, session caching, chunking |
| `utils.py` | Ingestion | I/O, path parsing, cleaning, chunking, cache, hash |
| `upload_to_pinecone.py` | Ingestion | Loads rag_chunks/, extracts metadata via OpenAI, embeds, upserts to Pinecone |
| `api.py` | Backend | **Monolithic** — routes + RAG pipeline + profile CRUD + content catalog |
| `query_rag.py` | Backend (CLI) | Standalone RAG CLI — ~60% duplicate logic with `api.py` |
| `retrieval_utils.py` | Backend | Filter detection & combining (shared by api.py and query_rag.py) |
| `config.py` | Shared | Env variable loading |
| `supabase_client.py` | Backend | Supabase client factory |
| `app.py` | Frontend | Streamlit UI — chat, scope selection, source display |

## What Should Be Split Out of `api.py` Next

`api.py` currently mixes 4 concerns. In the next phase, split into:

```
api/
  __init__.py          → FastAPI app factory, CORS, lifespan
  routes/
    __init__.py
    chat.py            → POST /chat
    profiles.py        → GET/POST/PUT /v1/profile
    catalog.py         → GET /levels, /modules, /sessions
    health.py          → GET /, /health
  rag/
    __init__.py
    pipeline.py        → rewrite, expand, HyDE, embed, dedup, post-filter
    search.py          → Pinecone query, fallback cascade, filter combining
    sources.py         → build_source_entry, format_sources, dedup_matches
  llm/
    __init__.py
    client.py          → OpenAI client init, chat completion helpers
  config.py            → shared config (or keep root config.py)
```

`query_rag.py` should be refactored to import from `api/rag/` instead of duplicating logic.

Files that stay at root level:
- `main.py`, `utils.py` — ingestion pipeline (separate concern, no API dependency)
- `upload_to_pinecone.py` — ingestion upload (separate concern)
- `app.py` — frontend (separate concern)
- `config.py`, `retrieval_utils.py`, `supabase_client.py` — shared utilities

## Phase 1 Status: Done vs Pending

| Item | Status |
|------|--------|
| Hierarchy-aware ingestion (level/category/module/session) | Done |
| `module_path` and `content_id` parsed and propagated | Done |
| RAG chunk generation with full metadata | Done |
| Pinecone metadata includes all hierarchy fields | Done |
| FastAPI backend with RAG pipeline | Done |
| Level/module scoped retrieval | Done |
| Streamlit frontend with scope selection | Done |
| Supabase profile CRUD endpoints | Done |
| **api.py split into modules** | **Pending** |
| **query_rag.py consolidated with api/rag/** | **Pending** |
| **Config unification (PINECONE_INDEX_NAME, env loading)** | **Pending** |
| **Frontend scope selector updated for 3-level hierarchy** | **Pending** |
| **Remove dead `from api import app` in main.py** | **Pending** |
