# Environment Variables — Canonical Reference

## How Env Loading Works

`config.py` runs `load_environment()` at import time, which loads `.env` then `.env.local` (overrides). All backend and frontend files that import from `config.py` share this mechanism.

Scripts that bypass `config.py` (`upload_to_pinecone.py`, `query_rag.py`, `test.py`) call `load_dotenv()` independently — this is a known inconsistency to fix in Phase 2.

## Required Variables

| Variable | Used In | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | `api.py`, `upload_to_pinecone.py`, `query_rag.py` | OpenAI API access for embeddings, chat, and metadata extraction |
| `PINECONE_API_KEY` | `api.py`, `upload_to_pinecone.py`, `query_rag.py` | Pinecone vector DB access |
| `PINECONE_INDEX_NAME` | `api.py`, `upload_to_pinecone.py`, `query_rag.py` | Pinecone index name (default: `iitm-modules-rag` in api.py only) |
| `SUPABASE_URL` | `supabase_client.py` | Supabase project URL |
| `SUPABASE_ANON_KEY` | `supabase_client.py` | Supabase anon/public key (for client-side) |
| `SUPABASE_SERVICE_ROLE_KEY` | `supabase_client.py` | Supabase service role key (for backend operations) |

## Optional Variables (Have Defaults)

| Variable | Default | Used In | Purpose |
|----------|---------|---------|---------|
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | `upload_to_pinecone.py` | Embedding model for chunk ingestion |
| `OPENAI_METADATA_MODEL` | `gpt-4o-mini` | `upload_to_pinecone.py` | LLM model for metadata extraction |
| `API_BASE_URL` | `https://iitm-curriculem-intelligence-layer.onrender.com` | `app.py` | Backend URL for the Streamlit frontend |

## Quick Start (.env.local for local dev)

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=iitm-modules-rag
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
```

## Known Inconsistencies (To Fix in Phase 2)

1. `PINECONE_INDEX_NAME` default exists in `api.py` but not in `upload_to_pinecone.py` or `query_rag.py`.
2. `upload_to_pinecone.py`, `query_rag.py`, and `test.py` call `load_dotenv()` directly instead of importing from `config.py`.
3. No validation/error message if `OPENAI_API_KEY` or `PINECONE_API_KEY` are missing — they fail at runtime with a cryptic OpenAI/Pinecone error.
