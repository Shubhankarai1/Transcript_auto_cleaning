# PRD: IITM Curriculum Intelligence Layer

## Objective

Build and maintain a lightweight RAG application that lets users ask questions over IITM curriculum lecture transcripts.

The system has two major parts:

1. A transcript processing pipeline that converts raw lecture transcripts into structured, RAG-ready chunks.
2. A FastAPI + Pinecone + OpenAI backend, with a Streamlit frontend, that retrieves relevant chunks and generates grounded answers with citations.

## Current Status

The project has evolved from a transcript-cleaning utility into a deployable curriculum Q&A system.

Implemented:

- Session-aware transcript cleaning and chunking.
- RAG chunk generation under `rag_chunks/`.
- Pinecone upload workflow.
- FastAPI backend in `api.py`.
- Streamlit frontend in `app.py`.
- Render-compatible API routing.
- Lightweight deployment dependencies without `sentence-transformers` or PyTorch.
- OpenAI-based reranking layer after Pinecone retrieval.
- Source citations in generated answers.
- Debug logging around retrieval, reranking, and OpenAI calls.

## Users

- Students reviewing IITM course content.
- Instructors or mentors answering curriculum-specific questions.
- Developers debugging retrieval quality and deployment behavior.

## Core User Flow

1. User opens the Streamlit app.
2. User selects either:
   - Global search across all content.
   - Filtered search by module and session.
3. Frontend sends a `POST /chat` request to the FastAPI backend.
4. Backend rewrites the query for retrieval.
5. Backend embeds the rewritten query with OpenAI embeddings.
6. Backend retrieves high-recall candidate chunks from Pinecone.
7. Backend reranks candidates using OpenAI.
8. Backend selects the best grounded chunks.
9. Backend generates an answer using only the selected context.
10. Frontend displays the answer and source citations.

## Backend API Requirements

### FastAPI App

The backend must initialize FastAPI normally:

```python
from fastapi import FastAPI

app = FastAPI()
```

Docs must remain enabled by default. Do not set `docs_url=None`.

### Required Routes

The backend must expose:

- `GET /`
- `GET /health`
- `GET /modules`
- `GET /sessions?module={module}`
- `POST /chat`

### Root Route

`GET /` must return:

```json
{"status": "API running"}
```

### Health Route

`GET /health` must return:

```json
{"status": "healthy"}
```

### Chat Route

`POST /chat` must support the frontend payload:

```json
{
  "question": "What is prompt engineering?",
  "mode": "global",
  "module": null,
  "session": null,
  "chat_history": []
}
```

It must also support a simple testing payload:

```json
{
  "query": "What is prompt engineering?"
}
```

The response must include:

```json
{
  "answer": "...",
  "sources": []
}
```

## Frontend Requirements

The Streamlit frontend lives in `app.py`.

It must call:

```python
BASE_URL = "https://iitm-curriculum-intelligence-layer.onrender.com"
API_BASE_URL = BASE_URL
```

Frontend API calls:

- `GET {API_BASE_URL}/modules`
- `GET {API_BASE_URL}/sessions`
- `POST {API_BASE_URL}/chat`

The frontend must preserve recent chat history and reset history when the user changes scope.

## RAG Retrieval Requirements

### Stage 1: Vector Retrieval

Use Pinecone for high-recall retrieval.

Requirements:

- Keep Pinecone retrieval logic unchanged.
- Keep OpenAI embedding model unchanged.
- Retrieve `RETRIEVAL_TOP_K = 25` candidate chunks.
- Include metadata in Pinecone results.
- Apply module/session filters when requested.

### Stage 2: OpenAI Reranking

After retrieval, rerank the 25 candidate chunks using OpenAI.

Function:

```python
def rerank_documents(query: str, docs: list[dict]) -> list[dict]:
    ...
```

Requirements:

- Use `gpt-4o-mini`.
- Score each document from `0` to `1`.
- Use only the first 500 characters of each chunk for scoring.
- Sort by rerank score descending.
- Remove documents with score below `0.6`.
- Return the top 5 documents.
- If reranking fails, return the original `docs[:5]`.
- Never crash the API due to reranking failure.

Reranking prompt:

```text
You are a strict relevance scorer.

Given a query and a document, score how relevant the document is to answering the query.

Rules:

Score between 0 and 1
1 = directly answers the query
0 = completely irrelevant
Be strict: partial matches should be <= 0.5
Do NOT explain
Output ONLY a number
```

## Answer Generation Requirements

The answer generation step must use only the selected reranked chunks.

System prompt requirements:

```text
Use ONLY the provided context.

Rules:

If answer is not clearly in context -> say 'Not in module'
Do NOT guess
Do NOT use prior knowledge
Cite sources inline
Be precise and structured
```

The answer must include inline citations such as:

```text
[CMS-S3-C84]
```

If the selected context does not clearly answer the question, the answer must be:

```text
Not in module
```

## Debug Logging Requirements

The backend should print simple deployment-friendly debug logs:

- Request received.
- Retrieved docs count.
- Reranked scores.
- Final selected docs.
- Before OpenAI call.
- After OpenAI call.
- Error details for caught exceptions.

These logs are intended to identify whether slowdowns occur during retrieval, reranking, or generation.

## Transcript Processing Requirements

Raw transcripts are stored under:

```text
input/{module}/session_{session_number}.txt
```

Supported modules currently include:

- `cms`
- `map`
- `wdp`

The transcript pipeline must:

- Load module/session transcript files.
- Split transcript text into manageable chunks.
- Clean and structure transcript content with OpenAI.
- Preserve module and session metadata.
- Merge cleaned outputs per module.
- Generate RAG-ready chunks under `rag_chunks/{module}/`.

RAG chunk files should include:

- Module
- Session
- Topic
- Chunk number
- Cleaned explanatory content
- Key points
- Student doubts where available

## Pinecone Requirements

The upload workflow must:

- Read generated RAG chunks.
- Embed chunk text using OpenAI embeddings.
- Upload vectors to Pinecone.
- Preserve metadata:
  - `module`
  - `session`
  - `chunk`
  - `text`

The API currently uses the Pinecone index:

```python
iitm-modules-rag
```

## Deployment Requirements

The backend must remain lightweight for Render free-tier deployment.

Requirements:

- Do not include `sentence-transformers`.
- Do not include PyTorch.
- Do not load local cross-encoder models.
- Use OpenAI reranking instead of local reranking.
- Keep startup lightweight.
- Ensure the deployed app exposes `GET /`, `GET /docs`, and `POST /chat`.

If Render is configured as `uvicorn main:app`, `main.py` must expose the FastAPI app imported from `api.py`.

Recommended backend start command:

```text
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Compatibility start command:

```text
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Dependency Requirements

API dependencies should remain minimal:

```text
fastapi
uvicorn
requests
openai
pinecone
python-dotenv
```

General project dependencies may include Streamlit for the frontend:

```text
streamlit
```

Avoid heavy ML dependencies unless the deployment target changes.

## Success Criteria

The project is successful when:

- `GET /` returns `{"status": "API running"}`.
- `GET /docs` opens FastAPI docs.
- `POST /chat` returns an answer and sources.
- Frontend successfully calls `BASE_URL + "/chat"`.
- Pinecone retrieval returns 25 candidates before reranking.
- OpenAI reranking selects more relevant top 5 chunks.
- Answers cite retrieved sources inline.
- Unsupported answers return `Not in module`.
- Render deployment does not hang during dependency installation.

## Out of Scope For Current Version

- Local cross-encoder reranking.
- `sentence-transformers`.
- PyTorch-based model hosting.
- User authentication.
- Admin dashboard.
- Real-time transcript upload UI.
- Multi-tenant data isolation.

## Future Enhancements

- Batch reranking to reduce OpenAI latency.
- More structured JSON scoring for reranking.
- Evaluation set for retrieval quality.
- Automated regression tests for answer grounding.
- Admin UI for uploading new curriculum material.
- Observability with request IDs and latency timings.
- Configurable Pinecone index name through environment variables.
