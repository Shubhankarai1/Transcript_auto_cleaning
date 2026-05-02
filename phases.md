# Project Phases: Course RAG System

This file tracks what is already built and what is still pending for the current RAG app. The original Aegis example used corporate policy documents, but this project uses course/transcript data, so policy-specific fields are intentionally ignored.

## Phase 1: Ingestion And Chunking

### Done

- Raw course transcripts are stored module-wise under `input/`.
- Cleaned and structured chunk files are generated under `rag_chunks/`.
- Chunks are organized by module and session.

### Pending

- Add Markdown/header-aware chunking where applicable.
- Add token-based chunk sizing instead of only file/text based chunking.
- Add 10-15% overlap between sequential chunks.
- Add stronger table/list preservation if future source data contains tables or structured blocks.

## Phase 2: Metadata Extraction

### Done

- Metadata extraction exists in `upload_to_pinecone.py`.
- Current metadata includes:
  - `document_id`
  - `module`
  - `topic`
  - `subtopic`
  - `session`
  - `chunk_id`
  - `keywords`
  - `source_type`
  - `difficulty`
- Chunk text is stored in Pinecone metadata for retrieval and source display.

### Pending

- Add richer course-specific metadata if needed:
  - course/topic group
  - concept name
  - session title
  - instructor/session date if available
  - prerequisite concept
- Add metadata validation before upload.
- Add a way to reprocess only changed chunks.

## Phase 3: Embedding And Vector Store

### Done

- OpenAI embeddings are used.
- Pinecone is used as the vector database.
- Batch upsert to Pinecone is implemented.
- Metadata is uploaded with each vector.

### Pending

- Consider upgrading from `text-embedding-3-small` to `text-embedding-3-large` for better retrieval quality.
- Make Pinecone index name configurable everywhere.
- Add upload logs/report showing how many chunks were inserted per module/session.

## Phase 4: Basic Retrieval

### Done

- FastAPI backend retrieves chunks from Pinecone.
- Current backend retrieves top 25 candidates and passes final top 5 chunks.
- Sources are returned with answer.
- Global and filtered retrieval modes exist.

### Pending

- Add better debug logging for retrieved context during testing.
- Add score visibility only inside debug/source UI.
- Add automated test questions for retrieval quality.

## Phase 5: Query Transformation

### Done

- A basic query rewrite step exists in `api.py`.
- Multi-query expansion exists in `query_rag.py`, but it is not yet part of the live API path.

### Pending

- Move multi-query expansion into `api.py`.
- Generate 3-4 alternate query versions.
- Retrieve for each query variant.
- Pool and deduplicate results across all query variants.
- Add HyDE retrieval for vague questions.

## Phase 6: Metadata Filtering

### Done

- Query-based lightweight filter detection exists in `retrieval_utils.py`.
- Filters currently support module/session/topic-style hints.
- UI supports filtered mode by module and session.

### Pending

- Improve intent detection for course concepts.
- Add concept/topic pre-filtering where metadata is reliable.
- Avoid over-filtering when the user asks broad questions.
- Add fallback to global search when filtered search finds weak/no results.

## Phase 7: Reranking

### Done

- Broad retrieval top-k is already configured.
- Reranking concepts exist in the project notes and previous work, but reranking is not active in the current API.

### Pending

- Add a reranker after Pinecone retrieval.
- Retrieve top 25 candidates.
- Rerank against the original user question.
- Pass only top 5 reranked chunks to the final answer model.
- Compare results with and without reranking using fixed test questions.

## Phase 8: Answer Generation

### Done

- The backend generates answers using retrieved context.
- The current answer prompt is teaching-style and detailed.
- Sources are returned separately from the answer.

### Pending

- Add citation consistency rules.
- Add handling for partial context without over-refusing.
- Add answer quality tests for common questions like:
  - what is LangChain
  - what is reranking
  - explain Pinecone retrieval
  - what is LangGraph

## Phase 9: Frontend UX

### Done

- Streamlit chat interface exists.
- Answer is displayed first.
- Sources/debug info are hidden inside a closed expander.
- Raw backend JSON is no longer dumped directly in the chat.

### Pending

- Improve source formatting inside the expander.
- Show source citation, module, session, chunk, and score cleanly.
- Optionally add a short source preview per retrieved chunk.

## Recommended Next Order

1. Add multi-query expansion to the live `api.py` path.
2. Add result pooling and deduplication.
3. Add reranking.
4. Improve course-specific metadata.
5. Improve chunking with overlap and better structure preservation.
6. Add test questions and compare retrieval quality after each change.
