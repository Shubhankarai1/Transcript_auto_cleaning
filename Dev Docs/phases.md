# Project Phases: Current Build Vs Pending Work

This file reflects the current state of the transcript/course RAG system against the checklist in `project_instructions.md`. It replaces the older stale version that repeated outdated status.

## Phase 0: Transcript Cleaning Pipeline

### Built

- Transcript inputs are organized under `input/<module>/session_<n>.txt`.
- `main.py` processes transcripts module-by-module and session-by-session.
- Raw transcript chunks are saved under `chunks/<module>/`.
- Cleaned session outputs are cached under `output/sessions/<module>/`.
- Final merged module outputs are written under `output/` as `*_final_cleaned.txt`.
- Session-level cache/hash reuse is implemented, so unchanged transcript sessions are skipped on rerun.

### Pending

- Add explicit validation/reporting for malformed transcript inputs before processing.
- Add automated checks for cleaning quality regressions.
- Add a clearer operator-facing processing summary per run.

## Phase 1: Ingestion Engine And Chunking

### Built

- Cleaned module outputs are re-chunked into RAG-ready files under `rag_chunks/<module>/`.
- Chunking is structure-aware at the transcript level:
  - split by `### Session <n>`
  - split by `### Topic: ...`
  - then split large topic bodies by paragraph groups
- Chunk files store lightweight headers:
  - module
  - session
  - topic
  - chunk number

### Pending

- Add markdown-header-aware chunking beyond the current session/topic patterns.
- Add token-aware chunk sizing instead of only word/paragraph heuristics.
- Add 10-15% overlap between sequential chunks.
- Add preservation rules for tables, lists, or other structured blocks if future source material contains them.
- Add a rerunnable chunk manifest so only changed RAG chunks need regeneration.

## Phase 2: Metadata Extraction And Tagging

### Built

- `upload_to_pinecone.py` extracts semantic metadata for each chunk using OpenAI.
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
- Metadata normalization exists for:
  - keyword cleanup/deduplication
  - fallback keyword generation
  - difficulty normalization

### Pending

- Add richer course-specific metadata if it becomes available:
  - session title
  - concept name
  - instructor
  - date/version
  - prerequisite or related concept
- Add stricter metadata validation and failure reporting before upload.
- Add partial reprocessing so only changed chunks are re-embedded and re-uploaded.

## Phase 3: Embeddings And Vector Store

### Built

- OpenAI embeddings are used for both ingestion and retrieval.
- Pinecone is the active vector store.
- Batch upsert to Pinecone is implemented.
- Chunk text is stored in metadata for downstream answer generation and source display.
- Upload logging shows progress across batches and final index stats.

### Pending

- Unify Pinecone index configuration across the codebase.
  - `upload_to_pinecone.py` uses `PINECONE_INDEX_NAME`
  - `api.py` still hardcodes `iitm-modules-rag`
- Evaluate whether to move from `text-embedding-3-small` to `text-embedding-3-large`.
- Add safer idempotent sync behavior for updates/deletes in Pinecone.

## Phase 4: Retrieval Baseline

### Built

- FastAPI backend is live in `api.py`.
- The API supports:
  - global retrieval
  - filtered retrieval by module and session
- Retrieval currently:
  - embeds multiple search queries
  - queries Pinecone with `top_k = 25`
  - pools results
  - deduplicates matches
  - keeps final top 5 matches for answer generation
- Source payloads include:
  - citation
  - module
  - session
  - chunk
  - score
  - matched query

### Pending

- Add stronger retrieval diagnostics for offline evaluation.
- Add thresholding or weak-match handling before answering.
- Add automated retrieval-quality checks using fixed benchmark questions.
- Extend retrieval filtering to support `level + module` scoping for the new multi-level content structure.

## Phase 5: Query Transformation

### Built

- Query rewriting is live in `api.py`.
- Multi-query expansion is live in `api.py`.
- HyDE generation is live in `api.py`.
- Retrieval searches across:
  - original question
  - rewritten query
  - HyDE query
  - alternate expanded queries
- Query variants are normalized and deduplicated before retrieval.

### Pending

- Tune expansion quality and guard against low-value alternate queries.
- Measure retrieval lift from each query strategy separately.
- Add fallback behavior when rewrite/HyDE outputs are poor but retrieval should still proceed cleanly.

## Phase 6: Metadata Filtering

### Built

- Lightweight query-based filter detection exists in `retrieval_utils.py`.
- Filter hints currently support:
  - module hints
  - session/week hints
  - a few topic-specific keyword mappings
- Explicit UI filters and auto-detected query filters are merged together.

### Pending

- Improve concept/topic intent detection beyond keyword rules.
- Add more reliable topic-level pre-filtering once metadata quality is proven.
- Add fallback from filtered retrieval to broader retrieval when filters are too restrictive.
- Add smarter version/date-aware post-filtering if the content base grows.

## Phase 7: Reranking

### Built

- Broad candidate retrieval already fetches 25 matches before final selection.

### Pending

- Add an actual reranker after Pinecone retrieval.
- Score candidate chunks against the original user query using a cross-encoder or rerank API.
- Reorder pooled retrieval results using rerank scores instead of raw vector similarity alone.
- Compare answer quality before and after reranking on fixed test questions.

## Phase 8: Answer Generation

### Built

- The backend answers questions using retrieved context plus an instructor-style prompt.
- The API returns the answer with structured source metadata.
- The prompt encourages:
  - detailed teaching-style explanations
  - synthesis across chunks
  - use of partial context instead of immediate refusal

### Pending

- Enforce citation behavior in the final answer text instead of only returning sources separately.
- Add explicit hallucination-control rules for unsupported claims.
- Add evaluation for answer quality on representative course questions.
- Add better fallback language than the current generic `Not in module.` response.

## Phase 9: Frontend UX

### Built

- Streamlit chat UI exists in `app.py`.
- Users can switch between:
  - all-content mode
  - module+session filtered mode
- Module and session options are loaded from the API.
- Chat history is maintained per scope.
- Answers and source summaries are rendered in the chat UI.

### Pending

- Replace the current advanced-only module selection flow with a two-step selector:
  - `Foundations`
  - `Intermediate`
  - `Advanced`
- After level selection, show only the modules for that level:
  - `Foundations`
    - `ai_foundations_curriculum`
    - `prompt_engineering`
    - `ai_ethics_safety_and_data_privacy`
    - `finance_chatgpt_excel_skills`
    - `operations_process_mapping_and_automated_reporting`
    - `hr_ai_enhanced_jd_design_and_skills_gap_mapping`
  - `Intermediate`
    - `human_in_the_loop_designing_hybrid_systems`
    - `ai_data_analysis_extracting_insights`
    - `project_management_predictive_resource_allocation_and_automated_risk_tracking`
    - `customer_facing_ai_sentiment_analysis_and_crm_integration`
  - `Advanced`
    - `cms`
    - `map`
    - `wdp`
- Pass both `level` and `module` from the Streamlit UI into backend retrieval so users can ask module-specific questions inside each learning level.
- Move source/debug details back into a cleaner collapsed UI instead of appending them directly to the answer.
- Improve source formatting with score and short previews where useful.
- Add clearer retrieval/debug view for evaluation without cluttering the normal chat experience.
- Remove stale UI text and dead code paths in the Streamlit app.

## Recommended Next Order

1. Implement reranking.
2. Add filtered-to-global fallback when retrieval is weak.
3. Improve chunking with overlap and better structural preservation.
4. Unify Pinecone/index/model configuration across ingestion and API code.
5. Add retrieval and answer evaluation using fixed benchmark questions.
