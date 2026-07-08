# Content Upgrade Plan: Multi-Level Internal Knowledge Base

This document explains how to extend the current transcript-based RAG system into a multi-level internal knowledge base that supports:

- beginner
- intermediate
- advanced

The current repository already supports ingestion, chunking, metadata extraction, embedding, Pinecone upload, retrieval, and Streamlit-based question answering. However, the existing content is mostly advanced and transcript-driven. To support the future platform, the content system must be upgraded so the AI can answer questions appropriately for different learner levels.

## Core Goal

Build a content pipeline that allows new learning material to be uploaded and processed through the same standard flow:

1. content input
2. cleaning or normalization
3. chunking
4. metadata extraction
5. embedding
6. Pinecone upload
7. retrieval at question time
8. level-aware answer generation in the UI

## Current Situation

The current project structure is optimized for advanced IITM transcript content:

- `input/<module>/session_<n>.txt`
- `output/<module>_final_cleaned.txt`
- `rag_chunks/<module>/...`

This works for the existing advanced content, but it does not yet support:

- multiple learner levels
- multiple content types
- level-aware retrieval
- roadmap-driven content recommendations

## Target Content Model

The system should classify content using at least these dimensions:

- `level`
  - beginner
  - intermediate
  - advanced
- `module`
  - example: `cms`, `map`, `wdp`, `ai_foundations`, `prompting_basics`
- `session`
  - optional for non-transcript content
- `topic`
- `source_type`
  - lecture
  - curated_note
  - lesson
  - summary
  - assessment_support

## Recommended Folder Strategy

The current `input/` structure should evolve to support level-based content.

Recommended future structure:

```text
input/
  beginner/
    ai_foundations/
      lesson_1.txt
      lesson_2.txt
    prompt_basics/
      lesson_1.txt
  intermediate/
    business_ai/
      lesson_1.txt
    rag_intro/
      lesson_1.txt
  advanced/
    cms/
      session_1.txt
      session_2.txt
    map/
      session_2.txt
    wdp/
      session_1.txt
```

This structure makes level a first-class property instead of assuming all content belongs to the same difficulty layer.

## Recommended Metadata Additions

Every chunk uploaded to Pinecone should eventually include:

- `level`
- `module`
- `session`
- `topic`
- `chunk`
- `text`
- `source_type`
- `difficulty`
- `audience`
- `content_origin`

Recommended examples:

- `level = beginner`
- `source_type = lesson`
- `content_origin = internal_curated`

or

- `level = advanced`
- `source_type = lecture`
- `content_origin = transcript`

## Upload And Processing Flow

All new content, regardless of learner level, should pass through the usual process. The difference is that the pipeline must become level-aware.

### Step 1: Add Source Content

Place new content files in the correct level and module folder.

Examples:

- `input/beginner/ai_foundations/lesson_1.txt`
- `input/intermediate/rag_intro/lesson_1.txt`
- `input/advanced/wdp/session_1.txt`

### Step 2: Normalize Or Clean Content

The current cleaning flow is transcript-oriented. That is appropriate for advanced lecture transcripts, but beginner and intermediate content may be:

- manually written lessons
- internal summaries
- guided concept notes

So the cleaning phase should support two modes:

- transcript cleaning for raw session transcripts
- lightweight normalization for curated lessons

Normalization can include:

- heading cleanup
- paragraph cleanup
- bullet preservation
- section standardization

### Step 3: Chunk Content

All content still needs chunking before upload.

Chunking rules should evolve as follows:

- transcript content:
  - split by session
  - split by topic
  - split by paragraph groups
- lesson content:
  - split by heading
  - split by subtopic
  - preserve examples and structured lists

The chunking output should preserve:

- level
- module
- topic
- source type

### Step 4: Generate Metadata

Metadata extraction should identify:

- topic
- subtopic
- learner level
- content type
- keywords
- concept aliases

For curated beginner and intermediate content, some metadata may be explicitly declared rather than inferred entirely by an LLM.

### Step 5: Create Embeddings

Once chunked and tagged, the content follows the same embedding flow already used by the project.

That means:

- generate embeddings
- package metadata
- prepare upload payload

No special retrieval system is needed for beginner or intermediate content. It should use the same vector database and the same overall pipeline.

### Step 6: Upload To Pinecone

All chunk variants should be uploaded into the same retrieval ecosystem, but with strong metadata support so the system can filter or prioritize by level.

Important rule:

- do not upload chunks without the metadata needed to distinguish beginner, intermediate, and advanced content

## Retrieval And Question Answering Changes

Yes, the retrieval logic must change so that content works correctly when users ask questions.

Right now, retrieval is mostly module/session aware. In the upgraded platform, retrieval should also become level-aware.

### Recommended Retrieval Behavior

When a user asks a question, the system should prefer content based on learner level.

Examples:

- beginner user:
  - prefer beginner chunks first
  - fall back to intermediate if needed
  - use advanced only if nothing else exists
- intermediate user:
  - prefer intermediate chunks
  - allow beginner for foundations
  - allow advanced for deeper explanations
- advanced user:
  - prefer advanced chunks
  - allow intermediate when helpful

### Required Metadata Filters

The retriever should eventually support filters like:

- `level = beginner`
- `level = intermediate`
- `level = advanced`
- `module = ...`
- `session = ...`

### Answering Rules

The final answer generation should use:

- retrieved content
- learner level
- tone and complexity guidance

Important:

- learner level should affect explanation style
- learner level should not override grounding
- the system must still answer from retrieved context only

## Streamlit UI Changes

Yes, the Streamlit UI will also need to change.

The current UI is mainly a chat interface with module/session filtering. That is not enough once content is organized by learning level.

### Minimum Required UI Changes

- add learner level awareness in the interface
- allow the system to know whether the user is beginner, intermediate, or advanced
- allow filtering or automatic scoping based on level
- display recommended modules or content sections based on level

### Practical Options

Option 1:

- user explicitly selects level in the UI

Option 2:

- user level is inferred from profile and assessment
- UI automatically uses that level during retrieval

For the future platform, Option 2 is better. For near-term implementation, Option 1 is a fast transitional step.

### Likely UI Areas To Change

- sidebar filters
- dashboard cards
- mentor chat payload
- source display
- recommended learning path sections

## Backend Changes Needed

The backend should be extended to support level-aware content.

At minimum, this means:

- add `level` to ingestion metadata
- add `level` to Pinecone metadata
- add `level` to retrieval filters
- allow `/chat` payloads to include learner level
- make answer generation prompts aware of learner level

## Content Strategy Recommendation

Do not treat beginner, intermediate, and advanced as only presentation layers.

They should be treated as distinct content libraries with their own source materials and metadata.

That means:

- advanced:
  - existing IITM transcript corpus
- intermediate:
  - curated bridge content
  - practical notes
  - applied explanations
- beginner:
  - foundational lessons
  - simplified concept guides
  - glossary-style support content

## Migration Recommendation

Do this in stages rather than changing everything at once.

### Stage 1

- keep current advanced pipeline working
- tag existing advanced content explicitly as `level = advanced`

### Stage 2

- introduce beginner and intermediate source folders
- add chunking and metadata support for lesson-style content

### Stage 3

- update upload pipeline to include level metadata
- re-upload all content with the new schema

### Stage 4

- update retrieval logic to prefer level-appropriate chunks

### Stage 5

- update Streamlit UI to send learner level or read it from profile

## What Must Be True For This To Work

For the upgraded system to work properly:

- new content must enter through a standard ingestion path
- chunks must carry level metadata
- embeddings must be regenerated for the upgraded content set
- Pinecone upload must preserve the new metadata
- retrieval must use level-aware filtering or ranking
- the UI must pass learner context into the backend

## Recommended Next Decisions

1. Decide the future folder structure for beginner, intermediate, and advanced content.
2. Decide whether beginner and intermediate content will be written manually, summarized from advanced content, or both.
3. Extend the ingestion schema to include `level` and `source_type`.
4. Update the retrieval contract so chat requests can include learner level.
5. Update the Streamlit UI to support level selection now, and profile-based level later.

## Bottom Line

Yes, beginner and intermediate content can and should be added through the same general process already used in the project:

- ingest
- clean or normalize
- chunk
- embed
- upload
- retrieve
- answer

But to make that work properly, the content model, metadata model, retrieval logic, and Streamlit UI all need to be upgraded to support level-aware learning content rather than only the current advanced transcript corpus.
