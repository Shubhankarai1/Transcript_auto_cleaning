# Implementation Plan: Multi-Level Course Project Version

This document defines the simplest practical way to extend the current project into a multi-level learning system for a course project.

The goal is not to build a full SaaS platform. The goal is to demonstrate that the existing RAG system can support:

- beginner content
- intermediate content
- advanced content

with level-aware retrieval and a simple UI update.

## Project Goal

Show that:

1. different content levels can be added to the knowledge base
2. all content still goes through the normal RAG pipeline
3. the user can choose a level in the UI
4. the backend retrieves content based on that level
5. the final answer is adjusted to match the chosen learner level

## What We Will Not Build

For this course project, we do not need:

- full authentication
- Supabase integration
- profile persistence
- adaptive roadmaps
- advanced analytics
- production-grade admin tooling
- perfect content governance

## Minimal Scope

We only need enough functionality to clearly prove the concept.

That means:

- add beginner and intermediate content
- keep existing advanced content
- tag content with learner level
- use level during retrieval
- add a level selector in Streamlit

## Phase 1: Organize Content By Level

### Objective

Create a simple structure for storing content at three difficulty levels.

### Plan

Use a folder structure such as:

```text
input/
  beginner/
    topic_a/
      lesson_1.txt
  intermediate/
    topic_a/
      lesson_1.txt
  advanced/
    cms/
      session_1.txt
```

### Notes

- Existing advanced transcript content can be moved or mapped into `advanced/`
- Beginner and intermediate content can be short curated lesson files
- The content does not need to be large for the demo

### Outcome

The project now has clearly separated content for all three levels.

## Phase 2: Update The Ingestion Pipeline

### Objective

Make the pipeline understand learner level while preserving the usual chunking flow.

### Plan

Update the pipeline so that each content file is processed with:

- level
- module/topic
- session or lesson id
- text content

Each chunk should carry level metadata.

### Outcome

The chunking process works for all three levels, not only the current advanced transcript files.

## Phase 3: Update Pinecone Upload Metadata

### Objective

Store level information in the vector database.

### Plan

When uploading chunks, ensure Pinecone metadata includes:

- `level`
- `module`
- `session`
- `chunk`
- `text`

### Outcome

Every stored vector can be filtered or prioritized by learner level.

## Phase 4: Update Retrieval Logic

### Objective

Make the backend search by level.

### Plan

Update the chat request flow so that the backend can receive:

- question
- level
- optional module/session filters

Retrieval behavior:

- if user selects beginner, search beginner content first
- if user selects intermediate, search intermediate content first
- if user selects advanced, search advanced content first

For this course project, simple filtering is enough.

### Outcome

The backend retrieves content based on learner level.

## Phase 5: Update Answer Style

### Objective

Make the final response match the learner level.

### Plan

Add simple prompt guidance:

- beginner: explain simply and clearly
- intermediate: explain with some depth and examples
- advanced: explain in a more technical way

The answer must still remain grounded in retrieved content.

### Outcome

The same system gives different explanation styles for different learners.

## Phase 6: Update Streamlit UI

### Objective

Let the user choose their learner level.

### Plan

Add a simple selector such as:

- Beginner
- Intermediate
- Advanced

Send the chosen level to the backend with the chat request.

Optional:

- show the chosen level in the UI header
- show recommended content scope based on level

### Outcome

The frontend becomes level-aware with a very small change.

## Phase 7: Demo Preparation

### Objective

Prepare a clean course-project demonstration.

### Plan

Create a few sample questions for each level and show:

- beginner answer
- intermediate answer
- advanced answer

This demonstrates that:

- the content exists
- the pipeline processed it
- retrieval uses the selected level
- the UI supports the concept

### Outcome

The project is demo-ready.

## Suggested Order Of Work

1. organize beginner/intermediate/advanced content folders
2. update chunking to preserve level metadata
3. update Pinecone upload to store level metadata
4. re-upload the content
5. update backend retrieval to use level
6. update prompt style by level
7. update Streamlit UI with a level selector
8. prepare demo questions

## Minimum Technical Changes

Likely files affected:

- `main.py`
  - content loading and chunk generation
- `upload_to_pinecone.py`
  - metadata and vector upload
- `api.py`
  - retrieval filters and response prompt
- `app.py`
  - level selector and request payload

## Simpler Alternative If Time Is Very Limited

If there is very little time, use this reduced approach:

- keep the current advanced content exactly as it is
- add only a small set of beginner and intermediate files
- tag them manually with level
- add a level dropdown in the UI
- filter retrieval by level

This is enough for a solid course project demonstration.

## Final Recommendation

For this project, do not overbuild.

Implement the minimum version that proves the core idea:

- three content levels
- one shared RAG pipeline
- level-aware retrieval
- simple UI support

That is the right level of complexity for a course project.
