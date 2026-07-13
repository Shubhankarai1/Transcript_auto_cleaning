# Transcript Auto Cleaning

commands to run 
uvicorn main:app --reload
streamlit run app.py

Python pipeline for cleaning, structuring, and indexing lecture transcripts using a local Ollama model, with hierarchy-aware metadata for RAG ingestion into Pinecone.

## What It Does

- Reads transcript files from `input/` organized as `level/category/module/session_<n>.txt`
- Extracts hierarchy metadata: `level`, `category`, `module`, `session`, `module_path`, `content_id`
- Splits each transcript into roughly 1200-word chunks
- Saves chunk files to `chunks/<module>/`
- Cleans and structures each chunk with Ollama
- Stores one cleaned output per session in `output/sessions/<module>/`
- Skips sessions that already have a saved cleaned output on rerun
- Generates one final cleaned file per module in `output/` with level/category header
- Produces RAG chunks in `rag_chunks/<module>/` with full metadata headers
- Uploads chunks to Pinecone with hierarchy metadata for filtered retrieval

## Project Structure

```text
input/
  level_1_foundations/
    common_modules/
      ai_ethics_safety_and_data_privacy/
        session_1.txt
        session_2.txt
      ai_foundations_curriculum/
      prompt_engineering/
    subject_matter_expertise/
      finance_chatgpt_excel_skills/
      hr_ai_enhanced_jd_design_and_skills_gap_mapping/
      operations_process_mapping_and_automated_reporting/
  level_2_intermediate/
    common_modules/
      ai_data_analysis_extracting_insights/
      human_in_the_loop_designing_hybrid_systems/
    role_specific/
      customer_facing_ai_sentiment_analysis_and_crm_integration/
      project_management_predictive_resource_allocation_and_automated_risk_tracking/
  level_3_advanced/
    cms/
      session_1.txt
    map/
    wdp/
chunks/
  <module>/
    session_<n>_chunk_<m>.txt
output/
  <module>_final_cleaned.txt
  sessions/
    <module>/
      session_<n>_cleaned.txt
      session_<n>_cache.json
rag_chunks/
  <module>/
    <module>_session_<n>_chunk_<m>.txt
main.py
utils.py
upload_to_pinecone.py
config.py
requirements.txt
.env
.gitignore
README.md
```

## Hierarchy Metadata Fields

| Field | Parsed | Propagated | In Pinecone |
|---|---|---|---|
| `level` | `level_1_foundations` → `beginner`, `level_2_intermediate` → `intermediate`, `level_3_advanced` → `advanced` | Yes | Yes |
| `category` | Second path component (`common_modules`, `subject_matter_expertise`, `role_specific`); `None` for advanced modules | Yes (fallback `"advanced"` when `None`) | Yes |
| `module` | Module folder name | Yes | Yes |
| `session` | From `session_<n>.txt` filename | Yes | Yes |
| `chunk` | Generated during RAG chunking | Yes | Yes |
| `module_path` | `<category>/<module>` (non-advanced) or `<module>` (advanced) | **Parsed only — not yet propagated downstream** | No |
| `content_id` | `level_<n>_<name>/<category>/<module>/session_<n>` | **Parsed only — not yet propagated downstream** | No |

## Setup

1. Create and activate a virtual environment if desired.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running locally and the target model is available:

```bash
ollama pull mistral
ollama serve
```

## Add Transcripts

Place transcript files inside the appropriate hierarchy under `input/`.

**Level 1 (Beginner) — Foundations:**
```text
input/level_1_foundations/common_modules/<module_name>/session_<n>.txt
input/level_1_foundations/subject_matter_expertise/<module_name>/session_<n>.txt
```

**Level 2 (Intermediate):**
```text
input/level_2_intermediate/common_modules/<module_name>/session_<n>.txt
input/level_2_intermediate/role_specific/<module_name>/session_<n>.txt
```

**Level 3 (Advanced) — no category folder:**
```text
input/level_3_advanced/cms/session_<n>.txt
input/level_3_advanced/map/session_<n>.txt
input/level_3_advanced/wdp/session_<n>.txt
```

Rules:
- Folder names use lowercase letters, numbers, and underscores only.
- Session files must be named `session_<number>.txt`.
- Sessions are merged in ascending numeric order inside each module.
- For `level_3_advanced`, only `cms`, `map`, `wdp` are valid module names.

## Run

```bash
python main.py       # Clean transcripts and produce RAG chunks
python upload_to_pinecone.py   # Upload RAG chunks to Pinecone
```

The cleaning script sends chunk-cleaning requests to:

```text
http://localhost:11434/api/generate
```

using model `mistral`.

## Output

- Raw chunks are saved in `chunks/<module>/`
- Per-session cleaned files are saved in `output/sessions/<module>/`
- Final cleaned module files are saved in `output/` as `<module>_final_cleaned.txt` with a header:

```text
Level: beginner
Category: common_modules

### Session 1
...
```

- RAG chunks ready for Pinecone are saved in `rag_chunks/<module>/` with a header:

```text
Level: beginner
Category: common_modules
Module: ai_ethics_safety_and_data_privacy
Session: 1
Topic: Introduction to AI Ethics
Chunk: 1
```

## Notes

- No content is mixed between modules.
- Each module gets exactly one standalone final output file.
- On rerun, sessions are reprocessed only when their source transcript has changed.
- The pipeline stores a source hash in `output/sessions/<module>/session_<n>_cache.json` and compares it to the current transcript.
- If you want to force reprocess a session, delete its matching cleaned output file or the corresponding cache metadata file and rerun the script.
