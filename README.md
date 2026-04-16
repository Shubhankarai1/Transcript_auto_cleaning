# Transcript Auto Cleaning

Python pipeline for cleaning and structuring lecture transcripts module by module using a local Ollama model.

## What It Does

- Reads transcript files from `input/`
- Treats each subfolder in `input/` as one module
- Reads session files named `session_<number>.txt` inside each module folder
- Splits each transcript into roughly 1200-word chunks
- Saves chunk files to `chunks/<module>/`
- Cleans and structures each chunk with Ollama
- Stores one cleaned output per session in `output/sessions/<module>/`
- Skips sessions that already have a saved cleaned output on rerun
- Generates one final cleaned file per module in `output/`

## Project Structure

```text
input/
  cms/
    session_1.txt
    session_2.txt
  ma/
    session_1.txt
chunks/
  cms/
  ma/
output/
  cms_final_cleaned.txt
  ma_final_cleaned.txt
  sessions/
    cms/
      session_1_cleaned.txt
    ma/
      session_1_cleaned.txt
main.py
utils.py
requirements.txt
.env
.gitignore
README.md
```

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

Place transcript files in module folders inside `input/`.

```text
input/
  cms/
    session_1.txt
    session_2.txt
  ma/
    session_1.txt
  contextual_management_systems/
    session_1.txt
```

Rules:
- Folder names are the module names.
- Module folder names should use lowercase letters, numbers, and underscores only.
- Session files must be named `session_<number>.txt`.
- Sessions are merged in ascending numeric order inside each module.

## Run

```bash
python main.py
```

The script sends chunk-cleaning requests to:

```text
http://localhost:11434/api/generate
```

using model `mistral`.

## Output

- Raw chunks are saved in `chunks/<module>/`
- Per-session cleaned files are saved in `output/sessions/<module>/`
- Final cleaned module files are saved in `output/` as:

```text
cms_final_cleaned.txt
ma_final_cleaned.txt
xyz_final_cleaned.txt
```

## Final Output Format

Each module output contains only that module's sessions:

```text
### Session 1

[cleaned session 1 content]

---

### Session 2

[cleaned session 2 content]
```

## Notes

- No content is mixed between modules.
- Each module gets exactly one standalone final output file.
- On rerun, sessions are reprocessed only when their source transcript has changed.
- The pipeline stores a source hash in `output/sessions/<module>/session_<n>_cache.json` and compares it to the current transcript.
- If you want to force reprocess a session, delete its matching cleaned output file or the corresponding cache metadata file and rerun the script.
