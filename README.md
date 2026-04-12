# Transcript Auto Cleaning

Python pipeline for cleaning and structuring lecture transcripts session by session using a local Ollama model.

## What It Does

- Reads transcript files from `input/`
- Detects session numbers from filenames like `session_1.txt`
- Splits each transcript into roughly 1200-word chunks
- Saves chunk files to `chunks/` using strict naming:
  - `session_1_chunk_1.txt`
  - `session_1_chunk_2.txt`
- Cleans and structures each chunk with Ollama
- Merges cleaned chunks into `output/final_cleaned.txt`

## Project Structure

```text
input/
chunks/
output/
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
ollama pull llama3
ollama serve
```

## Add Transcripts

Place transcript files in `input/` using this format:

```text
session_1.txt
session_2.txt
```

Each file should contain one session transcript.

## Run

```bash
python main.py
```

The script sends chunk-cleaning requests to:

```text
http://localhost:11434/api/generate
```

using model `llama3`.

## Output

- Raw chunks are saved in `chunks/`
- Final merged cleaned document is saved to:

```text
output/final_cleaned.txt
```

## Chunk File Format

Each chunk file name strictly follows:

```text
session_<session_number>_chunk_<chunk_number>.txt
```

Each chunk file also starts with:

```text
Session <session_number> - Chunk <chunk_number>
```

## Example Final Output

```text
### Session 1

### Topic: Introduction to Photosynthesis

Explanation:
Plants convert light energy into chemical energy.

Key Points:
- Role of chlorophyll
- Light-dependent reactions

Student Doubts:
Q: Why are leaves green?
A: Because chlorophyll reflects green wavelengths.
```

## Notes

- Ollama must be installed and running locally.
- The `llama3` model must be available locally, or you can change the default model in [`utils.py`](/abs/path/C:/Users/Admin/Documents/AI%20Projects/Transcript_auro_cleaning/utils.py).
- Session order and chunk order are preserved in the final merged file.
- If no transcript files are found, the script exits with a clear message.
