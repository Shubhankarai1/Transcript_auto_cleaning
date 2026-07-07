You are a senior Python engineer. Build a complete working project based on the provided PRD.md and requirements.txt.

## 🎯 Goal

Create a Python pipeline that:

* Reads transcript files from /input
* Splits into chunks (session-aware naming)
* Cleans + structures using OpenAI API
* Saves chunk files
* Merges into final_cleaned.txt

---

## 📁 Project Requirements

### 1. Folder Structure

Create this exact structure:

project/
│
├── input/
├── chunks/
├── output/
│
├── main.py
├── utils.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md

---

### 2. .env Setup

Create a `.env` file with:

OPENAI_API_KEY=your_api_key_here

Use python-dotenv to load it.

---

### 3. .gitignore

Create a `.gitignore` file with:

.env
**pycache**/
*.pyc
chunks/
output/
.venv/

---

### 4. Implementation Requirements

#### A. File Loader

* Read all files from /input
* Each file name format: session_X.txt

---

#### B. Session Detection

* Extract session number from filename

---

#### C. Chunking

* Split text into ~1200 word chunks
* Return list of chunks

---

#### D. Chunk Saving

Save each chunk as:

chunks/crms_session_<session_number>chunk-<chunk_number>.txt

---

#### E. OpenAI Cleaning Module

Use OpenAI Python SDK.

Model: gpt-4o-mini (or latest efficient model)

Prompt:

"You are cleaning a lecture transcript.

Rules:

* Remove filler words
* Keep only meaningful teaching content
* Separate student doubts
* Structure clearly

Output format:

### Topic:

Explanation:
Key Points:
Student Doubts:

Transcript:
{chunk}"

---

#### F. Processing Loop

For each session:

* Split → Save chunks → Clean → Store results

Use tqdm for progress bar

---

#### G. Merge Output

* Combine all cleaned chunks
* Maintain session order

Format:

### Session X

[cleaned chunks]

---

#### H. Final Output

Save to:

output/final_cleaned.txt

---

### 5. Code Quality

* Modular (utils.py for logic)

* Clear functions:

  * load_files()
  * split_text()
  * clean_chunk()
  * save_chunk()
  * merge_output()

* Add error handling

* Add print/logging statements

---

### 6. README.md

Include:

* Setup instructions
* How to add transcripts
* How to run project
* Example output

---

### 7. Execution

User should be able to run:

pip install -r requirements.txt
python main.py

---

## ⚠️ Important

* Do NOT overcomplicate
* Keep it simple and runnable
* Ensure it works end-to-end

---

Now implement the full project.
#### D. Chunk Naming (STRICT REQUIREMENT — DO NOT CHANGE)

Each chunk MUST follow EXACT naming:

crms_session_<session_number>chunk-<chunk_number>.txt

Examples:

* crms_session_1chunk-1.txt
* crms_session_1chunk-2.txt
* crms_session_2chunk-1.txt

Rules:

* Use lowercase only
* Use underscores (_) exactly as shown
* No spaces
* No deviations in naming format

If this format is not followed, the implementation is incorrect.

---

Additionally, inside the file content, prepend:

Session <session_number> - Chunk <chunk_number>

at the top of each chunk file.
