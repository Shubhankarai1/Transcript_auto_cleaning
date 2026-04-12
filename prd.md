# PRD: Transcript Auto Cleaning + Structuring Pipeline (Session-Aware, RAG Ready)

## 🎯 Objective

Build a Python tool that:

1. Takes raw lecture transcripts (per session)
2. Automatically splits them into chunks
3. Labels each chunk with session-aware naming
4. Cleans and structures each chunk using an LLM
5. Merges outputs into a final clean document
6. Outputs a file ready for upload into AnythingLLM

---

## 📥 Input

* Format: `.txt`
* One transcript per session
* Example:

  * session_1.txt
  * session_2.txt

---

## 📤 Output

### 1. Chunk Files

Each chunk MUST follow this naming format:

session_<session_number>*chunk*<chunk_number>.txt

Example:

* session_1_chunk_1.txt
* session_1_chunk_2.txt
* session_2_chunk_1.txt

---

### 2. Final Cleaned Output

* File: `final_cleaned.txt`
* Structured format:

### Session 1

#### Topic: [Topic Name]

Explanation:
...

Key Points:

* ...
* ...

Student Doubts:
Q:
A:

---

---

## ⚙️ Functional Requirements

### 1. File Loader

* Read all `.txt` files from `/input`
* Each file represents ONE session

---

### 2. Session Identification

* Extract session number from filename
  Example:

  * session_1.txt → session = 1

---

### 3. Chunking Module

* Split transcript into chunks:

  * Size: ~1200 words
* Ensure:

  * No mid-sentence splits (optional improvement)
* Output:

  * List of chunks per session

---

### 4. Chunk Naming Logic

Each chunk MUST be saved as:

session_<session_number>*chunk*<chunk_index>.txt

Where:

* session_number = extracted from file
* chunk_index = incremental (1, 2, 3…)

---

### 5. Cleaning + Structuring (Core AI Module)

Use LLM (OpenAI API)

Prompt:

"You are cleaning a lecture transcript.

Rules:

* Remove filler words and noise
* Keep only meaningful teaching content
* Separate teacher explanation and student doubts
* Organize into structured format

Output format:

### Topic:

Explanation:
Key Points:
Student Doubts:

Transcript:
{chunk}"

---

### 6. Processing Loop

For each session:

* Split into chunks
* For each chunk:

  * Save raw chunk file (with session naming)
  * Send to LLM
  * Store cleaned output

---

### 7. Merge Module

* Combine cleaned outputs in this order:

  * Session-wise
  * Chunk order preserved

Structure:

### Session 1

[All cleaned chunks]

### Session 2

[All cleaned chunks]

---

### 8. Final Refinement Pass (Optional)

Prompt:

"Clean this document:

* Remove duplicate topics
* Merge similar concepts
* Improve clarity
* Keep structured format"

---

### 9. Export Module

Save:

* `/output/final_cleaned.txt`

---

## 🧱 Tech Stack

* Python
* OpenAI API (GPT-4 / GPT-4o-mini)
* Optional:

  * tqdm (progress)
  * dotenv (API keys)

---

## 📂 Folder Structure

project/
│
├── input/
│   ├── session_1.txt
│   ├── session_2.txt
│
├── chunks/
│   ├── session_1_chunk_1.txt
│   ├── session_1_chunk_2.txt
│
├── output/
│   └── final_cleaned.txt
│
├── main.py
├── utils.py
└── .env

---

## 🚀 Execution Flow

1. Add session files to `/input`

2. Run:

   python main.py

3. Script:

   * Detects sessions
   * Splits into chunks
   * Names them correctly
   * Cleans via LLM
   * Merges output

4. Final file available in `/output`

---

## 🔥 Success Criteria

* Each chunk clearly labeled by session
* Output structured and readable
* Ready for direct upload into AnythingLLM
* No manual intervention required

---

## ⚠️ Notes

* Session-based naming is critical for:

  * Traceability
  * Debugging
  * Context-aware querying

---

## 💡 Future Enhancements

* Multi-threaded chunk processing
* UI dashboard
* Direct ingestion into vector DB
* Speaker detection (Teacher vs Student)

---
