import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field
from retrieval_utils import combine_filters, detect_filters


# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Init clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("iitm-modules-rag")


# Config
MODULE_SESSIONS = {
    "cms": [1, 2, 3, 4],
    "map": [2, 3],
    "wdp": [1, 2, 3, 4, 5],
}

CHUNK_ID_PATTERN = re.compile(r"_chunk_(\d+)", re.IGNORECASE)
MAX_HISTORY_MESSAGES = 10
SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful AI assistant"}
RETRIEVAL_TOP_K = 25
FINAL_TOP_K = 5


# -------------------- SCHEMAS --------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    mode: str = "global"
    module: Optional[str] = None
    session: Optional[int] = None
    chat_history: list[ChatMessage] = Field(default_factory=list)


# -------------------- HEALTH --------------------

@app.get("/")
def root():
    return {"status": "API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# -------------------- META --------------------

@app.get("/modules")
def get_modules():
    return sorted(MODULE_SESSIONS.keys())


@app.get("/sessions")
def get_sessions(module: str = Query(...)):
    sessions = MODULE_SESSIONS.get(module)
    if sessions is None:
        raise HTTPException(status_code=404, detail=f"Unknown module: {module}")
    return sessions


# -------------------- HELPERS --------------------

def build_filter(req: ChatRequest) -> Optional[dict[str, Any]]:
    if req.mode == "global":
        return None

    if req.mode != "filtered":
        raise HTTPException(status_code=400, detail="Invalid mode")

    if not req.module or req.session is None:
        raise HTTPException(status_code=400, detail="Module + session required")

    return {"module": req.module, "session": req.session}


def extract_chunk_number(match: dict[str, Any]) -> Optional[int]:
    metadata = match.get("metadata", {})
    if metadata.get("chunk"):
        return int(metadata["chunk"])

    match_id = str(match.get("id", ""))
    m = CHUNK_ID_PATTERN.search(match_id)
    return int(m.group(1)) if m else None


def build_source_entry(match: dict[str, Any], index_position: int):
    metadata = match.get("metadata", {})
    module = metadata.get("module")
    session = metadata.get("session")
    chunk = extract_chunk_number(match)

    citation = f"{str(module).upper()}-S{session}-C{chunk or index_position}"

    return {
        "id": match.get("id"),
        "score": match.get("score"),
        "text": metadata.get("text", ""),
        "module": module,
        "session": session,
        "chunk": chunk,
        "citation": citation,
    }


def build_context(sources):
    return "\n\n".join(
        f"[{s['citation']}]\n{s['text']}" for s in sources if s.get("text")
    )


def normalize_chat_history(chat_history):
    return [
        {"role": m.role, "content": m.content}
        for m in chat_history
        if m.role in {"user", "assistant"} and m.content
    ][-MAX_HISTORY_MESSAGES:]


def retrieve_context(query, embedding, base_filter):
    filters = combine_filters(base_filter, detect_filters(query))

    results = index.query(
        vector=embedding,
        top_k=RETRIEVAL_TOP_K,
        include_metadata=True,
        filter=filters if filters else None,
    )

    matches = sorted(results.get("matches", []), key=lambda x: x["score"], reverse=True)
    return matches[:FINAL_TOP_K]


# -------------------- MAIN CHAT --------------------

@app.post("/chat")
def chat(req: ChatRequest):

    question = (req.question or req.query or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        # Rewrite
        rewrite = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": question}],
        )
        rewritten = rewrite.choices[0].message.content or question

        # Embed
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=rewritten,
        ).data[0].embedding

        # Retrieve
        docs = retrieve_context(rewritten, embedding, build_filter(req))

        if not docs:
            return {"answer": "Not in module.", "sources": []}

        sources = [build_source_entry(d, i) for i, d in enumerate(docs, 1)]
        context = build_context(sources)

        # Answer
        answer_prompt = f"""
You are an expert instructor explaining concepts from a course.

Your goal is to generate a HIGH-QUALITY, DETAILED, TEACHING-STYLE answer using the provided context.

STRICT INSTRUCTIONS:

1. Always prioritize DEPTH and CLARITY over brevity.
2. Do NOT limit answers to 7-10 points.
3. Explain concepts as if teaching a student who is seeing this for the first time.
4. Use a natural explanation flow:
   - Start with a simple overview
   - Then break down the concept step-by-step
   - Then explain WHY it matters
   - Then, if relevant, give an example or analogy

5. Use structured formatting ONLY where helpful:
   - Headings
   - Subsections
   - Bullet points, but do not force them

6. If context is available:
   - Use it fully
   - Combine ideas across chunks
   - Do NOT say "not in module" unless absolutely no relevant info exists

7. If partial context is available:
   - Answer using available context
   - Then intelligently fill gaps using reasoning
   - NEVER abruptly stop

8. Avoid generic short answers.
9. Avoid robotic formatting.
10. Write like a human instructor, not a checklist generator.

---

CONTEXT:
{context}

---

QUESTION:
{question}

---

Now generate a COMPLETE and DETAILED answer.
""".strip()

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": answer_prompt},
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
