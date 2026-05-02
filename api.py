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
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
QUERY_EXPANSION_COUNT = 3
HYDE_MAX_TOKENS = 180


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
        "matched_query": match.get("matched_query"),
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


def parse_query_lines(text: str) -> list[str]:
    """Parse one-query-per-line LLM output into clean query strings."""
    queries: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[\).\:-])\s*", "", line).strip()
        cleaned = cleaned.strip("\"'")
        if cleaned:
            queries.append(cleaned)
    return queries


def unique_queries(queries: list[str]) -> list[str]:
    """Keep query variants in order while removing near-identical duplicates."""
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(query.strip())
    return unique


def rewrite_query(question: str) -> str:
    rewrite = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": question}],
    )
    return (rewrite.choices[0].message.content or question).strip()


def expand_query(question: str, rewritten: str) -> list[str]:
    prompt = f"""
Generate {QUERY_EXPANSION_COUNT} alternate search queries for retrieving relevant course transcript chunks.

Preserve the user's intent. Use different wording and related course terms where useful.
Return only the alternate queries, one per line. Do not number them.

Original question:
{question}

Current rewritten query:
{rewritten}
""".strip()

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception:
        logging.exception("Query expansion failed; falling back to base queries")
        return []

    expanded = parse_query_lines(response.choices[0].message.content or "")
    return expanded[:QUERY_EXPANSION_COUNT]


def generate_hyde_query(question: str, rewritten: str) -> str:
    prompt = f"""
Write a concise hypothetical course-transcript passage that would directly answer the student's question.

Use likely terminology from AI agent workflows, RAG, retrieval, planning, LangChain/LangGraph, Pinecone, reranking, or multi-agent systems only when relevant.
Do not mention that this is hypothetical. Do not add citations. Keep it under 120 words.

Student question:
{question}

Rewritten query:
{rewritten}
""".strip()

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=HYDE_MAX_TOKENS,
        )
    except Exception:
        logging.exception("HyDE generation failed; continuing without HyDE")
        return ""

    return (response.choices[0].message.content or "").strip()


def build_search_queries(question: str, rewritten: str) -> tuple[list[str], str]:
    hyde_query = generate_hyde_query(question, rewritten)
    expanded = expand_query(question, rewritten)
    queries = unique_queries([question, rewritten, hyde_query, *expanded])
    logging.info("Using %d retrieval queries: %s", len(queries), queries)
    return queries, hyde_query


def create_embeddings(queries: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=queries,
    )
    return [item.embedding for item in response.data]


def deduplicate_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate pooled query results, retaining the highest score per chunk."""
    best_by_key: dict[str, dict[str, Any]] = {}

    for match in matches:
        metadata = match.get("metadata", {})
        key = str(match.get("id") or metadata.get("text") or "")
        if not key:
            continue

        existing = best_by_key.get(key)
        if existing is None or match.get("score", 0) > existing.get("score", 0):
            best_by_key[key] = match

    return list(best_by_key.values())


def retrieve_context(original_query: str, search_queries: list[str], base_filter):
    filters = combine_filters(base_filter, detect_filters(original_query))
    embeddings = create_embeddings(search_queries)

    pooled_matches: list[dict[str, Any]] = []
    for query, embedding in zip(search_queries, embeddings):
        results = index.query(
            vector=embedding,
            top_k=RETRIEVAL_TOP_K,
            include_metadata=True,
            filter=filters if filters else None,
        )
        for match in results.get("matches", []):
            pooled_matches.append(
                {
                    "id": match.get("id"),
                    "score": match.get("score"),
                    "metadata": match.get("metadata", {}),
                    "matched_query": query,
                }
            )

    matches = sorted(
        deduplicate_matches(pooled_matches),
        key=lambda x: x["score"],
        reverse=True,
    )
    logging.info(
        "Retrieved %d pooled matches and retained %d unique matches",
        len(pooled_matches),
        len(matches),
    )
    return matches[:FINAL_TOP_K]


# -------------------- MAIN CHAT --------------------

@app.post("/chat")
def chat(req: ChatRequest):

    question = (req.question or req.query or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        # Rewrite
        rewritten = rewrite_query(question)
        search_queries, hyde_query = build_search_queries(question, rewritten)

        # Retrieve
        docs = retrieve_context(question, search_queries, build_filter(req))

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
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": answer_prompt},
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
            "retrieval_queries": search_queries,
            "hyde_query": hyde_query,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
