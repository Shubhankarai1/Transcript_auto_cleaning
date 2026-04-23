import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()


app = FastAPI()


# Init clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("iitm-modules-rag")  # your index name


# Simple module/session mapping used by the UI selectors
MODULE_SESSIONS = {
    "cms": [1, 2, 3, 4],
    "map": [2, 3],
    "wdp": [1, 2, 3, 4, 5],
}

CHUNK_ID_PATTERN = re.compile(r"_chunk_(\d+)", re.IGNORECASE)
MAX_HISTORY_MESSAGES = 10
SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful AI assistant"}


# Request schema
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    mode: str
    module: Optional[str] = None
    session: Optional[int] = None
    chat_history: list[ChatMessage] = Field(default_factory=list)


# Basic health endpoint
@app.get("/")
def home() -> dict[str, str]:
    return {"status": "API running"}


# List available modules
@app.get("/modules")
def get_modules() -> list[str]:
    return sorted(MODULE_SESSIONS.keys())


# List sessions for a specific module
@app.get("/sessions")
def get_sessions(module: str = Query(..., description="Module name")) -> list[int]:
    sessions = MODULE_SESSIONS.get(module)
    if sessions is None:
        raise HTTPException(status_code=404, detail=f"Unknown module: {module}")
    return sessions


def build_filter(req: ChatRequest) -> Optional[dict[str, Any]]:
    if req.mode == "global":
        return None

    if req.mode != "filtered":
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'global' or 'filtered'.",
        )

    if not req.module or req.session is None:
        raise HTTPException(
            status_code=400,
            detail="Filtered mode requires both module and session.",
        )

    allowed_sessions = MODULE_SESSIONS.get(req.module)
    if allowed_sessions is None:
        raise HTTPException(status_code=404, detail=f"Unknown module: {req.module}")

    if req.session not in allowed_sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session {req.session} for module '{req.module}'.",
        )

    return {
        "module": req.module,
        "session": req.session,
    }


def extract_chunk_number(match: dict[str, Any]) -> Optional[int]:
    metadata = match.get("metadata", {})
    chunk_value = metadata.get("chunk")
    if chunk_value is not None:
        try:
            return int(chunk_value)
        except (TypeError, ValueError):
            pass

    match_id = str(match.get("id", ""))
    chunk_match = CHUNK_ID_PATTERN.search(match_id)
    if chunk_match:
        return int(chunk_match.group(1))

    return None


def build_source_entry(match: dict[str, Any], index_position: int) -> dict[str, Any]:
    metadata = match.get("metadata", {})
    module = metadata.get("module")
    session = metadata.get("session")
    chunk = extract_chunk_number(match)

    module_label = str(module).upper() if module else "UNKNOWN"
    session_label = str(session) if session is not None else "?"
    chunk_label = str(chunk) if chunk is not None else str(index_position)
    citation = f"{module_label}-S{session_label}-C{chunk_label}"

    return {
        "id": match.get("id"),
        "score": match.get("score"),
        "text": metadata.get("text", ""),
        "module": module,
        "session": session,
        "chunk": chunk,
        "citation": citation,
    }


def build_context(sources: list[dict[str, Any]]) -> str:
    context_parts: list[str] = []
    for source in sources:
        text = source.get("text", "")
        if not text:
            continue
        context_parts.append(f"[{source['citation']}]\n{text}")
    return "\n\n".join(context_parts)


def normalize_chat_history(chat_history: list[ChatMessage]) -> list[dict[str, str]]:
    normalized_history: list[dict[str, str]] = []
    for message in chat_history:
        role = message.role.strip().lower()
        content = message.content.strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized_history.append({"role": role, "content": content})
    return normalized_history[-MAX_HISTORY_MESSAGES:]


def ensure_latest_user_message(
    chat_history: list[dict[str, str]], question: str
) -> list[dict[str, str]]:
    if chat_history and chat_history[-1]["role"] == "user" and chat_history[-1]["content"] == question:
        return chat_history
    return [*chat_history, {"role": "user", "content": question}][-MAX_HISTORY_MESSAGES:]


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pinecone_filter = build_filter(req)
    chat_history = ensure_latest_user_message(normalize_chat_history(req.chat_history), question)

    try:
        # 1. Rewrite the user query for retrieval
        rewrite_messages = [
            {
                "role": "system",
                "content": """
You are a query rewriting assistant for a RAG system.

Use the recent conversation to understand the latest user message.

Your job:
- Convert vague or short queries into specific, detailed queries
- Add missing context if implied
- Make it optimized for semantic search

Rules:
- Keep original intent
- Do not answer the question
- Only rewrite the latest user question
- Return only the rewritten query
""".strip(),
            },
            *chat_history,
        ]
        rewrite_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=rewrite_messages,
        )
        rewritten_query = rewrite_response.choices[0].message.content or question

        print("Original:", question)
        print("Rewritten:", rewritten_query)

        # 2. Embed rewritten query
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=rewritten_query,
        ).data[0].embedding

        # 3. Query Pinecone with or without filter
        query_kwargs: dict[str, Any] = {
            "vector": embedding,
            "top_k": 15,
            "include_metadata": True,
        }
        if pinecone_filter is not None:
            query_kwargs["filter"] = pinecone_filter

        results = index.query(**query_kwargs)
        matches = sorted(
            results.get("matches", []),
            key=lambda match: match.get("score", 0),
            reverse=True,
        )

        for match in matches:
            print("Score:", match.get("score"))

        filtered_matches = matches[:5]

        if not filtered_matches:
            return {
                "answer": "Not in module.",
                "sources": [],
            }

        source_entries = [
            build_source_entry(match, index_position)
            for index_position, match in enumerate(filtered_matches, start=1)
        ]

        # 4. Extract context
        context = build_context(source_entries)

        # 5. Generate answer
        messages = [
            SYSTEM_MESSAGE,
            {
                "role": "system",
                "content": f"""
Use the retrieved context to answer the latest user question in a detailed and structured way.

Rules:
- Explain concepts step-by-step
- Use examples where possible
- Expand ideas clearly instead of only summarizing
- Keep the answer grounded in the retrieved context
- Cite factual statements inline using the source labels from the context, for example [CMS-S3-C84]
- Include citations throughout the answer, not only at the end
- End with a short line starting with "Sources used:" and list the unique citations you relied on
- If the context does not contain the answer, say that clearly

Context:
{context}
""".strip(),
            },
            *chat_history,
        ]
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc

    return {
        "answer": response.choices[0].message.content,
        "sources": source_entries,
    }
