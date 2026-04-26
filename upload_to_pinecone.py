import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
METADATA_MODEL = os.getenv("OPENAI_METADATA_MODEL", "gpt-4o-mini")
EMBEDDING_BATCH_SIZE = 100
VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}
DEFAULT_SOURCE_TYPE = "lecture"
METADATA_PROMPT_TEMPLATE = """Analyze the following educational chunk and extract:

* Main topic
* Subtopic
* 5-10 keywords
* Difficulty level (beginner/intermediate/advanced)

Return ONLY valid JSON.

Chunk:
{chunk_text}"""

COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not set in .env")

# Connect to index
index = pinecone_client.Index(index_name)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_chunks(base_dir: Path) -> list[dict[str, Any]]:
    """Load all chunk files from rag_chunks/ subfolders with parsed identifiers."""
    chunks: list[dict[str, Any]] = []
    rag_dir = base_dir / "rag_chunks"
    if not rag_dir.exists():
        logging.warning("rag_chunks directory not found")
        return chunks

    for module_dir in sorted(path for path in rag_dir.iterdir() if path.is_dir()):
        module_name = module_dir.name.lower()
        for chunk_file in sorted(module_dir.glob("*.txt")):
            session_id = extract_session_from_filename(chunk_file.name)
            chunk_index = extract_chunk_from_filename(chunk_file.name)

            if session_id is None or chunk_index is None:
                logging.warning("Skipping chunk with invalid filename format: %s", chunk_file.name)
                continue

            try:
                content = chunk_file.read_text(encoding="utf-8").strip()
            except Exception as exc:
                logging.error("Failed to read %s: %s", chunk_file, exc)
                continue

            chunk_text = extract_chunk_text(content)
            if not chunk_text:
                logging.warning("Skipping empty chunk text: %s", chunk_file.name)
                continue

            chunks.append(
                {
                    "path": chunk_file,
                    "module_name": module_name,
                    "session_id": session_id,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                }
            )

    return chunks


def extract_session_from_filename(filename: str) -> int | None:
    match = re.search(r"_session_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def extract_chunk_from_filename(filename: str) -> int | None:
    match = re.search(r"_chunk_(\d+)\.txt$", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_chunk_text(content: str) -> str:
    """Extract only the text body from the stored chunk file."""
    lines = content.splitlines()
    text_start = 0
    for index, line in enumerate(lines):
        if index >= 4 and not line.strip():
            text_start = index + 1
            break
    return "\n".join(lines[text_start:]).strip()


def build_document_id(module_name: str, session_id: int) -> str:
    return f"{module_name}_{session_id}"


def build_chunk_id(module_name: str, session_id: int, chunk_index: int) -> str:
    return f"{module_name}_{session_id}_{chunk_index}"


def _load_json_object(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("Metadata response is not a JSON object")

    return payload


def _fallback_keywords(chunk_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", chunk_text.lower()):
        if token in COMMON_STOPWORDS or token in seen:
            continue
        seen.add(token)
        candidates.append(token)
        if len(candidates) == 10:
            break

    while len(candidates) < 5:
        candidates.append(f"keyword_{len(candidates) + 1}")

    return candidates


def _normalize_keywords(raw_keywords: Any, chunk_text: str) -> list[str]:
    if isinstance(raw_keywords, list):
        keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]
    elif isinstance(raw_keywords, str):
        keywords = [item.strip() for item in re.split(r",|\n", raw_keywords) if item.strip()]
    else:
        keywords = []

    normalized: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        lowered = keyword.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(keyword)
        if len(normalized) == 10:
            break

    if len(normalized) < 5:
        normalized = _fallback_keywords(chunk_text)

    return normalized[:10]


def _normalize_difficulty(raw_difficulty: Any) -> str:
    difficulty = str(raw_difficulty or "").strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        return "intermediate"
    return difficulty


def extract_metadata(chunk_text: str, session_id: int, module_name: str) -> dict[str, Any]:
    """
    Extract semantic metadata for a chunk using OpenAI.

    `chunk_id` is assigned in `process_chunk()` because it depends on the
    chunk index from the current chunking pass.
    """
    prompt = METADATA_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    response = openai_client.chat.completions.create(
        model=METADATA_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured metadata for educational transcript chunks. "
                    "Return only valid JSON with keys: topic, subtopic, keywords, difficulty."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw_response = response.choices[0].message.content or "{}"
    parsed_metadata = _load_json_object(raw_response)

    topic = str(parsed_metadata.get("topic") or "General topic").strip()
    subtopic = str(parsed_metadata.get("subtopic") or topic).strip()
    keywords = _normalize_keywords(parsed_metadata.get("keywords"), chunk_text)
    difficulty = _normalize_difficulty(parsed_metadata.get("difficulty"))

    return {
        "document_id": build_document_id(module_name, session_id),
        "module": module_name,
        "topic": topic,
        "subtopic": subtopic,
        "session": int(session_id),
        "chunk_id": "",
        "keywords": keywords,
        "source_type": DEFAULT_SOURCE_TYPE,
        "difficulty": difficulty,
    }


def process_chunk(
    chunk_text: str,
    session_id: int,
    module_name: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Return the structured chunk object expected by the embedding pipeline."""
    metadata = extract_metadata(chunk_text, session_id, module_name)
    metadata["chunk_id"] = build_chunk_id(module_name, session_id, chunk_index)

    logging.info(
        "Metadata created for %s | topic=%s | subtopic=%s | difficulty=%s | keywords=%s",
        metadata["chunk_id"],
        metadata["topic"],
        metadata["subtopic"],
        metadata["difficulty"],
        ", ".join(metadata["keywords"]),
    )

    return {
        "chunk_text": chunk_text,
        "metadata": metadata,
    }


def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of chunk texts using OpenAI."""
    response = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def embed_and_store(processed_chunks: list[dict[str, Any]], batch_size: int = EMBEDDING_BATCH_SIZE) -> None:
    """Embed processed chunks and upsert chunk text plus metadata to Pinecone."""
    if not processed_chunks:
        logging.info("No processed chunks to embed")
        return

    total_uploaded = 0

    for start_index in range(0, len(processed_chunks), batch_size):
        batch = processed_chunks[start_index : start_index + batch_size]
        embeddings = create_embeddings([item["chunk_text"] for item in batch])
        vectors = []

        for item, embedding in zip(batch, embeddings):
            metadata = dict(item["metadata"])
            chunk_index = int(metadata["chunk_id"].rsplit("_", 1)[-1])

            # Keep text and numeric chunk index in Pinecone metadata for downstream retrieval.
            storage_metadata = {
                **metadata,
                "chunk": chunk_index,
                "text": item["chunk_text"],
            }
            vectors.append(
                {
                    "id": metadata["chunk_id"],
                    "values": embedding,
                    "metadata": storage_metadata,
                }
            )

        index.upsert(vectors=vectors)
        total_uploaded += len(vectors)
        logging.info("Uploaded %s/%s vectors to Pinecone", total_uploaded, len(processed_chunks))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_chunks = load_chunks(base_dir)
    if not raw_chunks:
        logging.info("No chunks found to upload")
        return

    processed_chunks: list[dict[str, Any]] = []
    failed_files: list[str] = []

    for chunk in raw_chunks:
        try:
            processed_chunks.append(
                process_chunk(
                    chunk_text=chunk["chunk_text"],
                    session_id=chunk["session_id"],
                    module_name=chunk["module_name"],
                    chunk_index=chunk["chunk_index"],
                )
            )
        except Exception as exc:
            logging.error("Failed to create metadata for %s: %s", chunk["path"], exc)
            failed_files.append(str(chunk["path"]))

    logging.info("Prepared %s structured chunks for embedding", len(processed_chunks))

    if not processed_chunks:
        logging.info("No processed chunks available for embedding")
        return

    try:
        embed_and_store(processed_chunks)
        logging.info("Successfully uploaded %s vectors to Pinecone", len(processed_chunks))
        logging.info("Index stats: %s", index.describe_index_stats())
    except Exception as exc:
        logging.error("Failed to upload vectors: %s", exc)
        raise

    if failed_files:
        logging.warning("Failed to process %s files: %s", len(failed_files), failed_files)

    logging.info("Completed successfully")


if __name__ == "__main__":
    main()
