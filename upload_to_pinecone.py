import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not set in .env")

# Connect to index
index = pinecone_client.Index(index_name)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_chunks(base_dir: Path) -> List[Dict[str, str]]:
    """Load all chunk files from rag_chunks/ subfolders."""
    chunks = []
    rag_dir = base_dir / "rag_chunks"
    if not rag_dir.exists():
        logging.warning("rag_chunks directory not found")
        return chunks

    for module_dir in rag_dir.iterdir():
        if module_dir.is_dir():
            for chunk_file in module_dir.glob("*.txt"):
                try:
                    content = chunk_file.read_text(encoding="utf-8").strip()
                    if content:
                        chunks.append({"path": chunk_file, "content": content})
                except Exception as e:
                    logging.error(f"Failed to read {chunk_file}: {e}")
    return chunks


def extract_metadata(content: str) -> Optional[Dict[str, str]]:
    """Extract metadata from chunk content."""
    lines = content.split("\n")
    metadata = {}
    try:
        metadata["module"] = lines[0].split(": ")[1]
        metadata["session"] = int(lines[1].split(": ")[1])
        metadata["topic"] = lines[2].split(": ")[1]
        metadata["chunk_id"] = int(lines[3].split(": ")[1])
        # Text starts after the blank line
        text_start = content.find("\n\n") + 2
        metadata["text"] = content[text_start:].strip()
        return metadata
    except (IndexError, ValueError) as e:
        logging.error(f"Failed to extract metadata: {e}")
        return None


def create_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    try:
        response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to create embedding: {e}")
        raise


def main():
    base_dir = Path(__file__).resolve().parent
    chunks = load_chunks(base_dir)
    if not chunks:
        logging.info("No chunks found to upload")
        return

    vectors = []
    failed_files = []

    for chunk in chunks:
        metadata = extract_metadata(chunk["content"])
        if not metadata:
            failed_files.append(str(chunk["path"]))
            continue

        try:
            embedding = create_embedding(metadata["text"])
            unique_id = f"{metadata['module']}_{metadata['session']}_{metadata['chunk_id']}"
            vector = {
                "id": unique_id,
                "values": embedding,
                "metadata": {
                    "module": metadata["module"],
                    "text": metadata["text"],
                    "session": metadata["session"],
                    "chunk": metadata["chunk_id"],
                },
            }
            vectors.append(vector)
        except Exception as e:
            logging.error(f"Failed to process {chunk['path']}: {e}")
            failed_files.append(str(chunk["path"]))

    logging.info(f"Created {len(vectors)} vectors")

    if not vectors:
        logging.info("No vectors to upload")
        return

    try:
        print(f"Uploading {len(vectors)} vectors...")
        index.upsert(vectors=vectors)
        print("Upload complete")
        logging.info(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
        # Verify
        stats = index.describe_index_stats()
        logging.info(f"Index stats: {stats}")
    except Exception as e:
        logging.error(f"Failed to upload vectors: {e}")
        raise

    if failed_files:
        logging.warning(f"Failed to process {len(failed_files)} files: {failed_files}")

    logging.info("Completed successfully")


if __name__ == "__main__":
    main()