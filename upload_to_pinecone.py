import os
import logging
import re
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


def load_chunks(base_dir: Path) -> List[Dict]:
    """Load all chunk files from rag_chunks/ subfolders with module info."""
    chunks = []
    rag_dir = base_dir / "rag_chunks"
    if not rag_dir.exists():
        logging.warning("rag_chunks directory not found")
        return chunks

    for module_dir in rag_dir.iterdir():
        if module_dir.is_dir():
            module = module_dir.name.lower()
            for chunk_file in module_dir.glob("*.txt"):
                try:
                    content = chunk_file.read_text(encoding="utf-8").strip()
                    if content:
                        chunks.append({
                            "path": chunk_file,
                            "module": module,
                            "filename": chunk_file.name,
                            "content": content
                        })
                except Exception as e:
                    logging.error(f"Failed to read {chunk_file}: {e}")
    return chunks


def extract_session_from_filename(filename: str) -> Optional[int]:
    """Extract session number from filename (e.g. cms_session_1_chunk_5.txt → 1)."""
    match = re.search(r"_session_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def extract_chunk_text(content: str) -> str:
    """Extract only the text content (skip metadata header lines)."""
    lines = content.split("\n")
    # Skip first 4 lines: Module, Session, Topic, Chunk
    # Then skip the blank line
    text_start = 0
    for i, line in enumerate(lines):
        if i >= 4 and line.strip() == "":
            text_start = i + 1
            break
    
    return "\n".join(lines[text_start:]).strip()


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
        # Extract session from filename
        session = extract_session_from_filename(chunk["filename"])
        if session is None:
            logging.error(f"Could not extract session from {chunk['filename']}")
            failed_files.append(str(chunk["path"]))
            continue

        # Extract text content
        text_content = extract_chunk_text(chunk["content"])
        if not text_content:
            logging.warning(f"Empty text content in {chunk['filename']}")
            failed_files.append(str(chunk["path"]))
            continue

        try:
            # Create embedding
            embedding = create_embedding(text_content)
            
            # Create metadata with only module, session, text
            metadata = {
                "module": chunk["module"],
                "session": session,
                "text": text_content
            }
            print(metadata)
            
            # Create vector
            unique_id = f"{chunk['module']}_{session}_{chunk['filename']}"
            vector = {
                "id": unique_id,
                "values": embedding,
                "metadata": metadata
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
        print(f"\nUploading {len(vectors)} vectors...")
        print(f"Sample vector: {vectors[0]}")
        index.upsert(vectors=vectors)
        print("Upload complete\n")
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