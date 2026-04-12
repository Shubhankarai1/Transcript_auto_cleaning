from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

import requests


SESSION_FILE_PATTERN = re.compile(r"^session_(\d+)\.txt$", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_directories(base_dir: Path) -> Dict[str, Path]:
    paths = {
        "input": base_dir / "input",
        "chunks": base_dir / "chunks",
        "output": base_dir / "output",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def extract_session_number(filename: str) -> int:
    match = SESSION_FILE_PATTERN.match(filename)
    if not match:
        raise ValueError(
            f"Invalid file name '{filename}'. Expected format: session_<number>.txt"
        )
    return int(match.group(1))


def load_files(input_dir: Path) -> List[Dict[str, str | int]]:
    session_files: List[Dict[str, str | int]] = []

    for path in sorted(input_dir.glob("*.txt"), key=lambda item: item.name.lower()):
        session_number = extract_session_number(path.name)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logging.warning("Skipping empty transcript file: %s", path.name)
            continue

        session_files.append(
            {
                "session_number": session_number,
                "filename": path.name,
                "text": text,
            }
        )

    session_files.sort(key=lambda item: int(item["session_number"]))
    return session_files


def split_text(text: str, chunk_size: int = 1200) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    if len(sentences) == 1:
        return _split_by_words(words, chunk_size)

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_word_count = len(sentence.split())

        if sentence_word_count > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []
                current_word_count = 0
            chunks.extend(_split_by_words(sentence.split(), chunk_size))
            continue

        if current_word_count + sentence_word_count > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences).strip())
            current_sentences = [sentence]
            current_word_count = sentence_word_count
        else:
            current_sentences.append(sentence)
            current_word_count += sentence_word_count

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return chunks


def _split_by_words(words: List[str], chunk_size: int) -> List[str]:
    return [
        " ".join(words[index : index + chunk_size]).strip()
        for index in range(0, len(words), chunk_size)
    ]


def save_chunk(chunks_dir: Path, session_number: int, chunk_number: int, chunk_text: str) -> Path:
    file_path = chunks_dir / f"session_{session_number}_chunk_{chunk_number}.txt"
    header = f"Session {session_number} - Chunk {chunk_number}\n\n"
    file_path.write_text(f"{header}{chunk_text.strip()}\n", encoding="utf-8")
    logging.info("Saved chunk file: %s", file_path.name)
    return file_path


def clean_chunk(
    chunk: str,
    model: str = "llama3",
) -> str:
    prompt = (
        "You are cleaning a lecture transcript.\n\n"
        "Rules:\n"
        "- Remove filler words\n"
        "- Keep only meaningful teaching content\n"
        "- Separate student doubts\n"
        "- Structure clearly\n"
        "- Be concise and well formatted\n\n"
        "Output format:\n"
        "### Topic:\n"
        "Explanation:\n"
        "Key Points:\n"
        "Student Doubts:\n"
        "\n"
        f"Transcript:\n{chunk}"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()

    cleaned_text = response.json().get("response", "").strip()
    if not cleaned_text:
        raise ValueError("Ollama returned an empty response while cleaning a chunk.")
    return cleaned_text


def merge_output(cleaned_by_session: Dict[int, List[str]]) -> str:
    sections: List[str] = []
    for session_number in sorted(cleaned_by_session):
        session_chunks = cleaned_by_session[session_number]
        session_body = "\n\n---\n\n".join(session_chunks)
        sections.append(f"### Session {session_number}\n\n{session_body}")
    return "\n\n---\n\n".join(sections).strip() + "\n"


def save_final_output(output_dir: Path, content: str) -> Path:
    final_path = output_dir / "final_cleaned.txt"
    final_path.write_text(content, encoding="utf-8")
    logging.info("Saved final cleaned output: %s", final_path)
    return final_path
