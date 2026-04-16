from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import requests


SESSION_FILE_PATTERN = re.compile(r"^session_(\d+)\.txt$", re.IGNORECASE)
MODULE_DIR_PATTERN = re.compile(r"^[a-z0-9_]+$", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
SESSION_SECTION_PATTERN = re.compile(r"(?m)^### Session (\d+)\n\n")


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
        "session_output": base_dir / "output" / "sessions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def normalize_module_name(module_name: str) -> str:
    return module_name.strip().lower()


def format_module_name(module_name: str) -> str:
    return normalize_module_name(module_name).upper()


def extract_session_number(filename: str) -> int:
    match = SESSION_FILE_PATTERN.match(filename)
    if not match:
        raise ValueError(
            f"Invalid file name '{filename}'. Expected format: session_<number>.txt"
        )
    return int(match.group(1))


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_session_hash_path(
    session_output_dir: Path,
    module_name: str,
    session_number: int,
) -> Path:
    module_key = normalize_module_name(module_name)
    module_session_dir = session_output_dir / module_key
    module_session_dir.mkdir(parents=True, exist_ok=True)
    return module_session_dir / f"session_{session_number}_cache.json"


def load_session_hash(
    session_output_dir: Path,
    module_name: str,
    session_number: int,
) -> Optional[Dict[str, str]]:
    hash_path = get_session_hash_path(
        session_output_dir,
        module_name,
        session_number,
    )
    if not hash_path.exists():
        return None

    try:
        return json.loads(hash_path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def save_session_hash(
    session_output_dir: Path,
    module_name: str,
    session_number: int,
    source_hash: str,
) -> Path:
    hash_path = get_session_hash_path(
        session_output_dir,
        module_name,
        session_number,
    )
    hash_path.write_text(
        json.dumps({"source_hash": source_hash}, indent=2),
        encoding="utf-8",
    )
    logging.info("Saved session cache metadata: %s", hash_path)
    return hash_path


def load_cached_session_output(
    session_output_dir: Path,
    module_name: str,
    session_number: int,
    transcript_text: str,
    transcript_path: str | None = None,
) -> str | None:
    session_path = get_session_output_path(
        session_output_dir,
        module_name,
        session_number,
    )
    if not session_path.exists():
        return None

    current_hash = compute_text_hash(transcript_text)
    metadata = load_session_hash(
        session_output_dir,
        module_name,
        session_number,
    )

    if metadata and metadata.get("source_hash") == current_hash:
        return session_path.read_text(encoding="utf-8").strip()

    if metadata is None and transcript_path is not None:
        transcript_mtime = Path(transcript_path).stat().st_mtime
        if session_path.stat().st_mtime >= transcript_mtime:
            save_session_hash(
                session_output_dir,
                module_name,
                session_number,
                current_hash,
            )
            return session_path.read_text(encoding="utf-8").strip()

    return None


def load_files(input_dir: Path) -> List[Dict[str, str | int]]:
    session_files: List[Dict[str, str | int]] = []

    for module_dir in sorted(
        (path for path in input_dir.iterdir() if path.is_dir()),
        key=lambda item: item.name.lower(),
    ):
        module_name = normalize_module_name(module_dir.name)
        if not MODULE_DIR_PATTERN.match(module_name):
            logging.warning(
                "Skipping module folder with unsupported name: %s",
                module_dir.name,
            )
            continue

        for session_path in sorted(
            module_dir.glob("*.txt"),
            key=lambda item: item.name.lower(),
        ):
            session_number = extract_session_number(session_path.name)
            text = session_path.read_text(encoding="utf-8").strip()
            if not text:
                logging.warning("Skipping empty transcript file: %s", session_path)
                continue

            session_files.append(
                {
                    "module_name": module_name,
                    "session_number": session_number,
                    "filename": session_path.name,
                    "path": str(session_path),
                    "text": text,
                }
            )

    session_files.sort(
        key=lambda item: (
            str(item["module_name"]).lower(),
            int(item["session_number"]),
        )
    )
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


def save_chunk(
    chunks_dir: Path,
    module_name: str,
    session_number: int,
    chunk_number: int,
    chunk_text: str,
) -> Path:
    module_key = normalize_module_name(module_name)
    module_chunks_dir = chunks_dir / module_key
    module_chunks_dir.mkdir(parents=True, exist_ok=True)
    file_path = (
        module_chunks_dir
        / f"{module_key}_session_{session_number}chunk-{chunk_number}.txt"
    )
    header = f"Session {session_number} - Chunk {chunk_number}\n\n"
    file_path.write_text(f"{header}{chunk_text.strip()}\n", encoding="utf-8")
    logging.info("Saved chunk file: %s", file_path)
    return file_path


def clean_chunk(
    chunk: str,
    model: str = "mistral",
) -> str:
    prompt = (
        "You are cleaning a lecture transcript.\n\n"
        "Rules:\n"
        "- Remove filler words\n"
        "- Keep only meaningful teaching content\n"
        "- Separate student doubts\n"
        "- Structure clearly\n"
        "- Be concise and well formatted\n"
        "- STRICTLY follow the output format. Do not skip any section\n\n"
        "Output format:\n\n"
        "### Topic:\n"
        "Explanation:\n"
        "Key Points:\n"
        "Student Doubts:\n\n"
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


def merge_session_chunks(cleaned_chunks: List[str]) -> str:
    return "\n\n---\n\n".join(cleaned_chunks).strip()


def get_session_output_path(
    session_output_dir: Path,
    module_name: str,
    session_number: int,
) -> Path:
    module_key = normalize_module_name(module_name)
    module_session_dir = session_output_dir / module_key
    module_session_dir.mkdir(parents=True, exist_ok=True)
    return module_session_dir / f"session_{session_number}_cleaned.txt"


def load_session_output(
    session_output_dir: Path,
    module_name: str,
    session_number: int,
) -> str | None:
    session_path = get_session_output_path(
        session_output_dir,
        module_name,
        session_number,
    )
    if not session_path.exists():
        return None
    return session_path.read_text(encoding="utf-8").strip()


def save_session_output(
    session_output_dir: Path,
    module_name: str,
    session_number: int,
    content: str,
    source_text: str | None = None,
) -> Path:
    session_path = get_session_output_path(session_output_dir, module_name, session_number)
    session_path.write_text(content.strip() + "\n", encoding="utf-8")
    logging.info("Saved session output: %s", session_path)

    if source_text is not None:
        source_hash = compute_text_hash(source_text)
        save_session_hash(session_output_dir, module_name, session_number, source_hash)

    return session_path


def bootstrap_module_session_cache(output_dir: Path, session_output_dir: Path) -> int:
    restored = 0
    separator = "\n\n---\n\n"

    for final_path in sorted(output_dir.glob("*_final_cleaned.txt")):
        module_name = normalize_module_name(
            final_path.name[: -len("_final_cleaned.txt")]
        )
        module_session_dir = session_output_dir / module_name
        existing_cache_files = list(module_session_dir.glob("session_*_cleaned.txt"))
        if existing_cache_files:
            continue

        content = final_path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        matches = list(SESSION_SECTION_PATTERN.finditer(content))
        if not matches:
            continue

        for index, match in enumerate(matches):
            session_number = int(match.group(1))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            session_body = content[start:end]
            if session_body.endswith(separator):
                session_body = session_body[: -len(separator)]

            session_body = session_body.strip()
            if not session_body:
                continue

            save_session_output(
                session_output_dir,
                module_name,
                session_number,
                session_body,
            )
            restored += 1

    if restored:
        logging.info("Bootstrapped %s session cache files from module outputs.", restored)
    return restored


def merge_module_output(cleaned_sessions: Dict[int, str]) -> str:
    sections: List[str] = []
    for session_number in sorted(cleaned_sessions):
        session_body = cleaned_sessions[session_number].strip()
        sections.append(f"### Session {session_number}\n\n{session_body}")
    return "\n\n---\n\n".join(sections).strip() + "\n"


def save_module_output(output_dir: Path, module_name: str, content: str) -> Path:
    module_key = normalize_module_name(module_name)
    final_path = output_dir / f"{module_key}_final_cleaned.txt"
    final_path.write_text(content, encoding="utf-8")
    logging.info("Saved module output: %s", final_path)
    return final_path
