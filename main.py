from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from api import app
from utils import (
    bootstrap_module_session_cache,
    clean_chunk,
    ensure_directories,
    load_files,
    load_cached_session_output,
    merge_module_output,
    merge_session_chunks,
    save_chunk,
    save_module_output,
    save_session_output,
    setup_logging,
    split_text,
)


def split_by_session(text: str) -> list[str]:
    session_pattern = re.compile(r"(?m)^###\s*Session\s+\d+.*$")
    matches = list(session_pattern.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []

    sessions: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        session_text = text[start:end].strip()
        if session_text:
            sessions.append(session_text)
    return sessions


def split_by_topic(session_text: str) -> list[str]:
    topic_pattern = re.compile(r"(?m)^###\s*Topic\s*:\s*.*$")
    matches = list(topic_pattern.finditer(session_text))
    if not matches:
        return [session_text.strip()] if session_text.strip() else []

    topics: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(session_text)
        topic_text = session_text[start:end].strip()
        if topic_text:
            topics.append(topic_text)
    return topics


def extract_session_number(session_text: str) -> int:
    match = re.search(r"^###\s*Session\s+(\d+)", session_text, re.MULTILINE)
    if not match:
        raise ValueError("Unable to extract session number from session text")
    return int(match.group(1))


def extract_topic_name(topic_text: str) -> str:
    match = re.search(r"^###\s*Topic\s*:\s*(.+)$", topic_text, re.MULTILINE)
    if not match:
        raise ValueError("Unable to extract topic name from topic text")
    return match.group(1).strip()


def _split_large_paragraph(paragraph: str, max_words: int = 400) -> list[str]:
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + max_words]
        chunks.append(" ".join(chunk_words).strip())
        start += max_words
    return chunks


def _split_by_paragraphs(text: str, max_words: int = 400) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    result: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())
        if paragraph_word_count > max_words:
            if current_chunk:
                result.append("\n\n".join(current_chunk).strip())
                current_chunk = []
                current_word_count = 0
            result.extend(_split_large_paragraph(paragraph, max_words))
            continue

        if current_word_count + paragraph_word_count > max_words and current_chunk:
            result.append("\n\n".join(current_chunk).strip())
            current_chunk = [paragraph]
            current_word_count = paragraph_word_count
        else:
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count

    if current_chunk:
        result.append("\n\n".join(current_chunk).strip())

    return result


def save_rag_chunk(
    module: str,
    session_number: int,
    topic_name: str,
    chunk_id: int,
    body_text: str,
    base_dir: Path,
) -> Path:
    module_dir = base_dir / "rag_chunks" / module
    module_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{module}_session_{session_number}_chunk_{chunk_id}.txt"
    file_path = module_dir / filename

    sanitized_body = re.sub(r"^###\s*Topic\s*:\s*.*$\n", "", body_text, flags=re.MULTILINE).strip()
    content = (
        f"Module: {module}\n"
        f"Session: {session_number}\n"
        f"Topic: {topic_name}\n"
        f"Chunk: {chunk_id}\n\n"
        f"{sanitized_body}\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


def process_rag_chunking(base_dir: Path) -> None:
    output_dir = base_dir / "output"
    rag_root = base_dir / "rag_chunks"
    rag_root.mkdir(parents=True, exist_ok=True)

    final_files = sorted(output_dir.glob("*_final_cleaned.txt"))
    if not final_files:
        logging.info("No final_cleaned files found for RAG chunking in %s", output_dir)
        return

    for final_file in final_files:
        module = final_file.stem.replace("_final_cleaned", "")
        logging.info("Processing %s...", final_file.name)
        text = final_file.read_text(encoding="utf-8")
        session_texts = split_by_session(text)
        chunk_id = 0
        saved_paths: list[Path] = []

        for session_text in session_texts:
            session_number = extract_session_number(session_text)
            topic_texts = split_by_topic(session_text)

            for topic_text in topic_texts:
                topic_name = extract_topic_name(topic_text)
                body_text = topic_text
                topic_chunks = _split_by_paragraphs(body_text, max_words=400)

                for subchunk in topic_chunks:
                    chunk_id += 1
                    saved_path = save_rag_chunk(
                        module,
                        session_number,
                        topic_name,
                        chunk_id,
                        subchunk,
                        base_dir,
                    )
                    saved_paths.append(saved_path)

        logging.info("Created %s chunks for %s", len(saved_paths), module)
        logging.info("Saved to %s/%s", rag_root, module)


def main() -> None:
    setup_logging()
    base_dir = Path(__file__).resolve().parent
    paths = ensure_directories(base_dir)
    bootstrap_module_session_cache(paths["output"], paths["session_output"])

    logging.info("Loading transcript files from %s", paths["input"])
    session_files = load_files(paths["input"])
    if not session_files:
        legacy_files = sorted(paths["input"].glob("*.txt"))
        if legacy_files:
            logging.warning(
                "Found flat transcript files in %s, but the pipeline now expects module folders like input/cms/session_1.txt.",
                paths["input"],
            )
        logging.warning(
            "No valid transcript files found in %s. Add files in module folders like input/cms/session_1.txt and rerun.",
            paths["input"],
        )
        return

    cleaned_by_module: dict[str, dict[int, str]] = defaultdict(dict)

    for session in session_files:
        module_name = str(session["module_name"])
        session_number = int(session["session_number"])
        transcript_text = str(session["text"])

        existing_output = load_cached_session_output(
            paths["session_output"],
            module_name,
            session_number,
            transcript_text,
            session["path"],
        )
        if existing_output is not None:
            logging.info(
                "Skipping %s/session_%s because the transcript source hash matches cached output.",
                module_name,
                session_number,
            )
            cleaned_by_module[module_name][session_number] = existing_output
            continue

        logging.info("Processing %s/session_%s", module_name, session_number)
        chunks = split_text(transcript_text, chunk_size=1200)
        cleaned_chunks: list[str] = []

        for chunk_index, chunk_text in enumerate(
            tqdm(
                chunks,
                desc=f"{module_name}/session_{session_number}",
                unit="chunk",
            ),
            start=1,
        ):
            save_chunk(
                paths["chunks"],
                module_name,
                session_number,
                chunk_index,
                chunk_text,
            )

            try:
                cleaned_text = clean_chunk(chunk_text)
            except Exception as exc:
                logging.exception(
                    "Failed to clean %s/session_%s chunk %s: %s",
                    module_name,
                    session_number,
                    chunk_index,
                    exc,
                )
                raise

            cleaned_chunks.append(cleaned_text)

        session_output = merge_session_chunks(cleaned_chunks)
        save_session_output(
            paths["session_output"],
            module_name,
            session_number,
            session_output,
            source_text=transcript_text,
        )
        cleaned_by_module[module_name][session_number] = session_output

    for module_name, cleaned_sessions in sorted(cleaned_by_module.items()):
        module_output = merge_module_output(cleaned_sessions)
        save_module_output(paths["output"], module_name, module_output)

    process_rag_chunking(base_dir)
    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
