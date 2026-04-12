from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from utils import (
    clean_chunk,
    ensure_directories,
    load_files,
    merge_output,
    save_chunk,
    save_final_output,
    setup_logging,
    split_text,
)


def main() -> None:
    setup_logging()
    base_dir = Path(__file__).resolve().parent
    paths = ensure_directories(base_dir)

    logging.info("Loading transcript files from %s", paths["input"])
    session_files = load_files(paths["input"])
    if not session_files:
        logging.warning(
            "No valid transcript files found in %s. Add files like session_1.txt and rerun.",
            paths["input"],
        )
        return

    cleaned_by_session: dict[int, list[str]] = {}

    for session in session_files:
        session_number = int(session["session_number"])
        transcript_text = str(session["text"])

        logging.info("Processing session %s", session_number)
        chunks = split_text(transcript_text, chunk_size=1200)
        cleaned_by_session[session_number] = []

        for chunk_index, chunk_text in enumerate(
            tqdm(
                chunks,
                desc=f"Session {session_number}",
                unit="chunk",
            ),
            start=1,
        ):
            save_chunk(paths["chunks"], session_number, chunk_index, chunk_text)

            try:
                cleaned_text = clean_chunk(chunk_text)
            except Exception as exc:
                logging.exception(
                    "Failed to clean session %s chunk %s: %s",
                    session_number,
                    chunk_index,
                    exc,
                )
                raise

            cleaned_by_session[session_number].append(cleaned_text)

    merged_output = merge_output(cleaned_by_session)
    save_final_output(paths["output"], merged_output)
    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
