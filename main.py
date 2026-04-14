from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from utils import (
    clean_chunk,
    ensure_directories,
    load_files,
    load_session_output,
    merge_module_output,
    merge_session_chunks,
    save_chunk,
    save_module_output,
    save_session_output,
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

        existing_output = load_session_output(
            paths["session_output"],
            module_name,
            session_number,
        )
        if existing_output is not None:
            logging.info(
                "Skipping %s/session_%s because an existing cleaned output was found.",
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
        )
        cleaned_by_module[module_name][session_number] = session_output

    for module_name, cleaned_sessions in sorted(cleaned_by_module.items()):
        module_output = merge_module_output(cleaned_sessions)
        save_module_output(paths["output"], module_name, module_output)

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
