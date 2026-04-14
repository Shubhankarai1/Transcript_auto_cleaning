from __future__ import annotations

import logging
import re
from pathlib import Path


SOURCE_PREFIX = "crms"
TARGET_MODULE = "cms"
INPUT_FILE_PATTERN = re.compile(rf"^{SOURCE_PREFIX}_session_(\d+)\.txt$", re.IGNORECASE)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def migrate_input(input_dir: Path) -> None:
    module_dir = input_dir / TARGET_MODULE
    module_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(input_dir.glob("*.txt"), key=lambda item: item.name.lower()):
        match = INPUT_FILE_PATTERN.match(path.name)
        if not match:
            logging.info("Ignoring non-matching input file: %s", path.name)
            continue

        session_number = match.group(1)
        target_path = module_dir / f"session_{session_number}.txt"

        if target_path.exists():
            logging.warning(
                "Skipping move because target already exists: %s",
                target_path,
            )
            continue

        path.rename(target_path)
        logging.info("Moved %s -> %s", path.name, target_path)


def migrate_output(output_dir: Path) -> None:
    legacy_path = output_dir / "final_cleaned.txt"
    target_path = output_dir / f"{TARGET_MODULE}_final_cleaned.txt"

    if target_path.exists():
        if legacy_path.exists():
            logging.warning(
                "Keeping existing module output and leaving legacy file untouched to avoid overwrite: %s",
                legacy_path,
            )
        return

    if not legacy_path.exists():
        logging.info("No legacy final output found at %s", legacy_path)
        return

    legacy_path.rename(target_path)
    logging.info("Renamed %s -> %s", legacy_path.name, target_path.name)


def main() -> None:
    setup_logging()
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    migrate_input(input_dir)
    migrate_output(output_dir)
    logging.info("Migration completed.")


if __name__ == "__main__":
    main()
