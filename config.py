from __future__ import annotations

import os

from dotenv import load_dotenv


def load_environment() -> None:
    # Local overrides should win during development.
    load_dotenv('.env', override=False)
    load_dotenv('.env.local', override=True)


load_environment()


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()
