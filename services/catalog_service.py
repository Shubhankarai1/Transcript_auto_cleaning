from __future__ import annotations

from pathlib import Path
from typing import Any

from utils import load_files


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'input'
LEVEL_ORDER = {'beginner': 1, 'intermediate': 2, 'advanced': 3}


def get_content_catalog() -> dict[str, dict[str, list[int]]]:
    catalog: dict[str, dict[str, list[int]]] = {}
    if not INPUT_DIR.exists():
        return catalog

    for item in load_files(INPUT_DIR):
        level = str(item.get('level') or 'advanced')
        module = str(item.get('module_name') or '')
        session = int(item['session_number'])
        if not module:
            continue
        level_modules = catalog.setdefault(level, {})
        module_sessions = level_modules.setdefault(module, [])
        if session not in module_sessions:
            module_sessions.append(session)

    for modules in catalog.values():
        for sessions in modules.values():
            sessions.sort()

    return catalog
