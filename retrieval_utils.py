from __future__ import annotations

import logging
import re
from typing import Any


TOPIC_FILTER_HINTS: dict[str, dict[str, Any]] = {
    "multi-query": {"module": "map", "topic": "Multi-Query Expansion"},
    "multi query": {"module": "map", "topic": "Multi-Query Expansion"},
    "mqe": {"module": "map", "topic": "Multi-Query Expansion"},
    "cross encoder": {"module": "map"},
    "rerank": {"module": "map"},
    "reranking": {"module": "map"},
    "langflow": {"module": "wdp"},
    "langgraph": {"module": "wdp"},
    "pinecone": {"module": "cms"},
}

NAMED_MODULE_PATTERN = re.compile(r"\bmodule\s+(cms|map|wdp)\b", re.IGNORECASE)
NUMERIC_MODULE_PATTERN = re.compile(r"\bmodule\s+(\d+)\b", re.IGNORECASE)
WEEK_PATTERN = re.compile(r"\bweek\s+(\d+)\b", re.IGNORECASE)
SESSION_PATTERN = re.compile(r"\bsession\s+(\d+)\b", re.IGNORECASE)


def detect_filters(query: str) -> dict[str, Any] | None:
    """Infer lightweight Pinecone metadata filters directly from the query."""
    normalized_query = query.strip().lower()
    if not normalized_query:
        print("Detected filters: none")
        logging.info("Detected filters: none")
        return None

    filters: dict[str, Any] = {}

    named_module_match = NAMED_MODULE_PATTERN.search(normalized_query)
    numeric_module_match = NUMERIC_MODULE_PATTERN.search(normalized_query)
    week_match = WEEK_PATTERN.search(normalized_query)

    if named_module_match:
        filters["module"] = named_module_match.group(1).lower()
    elif numeric_module_match:
        filters["module"] = numeric_module_match.group(1)
    elif week_match:
        filters["module"] = week_match.group(1)
    elif re.search(r"\bcms\b", normalized_query):
        filters["module"] = "cms"
    elif re.search(r"\bwdp\b", normalized_query):
        filters["module"] = "wdp"

    session_match = SESSION_PATTERN.search(normalized_query)
    if session_match:
        filters["session"] = int(session_match.group(1))

    for keyword, keyword_filters in TOPIC_FILTER_HINTS.items():
        if keyword in normalized_query:
            for key, value in keyword_filters.items():
                filters.setdefault(key, value)
            break

    detected_filters = filters or None
    print(f"Detected filters: {detected_filters or 'none'}")
    logging.info("Detected filters: %s", detected_filters)
    return detected_filters


def combine_filters(
    base_filter: dict[str, Any] | None,
    detected_filter: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge explicit filters with query-detected filters, preferring explicit values."""
    if base_filter is None and detected_filter is None:
        return None

    merged_filters: dict[str, Any] = {}
    if detected_filter:
        merged_filters.update(detected_filter)

    if base_filter:
        for key, value in base_filter.items():
            existing_value = merged_filters.get(key)
            if existing_value is not None and existing_value != value:
                logging.info(
                    "Ignoring detected filter %s=%r because explicit filter uses %r",
                    key,
                    existing_value,
                    value,
                )
            merged_filters[key] = value

    return merged_filters or None
