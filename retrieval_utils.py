from __future__ import annotations

import logging
import re
from typing import Any


TOPIC_FILTER_HINTS: dict[str, dict[str, Any]] = {
    'multi-query': {'module': 'map', 'topic_header': 'Multi-Query Expansion'},
    'multi query': {'module': 'map', 'topic_header': 'Multi-Query Expansion'},
    'mqe': {'module': 'map', 'topic_header': 'Multi-Query Expansion'},
    'cross encoder': {'module': 'map', 'topic_header': 'Explanation of Cross Encoder in Retrieval and Reranking'},
    'rerank': {'module': 'map'},
    'reranking': {'module': 'map'},
    'langflow': {'module': 'wdp'},
    'langgraph': {'module': 'wdp'},
    'pinecone': {'module': 'cms'},
    'history maintenance': {'module': 'cms', 'concept_name': 'Building a Chatbot with History Maintenance'},
    'chat history': {'module': 'cms'},
    'tool calling': {'module': 'wdp'},
    'multi-agent': {'module': 'wdp'},
    'multi agent': {'module': 'wdp'},
}
INSTRUCTOR_HINTS: dict[str, str] = {
    'samil bub': 'Samil Bub',
}
CONTENT_TYPE_HINTS: dict[str, dict[str, Any]] = {
    'student doubts': {'content_type': {'$in': ['student_doubts', 'mixed']}},
    'student doubt': {'content_type': {'$in': ['student_doubts', 'mixed']}},
    'doubts asked': {'content_type': {'$in': ['student_doubts', 'mixed']}},
    'questions asked': {'content_type': {'$in': ['student_doubts', 'mixed']}},
    'key points': {'content_type': {'$in': ['key_points', 'mixed']}},
    'main points': {'content_type': {'$in': ['key_points', 'mixed']}},
    'summary points': {'content_type': {'$in': ['key_points', 'mixed']}},
    'explain': {'content_type': {'$in': ['explanation', 'mixed']}},
    'explanation': {'content_type': {'$in': ['explanation', 'mixed']}},
}
LEVEL_HINTS: dict[str, str] = {
    'foundations': 'beginner',
    'foundation': 'beginner',
    'beginner': 'beginner',
    'intermediate': 'intermediate',
    'advanced': 'advanced',
}

NAMED_MODULE_PATTERN = re.compile(r'\bmodule\s+(cms|map|wdp)\b', re.IGNORECASE)
NUMERIC_MODULE_PATTERN = re.compile(r'\bmodule\s+(\d+)\b', re.IGNORECASE)
WEEK_PATTERN = re.compile(r'\bweek\s+(\d+)\b', re.IGNORECASE)
SESSION_PATTERN = re.compile(r'\bsession\s+(\d+)\b', re.IGNORECASE)


def detect_filters(query: str) -> dict[str, Any] | None:
    """Infer lightweight Pinecone metadata filters directly from the query."""
    normalized_query = query.strip().lower()
    if not normalized_query:
        logging.info('Detected filters: none')
        return None

    filters: dict[str, Any] = {}

    for level_keyword, level_name in LEVEL_HINTS.items():
        if re.search(rf'\b{re.escape(level_keyword)}\b', normalized_query):
            filters['level'] = level_name
            break

    named_module_match = NAMED_MODULE_PATTERN.search(normalized_query)
    numeric_module_match = NUMERIC_MODULE_PATTERN.search(normalized_query)
    week_match = WEEK_PATTERN.search(normalized_query)

    if named_module_match:
        filters['module'] = named_module_match.group(1).lower()
    elif numeric_module_match:
        filters['module'] = numeric_module_match.group(1)
    elif week_match:
        filters['module'] = week_match.group(1)
    elif re.search(r'\bcms\b', normalized_query):
        filters['module'] = 'cms'
    elif re.search(r'\bmap\b', normalized_query):
        filters['module'] = 'map'
    elif re.search(r'\bwdp\b', normalized_query):
        filters['module'] = 'wdp'

    session_match = SESSION_PATTERN.search(normalized_query)
    if session_match:
        filters['session'] = int(session_match.group(1))

    for instructor_keyword, instructor_name in INSTRUCTOR_HINTS.items():
        if instructor_keyword in normalized_query:
            filters.setdefault('instructor', instructor_name)
            break

    for phrase, content_filters in CONTENT_TYPE_HINTS.items():
        if phrase in normalized_query:
            for key, value in content_filters.items():
                filters.setdefault(key, value)
            break

    for keyword, keyword_filters in TOPIC_FILTER_HINTS.items():
        if keyword in normalized_query:
            for key, value in keyword_filters.items():
                filters.setdefault(key, value)
            break

    detected_filters = filters or None
    logging.info('Detected filters: %s', detected_filters)
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
                    'Ignoring detected filter %s=%r because explicit filter uses %r',
                    key,
                    existing_value,
                    value,
                )
            merged_filters[key] = value

    return merged_filters or None
