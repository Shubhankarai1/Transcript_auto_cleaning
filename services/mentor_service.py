from __future__ import annotations

import logging
import re
from typing import Any

from openai import OpenAI
from pinecone import Pinecone

from config import get_env
from retrieval_utils import combine_filters, detect_filters


openai_client = OpenAI(api_key=get_env('OPENAI_API_KEY'))
pc = Pinecone(api_key=get_env('PINECONE_API_KEY'))
index = pc.Index(get_env('PINECONE_INDEX_NAME', 'iitm-modules-rag'))

CHUNK_ID_PATTERN = re.compile(r'_chunk_(\d+)', re.IGNORECASE)
MAX_HISTORY_MESSAGES = 10
RETRIEVAL_TOP_K = 25
FINAL_TOP_K = 5
LLM_MODEL = 'gpt-4o-mini'
EMBEDDING_MODEL = 'text-embedding-3-small'
QUERY_EXPANSION_COUNT = 3
HYDE_MAX_TOKENS = 180
LEVEL_ORDER = {'beginner': 1, 'intermediate': 2, 'advanced': 3}

_log = logging.getLogger(__name__)


def build_fallback_filters(
    mode: str,
    level: str | None,
    module: str | None,
    session: int | None,
) -> list[tuple[str, dict[str, Any] | None]]:
    if mode == 'global':
        return [('global', None)]

    if not level or not module:
        return [('global', None)]

    fallback: list[tuple[str, dict[str, Any] | None]] = [
        ('level_module', {'level': level, 'module': module}),
        ('level_only', {'level': level}),
        ('module_only', {'module': module}),
        ('global', None),
    ]
    if session is not None:
        fallback.insert(0, ('level_module_session', {'level': level, 'module': module, 'session': session}))
    return fallback


def extract_chunk_number(match: dict[str, Any]) -> int | None:
    metadata = match.get('metadata', {})
    if metadata.get('chunk'):
        return int(metadata['chunk'])
    match_id = str(match.get('id', ''))
    m = CHUNK_ID_PATTERN.search(match_id)
    return int(m.group(1)) if m else None


def build_source_entry(match: dict[str, Any], index_position: int):
    metadata = match.get('metadata', {})
    level = metadata.get('level')
    module = metadata.get('module')
    session = metadata.get('session')
    chunk = extract_chunk_number(match)
    citation = f"{str(module).upper()}-S{session}-C{chunk or index_position}"
    return {
        'id': match.get('id'),
        'score': match.get('score'),
        'text': metadata.get('text', ''),
        'level': level,
        'module': module,
        'session': session,
        'chunk': chunk,
        'citation': citation,
        'matched_query': match.get('matched_query'),
    }


def build_context(sources: list[dict[str, Any]]) -> str:
    return '\n\n'.join(
        f"[{source['citation']}]\n{source['text']}" for source in sources if source.get('text')
    )


def trim_sources_for_ui(sources: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for source in sources[:limit]:
        trimmed.append({
            'citation': source.get('citation'),
            'level': source.get('level'),
            'module': source.get('module'),
            'session': source.get('session'),
            'chunk': source.get('chunk'),
            'text': source.get('text', ''),
        })
    return trimmed


def normalize_chat_history(chat_history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {'role': m['role'], 'content': m['content']}
        for m in chat_history
        if m.get('role') in {'user', 'assistant'} and m.get('content')
    ][-MAX_HISTORY_MESSAGES:]


def parse_query_lines(text: str) -> list[str]:
    queries: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r'^\s*(?:[-*]|\d+[\).\:-])\s*', '', line).strip()
        cleaned = cleaned.strip('"\'')
        if cleaned:
            queries.append(cleaned)
    return queries


def unique_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        normalized = re.sub(r'\s+', ' ', query.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(query.strip())
    return unique


def rewrite_query(question: str) -> str:
    rewrite = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': question}],
    )
    return (rewrite.choices[0].message.content or question).strip()


def expand_query(question: str, rewritten: str) -> list[str]:
    prompt = f"""
Generate {QUERY_EXPANSION_COUNT} alternate search queries for retrieving relevant course transcript chunks.

Preserve the user's intent. Use different wording and related course terms where useful.
Return only the alternate queries, one per line. Do not number them.

Original question:
{question}

Current rewritten query:
{rewritten}
""".strip()

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.3,
        )
    except Exception:
        _log.exception('Query expansion failed; falling back to base queries')
        return []

    expanded = parse_query_lines(response.choices[0].message.content or '')
    return expanded[:QUERY_EXPANSION_COUNT]


def generate_hyde_query(question: str, rewritten: str) -> str:
    prompt = f"""
Write a concise hypothetical course-transcript passage that would directly answer the student's question.

Use likely terminology from AI agent workflows, RAG, retrieval, planning, LangChain/LangGraph, Pinecone, reranking, or multi-agent systems only when relevant.
Do not mention that this is hypothetical. Do not add citations. Keep it under 120 words.

Student question:
{question}

Rewritten query:
{rewritten}
""".strip()

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.2,
            max_tokens=HYDE_MAX_TOKENS,
        )
    except Exception:
        _log.exception('HyDE generation failed; continuing without HyDE')
        return ''

    return (response.choices[0].message.content or '').strip()


def build_search_queries(question: str, rewritten: str) -> tuple[list[str], str]:
    hyde_query = generate_hyde_query(question, rewritten)
    expanded = expand_query(question, rewritten)
    queries = unique_queries([question, rewritten, hyde_query, *expanded])
    _log.info('Using %d retrieval queries: %s', len(queries), queries)
    return queries, hyde_query


def create_embeddings(queries: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=queries,
    )
    return [item.embedding for item in response.data]


def deduplicate_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[str, dict[str, Any]] = {}
    for match in matches:
        metadata = match.get('metadata', {})
        key = str(match.get('id') or metadata.get('text') or '')
        if not key:
            continue
        existing = best_by_key.get(key)
        if existing is None or match.get('score', 0) > existing.get('score', 0):
            best_by_key[key] = match
    return list(best_by_key.values())


def _metadata_matches_filter_value(metadata_value: Any, filter_value: Any) -> bool:
    if isinstance(filter_value, dict):
        allowed_values = filter_value.get('$in')
        if isinstance(allowed_values, list):
            return metadata_value in allowed_values
        return False
    return metadata_value == filter_value


def _match_strength(match: dict[str, Any], filters: dict[str, Any] | None) -> int:
    if not filters:
        return 0
    metadata = match.get('metadata', {})
    strength = 0
    for key, filter_value in filters.items():
        if _metadata_matches_filter_value(metadata.get(key), filter_value):
            strength += 1
    return strength


def post_filter_matches(matches: list[dict[str, Any]], filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not matches or not filters:
        return matches

    ranked_with_strength = [(match, _match_strength(match, filters)) for match in matches]
    max_strength = max((strength for _, strength in ranked_with_strength), default=0)
    if max_strength <= 0:
        return matches

    strongest = [match for match, strength in ranked_with_strength if strength == max_strength]
    if len(strongest) >= FINAL_TOP_K:
        _log.info('Post-filtering retained %d strongest matches at metadata strength %d', len(strongest), max_strength)
        return strongest

    near_strongest = [match for match, strength in ranked_with_strength if strength >= max_strength - 1]
    if len(near_strongest) >= FINAL_TOP_K:
        _log.info('Post-filtering retained %d near-strongest matches at metadata strength >= %d', len(near_strongest), max_strength - 1)
        return near_strongest

    _log.info('Post-filtering left too few matches after metadata pruning; keeping broader pool')
    return matches


def retrieve_context_for_filter(
    original_query: str,
    search_queries: list[str],
    base_filter: dict[str, Any] | None,
):
    filters = combine_filters(base_filter, detect_filters(original_query))
    embeddings = create_embeddings(search_queries)

    pooled_matches: list[dict[str, Any]] = []
    for query, embedding in zip(search_queries, embeddings):
        results = index.query(
            vector=embedding,
            top_k=RETRIEVAL_TOP_K,
            include_metadata=True,
            filter=filters if filters else None,
        )
        for match in results.get('matches', []):
            pooled_matches.append({
                'id': match.get('id'),
                'score': match.get('score'),
                'metadata': match.get('metadata', {}),
                'matched_query': query,
            })

    matches = sorted(
        deduplicate_matches(pooled_matches),
        key=lambda x: x['score'],
        reverse=True,
    )
    post_filtered = sorted(
        post_filter_matches(matches, filters),
        key=lambda x: x['score'],
        reverse=True,
    )
    _log.info(
        'Retrieved %d pooled matches, retained %d unique, kept %d after post-filtering',
        len(pooled_matches), len(matches), len(post_filtered),
    )
    return post_filtered[:FINAL_TOP_K], filters


def has_strong_enough_results(matches: list[dict[str, Any]]) -> bool:
    if not matches:
        return False
    if len(matches) >= 3:
        return True
    top_score = float(matches[0].get('score') or 0.0)
    return top_score >= 0.45


def retrieve_context_with_fallback(
    mode: str,
    level: str | None,
    module: str | None,
    session: int | None,
    original_query: str,
    search_queries: list[str],
):
    fallback_filters = build_fallback_filters(mode, level, module, session)
    last_matches: list[dict[str, Any]] = []
    last_filters: dict[str, Any] | None = None
    last_scope = fallback_filters[-1][0]

    for scope_name, base_filter in fallback_filters:
        matches, applied_filters = retrieve_context_for_filter(original_query, search_queries, base_filter)
        _log.info('Retrieval scope=%s applied_filters=%s returned %d matches', scope_name, applied_filters, len(matches))
        last_matches = matches
        last_filters = applied_filters
        last_scope = scope_name
        if has_strong_enough_results(matches):
            return matches, scope_name, applied_filters

    return last_matches, last_scope, last_filters


def build_answer_prompt(context: str, question: str) -> str:
    return f"""
You are an expert instructor explaining concepts from a course.

Your goal is to generate a HIGH-QUALITY, DETAILED, TEACHING-STYLE answer using the provided context.

STRICT INSTRUCTIONS:

1. Always prioritize DEPTH and CLARITY over brevity.
2. Do NOT limit answers to 7-10 points.
3. Explain concepts as if teaching a student who is seeing this for the first time.
4. Use a natural explanation flow:
   - Start with a simple overview
   - Then break down the concept step-by-step
   - Then explain WHY it matters
   - Then, if relevant, give an example or analogy

5. Use structured formatting ONLY where helpful:
   - Headings
   - Subsections
   - Bullet points, but do not force them

6. If context is available:
   - Use it fully
   - Combine ideas across chunks
   - Do NOT say 'not in module' unless absolutely no relevant info exists

7. If partial context is available:
   - Answer using available context
   - Then intelligently fill gaps using reasoning
   - NEVER abruptly stop

8. Avoid generic short answers.
9. Avoid robotic formatting.
10. Write like a human instructor, not a checklist generator.
11. When you use a factual point from the context, cite it inline using the context labels, for example [MAP-S3-C42].
12. Use only the most relevant citations. Do not cite every sentence and do not invent citations.
13. Do not add a separate 'Sources', 'Courses', or 'Debug' section inside the answer.

---

CONTEXT:
{context}

---

QUESTION:
{question}

---

Now generate a COMPLETE and DETAILED answer.
""".strip()


def generate_answer(question: str, context: str) -> str:
    prompt = build_answer_prompt(context, question)
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.choices[0].message.content
