from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.mentor_service import (
    build_context,
    build_search_queries,
    build_source_entry,
    generate_answer,
    retrieve_context_with_fallback,
    rewrite_query,
    trim_sources_for_ui,
)


_log = logging.getLogger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    mode: str = 'global'
    level: Optional[str] = None
    module: Optional[str] = None
    session: Optional[int] = None
    chat_history: list[ChatMessage] = Field(default_factory=list)


@router.post('/chat')
def chat(req: ChatRequest):
    question = (req.question or req.query or '').strip()
    if not question:
        raise HTTPException(status_code=400, detail='Empty question')

    try:
        rewritten = rewrite_query(question)
        search_queries, hyde_query = build_search_queries(question, rewritten)

        docs, retrieval_scope, applied_filters = retrieve_context_with_fallback(
            req.mode, req.level, req.module, req.session,
            question, search_queries,
        )

        if not docs:
            return {'answer': 'Not in module.', 'sources': []}

        sources = [build_source_entry(doc, i) for i, doc in enumerate(docs, 1)]
        context = build_context(sources)
        answer = generate_answer(question, context)
        ui_sources = trim_sources_for_ui(sources)

        return {
            'answer': answer,
            'sources': ui_sources,
            'retrieval_queries': search_queries,
            'hyde_query': hyde_query,
            'retrieval_scope': retrieval_scope,
            'applied_filters': applied_filters,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
