from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Sequence

from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME)


def _document_text(document: dict[str, Any]) -> str:
    metadata = document.get("metadata", {})
    return str(metadata.get("text", "")).strip()


def rerank(
    query: str,
    documents: Sequence[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rerank retrieved documents using a cross-encoder."""
    if not documents:
        print("Rerank scores: []")
        logging.info("Rerank skipped because no documents were provided")
        return []

    valid_documents = [document for document in documents if _document_text(document)]
    if not valid_documents:
        print("Rerank scores: []")
        logging.info("Rerank skipped because all retrieved documents were empty")
        return []

    pairs = [(query, _document_text(document)) for document in valid_documents]
    scores = get_reranker_model().predict(pairs)
    ranked = sorted(zip(valid_documents, scores), key=lambda item: item[1], reverse=True)

    rerank_scores: list[dict[str, float | str | None]] = []
    reranked_documents: list[dict[str, Any]] = []

    for document, score in ranked:
        rerank_scores.append({"id": document.get("id"), "score": float(score)})

    print(f"Rerank scores: {rerank_scores}")
    logging.info("Rerank scores: %s", rerank_scores)

    for document, score in ranked[:top_k]:
        reranked_document = dict(document)
        reranked_document["rerank_score"] = float(score)
        reranked_documents.append(reranked_document)

    return reranked_documents
