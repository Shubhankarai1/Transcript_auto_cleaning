import os
import logging
import re
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from reranker import rerank
from retrieval_utils import combine_filters, detect_filters

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5
RETRIEVAL_TOP_K = 25

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not set in .env")

index = pinecone_client.Index(index_name)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
CHUNK_ID_PATTERN = re.compile(r"_chunk_(\d+)", re.IGNORECASE)


def expand_query(user_query: str) -> List[str]:
    """Generate 3 alternative versions of the query using OpenAI."""
    prompt = f"""Generate 3 alternative phrasings of this query that capture the same intent but use different wording:

Original query: "{user_query}"

Return ONLY the 3 alternatives, one per line, without numbering or additional text."""

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        expanded = response.choices[0].message.content.strip().split("\n")
        expanded = [q.strip() for q in expanded if q.strip()]
        return expanded[:3]
    except Exception as e:
        logging.error(f"Failed to expand query: {e}")
        return []


def create_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    try:
        response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to create embedding: {e}")
        raise


def retrieve_from_pinecone(queries: List[str], metadata_filter: Dict | None = None) -> List[Dict]:
    """Query Pinecone with multiple queries and collect results."""
    all_matches = []
    for query in queries:
        try:
            embedding = create_embedding(query)
            query_kwargs = {
                "vector": embedding,
                "top_k": RETRIEVAL_TOP_K,
                "include_metadata": True,
            }
            if metadata_filter is not None:
                query_kwargs["filter"] = metadata_filter

            results = index.query(**query_kwargs)
            for match in results.get("matches", []):
                all_matches.append(
                    {
                        "id": match["id"],
                        "score": match["score"],
                        "metadata": match.get("metadata", {}),
                    }
                )
        except Exception as e:
            logging.error(f"Failed to query Pinecone: {e}")
    return all_matches


def retrieve_context(query: str, expanded_queries: List[str]) -> List[Dict]:
    detected_filters = detect_filters(query)
    final_filter = combine_filters(None, detected_filters)

    print(f"Applied filters: {final_filter or 'none'}")
    logging.info("Applied filters: %s", final_filter)

    matches = retrieve_from_pinecone(expanded_queries, metadata_filter=final_filter)
    logging.info(f"Retrieved {len(matches)} total matches before deduplication")

    unique_matches = deduplicate_chunks(matches)
    print(f"Unique chunks retrieved: {len(unique_matches)}")
    logging.info(f"Retained {len(unique_matches)} unique matches before reranking")

    candidate_matches = sorted(
        unique_matches,
        key=lambda match: match.get("score", 0),
        reverse=True,
    )[:RETRIEVAL_TOP_K]

    return rerank(query, candidate_matches, top_k=TOP_K)


def deduplicate_chunks(matches: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks based on text content."""
    seen_texts = set()
    unique_matches = []
    for match in matches:
        text = match.get("metadata", {}).get("text", "")
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_matches.append(match)
    return unique_matches


def extract_chunk_number(match: Dict) -> int | None:
    metadata = match.get("metadata", {})
    chunk_value = metadata.get("chunk")
    if chunk_value is not None:
        try:
            return int(chunk_value)
        except (TypeError, ValueError):
            pass

    match_id = str(match.get("id", ""))
    chunk_match = CHUNK_ID_PATTERN.search(match_id)
    if chunk_match:
        return int(chunk_match.group(1))

    return None


def build_source_entry(match: Dict, index_position: int) -> Dict:
    metadata = match.get("metadata", {})
    module = metadata.get("module", "unknown")
    session = metadata.get("session", "unknown")
    chunk_num = extract_chunk_number(match)

    return {
        "id": match.get("id"),
        "score": match.get("score"),
        "module": module,
        "session": session,
        "chunk": chunk_num,
        "text": metadata.get("text", ""),
        "citation": f"{str(module).upper()}-S{session}-C{chunk_num if chunk_num is not None else index_position}",
    }


def build_context(sources: List[Dict]) -> str:
    """Join top chunks into a single context string with citation labels."""
    context_parts = []
    for source in sources:
        text = source.get("text", "")
        if not text:
            continue
        context_parts.append(f"[{source['citation']}]\n{text}")

    return "\n\n".join(context_parts)


def answer_with_context(user_query: str, context: str) -> str:
    """Call OpenAI LLM with context to answer the query."""
    system_prompt = "You are an expert teacher."

    user_message = f"""Use the retrieved context to answer the question in a detailed and structured way.

Rules:
- Explain concepts step-by-step
- Use examples where possible
- Expand ideas clearly (not just summary)
- Minimum 150-300 words
- If multiple chunks are retrieved, combine them into a single explanation
- Cite factual statements inline using the source labels from the context, for example [CMS-S3-C84]
- Include citations throughout the answer, not only at the end
- End with a short line starting with "Sources used:" and list the unique citations you relied on
- If the context does not contain the answer, say that clearly

Context:
{context}

Question:
{user_query}"""

    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Failed to get answer: {e}")
        return "Error generating answer"


def main():
    print("=" * 60)
    user_query = input("Enter your question: ")
    user_query = user_query.strip()
    
    if not user_query:
        print("Please enter a valid question.")
        return
    
    print(f"\n{'='*60}")
    print(f"Original Query: {user_query}")
    print(f"{'='*60}\n")

    # Expand query
    expanded_queries = expand_query(user_query)
    print(f"Expanded Queries:")
    for i, q in enumerate(expanded_queries, 1):
        print(f"  {i}. {q}")
    print()

    # Build all queries to search
    all_queries = [user_query] + expanded_queries
    logging.info(f"Searching with {len(all_queries)} queries")

    # Retrieve, pre-filter, and rerank
    reranked_matches = retrieve_context(user_query, all_queries)
    logging.info(f"Retrieved {len(reranked_matches)} reranked matches")

    if not reranked_matches:
        print("No results found in Pinecone")
        return

    print()

    # Build context
    source_entries = [
        build_source_entry(match, index_position)
        for index_position, match in enumerate(reranked_matches, start=1)
    ]
    context = build_context(source_entries)
    if not context:
        print("No valid context found")
        return

    # Get answer from LLM
    answer = answer_with_context(user_query, context)

    print(f"{'='*60}")
    print("FINAL ANSWER:")
    print(f"{'='*60}")
    print(answer)
    print("\nRetrieved Sources:")
    for source in source_entries:
        print(
            f"- {source['citation']} | module={source['module']} | "
            f"session={source['session']} | chunk={source['chunk']}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
