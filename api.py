from dotenv import load_dotenv
import os

load_dotenv()
print("KEY:", os.getenv("OPENAI_API_KEY"))  # 👈 ADD THIS HERE

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

app = FastAPI()

# Init clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("iitm-modules-rag")  # your index name


# Request schema
class ChatRequest(BaseModel):
    question: str
    module: str
    session: int


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Embed user query
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.question
    ).data[0].embedding

    # 2. Query Pinecone with filter
    results = index.query(
        vector=embedding,
        top_k=8,
        include_metadata=True,
        filter={
            "module": req.module,
            "session": req.session
        }
    )

    # 3. Extract context
    context = "\n\n".join([
        match["metadata"]["text"]
        for match in results["matches"]
    ])

    # 4. Generate answer
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are an AI assistant answering questions using ONLY the provided context.

STRICT RULES:
- If context is EMPTY → say: "Not in module."
- If context is NOT EMPTY → you MUST answer (never say "Not in module")

INSTRUCTIONS:
- Combine information from multiple context chunks
- Do NOT copy blindly — summarize and explain
- Be clear and structured

FORMAT:

Answer:
<clear explanation>

Key Points:
- point 1
- point 2
- point 3

Student Doubts:
- if present in context

IMPORTANT:
- Do not hallucinate
- Do not ignore context
- If ANY relevant info exists → produce an answer
"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{req.question}"
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "score": match["score"],
                "text": match["metadata"].get("text", "")
            }
            for match in results.get("matches", [])
        ]
    }