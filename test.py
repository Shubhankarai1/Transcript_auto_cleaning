from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("iitm-modules-rag")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

query = "context"
embedding = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
).data[0].embedding

res = index.query(
    vector=embedding,
    top_k=3,
    include_metadata=True
)

# 🔥 PRINT ACTUAL METADATA
for match in res["matches"]:
    print("\n---")
    print(match["metadata"])