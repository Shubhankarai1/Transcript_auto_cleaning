from pinecone import Pinecone
from openai import OpenAI

pc = Pinecone(api_key="pcsk_6Xi9vL_2DcEaqQn3LMtiDTq9hKv9fSiaptadwQN9hDrpAPmZpp9uYTY9YCUJ4xbP1WFULy")
index = pc.Index("iitm-modules-rag")

client = OpenAI(api_key="OPENAI_API_KEY")

# create REAL query embedding
query = "what is context in AI"
embedding = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
).data[0].embedding

# now search
res = index.query(
    vector=embedding,
    top_k=3,
    include_metadata=True
)

print(res)