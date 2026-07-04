from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(
    "localhost",
    port=6333
)

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

query = "How do I manage containers?"

query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name="knowledge",
    query=query_vector,
    limit=3,
).points

for hit in results:
    print()
    print("Score:", hit.score)
    print("Text :", hit.payload["text"])
