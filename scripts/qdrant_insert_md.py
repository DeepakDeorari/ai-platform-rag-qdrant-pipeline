from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)

collection_name = "knowledge"

try:
    client.delete_collection(collection_name)
except:
    pass

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
    ),
)

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

documents = [
    "Kubernetes is a container orchestration platform",
    "Docker is a container runtime",
    "Grafana is used for observability dashboards",
    "Prometheus collects metrics",
    "Ollama runs local large language models",
]

points = []

for idx, doc in enumerate(documents):
    vector = model.encode(doc).tolist()

    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "text": doc
            }
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points,
)

print("Inserted", len(points), "documents")
