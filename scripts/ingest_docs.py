from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
)
from sentence_transformers import SentenceTransformer


# -------------------------
# Configuration
# -------------------------

COLLECTION_NAME = "knowledge"
DOCS_DIR = "docs"
CHUNK_SIZE = 100  # words


# -------------------------
# Chunking Function
# -------------------------

def chunk_text(text, chunk_size=100):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# -------------------------
# Qdrant
# -------------------------

client = QdrantClient(
    host="localhost",
    port=6333,
)

# -------------------------
# Embedding Model
# -------------------------

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

embedding_size = model.get_sentence_embedding_dimension()

# -------------------------
# Create Collection
# -------------------------

collections = [
    c.name
    for c in client.get_collections().collections
]

if COLLECTION_NAME not in collections:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_size,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection: {COLLECTION_NAME}")

# -------------------------
# Read Documents
# -------------------------

docs_path = Path(DOCS_DIR)

points = []

for file in docs_path.glob("*.md"):

    print(f"Processing {file.name}")

    text = file.read_text(encoding="utf-8")

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
    )

    print(f"  Chunks: {len(chunks)}")

    for chunk_number, chunk in enumerate(chunks):

        vector = model.encode(
            chunk
        ).tolist()

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "source": file.name,
                    "chunk": chunk_number,
                    "text": chunk,
                },
            )
        )

# -------------------------
# Insert into Qdrant
# -------------------------

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

print()
print(f"Inserted {len(points)} chunks into Qdrant")
