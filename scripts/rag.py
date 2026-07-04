from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests

# ----------------------------
# Configuration
# ----------------------------

COLLECTION_NAME = "knowledge"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

OLLAMA_MODEL = "phi3"

MIN_SCORE = 0.5

VECTOR_RESULTS = 10
FINAL_RESULTS = 3

# ----------------------------
# Connect Qdrant
# ----------------------------

client = QdrantClient(
    host="localhost",
    port=6333,
    check_compatibility=False
)

# ----------------------------
# Models
# ----------------------------

print("Loading embedding model...")

embedder = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

# ----------------------------
# User Question
# ----------------------------

question = input("Question: ").strip()

if not question:
    print("Please enter a question.")
    exit(1)

# ----------------------------
# Embed Question
# ----------------------------

query_vector = embedder.encode(
    question
).tolist()

# ----------------------------
# Vector Search
# ----------------------------

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=VECTOR_RESULTS,
).points

if not results:
    print("\nNo documents found.")
    exit(0)

if results[0].score < MIN_SCORE:
    print("\nNo relevant information found.")
    print(f"Best score: {results[0].score:.4f}")
    exit(0)

# ----------------------------
# Reranking
# ----------------------------

pairs = [
    (question, hit.payload["text"])
    for hit in results
]

rerank_scores = reranker.predict(
    pairs
)

reranked = []

for hit, score in zip(results, rerank_scores):
    reranked.append(
        {
            "hit": hit,
            "rerank_score": float(score)
        }
    )

reranked.sort(
    key=lambda x: x["rerank_score"],
    reverse=True
)

top_hits = reranked[:FINAL_RESULTS]

# ----------------------------
# Show Retrieved Chunks
# ----------------------------

print("\nRetrieved Chunks (After Reranking):\n")

for item in top_hits:

    hit = item["hit"]

    print(
        f"Source : "
        f"{hit.payload.get('source', 'unknown')}"
    )

    print(
        f"Vector Score : "
        f"{hit.score:.4f}"
    )

    print(
        f"Rerank Score : "
        f"{item['rerank_score']:.4f}"
    )

    print(
        hit.payload["text"][:300]
    )

    print("-" * 80)

# ----------------------------
# Build Context
# ----------------------------

context = "\n\n".join(
    item["hit"].payload["text"]
    for item in top_hits
)

# ----------------------------
# Prompt
# ----------------------------

prompt = f"""
You are a helpful assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present in the context say:
  "I don't know based on the provided documents."
- Do not make up information.
- Keep answers concise.

Context:
{context}

Question:
{question}

Answer:
"""

# ----------------------------
# Ask Ollama
# ----------------------------

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    },
)

response.raise_for_status()

answer = response.json()["response"]

# ----------------------------
# Show Answer
# ----------------------------

print("\nAnswer:\n")

print(answer)

# ----------------------------
# Sources
# ----------------------------

print("\nSources:\n")

for item in top_hits:

    hit = item["hit"]

    print(
        f"- {hit.payload.get('source', 'unknown')} "
        f"(vector={hit.score:.4f}, "
        f"rerank={item['rerank_score']:.4f})"
    )
