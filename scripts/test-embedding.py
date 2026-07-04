from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

text = "Kubernetes is a container orchestration platform"

embedding = model.encode(text)

print(len(embedding))
print(embedding[:10])
