from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

question = "Where do pods run?"

chunks = [
    "Pods run on worker nodes",
    "Docker is a container runtime",
    "Prometheus stores metrics"
]

pairs = [(question, c) for c in chunks]

scores = model.predict(pairs)

for chunk, score in zip(chunks, scores):
    print(score, chunk)
