from pathlib import Path
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

COLLECTION_NAME = "knowledge"

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

client = QdrantClient(
    "localhost",
    port=6333,
    check_compatibility=False
)


def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


pdf_dir = Path("pdfs")

point_id = 1000

for pdf_file in pdf_dir.glob("*.pdf"):

    print(f"\nProcessing {pdf_file.name}")

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    chunks = chunk_text(text)

    print(f"Chunks: {len(chunks)}")

    vectors = model.encode(chunks)

    points = []

    for chunk, vector in zip(chunks, vectors):

        points.append(
            PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "source": pdf_file.name,
                    "text": chunk
                }
            )
        )

        point_id += 1

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

print("\nPDF ingestion complete")
