# RAG Pipeline with Qdrant, Embeddings & Local LLM

A Retrieval-Augmented Generation (RAG) pipeline built using **Qdrant**, **Sentence Transformers**, and **Ollama**. This project demonstrates how modern AI applications retrieve relevant knowledge from a vector database before generating answers with a Large Language Model (LLM).

Instead of relying solely on an LLM's internal knowledge, the application retrieves semantically relevant document chunks from a vector database, reranks them using a Cross Encoder, and uses the retrieved context to generate grounded responses.

---

# Architecture

```
                    User Question
                          │
                          ▼
               Sentence Transformer
            (BAAI/bge-small-en-v1.5)
                          │
                          ▼
                  Query Embedding
                          │
                          ▼
                     Qdrant Search
                  (Top-K Retrieval)
                          │
                          ▼
                 Cross Encoder Reranker
                          │
                          ▼
              Most Relevant Document Chunks
                          │
                          ▼
                  Prompt Construction
                          │
                          ▼
                Ollama (Phi3 / Llama3)
                          │
                          ▼
                   Final AI Response
```

---

# Features

* Semantic search using vector embeddings
* Qdrant vector database
* Markdown document ingestion
* PDF document ingestion
* Intelligent document chunking
* Sentence Transformer embeddings
* Cross Encoder reranking
* Local LLM inference with Ollama
* Retrieval-Augmented Generation (RAG)
* Source attribution for generated answers
* Hallucination reduction through contextual grounding

---

# Tech Stack

* Python
* Qdrant
* Sentence Transformers
* BAAI/bge-small-en-v1.5
* Cross Encoder (MS MARCO MiniLM)
* Ollama
* Phi3
* Docker

---

# Repository Structure

```
rag-qdrant-pipeline/

├── docs/
│   ├── kubernetes.md
│   ├── prometheus.md
│   └── tempo.md
│
├── pdfs/
│   └── Kubernetes-eBook.pdf
│
├── screenshots/
│   ├── architecture.png
│   ├── retrieval.png
│   ├── reranker.png
│   └── rag-answer.png
│
├── scripts/
│   ├── test_embedding.py
│   ├── qdrant_insert_md.py
│   ├── qdrant_search.py
│   ├── ingest_docs.py
│   ├── pdf_ingest.py
│   ├── reranker_test.py
│   └── rag.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Project Workflow

1. Documents (Markdown or PDF) are loaded.
2. Documents are split into manageable chunks.
3. Each chunk is converted into an embedding using Sentence Transformers.
4. Embeddings are stored in Qdrant.
5. A user submits a question.
6. The question is embedded.
7. Qdrant retrieves the most semantically similar chunks.
8. A Cross Encoder reranks the retrieved chunks.
9. The highest-ranked chunks are added to the prompt.
10. Ollama generates the final grounded response.

---

# Running the Project

## 1. Start Qdrant

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  qdrant/qdrant
```

---

## 2. Install Dependencies

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Start Ollama

```bash
ollama serve
```

Pull a model if required:

```bash
ollama pull phi3
```

---

## 4. Ingest Markdown Documents

```bash
python scripts/ingest_docs.py
```

---

## 5. Ingest PDF Documents

```bash
python scripts/pdf_ingest.py
```

---

## 6. Run Semantic Search

```bash
python scripts/qdrant_search.py
```

---

## 7. Run the Complete RAG Pipeline

```bash
python scripts/rag.py
```

---

# Example

Question

```
What is a Kubernetes Pod?
```

Retrieved Chunks

```
Source: kubernetes.md

Pods are the smallest deployable units in Kubernetes.

Pods run on Worker Nodes.
```

Generated Answer

```
A Kubernetes Pod is the smallest deployable unit in Kubernetes.
Pods run on Worker Nodes and can contain one or more containers.
```

---

# What I Learned

Through this project I implemented the core components of a modern Retrieval-Augmented Generation (RAG) system, including:

* Vector embeddings
* Semantic similarity search
* Vector databases
* Document chunking
* PDF parsing
* Retrieval pipelines
* Cross Encoder reranking
* Prompt construction
* Grounded response generation
* Local LLM inference

---

# Future Improvements

* Hybrid Search (Vector + BM25)
* Metadata Filtering
* Query Rewriting
* Parent Document Retrieval
* Multi-Query Retrieval
* Agentic RAG
* vLLM integration
* Kubernetes deployment
* OpenTelemetry instrumentation
* Grafana observability
* Production REST API

---

# Screenshots

Add screenshots of:

* Qdrant retrieval
* Retrieved chunks
* Reranker output
* Final RAG answer
* Architecture diagram

---

# License

This project is licensed under the MIT License.
