from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json

# ── Load chunks ────────────────────────────────────────────
print("Loading chunks...")
with open("textile_chunks.json") as f:
    chunks = json.load(f)
print(f"Total chunks loaded: {len(chunks)}")

# ── Initialize Ollama embeddings ───────────────────────────
print("\nInitializing nomic-embed-text via Ollama...")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ── Initialize ChromaDB with LangChain ────────────────────
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./textile_nomic_db",
    collection_name="textile_engineering"
)

# ── Convert chunks to LangChain Document objects ──────────
# LangChain's Chroma wrapper needs Document objects
# Document has two parts:
#   page_content → the actual text to embed
#   metadata     → everything else (course, professor etc)

print("Converting chunks to Documents...")
documents = []
for chunk in chunks:
    doc = Document(
        page_content=chunk["chunk_text"],
        metadata={
            "course_id":    chunk["course_id"],
            "course_title": chunk["course_title"],
            "professor":    chunk["professor"],
            "institute":    chunk["institute"],
            "lecture_name": chunk["lecture_name"],
            "source_type":  chunk["source_type"],
            "source_url":   chunk["source_url"],
            "chunk_index":  chunk["chunk_index"],
            "chunk_total":  chunk["chunk_total"],
        }
    )
    documents.append(doc)

print(f"Documents ready: {len(documents)}")

# ── Ingest in batches ──────────────────────────────────────
# nomic-embed-text via Ollama is slower than MiniLM
# because it runs locally via HTTP calls to Ollama
# batch size of 100 is safe
BATCH_SIZE = 100
total = len(documents)

print("\nEmbedding and storing...")
for i in range(0, total, BATCH_SIZE):
    batch = documents[i : i + BATCH_SIZE]
    vectorstore.add_documents(batch)
    print(f"  Stored: {min(i + BATCH_SIZE, total):5d} / {total}")

# ── Persist to disk ────────────────────────────────────────
vectorstore.persist()
print(f"\n{'='*50}")
print(f"✅ Total stored: {total}")
print(f"📁 Saved at:    ./textile_nomic_db")

# ── Test query ─────────────────────────────────────────────
print("\n--- Testing with sample query ---")
results = vectorstore.similarity_search(
    "what is ring spinning mechanism",
    k=3
)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Course:  {doc.metadata['course_title']}")
    print(f"  Lecture: {doc.metadata['lecture_name']}")
    print(f"  Source:  {doc.metadata['source_type']}")
    print(f"  Text:    {doc.page_content[:150]}...")