import chromadb
import ollama
import pymupdf
import sys
import uuid
import logging
import os

EMBEDDING_MODEL = "nomic-embed-text"
LANGUAGE_MODEL = "qwen2.5:3b-instruct"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text(t:str):
    return t.replace("-\n", "")

def extract_text(pdf_path:str):
    doc = pymupdf.open(pdf_path)
    pages = []

    for i, p in enumerate(doc):
        p:pymupdf.Page = p
        t = p.get_text(flags=0)
        if t.strip():
            t = clean_text(t)
            pages.append({"page": i+1, "text": t})

    doc.close()
    return pages

def chunk_text(t:str):
    chunks:list[str] = []
    start:int = 0

    while start < len(t):
        end = start + CHUNK_SIZE
        chunk = t[start:end]
        chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

def gen_embeddings(chunks:list[str]):
    embeddings = []

    for c in chunks:
        embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=c)
        embeddings.append(embedding["embedding"])

    return embeddings

def build(pdf_path:str):
    logger.info(f"Building {pdf_path}")

    pages = extract_text(pdf_path)

    for p in pages:
        p["chunks"] = chunk_text(p["text"])
        p["embeddings"] = gen_embeddings(p["chunks"])

        logger.info(f"Chunked and embedded page {p['page']}")

    cl = chromadb.PersistentClient("./chromadb")
    fname, _ = os.path.splitext(os.path.basename(pdf_path))
    coll = cl.get_or_create_collection(f"{fname}-docs")

    logger.info("Built vector database")

    for p in pages:
        for i in range(len(p["chunks"])):
            uuid_generated = str(uuid.uuid4())
            coll.add(
                ids=[uuid_generated],
                embeddings=[p["embeddings"][i]],
                documents=[p["chunks"][i]],
                metadatas=[{"source": pdf_path, "page": p["page"]}]
            )

            logger.info(f"Added {uuid_generated} to collection")

    logger.info(f"Added {coll.count()} items to {coll.name}")

def get(pdf_path:str):
    cl = chromadb.PersistentClient("./chromadb")
    fname, _ = os.path.splitext(os.path.basename(pdf_path))
    coll = cl.get_collection(f"{fname}-docs")

    logger.info(f"Loaded {coll.count()} from {coll.name}")

    data = coll.get(
        include=["embeddings", "documents", "metadatas"]
    )

if __name__ == "__main__":
    #build(sys.argv[1])
    get(sys.argv[1])