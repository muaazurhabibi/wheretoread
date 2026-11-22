import chromadb
import ollama
import pymupdf
import sys
import uuid
import logging

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
    pages = extract_text(pdf_path)

    for p in pages:
        p["chunks"] = chunk_text(p["text"])
        p["embeddings"] = gen_embeddings(p["chunks"])

    cl = chromadb.PersistentClient("./chroma")
    coll = cl.get_or_create_collection("docs")

    for pn, p in enumerate(pages):
        for i in range(len(p["chunks"])):
            coll.add(
                ids=[str(uuid.uuid4())],
                embeddings=[p["embeddings"][i]],
                documents=[p["chunks"][i]],
                metadatas=[{"source": pdf_path, "page": pn}]
            )

if __name__ == "__main__":
    build(sys.argv[1])