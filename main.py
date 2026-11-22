import chromadb
import ollama
import pymupdf

EMBEDDING_MODEL = "nomic-embed-text"
LANGUAGE_MODEL = "qwen2.5:3b-instruct"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

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
        embeddings.append(embedding)

    return embeddings

if __name__ == "__main__":
    pages = extract_text("C:\\Users\\ADMIN\\Downloads\\Calculus by Thomas and Finny.pdf")
    for p in pages:
        p["chunks"] = chunk_text(p["text"])
        p["embeddings"] = gen_embeddings(p["chunks"])
    