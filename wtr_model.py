import ollama
import logging
import subprocess
import urllib
import time

import wtr_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def begin_ollama():
    process = subprocess.Popen(
        ["ollama", "serve"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL
    )

    url = "http://localhost:11434"
    for _ in range(10):
        try:
            urllib.request.urlopen(url)
            print("Ollama server is running.")
            return process
        except Exception:
            time.sleep(1)

    return None

def retrieve(query:str, n:int, coll_name:str):
    cl = wtr_data.cl
    coll = cl.get_collection(coll_name)

    q_embedded = ollama.embeddings(wtr_data.EMBEDDING_MODEL, query)["embedding"]
    res = coll.query(query_embeddings=[q_embedded], n_results=n)

    chunks = res['documents'][0]
    metas = res['metadatas'][0]

    logger.info("Got context from vector database")

    return list(zip(chunks, metas))

def answer(query:str, n:int, coll_name:str):
    retrievals = retrieve(query, n, coll_name)
    context = f"\n\n".join(f"[Source: {meta['source']}, Page: {meta['page']}]\n\n{text}" for text, meta in retrievals)

    prompt = f"Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.\n\nContext: {context}\n\n\nUser query: {query}\n\n\n Answer"

    logger.info("Thinking...")
    res = ollama.chat(wtr_data.LANGUAGE_MODEL, messages=[{"role": "user", "content": prompt}])

    return res.message.content