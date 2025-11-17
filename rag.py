from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
load_dotenv()

from db import insert_document, search_similar_by_embedding

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    try:
        # Prefer langchain splitter if available
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    except Exception:
        # Fallback naive splitter
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i : i + chunk_size])
            i += chunk_size - chunk_overlap
        return chunks


def embed_and_store(title: str, text: str, metadata: Dict[str, Any] = None):
    """Chunk text, compute embeddings (using OpenAI if available), and store chunks in Postgres vector DB via insert_document."""
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = None
        try:
            if OPENAI_API_KEY:
                import openai

                openai.api_key = OPENAI_API_KEY
                resp = openai.Embedding.create(input=chunk, model=embedding_model)
                embedding = resp["data"][0]["embedding"]
        except Exception as e:
            print("Embedding failed for chunk", i, e)

        md = dict((metadata or {}).copy())
        md.update({"chunk_index": i})
        insert_document(title=f"{title} - chunk {i}", content=chunk, metadata=md, embedding=embedding)


def search_rag(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Compute embedding for the query and return top-k similar chunks from the DB."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY required for embed-based search")
    import openai

    openai.api_key = OPENAI_API_KEY
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    resp = openai.Embedding.create(input=query, model=embedding_model)
    emb = resp["data"][0]["embedding"]
    results = search_similar_by_embedding(emb, k=k)
    return results

