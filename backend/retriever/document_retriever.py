from sentence_transformers import SentenceTransformer
from backend.retriever.pinecone_setup import get_index
import os
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
index = get_index()

def embed_and_upsert(docs):
    """
    docs: List[str] of FULL documents loaded from disk.
    We embed each doc, but only upsert a small snippet (first 1k chars) as metadata.
    """
    for i, doc in enumerate(docs):
        vec = model.encode(doc).tolist()
        # only keep a small preview in metadata to stay under size limit
        snippet = doc[:1000]  # ~1 KB of text
        meta = {"text_snippet": snippet}
        index.upsert([(f"id-{i}", vec, meta)])

def query_pinecone(query, top_k=3):
    """
    Returns the stored snippet for each match.
    You can adjust how you fetch the full text from local docs
    if you need more context for QA.
    """
    vec = model.encode(query).tolist()
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    # return just the snippet
    return [match["metadata"]["text_snippet"] for match in res["matches"]]
