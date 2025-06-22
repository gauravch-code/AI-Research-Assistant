import os
from dotenv import load_dotenv

from backend.retriever.document_retriever import query_pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

load_dotenv()

# Chunk splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

# HuggingFace pipeline for generation
hf_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device_map="auto",
    max_length=512,
    do_sample=False,
    temperature=0.7
)

PROMPT_TMPL = (
    "You are a knowledgeable AI assistant. Use ONLY the context below to answer.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer (cite sources if helpful):"
)
import os
from dotenv import load_dotenv

# … your existing imports and answer_query here …

# New: load a single document by filename (without “.txt”)
def summarize_doc(doc_name: str, sentence_count: int = 5) -> str:
    """
    Summarize exactly one file under data/processed_docs/{doc_name}.txt
    """

    path = os.path.join("data", "processed_docs", f"{doc_name}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such document: {doc_name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Build a simple prompt
    prompt = (
        f"You are an expert summarizer. Please produce a concise summary "
        f"in {sentence_count} sentences of the following paper:\n\n{text}"
    )

    # 3) Call your HF pipeline directly (reuse hf_pipe from above)
    summary = hf_pipe(prompt)[0]["generated_text"].strip()
    # Ensure it ends with a period
    if not summary.endswith("."):
        summary += "."
    return summary


def answer_query(query: str) -> str:
    # Retrieve top-3 snippets from Pinecone
    snippets = query_pinecone(query, top_k=3)

    # Chunk them to respect T5's input limits
    chunks = []
    for snip in snippets:
        chunks.extend(splitter.split_text(snip))

    # Join chunks into one context string
    context = "\n\n---\n\n".join(chunks)

    # Build and run the prompt
    prompt = PROMPT_TMPL.format(context=context, question=query)
    output = hf_pipe(prompt)[0]["generated_text"]
    return output
