import os
from dotenv import load_dotenv

# 1) Retrieval
from backend.retriever.document_retriever import query_pinecone
# 2) PDF→text
from backend.utils.pdf_loader import pdf_to_text
# 3) Text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 4) Chat model & message schema
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY is missing in your .env")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_KEY,
    temperature=0.6
)

# 5) Prompt template
PROMPT_TMPL = (
    "You are a knowledgeable AI assistant. Use ONLY the context below to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer (cite sources if helpful):"
)

# 6) Chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=200,
    separators=["\n\n", "\n", ". ", "? ", "! "]
)

def _chat(prompt: str) -> str:
    """
    Send a single-user message to ChatOpenAI and return the assistant's response content.
    """
    from langchain.schema import HumanMessage

    messages = [HumanMessage(content=prompt)]
    # ChatOpenAI returns an AIMessage directly
    ai_message = llm(messages)
    return ai_message.content.strip()


def answer_on_text(text: str, query: str) -> str:
    chunks = splitter.split_text(text)
    context = "\n\n---\n\n".join(chunks)
    prompt = PROMPT_TMPL.format(context=context, question=query)
    return _chat(prompt).strip()

def answer_query(query: str) -> str:
    snippets = query_pinecone(query, top_k=3)
    text = "\n\n---\n\n".join(snippets)
    return answer_on_text(text, query)

def answer_on_pdf(pdf_path: str, query: str) -> str:
    text = pdf_to_text(pdf_path)
    return answer_on_text(text, query)

def summarize_on_text(text: str, sentence_count: int = 5) -> str:
    chunks = splitter.split_text(text)
    context = "\n\n---\n\n".join(chunks)
    prompt = (
        f"You are an expert assistant. Summarize the following context "
        f"in {sentence_count} sentences, focusing on the core ideas.\n\n"
        f"{context}"
    )
    return _chat(prompt).strip()

def summarize_pdf(pdf_path: str, sentence_count: int = 5) -> str:
    text = pdf_to_text(pdf_path)
    return summarize_on_text(text, sentence_count)
