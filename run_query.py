from backend.utils.document_loader import load_documents
from backend.retriever.pinecone_setup import get_index
from backend.retriever.document_retriever import embed_and_upsert
from backend.rag_pipeline.rag_engine import answer_query

def main():
    print("⏳ Loading documents...")
    docs = load_documents("data/processed_docs")

    print("🧠 Fetching Pinecone index...")
    get_index()

    print("📤 Embedding & uploading...")
    embed_and_upsert(docs)

    print("\n✅ Ready! Type your question below (or 'exit' to quit).")
    while True:
        query = input("\nEnter your academic question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        print("🤖 Thinking...")
        try:
            answer = answer_query(query)
            print(f"\n📝 Answer:\n{answer}")
        except Exception as e:
            print(f"⚠️  Error while answering: {e}")

if __name__ == "__main__":
    main()
