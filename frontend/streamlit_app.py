import os, sys
import streamlit as st

# make sure backend/ is on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.utils.pdf_loader import save_uploaded_pdf
from backend.utils.document_loader import load_documents
from backend.retriever.document_retriever import embed_and_upsert
from backend.retriever.pinecone_setup import get_index
from backend.rag_pipeline.rag_engine import answer_query, answer_on_pdf, summarize_on_text

st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📚 Smart Research Assistant")

# -------------------------------------------------------------------
# 1) Upload files into session_state.corpus
# -------------------------------------------------------------------
if "corpus_files" not in st.session_state:
    st.session_state.corpus_files = []  # list of filenames

st.sidebar.header("📁 Corpus Management")
uploaded = st.sidebar.file_uploader(
    "Upload TXT or PDF", type=["txt","pdf"], accept_multiple_files=True
)
if uploaded:
    for f in uploaded:
        # PDF → TXT conversion
        if f.type == "application/pdf":
            txt_name = save_uploaded_pdf(f)
        else:  # plain text
            path = os.path.join("data","processed_docs", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            txt_name = f.name
        if txt_name not in st.session_state.corpus_files:
            st.session_state.corpus_files.append(txt_name)
    st.sidebar.success(f"Corpus now has: {len(st.session_state.corpus_files)} files")

# -------------------------------------------------------------------
# 2) (Re)embed corpus when it changes
# -------------------------------------------------------------------
if st.sidebar.button("🔄 Rebuild Index"):
    docs = load_documents("data/processed_docs")
    get_index(); embed_and_upsert(docs)
    st.sidebar.success("Index rebuilt!")

# -------------------------------------------------------------------
# 3) Main panel: choose action
# -------------------------------------------------------------------
action = st.radio("What would you like to do?", ["Ask a Question","Summarize a Document"])

if action == "Ask a Question":
    query = st.text_input("🔍 Your question:", "")
    if st.button("Ask"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤖 Thinking over your corpus..."):
                try:
                    ans = answer_query(query).strip()
                    if not ans.endswith("."):
                        ans += "."
                    st.markdown("### 📝 Answer:")
                    st.write(ans)
                except Exception as e:
                    st.error(f"Error: {e}")

else:  # Summarize
    if not st.session_state.corpus_files:
        st.info("Upload at least one document to summarize.")
    else:
        doc = st.selectbox("Select document", st.session_state.corpus_files)
        n = st.slider("Sentences in summary",1,10,5)
        if st.button("Summarize"):
            with st.spinner("📝 Summarizing..."):
                try:
                    # if it’s a PDF upload, we answer on_pdf, otherwise load its text
                    path = os.path.join("data","processed_docs", doc)
                    if doc.lower().endswith(".pdf"):
                        summary = answer_on_pdf(path, f"Please summarize this document in {n} sentences")
                    else:
                        # read .txt
                        text = open(path,encoding="utf-8").read()
                        summary = summarize_on_text(text, sentence_count=n)
                    summary = summary.strip()
                    if not summary.endswith("."):
                        summary += "."
                    st.markdown("### 📄 Summary:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
