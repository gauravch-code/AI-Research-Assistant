import os, sys

# Make sure the repo root is on PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from backend.rag_pipeline.rag_engine import answer_query, summarize_doc

# Get list of available docs
docs = [
    fname[:-4]  # strip “.txt”
    for fname in os.listdir("data/processed_docs")
    if fname.endswith(".txt")
]

st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📚 Smart Research Assistant")

# Let user choose between “QA” and “Summarize”
mode = st.radio("Mode", ["Ask a question", "Summarize a paper"])

if mode == "Ask a question":
    query = st.text_input("🔍 Your question:", "")
    if st.button("Ask"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤖 Thinking..."):
                try:
                    answer = answer_query(query).strip()
                    if not answer.endswith("."):
                        answer += "."
                    st.markdown("### 📝 Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {e}")

else:  # Summarize mode
    doc_choice = st.selectbox("Select paper to summarize", docs)
    count = st.slider("Number of sentences", 1, 10, 5)
    if st.button("Summarize"):
        with st.spinner("📝 Summarizing..."):
            try:
                summary = summarize_doc(doc_choice, sentence_count=count)
                st.markdown("### 📄 Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error: {e}")
