# AI Research Assistant

A Retrieval-Augmented Generation (RAG) system that enables academic question answering and document summarization over a custom corpus of text and PDF files. The project integrates Pinecone for vector search, LangChain for orchestration, and OpenAI’s GPT-3.5-Turbo (with a local Flan-T5 fallback) to generate grounded responses. A Streamlit interface provides an interactive user experience.

## Features

- **Document Ingestion**  
  Supports plain text and PDF files. PDFs are automatically converted to text.

- **Vector Search**  
  Uses Pinecone to index document embeddings for semantic retrieval.

- **Question Answering**  
  Retrieves relevant passages from the corpus and generates answers grounded in source material.

- **Document Summarization**  
  Summarizes any document (text or PDF) into a specified number of sentences.

- **Model Fallback**  
  When `OPENAI_API_KEY` is set, uses GPT-3.5-Turbo; otherwise falls back to a local `google/flan-t5-base` pipeline.

- **Interactive UI**  
  Streamlit app for uploading files, rebuilding the index, asking questions, and summarizing documents.

## Repository Structure

```

.
├── backend
│   ├── rag_pipeline
│   │   └── rag_engine.py  
│   ├── retriever
│   │   ├── pinecone_setup.py      
│   │   └── document_retriever.py 
│   └── utils
│       ├── document_loader.py     
│       └── pdf_loader.py          
├── data
│   └── processed_docs            
├── frontend
│   └── streamlit_app.py          
├── run_query.py                  
├── requirements.txt              
├── .env                          
└── README.md

````

## Prerequisites

- Python 3.8 or later  
- A Pinecone account and API key  
- (Optional) An OpenAI account and API key for GPT-3.5-Turbo  
- Recommended hardware: CPU is sufficient; a GPU will accelerate local model inference

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/gauravch-code/AI-Research-Assistant.git
   cd AI-Research-Assistant
````

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a file named `.env` in the project root with the following entries:

   ```
   PINECONE_API_KEY=your-pinecone-key
   PINECONE_ENVIRONMENT=your-pinecone-environment
   OPENAI_API_KEY=sk-...          # Optional, required for GPT-3.5-Turbo
   ```

## Usage

### Command-Line Interface

```bash
python run_query.py
```

* Loads and indexes all files in `data/processed_docs/`.
* Prompts for your question.
* Returns a grounded answer based on indexed documents.

### Streamlit Web Application

```bash
streamlit run frontend/streamlit_app.py
```

1. Upload `.txt` or `.pdf` files via the sidebar.
2. Click **Rebuild Index** to embed and index all documents.
3. Choose between **Ask a Question** or **Summarize a Document**.

   * **Ask a Question**: enter a query about the full corpus.
   * **Summarize a Document**: select a file and specify the number of sentences.

## How It Works

1. **Embedding**: each document is converted to embeddings using a HuggingFace embedding model (`all-MiniLM-L6-v2`).
2. **Indexing**: embeddings are stored in Pinecone for efficient similarity search.
3. **Retrieval**: given a query, top-k relevant passages are fetched from Pinecone.
4. **Chunking**: lengthy passages are split into manageable chunks to respect model context limits.
5. **Generation**:

   * **Q\&A**: the model (GPT-3.5-Turbo or Flan-T5) consumes the retrieved chunks and the question to produce an answer.
   * **Summarization**: the model is prompted to condense the context into the specified number of sentences.

## Customization and Extension

* **Model configuration**: change `model_name` in `rag_engine.py` to switch to GPT-4 or another local model.
* **Retrieval enhancements**: implement hybrid or reranked retrieval strategies.
* **Fine-tuning**: integrate PEFT or LoRA for domain-specific model adaptation.
* **UI improvements**: add feedback collection, usage analytics, or custom styling to the Streamlit app.

## Contributing

1. Fork the repository and create a new branch.
2. Implement your changes, including tests and documentation updates.
3. Submit a pull request for review.

## Contact

- **Author**: Gaurav Chintakunta  
- **Email**: [gchin6@uic.edu](mailto:gchin6@uic.edu)  
- **GitHub**: [gauravch-code](https://github.com/gauravch-code)

For any questions or feedback, please open an issue on GitHub or reach out via email.

```
```
