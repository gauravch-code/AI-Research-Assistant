import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Instantiate client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

INDEX_NAME = "research-assistant-index"

def get_index():
    # Simply return the index instance you've created manually
    return pc.Index(INDEX_NAME)

if __name__ == "__main__":
    idx = get_index()
    print("✅ Fetched Pinecone index:", INDEX_NAME)
