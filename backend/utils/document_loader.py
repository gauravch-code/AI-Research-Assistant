import os, sys

def load_documents(directory):
    docs = []
    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs
