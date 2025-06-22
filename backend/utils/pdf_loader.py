# backend/utils/pdf_loader.py

import os
from PyPDF2 import PdfReader

def pdf_to_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text_pages = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text_pages.append(txt)
    return "\n\n".join(text_pages)

def save_uploaded_pdf(uploaded_file, output_folder="data/processed_docs"):
    """
    Save a Streamlit-uploaded file object to disk and convert to .txt.
    Returns the path to the .txt file.
    """
    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # convert
    txt = pdf_to_text(pdf_path)
    txt_name = uploaded_file.name.replace(".pdf", ".txt")
    txt_path = os.path.join(output_folder, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt_name  # filename of the txt in processed_docs
