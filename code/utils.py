import os
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")

def load_pdf(pdf_name):
    loader = PyPDFLoader(pdf_name, mode='single')
    pages = loader.load()
    return pages

def load_all_pdf_data(pdfs_directory = DATA_DIR):
    loader = PyPDFDirectoryLoader(pdfs_directory)
    documents = loader.load()
    all_text = "\n".join(doc.page_content for doc in documents)
    return all_text

