import os
import shutil
import chromadb
from paths import VECTOR_DB_DIR
from utils import load_all_pdf_data
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# initialize db and collection
def initialize_db(persist_directory=VECTOR_DB_DIR, collection_name="Tea-Documents", delete_existing=True):
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)
    
    os.makedirs(persist_directory, exist_ok=True)

    client = chromadb.PersistentClient(persist_directory)

    collection = client.get_or_create_collection(
        name=collection_name, configuration={"hnsw": {"space": "cosine"}}
    )

    return collection


# chunk pdfs
def chunk_pdfs(pdf_data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return text_splitter.split_text(pdf_data)


# embed documents


def embend_documents(documents, model_name = 'text-embedding-3-small'):
    model = OpenAIEmbeddings(model=model_name)
    embeddings = model.embed_documents(documents)
    return embeddings


# insert documents into vector db
def insert_pdfs(collection: chromadb.Collection, pdfs):
    next_id = collection.count()

    chunked_pdfs = chunk_pdfs(pdfs)
    embeddings = embend_documents(chunked_pdfs)

    ids = list(range(next_id, next_id + len(chunked_pdfs)))
    ids = [f"document_{id}" for id in ids]
    collection.add(
        embeddings=embeddings, ids=ids, documents=chunked_pdfs  # type: ignore
    )


def main():
    collection = initialize_db(VECTOR_DB_DIR, "tea-documents")
    pdfs = load_all_pdf_data()
    insert_pdfs(collection, pdfs)

    print("Total documents in collection: ", collection.count())


if __name__ == "__main__":
    main()
