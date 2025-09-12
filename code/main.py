import os
from typing import List
import chromadb
from dotenv import load_dotenv
from paths import VECTOR_DB_DIR, PROMPT_CONFIG_FPATH
from prompt_builder import build_prompt_from_config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from utils import load_yaml_config

load_dotenv()


def get_collection(collection_name="tea-documents") -> chromadb.Collection:
    client = chromadb.PersistentClient(VECTOR_DB_DIR)
    
    return client.get_collection(collection_name)


def get_embedder():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def retrieve_relevant_documents(
    query: str, top_k: int = 5, threshold: float = 0.6
) -> List[str]:
    """Return the most relevant chunks below a cosine-distance threshold."""
    collection = get_collection()
    embedder = get_embedder()

    q_emb = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "distances"],
    )

    docs: List[str] = []
    if not results:
        return docs
    documents_field = results.get("documents")
    distances_field = results.get("distances")
    if (
        not documents_field
        or not isinstance(documents_field, list)
        or not documents_field
    ):
        return docs
    docs_raw = documents_field[0] or []
    dists = (
        distances_field[0]
        if distances_field and isinstance(distances_field, list) and distances_field
        else []
    )

    if dists:
        for doc, dist in zip(docs_raw, dists):
            try:
                if dist < threshold:
                    docs.append(doc)
            except Exception:
                continue
        # Fallback: if nothing passed the threshold but we have results, return top_k unfiltered
        if not docs and docs_raw:
            docs.extend(docs_raw[:top_k])
    else:
        # If distances weren't returned, just take top_k docs
        docs.extend(docs_raw[:top_k])
    return docs


def respond_to_query(query, prompt_config, n_results=5, threshold=0.7, documents=None):

    if documents is not None:
        relevant_documents = documents
    else:
        relevant_documents = retrieve_relevant_documents(query, n_results, threshold)

    print("Relevant documents: \n")
    for doc in relevant_documents:
        print(doc)
    print("-" * 100)

    input_data = (
        f"Relevant documents:\n\n{relevant_documents}\n\nUser's question: \n\n{query}"
    )

    rag_assistant_prompt = build_prompt_from_config(prompt_config, input_data)

    llm = ChatOpenAI(model="gpt-5-nano")

    response = llm.invoke(rag_assistant_prompt)
    return response.content


def main():
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
    rag_prompt = prompt_config["rag_assistant_prompt"]

    print(
        "Welcome to Tea Knowledge Assistant that will answer any of your tea questions with high accuracy\n (type 'exit' to quit)"
    )
    exit_app = False
    while not exit_app:
        query = input("Enter Question: ")
        if query == "exit":
            exit_app = True
        exit()
        response = respond_to_query(query, rag_prompt)
        print(response)


if __name__ == "__main__":
    main()
