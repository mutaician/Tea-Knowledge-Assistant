import os
from langchain_community.document_loaders import PyPDFDirectoryLoader

# You need to install: tiktoken, langchain-community
import tiktoken

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def get_all_pdf_text(data_dir):
	loader = PyPDFDirectoryLoader(data_dir)
	documents = loader.load()
	return "\n".join(doc.page_content for doc in documents)

def count_tokens(text, encoding_name="cl100k_base"):
	encoding = tiktoken.get_encoding(encoding_name)
	return len(encoding.encode(text))

def main():
	text = get_all_pdf_text(DATA_DIR)
	total_tokens = count_tokens(text)
	cost_per_million = 0.02  # USD for text-embedding-3-small
	cost = (total_tokens / 1_000_000) * cost_per_million
	print(f"Total tokens: {total_tokens}")
	print(f"Estimated embedding cost (USD): {cost:.4f}")

if __name__ == "__main__":
	main()
