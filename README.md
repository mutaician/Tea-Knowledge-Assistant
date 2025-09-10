# Tea-Knowledge-Assistant
Retrieval-augmented question-answering system that unifies Wikipedia’s comprehensive tea content with Kenya-specific guidelines from the Tea Board of Kenya

## Problem Statement
Accessing reliable, Kenya-relevant knowledge on tea cultivation, processing, and regulation is difficult because key information is fragmented across sources. Farmers and students often rely on Wikipedia for general knowledge, while critical guidance—such as greenleaf quality requirements and cultivation manuals—is published by the Tea Board of Kenya in long PDFs or notices. Manually searching through these documents is slow and makes it harder to find answers to practical questions like pest control methods, harvesting standards, or processing differences.

## Suggested Solution
The Tea Knowledge Assistant is a retrieval-augmented question-answering system that unifies Wikipedia’s comprehensive tea content with Kenya-specific guidelines from the Tea Board of Kenya. By embedding Wikipedia articles (e.g., Tea, Tea processing, Tea production in Kenya) and summarizing key Tea Board documents (cultivation manuals, quality requirements, regulations), the assistant provides quick, cited answers to user queries. This setup ensures global background knowledge while highlighting Kenya’s local standards and practices.

## Impact

For farmers: easy access to cultivation and quality requirement details without sifting through long PDFs.

For students: structured answers combining global tea knowledge with local Kenyan context.

For researchers/policy makers: a consolidated, conversational entry point into scattered but critical resources.

## Why RAG fits

Because the knowledge is scattered across Wikipedia pages and government publications, a retrieval-based system ensures answers are grounded in the most relevant text chunks, reducing noise and making critical guidance more discoverable.

## 🛠️ Technical Architecture

### Core Components
- **Document Ingestion**: PDF processing and text chunking using LangChain
- **Vector Database**: ChromaDB for efficient similarity search with cosine distance
- **Embeddings**: OpenAI `text-embedding-3-small` for semantic document representation
- **Language Model**: OpenAI GPT models for generating contextual responses
- **Prompt Engineering**: YAML-configured prompt templates for consistent responses

### Data Pipeline
1. **Ingestion Phase**: Load PDFs → Split into chunks (1000 chars, 200 overlap) → Generate embeddings → Store in ChromaDB
2. **Query Phase**: User question → Semantic search → Retrieve relevant chunks → Build context → Generate response

### Key Features
- **Semantic Search**: Cosine similarity-based retrieval with configurable thresholds
- **Configurable Prompts**: YAML-based prompt templates for different use cases
- **Interactive CLI**: Command-line interface for real-time Q&A
- **Source Attribution**: Responses include document source information
- **Modular Design**: Separate concerns for ingestion, retrieval, and response generation

## Quick Start

### Prerequisites
- Python 3.13+
- OpenAI API key
- uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mutaician/Tea-Knowledge-Assistant
   cd Tea-Knowledge-Assistant
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Data Preparation

**Ingest documents** into the vector database:
   ```bash
   uv run code/ingest.py
   ```

### Usage

**Start the interactive assistant:**
```bash
uv run code/main.py
```

**Example interaction:**
```
Welcome to Tea Knowledge Assistant that will answer any of your tea questions with high accuracy
(type 'exit' to quit)
Enter Question: What are the optimal conditions for tea cultivation in Kenya?
```

## Project Structure

```
tea-knowledge-assistant/
├── code/
│   ├── main.py              # Main application entry point
│   ├── ingest.py            # Document ingestion pipeline
│   ├── prompt_builder.py     # Prompt construction utilities
│   ├── utils.py             # Helper functions (PDF loading, YAML parsing)
│   ├── paths.py             # Directory path constants
│   ├── checktokens.py       # Token usage analysis
│   └── config/
│       └── prompt_config.yaml # LLM prompt configurations
├── data/                    # PDF documents directory
├── outputs/
│   └── vector_db/          # ChromaDB vector database
├── pyproject.toml          # Project dependencies and metadata
├── uv.lock                 # uv dependency lock file
├── .env                    # Environment variables (API keys)
└── README.md
```

## Configuration

### Prompt Configuration
The assistant uses YAML-based prompt templates located in `code/config/prompt_config.yaml`. The configuration includes:
- **Role definition**: Assistant's persona and expertise
- **Instruction set**: Task-specific guidelines
- **Output constraints**: Response formatting rules
- **Style guidelines**: Tone and structure preferences

### Vector Database Settings
- **Chunk size**: 1000 characters with 200 character overlap
- **Similarity threshold**: Configurable relevance filtering
- **Embedding model**: OpenAI text-embedding-3-small
- **Search space**: Cosine similarity



### Adding New Documents
1. Place new PDF files in the `data/` directory
2. Run the ingestion script: `uv run code/ingest.py`
3. The new documents will be automatically chunked and embedded

## Performance Considerations

- **Chunk Size**: 1000 characters balances context preservation with retrieval precision
- **Similarity Threshold**: 0.6-0.7 provides good balance between relevance and coverage
- **Top-K Retrieval**: Configurable number of chunks retrieved per query
- **Embedding Model**: text-embedding-3-small offers good performance-cost ratio

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## License & Data Usage

This project combines publicly available content from multiple sources:

**Wikipedia articles** (e.g., Tea, Tea production in Kenya) are used under the terms of the Creative Commons Attribution-ShareAlike 4.0 International License. Attribution is preserved as required.

**Tea Board of Kenya publications** (e.g., cultivation manuals, quality requirement notices, regulations) are included for educational and research purposes only. These documents are © Tea Board of Kenya unless otherwise stated. Redistribution or commercial use may require prior permission from the Tea Board.

**The code for this project is licensed under the MIT License**, unless specified otherwise.

## Acknowledgments

- **Data Sources**: Wikipedia and Tea Board of Kenya for comprehensive tea knowledge
- **Open Source Libraries**: LangChain, ChromaDB, and OpenAI for powering the RAG pipeline
- **uv**: Modern Python package management
