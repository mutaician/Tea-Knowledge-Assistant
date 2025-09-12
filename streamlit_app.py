import streamlit as st
import os
import sys
from pathlib import Path

# Add the code directory to Python path for imports
sys.path.append(str(Path(__file__).parent / "code"))

# Import project modules
try:
    from ingest import main as ingest_main
    from main import respond_to_query, retrieve_relevant_documents
    from utils import load_yaml_config
    from paths import PROMPT_CONFIG_FPATH
    INGEST_AVAILABLE = True
    MAIN_AVAILABLE = True
except ImportError as e:
    INGEST_AVAILABLE = False
    MAIN_AVAILABLE = False
    st.error(f"Could not import required modules: {e}. Please check the code directory.")

# Page configuration
st.set_page_config(
    page_title="Tea Knowledge Assistant",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tea theme
st.markdown("""
<style>
    .main-header {
        color: #2E8B57;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        color: #3CB371;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        background-color: rgba(46, 139, 87, 0.1);
        border: 2px solid #2E8B57;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: inherit;
    }
    .status-success {
        color: #2E8B57;
        font-weight: bold;
    }
    .status-warning {
        color: #FF6B35;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #3CB371;
    }
</style>
""", unsafe_allow_html=True)

def check_system_status():
    """Check API key and vector database status"""
    status = {}

    # Check API key
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY' in content and len(content.strip()) > 0:
                status['api_key'] = True
            else:
                status['api_key'] = False
    else:
        status['api_key'] = False

    # Check vector database
    vector_db_path = Path('outputs/vector_db')
    status['vector_db'] = vector_db_path.exists() and len(list(vector_db_path.glob('*'))) > 0

    return status

def process_documents():
    """Process documents using the existing ingestion pipeline"""
    if not INGEST_AVAILABLE:
        st.error("Ingestion module not available")
        return False

    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Starting document processing...")

        # Run ingestion (this might take time)
        progress_bar.progress(25)
        status_text.text("Loading and chunking documents...")

        progress_bar.progress(50)
        status_text.text("Generating embeddings...")

        # Call the ingestion function
        ingest_main()

        progress_bar.progress(100)
        status_text.text("Document processing complete!")

        # Clear progress indicators after a delay
        import time
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

        return True

    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        return False

def qa_interface():
    """Q&A interface for asking questions about tea"""
    if not MAIN_AVAILABLE:
        st.error("Q&A functionality not available")
        return

    # Load prompt config
    try:
        prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        rag_prompt = prompt_config["rag_assistant_prompt"]
    except Exception as e:
        st.error(f"Could not load prompt configuration: {e}")
        return

    st.markdown("### 💬 Ask a Question")
    st.markdown("Ask any question about tea cultivation, processing, or related topics!")

    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What are the optimal conditions for tea cultivation in Kenya?",
        key="question_input"
    )

    # Submit button
    if st.button("🔍 Get Answer", type="primary") and question.strip():
        try:
            with st.spinner("Searching documents and generating answer..."):
                # Get relevant documents (only once)
                relevant_docs = retrieve_relevant_documents(question, top_k=3, threshold=0.6)

                # Generate response using the retrieved documents (no double retrieval)
                response = respond_to_query(question, rag_prompt, n_results=3, threshold=0.6, documents=relevant_docs)

            # Display results
            st.markdown("---")

            # Show retrieved documents
            if relevant_docs:
                st.markdown("### 📄 Relevant Information Found")
                with st.expander("View source documents", expanded=False):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Document {i}:**")
                        # Show first 500 characters of each document
                        preview = doc[:500] + "..." if len(doc) > 500 else doc
                        st.text_area(f"Document {i} preview", preview, height=100, disabled=True)
            else:
                st.warning("No relevant documents found for this question.")

            # Show AI answer
            st.markdown("### 🤖 Answer")
            st.markdown(str(response))

        except Exception as e:
            st.error("❌ Failed to process your question")
            with st.expander("Error Details", expanded=False):
                st.code(f"Error: {str(e)}")
            st.info("💡 **Troubleshooting tips:**\n"
                   "• Check your OpenAI API key is valid\n"
                   "• Ensure documents are properly processed\n"
                   "• Try refreshing the page\n"
                   "• Check your internet connection")

    elif not question.strip() and st.session_state.get('question_input', '').strip():
        st.info("Please enter a question to get started!")

# Main content
st.markdown('<h1 class="main-header">🍵 Tea Knowledge Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your AI-powered guide to tea cultivation, processing, and knowledge</p>', unsafe_allow_html=True)

# Initialize session state for processing status
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# System status check
status = check_system_status()

# Update status if processing was completed
if st.session_state.processing_complete:
    status = check_system_status()  # Re-check after processing
    if status['vector_db']:
        st.session_state.processing_complete = False  # Reset flag

# API Key Setup Instructions (only show if API key is missing)
if not status['api_key']:
    st.markdown("""
    <div class="status-box">
    <h3>🔑 Setup Instructions</h3>
    <p><strong>To use this application, you need to add your OpenAI API key:</strong></p>
    <ol>
    <li>Create a <code>.env</code> file in the project root directory</li>
    <li>Add your API key: <code>OPENAI_API_KEY=your_api_key_here</code></li>
    <li>Restart the application</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# System Status
st.markdown("### 📊 System Status")

col1, col2 = st.columns(2)

with col1:
    if status['api_key']:
        st.success("✅ OpenAI API Key: Found")
    else:
        st.error("❌ OpenAI API Key: Missing")

with col2:
    if status['vector_db']:
        st.success("✅ Document Database: Ready")
    else:
        st.warning("⚠️ Document Database: Needs Processing")

# Navigation logic
all_ready = status['api_key'] and status['vector_db']

if all_ready:
    st.markdown("---")
    st.markdown("### 🎯 Ready to Ask Questions!")
    st.success("All systems are ready! You can now ask questions about tea.")

    # Q&A Interface
    qa_interface()

elif status['api_key'] and not status['vector_db']:
    st.markdown("---")
    st.markdown("### 📚 Document Processing Needed")
    st.info("Your API key is configured, but the documents need to be processed first.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents... This may take a few minutes."):
                success = process_documents()
                if success:
                    st.session_state.processing_complete = True
                    st.success("✅ Documents processed successfully!")
                    st.rerun()  # Refresh the page to update status
                else:
                    st.error("❌ Document processing failed. Please check the logs.")

    with col2:
        st.markdown("""
        **What happens during processing:**
        - 📄 Load PDF documents from `data/` folder
        - ✂️ Split documents into smaller chunks
        - 🧠 Generate embeddings using OpenAI
        - 💾 Store in vector database for fast retrieval
        """)

else:
    st.markdown("---")
    st.markdown("### ⚠️ Setup Required")
    st.warning("Please complete the setup steps above to continue.")

# Footer
st.markdown("---")
st.markdown("🍵 **Tea Knowledge Assistant** | Built with Streamlit & RAG Technology")
st.markdown("*Demonstrating retrieval-augmented generation for agricultural knowledge*")
