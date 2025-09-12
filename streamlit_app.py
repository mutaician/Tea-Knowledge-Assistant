import streamlit as st
import os
import sys
from pathlib import Path

# Add the code directory to Python path for imports
sys.path.append(str(Path(__file__).parent / "code"))

# Import project modules
try:
    from ingest import main as ingest_main
    INGEST_AVAILABLE = True
except ImportError:
    INGEST_AVAILABLE = False
    st.error("Could not import ingestion module. Please check the code directory.")

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
    st.markdown("### 🎯 Ready to Start!")
    st.success("All systems are ready! You can now ask questions about tea.")
    if st.button("🚀 Continue to Q&A", type="primary"):
        st.info("Q&A interface will be implemented in the next step!")

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
st.markdown("*Built with Streamlit and powered by RAG technology*")
