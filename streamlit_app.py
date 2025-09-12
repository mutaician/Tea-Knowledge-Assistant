import streamlit as st
import os
from pathlib import Path

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

# Main content
st.markdown('<h1 class="main-header">🍵 Tea Knowledge Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your AI-powered guide to tea cultivation, processing, and knowledge</p>', unsafe_allow_html=True)

# System status check
status = check_system_status()

# API Key Setup Instructions
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
    if st.button("🔄 Process Documents", type="primary"):
        st.info("Document processing will be implemented in the next step!")

else:
    st.markdown("---")
    st.markdown("### ⚠️ Setup Required")
    st.warning("Please complete the setup steps above to continue.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and powered by RAG technology*")
