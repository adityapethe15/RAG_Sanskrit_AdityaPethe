"""
Sanskrit Semantic Vector RAG - Streamlit Frontend
Simplified RAG-Only Interface
"""
import streamlit as st
import requests
import json
from typing import List
import os

# Configure page
st.set_page_config(
    page_title="Sanskrit Semantic Vector RAG",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# API URL configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def call_api(endpoint: str, method: str = "GET", data: dict = None):
    """Make API calls"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data, timeout=60)
        else:
            response = requests.get(url, timeout=60)
        
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.title("🕉️ Sanskrit RAG")
    st.write("Semantic Vector Retrieval-Augmented Generation System")
    
    # API Status
    api_health = check_api_health()
    status_icon = "🟢" if api_health else "🔴"
    st.write(f"{status_icon} API Status: {'Connected' if api_health else 'Disconnected'}")
    
    st.divider()
    st.write("**Configuration**")
    api_url_input = st.text_input("API URL", value=API_BASE_URL)
    if api_url_input != API_BASE_URL:
        os.environ["API_BASE_URL"] = api_url_input
    
    st.divider()
    st.write("**System Information**")
    st.write("• **Model**: Sentence Transformers (all-MiniLM-L6-v2)")
    st.write("• **Vector Store**: FAISS")
    st.write("• **Language**: Sanskrit + Transliteration")


# Main title
st.title("🕉️ Sanskrit Semantic Vector RAG")
st.write("**Retrieval-Augmented Generation for Sanskrit Texts**")
st.write("Upload documents → Ask questions → Get context-grounded answers with source references")

st.divider()

# Tab arrangement - Simplified
tab1, tab2 = st.tabs([
    "📥 Upload & Ingest Documents",
    "❓ Query Knowledge Base"
])

# ============================================================================
# Tab 1: Upload & Ingest
# ============================================================================
with tab1:
    st.header("Upload & Ingest Documents")
    st.write("Add documents to your knowledge base for semantic retrieval")
    
    st.divider()
    
    # Two sub-sections
    col1, col2 = st.columns(2, gap="large")
    
    # Section 1: Upload files
    with col1:
        st.subheader("📁 Upload File")
        st.write("Upload PDF or TXT files")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["txt", "pdf"],
            key="file_uploader",
            label_visibility="collapsed"
        )
        doc_name = st.text_input(
            "Document Identifier",
            value="document_1",
            help="Unique name for this document"
        )
        
        if uploaded_file:
            st.write(f"📄 Selected: **{uploaded_file.name}** ({uploaded_file.size/1024:.1f} KB)")
            
            if st.button("📤 Upload & Ingest", use_container_width=True):
                with st.spinner("Processing document..."):
                    files = {"file": uploaded_file}
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/upload-document",
                            files=files,
                            params={"source": doc_name},
                            timeout=50000
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Document successfully ingested!")
                            
                            data = result.get("data", {})
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Document", doc_name)
                            with col_b:
                                st.metric("Size", f"{data.get('size', 0)//1024} KB")
                            with col_c:
                                st.metric("Chunks", data.get('pages', 0))
                        else:
                            st.error(f"❌ Upload failed with status {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("ℹ️ Select a PDF or TXT file to upload")
    
    # Section 2: Paste text
    with col2:
        st.subheader("📝 Paste Text Content")
        st.write("Or paste document content directly")
        
        doc_content = st.text_area(
            "Document content",
            placeholder="Paste Sanskrit text, transliteration, or other content...",
            height=150,
            key="paste_content",
            label_visibility="collapsed"
        )
        doc_source = st.text_input(
            "Document Name",
            value="pasted_document",
            help="Identifier for this document",
            key="paste_source"
        )
        
        if doc_content:
            st.write(f"📝 Content: **{len(doc_content)} characters**")
            
            if st.button("📥 Ingest Text", use_container_width=True):
                with st.spinner("Ingesting document..."):
                    result = call_api("/rag/ingest", "POST", {
                        "content": doc_content,
                        "source": doc_source
                    })
                    if result and result.get("success"):
                        st.success("✅ Document successfully ingested!")
                        st.write(f"Source: **{doc_source}**")
                    else:
                        st.error("❌ Failed to ingest document")
        else:
            st.info("ℹ️ Paste document content to ingest")


# ============================================================================
# Tab 2: Query
# ============================================================================
with tab2:
    st.header("Query Your Knowledge Base")
    st.write("Ask questions to retrieve relevant documents and get context-grounded answers")
    
    st.divider()
    
    # Query input section
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        query_text = st.text_area(
            "Enter your question",
            placeholder="Ask a question about your uploaded document...",
            height=100,
            key="query_input"
        )
        st.caption("💡 Ask specific questions about your uploaded document content")
    
    with col2:
        st.write("**Retrieval Settings**")
        k_results = st.slider(
            "Results to retrieve",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of document chunks to retrieve"
        )
    
    # Query button
    if st.button("🔍 Execute Query", use_container_width=True, type="primary"):
        if query_text:
            with st.spinner("Processing query and retrieving documents..."):
                result = call_api("/rag/query", "POST", {
                    "query": query_text,
                    "k": k_results
                })
                
                if result and result.get("success"):
                    data = result.get("data", {})
                    
                    st.divider()
                    
                    # ===== QUERY ECHO =====
                    st.markdown(f"**Query:** {data.get('query', query_text)}")
                    
                    st.write("")  # spacer
                    
                    # ===== ANSWER SECTION (MAIN) =====
                    st.markdown("### 📖 Answer")
                    answer = data.get('answer', '')
                    if answer:
                        # Display answer in a clean highlighted box
                        st.success(answer)
                    else:
                        st.warning("No answer could be generated for this query.")
                    
                    st.write("")  # spacer
                    
                    # ===== SOURCES SECTION — SHORT, INLINE =====
                    sources = data.get('sources', [])
                    
                    st.markdown("### 📚 Sources")
                    if sources:
                        for source in sources:
                            rank = source.get('rank', 0)
                            snippet = source.get('snippet', '')
                            st.markdown(f"**Rank {rank}** – {snippet}")
                    else:
                        st.markdown("*None*")
                    
                else:
                    st.error("❌ Query Failed - Please check API connection and try again.")
        else:
            st.warning("⚠️ Please enter a question")


# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; margin-top: 2rem; color: gray;'>
    <p><strong>Sanskrit Semantic Vector RAG System</strong></p>
    <p>Retrieval-Augmented Generation Pipeline</p>
    <p style='font-size: 0.85em;'>Built with FastAPI, Streamlit, FAISS, and Sentence Transformers</p>
    <p style='font-size: 0.9em; margin-top: 1rem;'>नमस्ते | स्वागतम्</p>
    </div>
""", unsafe_allow_html=True)
