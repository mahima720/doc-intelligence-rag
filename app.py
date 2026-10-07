import os
import faiss
import numpy as np
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# App Configuration
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .response-card {
        border-radius: 8px;
        padding: 1.2rem;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Client Initialization
@st.cache_resource
def get_genai_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API Key not found. Please define GOOGLE_API_KEY in your .env file.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_genai_client()

# 2. Text Extraction
def extract_pdf_text(pdf_docs):
    combined_text = ""
    for pdf in pdf_docs:
        pdf_bytes = pdf.getvalue()
        try:
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    page_text = page.get_text()
                    if page_text:
                        combined_text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return combined_text.strip() if combined_text.strip() else None

# 3. Chunking
def chunk_text(text, chunk_size=1000, overlap=200):
    if not text:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        segment = text[i : i + chunk_size]
        if len(segment.strip()) > 10:
            chunks.append(segment)
    return chunks

# 4. Vector Store Creation
def build_vector_store(text_chunks):
    if not text_chunks:
        return None, None
        
    try:
        raw_embeddings = []
        batch_size = 32

        progress_bar = st.progress(0, text="Generating embeddings...")
        total_chunks = len(text_chunks)

        for i in range(0, total_chunks, batch_size):
            batch = text_chunks[i : i + batch_size]
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            raw_embeddings.extend([e.values for e in result.embeddings])
            
            percent_complete = min(1.0, (i + batch_size) / total_chunks)
            progress_bar.progress(percent_complete, text=f"Embedding chunks: {min(i + batch_size, total_chunks)}/{total_chunks}")
            
        progress_bar.empty()

        embeddings = np.array(raw_embeddings, dtype=np.float32)
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, text_chunks
    except Exception as e:
        st.error(f"Failed to generate embeddings: {e}")
        return None, None

# 5. Question Answering
def answer_question(user_question, index, chunks):
    try:
        query_result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=user_question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = np.array([query_result.embeddings[0].values], dtype=np.float32)
        
        # Retrieve top relevant passages
        k = min(5, len(chunks))
        distances, indices = index.search(query_vector, k=k)
        
        context = "\n---\n".join([chunks[i] for i in indices[0] if 0 <= i < len(chunks)])
        
        prompt = f"""
Answer the question based strictly on the provided context.
If the information is not present in the context, explicitly respond: 'The answer is not available in the provided documents.'

Context:
{context}

Question: {user_question}
"""
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            response_stream = client.models.generate_content_stream(
                model=GENERATION_MODEL, 
                contents=prompt
            )
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            
    except Exception as e:
        st.error(f"Search failed: {e}")

def main():
    # Session state initialization
    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.chunks = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    # Sidebar controls
    with st.sidebar:
        st.title("Documents")
        st.caption("Upload source documents to build your knowledge base.")
        
        uploaded_files = st.file_uploader(
            "Select PDFs", 
            accept_multiple_files=True, 
            type=["pdf"],
            help="Upload one or more standard text-based PDF files."
        )
        
        process_clicked = st.button("Process & Index", type="primary", use_container_width=True)
        
        if process_clicked:
            if not uploaded_files:
                st.warning("Please upload at least one PDF file first.")
            else:
                with st.spinner("Extracting text and indexing content..."):
                    raw_text = extract_pdf_text(uploaded_files)
                    
                    if not raw_text:
                        st.error("No readable text found. Scanned image PDFs are not supported without OCR.")
                    else:
                        chunks = chunk_text(raw_text)
                        index, valid_chunks = build_vector_store(chunks)
                        
                        if index:
                            st.session_state.index = index
                            st.session_state.chunks = valid_chunks
                            st.session_state.processed_files = [f.name for f in uploaded_files]
                            st.success(f"Successfully indexed {len(valid_chunks)} chunks across {len(uploaded_files)} file(s).")

        if st.session_state.processed_files:
            st.divider()
            st.caption("Currently Indexed Files:")
            for name in st.session_state.processed_files:
                st.markdown(f"- 📄 `{name}`")

    # Main Area
    st.markdown('<div class="main-header">Document Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Ask questions and retrieve insights grounded strictly in your indexed documents.</div>', unsafe_allow_html=True)

    # Search Form with explicit button
    with st.form(key="search_form", clear_on_submit=False):
        col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
        
        with col1:
            query = st.text_input(
                "Your Question:", 
                placeholder="e.g., What are the key findings mentioned in section 2?",
                label_visibility="collapsed"
            )
        with col2:
            submit_search = st.form_submit_button("Search", type="primary", use_container_width=True)

    # Handling Query Execution
    if submit_search:
        if not query.strip():
            st.warning("Please enter a question before searching.")
        elif not st.session_state.index:
            st.info("No documents indexed yet. Upload and process documents from the sidebar to begin.")
        else:
            with st.chat_message("user"):
                st.write(query)
            answer_question(query, st.session_state.index, st.session_state.chunks)

if __name__ == "__main__":
    main()