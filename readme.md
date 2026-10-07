# Document Intelligence: Chat with Multiple PDFs 📄🤖

An enterprise-ready Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **Google Gemini**, and **FAISS**. This application ingests multiple PDF documents, segments text with sliding overlap, indexes dense vector embeddings locally in memory, and performs grounded question answering with context-enforced streaming responses.

---

## 🌟 Key Features

* **Multi-Document Ingestion**: Extracts text from multiple PDF documents simultaneously using PyMuPDF (`fitz`).
* **Context-Preserving Chunking**: Splits dense text into overlapping windows (1,000–1,500 characters with 200-character overlap) to prevent mid-sentence context loss.
* **Resilient Batched Embeddings**: Uses Google's `gemini-embedding-001` with request batching and automated exponential backoff to handle rate limits seamlessly.
* **Sub-Millisecond Vector Retrieval**: Employs local in-memory FAISS (`IndexFlatL2`) for fast, zero-database similarity search.
* **Hallucination-Guarded Answering**: Queries Google's `gemini-2.5-flash` model with strict system prompts that instruct the model to state if information is missing rather than fabricating answers.
* **Polished, Minimal UI**: Clean interface built with `st.form` controls, progress indicators, status badges, and conversational message containers without vendor clutter.

---

## 🏗️ System Architecture

```
[Uploaded PDFs]
       │
       ▼
 [PyMuPDF (fitz)] ──► Extracts clean text streams
       │
       ▼
 [Sliding Window Chunking] ──► Generates overlapping chunks (size: 1000, overlap: 200)
       │
       ▼
 [Google GenAI Embedding API] ──► Batched requests (32–64 items) with retry backoff
       │
       ▼
  [FAISS IndexFlatL2] ──► In-memory vector index stored in st.session_state
       │
       ▲
 [User Search Query]
       │
       ▼
 [Query Embedding] ──► FAISS Top-K Search (k=5) ──► Context Formulation ──► [Gemini 2.5 Flash] ──► Streamed Response
```

---

## 📋 Prerequisites

* **Python**: `3.10` or higher
* **Google AI Studio API Key**: Free tier or paid key from [Google AI Studio](https://aistudio.google.com/)

---

## 🚀 Installation & Setup

### 1. Clone or Open Project Directory

```bash
cd "C:\Projects\Chat with Multiple PDF Gemini"
```

### 2. Configure Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit pymupdf faiss-cpu numpy google-genai python-dotenv
```

### 4. Configure Environment Variables

Create a `.env` file in the root project folder:

```env
GOOGLE_API_KEY=AIzaSyYourActualKeyHere
```

> **Security Note:** Never commit your `.env` file to version control. Ensure `.env` is listed in your `.gitignore` file.

---

## 🖥️ Running the Application

Always run the app using Python's module runner to prevent virtual environment wrapper path errors:

```bash
python -m streamlit run app.py
```

Access the web interface at `http://localhost:8501`.

---

## 📖 Usage Instructions

1. **Upload Documents**: Open the left sidebar, click **Select PDFs**, and select one or more text-based PDFs.
2. **Process & Index**: Click **Process & Index**. A progress bar tracks the embedding creation across batches.
3. **Ask Questions**: Enter your question in the search input and click **Search** (or press **Enter**).
4. **Inspect Answers**: The assistant streams answers retrieved strictly from your uploaded files. If a fact cannot be found in the documents, it will explicitly notify you.

---

## ⚙️ Configuration Parameters

| Parameter | Default | Location | Description |
| :--- | :--- | :--- | :--- |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Header | Vector embedding model |
| `GENERATION_MODEL` | `gemini-2.5-flash` | Header | LLM used for question answering |
| `chunk_size` | `1000` | `chunk_text()` | Character count per document segment |
| `overlap` | `200` | `chunk_text()` | Character overlap between consecutive segments |
| `batch_size` | `64` | `build_vector_store()` | Number of chunks per embedding call to prevent quota overflow |
| `k` | `5` | `answer_question()` | Number of nearest neighbors retrieved from FAISS |

---

## 🧠 Key Learnings & Engineering Takeaways

### 1. API Evolution & SDK Shifts
* **Unified SDK Migration**: Moving from `google-generativeai` to the unified `google-genai` SDK changes data contracts. Embeddings now return structured `ContentEmbedding` objects where raw floats must be explicitly extracted via `[e.values for e in result.embeddings]`.
* **Model ID Deprecations**: Legacy embedding endpoints like `text-embedding-004` return `404 NOT_FOUND` under certain SDK endpoint versions. Using `gemini-embedding-001` or validating available capabilities via `client.models.list()` prevents routing failures.

### 2. Quota Management & Ingestion Resilience
* **Free Tier Rate Limits (`429 RESOURCE_EXHAUSTED`)**: Embedding multi-page PDFs chunk-by-chunk triggers per-minute request limits. Batching chunks (32–64 per request) reduces total API calls by up to 98%.
* **Exponential Backoff**: Transient quota spikes should be intercepted with automated retry logic (`time.sleep`) rather than failing the upload flow.

### 3. Streamlit State & Caching Hazards
* **Avoiding `@st.cache_resource` on Dynamic FAISS Indexes**: Passing unhashable chunk arrays into cached functions causes hashing exceptions, and caching raw C++ FAISS memory pointers can cause state corruption across browser sessions. Retaining indexes inside `st.session_state` provides clean per-session isolation.
* **Form-Based Execution**: Wrapping queries inside `st.form` prevents Streamlit from re-running the entire script on every keystroke, executing inference only on deliberate submission.

### 4. Proactive API Key Protection
* Cloud providers employ automated scanners that continuously monitor public repositories. Exposing an API key triggers instant, irreversible revocation (`403 PERMISSION_DENIED: Your API key was reported as leaked`). API keys must be isolated in `.env` files and guarded by `.gitignore`.

### 5. Launcher Integrity in Python Virtual Environments
* Moving or renaming project directories invalidates hardcoded absolute paths inside `venv\Scripts\*.exe` launchers (causing `Fatal error in launcher`). Launching through `python -m streamlit run app.py` dynamically resolves against the current interpreter path and prevents this failure.

---

## 📊 Conclusions

1. **Chunking Governs Both Recall and Ingestion Overhead**: Chunk sizing is an architectural trade-off. Overly small chunks multiply API requests and fragment context; overly large chunks dilute vector specificity. A window of 1,000–1,500 characters with 200-character overlap strikes an optimal balance for document QA.
2. **In-Memory Vector Stores Suit Ephemeral Document Workflows**: For interactive, session-scoped document review, lightweight in-memory FAISS structures eliminate the deployment, connection, and operational complexity of external vector databases.
3. **Defensive Streaming Controls Prevent UI Crashes**: When streaming generative model output via `generate_content_stream`, token chunks containing metadata, safety notifications, or empty contents must be guarded with `if chunk.text:` to prevent string concatenation runtime exceptions.

---

## 🛠️ Troubleshooting Matrix

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `400 INVALID_ARGUMENT (API_KEY_INVALID)` | Formatting error or malformed characters in `.env` | Ensure `.env` contains `GOOGLE_API_KEY=AIzaSy...` without spaces or quotes, and clear Streamlit cache. |
| `403 PERMISSION_DENIED (Leaked Key)` | Key was committed publicly and permanently revoked | Generate a new key in Google AI Studio, update `.env`, and add `.env` to `.gitignore`. |
| `404 NOT_FOUND (Model)` | Model name unsupported on the active API endpoint | Set embedding model to `gemini-embedding-001` or check supported models using `client.models.list()`. |
| `429 RESOURCE_EXHAUSTED` | Exceeded Free Tier requests per minute | Ingest embeddings in larger batches (`batch_size = 64`) and allow the exponential retry backoff to complete. |
| `Fatal error in launcher` | Virtual environment directory was moved or renamed | Run using `python -m streamlit run app.py`. |
| `No text found in PDF` | Scanned image PDF without embedded text layer | Use standard selectable PDFs or run an OCR preprocessing pipeline (e.g., Tesseract). |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).