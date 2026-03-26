# Sanskrit Semantic Vector RAG Project

Advanced Retrieval-Augmented Generation (RAG) system for Sanskrit text with semantic embeddings and intelligent summarization.

## 🎯 Features

- **Sanskrit Text Processing**: Tokenization, normalization, sentence segmentation, Devanagari detection
- **Semantic Embeddings**: Using transformer-based models (Sentence Transformers)
- **Vector Store**: Efficient semantic search with FAISS
- **Summarization**: Extractive summarization based on word frequency
- **RAG Pipeline**: Retrieve relevant documents and generate context
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Web UI**: Streamlit frontend for easy interaction

## 📁 Project Structure

```
Sanskrit_Summariser/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── sanskrit_processor.py   # Sanskrit NLP processing
│   ├── vector_rag.py          # Vector store & RAG pipeline
│   └── __init__.py
├── frontend/
│   └── app.py                  # Streamlit UI
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Windows/Mac/Linux

### Installation

1. **Clone or create the project directory**
```bash
cd Sanskrit_Summariser
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required NLTK data**
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. **Set up environment variables**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your API keys
# (For basic testing, you can skip this and use default models)
```

## 🏃 Running the Project

### Option 1: Run Both Backend and Frontend (Recommended)

**Terminal 1 - Start Backend API:**
```bash
cd backend
python main.py
```
The API will be available at: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
streamlit run app.py
```
The UI will open at: `http://localhost:8501`

### Option 2: Backend Only (For API Testing)

```bash
cd backend
python main.py
```

Use tools like **Postman** or **cURL** to test endpoints.

### Option 3: Frontend Only (With Remote Backend)

```bash
cd frontend

# Set API URL environment variable
set API_BASE_URL=http://your-backend-url:8000  # Windows
export API_BASE_URL=http://your-backend-url:8000  # Mac/Linux

streamlit run app.py
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Text Processing
```bash
POST /process/tokenize
POST /process/preprocess
POST /process/detect-script
POST /process/split-sentences
```

**Example:**
```bash
curl -X POST "http://localhost:8000/process/tokenize" \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते विश्वम्"}'
```

### Summarization
```bash
POST /summarize
```

**Request:**
```json
{
  "text": "Your long text here...",
  "num_sentences": 3
}
```

### Embeddings
```bash
POST /embed/text
POST /embed/batch
```

### RAG System
```bash
POST /rag/ingest
POST /rag/retrieve
POST /rag/query
POST /upload-document
```

## 🔄 Workflow

### 1. Ingest Documents
```python
# Via API
POST /rag/ingest
{
  "content": "Sanskrit text content...",
  "source": "shastras"
}

# Via Frontend: Go to RAG System → Ingest tab
```

### 2. Query the System
```python
# Via API
POST /rag/query
{
  "query": "Sanskrit grammar",
  "k": 5
}

# Returns:
# - Retrieved documents
# - Similarity scores
# - Generated context for LLM
```

### 3. Process Text
```python
# Tokenize
POST /process/tokenize

# Preprocess (with stopword removal)
POST /process/preprocess

# Get summary
POST /summarize
```

## 🧪 Testing

### Quick Test with Python

```python
import requests

API_URL = "http://localhost:8000"

# Test health
response = requests.get(f"{API_URL}/health")
print(response.json())

# Tokenize Sanskrit text
data = {"text": "नमस्ते विश्वम्"}
response = requests.post(f"{API_URL}/process/tokenize", json=data)
print(response.json())

# Ingest document
doc_data = {
    "content": "Long Sanskrit text here...",
    "source": "vedas"
}
response = requests.post(f"{API_URL}/rag/ingest", json=doc_data)
print(response.json())

# Query
query_data = {"query": "Sanskrit", "k": 5}
response = requests.post(f"{API_URL}/rag/query", json=query_data)
print(response.json())
```

## 🔧 Configuration

### Environment Variables (.env)

```
# API Keys (Optional for basic functionality)
OPENAI_API_KEY=your-key
HUGGING_FACE_API_KEY=your-key

# Embedding Model (Fixed - all-MiniLM-L6-v2)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384

# Port
PORT=8000
```

## 📚 How It Works

### 1. **Text Processing Pipeline**
- Input Sanskrit text → Normalize → Tokenize → Remove stopwords → Output

### 2. **Embedding Generation**
- Text → Sentence-Transformers (all-MiniLM-L6-v2) → 384-dim vector

### 3. **Vector Storage**
- Embeddings stored in FAISS (Facebook AI Similarity Search)
- Fast semantic similarity search in milliseconds

### 4. **RAG Retrieval**
```
Query → Embed → Search Vector Store → Retrieve Top-K → Generate Context
```

### 5. **Summarization**
- Calculate word frequency → Score sentences → Return top N sentences

## 🛠️ Customization

### Change Embedding Model
Edit `backend/vector_rag.py`:
```python
embedding_manager = EmbeddingManager(
    model_name="sentence-transformers/multilingual-MiniLM-L12-v2"
)
```

### Add Custom Stopwords
Edit `backend/sanskrit_processor.py`:
```python
stopwords = {
    'iti', 'api', 'ca', 'yatha',
    # Add your custom stopwords
}
```

### Integrate LLM
Add to `backend/main.py`:
```python
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
response = llm(context)
```

## 📊 Performance

- **Embeddings**: ~100ms per text
- **Indexing**: ~1000 docs/second
- **Retrieval**: ~10ms for semantic search
- **Summarization**: ~50ms

## 🐛 Troubleshooting

### API not starting
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Mac/Linux
```

### Streamlit not connecting to API
- Verify Backend is running: `http://localhost:8000/health`
- Check API_BASE_URL in sidebar
- Ensure CORS is enabled (it is by default)

### Memory issues with embeddings
- Reduce batch size
- Process documents in chunks
- Use GPU: Set `device='cuda'` in SentenceTransformer

## 📝 Example Usage Scenarios

### Scenario 1: Sanskrit Shastras Analysis
```bash
1. Ingest Vedas/Upanishads via document upload
2. Query: "प्रमाण" (proofs)
3. System retrieves relevant scriptures
4. Generates summary with context
```

### Scenario 2: Sanskrit Grammar Learning
```bash
1. Upload grammar texts
2. Query: "समास" (compounds)
3. Get relevant sections
4. Refer to web UI for learning
```

### Scenario 3: Classical Text Processing
```bash
1. Process Mahabharata/Ramayana
2. Query for character mentions
3. Extract summaries of episodes
4. Generate contextual information
```

## 🚀 Deployment

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "backend/main.py"]
```

### Heroku
```bash
git push heroku main
```

### AWS/GCP
Deploy FastAPI + Streamlit on serverless or container services

## 📖 Documentation

- [FastAPI Docs](http://localhost:8000/docs) - Interactive API documentation
- [Streamlit Guide](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Fine-tuned Sanskrit embedding models
- LLM integration (GPT, Llama)
- Multi-language support
- Advanced summarization algorithms
- UI improvements

## 📄 License

MIT License - feel free to use this project!

## 🙏 Credits

- Transformers: Hugging Face
- Embeddings: Sentence Transformers
- Vector Search: Facebook FAISS
- Framework: FastAPI, Streamlit

## 📧 Support

For issues or questions:
1. Check [troubleshooting](#troubleshooting)
2. Review API docs at `/docs`
3. Check logs in backend/frontend terminals

---

**नमस्ते! Happy Sanskrit Processing! 🕉️**
