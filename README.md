# Sanskrit Document Retrieval-Augmented Generation (RAG) System

An end-to-end **CPU-based Retrieval-Augmented Generation (RAG) system** designed to process and answer queries from Sanskrit documents using semantic search and lightweight language models.

---

## 🎯 Objective

This project implements a complete RAG pipeline that:

* Ingests Sanskrit documents (TXT/PDF)
* Processes and indexes them for semantic retrieval
* Accepts user queries (English / Sanskrit / mixed)
* Retrieves relevant context
* Generates context-aware answers using a lightweight CPU-based LLM

The system follows a modular RAG architecture separating retrieval and generation components.

---

## ⚙️ System Architecture

User Query
→ Embedding Generation
→ FAISS Vector Retrieval
→ Top-K Context Selection
→ Context + Query Fusion
→ LLM-based Answer Generation

---

## 🚀 Features

* **Sanskrit Text Processing**

  * Tokenization, normalization, sentence segmentation
  * Devanagari script handling

* **Semantic Retrieval**

  * Embeddings using multilingual Sentence Transformers
  * Fast similarity search using FAISS

* **RAG Pipeline**

  * Context-based retrieval
  * Query-aware answer generation (designed to avoid simple summarization)

* **LLM-Based Answer Generation**

  * Uses FLAN-T5 (CPU-friendly)
  * Generates short, relevant answers (2–3 lines)

* **Frontend + API**

  * FastAPI backend
  * Streamlit UI for interaction

---

## 🧠 Tech Stack

* Backend: FastAPI
* Frontend: Streamlit
* Embeddings: sentence-transformers (multilingual MiniLM)
* Vector Store: FAISS
* LLM: google/flan-t5-small
* Language: Python

---

## 📁 Project Structure

backend/
├── main.py                # FastAPI server
├── vector_rag.py         # RAG pipeline (retrieval + generation)
├── sanskrit_processor.py # Sanskrit preprocessing

frontend/
├── app.py                # Streamlit UI

config.py
requirements.txt
README.md

---

## 🏃 How to Run

### 1. Create Virtual Environment

python -m venv venv
venv\Scripts\activate   (Windows)

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Start Backend

cd backend
python main.py

### 4. Start Frontend

cd frontend
streamlit run app.py

---

## 🔄 Workflow

### Step 1: Ingest Documents

* Upload or paste Sanskrit text
* System splits into chunks
* Converts into embeddings
* Stores in FAISS vector database

### Step 2: Query System

* User enters a question
* Query is converted to embedding
* Top-K relevant chunks are retrieved

### Step 3: Answer Generation

* Retrieved context + query passed to LLM
* LLM generates a concise answer
* Sources are returned for transparency

---

## 🧪 Example Usage

### Step 1: Ingest Document

Upload or paste text such as:

Kalidasa was a famous Sanskrit poet and playwright known for works like Abhijnanasakuntalam.

---

### Step 2: Ask Query

Example queries:

* Who is Kalidasa?
* What is the document about?

---

### Step 3: Expected Output

Kalidasa is a famous Sanskrit poet and playwright known for his literary works.

---

## 📌 Important Design Decisions

* FAISS → Fast and efficient vector search on CPU
* Multilingual Embeddings → Supports Sanskrit + English queries
* FLAN-T5 → Lightweight, instruction-following LLM for CPU
* Chunking Strategy → Improves retrieval accuracy

---

## ⚠️ Limitations

* Sanskrit embeddings are not highly optimized (limited pretrained support)
* Similarity scores may be lower for Sanskrit queries
* Performance is better with English or mixed queries
* Pure Sanskrit query understanding may be weaker

---

## 🔮 Future Improvements

* Use domain-specific Sanskrit embeddings
* Add translation layer (Sanskrit → English → RAG)
* Improve ranking with hybrid retrieval
* Upgrade to stronger LLM (if GPU available)

---

## 📊 Performance

* Fully CPU-based (no GPU required)
* Fast retrieval using FAISS
* Lightweight inference with FLAN-T5

---

## ✅ Assignment Compliance

✔ End-to-end RAG pipeline
✔ Sanskrit document ingestion
✔ Semantic retrieval
✔ LLM-based answer generation
✔ CPU-only execution
✔ Modular architecture
✔ Clear setup and execution instructions

---

## 🙏 Conclusion

This project demonstrates a complete implementation of a **Retrieval-Augmented Generation system** for Sanskrit documents, focusing on modular design, efficient CPU execution, and practical handling of low-resource language challenges.

---

**नमस्ते 🙏**
