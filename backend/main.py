"""
Main FastAPI Application
Sanskrit Semantic Vector RAG System
"""
import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from dotenv import load_dotenv

# Support both direct script execution and module import
try:
    from sanskrit_processor import SanskritProcessor, SanskritSummarizer
    from vector_rag import EmbeddingManager, VectorStore, RAGPipeline
except ImportError:
    from .sanskrit_processor import SanskritProcessor, SanskritSummarizer
    from .vector_rag import EmbeddingManager, VectorStore, RAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Sanskrit Semantic Vector RAG",
    description="Retrieval-Augmented Generation system for Sanskrit text",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger.info("Initializing components...")
try:
    sanskrit_processor = SanskritProcessor()
    logger.info("✓ Sanskrit processor initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize Sanskrit processor: {e}")
    raise

try:
    summarizer = SanskritSummarizer(sanskrit_processor)
    logger.info("✓ Summarizer initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize summarizer: {e}")
    raise

try:
    embedding_manager = EmbeddingManager()
    logger.info("✓ Embedding manager initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize embedding manager: {e}")
    raise

try:
    # Create vector store directory if it doesn't exist
    vector_store_path = "vector_store"
    os.makedirs(vector_store_path, exist_ok=True)
    
    vector_store = VectorStore(embedding_manager)
    logger.info("✓ Vector store initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize vector store: {e}")
    raise

try:
    rag_pipeline = RAGPipeline(embedding_manager, vector_store)
    logger.info("✓ RAG pipeline initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize RAG pipeline: {e}")
    raise

# Load existing vector store if available
try:
    vector_store.load()
    logger.info("✓ Loaded existing vector store")
except Exception as e:
    logger.info(f"  No existing vector store to load: {e}")



# Pydantic Models
class TextRequest(BaseModel):
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "नमस्ते विश्वम्। यह एक संस्कृत पाठ है।"
            }
        }


class SummaryRequest(BaseModel):
    text: str
    num_sentences: int = 3
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Long Sanskrit text here...",
                "num_sentences": 3
            }
        }


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    
    class Config:
        schema_extra = {
            "example": {
                "query": "संस्कृत grammar",
                "k": 5
            }
        }


class DocumentRequest(BaseModel):
    content: str
    source: str = "unknown"
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Full document content...",
                "source": "shastras"
            }
        }


class ProcessingResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sanskrit Semantic Vector RAG",
        "version": "0.1.0"
    }


# Sanskrit Processing Endpoints
@app.post("/process/tokenize", response_model=ProcessingResponse)
async def tokenize_text(request: TextRequest):
    """Tokenize Sanskrit text"""
    try:
        tokens = sanskrit_processor.tokenize(request.text)
        return ProcessingResponse(
            success=True,
            message="Text tokenized successfully",
            data={"tokens": tokens, "count": len(tokens)}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/process/preprocess", response_model=ProcessingResponse)
async def preprocess_text(request: TextRequest):
    """Complete preprocessing: normalize, tokenize, remove stopwords"""
    try:
        processed = sanskrit_processor.preprocess(request.text)
        return ProcessingResponse(
            success=True,
            message="Text preprocessed successfully",
            data={"preprocessed_tokens": processed, "count": len(processed)}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/process/detect-script", response_model=ProcessingResponse)
async def detect_script(request: TextRequest):
    """Detect if text is in Devanagari script"""
    try:
        is_devanagari = sanskrit_processor.detect_devanagari(request.text)
        return ProcessingResponse(
            success=True,
            message="Script detected",
            data={"is_devanagari": is_devanagari}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/process/split-sentences", response_model=ProcessingResponse)
async def split_sentences(request: TextRequest):
    """Split text into sentences"""
    try:
        sentences = sanskrit_processor.split_sentences(request.text)
        return ProcessingResponse(
            success=True,
            message="Sentences split successfully",
            data={"sentences": sentences, "count": len(sentences)}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Summarization Endpoints
@app.post("/summarize", response_model=ProcessingResponse)
async def summarize_text(request: SummaryRequest):
    """Summarize Sanskrit text"""
    try:
        summary = summarizer.summarize(request.text, request.num_sentences)
        return ProcessingResponse(
            success=True,
            message="Text summarized successfully",
            data={"summary": summary}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Embedding Endpoints
@app.post("/embed/text", response_model=ProcessingResponse)
async def embed_text(request: TextRequest):
    """Generate embedding for text"""
    try:
        embedding = embedding_manager.embed_text(request.text)
        return ProcessingResponse(
            success=True,
            message="Text embedded successfully",
            data={
                "embedding_dimension": int(embedding.shape[0]),
                "embedding": embedding.tolist()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/embed/batch", response_model=ProcessingResponse)
async def embed_batch(texts: List[str]):
    """Generate embeddings for multiple texts"""
    try:
        embeddings = embedding_manager.embed_batch(texts)
        return ProcessingResponse(
            success=True,
            message="Texts embedded successfully",
            data={
                "count": len(texts),
                "embedding_dimension": int(embeddings.shape[1]),
                "embeddings": embeddings.tolist()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# RAG Endpoints
@app.post("/rag/ingest", response_model=ProcessingResponse)
async def ingest_document(request: DocumentRequest):
    """Ingest a document into the RAG system"""
    try:
        logger.info(f"Ingesting document from source: {request.source}")
        
        # Validate content is not empty
        if not request.content or not request.content.strip():
            raise ValueError("Document content cannot be empty")
        
        logger.info(f"Content length: {len(request.content)} chars")
        rag_pipeline.ingest_document(request.content, request.source)
        logger.info("✓ Document ingested successfully")
        
        return ProcessingResponse(
            success=True,
            message="Document ingested successfully",
            data={"source": request.source}
        )
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Error ingesting document: {error_msg}", exc_info=True)
        raise HTTPException(status_code=400, detail=error_msg)


@app.post("/rag/retrieve", response_model=ProcessingResponse)
async def retrieve_documents(request: QueryRequest):
    """Retrieve documents similar to query with source attribution"""
    try:
        results = rag_pipeline.retrieve(request.query, request.k)
        return ProcessingResponse(
            success=True,
            message="Documents retrieved successfully",
            data={
                "query": request.query,
                "results_count": len(results),
                "results": results
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rag/query", response_model=ProcessingResponse)
async def rag_query(request: QueryRequest):
    """Query the RAG system and get LLM-generated answer with source references"""
    try:
        logger.info(f"Query: {request.query}")
        
        # Use top_k=3 if not provided or invalid
        k = request.k if request.k and request.k > 0 else 3
        
        # Retrieve relevant documents
        retrieved = rag_pipeline.retrieve(request.query, k)
        
        # Generate answer using question + combined context
        answer = rag_pipeline.generate_answer(request.query, k)
        
        # Get clean source snippets
        snippets = rag_pipeline.get_source_snippets(retrieved)
        
        return ProcessingResponse(
            success=True,
            message="Query processed successfully",
            data={
                "query": request.query,
                "answer": answer,
                "sources": snippets,
                "results_count": len(retrieved)
            }
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# File upload endpoint
@app.post("/upload-document", response_model=ProcessingResponse)
async def upload_document(file: UploadFile = File(...), source: str = "uploaded"):
    """Upload and ingest a document (PDF or TXT)"""
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Validate file
        if not file or not file.filename:
            raise ValueError("No file provided")
        
        content = await file.read()
        logger.info(f"File size: {len(content)} bytes")
        
        if not content:
            raise ValueError("File is empty")
        
        # --- Helper: check if extracted text is actually readable ---
        def _is_readable_text(t: str) -> bool:
            """Return True if text looks like real readable content,
            not raw PDF binary / font metadata."""
            if not t or len(t.strip()) < 20:
                return False
            sample = t[:2000]
            # Count printable vs. non-printable characters
            printable = sum(1 for c in sample if c.isprintable() or c in '\n\r\t')
            ratio = printable / max(len(sample), 1)
            # Check for PDF-object markers that indicate raw binary
            garbage_markers = ['endobj', '/BaseFont', '/Subtype', '/Type/Font',
                               'stream\n', '/Filter', '/FlateDecode']
            garbage_hits = sum(1 for m in garbage_markers if m in sample)
            if garbage_hits >= 2:
                logger.warning(f"Text looks like raw PDF objects ({garbage_hits} markers found)")
                return False
            if ratio < 0.7:
                logger.warning(f"Text has low printable ratio: {ratio:.2f}")
                return False
            return True
        
        # Detect file type and extract text
        filename_lower = file.filename.lower()
        is_pdf = filename_lower.endswith('.pdf') or content.startswith(b'%PDF')
        text = None
        
        # Try PDF extraction first if it's a PDF
        if is_pdf:
            logger.info("Detected PDF file, extracting text...")
            text = None
            
            # Try PyMuPDF (fitz) first - most reliable for text extraction
            try:
                import fitz
                import io
                
                pdf_file = io.BytesIO(content)
                doc = fitz.open(stream=pdf_file, filetype="pdf")
                logger.info(f"PDF has {len(doc)} pages")
                
                text_parts = []
                for page_num, page in enumerate(doc):
                    try:
                        # Use get_text("text") for best extraction quality
                        page_text = page.get_text("text")
                        if page_text and page_text.strip():
                            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                            logger.info(f"Extracted {len(page_text)} chars from page {page_num + 1}")
                    except Exception as page_error:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {page_error}")
                
                if text_parts:
                    candidate = "\n\n".join(text_parts)
                    if _is_readable_text(candidate):
                        text = candidate
                        logger.info(f"✓ PyMuPDF extraction successful: {len(text)} total chars")
                    else:
                        logger.warning("PyMuPDF extracted text but it looks like binary/garbage")
                else:
                    logger.warning("No text extracted with PyMuPDF")
                    
            except ImportError:
                logger.warning("PyMuPDF not available, trying pypdf...")
                try:
                    import io
                    from pypdf import PdfReader
                    
                    pdf_file = io.BytesIO(content)
                    reader = PdfReader(pdf_file)
                    logger.info(f"PDF has {len(reader.pages)} pages")
                    
                    text_parts = []
                    for page_num, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                                logger.info(f"Extracted {len(page_text)} chars from page {page_num + 1}")
                        except Exception as page_error:
                            logger.warning(f"Could not extract text from page {page_num + 1}: {page_error}")
                    
                    if text_parts:
                        candidate = "\n\n".join(text_parts)
                        if _is_readable_text(candidate):
                            text = candidate
                            logger.info(f"✓ pypdf extraction successful: {len(text)} total chars")
                        else:
                            logger.warning("pypdf extracted text but it looks like binary/garbage")
                            
                except Exception as pypdf_error:
                    logger.warning(f"pypdf extraction failed: {pypdf_error}")
            
            # Try pdfplumber as fallback
            if text is None:
                logger.warning("Previous methods failed, trying pdfplumber...")
                try:
                    import io
                    import pdfplumber
                    
                    with pdfplumber.open(io.BytesIO(content)) as pdf:
                        logger.info(f"PDF has {len(pdf.pages)} pages")
                        text_parts = []
                        for page_num, page in enumerate(pdf.pages):
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                                logger.info(f"Extracted {len(page_text)} chars from page {page_num + 1}")
                        
                        if text_parts:
                            candidate = "\n\n".join(text_parts)
                            if _is_readable_text(candidate):
                                text = candidate
                                logger.info(f"✓ pdfplumber extraction successful: {len(text)} total chars")
                            else:
                                logger.warning("pdfplumber extracted text but it looks like binary/garbage")
                except Exception as pdfplumber_error:
                    logger.warning(f"pdfplumber extraction failed: {pdfplumber_error}")
            
            if text is None:
                logger.error("All PDF extraction methods failed to produce readable text")
                raise ValueError(
                    "Could not extract readable text from this PDF. "
                    "The PDF may be image-based (scanned). "
                    "Please try copy-pasting the text content instead."
                )
        
        # For non-PDF files, try text decoding
        if text is None and not is_pdf:
            logger.info("Attempting to decode as text file...")
            try:
                text = content.decode('utf-8')
                logger.info(f"✓ Decoded as UTF-8: {len(text)} chars")
            except UnicodeDecodeError:
                try:
                    text = content.decode('latin-1')
                    logger.info(f"✓ Decoded as Latin-1: {len(text)} chars")
                except Exception as decode_error:
                    logger.error(f"Could not decode file: {decode_error}")
                    raise ValueError("Could not read file content - not a valid text file")
        
        if not text or not text.strip():
            raise ValueError("File content is empty after extraction")
        
        # Final quality check
        if not _is_readable_text(text):
            raise ValueError(
                "Extracted content does not appear to be readable text. "
                "Please try a different file or paste the text content directly."
            )
        
        logger.info(f"Ingesting document from: {source or file.filename}")
        rag_pipeline.ingest_document(text, source=source or file.filename)
        logger.info("✓ Document uploaded and ingested successfully")
        
        return ProcessingResponse(
            success=True,
            message="Document uploaded and ingested successfully",
            data={"filename": file.filename, "size": len(text), "pages": len(text.split("--- Page"))}
        )
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Error uploading document: {error_msg}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Upload failed: {error_msg}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Sanskrit Semantic Vector RAG",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "processing": "/process/*",
            "summarization": "/summarize",
            "embeddings": "/embed/*",
            "rag": "/rag/*"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
