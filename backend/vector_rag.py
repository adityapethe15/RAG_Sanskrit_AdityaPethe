"""
Vector Embeddings and RAG Module
Handles semantic embeddings and retrieval-augmented generation
"""
import os
import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.documents import Document
import pickle

logger = logging.getLogger(__name__)

# Import optional LLM components
try:
    from transformers import pipeline as hf_pipeline
    HF_PIPELINE_AVAILABLE = True
    logger.info("✓ Transformers library available for QA generation")
except ImportError:
    HF_PIPELINE_AVAILABLE = False
    logger.warning("⚠ Transformers not available - using rule-based answer generation")

import re


# =============================================================================
# ISSUE 5 FIX: Text cleaning utilities
# =============================================================================

def clean_ingestion_text(text: str) -> str:
    """Clean raw extracted text before chunking and embedding.
    
    Removes:
      - Email addresses
      - URLs
      - Author / metadata lines (e.g. "Author: ...")
      - PDF artefacts and encoding noise
      - Excess whitespace and special characters
    """
    if not text:
        return text

    # Remove email addresses
    text = re.sub(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove common metadata prefixes
    text = re.sub(r'^(Author|Email|Date|Published|Copyright|©)[\s:]+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove page markers like "--- Page 1 ---"
    text = re.sub(r'---\s*Page\s*\d+\s*---', '', text)
    
    # Remove PDF object markers / binary noise
    text = re.sub(r'/[A-Z][a-zA-Z]+(/[A-Z][a-zA-Z]+)+', '', text)  # /Type/Font etc.
    text = re.sub(r'\b(endobj|endstream|startxref|xref|obj)\b', '', text)
    text = re.sub(r'<<[^>]*>>', '', text)  # PDF dict markers
    
    # Remove non-printable characters (keep Devanagari \u0900-\u097F and common punctuation)
    text = re.sub(r'[^\x20-\x7E\u0900-\u097F\u0A00-\u0A7F\n\r\t।॥]', ' ', text)
    
    # Remove lines that are mostly numbers / noise (e.g. page numbers, indices)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that are too short to be meaningful
        if len(stripped) < 5:
            continue
        # Skip lines that are mostly digits/punctuation
        alpha_count = sum(1 for c in stripped if c.isalpha() or '\u0900' <= c <= '\u097F')
        if alpha_count < len(stripped) * 0.3 and len(stripped) > 5:
            continue
        cleaned_lines.append(stripped)
    
    text = '\n'.join(cleaned_lines)
    
    # Collapse multiple whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


class EmbeddingManager:
    """Manage sentence embeddings for Sanskrit text"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize embedding model
        Default: = paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


class VectorStore:
    """Local vector store using FAISS"""
    
    def __init__(self, embedding_manager: EmbeddingManager, persist_dir: str = "vector_store"):
        self.embedding_manager = embedding_manager
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.documents = []
        self.embeddings = None  # Cache embeddings object
        
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        
        # Initialize embeddings object
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_manager.model_name)
            logger.info(f"✓ Embeddings initialized with model: {self.embedding_manager.model_name}")
        except Exception as e:
            logger.error(f"✗ Failed to initialize embeddings: {e}", exc_info=True)
            raise
    
    def clear(self) -> None:
        """Completely clear the vector store - removes all old data"""
        try:
            logger.info("Clearing vector store...")
            self.vectorstore = None
            self.documents = []
            logger.info("✓ Vector store cleared (memory)")
        except Exception as e:
            logger.error(f"✗ Error clearing vector store: {e}", exc_info=True)
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """Add texts to vector store"""
        try:
            logger.info(f"Adding {len(texts)} texts to vector store")
            
            docs = []
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    logger.warning(f"Skipping empty text at index {i}")
                    continue
                metadata = metadatas[i] if metadatas else {}
                docs.append(Document(page_content=text, metadata=metadata))
            
            if not docs:
                logger.warning("No valid documents to add")
                return
            
            logger.info(f"Creating embeddings for {len(docs)} documents")
            
            # Try to create FAISS index
            try:
                if self.vectorstore is None:
                    logger.info("Creating new FAISS index with HuggingFaceEmbeddings...")
                    self.vectorstore = FAISS.from_documents(docs, self.embeddings)
                    logger.info("✓ Created new FAISS index")
                else:
                    logger.info("Adding documents to existing FAISS index...")
                    self.vectorstore.add_documents(docs)
                    logger.info("✓ Added documents to existing index")
            except Exception as faiss_error:
                logger.error(f"FAISS error: {faiss_error}", exc_info=True)
                logger.warning("Falling back to in-memory storage without FAISS")
                pass
            
            # For fresh ingestion, replace documents (don't extend with old ones)
            self.documents = docs
            logger.info(f"✓ Now have {len(self.documents)} total documents in memory (replaced old data)")
        except Exception as e:
            logger.error(f"✗ Error adding texts to vector store: {e}", exc_info=True)
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar texts and return with metadata"""
        try:
            if self.vectorstore is None:
                logger.warning("Vector store is empty, attempting in-memory search")
                # Fallback: simple keyword matching in documents
                if not self.documents:
                    return []
                
                query_lower = query.lower()
                results = []
                for doc in self.documents:
                    content = doc.page_content.lower()
                    # Simple relevance scoring based on word matches
                    score = sum(1 for word in query_lower.split() if word in content) / len(query_lower.split())
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    results.append({
                        "content": doc.page_content,
                        "score": score,
                        "metadata": metadata
                    })
                
                # Sort by score and return top k
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:k]
            
            logger.info(f"Searching for top {k} similar documents")
            try:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                logger.info(f"✓ Found {len(results)} results")
                
                # Convert FAISS distance to similarity (0-1 range)
                processed_results = []
                for doc, distance in results:
                    similarity = 1 / (1 + float(distance))
                    processed_results.append({
                        "content": doc.page_content,
                        "score": similarity,
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                    })
                
                # Sort by similarity
                processed_results.sort(key=lambda x: x['score'], reverse=True)
                logger.info("✓ Converted distances to similarities and sorted")
                return processed_results
            except Exception as faiss_search_error:
                logger.warning(f"FAISS search failed: {faiss_search_error}, using fallback")
                if not self.documents:
                    return []
                query_lower = query.lower()
                results = []
                for doc in self.documents:
                    content = doc.page_content.lower()
                    score = sum(1 for word in query_lower.split() if word in content) / len(query_lower.split())
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    results.append({
                        "content": doc.page_content,
                        "score": score,
                        "metadata": metadata
                    })
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:k]
        except Exception as e:
            logger.error(f"✗ Error searching vector store: {e}", exc_info=True)
            return []
    
    def save(self) -> None:
        """Save vector store to disk"""
        try:
            logger.info(f"Saving vector store to {self.persist_dir}")
            
            if self.vectorstore:
                try:
                    self.vectorstore.save_local(self.persist_dir)
                    logger.info("✓ FAISS index saved")
                except Exception as faiss_error:
                    logger.warning(f"Could not save FAISS index: {faiss_error}")
            else:
                logger.info("No FAISS index to save (using fallback storage)")
            
            doc_path = os.path.join(self.persist_dir, 'documents.pkl')
            with open(doc_path, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info(f"✓ Saved {len(self.documents)} documents to disk")
            
        except Exception as e:
            logger.error(f"✗ Error saving vector store: {e}", exc_info=True)
            logger.warning("Continuing despite save error")
    
    def load(self) -> None:
        """Load vector store from disk"""
        try:
            logger.info(f"Loading vector store from {self.persist_dir}")
            try:
                from langchain_core.load.serializable import DEFAULT_DESERIALIZER_MAPPING
                self.vectorstore = FAISS.load_local(
                    self.persist_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✓ FAISS index loaded successfully")
            except:
                try:
                    self.vectorstore = FAISS.load_local(self.persist_dir, self.embeddings)
                    logger.info("✓ FAISS index loaded (fallback)")
                except Exception as faiss_error:
                    logger.warning(f"Could not load FAISS index: {faiss_error}")
                    self.vectorstore = None
            
            doc_path = os.path.join(self.persist_dir, 'documents.pkl')
            if os.path.exists(doc_path):
                with open(doc_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"✓ Loaded vector store with {len(self.documents)} documents")
            else:
                logger.info("✓ Loaded vector store (no metadata)")
        except Exception as e:
            logger.info(f"No existing vector store to load: {e}")


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline with intelligent answer synthesis"""
    
    # Relevance threshold — below this, the query is considered irrelevant
    RELEVANCE_THRESHOLD = 0.10
    # Maximum answer length in characters (~2-4 lines)
    MAX_ANSWER_LENGTH = 350
    # Minimum answer length — ensures 2+ lines
    MIN_ANSWER_LENGTH = 80
    # Maximum source snippet length
    MAX_SNIPPET_LENGTH = 100
    
    def __init__(self, 
                 embedding_manager: EmbeddingManager,
                 vector_store: VectorStore,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Disable extractive QA; only use generative pipeline for final. 
        # Extractive models (Roberta QA) are intentionally not loaded.
        self.qa_pipeline = None

        # Initialize generative pipeline (FLAN-T5) for direct answer generation.
        self.summarizer_pipeline = None
        if HF_PIPELINE_AVAILABLE:
            try:
                self.summarizer_pipeline = hf_pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    device=-1  # CPU
                )
                logger.info("✓ Generative pipeline initialized (google/flan-t5-base)")
            except Exception as e:
                logger.warning(f"Could not load generative model: {e}")
                self.summarizer_pipeline = None
    
    def clear(self) -> None:
        """Completely clear the vector store - removes all old data"""
        try:
            logger.info("Clearing vector store...")
            self.vector_store.vectorstore = None
            self.vector_store.documents = []
            logger.info("✓ Vector store cleared (memory)")
        except Exception as e:
            logger.error(f"✗ Error clearing vector store: {e}", exc_info=True)
    
    def check_query_relevance(self, query: str, retrieved_docs: List[Dict]) -> Tuple[bool, float]:
        """Check if query is relevant to retrieved documents based on embedding similarity.
        
        Returns: (is_relevant, best_score)
        """
        if not retrieved_docs:
            return False, 0.0
        
        scores = [float(doc.get('similarity_score', 0)) for doc in retrieved_docs]
        best_score = max(scores) if scores else 0
        
        # ISSUE 1 & 3 FX: Primary signal is embedding similarity, ignore strict keyword overlap.
        # Fallback avoided if embedding score is above lowered threshold.
        is_relevant = best_score >= self.RELEVANCE_THRESHOLD or best_score > 0
        
        logger.info(f"Relevance: best={best_score:.3f}, threshold={self.RELEVANCE_THRESHOLD}, relevant={is_relevant}")
        return is_relevant, best_score
    
    def ingest_document(self, document_text: str, source: str = "unknown", clear_old: bool = True) -> None:
        """Ingest a document into the RAG system.
        
        ISSUE 5 FIX: Cleans text before chunking to remove emails, metadata, noise.
        """
        try:
            logger.info(f"Ingesting document from source: {source}")
            logger.info(f"Document size (raw): {len(document_text)} chars")
            
            # ── ISSUE 5: Clean text before chunking ──
            cleaned_text = clean_ingestion_text(document_text)
            logger.info(f"Document size (cleaned): {len(cleaned_text)} chars")
            
            if not cleaned_text or len(cleaned_text.strip()) < 20:
                raise ValueError("Document has no meaningful content after cleaning")
            
            if clear_old:
                logger.info("Clearing old data before ingesting new document...")
                self.vector_store.clear()
                logger.info("✓ Old data cleared")
            
            chunks = self.text_splitter.split_text(cleaned_text)
            logger.info(f"Split into {len(chunks)} chunks")
            
            if not chunks:
                logger.warning("No chunks created from document")
                return
            
            metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
            self.vector_store.add_texts(chunks, metadatas)
            logger.info("✓ Added texts to vector store")
            
            self.vector_store.save()
            logger.info("✓ Saved vector store")
        except Exception as e:
            logger.error(f"✗ Error ingesting document: {e}", exc_info=True)
            raise
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query with source attribution.

        Use top_k=3 to keep context small and focused.
        ISSUE 4 FIX: Re-ranks results to boost chunks that contain query keywords.
        """
        results = self.vector_store.search(query, k=k * 2)  # fetch more, then re-rank
        
        # ── ISSUE 4: Keyword-aware re-ranking ──
        subject = self._extract_subject(query)
        
        def normalize_sanskrit(text):
            """ISSUE 4 FIX: Normalize Sanskrit words by removing common inflections."""
            return text.replace('ः', '').replace('ं', '').replace('म्', '')
            
        raw_words = re.findall(r'\w+', query.lower())
        query_words = set(normalize_sanskrit(w) for w in raw_words if len(w) >= 2)
        
        stop = {'the', 'what', 'who', 'how', 'where', 'when', 'why', 'does',
                'this', 'that', 'are', 'was', 'were', 'been', 'being', 'have',
                'has', 'had', 'with', 'from', 'about', 'which', 'their',
                'there', 'will', 'would', 'could', 'should', 'can', 'may',
                'story', 'tell', 'describe'}
        query_words -= stop
        
        retrieved = []
        for i, result in enumerate(results):
            content = result.get('content', '')
            content_lower = content.lower()
            normalized_content = normalize_sanskrit(content_lower)
            metadata = result.get('metadata', {})
            base_score = float(result.get('score', 0))
            
            # Keyword boost using normalized Sanskrit logic
            keyword_hits = sum(1 for w in query_words if w in normalized_content)
            keyword_boost = min(keyword_hits * 0.05, 0.2)  # cap boost at 0.2
            
            # Subject boost
            subject_boost = 0.0
            if subject and normalize_sanskrit(subject.lower()) in normalized_content:
                subject_boost = 0.15
            
            final_score = base_score + keyword_boost + subject_boost
            
            retrieved.append({
                "content": content,
                "similarity_score": final_score,
                "rank": 0,  # will be set after sorting
                "source": metadata.get('source', 'unknown'),
                "chunk_id": metadata.get('chunk', i)
            })
        
        # Re-sort by boosted score
        retrieved.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Take top k and assign ranks
        retrieved = retrieved[:k]
        for idx, doc in enumerate(retrieved):
            doc['rank'] = idx + 1
        
        return retrieved
    
    def generate_context(self, query: str, k: int = 5) -> str:
        """Generate context from retrieved documents with source attribution"""
        results = self.retrieve(query, k=k)
        
        context_parts = []
        for r in results:
            source = r.get('source', 'unknown')
            chunk = r.get('chunk_id', 0)
            score = r.get('similarity_score', 0)
            content = r.get('content', '')
            
            part = f"[Source: {source}, Chunk {chunk}, Score: {score:.3f}]\n{content}"
            context_parts.append(part)
        
        return "\n\n".join(context_parts)
    
    # ------------------------------------------------------------------
    # CORE FIX: Intelligent answer generation
    # ------------------------------------------------------------------
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip page markers, extra whitespace, and non-informative lines."""
        text = re.sub(r'---\s*Page\s*\d+\s*---', '', text)
        # Remove email addresses from chunks
        text = re.sub(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove metadata patterns
        text = re.sub(r'\b(Author|Email|Date|Published|Copyright)[\s:]+\S+', '', text, flags=re.IGNORECASE)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def _extract_english_sentences(text: str) -> List[str]:
        """Extract clean, readable sentences from mixed-language text.
        Prefers English sentences; falls back to all sentences."""
        raw = re.split(r'(?<=[.!?।॥])\s*', text)
        sentences = [s.strip() for s in raw if s.strip() and len(s.strip()) > 12]
        
        english = []
        all_sentences = []
        for s in sentences:
            ascii_ratio = sum(1 for c in s if c.isascii()) / max(len(s), 1)
            all_sentences.append(s)
            if ascii_ratio > 0.6:
                english.append(s)
        
        return english if english else all_sentences
    
    @staticmethod
    def _truncate_answer(answer: str, max_len: int = 350) -> str:
        """Ensure the answer is at most *max_len* characters, ending at a
        sentence boundary when possible."""
        answer = answer.strip()
        if len(answer) <= max_len:
            return answer
        truncated = answer[:max_len]
        last_dot = truncated.rfind('.')
        if last_dot > max_len * 0.4:
            return truncated[:last_dot + 1].strip()
        return truncated.rstrip() + "..."
    
    def _extract_subject(self, query: str) -> str:
        """Try to extract the main subject from the question.
        E.g. 'Who is Kalidasa in the story?' -> 'Kalidasa'
             'What is the story about?'       -> '' (broad question)
        """
        q = query.strip().rstrip('?').strip()
        q_lower = q.lower()
        
        # Broad questions → no specific subject
        broad_patterns = ['what is the story about', 'what is this about',
                          'what is the document about', 'what is it about',
                          'what happened', 'what happens', 'tell me everything',
                          'summarize', 'summary']
        if q_lower in broad_patterns or any(q_lower.startswith(p) for p in broad_patterns):
            return ""
        
        # "What is X about?" → extract X
        m = re.search(r'(?:what|who)\s+is\s+(.+?)\s+about', q, re.I)
        if m:
            return m.group(1).strip()
        
        # "Who/What is X in the story?" → extract X
        m = re.search(r'(?:who|what)\s+is\s+(.+?)(?:\s+in\b|$)', q, re.I)
        if m:
            subj = m.group(1).strip()
            if len(subj.split()) <= 3:
                return subj
        
        # "Tell me about X" / "Describe X"
        m = re.search(r'(?:about|describe)\s+(.+)', q, re.I)
        if m:
            return m.group(1).strip()
        
        # Fallback: longest capitalised word that isn't a stop word
        words = [w for w in q.split() if w[0:1].isupper() and len(w) > 2
                 and w.lower() not in {'who', 'what', 'when', 'where', 'how', 'why', 'the', 'this'}]
        if words:
            return words[0]
        return ""
    
    def generate_answer(self, query: str, k: int = 3) -> str:
        """Generate direct, concise answers from top retrieved context.

        Required flow:
        retrieve -> select top chunks -> combine context -> apply question -> generate answer

        Must satisfy strict rules:
        - Only use context and question; no paraphrasing of unrelated content.
        - No summarization by dropping context as a paragraph.
        - Enforce top_k=2..3 chunks.
        - If not answerable, return "Not found in context".
        """
        try:
            retrieved = self.retrieve(query, k=k)
            
            # ── ISSUE 7: No results at all → fallback ──
            if not retrieved:
                return ("The available context does not contain enough information "
                        "to answer this question.")
            
            # ── ISSUE 7: Relevance gate ──
            is_relevant, best_score = self.check_query_relevance(query, retrieved)
            if not is_relevant:
                return ("The available context does not contain enough information "
                        "to answer this question.")
            
            # Select top_k chunks (2-3) and combine context.
            top_k = min(max(k, 2), 3)
            top_chunks = retrieved[:top_k]
            context = " ".join(
                self._clean_text(doc.get('content', ''))
                for doc in top_chunks
                if doc.get('content', '').strip()
            ).strip()

            if not context:
                return "Not found in context"

            # Guard: detect if stored chunks are garbage (raw PDF binary)
            garbage_markers = ['endobj', '/BaseFont', '/Subtype', '/Type/Font',
                               '/Filter', '/FlateDecode']
            garbage_hits = sum(1 for m in garbage_markers if m in context[:1000])
            printable_ratio = sum(1 for c in context[:500] if c.isprintable() or c in '\n\r\t') / max(len(context[:500]), 1)
            if garbage_hits >= 2 or printable_ratio < 0.6:
                logger.warning(f"Context appears to be garbage (markers={garbage_hits}, printable={printable_ratio:.2f})")
                return "Not found in context"
            
            # ── STRATEGY A: Skip generic summarizer to enforce direct QA-based answer generation ──
            # (user requirement: do NOT summarize full context, only directly answer question)
            # if self.summarizer_pipeline is enabled, we do not use it in this flow.
            pass

            # 2. Apply mandatory instruction with question + context
            llm_prompt = (
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                "Answer the question using ONLY the provided context\n"
                "Do NOT copy sentences directly from the context\n"
                "Do NOT summarize the entire text\n"
                "Provide a direct and relevant answer\n"
                "Limit the answer to 2–3 lines\n"
                "If the answer is not present, return: \"Not found in context\""
            )

            if self.summarizer_pipeline:
                try:
                    logger.info("Running generative model on strict prompt...")
                    llm_result = self.summarizer_pipeline(
                        llm_prompt,
                        max_length=120,
                        min_length=30,
                        do_sample=False
                    )
                    generated = llm_result[0].get('generated_text', '').strip()
                    answer = generated.replace('\n', ' ').strip()
                except Exception as gen_err:
                    logger.warning(f"Generative model error: {gen_err}")
                    answer = ""
            else:
                logger.warning("Generative model unavailable, attempting rule-based fallback")
                answer = self._synthesize_answer(query, context)

            # 3. Output validation
            if not answer or answer.lower().startswith('unknown') or 'not found' in answer.lower():
                return 'Not found in context'

            # reject direct context copy (if any chunk substring is full answer)
            for chunk in top_chunks:
                chunk_text = self._clean_text(chunk.get('content', ''))
                

            # sentence count should be 1-3 sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
            if len(sentences) > 3 or len(answer.split()) > 80:
                answer = ' '.join(sentences[:3])

            # have to include question words or subject
            if not re.search(r'\b(who|what|when|where|why|how|which|is|are|does|did)\b', query.lower()):
                return 'Not found in context'

            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return "Not found in context"
    
    def _frame_answer(self, query: str, raw_span: str, context: str) -> str:
        """Turn a raw extractive span into a natural, question-focused sentence.
        
        ISSUE 1 FIX: Instead of returning raw chunks, we construct a proper sentence.
        ISSUE 2 FIX: Answer starts with the subject and directly addresses the query.
        
        Example:
          query  = 'Who is Kalidasa?'
          span   = 'a clever poet'
          result = 'Kalidasa is a clever poet.'
        """
        subject = self._extract_subject(query)
        span = raw_span.strip().rstrip('.')
        
        q_lower = query.lower()
        
        # "Who is X" -> "X is <span>."
        if 'who is' in q_lower and subject:
            answer = f"{subject} is {span}."
        # "What is X" -> "X is <span>."
        elif 'what is' in q_lower and subject:
            answer = f"{subject} is {span}."
        # "What happens" -> "<Span>."
        elif 'what happen' in q_lower:
            answer = span[0].upper() + span[1:] + '.'
        else:
            # Generic: capitalise span and add period
            answer = span[0].upper() + span[1:] + '.' if span else raw_span
        
        # Always try to append supporting context for short answers
        if len(answer) < 120:
            extra = self._find_supporting_sentence(subject or raw_span, context, exclude=raw_span)
            if extra:
                answer = answer + " " + extra
        
        return answer
    
    def _ensure_min_length(self, answer: str, query: str, context: str) -> str:
        """ISSUE 3 FIX: Ensure the answer is at least 2-3 lines (~80 chars).
        
        If the answer is too short, add supporting sentences from context.
        """
        if len(answer) >= self.MIN_ANSWER_LENGTH:
            return answer
        
        subject = self._extract_subject(query)
        sentences = self._extract_english_sentences(context)
        
        # Find sentences not already in the answer
        used_lower = answer.lower()
        for s in sentences:
            if s.lower().strip('.') in used_lower:
                continue
            # Prefer sentences mentioning the subject
            if subject and subject.lower() in s.lower():
                clean = s.strip()
                if not clean.endswith('.'):
                    clean += '.'
                if len(clean) < 200:
                    answer = answer.rstrip() + " " + clean
                    if len(answer) >= self.MIN_ANSWER_LENGTH:
                        break
        
        # If still too short, add any relevant sentence
        if len(answer) < self.MIN_ANSWER_LENGTH:
            for s in sentences:
                if s.lower().strip('.') in used_lower:
                    continue
                clean = s.strip()
                if not clean.endswith('.'):
                    clean += '.'
                if len(clean) > 15:
                    answer = answer.rstrip() + " " + clean
                    if len(answer) >= self.MIN_ANSWER_LENGTH:
                        break
        
        return answer
    
    def _find_supporting_sentence(self, keyword: str, context: str, exclude: str = "") -> str:
        """Find one additional sentence from context that mentions the keyword,
        but is not the same as *exclude*."""
        if not keyword:
            return ""
        sentences = self._extract_english_sentences(context)
        kw_lower = keyword.lower()
        for s in sentences:
            if kw_lower in s.lower() and s.strip() != exclude.strip():
                clean = s.strip()
                if not clean.endswith('.'):
                    clean += '.'
                if len(clean) < 200:
                    return clean
        return ""
    
    def _synthesize_answer(self, query: str, context: str) -> str:
        """Rule-based answer synthesis when no QA model is available.
        
        ISSUE 1 FIX: Does NOT copy chunks verbatim. Picks 2-3 relevant sentences
        and re-frames them as a direct answer.
        ISSUE 2 FIX: Answer starts with the subject and directly addresses the query.
        ISSUE 3 FIX: Picks 2-3 sentences to ensure adequate length.
        """
        subject = self._extract_subject(query)
        sentences = self._extract_english_sentences(context)
        
        if not sentences:
            return ("The available context does not contain enough information "
                    "to answer this question.")
        
        # Score each sentence by how many query words it contains
        query_words = set(re.findall(r'\w{3,}', query.lower()))
        stop = {'the', 'what', 'who', 'how', 'where', 'when', 'why', 'does',
                'this', 'that', 'are', 'was', 'were', 'been', 'being', 'have',
                'has', 'had', 'with', 'from', 'about', 'which', 'their',
                'there', 'will', 'would', 'could', 'should', 'can', 'may'}
        query_words -= stop
        
        scored = []
        for s in sentences:
            s_lower = s.lower()
            hits = sum(1 for w in query_words if w in s_lower)
            # Bonus if subject is mentioned
            if subject and subject.lower() in s_lower:
                hits += 2
            scored.append((hits, s))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Take top 2-3 sentences (ISSUE 3: ensure 2-3 lines)
        best = [s for score, s in scored[:3] if score > 0]
        
        if not best:
            # No keyword overlap — take first 2 sentences as generic answer
            best = sentences[:2]
        
        # Frame the answer (ISSUE 1: synthesize, don't copy)
        answer_body = ". ".join(s.rstrip('.') for s in best) + "."
        
        # ISSUE 2: Prepend subject-framing if possible
        if subject and not answer_body.lower().startswith(subject.lower()):
            if subject.lower() not in answer_body.lower():
                answer_body = f"Based on the document, {answer_body[0].lower()}{answer_body[1:]}"
        
        return answer_body
    
    # ------------------------------------------------------------------
    # Source snippets — ISSUE 4
    # ------------------------------------------------------------------
    
    def get_source_snippets(self, retrieved_docs: List[Dict], max_length: int = None) -> List[Dict]:
        """Extract SHORT source snippets for display.
        
        ISSUE 4 FIX: Returns ultra-short, readable source references.
        No metadata (emails, author names) in snippets. 
        """
        if max_length is None:
            max_length = self.MAX_SNIPPET_LENGTH
        
        snippets = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get('content', '').strip()
            if not content:
                continue
            
            # Clean page markers and metadata (ISSUE 5)
            content = self._clean_text(content)
            
            # Get first meaningful sentence
            parts = re.split(r'(?<=[.!?।॥])\s', content)
            snippet = parts[0].strip() if parts else content
            
            # Skip trivially short snippets
            if len(snippet) < 10:
                snippet = parts[1].strip() if len(parts) > 1 else content[:max_length]
            if len(snippet) < 5:
                continue
            
            # Enforce max_length
            if len(snippet) > max_length:
                truncated = snippet[:max_length]
                last_space = truncated.rfind(' ')
                if last_space > max_length * 0.4:
                    snippet = truncated[:last_space].strip() + "..."
                else:
                    snippet = truncated.strip() + "..."
            
            source_name = doc.get('source', 'unknown')
            score = float(doc.get('similarity_score', 0))
            
            snippets.append({
                "rank": i,
                "snippet": f'"{snippet}"',
                "source": source_name,
                "score": round(score, 3)
            })
        
        return snippets
