"""
Configuration settings for Sanskrit Semantic Vector RAG
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration"""
    
    # API Configuration
    API_TITLE = "Sanskrit Semantic Vector RAG"
    API_VERSION = "0.1.0"
    API_DESCRIPTION = "Retrieval-Augmented Generation system for Sanskrit text"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 384))
    
    # RAG Configuration
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "backend/vector_store")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    
    # LLM Configuration (if using OpenAI)
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Pinecone Configuration (if using Pinecone)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "sanskrit-rag")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
    
    # HuggingFace Configuration
    HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY", "")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
    
    # CORS
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "*"  # Allow all in dev
    ]
    
    # Text Processing
    LANGUAGE = "sa"  # Sanskrit
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if not cls.EMBEDDING_MODEL:
            errors.append("EMBEDDING_MODEL not set")
        
        if cls.VECTOR_DIMENSION <= 0:
            errors.append("VECTOR_DIMENSION must be positive")
        
        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    CORS_ORIGINS = [
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ]


def get_config():
    """Get configuration based on environment"""
    env = os.getenv("ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig
    else:
        return DevelopmentConfig


# Export active config
ACTIVE_CONFIG = get_config()

if __name__ == "__main__":
    print("Current Configuration:")
    print(f"Environment: {os.getenv('ENV', 'development')}")
    print(f"Debug: {ACTIVE_CONFIG.DEBUG}")
    print(f"API Title: {ACTIVE_CONFIG.API_TITLE}")
    print(f"API Version: {ACTIVE_CONFIG.API_VERSION}")
    print(f"Embedding Model: {ACTIVE_CONFIG.EMBEDDING_MODEL}")
    print(f"Vector Dimension: {ACTIVE_CONFIG.VECTOR_DIMENSION}")
    print(f"Port: {ACTIVE_CONFIG.PORT}")
