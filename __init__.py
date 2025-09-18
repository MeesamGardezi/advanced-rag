# Enhanced Construction RAG System
# A RAG (Retrieval Augmented Generation) system for construction job cost data

__version__ = "2.0.0"
__author__ = "Construction RAG Team"
__description__ = "Enhanced RAG system with intelligent agents and hybrid retrieval for construction data"

# Import existing components
from database import initialize_firebase, get_chroma_collection, get_firebase_db
from embedding_service import EmbeddingService  
from rag_service import RAGService

# Import new enhanced components
from core import Config, QdrantClient
from agents import QueryRouter, MultiAgentCoordinator
from retrieval import HybridRetriever, CorrectiveRAG
from processing import AdvancedChunking

__all__ = [
    # Existing components
    "initialize_firebase",
    "get_chroma_collection", 
    "get_firebase_db",
    "EmbeddingService",
    "RAGService",
    # New enhanced components
    "Config",
    "QdrantClient",
    "QueryRouter", 
    "MultiAgentCoordinator",
    "HybridRetriever",
    "CorrectiveRAG",
    "AdvancedChunking"
]
