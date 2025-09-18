#!/bin/bash

# Enhanced Construction RAG - Project Structure Setup
echo "🚀 Setting up Enhanced Construction RAG project structure..."

# Create core directory structure
echo "📁 Creating directory structure..."

mkdir -p core
mkdir -p agents  
mkdir -p retrieval
mkdir -p processing

# Create core files
echo "📄 Creating core files..."

# Core module files
touch core/__init__.py
touch core/config.py
touch core/qdrant_client.py

# Agents module files  
touch agents/__init__.py
touch agents/query_router.py
touch agents/multi_agent.py

# Retrieval module files
touch retrieval/__init__.py
touch retrieval/hybrid_retriever.py
touch retrieval/corrective_rag.py

# Processing module files
touch processing/__init__.py
touch processing/advanced_chunking.py

# Create .env.example if it doesn't exist
if [ ! -f .env.example ]; then
    touch .env.example
    echo "📝 Created .env.example"
fi

# Add initial content to __init__.py files
echo "✨ Adding module initialization content..."

# Core module init
cat > core/__init__.py << 'EOF'
"""
Enhanced Construction RAG - Core Module
Provides configuration management and database connections
"""

from .config import Config
from .qdrant_client import QdrantClient

__all__ = ['Config', 'QdrantClient']
EOF

# Agents module init
cat > agents/__init__.py << 'EOF'
"""
Enhanced Construction RAG - Agents Module
Provides intelligent query routing and multi-agent coordination
"""

from .query_router import QueryRouter
from .multi_agent import MultiAgentCoordinator

__all__ = ['QueryRouter', 'MultiAgentCoordinator']
EOF

# Retrieval module init
cat > retrieval/__init__.py << 'EOF'
"""
Enhanced Construction RAG - Retrieval Module
Provides hybrid retrieval and corrective RAG capabilities
"""

from .hybrid_retriever import HybridRetriever
from .corrective_rag import CorrectiveRAG

__all__ = ['HybridRetriever', 'CorrectiveRAG']
EOF

# Processing module init
cat > processing/__init__.py << 'EOF'
"""
Enhanced Construction RAG - Processing Module
Provides advanced document processing and chunking
"""

from .advanced_chunking import AdvancedChunking

__all__ = ['AdvancedChunking']
EOF

# Update main __init__.py to include new modules
cat > __init__.py << 'EOF'
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
EOF

echo "✅ Project structure created successfully!"
echo ""
echo "📊 Structure summary:"
echo "├── core/ (3 files)"
echo "├── agents/ (3 files)" 
echo "├── retrieval/ (3 files)"
echo "├── processing/ (2 files)"
echo "└── Enhanced __init__.py"
echo ""
echo "🎯 Next steps:"
echo "1. Update requirements.txt with new dependencies"
echo "2. Configure .env.example with new settings" 
echo "3. Start implementing the enhanced components"
echo ""
echo "🚀 Ready to enhance your Construction RAG system!"