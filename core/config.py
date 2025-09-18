import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    # Firebase settings
    firebase_project_id: str
    firebase_private_key_id: str
    firebase_private_key: str
    firebase_client_email: str
    firebase_client_id: str
    firebase_auth_uri: str = "https://accounts.google.com/o/oauth2/auth"
    firebase_token_uri: str = "https://oauth2.googleapis.com/token"
    
    # Qdrant settings
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "construction_rag_v2"
    qdrant_timeout: int = 10
    
    # ChromaDB settings (for migration/compatibility)
    chroma_persist_path: str = "./chroma_storage"
    chroma_collection_name: str = "construction_rag"

@dataclass 
class EmbeddingConfig:
    """Embedding and AI model configuration"""
    # OpenAI settings
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    completion_model_simple: str = "gpt-3.5-turbo"
    completion_model_complex: str = "gpt-4o"
    max_tokens: int = 600
    temperature: float = 0.1
    
    # Model routing thresholds
    simple_query_threshold: float = 0.7
    batch_size: int = 100
    
    # Cost optimization
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 7200  # 2 hours

@dataclass
class RetrievalConfig:
    """Retrieval system configuration"""
    # Hybrid search settings
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    n_results_default: int = 5
    n_results_max: int = 20
    
    # Query routing
    enable_query_routing: bool = True
    route_confidence_threshold: float = 0.8
    
    # Corrective RAG settings
    enable_corrective_rag: bool = True
    document_relevance_threshold: float = 0.6
    max_correction_attempts: int = 2
    
    # Reranking
    enable_authority_reranking: bool = True
    authority_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.authority_weights is None:
            self.authority_weights = {
                'building_code': 1.0,
                'specification': 0.9,
                'standard': 0.8,
                'guideline': 0.7,
                'consumed': 0.8,  # Actual cost data
                'estimate': 0.7,  # Estimated data
                'schedule': 0.6,  # Schedule data
                'reference': 0.5
            }

@dataclass
class AgentConfig:
    """Multi-agent system configuration"""
    # Agent settings
    enable_multi_agent: bool = True
    max_concurrent_agents: int = 3
    agent_timeout_seconds: int = 30
    
    # Specialized agents
    enable_specification_agent: bool = True
    enable_compliance_agent: bool = True
    enable_cost_agent: bool = True
    enable_schedule_agent: bool = True
    
    # Query classification
    classification_confidence_threshold: float = 0.75

@dataclass
class CacheConfig:
    """Caching configuration"""
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Semantic cache settings
    enable_semantic_cache: bool = True
    cache_prefix: str = "construction_rag"
    default_ttl: int = 7200  # 2 hours
    max_cache_size: int = 10000

@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    # Chunking settings
    enable_semantic_chunking: bool = True
    chunk_sizes: List[int] = None
    chunk_overlap: int = 200
    
    # Multi-modal processing
    enable_multimodal: bool = False
    supported_file_types: List[str] = None
    max_file_size_mb: int = 50
    
    def __post_init__(self):
        if self.chunk_sizes is None:
            self.chunk_sizes = [2048, 512, 128]
        if self.supported_file_types is None:
            self.supported_file_types = ['pdf', 'docx', 'txt', 'md']

class Config:
    """Main configuration class that loads and manages all settings"""
    
    def __init__(self):
        self.database = self._load_database_config()
        self.embedding = self._load_embedding_config()
        self.retrieval = self._load_retrieval_config()
        self.agents = self._load_agent_config()
        self.cache = self._load_cache_config()
        self.processing = self._load_processing_config()
        
        # System settings
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Validate critical settings
        self._validate_config()
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment"""
        return DatabaseConfig(
            firebase_project_id=os.getenv("FIREBASE_PROJECT_ID"),
            firebase_private_key_id=os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            firebase_private_key=os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
            firebase_client_email=os.getenv("FIREBASE_CLIENT_EMAIL"),
            firebase_client_id=os.getenv("FIREBASE_CLIENT_ID"),
            firebase_auth_uri=os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            firebase_token_uri=os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            qdrant_url=os.getenv("QDRANT_URL", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "construction_rag_v2"),
            qdrant_timeout=int(os.getenv("QDRANT_TIMEOUT", "10")),
            chroma_persist_path=os.getenv("CHROMA_PERSIST_PATH", "./chroma_storage"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "construction_rag")
        )
    
    def _load_embedding_config(self) -> EmbeddingConfig:
        """Load embedding configuration from environment"""
        return EmbeddingConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            completion_model_simple=os.getenv("COMPLETION_MODEL_SIMPLE", "gpt-3.5-turbo"),
            completion_model_complex=os.getenv("COMPLETION_MODEL_COMPLEX", "gpt-4o"),
            max_tokens=int(os.getenv("MAX_TOKENS", "600")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            simple_query_threshold=float(os.getenv("SIMPLE_QUERY_THRESHOLD", "0.7")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            enable_semantic_cache=os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true",
            cache_similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95")),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "7200"))
        )
    
    def _load_retrieval_config(self) -> RetrievalConfig:
        """Load retrieval configuration from environment"""
        return RetrievalConfig(
            semantic_weight=float(os.getenv("SEMANTIC_WEIGHT", "0.7")),
            keyword_weight=float(os.getenv("KEYWORD_WEIGHT", "0.3")),
            n_results_default=int(os.getenv("N_RESULTS_DEFAULT", "5")),
            n_results_max=int(os.getenv("N_RESULTS_MAX", "20")),
            enable_query_routing=os.getenv("ENABLE_QUERY_ROUTING", "true").lower() == "true",
            route_confidence_threshold=float(os.getenv("ROUTE_CONFIDENCE_THRESHOLD", "0.8")),
            enable_corrective_rag=os.getenv("ENABLE_CORRECTIVE_RAG", "true").lower() == "true",
            document_relevance_threshold=float(os.getenv("DOC_RELEVANCE_THRESHOLD", "0.6")),
            max_correction_attempts=int(os.getenv("MAX_CORRECTION_ATTEMPTS", "2")),
            enable_authority_reranking=os.getenv("ENABLE_AUTHORITY_RERANKING", "true").lower() == "true"
        )
    
    def _load_agent_config(self) -> AgentConfig:
        """Load agent configuration from environment"""
        return AgentConfig(
            enable_multi_agent=os.getenv("ENABLE_MULTI_AGENT", "true").lower() == "true",
            max_concurrent_agents=int(os.getenv("MAX_CONCURRENT_AGENTS", "3")),
            agent_timeout_seconds=int(os.getenv("AGENT_TIMEOUT_SECONDS", "30")),
            enable_specification_agent=os.getenv("ENABLE_SPECIFICATION_AGENT", "true").lower() == "true",
            enable_compliance_agent=os.getenv("ENABLE_COMPLIANCE_AGENT", "true").lower() == "true",
            enable_cost_agent=os.getenv("ENABLE_COST_AGENT", "true").lower() == "true",
            enable_schedule_agent=os.getenv("ENABLE_SCHEDULE_AGENT", "true").lower() == "true",
            classification_confidence_threshold=float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.75"))
        )
    
    def _load_cache_config(self) -> CacheConfig:
        """Load cache configuration from environment"""
        return CacheConfig(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            enable_semantic_cache=os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true",
            cache_prefix=os.getenv("CACHE_PREFIX", "construction_rag"),
            default_ttl=int(os.getenv("DEFAULT_CACHE_TTL", "7200")),
            max_cache_size=int(os.getenv("MAX_CACHE_SIZE", "10000"))
        )
    
    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration from environment"""
        chunk_sizes_str = os.getenv("CHUNK_SIZES", "2048,512,128")
        chunk_sizes = [int(x.strip()) for x in chunk_sizes_str.split(",")]
        
        file_types_str = os.getenv("SUPPORTED_FILE_TYPES", "pdf,docx,txt,md")
        file_types = [x.strip() for x in file_types_str.split(",")]
        
        return ProcessingConfig(
            enable_semantic_chunking=os.getenv("ENABLE_SEMANTIC_CHUNKING", "true").lower() == "true",
            chunk_sizes=chunk_sizes,
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            enable_multimodal=os.getenv("ENABLE_MULTIMODAL", "false").lower() == "true",
            supported_file_types=file_types,
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        )
    
    def _validate_config(self):
        """Validate critical configuration settings"""
        errors = []
        
        # Check required Firebase settings
        if not self.database.firebase_project_id:
            errors.append("FIREBASE_PROJECT_ID is required")
        if not self.database.firebase_private_key:
            errors.append("FIREBASE_PRIVATE_KEY is required")
        if not self.database.firebase_client_email:
            errors.append("FIREBASE_CLIENT_EMAIL is required")
        
        # Check required OpenAI settings
        if not self.embedding.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    
    def get_firebase_credentials(self) -> Dict[str, Any]:
        """Get Firebase credentials as a dictionary"""
        return {
            "type": "service_account",
            "project_id": self.database.firebase_project_id,
            "private_key_id": self.database.firebase_private_key_id,
            "private_key": self.database.firebase_private_key,
            "client_email": self.database.firebase_client_email,
            "client_id": self.database.firebase_client_id,
            "auth_uri": self.database.firebase_auth_uri,
            "token_uri": self.database.firebase_token_uri,
        }
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def get_qdrant_url(self) -> str:
        """Get full Qdrant URL"""
        if self.database.qdrant_url.startswith("http"):
            return self.database.qdrant_url
        return f"http://{self.database.qdrant_url}:{self.database.qdrant_port}"

# Global configuration instance
config = Config()