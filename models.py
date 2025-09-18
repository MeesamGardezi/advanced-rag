from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum

# Enhanced enums for the new system
class DataType(str, Enum):
    """Construction data types"""
    CONSUMED = "consumed"
    ESTIMATE = "estimate"
    SCHEDULE = "schedule"
    ALL = "all"

class QueryType(str, Enum):
    """Query classification types"""
    SPECIFICATIONS = "specifications"
    BUILDING_CODES = "building_codes"
    COMPLIANCE_CHECK = "compliance_check"
    COST_ANALYSIS = "cost_analysis"
    SCHEDULE_PLANNING = "schedule_planning"
    GENERAL = "general"
    COMPARISON = "comparison"
    MULTI_DOMAIN = "multi_domain"

class QueryComplexity(str, Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class AgentType(str, Enum):
    """Available agent types"""
    SPECIFICATION = "specification"
    COMPLIANCE = "compliance"
    COST = "cost"
    SCHEDULE = "schedule"
    GENERAL = "general"

class ProcessingStatus(str, Enum):
    """Processing status indicators"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

# Basic models (backward compatible)
class DocumentCreate(BaseModel):
    """Model for creating a new document"""
    text: str = Field(..., description="The text content to embed")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata for the document")

class QueryRequest(BaseModel):
    """Basic query request model (backward compatible)"""
    question: str = Field(..., description="The question to ask about the construction data")
    n_results: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to return")

class DocumentSource(BaseModel):
    """Model for document sources in query responses"""
    job_name: str
    company_id: str
    job_id: str
    cost_code: Optional[str] = None
    amount: Optional[str] = None
    last_updated: Optional[str] = None

class QueryResponse(BaseModel):
    """Basic query response model (backward compatible)"""
    question: str
    answer: str
    sources: List[DocumentSource]
    relevant_chunks: List[str]

# Enhanced models for the new system
class QueryClassificationResult(BaseModel):
    """Query classification result"""
    query_type: QueryType
    complexity: QueryComplexity
    data_types: List[DataType]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    suggested_agents: List[AgentType]
    retrieval_strategy: str

class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with advanced options"""
    question: str = Field(..., description="The construction query to ask")
    n_results: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to return")
    data_types: Optional[List[DataType]] = Field(default=None, description="Filter by specific data types")
    use_multi_agent: Optional[bool] = Field(default=None, description="Force multi-agent processing")
    use_corrective_rag: Optional[bool] = Field(default=None, description="Enable corrective RAG")
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for processing")

class AgentResponse(BaseModel):
    """Response from a specialized agent"""
    agent_type: AgentType
    response: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    execution_time: float = Field(..., ge=0.0)
    status: ProcessingStatus
    reasoning: str
    sources_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MultiAgentQueryRequest(BaseModel):
    """Request for multi-agent query processing"""
    question: str = Field(..., description="Question for specialized agents")
    agent_types: List[AgentType] = Field(..., description="Specific agents to use")
    parallel_execution: bool = Field(default=True, description="Execute agents in parallel")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ChunkMetadata(BaseModel):
    """Enhanced metadata for document chunks"""
    chunk_id: str
    parent_id: Optional[str] = None
    chunk_type: str
    chunk_level: int = Field(..., ge=0)
    chunk_index: int = Field(..., ge=0)
    original_document_id: str
    construction_category: str
    section_title: Optional[str] = None
    data_type: DataType
    job_context: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = Field(..., gt=0)
    processing_version: str = "2.0"

class ProcessedChunk(BaseModel):
    """A processed document chunk with enhanced metadata"""
    id: str
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None

class EnhancedQueryResponse(BaseModel):
    """Enhanced query response with detailed metadata"""
    question: str
    answer: str
    sources: List[DocumentSource]
    relevant_chunks: List[str]
    data_types_found: List[DataType]
    document_types_found: List[str]
    processing_time: float = Field(..., ge=0.0)
    
    # Enhanced metadata
    query_classification: Optional[QueryClassificationResult] = None
    agent_responses: Optional[List[AgentResponse]] = None
    correction_metadata: Optional[Dict[str, Any]] = None
    retrieval_strategy: Optional[str] = None
    confidence_score: Optional[float] = None

class StreamingQueryChunk(BaseModel):
    """Streaming query response chunk"""
    status: Literal["starting", "streaming", "complete", "error"]
    word: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_complete: bool = False
    error: Optional[str] = None
    full_answer: Optional[str] = None
    sources: Optional[List[DocumentSource]] = None
    processing_time: Optional[float] = None

class ProcessingRequest(BaseModel):
    """Enhanced data processing request"""
    company_id: Optional[str] = Field(default=None, description="Specific company to process")
    force_reprocess: bool = Field(default=False, description="Force reprocessing existing data")
    use_advanced_chunking: bool = Field(default=True, description="Use semantic chunking")
    batch_size: Optional[int] = Field(default=50, ge=1, le=100, description="Processing batch size")
    data_types: Optional[List[DataType]] = Field(default=None, description="Specific data types to process")

class ProcessingStats(BaseModel):
    """Processing statistics"""
    total_jobs_processed: int = Field(..., ge=0)
    total_documents_created: int = Field(..., ge=0)
    total_chunks_created: int = Field(..., ge=0)
    consumed_datasets: int = Field(..., ge=0)
    estimate_datasets: int = Field(..., ge=0)
    schedule_datasets: int = Field(..., ge=0)
    companies_processed: List[str]
    processing_time_seconds: float = Field(..., ge=0.0)
    errors: List[str] = Field(default_factory=list)

class JobSummary(BaseModel):
    """Enhanced job summary with metadata"""
    job_name: str
    job_id: str
    company_id: str
    data_types: List[DataType]
    chunk_count: int = Field(..., ge=0)
    categories: List[str] = Field(default_factory=list)
    last_updated: Optional[str] = None
    
    # Enhanced fields
    total_cost: Optional[float] = None
    estimated_cost: Optional[float] = None
    project_status: Optional[str] = None
    completion_rate: Optional[float] = Field(None, ge=0.0, le=1.0)

class DataTypeSummary(BaseModel):
    """Enhanced data type summary"""
    consumed: int = Field(..., ge=0)
    estimate: int = Field(..., ge=0)
    schedule: int = Field(..., ge=0)
    unknown: int = Field(..., ge=0)
    total_documents: int = Field(..., ge=0)
    
    # Enhanced statistics
    total_jobs: int = Field(..., ge=0)
    companies_count: int = Field(..., ge=0)
    processing_version: str = "2.0"
    last_updated: str

class SystemHealth(BaseModel):
    """System health status"""
    overall_status: Literal["healthy", "degraded", "error"]
    timestamp: str
    components: Dict[str, Dict[str, Any]]
    service_stats: Dict[str, Any] = Field(default_factory=dict)
    issues: Optional[List[str]] = None

class PerformanceMetrics(BaseModel):
    """System performance metrics"""
    query_metrics: Dict[str, Union[int, float]]
    cache_metrics: Dict[str, Any] = Field(default_factory=dict)
    model_routing_metrics: Dict[str, Any] = Field(default_factory=dict)
    system_health: Dict[str, bool] = Field(default_factory=dict)

class ConfigurationInfo(BaseModel):
    """System configuration information"""
    multi_agent_enabled: bool
    corrective_rag_enabled: bool
    semantic_chunking_enabled: bool
    semantic_caching_enabled: bool
    supported_data_types: List[DataType]
    version: str
    environment: str = "development"
    
    # Advanced configuration
    max_concurrent_agents: int = Field(..., gt=0)
    chunk_sizes: List[int] = Field(default_factory=list)
    cache_ttl_seconds: int = Field(..., gt=0)

class MigrationResult(BaseModel):
    """Database migration result"""
    status: Literal["completed", "failed", "already_completed"]
    migrated_count: int = Field(..., ge=0)
    errors: List[str] = Field(default_factory=list)
    total_source: int = Field(..., ge=0)
    enhanced_features_enabled: bool = True
    migration_time: Optional[float] = None

class CostCodeSummary(BaseModel):
    """Enhanced cost code summary"""
    cost_code: str
    description: str
    category: str
    total_jobs_used: int = Field(..., ge=0)
    average_amount: float = Field(..., ge=0.0)
    
    # Enhanced fields
    total_amount: float = Field(..., ge=0.0)
    frequency: int = Field(..., ge=0)
    construction_category: Optional[str] = None

class ConstructionInsights(BaseModel):
    """AI-generated insights about construction data"""
    data_coverage: Dict[str, Any]
    system_performance: Dict[str, float]
    recommendations: List[str]
    
    # Advanced insights
    cost_trends: Optional[Dict[str, Any]] = None
    schedule_patterns: Optional[Dict[str, Any]] = None
    efficiency_metrics: Optional[Dict[str, Any]] = None

class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: Literal["query", "query_result", "status", "error"]
    project_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class SearchFilters(BaseModel):
    """Advanced search filters"""
    data_types: Optional[List[DataType]] = None
    company_ids: Optional[List[str]] = None
    job_ids: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None
    cost_range: Optional[Dict[str, float]] = None
    categories: Optional[List[str]] = None
    
    # Semantic filters
    min_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    construction_categories: Optional[List[str]] = None

class QueryAnalytics(BaseModel):
    """Query analytics and debugging information"""
    query: str
    classification: QueryClassificationResult
    routing_strategy: Dict[str, Any]
    processing_path: List[str]
    performance_metrics: Dict[str, float]
    cache_hit: bool = False
    
    # Debug information
    debug_info: Optional[Dict[str, Any]] = None

class BatchProcessingRequest(BaseModel):
    """Batch processing request"""
    queries: List[str] = Field(..., min_length=1, max_length=10)
    shared_context: Optional[Dict[str, Any]] = None
    processing_options: Optional[Dict[str, Any]] = None

class BatchProcessingResponse(BaseModel):
    """Batch processing response"""
    results: List[EnhancedQueryResponse]
    batch_metrics: Dict[str, float]
    total_processing_time: float = Field(..., ge=0.0)

# Legacy model aliases for backward compatibility
JobCostEntry = DocumentSource  # Alias for backward compatibility
ProcessedJobData = JobSummary   # Alias for backward compatibility
EmbeddingStats = ProcessingStats  # Alias for backward compatibility

# Model configuration
class BaseModelConfig:
    """Base configuration for all models"""
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": "Enhanced Construction RAG API v2.0"
        }
    )

# Apply configuration to all models
for cls_name in list(globals().keys()):
    cls = globals()[cls_name]
    if (isinstance(cls, type) and 
        issubclass(cls, BaseModel) and 
        cls is not BaseModel):
        if not hasattr(cls, 'model_config'):
            cls.model_config = BaseModelConfig.model_config.copy()

# Validation functions
def validate_data_types(data_types: List[str]) -> List[DataType]:
    """Validate and convert data type strings to enums"""
    valid_types = []
    for dt in data_types:
        try:
            valid_types.append(DataType(dt))
        except ValueError:
            continue  # Skip invalid types
    return valid_types

def validate_agent_types(agent_types: List[str]) -> List[AgentType]:
    """Validate and convert agent type strings to enums"""
    valid_types = []
    for at in agent_types:
        try:
            valid_types.append(AgentType(at))
        except ValueError:
            continue  # Skip invalid types
    return valid_types

# Helper functions for model creation
def create_error_response(error: str, query: str = "") -> EnhancedQueryResponse:
    """Create standardized error response"""
    return EnhancedQueryResponse(
        question=query,
        answer=f"Error processing query: {error}",
        sources=[],
        relevant_chunks=[],
        data_types_found=[],
        document_types_found=[],
        processing_time=0.0,
        confidence_score=0.0
    )

def create_empty_response(query: str) -> EnhancedQueryResponse:
    """Create empty response for no results"""
    return EnhancedQueryResponse(
        question=query,
        answer="No relevant construction data found for your query. Please try rephrasing or check if data has been loaded.",
        sources=[],
        relevant_chunks=[],
        data_types_found=[],
        document_types_found=[],
        processing_time=0.0,
        confidence_score=0.0
    )