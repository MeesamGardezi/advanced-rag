import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime

# Enhanced imports
from core.config import config
from database import initialize_databases, get_database_stats, test_connections
from rag_service import get_rag_service
from models import DocumentCreate, QueryRequest, QueryResponse, DocumentSource

# Set up logging
logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

# Enhanced models for new functionality
class EnhancedQueryRequest(BaseModel):
    question: str = Field(..., description="The construction query to ask")
    n_results: Optional[int] = Field(default=5, ge=1, le=20, description="Number of results to return")
    data_types: Optional[List[str]] = Field(default=None, description="Filter by data types: consumed, estimate, schedule")
    use_multi_agent: Optional[bool] = Field(default=None, description="Force multi-agent processing")
    use_corrective_rag: Optional[bool] = Field(default=None, description="Force corrective RAG")

class AgentQueryRequest(BaseModel):
    question: str = Field(..., description="Question for specialized agent")
    agent_types: List[str] = Field(..., description="Agents to use: specification, compliance, cost, schedule")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class ProcessingRequest(BaseModel):
    company_id: Optional[str] = Field(default=None, description="Specific company to process")
    force_reprocess: bool = Field(default=False, description="Force reprocessing existing data")
    use_advanced_chunking: bool = Field(default=True, description="Use advanced semantic chunking")

class HealthResponse(BaseModel):
    overall_status: str
    timestamp: str
    components: Dict[str, Any]
    service_stats: Dict[str, Any]
    issues: Optional[List[str]] = None

class ConfigResponse(BaseModel):
    multi_agent_enabled: bool
    corrective_rag_enabled: bool
    semantic_chunking_enabled: bool
    semantic_caching_enabled: bool
    supported_data_types: List[str]
    version: str

# Global services
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global rag_service
    
    logger.info("🚀 Starting Enhanced Construction RAG System v2.0...")
    
    try:
        # Initialize all database connections
        success = initialize_databases()
        if not success:
            logger.error("❌ Failed to initialize databases")
            raise RuntimeError("Database initialization failed")
        
        # Initialize enhanced RAG service
        rag_service = get_rag_service()
        
        # Test all connections
        connection_results = test_connections()
        healthy_systems = sum(1 for status in connection_results.values() if status)
        
        logger.info(f"✅ System ready! ({healthy_systems}/{len(connection_results)} systems healthy)")
        logger.info("🎯 Enhanced features:")
        logger.info(f"   - Multi-Agent System: {'✅' if config.agents.enable_multi_agent else '❌'}")
        logger.info(f"   - Corrective RAG: {'✅' if config.retrieval.enable_corrective_rag else '❌'}")
        logger.info(f"   - Semantic Chunking: {'✅' if config.processing.enable_semantic_chunking else '❌'}")
        logger.info(f"   - Semantic Caching: {'✅' if config.embedding.enable_semantic_cache else '❌'}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Enhanced Construction RAG System...")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Construction RAG API",
    description="Advanced RAG system for construction data with multi-agent processing, corrective retrieval, and intelligent query routing",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoints
@app.get("/")
async def root():
    return {
        "message": "Enhanced Construction RAG API v2.0",
        "version": "2.0.0",
        "features": [
            "Multi-Agent Processing",
            "Corrective RAG", 
            "Intelligent Query Routing",
            "Semantic Chunking",
            "Hybrid Retrieval",
            "Real-time Streaming"
        ],
        "supported_data_types": ["consumed", "estimate", "schedule"],
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def comprehensive_health_check():
    """Comprehensive system health check"""
    try:
        health_status = await rag_service.health_check()
        return HealthResponse(**health_status)
    except Exception as e:
        return HealthResponse(
            overall_status="error",
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)},
            service_stats={},
            issues=[f"Health check failed: {e}"]
        )

@app.get("/config", response_model=ConfigResponse)
async def get_system_config():
    """Get current system configuration"""
    return ConfigResponse(
        multi_agent_enabled=config.agents.enable_multi_agent,
        corrective_rag_enabled=config.retrieval.enable_corrective_rag,
        semantic_chunking_enabled=config.processing.enable_semantic_chunking,
        semantic_caching_enabled=config.embedding.enable_semantic_cache,
        supported_data_types=["consumed", "estimate", "schedule"],
        version="2.0.0"
    )

# Enhanced query endpoints
@app.post("/query/enhanced")
async def enhanced_query(query: EnhancedQueryRequest):
    """Enhanced query with intelligent routing and multi-agent processing"""
    try:
        result = await rag_service.query(
            question=query.question,
            n_results=query.n_results,
            data_types=query.data_types,
            use_multi_agent=query.use_multi_agent,
            use_corrective_rag=query.use_corrective_rag
        )
        
        return {
            "question": query.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "relevant_chunks": result["chunks"],
            "data_types_found": result["data_types_found"],
            "document_types_found": result["document_types_found"],
            "processing_time": result["processing_time"],
            "metadata": result["metadata"]
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/agents")
async def multi_agent_query(query: AgentQueryRequest):
    """Direct multi-agent query processing"""
    try:
        if not config.agents.enable_multi_agent:
            raise HTTPException(status_code=503, detail="Multi-agent system is disabled")
        
        # Force multi-agent processing
        result = await rag_service.query(
            question=query.question,
            use_multi_agent=True
        )
        
        return {
            "question": query.question,
            "answer": result["answer"],
            "agents_used": result["metadata"].get("agent_coordination", {}).get("agents_used", []),
            "parallel_execution": result["metadata"].get("agent_coordination", {}).get("parallel_execution", False),
            "sources": result["sources"],
            "processing_time": result["processing_time"],
            "agent_details": result["metadata"].get("agent_coordination", {})
        }
        
    except Exception as e:
        logger.error(f"❌ Multi-agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def streaming_query(query: EnhancedQueryRequest):
    """Streaming query response for real-time results"""
    
    async def generate_streaming_response():
        try:
            # Start query processing
            yield f"data: {{'status': 'starting', 'message': 'Processing query...'}}\n\n"
            
            # Process query
            result = await rag_service.query(
                question=query.question,
                n_results=query.n_results,
                data_types=query.data_types
            )
            
            # Stream response in chunks
            answer = result["answer"]
            words = answer.split()
            
            for i, word in enumerate(words):
                chunk_data = {
                    'status': 'streaming',
                    'word': word,
                    'progress': (i + 1) / len(words),
                    'is_complete': False
                }
                yield f"data: {chunk_data}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Send completion data
            completion_data = {
                'status': 'complete',
                'full_answer': answer,
                'sources': [source.dict() for source in result["sources"]],
                'processing_time': result["processing_time"],
                'is_complete': True
            }
            yield f"data: {completion_data}\n\n"
            
        except Exception as e:
            error_data = {
                'status': 'error',
                'error': str(e),
                'is_complete': True
            }
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate_streaming_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# WebSocket endpoint for real-time communication
@app.websocket("/ws/construction/{project_id}")
async def construction_websocket(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time construction query processing"""
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_text()
            query_data = eval(data)  # In production, use json.loads with proper validation
            
            # Process query
            result = await rag_service.query(
                question=query_data.get("question", ""),
                data_types=query_data.get("data_types"),
                n_results=query_data.get("n_results", 5)
            )
            
            # Send response
            await websocket.send_json({
                "type": "query_result",
                "project_id": project_id,
                "result": {
                    "answer": result["answer"],
                    "sources": [source.dict() for source in result["sources"]],
                    "processing_time": result["processing_time"]
                }
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for project {project_id}")
    except Exception as e:
        logger.error(f"WebSocket error for project {project_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })

# Data processing endpoints
@app.post("/processing/enhanced")
async def enhanced_data_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Enhanced data processing with advanced chunking"""
    
    async def process_data():
        try:
            return await rag_service.process_firebase_data(request.company_id)
        except Exception as e:
            logger.error(f"Background processing failed: {e}")
    
    if request.force_reprocess:
        # Synchronous processing for immediate feedback
        try:
            stats = await rag_service.process_firebase_data(request.company_id)
            return {
                "message": "Enhanced data processing completed",
                "stats": stats,
                "processing_type": "synchronous"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Background processing
        background_tasks.add_task(process_data)
        return {
            "message": "Enhanced data processing started in background",
            "company_id": request.company_id or "all_companies",
            "processing_type": "background",
            "features": ["advanced_chunking", "intelligent_routing", "enhanced_metadata"]
        }

@app.post("/processing/migrate")
async def migrate_to_enhanced_system():
    """Migrate from ChromaDB to Qdrant with enhanced features"""
    try:
        result = await rag_service.migrate_from_chromadb()
        return {
            "migration_status": result["status"],
            "migrated_documents": result.get("migrated_count", 0),
            "errors": result.get("errors", []),
            "enhanced_features_enabled": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Statistics and monitoring endpoints
@app.get("/stats/comprehensive")
async def get_comprehensive_stats():
    """Get comprehensive system statistics"""
    try:
        stats = rag_service.get_collection_stats()
        db_stats = get_database_stats()
        
        return {
            "collection_stats": stats,
            "database_stats": db_stats,
            "system_config": {
                "multi_agent_enabled": config.agents.enable_multi_agent,
                "corrective_rag_enabled": config.retrieval.enable_corrective_rag,
                "advanced_chunking_enabled": config.processing.enable_semantic_chunking,
                "semantic_caching_enabled": config.embedding.enable_semantic_cache
            },
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/performance")
async def get_performance_metrics():
    """Get performance and usage metrics"""
    try:
        stats = rag_service.get_collection_stats()
        
        return {
            "query_metrics": {
                "total_queries": stats["rag_service_stats"]["queries_processed"],
                "multi_agent_queries": stats["rag_service_stats"]["multi_agent_queries"],
                "corrective_rag_activations": stats["rag_service_stats"]["corrective_rag_activations"],
                "average_response_time": stats["rag_service_stats"]["average_response_time"]
            },
            "cache_metrics": stats["embedding_stats"].get("cache_stats", {}),
            "model_routing_metrics": stats["embedding_stats"].get("model_routing_stats", {}),
            "system_health": stats["system_health"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Job and data management endpoints
@app.get("/jobs/enhanced")
async def get_enhanced_jobs_list():
    """Get enhanced list of jobs with detailed metadata"""
    try:
        jobs = await rag_service.get_available_jobs()
        data_summary = await rag_service.get_data_types_summary()
        
        return {
            "jobs": jobs,
            "summary": {
                "total_jobs": len(jobs),
                "data_types_summary": data_summary,
                "enhanced_processing": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-analysis/insights")
async def get_data_insights():
    """Get AI-powered insights about the construction data"""
    try:
        # Generate insights about the data collection
        stats = rag_service.get_collection_stats()
        qdrant_stats = stats.get("qdrant_stats", {})
        
        insights = {
            "data_coverage": {
                "total_documents": qdrant_stats.get("total_points", 0),
                "data_types": qdrant_stats.get("data_type_breakdown", {}),
                "processing_quality": "enhanced" if config.processing.enable_semantic_chunking else "standard"
            },
            "system_performance": {
                "average_response_time": stats["rag_service_stats"]["average_response_time"],
                "multi_agent_usage_rate": stats["rag_service_stats"]["multi_agent_queries"] / max(stats["rag_service_stats"]["queries_processed"], 1),
                "corrective_rag_rate": stats["rag_service_stats"]["corrective_rag_activations"] / max(stats["rag_service_stats"]["queries_processed"], 1)
            },
            "recommendations": []
        }
        
        # Generate recommendations based on stats
        if insights["system_performance"]["average_response_time"] > 3.0:
            insights["recommendations"].append("Consider enabling semantic caching to improve response times")
        
        if insights["system_performance"]["multi_agent_usage_rate"] < 0.3:
            insights["recommendations"].append("Enable multi-agent processing for more complex queries")
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Backward compatibility endpoints
@app.post("/documents", response_model=Dict[str, Any])
async def add_document(document: DocumentCreate):
    """Add a document manually (backward compatible)"""
    try:
        doc_id = await rag_service.add_document(
            text=document.text,
            metadata=document.metadata or {}
        )
        return {"success": True, "document_id": doc_id, "enhanced": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=QueryResponse)
async def search_documents(query: QueryRequest):
    """Basic search endpoint (backward compatible)"""
    try:
        result = await rag_service.query(
            question=query.question,
            n_results=query.n_results or 5
        )
        
        return QueryResponse(
            question=query.question,
            answer=result["answer"],
            sources=result["sources"],
            relevant_chunks=result["chunks"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings")
async def trigger_embedding_generation(
    background_tasks: BackgroundTasks,
    company_id: Optional[str] = None
):
    """Generate embeddings (backward compatible)"""
    try:
        async def process_data():
            return await rag_service.process_firebase_data(company_id)
        
        background_tasks.add_task(process_data)
        return {
            "message": "Enhanced embedding generation started in background",
            "company_id": company_id or "all_companies",
            "data_types": ["consumed", "estimate", "schedule"],
            "enhanced_features": ["semantic_chunking", "multi_agent_ready", "intelligent_routing"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/embeddings/clear")
async def clear_all_embeddings():
    """Clear all embeddings (backward compatible)"""
    try:
        from database import clear_qdrant_collection
        success = clear_qdrant_collection()
        
        return {
            "message": "Vector database cleared" if success else "Clear operation failed",
            "success": success,
            "database": "Qdrant (Enhanced)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics (backward compatible)"""
    try:
        stats = rag_service.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Development and debugging endpoints
@app.get("/debug/query-classification")
async def debug_query_classification(query: str = Query(..., description="Query to classify")):
    """Debug query classification system"""
    try:
        from agents.query_router import get_query_router
        from dataclasses import asdict
        
        router = get_query_router()
        classification = await router.classify_query(query)
        routing_strategy = router.get_routing_strategy(classification)
        
        return {
            "query": query,
            "classification": asdict(classification),
            "routing_strategy": routing_strategy,
            "debug_info": {
                "confidence_threshold": config.agents.classification_confidence_threshold,
                "multi_agent_enabled": config.agents.enable_multi_agent
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/system-status")
async def debug_system_status():
    """Debug system status and configuration"""
    return {
        "version": "2.0.0",
        "environment": config.environment,
        "features": {
            "multi_agent": {
                "enabled": config.agents.enable_multi_agent,
                "max_concurrent": config.agents.max_concurrent_agents,
                "available_agents": ["specification", "compliance", "cost", "schedule"]
            },
            "corrective_rag": {
                "enabled": config.retrieval.enable_corrective_rag,
                "max_attempts": config.retrieval.max_correction_attempts
            },
            "semantic_chunking": {
                "enabled": config.processing.enable_semantic_chunking,
                "chunk_sizes": config.processing.chunk_sizes
            },
            "caching": {
                "enabled": config.embedding.enable_semantic_cache,
                "similarity_threshold": config.embedding.cache_similarity_threshold
            }
        },
        "database": {
            "primary_vector_db": "Qdrant",
            "backup_vector_db": "ChromaDB",
            "document_db": "Firebase"
        }
    }

# Application startup message
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Enhanced Construction RAG API v2.0 on port {port}")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=config.debug,
        log_level=config.log_level.lower()
    )