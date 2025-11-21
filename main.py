"""
Agentic RAG System - Main API Application
FastAPI server with hybrid SQL + Vector agent system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field 
from typing import List, Optional, Dict, Any
import os
from contextlib import asynccontextmanager
import logging

# Import database components
from db_connection import init_database, close_database, health_check, get_job_data_summary
from database_schema import Job, Estimate

# Import agent components
from agent.query_processor import get_query_processor, QueryProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# PYDANTIC MODELS
# ==========================================

class AgentQueryRequest(BaseModel):
    """Request model for agent-powered queries"""
    question: str = Field(..., min_length=3, max_length=2000, description="User's question")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Conversation history (list of {question, answer} dicts)"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context (job_id, filters, etc.)"
    )


class AgentQueryResponse(BaseModel):
    """Response model for agent queries"""
    success: bool
    question: str
    answer: str
    data_type_hint: Optional[str] = None
    tools_used: List[Dict[str, str]]
    response_time_ms: float
    timestamp: str
    error: Optional[str] = None


class JobContextRequest(BaseModel):
    """Request to validate job context"""
    job_id: str = Field(..., description="Job identifier")


class StatisticsResponse(BaseModel):
    """Statistics response"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    success_rate_pct: float
    average_response_time_ms: float
    tools_usage: Dict[str, int]


# ==========================================
# GLOBAL STATE
# ==========================================

query_processor: Optional[QueryProcessor] = None


# ==========================================
# LIFESPAN MANAGEMENT
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Agentic RAG System...")
    
    global query_processor
    
    try:
        # Initialize PostgreSQL database
        logger.info("🐘 Initializing PostgreSQL...")
        init_database()
        
        # Initialize query processor (which initializes agent)
        logger.info("🤖 Initializing Agent System...")
        query_processor = get_query_processor()
        
        logger.info("✅ Agentic RAG System ready!")
        logger.info(f"   - PostgreSQL: Connected")
        logger.info(f"   - ChromaDB: Connected")
        logger.info(f"   - Agent: Initialized")
        logger.info(f"   - Tools: SQL Tool, Vector Tool")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Agentic RAG System...")
    close_database()
    logger.info("✅ Shutdown complete")


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Agentic Construction RAG API",
    description="Hybrid SQL + Vector RAG system with intelligent agent routing for construction data queries",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Agentic Construction RAG API v2.0",
        "description": "Hybrid SQL + Vector architecture with intelligent agent",
        "version": "2.0.0",
        "architecture": "PostgreSQL + ChromaDB + LangChain Agent",
        "features": [
            "Intelligent query routing (SQL vs Vector)",
            "Multi-step query orchestration",
            "Hybrid search (exact filters + semantic matching)",
            "100% accurate calculations",
            "Natural language understanding",
            "Conversation history support"
        ],
        "endpoints": {
            "query": "POST /query - Agent-powered query processing",
            "health": "GET /health - System health check",
            "stats": "GET /stats - Query processing statistics",
            "job_context": "POST /job/validate - Validate job context"
        }
    }


@app.get("/health")
async def health_check_endpoint():
    """Comprehensive health check"""
    try:
        # Check database health
        db_health = health_check()
        
        # Check agent health
        agent_healthy = query_processor is not None
        
        # Overall status
        overall_status = "healthy" if (
            db_health['status'] == 'healthy' and agent_healthy
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "database": db_health,
            "agent": {
                "status": "healthy" if agent_healthy else "unhealthy",
                "initialized": agent_healthy
            },
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Process a query using the intelligent agent system
    
    The agent automatically:
    - Routes to SQL tool for exact filters and calculations
    - Routes to Vector tool for semantic matching
    - Orchestrates multi-step workflows
    - Ensures 100% accurate results
    
    Examples:
    - "What's the total cleanup cost for Hammond project?"
    - "Show me all allowances in the Mall estimate"
    - "Compare electrical costs: estimate vs consumed"
    """
    try:
        if query_processor is None:
            raise HTTPException(
                status_code=503,
                detail="Agent system not initialized"
            )
        
        # Process query through agent
        result = await query_processor.process_query(
            question=request.question,
            chat_history=request.chat_history,
            context=request.context
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Query processing failed')
            )
        
        return AgentQueryResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/job/validate")
async def validate_job_context(request: JobContextRequest):
    """
    Validate job context and check available data types
    
    Returns job information and which data types are available
    (estimates, flooring, schedule, consumed)
    """
    try:
        if query_processor is None:
            raise HTTPException(
                status_code=503,
                detail="Agent system not initialized"
            )
        
        validation = query_processor.validate_job_context(request.job_id)
        
        if not validation['valid']:
            raise HTTPException(
                status_code=404,
                detail=validation['error']
            )
        
        return validation
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatisticsResponse)
async def get_statistics():
    """
    Get query processing statistics
    
    Returns:
    - Total queries processed
    - Success/failure counts
    - Average response time
    - Tool usage distribution (SQL only, Vector only, Hybrid)
    """
    try:
        if query_processor is None:
            raise HTTPException(
                status_code=503,
                detail="Agent system not initialized"
            )
        
        stats = query_processor.get_statistics()
        return StatisticsResponse(**stats)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stats/reset")
async def reset_statistics():
    """Reset query processing statistics"""
    try:
        if query_processor is None:
            raise HTTPException(
                status_code=503,
                detail="Agent system not initialized"
            )
        
        query_processor.reset_statistics()
        return {"message": "Statistics reset successfully"}
    
    except Exception as e:
        logger.error(f"Reset statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def list_jobs():
    """
    List all jobs in the database with basic information
    """
    try:
        from db_connection import session_scope
        
        with session_scope() as session:
            jobs = session.query(Job).all()
            
            return {
                "total_jobs": len(jobs),
                "jobs": [
                    {
                        "id": job.id,
                        "name": job.name,
                        "client_name": job.client_name,
                        "company_id": job.company_id,
                        "status": job.status,
                        "site_location": job.site_location
                    }
                    for job in jobs
                ]
            }
    
    except Exception as e:
        logger.error(f"List jobs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}/summary")
async def get_job_summary(job_id: str):
    """
    Get comprehensive summary for a specific job
    
    Returns counts of all data types available for this job
    """
    try:
        summary = get_job_data_summary(job_id)
        
        if 'error' in summary:
            raise HTTPException(status_code=404, detail=summary['error'])
        
        return summary
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples")
async def get_query_examples():
    """
    Get example queries that demonstrate the agent's capabilities
    """
    return {
        "basic_queries": [
            {
                "query": "List all jobs in the database",
                "description": "Simple data retrieval using SQL",
                "expected_tools": ["sql_tool"]
            },
            {
                "query": "What's the total estimated cost for Hammond project?",
                "description": "Job lookup + aggregation using SQL",
                "expected_tools": ["sql_tool"]
            },
            {
                "query": "How many estimate rows does Mall project have?",
                "description": "Count query using SQL",
                "expected_tools": ["sql_tool"]
            }
        ],
        "semantic_queries": [
            {
                "query": "What's the total cleanup cost for Hammond?",
                "description": "Semantic filtering (cleanup materials) + SQL aggregation",
                "expected_tools": ["sql_tool", "vector_tool"]
            },
            {
                "query": "Show me electrical work estimates for Mall project",
                "description": "Semantic matching for 'electrical work'",
                "expected_tools": ["sql_tool", "vector_tool"]
            },
            {
                "query": "Find all kitchen-related items in the estimate",
                "description": "Semantic search for kitchen work",
                "expected_tools": ["sql_tool", "vector_tool"]
            }
        ],
        "exact_filter_queries": [
            {
                "query": "Show me all allowances for Hammond",
                "description": "Filter by row_type = 'allowance'",
                "expected_tools": ["sql_tool"]
            },
            {
                "query": "What's in cost code 01-5020 for Mall?",
                "description": "Exact cost code match",
                "expected_tools": ["sql_tool"]
            },
            {
                "query": "Total for Roof Deck area in Hammond",
                "description": "Exact area match",
                "expected_tools": ["sql_tool"]
            }
        ],
        "comparison_queries": [
            {
                "query": "Compare estimated vs budgeted costs for Hammond",
                "description": "Calculate variance across all estimates",
                "expected_tools": ["sql_tool"]
            },
            {
                "query": "What's over budget in the Mall project?",
                "description": "Find rows where estimated > budgeted",
                "expected_tools": ["sql_tool"]
            }
        ],
        "complex_queries": [
            {
                "query": "What's the total for cleanup materials in Kitchen area of Hammond?",
                "description": "Hybrid: exact area filter + semantic 'cleanup' match",
                "expected_tools": ["sql_tool", "vector_tool"]
            },
            {
                "query": "Show me all electrical work allowances for Mall",
                "description": "Hybrid: exact row_type filter + semantic 'electrical' match",
                "expected_tools": ["sql_tool", "vector_tool"]
            }
        ]
    }


@app.get("/system/info")
async def system_info():
    """
    Get detailed system information
    """
    try:
        from db_connection import get_table_row_counts, get_db
        
        # Database info
        row_counts = get_table_row_counts()
        pool_status = get_db().get_pool_status()
        
        # Agent info
        stats = query_processor.get_statistics() if query_processor else {}
        
        return {
            "version": "2.0.0",
            "architecture": {
                "database": "PostgreSQL",
                "vector_db": "ChromaDB",
                "agent_framework": "LangChain",
                "llm": "GPT-4-Turbo"
            },
            "database": {
                "tables": row_counts,
                "connection_pool": pool_status
            },
            "agent": {
                "statistics": stats,
                "tools": ["SQL Tool (read-only)", "Vector Tool (semantic filtering)"]
            },
            "capabilities": [
                "Exact filtering (job_id, cost_code, area, row_type)",
                "Semantic matching (natural language concepts)",
                "Hybrid search (SQL + Vector)",
                "Multi-step reasoning",
                "Automatic calculation",
                "Conversation history"
            ]
        }
    
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# MIGRATION HELPER ENDPOINTS (Optional)
# ==========================================

@app.post("/admin/migrate-job")
async def trigger_job_migration(
    company_id: str = Query(..., description="Company ID"),
    job_id: str = Query(..., description="Job ID"),
    background_tasks: BackgroundTasks = None
):
    """
    Trigger ETL migration for a single job (admin endpoint)
    
    This runs the ETL script to migrate one job from Firestore to PostgreSQL
    """
    try:
        async def run_migration():
            from etl_migration import ETLMigration
            from database import initialize_firebase
            
            # Initialize Firebase
            initialize_firebase()
            
            # Run migration
            etl = ETLMigration()
            result = await etl.migrate_job(company_id, job_id, skip_if_exists=False)
            
            logger.info(f"Migration completed: {result}")
            return result
        
        if background_tasks:
            background_tasks.add_task(run_migration)
            return {
                "message": "Migration started in background",
                "company_id": company_id,
                "job_id": job_id
            }
        else:
            result = await run_migration()
            return {
                "message": "Migration completed",
                "result": result
            }
    
    except Exception as e:
        logger.error(f"Migration trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "status_code": 500
    }

# Add this after the /stats endpoints and before the /jobs endpoint

@app.get("/debug/chromadb")
async def debug_chromadb():
    """
    Debug endpoint to check ChromaDB contents and test searches
    
    This helps diagnose why semantic search isn't finding matches
    """
    try:
        from tools.vector_tool import vector_tool
        from database import get_chroma_collection
        
        # Get collection info
        collection_info = vector_tool.debug_collection_info()
        
        # Test search for roofing with various thresholds
        roofing_tests = {}
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            result = vector_tool.search_by_text(
                semantic_query="roofing roof shingles",
                data_type="estimate",
                n_results=10,
                similarity_threshold=threshold
            )
            roofing_tests[f"threshold_{threshold}"] = {
                'found': result['total_found'],
                'success': result['success']
            }
        
        # Test with specific job_id
        job_specific = vector_tool.search_by_text(
            semantic_query="roofing",
            data_type="estimate",
            job_id="4ZppggAAJuJMZNB8f2ZT",
            n_results=100,
            similarity_threshold=0.3
        )
        
        # Get a sample of ALL documents for the job
        collection = get_chroma_collection()
        job_docs = collection.get(
            where={"job_id": "4ZppggAAJuJMZNB8f2ZT"},
            limit=10,
            include=['metadatas', 'documents']
        )
        
        return {
            "collection_info": collection_info,
            "roofing_search_tests": roofing_tests,
            "job_specific_search": {
                "found": job_specific['total_found'],
                "results": job_specific.get('results', [])[:3] if job_specific['success'] else []
            },
            "job_documents_sample": {
                "count": len(job_docs['ids']) if job_docs['ids'] else 0,
                "sample": job_docs['documents'][:3] if job_docs.get('documents') else []
            },
            "cache_stats": vector_tool.get_cache_stats()
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/debug/test-semantic")
async def test_semantic_search(
    query: str = Query(..., description="Semantic search query"),
    threshold: float = Query(0.5, description="Similarity threshold"),
    job_id: Optional[str] = Query(None, description="Optional job ID filter")
):
    """
    Test semantic search directly with custom parameters
    """
    try:
        from tools.vector_tool import vector_tool
        
        result = vector_tool.search_by_text(
            semantic_query=query,
            data_type="estimate",
            job_id=job_id,
            n_results=20,
            similarity_threshold=threshold
        )
        
        return {
            "query": query,
            "threshold": threshold,
            "job_id": job_id,
            "success": result['success'],
            "total_found": result['total_found'],
            "results": result.get('results', [])[:10]  # First 10 results
        }
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/debug/test-simple")
async def test_simple():
    """Simple test to see what's in ChromaDB"""
    try:
        from database import get_chroma_collection
        collection = get_chroma_collection()
        
        # Just get ANY documents
        sample = collection.query(
            query_texts=["roofing"],
            n_results=5,
            include=['distances', 'documents']
        )
        
        results = []
        if sample['ids'] and sample['ids'][0]:
            for i in range(len(sample['ids'][0])):
                results.append({
                    'id': sample['ids'][0][i],
                    'distance': sample['distances'][0][i],
                    'similarity': 1.0 - sample['distances'][0][i],
                    'text_preview': sample['documents'][0][i][:100] if sample['documents'][0] else ""
                })
        
        return {
            "query": "roofing",
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Agentic RAG API on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )