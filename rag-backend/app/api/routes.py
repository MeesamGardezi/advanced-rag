"""
API routes for the Construction RAG system.
"""

from fastapi import APIRouter, HTTPException, status
from app.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    DeleteResponse,
    HealthResponse
)
from app.services import (
    get_ingestion_service,
    get_rag_service,
    get_qdrant_service,
    get_firebase_service
)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_job(request: IngestRequest):
    """
    Ingest job data into the vector database.
    
    Can either:
    - Fetch data from Firebase (set fetch_from_firebase=True)
    - Use provided job_data and comparison_data
    """
    try:
        ingestion_service = get_ingestion_service()
        
        result = ingestion_service.ingest_job(
            company_id=request.company_id,
            job_id=request.job_id,
            job_data=request.job_data,
            comparison_data=request.comparison_data,
            fetch_from_firebase=request.fetch_from_firebase
        )
        
        if result['status'] == 'error':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['message']
            )
        
        return IngestResponse(
            status=result['status'],
            chunks_created=result['chunks_created'],
            job_id=request.job_id,
            message=result.get('message')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting job: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a natural language question.
    """
    try:
        rag_service = get_rag_service()
        response = rag_service.query(request)
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.delete("/job/{job_id}", response_model=DeleteResponse)
async def delete_job(job_id: str):
    """
    Delete all data for a specific job from the vector database.
    """
    try:
        qdrant_service = get_qdrant_service()
        deleted_count = qdrant_service.delete_by_job_id(job_id)
        
        return DeleteResponse(
            status="success",
            deleted_count=deleted_count,
            job_id=job_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting job: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health of all services.
    """
    try:
        qdrant_service = get_qdrant_service()
        rag_service = get_rag_service()
        firebase_service = get_firebase_service()
        
        qdrant_status = "connected" if qdrant_service.health_check() else "disconnected"
        openai_status = "connected" if rag_service.health_check() else "disconnected"
        firebase_status = "connected" if firebase_service.health_check() else "disconnected"
        
        overall_status = "healthy" if all([
            qdrant_status == "connected",
            openai_status == "connected",
            firebase_status == "connected"
        ]) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            qdrant=qdrant_status,
            openai=openai_status,
            firebase=firebase_status
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking health: {str(e)}"
        )


@router.get("/collection/info")
async def get_collection_info():
    """
    Get information about the Qdrant collection.
    """
    try:
        qdrant_service = get_qdrant_service()
        info = qdrant_service.get_collection_info()
        return info
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting collection info: {str(e)}"
        )