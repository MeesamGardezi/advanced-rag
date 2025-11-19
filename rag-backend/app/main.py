"""
FastAPI application entry point for Construction RAG backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app.config import settings

# Create FastAPI app
app = FastAPI(
    title="Construction RAG API",
    description="RAG system for querying construction job estimates and budget comparisons",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Construction RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("🚀 Starting Construction RAG API...")
    print(f"📊 Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"🤖 OpenAI Model: {settings.openai_model}")
    print(f"🔢 Embedding Model: {settings.embedding_model}")
    print("✅ API is ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Shutting down Construction RAG API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )