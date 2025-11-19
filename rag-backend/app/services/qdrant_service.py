"""
Qdrant service for vector database operations.
Handles collection management, vector storage, and similarity search.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny
)
from typing import List, Dict, Optional
from app.config import settings
from app.models import DocumentChunk
import uuid


class QdrantService:
    """Service for interacting with Qdrant vector database."""
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.client = None
        self.collection_name = settings.qdrant_collection_name
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant server."""
        try:
            print(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )
            print("✅ Connected to Qdrant successfully")
            self._ensure_collection_exists()
        except Exception as e:
            print(f"❌ Error connecting to Qdrant: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Collection '{self.collection_name}' created successfully")
            else:
                print(f"✅ Collection '{self.collection_name}' already exists")
        except Exception as e:
            print(f"❌ Error ensuring collection exists: {e}")
            raise
    
    def insert_vectors(
        self,
        chunks: List[DocumentChunk],
        vectors: List[List[float]]
    ) -> int:
        """
        Insert document chunks with their vectors into Qdrant.
        
        Args:
            chunks: List of document chunks with content and metadata
            vectors: List of embedding vectors corresponding to chunks
            
        Returns:
            Number of vectors inserted
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")
        
        if not chunks:
            return 0
        
        try:
            points = []
            for chunk, vector in zip(chunks, vectors):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "content": chunk.content,
                        **chunk.metadata
                    }
                )
                points.append(point)
            
            # Upload points in batches for efficiency
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            print(f"✅ Inserted {len(points)} vectors into Qdrant")
            return len(points)
            
        except Exception as e:
            print(f"❌ Error inserting vectors: {e}")
            raise
    
    def search_similar(
        self,
        query_vector: List[float],
        company_id: str,
        job_id: Optional[str] = None,
        document_type: Optional[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Search for similar vectors with filtering.
        
        Args:
            query_vector: Query embedding vector
            company_id: Company ID for filtering
            job_id: Optional job ID for filtering
            document_type: Optional document type filter ('estimate' or 'comparison')
            top_k: Number of results to return
            
        Returns:
            List of search results with content and metadata
        """
        if top_k is None:
            top_k = settings.top_k_results
        
        try:
            # Build filter conditions
            must_conditions = [
                FieldCondition(
                    key="company_id",
                    match=MatchValue(value=company_id)
                )
            ]
            
            if job_id:
                must_conditions.append(
                    FieldCondition(
                        key="job_id",
                        match=MatchValue(value=job_id)
                    )
                )
            
            if document_type:
                must_conditions.append(
                    FieldCondition(
                        key="document_type",
                        match=MatchValue(value=document_type)
                    )
                )
            
            # Create filter
            search_filter = Filter(must=must_conditions) if must_conditions else None
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=top_k,
                score_threshold=settings.similarity_threshold
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "content": result.payload.get("content", ""),
                    "metadata": {
                        k: v for k, v in result.payload.items() if k != "content"
                    },
                    "score": result.score
                })
            
            print(f"✅ Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            print(f"❌ Error searching vectors: {e}")
            raise
    
    def delete_by_job_id(self, job_id: str) -> int:
        """
        Delete all vectors associated with a job ID.
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            Number of vectors deleted
        """
        try:
            # Search for all points with this job_id
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="job_id",
                            match=MatchValue(value=job_id)
                        )
                    ]
                ),
                limit=10000  # Max points to retrieve
            )
            
            points, _ = search_results
            point_ids = [point.id for point in points]
            
            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                print(f"✅ Deleted {len(point_ids)} vectors for job_id: {job_id}")
                return len(point_ids)
            else:
                print(f"ℹ️ No vectors found for job_id: {job_id}")
                return 0
                
        except Exception as e:
            print(f"❌ Error deleting vectors: {e}")
            raise
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            self.client.get_collections()
            return True
        except:
            return False


# Global instance
_qdrant_service = None


def get_qdrant_service() -> QdrantService:
    """Get or create the global Qdrant service instance."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service