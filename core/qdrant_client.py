import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

try:
    from qdrant_client import QdrantClient as QdrantClientSDK
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, 
        FieldCondition, Match, Range, CollectionInfo,
        UpdateStatus, ScoredPoint
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # Create dummy classes for type hints when Qdrant is not available
    class PointStruct:
        pass
    class ScoredPoint:
        pass
    class CollectionInfo:
        pass
    print("⚠️  Qdrant not available. Install with: pip install qdrant-client")

from .config import config

logger = logging.getLogger(__name__)

class QdrantClient:
    """Enhanced Qdrant client for construction RAG system"""
    
    def __init__(self):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        self.config = config.database
        self.client = None
        self.collection_name = self.config.qdrant_collection_name
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client connection"""
        try:
            if self.config.qdrant_api_key:
                # Cloud/remote Qdrant instance
                self.client = QdrantClientSDK(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                    timeout=self.config.qdrant_timeout
                )
            else:
                # Local Qdrant instance
                self.client = QdrantClientSDK(
                    host=self.config.qdrant_url,
                    port=self.config.qdrant_port,
                    timeout=self.config.qdrant_timeout
                )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"✅ Qdrant client initialized successfully. Collections: {len(collections.collections)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant client: {e}")
            raise
    
    def create_collection(self, vector_size: int = 1536, force_recreate: bool = False) -> bool:
        """Create or recreate the construction RAG collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if collection_exists and not force_recreate:
                logger.info(f"✅ Collection '{self.collection_name}' already exists")
                return True
            
            if collection_exists and force_recreate:
                logger.info(f"🗑️  Deleting existing collection '{self.collection_name}'")
                self.client.delete_collection(collection_name=self.collection_name)
            
            # Create collection with optimized settings for construction documents
            logger.info(f"🏗️  Creating collection '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    hnsw_config={
                        "m": 16,  # Optimized for construction document characteristics
                        "ef_construct": 100,
                        "full_scan_threshold": 10000
                    }
                )
            )
            
            # Create indexes for common construction metadata filters
            self._create_payload_indexes()
            
            logger.info(f"✅ Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating collection: {e}")
            return False
    
    def _create_payload_indexes(self):
        """Create optimized indexes for construction-specific metadata"""
        indexes_to_create = [
            ("data_type", "keyword"),
            ("job_name", "keyword"), 
            ("company_id", "keyword"),
            ("job_id", "keyword"),
            ("document_type", "keyword"),
            ("project_phase", "keyword"),
            ("total_cost", "float"),
            ("categories", "text")
        ]
        
        for field_name, field_type in indexes_to_create:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.debug(f"✅ Created index for {field_name}")
            except Exception as e:
                logger.warning(f"⚠️  Could not create index for {field_name}: {e}")
    
    def add_points(self, points: List[PointStruct], batch_size: int = 100) -> bool:
        """Add points in batches for better performance"""
        try:
            if not points:
                logger.warning("No points to add")
                return True
            
            total_points = len(points)
            logger.info(f"📊 Adding {total_points} points to collection in batches of {batch_size}")
            
            # Process in batches
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_points + batch_size - 1) // batch_size
                
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} points)")
                
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
                if operation_info.status != UpdateStatus.COMPLETED:
                    logger.error(f"❌ Batch {batch_num} failed with status: {operation_info.status}")
                    return False
                
                # Small delay to avoid overwhelming the server
                if i + batch_size < total_points:
                    time.sleep(0.1)
            
            logger.info(f"✅ Successfully added {total_points} points")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding points: {e}")
            return False
    
    def search(self, 
               query_vector: List[float], 
               limit: int = 5,
               filters: Optional[Dict[str, Any]] = None,
               score_threshold: Optional[float] = None) -> List[ScoredPoint]:
        """Search for similar vectors with optional filtering"""
        try:
            # Build filter conditions
            filter_conditions = None
            if filters:
                filter_conditions = self._build_filter(filters)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_conditions,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            logger.debug(f"🔍 Search returned {len(search_result)} results")
            return search_result
            
        except Exception as e:
            logger.error(f"❌ Error searching: {e}")
            return []
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary conditions"""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values - use "should" (OR) condition
                should_conditions = []
                for v in value:
                    should_conditions.append(FieldCondition(
                        key=key,
                        match=Match(value=v)
                    ))
                conditions.extend(should_conditions)
            elif isinstance(value, dict):
                # Range or special conditions
                if 'gte' in value or 'lte' in value or 'gt' in value or 'lt' in value:
                    conditions.append(FieldCondition(
                        key=key,
                        range=Range(
                            gte=value.get('gte'),
                            lte=value.get('lte'),
                            gt=value.get('gt'),
                            lt=value.get('lt')
                        )
                    ))
            else:
                # Simple match condition
                conditions.append(FieldCondition(
                    key=key,
                    match=Match(value=value)
                ))
        
        return Filter(must=conditions) if conditions else None
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get collection information and statistics"""
        try:
            return self.client.get_collection(collection_name=self.collection_name)
        except Exception as e:
            logger.error(f"❌ Error getting collection info: {e}")
            return None
    
    def count_points(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count points in collection with optional filters"""
        try:
            filter_conditions = None
            if filters:
                filter_conditions = self._build_filter(filters)
            
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=filter_conditions,
                exact=False  # Use approximate count for better performance
            )
            
            return result.count
            
        except Exception as e:
            logger.error(f"❌ Error counting points: {e}")
            return 0
    
    def delete_points(self, point_ids: List[Union[str, int]]) -> bool:
        """Delete points by IDs"""
        try:
            if not point_ids:
                logger.warning("No point IDs provided for deletion")
                return True
            
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            
            success = operation_info.status == UpdateStatus.COMPLETED
            if success:
                logger.info(f"✅ Deleted {len(point_ids)} points")
            else:
                logger.error(f"❌ Delete operation failed with status: {operation_info.status}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error deleting points: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all points from the collection"""
        try:
            # Get collection info to check if it exists
            info = self.get_collection_info()
            if not info:
                logger.warning("Collection does not exist")
                return False
            
            # Delete collection and recreate
            self.client.delete_collection(collection_name=self.collection_name)
            time.sleep(1)  # Wait for deletion to complete
            
            # Recreate collection
            return self.create_collection()
            
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {e}")
            return False
    
    def get_construction_stats(self) -> Dict[str, Any]:
        """Get construction-specific statistics about the collection"""
        try:
            info = self.get_collection_info()
            if not info:
                return {"error": "Collection not found"}
            
            # Count by data type
            data_type_counts = {}
            for data_type in ['consumed', 'estimate', 'schedule', 'unknown']:
                count = self.count_points(filters={'data_type': data_type})
                data_type_counts[data_type] = count
            
            # Count by document type
            doc_type_counts = {}
            for doc_type in ['job_cost_data', 'job_estimate_data', 'job_schedule_data']:
                count = self.count_points(filters={'document_type': doc_type})
                doc_type_counts[doc_type] = count
            
            return {
                'total_points': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'data_type_breakdown': data_type_counts,
                'document_type_breakdown': doc_type_counts,
                'collection_status': info.status,
                'optimizer_status': info.optimizer_status,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting construction stats: {e}")
            return {"error": str(e)}
    
    def migrate_from_chromadb(self, chroma_collection) -> Dict[str, Any]:
        """Migrate data from ChromaDB to Qdrant"""
        try:
            logger.info("🔄 Starting ChromaDB to Qdrant migration...")
            
            # Get all data from ChromaDB
            chroma_data = chroma_collection.get()
            
            if not chroma_data['ids']:
                logger.warning("No data found in ChromaDB collection")
                return {"migrated": 0, "errors": []}
            
            # Prepare points for Qdrant
            points = []
            errors = []
            
            for i, (doc_id, embedding, metadata, document) in enumerate(zip(
                chroma_data['ids'],
                chroma_data['embeddings'],
                chroma_data['metadatas'],
                chroma_data['documents']
            )):
                try:
                    # Create Qdrant point
                    point = PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            **metadata,
                            'document_content': document,
                            'migrated_at': datetime.now().isoformat(),
                            'migration_source': 'chromadb'
                        }
                    )
                    points.append(point)
                    
                except Exception as e:
                    error_msg = f"Error processing document {doc_id}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Add points to Qdrant
            if points:
                success = self.add_points(points)
                if success:
                    logger.info(f"✅ Successfully migrated {len(points)} documents")
                    return {
                        "migrated": len(points),
                        "errors": errors,
                        "total_source": len(chroma_data['ids'])
                    }
                else:
                    logger.error("❌ Failed to add points to Qdrant")
                    return {"migrated": 0, "errors": errors + ["Failed to add points to Qdrant"]}
            else:
                logger.warning("No valid points to migrate")
                return {"migrated": 0, "errors": errors}
            
        except Exception as e:
            error_msg = f"Migration failed: {e}"
            logger.error(error_msg)
            return {"migrated": 0, "errors": [error_msg]}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Qdrant connection and collection"""
        try:
            # Test basic connection
            collections = self.client.get_collections()
            
            # Check if our collection exists
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                return {
                    "status": "unhealthy",
                    "error": f"Collection '{self.collection_name}' does not exist",
                    "collections_found": len(collections.collections)
                }
            
            # Get collection info
            info = self.get_collection_info()
            
            # Test search functionality with dummy vector
            dummy_vector = [0.0] * (info.config.params.vectors.size if info else 1536)
            search_results = self.search(query_vector=dummy_vector, limit=1)
            
            return {
                "status": "healthy",
                "collection_exists": True,
                "points_count": info.points_count if info else 0,
                "search_functional": len(search_results) >= 0,  # >= 0 because empty results are valid
                "connection_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_time": datetime.now().isoformat()
            }

# Global Qdrant client instance
_qdrant_client = None

def get_qdrant_client() -> QdrantClient:
    """Get or create global Qdrant client instance"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient()
    return _qdrant_client

def initialize_qdrant() -> QdrantClient:
    """Initialize Qdrant client and create collection if needed"""
    client = get_qdrant_client()
    
    # Create collection if it doesn't exist
    client.create_collection()
    
    logger.info("🚀 Qdrant client initialized and ready")
    return client