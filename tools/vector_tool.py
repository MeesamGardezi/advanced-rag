"""
SemanticFilterTool: Vector-based semantic filtering of SQL candidates
Filters candidate rows using text similarity in ChromaDB
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import logging

from database import get_chroma_collection
from embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SemanticFilterTool:
    """
    Semantic filtering tool using vector embeddings
    Takes candidate rows from SQL and filters based on text similarity
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize semantic filter tool
        
        Args:
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
        """
        self.name = "vector_tool"
        self.description = """Filter candidate rows using semantic text similarity.
        
Use this tool when you need to:
1. Filter rows by semantic meaning (e.g., "cleanup materials", "electrical work")
2. Match natural language concepts to technical descriptions
3. Find semantically similar items across different wordings

Input:
- candidate_ids: List of row IDs from SQL query
- semantic_query: Natural language description of what you're looking for
- data_type: Type of data ('estimate', 'flooring_estimate', 'schedule', 'consumed')

Output:
- List of row IDs that semantically match the query

Examples:
- Filter estimates for "cleanup materials" (matches "debris removal", "site cleaning", etc.)
- Find "electrical work" (matches "wiring", "outlets", "lighting", etc.)
- Identify "kitchen renovation" items (matches various kitchen-related tasks)

DO NOT use for exact matching - use SQL tool for that.
"""
        
        self.collection = get_chroma_collection()
        self.embedding_service = EmbeddingService()
        self.similarity_threshold = similarity_threshold
        
        # Cache for frequent queries
        self._query_cache: Dict[str, List[str]] = {}
        self._cache_max_size = 100
    
    def _get_cache_key(self, semantic_query: str, data_type: str, threshold: float) -> str:
        """Generate cache key for query"""
        return f"{semantic_query.lower()}:{data_type}:{threshold:.2f}"
    
    def _add_to_cache(self, key: str, result: List[str]):
        """Add result to cache with size limit"""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[key] = result
    
    def _extract_text_fields(self, metadata: Dict[str, Any], data_type: str) -> str:
        """
        Extract searchable text from metadata based on data type
        
        Args:
            metadata: Document metadata from ChromaDB
            data_type: Type of data
        
        Returns:
            Combined text string for similarity comparison
        """
        text_parts = []
        
        if data_type == 'estimate':
            # Combine semantic fields from estimate
            text_parts.extend([
                metadata.get('area', ''),
                metadata.get('task_scope', ''),
                metadata.get('description', ''),
                metadata.get('cost_code', ''),
                metadata.get('notes_remarks', '')
            ])
        
        elif data_type == 'flooring_estimate':
            # Combine semantic fields from flooring estimate
            text_parts.extend([
                metadata.get('item_material_name', ''),
                metadata.get('vendor', ''),
                metadata.get('brand', ''),
                metadata.get('floor_type_id', ''),
                metadata.get('notes_remarks', '')
            ])
        
        elif data_type == 'schedule':
            # Combine semantic fields from schedule
            text_parts.extend([
                metadata.get('task', ''),
                metadata.get('task_type', '')
            ])
        
        elif data_type == 'consumed':
            # Combine semantic fields from consumed
            text_parts.extend([
                metadata.get('cost_code', ''),
                metadata.get('category', '')
            ])
        
        # Filter out empty strings and join
        return " ".join([part for part in text_parts if part])
    
    def filter_candidates(
        self,
        candidate_ids: List[str],
        semantic_query: str,
        data_type: str,
        job_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Filter candidate rows using semantic similarity
        
        Args:
            candidate_ids: List of document IDs from SQL query
            semantic_query: Natural language query for semantic matching
            data_type: Type of data to filter
            job_id: Optional job ID for additional filtering
            similarity_threshold: Override default threshold
        
        Returns:
            Dictionary with:
            - success: bool
            - matching_ids: List of IDs that match semantically
            - scores: Dict mapping ID to similarity score
            - total_candidates: Number of input candidates
            - filtered_count: Number of matches
        """
        logger.info(f"🔍 Vector Tool filtering candidates:")
        logger.info(f"   Candidates: {len(candidate_ids)}")
        logger.info(f"   Query: '{semantic_query}'")
        logger.info(f"   Data type: {data_type}")
        
        try:
            # Use provided threshold or default
            threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
            
            # Check cache
            cache_key = self._get_cache_key(semantic_query, data_type, threshold)
            if cache_key in self._query_cache:
                cached_ids = self._query_cache[cache_key]
                # Filter cached results by candidate_ids
                matching_ids = [id for id in cached_ids if id in candidate_ids]
                logger.info(f"✅ Cache hit: {len(matching_ids)} matches from cache")
                
                return {
                    'success': True,
                    'matching_ids': matching_ids,
                    'scores': {},  # Scores not available from cache
                    'total_candidates': len(candidate_ids),
                    'filtered_count': len(matching_ids),
                    'from_cache': True
                }
            
            # If no candidates provided, return empty
            if not candidate_ids:
                logger.warning("⚠️  No candidates provided for filtering")
                return {
                    'success': True,
                    'matching_ids': [],
                    'scores': {},
                    'total_candidates': 0,
                    'filtered_count': 0
                }
            
            # Build ChromaDB where filter
            where_filter = {'data_type': data_type}
            
            if job_id:
                where_filter['job_id'] = job_id
            
            # Query ChromaDB with semantic query
            # Request more results to ensure we get all candidates
            n_results = min(len(candidate_ids) * 2, 500)
            
            results = self.collection.query(
                query_texts=[semantic_query],
                n_results=n_results,
                where=where_filter,
                include=['metadatas', 'distances']
            )
            
            # Extract results
            returned_ids = results['ids'][0] if results['ids'] else []
            distances = results['distances'][0] if results['distances'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            logger.info(f"   ChromaDB returned: {len(returned_ids)} results")
            
            # Convert distances to similarity scores (1 - normalized_distance)
            # ChromaDB uses cosine distance, so similarity = 1 - distance
            scores = {}
            matching_ids = []
            
            for doc_id, distance, metadata in zip(returned_ids, distances, metadatas):
                # Only consider candidates from our input list
                if doc_id not in candidate_ids:
                    continue
                
                # Convert distance to similarity score
                similarity = 1.0 - distance
                scores[doc_id] = similarity
                
                # Filter by threshold
                if similarity >= threshold:
                    matching_ids.append(doc_id)
                    logger.debug(f"   ✅ Match: {doc_id} (score: {similarity:.3f})")
                else:
                    logger.debug(f"   ❌ Below threshold: {doc_id} (score: {similarity:.3f})")
            
            # Sort matching_ids by score (highest first)
            matching_ids.sort(key=lambda x: scores[x], reverse=True)
            
            # Cache results
            self._add_to_cache(cache_key, matching_ids)
            
            logger.info(f"✅ Semantic filtering complete:")
            logger.info(f"   Matches: {len(matching_ids)} / {len(candidate_ids)} candidates")
            logger.info(f"   Threshold: {threshold}")
            
            return {
                'success': True,
                'matching_ids': matching_ids,
                'scores': scores,
                'total_candidates': len(candidate_ids),
                'filtered_count': len(matching_ids),
                'from_cache': False,
                'threshold_used': threshold
            }
        
        except Exception as e:
            error_msg = f"Semantic filtering error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'matching_ids': [],
                'scores': {},
                'total_candidates': len(candidate_ids),
                'filtered_count': 0,
                'error': error_msg
            }
    
    def search_by_text(
        self,
        semantic_query: str,
        data_type: str,
        job_id: Optional[str] = None,
        n_results: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Search directly in ChromaDB without pre-filtering
        (Useful when you don't have SQL candidates)
        
        Args:
            semantic_query: Natural language query
            data_type: Type of data to search
            job_id: Optional job ID filter
            n_results: Number of results to return
            similarity_threshold: Override default threshold
        
        Returns:
            Dictionary with search results
        """
        logger.info(f"🔍 Vector search: '{semantic_query}' in {data_type}")
        
        try:
            threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
            
            # Build where filter
            where_filter = {'data_type': data_type}
            
            if job_id:
                where_filter['job_id'] = job_id
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[semantic_query],
                n_results=n_results,
                where=where_filter,
                include=['metadatas', 'distances', 'documents']
            )
            
            # Extract and format results
            ids = results['ids'][0] if results['ids'] else []
            distances = results['distances'][0] if results['distances'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            documents = results['documents'][0] if results['documents'] else []
            
            # Filter by threshold and format
            matching_results = []
            
            for doc_id, distance, metadata, document in zip(ids, distances, metadatas, documents):
                similarity = 1.0 - distance
                
                if similarity >= threshold:
                    matching_results.append({
                        'id': doc_id,
                        'similarity': similarity,
                        'metadata': metadata,
                        'text_preview': document[:200] + "..." if len(document) > 200 else document
                    })
            
            logger.info(f"✅ Found {len(matching_results)} matches above threshold {threshold}")
            
            return {
                'success': True,
                'results': matching_results,
                'total_found': len(matching_results)
            }
        
        except Exception as e:
            error_msg = f"Vector search error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'results': [],
                'total_found': 0,
                'error': error_msg
            }
    
    def get_similar_items(
        self,
        reference_id: str,
        n_results: int = 5,
        same_job_only: bool = True
    ) -> Dict[str, Any]:
        """
        Find items similar to a reference item
        
        Args:
            reference_id: ID of reference document
            n_results: Number of similar items to return
            same_job_only: Only return items from same job
        
        Returns:
            Dictionary with similar items
        """
        logger.info(f"🔍 Finding items similar to: {reference_id}")
        
        try:
            # Get reference document
            ref_doc = self.collection.get(ids=[reference_id])
            
            if not ref_doc['ids']:
                return {
                    'success': False,
                    'results': [],
                    'error': f"Reference document {reference_id} not found"
                }
            
            ref_metadata = ref_doc['metadatas'][0]
            ref_document = ref_doc['documents'][0]
            
            # Build where filter
            where_filter = {'data_type': ref_metadata.get('data_type')}
            
            if same_job_only:
                where_filter['job_id'] = ref_metadata.get('job_id')
            
            # Query for similar items
            results = self.collection.query(
                query_texts=[ref_document],
                n_results=n_results + 1,  # +1 because reference will be included
                where=where_filter,
                include=['metadatas', 'distances']
            )
            
            # Extract results (excluding reference itself)
            similar_items = []
            
            ids = results['ids'][0] if results['ids'] else []
            distances = results['distances'][0] if results['distances'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            for doc_id, distance, metadata in zip(ids, distances, metadatas):
                if doc_id == reference_id:
                    continue  # Skip reference itself
                
                similarity = 1.0 - distance
                similar_items.append({
                    'id': doc_id,
                    'similarity': similarity,
                    'metadata': metadata
                })
            
            logger.info(f"✅ Found {len(similar_items)} similar items")
            
            return {
                'success': True,
                'results': similar_items,
                'reference_id': reference_id
            }
        
        except Exception as e:
            error_msg = f"Similar items search error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'results': [],
                'error': error_msg
            }
    
    def clear_cache(self):
        """Clear the query cache"""
        self._query_cache.clear()
        logger.info("✅ Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self._query_cache),
            'max_size': self._cache_max_size,
            'cached_queries': list(self._query_cache.keys())
        }


# Singleton instance for easy import
vector_tool = SemanticFilterTool()


if __name__ == "__main__":
    # Test the Vector tool
    print("🧪 Testing SemanticFilterTool...\n")
    
    tool = SemanticFilterTool(similarity_threshold=0.7)
    
    # Test 1: Search by text (no pre-filtering)
    print("Test 1: Direct semantic search")
    result = tool.search_by_text(
        semantic_query="cleanup materials",
        data_type="estimate",
        n_results=5
    )
    print(f"Success: {result['success']}")
    print(f"Results: {result['total_found']}")
    
    if result['results']:
        print("\nTop match:")
        top = result['results'][0]
        print(f"  ID: {top['id']}")
        print(f"  Similarity: {top['similarity']:.3f}")
        print(f"  Preview: {top['text_preview']}\n")
    
    # Test 2: Filter candidates (hybrid approach)
    print("Test 2: Filter SQL candidates with semantic query")
    
    # Simulate SQL candidates
    candidate_ids = [
        "comp123_job456_est_row_1",
        "comp123_job456_est_row_2",
        "comp123_job456_est_row_3"
    ]
    
    result = tool.filter_candidates(
        candidate_ids=candidate_ids,
        semantic_query="electrical work",
        data_type="estimate"
    )
    print(f"Success: {result['success']}")
    print(f"Candidates: {result['total_candidates']}")
    print(f"Matches: {result['filtered_count']}")
    
    if result['matching_ids']:
        print("\nMatching IDs:")
        for match_id in result['matching_ids']:
            score = result['scores'].get(match_id, 0)
            print(f"  {match_id} (score: {score:.3f})")
    
    # Test 3: Cache statistics
    print("\nTest 3: Cache stats")
    stats = tool.get_cache_stats()
    print(f"Cache size: {stats['cache_size']}/{stats['max_size']}")
    
    print("\n✅ Vector Tool tests complete")