import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import asyncio

from core.config import config
from core.qdrant_client import get_qdrant_client
from embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Unified search result from hybrid retrieval"""
    id: str
    content: str
    metadata: Dict[str, Any]
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    combined_score: float = 0.0
    source: str = "hybrid"  # 'semantic', 'keyword', or 'hybrid'

class ConstructionKeywordExtractor:
    """Extract construction-specific keywords and terms"""
    
    def __init__(self):
        # Construction-specific terminology
        self.construction_terms = {
            'materials': [
                'concrete', 'steel', 'lumber', 'drywall', 'insulation', 'roofing',
                'electrical', 'plumbing', 'hvac', 'flooring', 'paint', 'siding'
            ],
            'activities': [
                'excavation', 'foundation', 'framing', 'rough-in', 'finish',
                'installation', 'demolition', 'site prep', 'cleanup'
            ],
            'roles': [
                'contractor', 'subcontractor', 'electrician', 'plumber',
                'carpenter', 'mason', 'roofer', 'painter'
            ],
            'measurements': [
                'sq ft', 'linear ft', 'cubic yard', 'ton', 'gallon', 'hour',
                'square foot', 'linear foot', 'cubic feet'
            ],
            'locations': [
                'basement', 'attic', 'kitchen', 'bathroom', 'bedroom',
                'living room', 'garage', 'exterior', 'interior'
            ]
        }
        
        # Cost code patterns
        self.cost_code_pattern = re.compile(r'\b\d{3}[SMLO]?\b')
        
        # Dollar amount patterns
        self.dollar_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?')
        
        # Measurement patterns
        self.measurement_pattern = re.compile(r'\b\d+(?:\.\d+)?\s*(?:sq\s*ft|linear\s*ft|cubic\s*yard|ton|gallon|hour|lf|sf|cy)\b', re.IGNORECASE)
    
    def extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract construction-specific keywords from text"""
        text_lower = text.lower()
        
        keywords = {
            'construction_terms': [],
            'cost_codes': [],
            'dollar_amounts': [],
            'measurements': [],
            'important_phrases': []
        }
        
        # Extract construction terms
        for category, terms in self.construction_terms.items():
            for term in terms:
                if term in text_lower:
                    keywords['construction_terms'].append(term)
        
        # Extract cost codes
        cost_codes = self.cost_code_pattern.findall(text)
        keywords['cost_codes'] = list(set(cost_codes))
        
        # Extract dollar amounts
        dollar_amounts = self.dollar_pattern.findall(text)
        keywords['dollar_amounts'] = list(set(dollar_amounts))
        
        # Extract measurements
        measurements = self.measurement_pattern.findall(text)
        keywords['measurements'] = list(set(measurements))
        
        # Extract important phrases (noun phrases, technical terms)
        important_phrases = self._extract_phrases(text)
        keywords['important_phrases'] = important_phrases[:10]  # Top 10 phrases
        
        return keywords
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract important phrases from text"""
        # Simple phrase extraction - in production, consider using spaCy or similar
        phrases = []
        
        # Extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', text)
        phrases.extend(quoted_phrases)
        
        # Extract capitalized phrases (likely proper nouns)
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        phrases.extend(capitalized_phrases)
        
        # Extract technical terms (word followed by specific suffixes)
        technical_terms = re.findall(r'\b\w+(?:ing|tion|sion|ment|ance|ence|ity|ness)\b', text)
        phrases.extend(technical_terms)
        
        return list(set(phrases))

class BM25Scorer:
    """BM25 scoring for keyword-based retrieval"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_frequencies = defaultdict(int)
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.total_docs = 0
        self.vocabulary = set()
    
    def fit(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents"""
        self.total_docs = len(documents)
        total_length = 0
        
        # Build document statistics
        for doc in documents:
            doc_id = doc['id']
            content = doc.get('content', '') + ' ' + str(doc.get('metadata', {}))
            
            # Tokenize and count terms
            terms = self._tokenize(content)
            self.doc_lengths[doc_id] = len(terms)
            total_length += len(terms)
            
            # Count term frequencies
            term_counts = Counter(terms)
            for term, count in term_counts.items():
                self.vocabulary.add(term)
                self.doc_frequencies[term] += 1
        
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        logger.info(f"🔍 BM25 index built: {self.total_docs} docs, {len(self.vocabulary)} unique terms")
    
    def score(self, query: str, documents: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Score documents using BM25"""
        query_terms = self._tokenize(query)
        if not query_terms:
            return [(doc['id'], 0.0) for doc in documents]
        
        scores = []
        
        for doc in documents:
            doc_id = doc['id']
            content = doc.get('content', '') + ' ' + str(doc.get('metadata', {}))
            doc_terms = self._tokenize(content)
            doc_length = len(doc_terms)
            
            # Calculate BM25 score
            score = 0.0
            term_counts = Counter(doc_terms)
            
            for term in query_terms:
                if term in term_counts:
                    tf = term_counts[term]
                    df = self.doc_frequencies.get(term, 0)
                    
                    if df > 0:
                        idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))
                        score += idf * (tf * (self.k1 + 1)) / (
                            tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                        )
            
            scores.append((doc_id, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]
        
        return tokens

class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.qdrant_client = get_qdrant_client()
        self.keyword_extractor = ConstructionKeywordExtractor()
        self.bm25_scorer = BM25Scorer()
        
        # Weights for combining scores
        self.semantic_weight = config.retrieval.semantic_weight
        self.keyword_weight = config.retrieval.keyword_weight
        
        # Cache for document corpus (for BM25)
        self.document_cache = {}
        self.cache_valid = False
        
        logger.info(f"🔍 Hybrid retriever initialized (semantic: {self.semantic_weight}, keyword: {self.keyword_weight})")
    
    async def retrieve(self, 
                      query: str,
                      n_results: int = 5,
                      filters: Optional[Dict[str, Any]] = None,
                      strategy: str = "hybrid") -> List[SearchResult]:
        """Perform hybrid retrieval"""
        
        if strategy == "semantic":
            return await self._semantic_only_retrieval(query, n_results, filters)
        elif strategy == "keyword":
            return await self._keyword_only_retrieval(query, n_results, filters)
        else:
            return await self._hybrid_retrieval(query, n_results, filters)
    
    async def _hybrid_retrieval(self, 
                               query: str,
                               n_results: int,
                               filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Perform hybrid semantic + keyword retrieval"""
        try:
            # Perform both searches concurrently
            semantic_task = self._semantic_search(query, n_results * 2, filters)
            keyword_task = self._keyword_search(query, n_results * 2, filters)
            
            semantic_results, keyword_results = await asyncio.gather(semantic_task, keyword_task)
            
            # Combine and rerank results
            combined_results = self._combine_results(semantic_results, keyword_results, query)
            
            # Apply construction-specific reranking
            reranked_results = self._apply_construction_reranking(combined_results, query)
            
            # Return top N results
            return reranked_results[:n_results]
            
        except Exception as e:
            logger.error(f"❌ Hybrid retrieval failed: {e}")
            # Fallback to semantic search
            return await self._semantic_only_retrieval(query, n_results, filters)
    
    async def _semantic_search(self, 
                              query: str,
                              n_results: int,
                              filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Perform semantic search using Qdrant"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                query_vector=query_embedding,
                limit=n_results,
                filters=filters
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                search_result = SearchResult(
                    id=str(result.id),
                    content=result.payload.get('document_content', ''),
                    metadata=result.payload,
                    semantic_score=result.score,
                    combined_score=result.score,
                    source="semantic"
                )
                results.append(search_result)
            
            logger.debug(f"🔍 Semantic search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    async def _keyword_search(self, 
                             query: str,
                             n_results: int,
                             filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Perform keyword search using BM25"""
        try:
            # Ensure BM25 index is ready
            await self._ensure_bm25_index()
            
            # Get all documents that match filters
            candidate_docs = await self._get_candidate_documents(filters)
            
            if not candidate_docs:
                logger.warning("No candidate documents for keyword search")
                return []
            
            # Score documents using BM25
            scored_docs = self.bm25_scorer.score(query, candidate_docs)
            
            # Convert to SearchResult objects
            results = []
            for doc_id, score in scored_docs[:n_results]:
                if score > 0:  # Only include documents with positive scores
                    doc = next((d for d in candidate_docs if d['id'] == doc_id), None)
                    if doc:
                        search_result = SearchResult(
                            id=doc_id,
                            content=doc.get('content', ''),
                            metadata=doc.get('metadata', {}),
                            keyword_score=score,
                            combined_score=score,
                            source="keyword"
                        )
                        results.append(search_result)
            
            logger.debug(f"🔍 Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Keyword search failed: {e}")
            return []
    
    async def _ensure_bm25_index(self):
        """Ensure BM25 index is built and up-to-date"""
        if not self.cache_valid:
            await self._rebuild_bm25_index()
    
    async def _rebuild_bm25_index(self):
        """Rebuild BM25 index from Qdrant collection"""
        try:
            logger.info("🔄 Rebuilding BM25 index...")
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection_info()
            if not collection_info:
                logger.warning("Collection not found for BM25 index")
                return
            
            # For now, we'll use a sample of documents due to potential memory constraints
            # In production, consider using a dedicated keyword search engine
            sample_size = min(1000, collection_info.points_count)
            
            # Get sample documents (this is simplified - in production you'd want pagination)
            dummy_vector = [0.0] * 1536  # Dummy vector for search
            search_results = self.qdrant_client.search(
                query_vector=dummy_vector,
                limit=sample_size
            )
            
            # Prepare documents for BM25
            documents = []
            for result in search_results:
                doc = {
                    'id': str(result.id),
                    'content': result.payload.get('document_content', ''),
                    'metadata': result.payload
                }
                documents.append(doc)
                self.document_cache[doc['id']] = doc
            
            # Build BM25 index
            if documents:
                self.bm25_scorer.fit(documents)
                self.cache_valid = True
                logger.info(f"✅ BM25 index built with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild BM25 index: {e}")
    
    async def _get_candidate_documents(self, filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get candidate documents for keyword search"""
        # For now, return cached documents
        # In production, you'd want to apply filters here
        candidates = list(self.document_cache.values())
        
        # Apply simple filtering if specified
        if filters:
            filtered_candidates = []
            for doc in candidates:
                metadata = doc.get('metadata', {})
                include_doc = True
                
                for key, value in filters.items():
                    if key in metadata:
                        if isinstance(value, list):
                            if metadata[key] not in value:
                                include_doc = False
                                break
                        else:
                            if metadata[key] != value:
                                include_doc = False
                                break
                
                if include_doc:
                    filtered_candidates.append(doc)
            
            candidates = filtered_candidates
        
        return candidates
    
    def _combine_results(self, 
                        semantic_results: List[SearchResult],
                        keyword_results: List[SearchResult],
                        query: str) -> List[SearchResult]:
        """Combine semantic and keyword search results"""
        # Create lookup for results
        semantic_lookup = {r.id: r for r in semantic_results}
        keyword_lookup = {r.id: r for r in keyword_results}
        
        # Get all unique document IDs
        all_ids = set(semantic_lookup.keys()) | set(keyword_lookup.keys())
        
        combined_results = []
        
        for doc_id in all_ids:
            semantic_result = semantic_lookup.get(doc_id)
            keyword_result = keyword_lookup.get(doc_id)
            
            # Determine which result to use as base
            if semantic_result and keyword_result:
                # Document found in both searches - combine scores
                base_result = semantic_result
                base_result.keyword_score = keyword_result.keyword_score
                
                # Normalize and combine scores
                normalized_semantic = self._normalize_score(semantic_result.semantic_score or 0.0, 'semantic')
                normalized_keyword = self._normalize_score(keyword_result.keyword_score or 0.0, 'keyword')
                
                combined_score = (
                    self.semantic_weight * normalized_semantic +
                    self.keyword_weight * normalized_keyword
                )
                
                base_result.combined_score = combined_score
                base_result.source = "hybrid"
                
            elif semantic_result:
                # Only in semantic results
                base_result = semantic_result
                base_result.combined_score = self.semantic_weight * self._normalize_score(semantic_result.semantic_score or 0.0, 'semantic')
                
            else:
                # Only in keyword results
                base_result = keyword_result
                base_result.combined_score = self.keyword_weight * self._normalize_score(keyword_result.keyword_score or 0.0, 'keyword')
            
            combined_results.append(base_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        logger.debug(f"🔄 Combined {len(semantic_results)} semantic + {len(keyword_results)} keyword = {len(combined_results)} total results")
        
        return combined_results
    
    def _normalize_score(self, score: float, score_type: str) -> float:
        """Normalize scores to 0-1 range for combination"""
        if score_type == 'semantic':
            # Semantic scores are already between 0-1
            return max(0.0, min(1.0, score))
        else:
            # BM25 scores need normalization
            # Simple sigmoid normalization
            return 1.0 / (1.0 + math.exp(-score / 2.0))
    
    def _apply_construction_reranking(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Apply construction-specific reranking"""
        if not config.retrieval.enable_authority_reranking:
            return results
        
        # Extract query keywords for relevance boosting
        query_keywords = self.keyword_extractor.extract_keywords(query)
        
        reranked_results = []
        
        for result in results:
            # Apply authority-based reranking
            authority_boost = self._get_authority_boost(result.metadata)
            
            # Apply keyword relevance boost
            keyword_boost = self._get_keyword_relevance_boost(result.content, result.metadata, query_keywords)
            
            # Apply data type preference boost
            data_type_boost = self._get_data_type_boost(result.metadata, query)
            
            # Calculate final score
            boosted_score = result.combined_score * authority_boost * keyword_boost * data_type_boost
            
            # Create new result with boosted score
            boosted_result = SearchResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                semantic_score=result.semantic_score,
                keyword_score=result.keyword_score,
                combined_score=boosted_score,
                source=result.source
            )
            
            reranked_results.append(boosted_result)
        
        # Sort by boosted scores
        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return reranked_results
    
    def _get_authority_boost(self, metadata: Dict[str, Any]) -> float:
        """Get authority boost based on document type and data reliability"""
        authority_weights = config.retrieval.authority_weights
        
        # Check document type
        doc_type = metadata.get('document_type', 'reference')
        data_type = metadata.get('data_type', 'unknown')
        
        # Get base authority from document type
        base_authority = 1.0
        for authority_key, weight in authority_weights.items():
            if authority_key in doc_type.lower() or authority_key in data_type.lower():
                base_authority = weight
                break
        
        # Additional boost for high-cost or high-impact items
        total_cost = metadata.get('total_cost', 0) or metadata.get('total_estimated_cost', 0)
        if isinstance(total_cost, (int, float)) and total_cost > 10000:
            base_authority *= 1.1  # 10% boost for high-value items
        
        return base_authority
    
    def _get_keyword_relevance_boost(self, content: str, metadata: Dict[str, Any], query_keywords: Dict[str, List[str]]) -> float:
        """Get boost based on keyword relevance"""
        boost = 1.0
        content_lower = content.lower()
        
        # Boost for cost codes match
        if query_keywords['cost_codes']:
            cost_codes = metadata.get('cost_codes', '')
            for code in query_keywords['cost_codes']:
                if code in cost_codes:
                    boost *= 1.15
        
        # Boost for dollar amount relevance
        if query_keywords['dollar_amounts'] and 'cost' in content_lower:
            boost *= 1.1
        
        # Boost for measurement relevance
        if query_keywords['measurements'] and any(term in content_lower for term in ['quantity', 'unit', 'measure']):
            boost *= 1.05
        
        return min(boost, 1.5)  # Cap boost at 50%
    
    def _get_data_type_boost(self, metadata: Dict[str, Any], query: str) -> float:
        """Get boost based on data type preference inferred from query"""
        query_lower = query.lower()
        data_type = metadata.get('data_type', '')
        
        # Boost consumed data for "actual" or "spent" queries
        if data_type == 'consumed' and any(term in query_lower for term in ['actual', 'spent', 'consumed', 'real']):
            return 1.2
        
        # Boost estimate data for "budget" or "planned" queries
        if data_type == 'estimate' and any(term in query_lower for term in ['budget', 'estimated', 'planned', 'forecast']):
            return 1.2
        
        # Boost schedule data for timeline queries
        if data_type == 'schedule' and any(term in query_lower for term in ['schedule', 'timeline', 'duration', 'deadline']):
            return 1.2
        
        return 1.0
    
    async def _semantic_only_retrieval(self, query: str, n_results: int, filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Fallback to semantic-only retrieval"""
        return await self._semantic_search(query, n_results, filters)
    
    async def _keyword_only_retrieval(self, query: str, n_results: int, filters: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Keyword-only retrieval for specific use cases"""
        return await self._keyword_search(query, n_results, filters)

# Global hybrid retriever instance  
_hybrid_retriever = None

def get_hybrid_retriever(embedding_service: EmbeddingService) -> HybridRetriever:
    """Get or create global hybrid retriever instance"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(embedding_service)
    return _hybrid_retriever