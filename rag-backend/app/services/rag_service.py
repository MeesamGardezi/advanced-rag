"""
RAG (Retrieval Augmented Generation) service for query processing and answer generation.
"""

from typing import List, Dict
from openai import OpenAI
from app.config import settings
from app.models import QueryRequest, QueryResponse
from app.services.embedding import get_embedding_service
from app.services.qdrant_service import get_qdrant_service
import re


class RAGService:
    """Service for RAG query pipeline."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.embedding_service = get_embedding_service()
        self.qdrant_service = get_qdrant_service()
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            request: Query request with question and filters
            
        Returns:
            Query response with answer and sources
        """
        try:
            # Step 1: Embed the question
            print(f"🔍 Processing query: {request.question}")
            query_vector = self.embedding_service.embed_text(request.question)
            
            # Step 2: Search for similar documents
            print(f"🔎 Searching for relevant documents...")
            search_results = self.qdrant_service.search_similar(
                query_vector=query_vector,
                company_id=request.company_id,
                job_id=request.job_id,
                top_k=settings.top_k_results
            )
            
            if not search_results:
                return QueryResponse(
                    answer="I couldn't find any relevant information to answer your question. Please ensure the job data has been ingested.",
                    sources=[],
                    confidence=0.0,
                    chunks_used=0
                )
            
            # Step 3: Build context from search results
            context = self._build_context(search_results)
            
            # Step 4: Generate answer using OpenAI
            print(f"🤖 Generating answer with {settings.openai_model}...")
            answer = self._generate_answer(request.question, context)
            
            # Step 5: Extract sources (cost codes, areas)
            sources = self._extract_sources(search_results)
            
            # Step 6: Calculate confidence based on search scores
            confidence = self._calculate_confidence(search_results)
            
            print(f"✅ Query processed successfully")
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                chunks_used=len(search_results)
            )
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            raise
    
    def _build_context(self, search_results: List[Dict]) -> str:
        """
        Build context string from search results.
        
        Args:
            search_results: List of similar documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for idx, result in enumerate(search_results, 1):
            content = result['content']
            score = result['score']
            
            context_parts.append(f"--- Document {idx} (Relevance: {score:.2f}) ---")
            context_parts.append(content)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using OpenAI API.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Generated answer
        """
        # Create prompt
        prompt = f"""You are a construction project assistant analyzing job estimates and budget comparisons.

Context from job documents:
{context}

User Question: {question}

Instructions:
- Provide a detailed, accurate answer based ONLY on the context provided above
- If you mention specific cost codes, amounts, or percentages, reference them clearly
- If you're comparing values, show the calculations
- If the context doesn't contain enough information to answer fully, say so clearly
- Use proper formatting with bullet points or numbered lists when appropriate
- Be specific with dollar amounts and percentages
- If discussing budget variances, clearly state if items are over or under budget

Answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful construction project assistant that provides accurate, detailed answers about job estimates and budgets."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
            raise
    
    def _extract_sources(self, search_results: List[Dict]) -> List[str]:
        """
        Extract unique sources (cost codes, areas) from search results.
        
        Args:
            search_results: List of similar documents
            
        Returns:
            List of unique source identifiers
        """
        sources = set()
        
        for result in search_results:
            metadata = result['metadata']
            
            # Add cost code
            if 'cost_code' in metadata:
                sources.add(metadata['cost_code'])
            
            # Add area
            if 'area' in metadata:
                sources.add(metadata['area'])
        
        return sorted(list(sources))
    
    def _calculate_confidence(self, search_results: List[Dict]) -> float:
        """
        Calculate confidence score based on search result scores.
        
        Args:
            search_results: List of similar documents with scores
            
        Returns:
            Confidence score between 0 and 1
        """
        if not search_results:
            return 0.0
        
        # Use average of top 3 scores, weighted towards the top result
        top_scores = [result['score'] for result in search_results[:3]]
        
        if len(top_scores) == 1:
            return top_scores[0]
        elif len(top_scores) == 2:
            return (top_scores[0] * 0.6 + top_scores[1] * 0.4)
        else:
            return (top_scores[0] * 0.5 + top_scores[1] * 0.3 + top_scores[2] * 0.2)
    
    def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            # Simple API test
            self.openai_client.models.list()
            return True
        except:
            return False


# Global instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service