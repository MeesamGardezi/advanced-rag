import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from openai import OpenAI

from core.config import config
from retrieval.hybrid_retriever import HybridRetriever, SearchResult

logger = logging.getLogger(__name__)

class DocumentRelevance(Enum):
    """Document relevance levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"

class CorrectionAction(Enum):
    """Actions to take when correction is needed"""
    ACCEPT = "accept"           # Results are good as-is
    EXPAND_QUERY = "expand"     # Expand the search query
    REWRITE_QUERY = "rewrite"   # Completely rewrite the query
    FILTER_RESULTS = "filter"   # Filter out irrelevant results
    SEARCH_BROADER = "broader"  # Search with broader parameters

@dataclass
class DocumentGrade:
    """Grade for a retrieved document"""
    document_id: str
    relevance: DocumentRelevance
    confidence: float
    reasoning: str
    authority_score: float
    technical_accuracy: float

@dataclass
class CorrectionDecision:
    """Decision on how to correct retrieval"""
    action: CorrectionAction
    confidence: float
    reasoning: str
    new_query: Optional[str] = None
    filter_suggestions: Optional[Dict[str, Any]] = None
    search_params: Optional[Dict[str, Any]] = None

class ConstructionDocumentGrader:
    """Grade documents for relevance and quality in construction context"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
        self.relevance_threshold = config.retrieval.document_relevance_threshold
    
    async def grade_documents(self, query: str, documents: List[SearchResult]) -> List[DocumentGrade]:
        """Grade all documents for relevance and quality"""
        if not documents:
            return []
        
        # Process documents in batches for efficiency
        batch_size = 5
        all_grades = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_grades = await self._grade_document_batch(query, batch)
            all_grades.extend(batch_grades)
        
        logger.debug(f"🏆 Graded {len(all_grades)} documents")
        return all_grades
    
    async def _grade_document_batch(self, query: str, documents: List[SearchResult]) -> List[DocumentGrade]:
        """Grade a batch of documents efficiently"""
        try:
            # Create grading prompt for batch processing
            grading_prompt = self._create_batch_grading_prompt(query, documents)
            
            response = self.openai_client.chat.completions.create(
                model=config.embedding.completion_model_simple,
                messages=[
                    {"role": "system", "content": "You are a construction domain expert who evaluates document relevance and quality."},
                    {"role": "user", "content": grading_prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            return self._parse_batch_grades(result, documents)
            
        except Exception as e:
            logger.error(f"❌ Error grading document batch: {e}")
            # Return default grades for all documents
            return [self._create_default_grade(doc.id) for doc in documents]
    
    def _create_batch_grading_prompt(self, query: str, documents: List[SearchResult]) -> str:
        """Create prompt for batch document grading"""
        prompt_parts = [
            f"Query: \"{query}\"",
            "",
            "Evaluate these construction documents for relevance and quality:",
            ""
        ]
        
        for i, doc in enumerate(documents, 1):
            # Create document summary
            content_preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            
            metadata_info = []
            if 'data_type' in doc.metadata:
                metadata_info.append(f"Type: {doc.metadata['data_type']}")
            if 'job_name' in doc.metadata:
                metadata_info.append(f"Job: {doc.metadata['job_name']}")
            if 'total_cost' in doc.metadata:
                metadata_info.append(f"Cost: ${doc.metadata['total_cost']:,.2f}")
            
            metadata_str = " | ".join(metadata_info)
            
            prompt_parts.extend([
                f"Document {i} (ID: {doc.id}):",
                f"Metadata: {metadata_str}",
                f"Content: {content_preview}",
                ""
            ])
        
        prompt_parts.extend([
            "For each document, provide:",
            "1. Relevance: high/medium/low/irrelevant",
            "2. Confidence: 0.0-1.0",
            "3. Authority: 0.0-1.0 (based on data reliability and source type)",
            "4. Technical: 0.0-1.0 (technical accuracy for the query)",
            "5. Reason: Brief explanation",
            "",
            "Consider:",
            "- Relevance to the specific construction query",
            "- Authority (consumed > estimate > schedule for cost queries)",
            "- Technical accuracy and completeness",
            "- Recency and data quality",
            "",
            "Format each response as:",
            "Doc X: relevance|confidence|authority|technical|reason",
            "",
            "Responses:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_batch_grades(self, response: str, documents: List[SearchResult]) -> List[DocumentGrade]:
        """Parse batch grading response"""
        grades = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('Doc'):
                continue
            
            try:
                # Parse format: "Doc X: relevance|confidence|authority|technical|reason"
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                
                doc_num_str = parts[0].replace('Doc', '').strip()
                doc_num = int(doc_num_str) - 1  # Convert to 0-based index
                
                if doc_num >= len(documents):
                    continue
                
                grade_parts = parts[1].split('|')
                if len(grade_parts) >= 5:
                    relevance_str = grade_parts[0].strip().lower()
                    confidence = float(grade_parts[1].strip())
                    authority = float(grade_parts[2].strip())
                    technical = float(grade_parts[3].strip())
                    reason = '|'.join(grade_parts[4:]).strip()
                    
                    # Parse relevance
                    relevance = DocumentRelevance.MEDIUM  # default
                    if relevance_str in ['high', 'medium', 'low', 'irrelevant']:
                        relevance = DocumentRelevance(relevance_str)
                    
                    grade = DocumentGrade(
                        document_id=documents[doc_num].id,
                        relevance=relevance,
                        confidence=max(0.0, min(1.0, confidence)),
                        reasoning=reason,
                        authority_score=max(0.0, min(1.0, authority)),
                        technical_accuracy=max(0.0, min(1.0, technical))
                    )
                    
                    grades.append(grade)
                
            except Exception as e:
                logger.warning(f"⚠️  Error parsing grade line '{line}': {e}")
                continue
        
        # Fill in any missing grades with defaults
        while len(grades) < len(documents):
            missing_doc = documents[len(grades)]
            grades.append(self._create_default_grade(missing_doc.id))
        
        return grades
    
    def _create_default_grade(self, document_id: str) -> DocumentGrade:
        """Create default grade when grading fails"""
        return DocumentGrade(
            document_id=document_id,
            relevance=DocumentRelevance.MEDIUM,
            confidence=0.5,
            reasoning="Default grade - grading failed",
            authority_score=0.5,
            technical_accuracy=0.5
        )

class QueryExpander:
    """Expand and improve queries for better retrieval"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
        
        # Construction domain synonyms and related terms
        self.construction_synonyms = {
            'cost': ['expense', 'budget', 'price', 'amount', 'fee'],
            'material': ['supplies', 'product', 'component', 'item'],
            'labor': ['work', 'workforce', 'manpower', 'hours'],
            'schedule': ['timeline', 'plan', 'duration', 'deadline'],
            'electrical': ['electric', 'wiring', 'power', 'lighting'],
            'plumbing': ['pipes', 'water', 'drainage', 'fixtures'],
            'concrete': ['cement', 'foundation', 'slab', 'pour'],
            'framing': ['lumber', 'wood', 'structure', 'frame']
        }
    
    async def expand_query(self, original_query: str, context: Dict[str, Any]) -> List[str]:
        """Generate expanded query variations"""
        try:
            # Create expansion prompt
            expansion_prompt = self._create_expansion_prompt(original_query, context)
            
            response = self.openai_client.chat.completions.create(
                model=config.embedding.completion_model_simple,
                messages=[
                    {"role": "system", "content": "You are a construction domain expert who helps improve search queries."},
                    {"role": "user", "content": expansion_prompt}
                ],
                max_tokens=400,
                temperature=0.3  # Slightly higher temperature for creative expansions
            )
            
            result = response.choices[0].message.content.strip()
            expanded_queries = self._parse_expanded_queries(result)
            
            # Add rule-based expansions
            rule_based_expansions = self._generate_rule_based_expansions(original_query)
            expanded_queries.extend(rule_based_expansions)
            
            # Remove duplicates and limit to top expansions
            unique_queries = list(dict.fromkeys(expanded_queries))  # Preserves order
            return unique_queries[:5]  # Return top 5 expansions
            
        except Exception as e:
            logger.error(f"❌ Query expansion failed: {e}")
            return [original_query]  # Return original query as fallback
    
    def _create_expansion_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Create prompt for query expansion"""
        prompt_parts = [
            f"Original query: \"{query}\"",
            "",
            "Context information:",
        ]
        
        if context.get('failed_results_summary'):
            prompt_parts.append(f"- Previous search found: {context['failed_results_summary']}")
        
        if context.get('data_types_needed'):
            prompt_parts.append(f"- Looking for data types: {', '.join(context['data_types_needed'])}")
        
        prompt_parts.extend([
            "",
            "Generate 3-5 improved search queries for construction data that:",
            "1. Use construction-specific terminology",
            "2. Are more specific and targeted",
            "3. Include relevant cost codes, materials, or trades",
            "4. Consider different ways to express the same concept",
            "5. Focus on actionable construction information",
            "",
            "Format as numbered list:",
            "1. [improved query]",
            "2. [improved query]",
            "etc.",
            "",
            "Improved queries:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_expanded_queries(self, response: str) -> List[str]:
        """Parse expanded queries from LLM response"""
        queries = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Match numbered list format
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                query = line.split('.', 1)[-1].strip()
                query = query.replace('-', '').strip()
                if query and len(query) > 10:  # Reasonable minimum length
                    queries.append(query)
        
        return queries
    
    def _generate_rule_based_expansions(self, query: str) -> List[str]:
        """Generate expansions using rule-based approach"""
        expansions = []
        query_lower = query.lower()
        
        # Add synonyms
        for term, synonyms in self.construction_synonyms.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Use top 2 synonyms
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expansions.append(expanded)
        
        # Add data type specific expansions
        if 'cost' in query_lower:
            expansions.extend([
                query + " consumed actual",
                query + " estimated budgeted",
                query.replace('cost', 'expense breakdown')
            ])
        
        if 'schedule' in query_lower:
            expansions.extend([
                query + " timeline tasks",
                query.replace('schedule', 'project timeline')
            ])
        
        return expansions[:3]  # Limit rule-based expansions

class CorrectiveRAG:
    """Self-correcting RAG system for construction queries"""
    
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever
        self.document_grader = ConstructionDocumentGrader()
        self.query_expander = QueryExpander()
        self.max_correction_attempts = config.retrieval.max_correction_attempts
        self.relevance_threshold = config.retrieval.document_relevance_threshold
    
    async def corrective_retrieve(self, 
                                 query: str,
                                 n_results: int = 5,
                                 filters: Optional[Dict[str, Any]] = None,
                                 strategy: str = "hybrid") -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Perform corrective retrieval with self-improvement"""
        
        if not config.retrieval.enable_corrective_rag:
            # Fallback to regular retrieval
            results = await self.retriever.retrieve(query, n_results, filters, strategy)
            return results, {"corrections_applied": 0, "final_query": query}
        
        correction_history = []
        current_query = query
        current_filters = filters
        current_strategy = strategy
        
        for attempt in range(self.max_correction_attempts + 1):
            logger.debug(f"🔄 Corrective retrieval attempt {attempt + 1}")
            
            # Perform retrieval
            results = await self.retriever.retrieve(
                current_query, 
                n_results * 2,  # Get more results for evaluation
                current_filters, 
                current_strategy
            )
            
            if not results:
                logger.warning("No results found, attempting query expansion")
                if attempt < self.max_correction_attempts:
                    expanded_queries = await self.query_expander.expand_query(
                        current_query, 
                        {"failed_results_summary": "no results found"}
                    )
                    if expanded_queries:
                        current_query = expanded_queries[0]
                        continue
                break
            
            # Grade the retrieved documents
            document_grades = await self.document_grader.grade_documents(current_query, results)
            
            # Analyze results and decide on corrections
            correction_decision = await self._analyze_and_decide(current_query, results, document_grades)
            
            correction_history.append({
                "attempt": attempt + 1,
                "query": current_query,
                "results_count": len(results),
                "decision": correction_decision.action.value,
                "reasoning": correction_decision.reasoning
            })
            
            # Apply correction or accept results
            if correction_decision.action == CorrectionAction.ACCEPT:
                # Filter results based on grades and return top N
                accepted_results = self._filter_results_by_grade(results, document_grades, n_results)
                
                return accepted_results, {
                    "corrections_applied": attempt,
                    "final_query": current_query,
                    "correction_history": correction_history,
                    "final_decision": correction_decision.action.value
                }
            
            elif attempt < self.max_correction_attempts:
                # Apply correction for next attempt
                current_query, current_filters, current_strategy = self._apply_correction(
                    correction_decision, current_query, current_filters, current_strategy
                )
            else:
                logger.warning("Max correction attempts reached, returning best available results")
                # Filter and return best results
                final_results = self._filter_results_by_grade(results, document_grades, n_results)
                
                return final_results, {
                    "corrections_applied": attempt + 1,
                    "final_query": current_query,
                    "correction_history": correction_history,
                    "final_decision": "max_attempts_reached"
                }
        
        # Fallback return
        return results[:n_results], {
            "corrections_applied": 0,
            "final_query": current_query,
            "correction_history": correction_history,
            "final_decision": "fallback"
        }
    
    async def _analyze_and_decide(self, 
                                 query: str, 
                                 results: List[SearchResult], 
                                 grades: List[DocumentGrade]) -> CorrectionDecision:
        """Analyze results and decide on correction action"""
        
        if not grades:
            return CorrectionDecision(
                action=CorrectionAction.EXPAND_QUERY,
                confidence=0.8,
                reasoning="No document grades available"
            )
        
        # Calculate quality metrics
        high_relevance_count = sum(1 for g in grades if g.relevance == DocumentRelevance.HIGH)
        medium_relevance_count = sum(1 for g in grades if g.relevance == DocumentRelevance.MEDIUM)
        low_relevance_count = sum(1 for g in grades if g.relevance == DocumentRelevance.LOW)
        irrelevant_count = sum(1 for g in grades if g.relevance == DocumentRelevance.IRRELEVANT)
        
        total_grades = len(grades)
        high_relevance_ratio = high_relevance_count / total_grades
        relevant_ratio = (high_relevance_count + medium_relevance_count) / total_grades
        
        avg_confidence = sum(g.confidence for g in grades) / total_grades
        avg_authority = sum(g.authority_score for g in grades) / total_grades
        avg_technical = sum(g.technical_accuracy for g in grades) / total_grades
        
        logger.debug(f"📊 Quality metrics: {high_relevance_ratio:.2f} high, {relevant_ratio:.2f} relevant, "
                    f"{avg_confidence:.2f} confidence, {avg_authority:.2f} authority")
        
        # Decision logic
        if high_relevance_ratio >= 0.4 and relevant_ratio >= 0.6:
            # Good results - accept them
            return CorrectionDecision(
                action=CorrectionAction.ACCEPT,
                confidence=0.9,
                reasoning=f"Good results: {high_relevance_ratio:.1%} high relevance, {relevant_ratio:.1%} relevant"
            )
        
        elif relevant_ratio >= 0.3 and avg_authority >= 0.6:
            # Decent results but could filter out poor ones
            return CorrectionDecision(
                action=CorrectionAction.FILTER_RESULTS,
                confidence=0.7,
                reasoning=f"Filter low quality: {relevant_ratio:.1%} relevant, authority {avg_authority:.2f}"
            )
        
        elif irrelevant_count > total_grades * 0.7:
            # Most results are irrelevant - rewrite query
            return CorrectionDecision(
                action=CorrectionAction.REWRITE_QUERY,
                confidence=0.8,
                reasoning=f"Poor relevance: {irrelevant_count}/{total_grades} irrelevant"
            )
        
        elif avg_technical < 0.4:
            # Low technical accuracy - expand query for better context
            return CorrectionDecision(
                action=CorrectionAction.EXPAND_QUERY,
                confidence=0.7,
                reasoning=f"Low technical accuracy: {avg_technical:.2f}"
            )
        
        else:
            # Default to broader search
            return CorrectionDecision(
                action=CorrectionAction.SEARCH_BROADER,
                confidence=0.6,
                reasoning="Results insufficient, searching broader"
            )
    
    def _apply_correction(self, 
                         decision: CorrectionDecision,
                         query: str,
                         filters: Optional[Dict[str, Any]],
                         strategy: str) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """Apply correction decision to query parameters"""
        
        if decision.action == CorrectionAction.REWRITE_QUERY:
            # Use the new query if provided, otherwise use original
            new_query = decision.new_query if decision.new_query else query
            return new_query, filters, strategy
        
        elif decision.action == CorrectionAction.EXPAND_QUERY:
            # This will be handled by actually calling the query expander in the main loop
            return query, filters, strategy
        
        elif decision.action == CorrectionAction.SEARCH_BROADER:
            # Remove restrictive filters and increase search scope
            broader_filters = None
            if filters:
                # Remove data_type filters for broader search
                broader_filters = {k: v for k, v in filters.items() if k != 'data_type'}
                if not broader_filters:
                    broader_filters = None
            
            return query, broader_filters, "hybrid"
        
        elif decision.action == CorrectionAction.FILTER_RESULTS:
            # No changes to query parameters - filtering happens in result processing
            return query, filters, strategy
        
        else:
            # Default - no changes
            return query, filters, strategy
    
    def _filter_results_by_grade(self, 
                                results: List[SearchResult], 
                                grades: List[DocumentGrade], 
                                n_results: int) -> List[SearchResult]:
        """Filter and rank results based on document grades"""
        
        # Create lookup for grades
        grade_lookup = {g.document_id: g for g in grades}
        
        # Score and filter results
        scored_results = []
        for result in results:
            grade = grade_lookup.get(result.id)
            if not grade:
                continue
            
            # Skip irrelevant documents
            if grade.relevance == DocumentRelevance.IRRELEVANT:
                continue
            
            # Calculate composite score
            relevance_score = {
                DocumentRelevance.HIGH: 1.0,
                DocumentRelevance.MEDIUM: 0.7,
                DocumentRelevance.LOW: 0.4,
                DocumentRelevance.IRRELEVANT: 0.0
            }.get(grade.relevance, 0.5)
            
            composite_score = (
                result.combined_score * 0.4 +
                relevance_score * 0.3 +
                grade.authority_score * 0.2 +
                grade.technical_accuracy * 0.1
            )
            
            # Update result with composite score
            result.combined_score = composite_score
            scored_results.append(result)
        
        # Sort by composite score and return top N
        scored_results.sort(key=lambda x: x.combined_score, reverse=True)
        return scored_results[:n_results]

# Global corrective RAG instance
_corrective_rag = None

def get_corrective_rag(hybrid_retriever: HybridRetriever) -> CorrectiveRAG:
    """Get or create global corrective RAG instance"""
    global _corrective_rag
    if _corrective_rag is None:
        _corrective_rag = CorrectiveRAG(hybrid_retriever)
    return _corrective_rag