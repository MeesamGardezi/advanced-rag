import re
import logging
from typing import Dict, List, Literal, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI

from core.config import config

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Construction query types for routing"""
    SPECIFICATIONS = "specifications"
    BUILDING_CODES = "building_codes"
    COMPLIANCE_CHECK = "compliance_check"
    COST_ANALYSIS = "cost_analysis"
    SCHEDULE_PLANNING = "schedule_planning"
    GENERAL = "general"
    COMPARISON = "comparison"
    MULTI_DOMAIN = "multi_domain"

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"      # Direct factual lookup
    MODERATE = "moderate"   # Single-step analysis
    COMPLEX = "complex"     # Multi-step reasoning

class DataTypeFilter(Enum):
    """Data type preferences for queries"""
    CONSUMED = "consumed"
    ESTIMATE = "estimate"
    SCHEDULE = "schedule"
    ALL = "all"

@dataclass
class QueryClassification:
    """Result of query classification"""
    query_type: QueryType
    complexity: QueryComplexity
    data_types: List[DataTypeFilter]
    confidence: float
    reasoning: str
    suggested_agents: List[str]
    retrieval_strategy: str

class ConstructionQueryRouter:
    """Intelligent query routing for construction domain"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
        self.confidence_threshold = config.agents.classification_confidence_threshold
        
        # Construction-specific patterns for quick classification
        self.query_patterns = {
            QueryType.COST_ANALYSIS: [
                r'\b(cost|expense|budget|price|spent|money|dollar|\$)\b',
                r'\b(material cost|labor cost|total cost)\b',
                r'\b(over budget|under budget|budget variance)\b'
            ],
            QueryType.SCHEDULE_PLANNING: [
                r'\b(schedule|timeline|duration|deadline|date)\b',
                r'\b(start date|end date|completion)\b',
                r'\b(behind schedule|on schedule|delay)\b',
                r'\b(task|activity|milestone)\b'
            ],
            QueryType.SPECIFICATIONS: [
                r'\b(specification|spec|requirement|standard)\b',
                r'\b(material specification|technical spec)\b',
                r'\b(drawing|blueprint|plan)\b'
            ],
            QueryType.BUILDING_CODES: [
                r'\b(building code|code requirement|regulation)\b',
                r'\b(permit|approval|inspection)\b',
                r'\b(compliance|conform|meet code)\b'
            ],
            QueryType.COMPLIANCE_CHECK: [
                r'\b(compliance|compliant|violat|meet requirement)\b',
                r'\b(code compliance|safety requirement)\b',
                r'\b(pass inspection|fail inspection)\b'
            ]
        }
        
        # Data type indicators
        self.data_type_patterns = {
            DataTypeFilter.CONSUMED: [
                r'\b(consumed|actual|spent|used|real cost)\b',
                r'\b(what was spent|actual expense|consumed cost)\b'
            ],
            DataTypeFilter.ESTIMATE: [
                r'\b(estimate|estimated|budget|projected)\b',
                r'\b(planned cost|budgeted|forecasted)\b'
            ],
            DataTypeFilter.SCHEDULE: [
                r'\b(schedule|timeline|task|activity)\b',
                r'\b(planned duration|scheduled|timing)\b'
            ]
        }
        
        # Complexity indicators
        self.complexity_patterns = {
            QueryComplexity.SIMPLE: [
                r'^(what is|define|explain|show me)\b',
                r'\b(total|sum|list|find)\b'
            ],
            QueryComplexity.MODERATE: [
                r'\b(compare|analyze|calculate|breakdown)\b',
                r'\b(which|where|when|how much)\b'
            ],
            QueryComplexity.COMPLEX: [
                r'\b(evaluate|assess|recommend|strategy)\b',
                r'\b(if.*then|what if|scenario|impact)\b',
                r'\b(optimize|improve|best practice)\b'
            ]
        }
    
    async def classify_query(self, query: str) -> QueryClassification:
        """Classify construction query for optimal routing"""
        try:
            logger.debug(f"🔍 Classifying query: {query[:100]}...")
            
            # Quick pattern-based classification
            pattern_classification = self._pattern_based_classification(query)
            
            # If pattern classification is confident enough, use it
            if pattern_classification.confidence >= self.confidence_threshold:
                logger.debug(f"✅ Pattern-based classification (confidence: {pattern_classification.confidence:.2f})")
                return pattern_classification
            
            # Use LLM for more complex classification
            llm_classification = await self._llm_based_classification(query)
            
            # Combine pattern and LLM insights
            final_classification = self._combine_classifications(pattern_classification, llm_classification)
            
            logger.info(f"🎯 Query classified as {final_classification.query_type.value} "
                       f"({final_classification.complexity.value}) with {final_classification.confidence:.2f} confidence")
            
            return final_classification
            
        except Exception as e:
            logger.error(f"❌ Error classifying query: {e}")
            # Return safe default classification
            return QueryClassification(
                query_type=QueryType.GENERAL,
                complexity=QueryComplexity.MODERATE,
                data_types=[DataTypeFilter.ALL],
                confidence=0.5,
                reasoning=f"Error in classification: {e}",
                suggested_agents=['general'],
                retrieval_strategy='hybrid'
            )
    
    def _pattern_based_classification(self, query: str) -> QueryClassification:
        """Fast pattern-based query classification"""
        query_lower = query.lower()
        
        # Score each query type
        type_scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            type_scores[query_type] = score
        
        # Determine best match
        best_type = max(type_scores, key=type_scores.get) if any(type_scores.values()) else QueryType.GENERAL
        type_confidence = min(type_scores[best_type] / 3.0, 1.0)  # Scale confidence
        
        # Classify complexity
        complexity, complexity_confidence = self._classify_complexity(query_lower)
        
        # Classify data type preference
        data_types = self._classify_data_types(query_lower)
        
        # Determine retrieval strategy
        retrieval_strategy = self._determine_retrieval_strategy(best_type, complexity)
        
        # Suggest agents
        suggested_agents = self._suggest_agents(best_type, data_types)
        
        # Overall confidence
        overall_confidence = (type_confidence + complexity_confidence) / 2.0
        
        return QueryClassification(
            query_type=best_type,
            complexity=complexity,
            data_types=data_types,
            confidence=overall_confidence,
            reasoning=f"Pattern-based: matched {type_scores[best_type]} indicators for {best_type.value}",
            suggested_agents=suggested_agents,
            retrieval_strategy=retrieval_strategy
        )
    
    def _classify_complexity(self, query_lower: str) -> Tuple[QueryComplexity, float]:
        """Classify query complexity based on patterns"""
        complexity_scores = {}
        
        for complexity, patterns in self.complexity_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            complexity_scores[complexity] = score
        
        # Default to moderate if no clear indicators
        if not any(complexity_scores.values()):
            return QueryComplexity.MODERATE, 0.5
        
        best_complexity = max(complexity_scores, key=complexity_scores.get)
        confidence = min(complexity_scores[best_complexity] / 2.0, 1.0)
        
        return best_complexity, confidence
    
    def _classify_data_types(self, query_lower: str) -> List[DataTypeFilter]:
        """Determine which data types the query is interested in"""
        data_type_scores = {}
        
        for data_type, patterns in self.data_type_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            data_type_scores[data_type] = score
        
        # Return data types with non-zero scores
        relevant_types = [dt for dt, score in data_type_scores.items() if score > 0]
        
        # If no specific preference, return all
        return relevant_types if relevant_types else [DataTypeFilter.ALL]
    
    def _determine_retrieval_strategy(self, query_type: QueryType, complexity: QueryComplexity) -> str:
        """Determine optimal retrieval strategy"""
        if complexity == QueryComplexity.SIMPLE:
            return "semantic"
        elif complexity == QueryComplexity.COMPLEX:
            return "hybrid_with_reranking"
        else:
            return "hybrid"
    
    def _suggest_agents(self, query_type: QueryType, data_types: List[DataTypeFilter]) -> List[str]:
        """Suggest which agents should handle this query"""
        agents = []
        
        # Map query types to agents
        type_agent_mapping = {
            QueryType.SPECIFICATIONS: ['specification'],
            QueryType.BUILDING_CODES: ['compliance'],
            QueryType.COMPLIANCE_CHECK: ['compliance'],
            QueryType.COST_ANALYSIS: ['cost'],
            QueryType.SCHEDULE_PLANNING: ['schedule'],
            QueryType.COMPARISON: ['cost', 'schedule'],
            QueryType.MULTI_DOMAIN: ['cost', 'schedule', 'compliance']
        }
        
        # Add type-specific agents
        if query_type in type_agent_mapping:
            agents.extend(type_agent_mapping[query_type])
        
        # Add data-type specific agents
        for data_type in data_types:
            if data_type == DataTypeFilter.COST_ANALYSIS:
                agents.append('cost')
            elif data_type == DataTypeFilter.SCHEDULE:
                agents.append('schedule')
        
        # Default to general agent if no specific agents
        return agents if agents else ['general']
    
    async def _llm_based_classification(self, query: str) -> QueryClassification:
        """Use LLM for sophisticated query classification"""
        try:
            classification_prompt = f"""
            Analyze this construction-related query and classify it:
            
            Query: "{query}"
            
            Classify into these categories:
            
            1. Query Type:
            - specifications: Technical specs, drawings, requirements
            - building_codes: Building codes, regulations, permits
            - compliance_check: Compliance verification, inspections
            - cost_analysis: Cost analysis, budget, expenses
            - schedule_planning: Timelines, schedules, project planning
            - comparison: Comparing multiple items/options
            - multi_domain: Involves multiple construction domains
            - general: General construction questions
            
            2. Complexity:
            - simple: Direct factual lookup ("What is the cost of X?")
            - moderate: Single-step analysis ("Compare costs of X and Y")
            - complex: Multi-step reasoning ("Analyze impact of schedule changes on budget")
            
            3. Data Types (can be multiple):
            - consumed: Actual/spent costs and time
            - estimate: Planned/budgeted costs and time
            - schedule: Timeline and task information
            - all: No specific preference
            
            Respond in this exact format:
            Type: [query_type]
            Complexity: [complexity]
            DataTypes: [data_type1,data_type2]
            Confidence: [0.0-1.0]
            Reasoning: [brief explanation]
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.embedding.completion_model_simple,
                messages=[
                    {"role": "system", "content": "You are a construction domain expert that classifies queries for optimal processing."},
                    {"role": "user", "content": classification_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            return self._parse_llm_response(result)
            
        except Exception as e:
            logger.error(f"❌ LLM classification failed: {e}")
            # Return default classification
            return QueryClassification(
                query_type=QueryType.GENERAL,
                complexity=QueryComplexity.MODERATE,
                data_types=[DataTypeFilter.ALL],
                confidence=0.3,
                reasoning=f"LLM classification failed: {e}",
                suggested_agents=['general'],
                retrieval_strategy='hybrid'
            )
    
    def _parse_llm_response(self, response: str) -> QueryClassification:
        """Parse LLM classification response"""
        try:
            lines = response.strip().split('\n')
            parsed = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    parsed[key.strip().lower()] = value.strip()
            
            # Parse query type
            query_type = QueryType(parsed.get('type', 'general'))
            
            # Parse complexity
            complexity = QueryComplexity(parsed.get('complexity', 'moderate'))
            
            # Parse data types
            data_types_str = parsed.get('datatypes', 'all')
            data_types = []
            for dt in data_types_str.split(','):
                dt = dt.strip()
                try:
                    data_types.append(DataTypeFilter(dt))
                except ValueError:
                    continue
            
            if not data_types:
                data_types = [DataTypeFilter.ALL]
            
            # Parse confidence
            confidence = float(parsed.get('confidence', '0.7'))
            confidence = max(0.0, min(1.0, confidence))
            
            reasoning = parsed.get('reasoning', 'LLM-based classification')
            
            # Determine retrieval strategy and agents
            retrieval_strategy = self._determine_retrieval_strategy(query_type, complexity)
            suggested_agents = self._suggest_agents(query_type, data_types)
            
            return QueryClassification(
                query_type=query_type,
                complexity=complexity,
                data_types=data_types,
                confidence=confidence,
                reasoning=reasoning,
                suggested_agents=suggested_agents,
                retrieval_strategy=retrieval_strategy
            )
            
        except Exception as e:
            logger.error(f"❌ Error parsing LLM response: {e}")
            return QueryClassification(
                query_type=QueryType.GENERAL,
                complexity=QueryComplexity.MODERATE,
                data_types=[DataTypeFilter.ALL],
                confidence=0.3,
                reasoning=f"Parse error: {e}",
                suggested_agents=['general'],
                retrieval_strategy='hybrid'
            )
    
    def _combine_classifications(self, pattern_result: QueryClassification, llm_result: QueryClassification) -> QueryClassification:
        """Combine pattern-based and LLM-based classifications"""
        # Use the result with higher confidence as primary
        if pattern_result.confidence >= llm_result.confidence:
            primary = pattern_result
            secondary = llm_result
        else:
            primary = llm_result
            secondary = pattern_result
        
        # Combine data types from both
        combined_data_types = list(set(primary.data_types + secondary.data_types))
        
        # Average confidence weighted by individual confidences
        combined_confidence = (
            primary.confidence * 0.7 + secondary.confidence * 0.3
        )
        
        # Combine reasoning
        combined_reasoning = f"Primary: {primary.reasoning}; Secondary: {secondary.reasoning}"
        
        return QueryClassification(
            query_type=primary.query_type,
            complexity=primary.complexity,
            data_types=combined_data_types,
            confidence=combined_confidence,
            reasoning=combined_reasoning,
            suggested_agents=primary.suggested_agents,
            retrieval_strategy=primary.retrieval_strategy
        )
    
    def get_routing_strategy(self, classification: QueryClassification) -> Dict[str, Any]:
        """Get complete routing strategy based on classification"""
        return {
            'classification': classification,
            'retrieval_config': {
                'strategy': classification.retrieval_strategy,
                'n_results': self._get_n_results(classification.complexity),
                'filters': self._get_filters(classification.data_types),
                'rerank': classification.complexity == QueryComplexity.COMPLEX
            },
            'agent_config': {
                'agents': classification.suggested_agents,
                'parallel': len(classification.suggested_agents) > 1 and classification.complexity == QueryComplexity.COMPLEX,
                'timeout': 30 if classification.complexity == QueryComplexity.COMPLEX else 15
            },
            'processing_hints': {
                'use_semantic_cache': classification.complexity == QueryComplexity.SIMPLE,
                'enable_fact_check': classification.query_type in [QueryType.COMPLIANCE_CHECK, QueryType.BUILDING_CODES],
                'expand_query': classification.complexity == QueryComplexity.COMPLEX
            }
        }
    
    def _get_n_results(self, complexity: QueryComplexity) -> int:
        """Get number of results based on complexity"""
        if complexity == QueryComplexity.SIMPLE:
            return 3
        elif complexity == QueryComplexity.MODERATE:
            return 5
        else:
            return 8
    
    def _get_filters(self, data_types: List[DataTypeFilter]) -> Optional[Dict[str, Any]]:
        """Get search filters based on data types"""
        if DataTypeFilter.ALL in data_types:
            return None
        
        # Convert to filter format
        type_values = []
        for dt in data_types:
            if dt != DataTypeFilter.ALL:
                type_values.append(dt.value)
        
        return {'data_type': type_values} if type_values else None

# Global query router instance
_query_router = None

def get_query_router() -> ConstructionQueryRouter:
    """Get or create global query router instance"""
    global _query_router
    if _query_router is None:
        _query_router = ConstructionQueryRouter()
    return _query_router