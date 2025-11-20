"""
Query Processor: High-level orchestration for query processing
Wraps the agent with additional business logic, validation, and formatting
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import re

from agent.agent_config import get_agent, ConstructionAgent
from db_connection import verify_job_exists, get_job_data_summary

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    High-level query processor that orchestrates the agent
    Handles validation, pre-processing, post-processing, and formatting
    """
    
    def __init__(self, agent: Optional[ConstructionAgent] = None):
        """
        Initialize query processor
        
        Args:
            agent: Optional pre-configured agent (will create if None)
        """
        self.agent = agent or get_agent()
        
        # Query statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0.0,
            'tools_usage': {
                'sql_only': 0,
                'vector_only': 0,
                'hybrid': 0
            }
        }
    
    def _preprocess_query(self, question: str) -> str:
        """
        Preprocess and normalize the query
        
        Args:
            question: Raw user question
        
        Returns:
            Preprocessed question
        """
        # Trim whitespace
        question = question.strip()
        
        # Ensure question ends with proper punctuation
        if not question[-1] in ['.', '?', '!']:
            question += '?'
        
        # Normalize common abbreviations
        replacements = {
            r'\bestimate\b': 'estimate',
            r'\bbudget\b': 'budgeted',
            r'\bactual\b': 'consumed',
            r'\bspent\b': 'consumed',
        }
        
        for pattern, replacement in replacements.items():
            question = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        return question
    
    def _extract_data_type_hint(self, question: str) -> Optional[str]:
        """
        Extract data type hint from question
        
        Args:
            question: User question
        
        Returns:
            Data type hint or None
        """
        question_lower = question.lower()
        
        # Check for explicit data type mentions
        if any(word in question_lower for word in ['estimate', 'estimated', 'budget', 'budgeted']):
            return 'estimate'
        
        if any(word in question_lower for word in ['consumed', 'actual', 'spent', 'paid']):
            return 'consumed'
        
        if any(word in question_lower for word in ['schedule', 'timeline', 'task', 'deadline']):
            return 'schedule'
        
        if any(word in question_lower for word in ['flooring', 'floor', 'carpet', 'tile']):
            return 'flooring_estimate'
        
        return None
    
    def _validate_query(self, question: str) -> tuple[bool, Optional[str]]:
        """
        Validate query before processing
        
        Args:
            question: User question
        
        Returns:
            (is_valid, error_message)
        """
        # Check minimum length
        if len(question.strip()) < 3:
            return False, "Question is too short. Please provide more details."
        
        # Check maximum length
        if len(question) > 2000:
            return False, "Question is too long. Please keep it under 2000 characters."
        
        # Check for SQL injection attempts (basic check)
        suspicious_patterns = [
            r';\s*DROP',
            r';\s*DELETE',
            r';\s*UPDATE',
            r'--\s*',
            r'/\*.*\*/',
            r'UNION\s+SELECT',
            r'EXEC\s*\(',
            r'EXECUTE\s*\('
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return False, "Query contains suspicious patterns. Please rephrase your question."
        
        return True, None
    
    def _postprocess_answer(self, answer: str, tools_used: List[Dict]) -> str:
        """
        Postprocess and enhance the answer
        
        Args:
            answer: Raw agent answer
            tools_used: List of tools used by agent
        
        Returns:
            Enhanced answer
        """
        # Ensure monetary values are formatted correctly
        # Find all dollar amounts and ensure they have commas and 2 decimals
        def format_money(match):
            amount = match.group(1)
            # Remove existing commas
            amount = amount.replace(',', '')
            # Convert to float
            try:
                value = float(amount)
                return f"${value:,.2f}"
            except ValueError:
                return match.group(0)
        
        # Pattern: $XXX or $XXX.XX (with or without commas)
        answer = re.sub(r'\$([0-9,]+\.?[0-9]*)', format_money, answer)
        
        # Add data source citation if not present
        if '[SQL Database]' not in answer and '[Semantic Search' not in answer:
            # Determine source from tools used
            tool_names = [t['tool'] for t in tools_used]
            
            if 'sql_tool' in tool_names and 'vector_tool' in tool_names:
                answer += "\n\n_Source: Hybrid search (SQL + Semantic filtering)_"
            elif 'sql_tool' in tool_names:
                answer += "\n\n_Source: SQL Database_"
            elif 'vector_tool' in tool_names:
                answer += "\n\n_Source: Semantic Search_"
        
        return answer
    
    def _analyze_tools_usage(self, tools_used: List[Dict]):
        """
        Analyze which tools were used and update statistics
        
        Args:
            tools_used: List of tools used
        """
        tool_names = [t['tool'] for t in tools_used]
        
        has_sql = 'sql_tool' in tool_names
        has_vector = 'vector_tool' in tool_names
        
        if has_sql and has_vector:
            self.stats['tools_usage']['hybrid'] += 1
        elif has_sql:
            self.stats['tools_usage']['sql_only'] += 1
        elif has_vector:
            self.stats['tools_usage']['vector_only'] += 1
    
    def _format_error_response(self, question: str, error: str) -> Dict[str, Any]:
        """
        Format error response in consistent structure
        
        Args:
            question: Original question
            error: Error message
        
        Returns:
            Formatted error response
        """
        return {
            'success': False,
            'question': question,
            'answer': f"I apologize, but I encountered an error processing your query: {error}",
            'data_type_hint': None,
            'tools_used': [],
            'response_time_ms': 0,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def process_query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the agent system
        
        Args:
            question: User's question
            chat_history: Optional conversation history
            context: Optional additional context (job_id, user preferences, etc.)
        
        Returns:
            Comprehensive response dictionary
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📝 QUERY PROCESSOR: Processing new query")
        logger.info(f"   Question: {question}")
        logger.info(f"   Has history: {bool(chat_history)}")
        logger.info(f"   Context: {context}")
        logger.info(f"{'='*80}\n")
        
        self.stats['total_queries'] += 1
        
        try:
            # Step 1: Validate query
            is_valid, error_msg = self._validate_query(question)
            if not is_valid:
                logger.warning(f"⚠️  Query validation failed: {error_msg}")
                return self._format_error_response(question, error_msg)
            
            # Step 2: Preprocess query
            processed_question = self._preprocess_query(question)
            if processed_question != question:
                logger.info(f"   Preprocessed: {processed_question}")
            
            # Step 3: Extract data type hint
            data_type_hint = self._extract_data_type_hint(processed_question)
            if data_type_hint:
                logger.info(f"   Data type hint: {data_type_hint}")
            
            # Step 4: Enhance question with context (if provided)
            enhanced_question = processed_question
            if context:
                if 'job_id' in context:
                    # Verify job exists
                    if verify_job_exists(context['job_id']):
                        enhanced_question += f" [Context: job_id='{context['job_id']}']"
                    else:
                        logger.warning(f"⚠️  Job ID in context not found: {context['job_id']}")
            
            # Step 5: Execute agent query
            logger.info("🤖 Executing agent query...")
            agent_result = self.agent.query(
                question=enhanced_question,
                chat_history=chat_history
            )
            
            if not agent_result['success']:
                self.stats['failed_queries'] += 1
                return self._format_error_response(
                    question,
                    agent_result.get('error', 'Unknown error')
                )
            
            # Step 6: Postprocess answer
            answer = self._postprocess_answer(
                agent_result['answer'],
                agent_result['tools_used']
            )
            
            # Step 7: Analyze tool usage
            self._analyze_tools_usage(agent_result['tools_used'])
            
            # Step 8: Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self.stats['successful_queries'] += 1
            total = self.stats['total_queries']
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (total - 1) + response_time_ms) / total
            )
            
            logger.info(f"\n✅ Query processed successfully")
            logger.info(f"   Response time: {response_time_ms:.0f}ms")
            logger.info(f"   Tools used: {[t['tool'] for t in agent_result['tools_used']]}\n")
            
            # Step 9: Format response
            return {
                'success': True,
                'question': question,
                'answer': answer,
                'data_type_hint': data_type_hint,
                'tools_used': agent_result['tools_used'],
                'response_time_ms': response_time_ms,
                'timestamp': datetime.utcnow().isoformat(),
                'intermediate_steps': agent_result.get('intermediate_steps', [])
            }
        
        except Exception as e:
            error_msg = f"Query processing error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            import traceback
            logger.error(traceback.format_exc())
            
            self.stats['failed_queries'] += 1
            return self._format_error_response(question, error_msg)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get query processing statistics
        
        Returns:
            Statistics dictionary
        """
        success_rate = (
            (self.stats['successful_queries'] / self.stats['total_queries'] * 100)
            if self.stats['total_queries'] > 0 else 0
        )
        
        return {
            'total_queries': self.stats['total_queries'],
            'successful_queries': self.stats['successful_queries'],
            'failed_queries': self.stats['failed_queries'],
            'success_rate_pct': round(success_rate, 2),
            'average_response_time_ms': round(self.stats['average_response_time'], 2),
            'tools_usage': self.stats['tools_usage']
        }
    
    def reset_statistics(self):
        """Reset query statistics"""
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0.0,
            'tools_usage': {
                'sql_only': 0,
                'vector_only': 0,
                'hybrid': 0
            }
        }
        logger.info("📊 Statistics reset")
    
    def validate_job_context(self, job_id: str) -> Dict[str, Any]:
        """
        Validate and get job context
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job context or error
        """
        if not verify_job_exists(job_id):
            return {
                'valid': False,
                'error': f"Job {job_id} not found in database"
            }
        
        summary = get_job_data_summary(job_id)
        
        return {
            'valid': True,
            'job_id': job_id,
            'job_name': summary['job_name'],
            'data_available': {
                'estimates': summary['estimates_count'] > 0,
                'flooring': summary['flooring_estimates_count'] > 0,
                'schedule': summary['schedule_items_count'] > 0,
                'consumed': summary['consumed_items_count'] > 0
            }
        }


# Global processor instance
_processor_instance: Optional[QueryProcessor] = None


def get_query_processor(force_new: bool = False) -> QueryProcessor:
    """
    Get or create global query processor instance
    
    Args:
        force_new: Force creation of new processor
    
    Returns:
        QueryProcessor instance
    """
    global _processor_instance
    
    if _processor_instance is None or force_new:
        _processor_instance = QueryProcessor()
    
    return _processor_instance


# Convenience function for one-off queries
async def process_query(
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a query using the global processor
    
    Args:
        question: User's question
        chat_history: Optional conversation history
        context: Optional additional context
    
    Returns:
        Query response
    """
    processor = get_query_processor()
    return await processor.process_query(question, chat_history, context)


if __name__ == "__main__":
    import asyncio
    
    async def test_processor():
        print("🧪 Testing Query Processor...\n")
        
        processor = QueryProcessor()
        
        # Test 1: Simple query
        print("Test 1: Simple query")
        result = await processor.process_query("How many jobs are in the database?")
        print(f"Success: {result['success']}")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Response time: {result['response_time_ms']:.0f}ms\n")
        
        # Test 2: Query with validation error
        print("Test 2: Query too short")
        result = await processor.process_query("Hi")
        print(f"Success: {result['success']}")
        print(f"Error: {result.get('error', 'None')}\n")
        
        # Test 3: Complex calculation query
        print("Test 3: Calculation query")
        result = await processor.process_query(
            "What's the total estimated cost for all projects?"
        )
        print(f"Success: {result['success']}")
        print(f"Tools used: {[t['tool'] for t in result['tools_used']]}")
        print(f"Answer: {result['answer'][:200]}...\n")
        
        # Test 4: Get statistics
        print("Test 4: Statistics")
        stats = processor.get_statistics()
        print(f"Total queries: {stats['total_queries']}")
        print(f"Success rate: {stats['success_rate_pct']}%")
        print(f"Avg response time: {stats['average_response_time_ms']:.0f}ms")
        print(f"Tools usage: {stats['tools_usage']}\n")
        
        print("✅ Query Processor tests complete")
    
    asyncio.run(test_processor())