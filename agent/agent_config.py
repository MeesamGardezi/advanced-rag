"""
LangChain Agent Configuration
Intelligent query router that orchestrates SQL and Vector tools
FIXED: Ensure semantic_query parameter is always passed to vector_tool
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import logging
import json

from tools.sql_tool import sql_tool
from tools.vector_tool import vector_tool

logger = logging.getLogger(__name__)


# System prompt for the agent - FIXED to ensure semantic_query is always passed
AGENT_SYSTEM_PROMPT = """You are a Construction Data Assistant with specialized tools for querying construction project data.

You have access to a PostgreSQL database (via SQL Tool) and semantic search (via Vector Tool).

DATABASE SCHEMA:
- jobs: Project metadata (id, name, client_name, company_id, status)
- estimates: Estimate line items (job_id, row_number, cost_code, area, task_scope, description, total, budgeted_total, variance, qty, rate, units, row_type)
- flooring_estimates: Flooring estimates (job_id, row_number, vendor, item_material_name, total_cost, sale_price, profit)
- schedule_items: Schedule/timeline (job_id, task, hours, consumed, percentage_complete, start_date, end_date)
- consumed_items: Actual costs (job_id, cost_code, amount, category)

YOUR TOOLS:

1. **sql_tool**: Execute read-only SQL queries
   - Use for: Exact filters, getting ALL matching rows, calculations (SUM, AVG, COUNT)
   - Returns: JSON with "data" field (all rows), "formatted_display" field (readable text), and "row_count"
   - IMPORTANT: Use the "data" field to extract IDs, not the formatted display

2. **vector_tool**: Semantic filtering using embeddings
   - Use for: Filtering rows by semantic meaning (natural language concepts)
   - REQUIRED PARAMETERS:
     * candidate_ids: Comma-separated list of IDs from SQL
     * semantic_query: Natural language description of what you're looking for (e.g., "cleanup materials", "electrical work")
   - Optional: data_type (default "estimate"), job_id
   - Returns: Filtered IDs based on text similarity

QUERY PROCESSING WORKFLOW:

**STEP 1: Get Job ID**
```sql
SELECT id, name FROM jobs WHERE name ILIKE '%job_name%' LIMIT 1
```

**STEP 2: Retrieve ALL Candidates with SQL**
```sql
SELECT id, description, cost_code FROM estimates WHERE job_id = 'xxx'
```

**STEP 3: Extract ALL IDs from SQL Data**
Parse the JSON response and extract ALL IDs from the "data" field (not formatted_display).

**STEP 4: Semantic Filtering**
CRITICAL: When calling vector_tool, you MUST provide:
- candidate_ids: The comma-separated list of ALL IDs
- semantic_query: The semantic concept from the user's question (e.g., "cleanup materials", "electrical work", "roofing")

Example vector_tool call:
```
vector_tool(
    candidate_ids="id1,id2,id3,...,id258",
    semantic_query="cleanup materials",
    data_type="estimate",
    job_id="4ZppggAAJuJMZNB8f2ZT"
)
```

**STEP 5: Calculate Results**
Use SQL to aggregate filtered results:
```sql
SELECT SUM(total) FROM estimates WHERE id IN ('id1', 'id2', ...)
```

EXAMPLES:

**Example: "What's the total cleanup cost for Hammond?"**
1. Get job_id: SELECT id FROM jobs WHERE name LIKE '%Hammond%'
2. Get ALL estimates: SELECT id, description, cost_code FROM estimates WHERE job_id = 'xxx'
3. Extract ALL 258 IDs from the JSON "data" field
4. Call vector_tool with:
   - candidate_ids: "id1,id2,id3,...,id258"  
   - semantic_query: "cleanup materials"
   - data_type: "estimate"
   - job_id: "xxx"
5. Calculate: SELECT SUM(total) FROM estimates WHERE id IN (filtered_ids)

CRITICAL RULES:
1. **ALWAYS include semantic_query parameter when calling vector_tool**
2. **Extract the semantic concept from the user's question for semantic_query**
3. **Use ALL IDs from the data field, not just what's shown in formatted_display**
4. **Format money as $X,XXX.XX**
"""


def create_sql_tool_wrapper() -> StructuredTool:
    """Create LangChain tool wrapper for SQL tool - FIXED to return full data"""
    
    def execute_sql(query: str) -> str:
        """Execute a read-only SQL query and return FULL results as JSON"""
        result = sql_tool.execute(query)
        
        if result['success']:
            # Return structured JSON with BOTH data and display
            response = {
                "success": True,
                "row_count": result['row_count'],
                "data": result['data'],  # ALL rows as list of dicts
                "formatted_display": result['formatted_result']  # Human-readable (may be truncated)
            }
            
            # Add helpful note if data was truncated in display
            if result['row_count'] > 50 and 'more rows' in result['formatted_result']:
                response['note'] = f"Display shows first 50 rows, but data field contains all {result['row_count']} rows"
            
            # Return as JSON string so agent can parse it
            return json.dumps(response)
        else:
            return json.dumps({
                "success": False,
                "error": result['error']
            })
    
    return StructuredTool.from_function(
        func=execute_sql,
        name="sql_tool",
        description=sql_tool.description + """

IMPORTANT: This tool returns JSON with:
- "data": List of ALL result rows (may be 100s of rows)
- "formatted_display": Human-readable text (may show only first 50 rows)
- "row_count": Total number of rows

Always parse the JSON and use the "data" field to extract information, not the formatted_display!
"""
    )


def create_vector_tool_wrapper() -> StructuredTool:
    """Create LangChain tool wrapper for Vector tool - FIXED parameter handling"""
    
    def filter_semantically(
        candidate_ids: str,
        semantic_query: str,
        data_type: str = "estimate",
        job_id: Optional[str] = None
    ) -> str:
        """
        Filter candidate rows using semantic similarity
        
        Args:
            candidate_ids: Comma-separated list of IDs (can be 100s of IDs)
            semantic_query: Natural language description (REQUIRED - e.g., "cleanup materials", "electrical work")
            data_type: Type of data (estimate, flooring_estimate, schedule, consumed)
            job_id: Optional job ID for additional filtering
        """
        # Ensure we have semantic_query
        if not semantic_query:
            return "Error: semantic_query is required. Please provide what you're looking for (e.g., 'cleanup materials')"
        
        # Parse candidate_ids (may be a very long list)
        # Handle case where the list might be truncated
        if candidate_ids.endswith('_est_row_'):
            # ID list was truncated, try to complete it
            logger.warning("ID list appears truncated, attempting to parse partial list")
            candidate_ids = candidate_ids.rstrip(',').rstrip('_est_row_')
        
        ids_list = [id.strip() for id in candidate_ids.split(',') if id.strip() and len(id.strip()) > 10]
        
        logger.info(f"📊 Vector tool received {len(ids_list)} candidate IDs for semantic query: '{semantic_query}'")
        
        if len(ids_list) == 0:
            return f"Error: No valid candidate IDs provided for semantic filtering of '{semantic_query}'"
        
        # If we get a huge number of candidates, log it
        if len(ids_list) > 100:
            logger.info(f"⚡ Processing large batch: {len(ids_list)} candidates")
        
        # Process the candidates
        result = vector_tool.filter_candidates(
            candidate_ids=ids_list,
            semantic_query=semantic_query,
            data_type=data_type,
            job_id=job_id
        )
        
        if result['success']:
            matching_ids = result['matching_ids']
            
            if not matching_ids:
                return f"No semantic matches found for '{semantic_query}' among {result['total_candidates']} candidates"
            
            # Format response
            response_lines = [
                f"Semantic filtering for '{semantic_query}':",
                f"Processed: {result['total_candidates']} total candidates",
                f"Matches: {result['filtered_count']} items match semantically",
                f"Threshold: {result.get('threshold_used', 0.7)}",
                f"\nMatching IDs ({len(matching_ids)} total):"
            ]
            
            # Show first 20 matches with scores
            for i, match_id in enumerate(matching_ids[:20]):
                score = result['scores'].get(match_id, 0)
                response_lines.append(f"  - {match_id} (similarity: {score:.3f})")
            
            if len(matching_ids) > 20:
                response_lines.append(f"  ... and {len(matching_ids) - 20} more matches")
                response_lines.append(f"\nAll {len(matching_ids)} matching IDs (comma-separated):")
                response_lines.append(",".join(matching_ids))
            
            return "\n".join(response_lines)
        else:
            return f"Error: {result.get('error', 'Unknown error')}"
    
    return StructuredTool.from_function(
        func=filter_semantically,
        name="vector_tool",
        description="""Semantic filtering tool for finding conceptually similar items.

REQUIRED PARAMETERS:
- candidate_ids: Comma-separated list of IDs from SQL query (can handle 100s of IDs)
- semantic_query: Natural language concept to filter for (e.g., "cleanup materials", "electrical work", "roofing")

OPTIONAL PARAMETERS:
- data_type: Type of data (default: "estimate")
- job_id: Job ID for additional filtering

IMPORTANT: Always extract the semantic concept from the user's question and pass it as semantic_query.
For example, if user asks about "cleanup costs", use semantic_query="cleanup materials"
"""
    )


class ConstructionAgent:
    """
    Intelligent construction data query agent
    Routes queries to appropriate tools and orchestrates multi-step workflows
    """
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.0,
        max_iterations: int = 15,  # Increased for complex queries
        verbose: bool = True
    ):
        """
        Initialize the agent
        
        Args:
            model: OpenAI model to use
            temperature: Model temperature (0.0 for deterministic)
            max_iterations: Maximum agent iterations
            verbose: Enable verbose logging
        """
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create tools
        self.tools = [
            create_sql_tool_wrapper(),
            create_vector_tool_wrapper()
        ]
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor with better error handling
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=max_iterations,
            verbose=verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            max_execution_time=60  # Add timeout
        )
        
        logger.info(f"✅ Construction Agent initialized with {len(self.tools)} tools")
    
    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Execute a query through the agent
        
        Args:
            question: User's question
            chat_history: Optional conversation history
        
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 AGENT QUERY: {question}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Prepare chat history
            history_messages = []
            if chat_history:
                for turn in chat_history:
                    history_messages.append(HumanMessage(content=turn.get('question', '')))
                    history_messages.append(AIMessage(content=turn.get('answer', '')))
            
            # Execute agent
            result = self.agent_executor.invoke({
                "input": question,
                "chat_history": history_messages
            })
            
            answer = result.get('output', 'I was unable to generate an answer.')
            intermediate_steps = result.get('intermediate_steps', [])
            
            # Extract tool usage information
            tools_used = []
            for step in intermediate_steps:
                if len(step) >= 2:
                    action = step[0]
                    # Truncate long inputs (like lists of IDs)
                    tool_input = str(action.tool_input)
                    if 'candidate_ids' in tool_input and len(tool_input) > 500:
                        # Count IDs instead of showing them all
                        id_count = tool_input.count('4ZppggAAJuJMZNB8f2ZT')
                        tool_input = tool_input[:200] + f"...[{id_count} IDs total]..." + tool_input[-50:]
                    
                    tools_used.append({
                        'tool': action.tool,
                        'input': tool_input
                    })
            
            logger.info(f"\n✅ Agent completed query")
            logger.info(f"Tools used: {[t['tool'] for t in tools_used]}")
            logger.info(f"Answer length: {len(answer)} chars\n")
            
            return {
                'success': True,
                'question': question,
                'answer': answer,
                'tools_used': tools_used,
                'intermediate_steps': intermediate_steps
            }
        
        except Exception as e:
            error_msg = f"Agent execution error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'question': question,
                'answer': f"I encountered an error: {error_msg}",
                'tools_used': [],
                'error': error_msg
            }
    
    def clear_history(self):
        """Clear conversation history (if implementing stateful conversations)"""
        pass


# Global agent instance (lazy initialization)
_agent_instance: Optional[ConstructionAgent] = None


def get_agent(
    model: str = "gpt-4-turbo-preview",
    temperature: float = 0.0,
    force_new: bool = False
) -> ConstructionAgent:
    """
    Get or create global agent instance
    
    Args:
        model: OpenAI model to use
        temperature: Model temperature
        force_new: Force creation of new agent
    
    Returns:
        ConstructionAgent instance
    """
    global _agent_instance
    
    if _agent_instance is None or force_new:
        _agent_instance = ConstructionAgent(
            model=model,
            temperature=temperature,
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "15")),  # Increased default
            verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true"
        )
    
    return _agent_instance


if __name__ == "__main__":
    # Test the agent
    print("🧪 Testing Construction Agent with full data extraction...\n")
    
    # Initialize agent
    agent = get_agent()
    
    # Test query that requires processing many rows
    print("Test: Query requiring semantic filtering of many rows")
    result = agent.query("What is the total estimated cost of cleanup materials in Hammond?")
    print(f"Success: {result['success']}")
    print(f"Tools used: {[t['tool'] for t in result['tools_used']]}")
    
    # Check if all IDs were processed
    for tool_call in result['tools_used']:
        if tool_call['tool'] == 'vector_tool':
            print(f"Vector tool call details: {tool_call['input'][:100]}...")
    
    print(f"Answer: {result['answer']}\n")
    
    print("✅ Agent test complete")