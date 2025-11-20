"""
LangChain Agent Configuration
Intelligent query router that orchestrates SQL and Vector tools
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import logging

from tools.sql_tool import sql_tool
from tools.vector_tool import vector_tool

logger = logging.getLogger(__name__)


# System prompt for the agent
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
   - Returns: Complete result sets (no top-K limitation)
   - Examples:
     * Get job_id: SELECT id FROM jobs WHERE name LIKE '%Hammond%'
     * Get ALL estimates: SELECT * FROM estimates WHERE job_id = 'xxx'
     * Calculate total: SELECT SUM(total) FROM estimates WHERE job_id = 'xxx'
     * Filter by exact values: WHERE cost_code = 'xxx' or area = 'Kitchen'

2. **vector_tool**: Semantic filtering using embeddings
   - Use for: Filtering rows by semantic meaning (natural language concepts)
   - Input: List of candidate IDs from SQL + semantic query
   - Returns: Filtered IDs based on text similarity
   - Examples:
     * Filter for "cleanup materials" (matches "debris removal", "site cleaning")
     * Find "electrical work" (matches "wiring", "outlets", "lighting")

QUERY PROCESSING WORKFLOW:

**STEP 1: Understand the Query**
- Identify if query needs exact filters (job name, cost codes, dates) or semantic filters (concepts like "cleanup")
- Determine what data type is needed (estimates, flooring, schedule, consumed)
- Decide if calculation is needed (total, average, count)

**STEP 2: Get Job ID (if needed)**
If query mentions a job name but you don't have the job_id:
```sql
SELECT id, name FROM jobs WHERE name ILIKE '%job_name%' LIMIT 1
```

**STEP 3: Retrieve Candidates with SQL**
Get ALL matching rows using exact filters:
```sql
SELECT * FROM estimates 
WHERE job_id = 'xxx'
[AND cost_code = 'xxx']  -- if exact cost code mentioned
[AND area = 'xxx']        -- if exact area mentioned
```

**STEP 4: Semantic Filtering (if needed)**
If query has semantic component (like "cleanup materials"), filter candidates:
- Extract candidate IDs from SQL results
- Call vector_tool with semantic query and candidate IDs
- Get filtered IDs that match semantically

**STEP 5: Calculate Results**
Use SQL to aggregate filtered results:
```sql
SELECT SUM(total) FROM estimates WHERE id IN ('id1', 'id2', ...)
```

**STEP 6: Format Response**
- Always format money as $X,XXX.XX (with commas and 2 decimals)
- Include units (sq ft, hours, each, etc.)
- Cite data sources: [SQL Database] or [Semantic Search + SQL]
- If variance exists, mention it: "Estimated: $X | Budgeted: $Y | Variance: $Z"
- For lists, show row numbers and key details

CRITICAL RULES:

1. **Always get job_id first** before querying estimates/schedule/consumed
2. **Use SQL for ALL mathematical operations** (SUM, AVG, COUNT, etc.)
3. **Never fabricate data** - only use what's in the database
4. **Handle "estimate" vs "consumed" carefully**:
   - "estimate"/"budget"/"projected" → estimates table (total, budgeted_total)
   - "consumed"/"actual"/"spent" → consumed_items table (amount)
5. **Row type distinction**:
   - "allowance" → row_type = 'allowance'
   - "estimate" → row_type = 'estimate'
6. **Format all monetary values**: $1,234.56 (never $1234.5 or 1234.56)
7. **Be specific about data sources**: Always cite [SQL Database] or [Semantic Search]

EXAMPLES:

**Example 1: Exact Cost Code Query**
Query: "What's the total for cost code 01-5020 in Hammond project?"
Workflow:
1. Get job_id: SELECT id FROM jobs WHERE name LIKE '%Hammond%'
2. Get estimates: SELECT * FROM estimates WHERE job_id = 'xxx' AND cost_code = '01-5020'
3. Calculate: SELECT SUM(total) FROM estimates WHERE job_id = 'xxx' AND cost_code = '01-5020'
4. Response: "Total for cost code 01-5020 in Hammond project: $12,450.00 [SQL Database]"

**Example 2: Semantic Query**
Query: "What's the total cleanup cost for Mall project?"
Workflow:
1. Get job_id: SELECT id FROM jobs WHERE name LIKE '%Mall%'
2. Get ALL candidates: SELECT id, description, cost_code FROM estimates WHERE job_id = 'xxx'
3. Semantic filter: vector_tool.filter_candidates(candidate_ids, "cleanup materials")
4. Calculate: SELECT SUM(total) FROM estimates WHERE id IN (filtered_ids)
5. Response: "Total cleanup costs for Mall project: $8,750.00 (3 line items) [Semantic Search + SQL]"

**Example 3: Allowances Only**
Query: "Show me all allowances for Hammond"
Workflow:
1. Get job_id: SELECT id FROM jobs WHERE name LIKE '%Hammond%'
2. Query: SELECT * FROM estimates WHERE job_id = 'xxx' AND row_type = 'allowance'
3. Response: List all allowance rows with details

**Example 4: Budget vs Actual**
Query: "Compare estimated vs consumed costs for electrical work in Mall"
Workflow:
1. Get job_id for Mall
2. Get estimate total (using semantic filter for "electrical")
3. Get consumed total (using semantic filter for "electrical")
4. Response: "Electrical costs for Mall: Estimated: $45,000 | Consumed: $42,500 | Under budget by $2,500"

EDGE CASES:

- **Zero results**: "No matching items found for [query] in [project]"
- **Ambiguous query**: Ask for clarification: "Did you mean estimated costs or actual consumed costs?"
- **Multiple jobs match**: List options: "Multiple projects match 'Hammond'. Did you mean: 1) Hammond 2508, 2) Hammond Renovation?"
- **No semantic matches**: Fall back to SQL-only: "No items semantically matched '[query]', showing all results from SQL query"

Remember: You're a helpful assistant focused on accuracy. Always use the tools to get real data, never guess or fabricate numbers."""


def create_sql_tool_wrapper() -> StructuredTool:
    """Create LangChain tool wrapper for SQL tool"""
    
    def execute_sql(query: str) -> str:
        """Execute a read-only SQL query on the construction database"""
        result = sql_tool.execute(query)
        
        if result['success']:
            return result['formatted_result']
        else:
            return f"Error: {result['error']}"
    
    return StructuredTool.from_function(
        func=execute_sql,
        name="sql_tool",
        description=sql_tool.description
    )


def create_vector_tool_wrapper() -> StructuredTool:
    """Create LangChain tool wrapper for Vector tool"""
    
    def filter_semantically(
        candidate_ids: str,
        semantic_query: str,
        data_type: str = "estimate",
        job_id: Optional[str] = None
    ) -> str:
        """
        Filter candidate rows using semantic similarity
        
        Args:
            candidate_ids: Comma-separated list of IDs (e.g., "id1,id2,id3")
            semantic_query: Natural language description
            data_type: Type of data (estimate, flooring_estimate, schedule, consumed)
            job_id: Optional job ID for additional filtering
        """
        # Parse candidate_ids
        ids_list = [id.strip() for id in candidate_ids.split(',') if id.strip()]
        
        if not ids_list:
            return "Error: No candidate IDs provided"
        
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
                f"Matches: {result['filtered_count']} / {result['total_candidates']} candidates",
                f"Threshold: {result.get('threshold_used', 0.7)}",
                f"\nMatching IDs:"
            ]
            
            for match_id in matching_ids:
                score = result['scores'].get(match_id, 0)
                response_lines.append(f"  - {match_id} (similarity: {score:.3f})")
            
            return "\n".join(response_lines)
        else:
            return f"Error: {result.get('error', 'Unknown error')}"
    
    return StructuredTool.from_function(
        func=filter_semantically,
        name="vector_tool",
        description=vector_tool.description
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
        max_iterations: int = 10,
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
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=max_iterations,
            verbose=verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True
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
                    tools_used.append({
                        'tool': action.tool,
                        'input': str(action.tool_input)[:200]  # Truncate for logging
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
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "10")),
            verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true"
        )
    
    return _agent_instance


if __name__ == "__main__":
    # Test the agent
    print("🧪 Testing Construction Agent...\n")
    
    # Initialize agent
    agent = get_agent()
    
    # Test query 1: Simple job lookup
    print("Test 1: Job lookup")
    result = agent.query("List all jobs in the database")
    print(f"Success: {result['success']}")
    print(f"Tools used: {[t['tool'] for t in result['tools_used']]}")
    print(f"Answer: {result['answer'][:200]}...\n")
    
    # Test query 2: Calculation query
    print("Test 2: Calculation query")
    result = agent.query("What's the total estimated cost for all projects?")
    print(f"Success: {result['success']}")
    print(f"Tools used: {[t['tool'] for t in result['tools_used']]}")
    print(f"Answer: {result['answer'][:200]}...\n")
    
    print("✅ Agent tests complete")