"""
SafeSQLTool: Read-only SQL query execution with safety guardrails
Retrieves ALL matching rows from PostgreSQL for precise calculations
UPDATED: Smarter LIMIT handling for complete data retrieval
"""

from typing import List, Dict, Any, Optional
import re
import logging

from db_connection import session_scope

logger = logging.getLogger(__name__)


class SafeSQLTool:
    """
    Safe SQL query execution tool with read-only enforcement
    Used by the agent to retrieve ALL candidate rows for calculations
    """
    
    # Destructive SQL keywords that are BLOCKED
    BLOCKED_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 
        'TRUNCATE', 'GRANT', 'REVOKE', 'CREATE', 'REPLACE'
    ]
    
    # Maximum rows to return for safety (increased from 1000 to 10000)
    MAX_ROWS = 10000
    
    # Default timeout in seconds
    DEFAULT_TIMEOUT = 30
    
    def __init__(self):
        self.name = "sql_tool"
        self.description = """Execute read-only SQL queries on the construction database.
        
Available tables:
- jobs: Project metadata (id, name, client_name, company_id, status)
- estimates: Estimate line items (job_id, cost_code, area, task_scope, total, budgeted_total, row_number)
- flooring_estimates: Flooring estimate items (job_id, vendor, item_material_name, total_cost, sale_price)
- schedule_items: Schedule/timeline (job_id, task, hours, consumed, percentage_complete, start_date, end_date)
- consumed_items: Actual costs (job_id, cost_code, amount, category)

Use this tool to:
1. Get ALL rows matching exact filters (job_id, cost_code, date ranges, cost thresholds)
2. Perform aggregations (SUM, AVG, COUNT, MIN, MAX)
3. Filter by exact matches on structured fields

IMPORTANT: This tool retrieves ALL matching rows (up to 10,000), NOT just top-K results.
DO NOT use for semantic/text matching - use vector_tool for that.

Examples:
- Get all estimates for a job: SELECT * FROM estimates WHERE job_id = 'job123'
- Sum estimate totals: SELECT SUM(total) FROM estimates WHERE job_id = 'job123'
- Get job ID by name: SELECT id FROM jobs WHERE name LIKE '%Hammond%'
- Filter by cost code: SELECT * FROM estimates WHERE cost_code = '01-5020'
"""
    
    def _validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL query for safety
        
        Returns:
            (is_valid, error_message)
        """
        query_upper = query.upper().strip()
        
        # Check for blocked keywords
        for keyword in self.BLOCKED_KEYWORDS:
            if re.search(r'\b' + keyword + r'\b', query_upper):
                return False, f"❌ Blocked keyword detected: {keyword}. Only SELECT queries allowed."
        
        # Must start with SELECT or WITH (for CTEs)
        if not (query_upper.startswith('SELECT') or query_upper.startswith('WITH')):
            return False, "❌ Only SELECT queries are allowed"
        
        # Check for multiple statements (SQL injection prevention)
        if ';' in query.rstrip(';'):
            return False, "❌ Multiple statements not allowed"
        
        return True, None
    
    def _should_add_limit(self, query: str) -> bool:
        """
        Determine if LIMIT should be added to query
        
        Returns False if:
        - Query already has LIMIT
        - Query is an aggregation (COUNT, SUM, AVG, MAX, MIN)
        - Query is getting specific IDs (WHERE id IN ...)
        - Query filters by job_id (likely wants all rows for that job)
        """
        query_upper = query.upper()
        
        # Already has LIMIT
        if 'LIMIT' in query_upper:
            return False
        
        # Aggregation queries don't need LIMIT
        if any(agg in query_upper for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
            return False
        
        # Filtering by specific IDs (wants specific rows)
        if 'WHERE ID IN' in query_upper or 'WHERE ID =' in query_upper:
            return False
        
        # Filtering by job_id (wants all rows for that job)
        if 'WHERE JOB_ID' in query_upper:
            return False
        
        # If selecting just a few columns (like id, name), probably wants all
        if re.search(r'SELECT\s+\w+\s*,?\s*\w*\s+FROM', query_upper):
            # Check if it's selecting all columns or just id columns
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query_upper)
            if select_match:
                columns = select_match.group(1)
                # If selecting just ID or very few columns, allow all rows
                if ',' not in columns or columns.count(',') <= 2:
                    return False
        
        return True
    
    def _add_limit_if_needed(self, query: str) -> str:
        """Add LIMIT clause if appropriate"""
        
        if not self._should_add_limit(query):
            return query
        
        query_upper = query.upper()
        
        # If query has ORDER BY, add LIMIT before it
        if 'ORDER BY' in query_upper:
            order_pos = query_upper.rfind('ORDER BY')
            return query[:order_pos] + f" LIMIT {self.MAX_ROWS} " + query[order_pos:]
        
        # Otherwise, add LIMIT at the end
        return query.rstrip(';') + f" LIMIT {self.MAX_ROWS}"
    
    def _format_results(self, rows: List[Dict[str, Any]]) -> str:
        """Format query results as readable text for the agent"""
        if not rows:
            return "No results found."
        
        # If single value (like COUNT or SUM), return it directly
        if len(rows) == 1 and len(rows[0]) == 1:
            key = list(rows[0].keys())[0]
            value = rows[0][key]
            return f"{key}: {value}"
        
        # Format as table
        result_lines = [f"Found {len(rows)} rows:\n"]
        
        # Header
        if rows:
            headers = list(rows[0].keys())
            result_lines.append(" | ".join(headers))
            result_lines.append("-" * 80)
        
        # Rows (limit to first 50 for readability in response)
        for row in rows[:50]:
            row_values = [str(v) if v is not None else 'NULL' for v in row.values()]
            result_lines.append(" | ".join(row_values))
        
        if len(rows) > 50:
            result_lines.append(f"\n... and {len(rows) - 50} more rows")
        
        return "\n".join(result_lines)
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute a read-only SQL query
        
        Args:
            query: SQL SELECT query
        
        Returns:
            Dictionary with:
            - success: bool
            - data: List of result rows (as dicts)
            - formatted_result: Human-readable string
            - row_count: Number of rows returned
            - error: Error message if failed
        """
        logger.info(f"🔍 SQL Tool executing query:\n{query}")
        
        try:
            # Step 1: Validate query
            is_valid, error_msg = self._validate_query(query)
            if not is_valid:
                logger.error(error_msg)
                return {
                    'success': False,
                    'data': [],
                    'formatted_result': error_msg,
                    'row_count': 0,
                    'error': error_msg
                }
            
            # Step 2: Add LIMIT if needed (smart logic)
            safe_query = self._add_limit_if_needed(query)
            
            if safe_query != query:
                logger.info(f"ℹ️  Added LIMIT {self.MAX_ROWS} to query for safety")
            else:
                logger.info(f"ℹ️  No LIMIT added - query will return all matching rows")
            
            # Step 3: Execute query using psycopg2 cursor
            with session_scope() as cursor:
                cursor.execute(safe_query)
                
                # Check if query returns rows
                if cursor.description:
                    # Fetch all results
                    rows = cursor.fetchall()
                    
                    # Convert to list of dicts (RealDictCursor returns RealDictRow objects)
                    data = [dict(row) for row in rows]
                    
                    formatted = self._format_results(data)
                    
                    logger.info(f"✅ Query successful: {len(data)} rows returned")
                    
                    return {
                        'success': True,
                        'data': data,
                        'formatted_result': formatted,
                        'row_count': len(data),
                        'error': None
                    }
                else:
                    # Query doesn't return rows (shouldn't happen for SELECT)
                    return {
                        'success': True,
                        'data': [],
                        'formatted_result': "Query executed successfully (no rows returned)",
                        'row_count': 0,
                        'error': None
                    }
        
        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'data': [],
                'formatted_result': error_msg,
                'row_count': 0,
                'error': error_msg
            }
    
    def get_job_id_by_name(self, job_name: str) -> Optional[str]:
        """
        Helper: Get job ID by name (fuzzy match)
        
        Args:
            job_name: Job name or partial name
        
        Returns:
            job_id if found, None otherwise
        """
        query = f"""
            SELECT id, name, client_name
            FROM jobs
            WHERE name ILIKE '%{job_name}%'
            LIMIT 1
        """
        
        result = self.execute(query)
        
        if result['success'] and result['data']:
            job = result['data'][0]
            logger.info(f"✅ Found job: {job['name']} (ID: {job['id']})")
            return job['id']
        
        logger.warning(f"⚠️  Job not found: {job_name}")
        return None
    
    def get_all_estimates_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Helper: Get ALL estimate rows for a job
        
        Args:
            job_id: Job identifier
        
        Returns:
            List of estimate rows
        """
        query = f"""
            SELECT 
                id, job_id, row_number, area, task_scope, cost_code,
                description, total, budgeted_total, variance,
                qty, rate, units, row_type, has_materials
            FROM estimates
            WHERE job_id = '{job_id}'
            ORDER BY row_number
        """
        
        result = self.execute(query)
        
        if result['success']:
            logger.info(f"✅ Retrieved {result['row_count']} estimate rows for job {job_id}")
            return result['data']
        
        logger.error(f"❌ Failed to retrieve estimates for job {job_id}")
        return []
    
    def calculate_total_for_job(self, job_id: str, cost_type: str = 'total') -> float:
        """
        Helper: Calculate total cost for a job
        
        Args:
            job_id: Job identifier
            cost_type: 'total' (estimated) or 'budgeted_total'
        
        Returns:
            Total amount
        """
        if cost_type not in ['total', 'budgeted_total']:
            cost_type = 'total'
        
        query = f"""
            SELECT SUM({cost_type}) as total_amount
            FROM estimates
            WHERE job_id = '{job_id}'
        """
        
        result = self.execute(query)
        
        if result['success'] and result['data']:
            amount = result['data'][0].get('total_amount', 0.0)
            return float(amount) if amount is not None else 0.0
        
        return 0.0
    
    def get_estimates_by_filter(
        self,
        job_id: str,
        cost_code: Optional[str] = None,
        area: Optional[str] = None,
        row_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Helper: Get estimates with exact filters
        
        Args:
            job_id: Job identifier
            cost_code: Cost code filter (optional)
            area: Area filter (optional)
            row_type: Row type filter: 'estimate' or 'allowance' (optional)
        
        Returns:
            List of matching estimate rows
        """
        conditions = [f"job_id = '{job_id}'"]
        
        if cost_code:
            conditions.append(f"cost_code = '{cost_code}'")
        
        if area:
            conditions.append(f"area = '{area}'")
        
        if row_type:
            conditions.append(f"row_type = '{row_type}'")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                id, row_number, area, task_scope, cost_code,
                description, total, budgeted_total, variance,
                qty, rate, units, row_type
            FROM estimates
            WHERE {where_clause}
            ORDER BY row_number
        """
        
        result = self.execute(query)
        
        if result['success']:
            return result['data']
        
        return []


# Singleton instance for easy import
sql_tool = SafeSQLTool()


if __name__ == "__main__":
    # Test the SQL tool
    print("🧪 Testing SafeSQLTool...\n")
    
    tool = SafeSQLTool()
    
    # Test 1: Valid SELECT query
    print("Test 1: Valid SELECT query")
    result = tool.execute("SELECT * FROM jobs LIMIT 5")
    print(f"Success: {result['success']}")
    print(f"Rows: {result['row_count']}")
    print(f"Result:\n{result['formatted_result']}\n")
    
    # Test 2: Blocked DELETE query
    print("Test 2: Blocked DELETE query")
    result = tool.execute("DELETE FROM jobs WHERE id = 'test'")
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}\n")
    
    # Test 3: Query without LIMIT (should NOT add for job_id filter)
    print("Test 3: Query with job_id filter (no LIMIT added)")
    result = tool.execute("SELECT * FROM estimates WHERE job_id = 'test123'")
    print(f"Success: {result['success']}")
    print(f"Rows: {result['row_count']}\n")
    
    # Test 4: Aggregation query
    print("Test 4: Aggregation query")
    result = tool.execute("SELECT COUNT(*) as total_jobs FROM jobs")
    print(f"Success: {result['success']}")
    print(f"Result:\n{result['formatted_result']}\n")
    
    print("✅ SQL Tool tests complete")