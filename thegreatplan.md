# Agentic RAG Migration Plan: Hybrid SQL + Vector Architecture

## Executive Summary

This document outlines the migration strategy from a naive vector-only RAG system to a hybrid agentic architecture that combines SQL databases for precise calculations with vector search for semantic understanding. The current system fails to reliably retrieve and calculate estimate/budget data due to top-K retrieval limitations and incomplete semantic matching. The proposed solution uses an intelligent agent that routes queries to specialized tools, ensuring 100% accuracy for calculations while maintaining powerful semantic search capabilities.

## Problem Statement

### Current System Limitations

The existing vector-based RAG system exhibits two critical failures:

\begin{itemize}
\item \textbf{Incomplete Retrieval}: Vector search returns only top-K results (typically 5-10 documents), but accurate cost calculations require ALL relevant rows (often 200+ estimate line items)
\item \textbf{Inaccurate Matching}: Vector similarity search misses relevant estimate rows due to semantic variations and retrieves irrelevant rows, resulting in incomplete and incorrect cost summations
\end{itemize}

### Impact

\begin{itemize}
\item Consumed/actual data queries: Work correctly
\item Budget/estimate data queries: Incomplete and inaccurate results
\item User trust: Compromised due to unreliable financial calculations
\end{itemize}

## Proposed Architecture

### High-Level Design

User Query → Intelligent Agent (Router) → Decision Engine
                                          ↓
                        ┌─────────────────┴─────────────────┐
                        ↓                                   ↓
                   SQL Tool                           Vector Tool
            (Precise Calculations)              (Semantic Understanding)
                        ↓                                   ↓
                   SQL Database                        ChromaDB
              (All Firestore Data)              (Text Fields Embedded)

### Core Components

#### 1. SQL Database

Stores ALL Firestore data in structured format for complete retrieval and precise calculations.

**Schema Design**:

\begin{table}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Table} & \textbf{Purpose} & \textbf{Key Fields} \\
\hline
jobs & Project metadata & id, name, client, status \\
estimates & Line item data & job\_id, costCode, total, budgetedTotal \\
schedule & Timeline data & job\_id, dates, percent\_complete \\
\hline
\end{tabular}
\caption{SQL Database Schema}
\end{table}

#### 2. Vector Database

Contains embeddings of text fields for semantic search and filtering.

**Embedded Fields**: area, taskScope, description, notesRemarks

#### 3. Intelligent Agent

LLM-based router that:
\begin{itemize}
\item Analyzes user query intent
\item Determines which tool(s) to use
\item Orchestrates multi-step workflows
\item Synthesizes results into coherent responses
\end{itemize}

## Data Model

### EstimateRow Structure

Based on Firestore schema, each estimate row contains:

**Structured Fields** (exact matching):
\begin{itemize}
\item costCode: Standardized cost code
\item qty, rate, total: Numeric values for calculations
\item budgetedRate, budgetedTotal: Budget values
\item units, rowType: Categorical values
\end{itemize}

**Text Fields** (semantic search):
\begin{itemize}
\item area: Location description (e.g., "Kitchen", "Main Hall")
\item taskScope: Work scope description
\item description: Detailed task description
\item notesRemarks: Additional context and notes
\end{itemize}

**Additional Fields**:
\begin{itemize}
\item materials: Array of material references
\item rowType: Either "estimate" or "allowance"
\end{itemize}

## Query Processing Flow

### Three-Step Hybrid Process

#### Step 1: SQL Candidate Retrieval

Agent identifies exact filters from query and retrieves ALL matching rows from SQL database.

**Example**: Query "Total cleanup costs for Mall project"

SELECT * FROM estimates e
JOIN jobs j ON e.job_id = j.id
WHERE j.name = 'Mall';

**Result**: 500 candidate estimate rows

#### Step 2: Vector Semantic Filtering

Agent applies semantic understanding to filter candidates to truly relevant items.

**Process**:
\begin{enumerate}
\item Embed user's semantic intent: "cleanup costs"
\item Embed text fields from candidates: area + taskScope + description + notesRemarks
\item Calculate similarity scores
\item Filter candidates where similarity > 0.7
\end{enumerate}

**Result**: 12 semantically relevant estimate rows

#### Step 3: SQL Aggregation

Agent performs precise mathematical operations on filtered results.

SELECT SUM(total) FROM estimates
WHERE id IN ('est_123', 'est_456', ...);

**Result**: $12,450.00 (exact sum)

### Decision Matrix

\begin{table}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Query Type} & \textbf{SQL Filter} & \textbf{Vector Filter} \\
\hline
Exact costCode & costCode = 'X' & None \\
Semantic concept & job\_id = 'X' & "cleanup materials" \\
Date range & date range & Optional \\
Cost threshold & cost > \$X & None \\
Multi-criteria & Multiple & "electrical work" \\
\hline
\end{tabular}
\caption{Tool Selection Logic}
\end{table}

## Implementation Components

### 1. Database Schema

CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    client TEXT,
    status TEXT CHECK(status IN ('active', 'completed', 'on_hold'))
);

CREATE TABLE estimates (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    cost_code TEXT NOT NULL,
    area TEXT,
    task_scope TEXT,
    description TEXT,
    notes_remarks TEXT,
    units TEXT,
    qty REAL,
    rate REAL,
    total REAL NOT NULL,
    budgeted_rate REAL,
    budgeted_total REAL,
    row_type TEXT CHECK(row_type IN ('estimate', 'allowance')),
    materials JSON,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE INDEX idx_estimates_job_id ON estimates(job_id);
CREATE INDEX idx_estimates_cost_code ON estimates(cost_code);

### 2. ETL Pipeline

**Migration Script**: Transfers ALL Firestore data to both SQL and Vector databases.

**Process**:
\begin{enumerate}
\item Begin SQL transaction
\item Fetch all EstimateRow documents from Firestore
\item For each document:
  \begin{itemize}
  \item Insert into SQL database (all fields)
  \item Embed text fields and insert into ChromaDB
  \end{itemize}
\item Commit transaction on success, rollback on error
\item Verify counts: SQL rows + Vector docs = Firestore docs
\end{enumerate}

**Safety Features**:
\begin{itemize}
\item Transaction-based migration with rollback support
\item Conflict handling for re-runs
\item Post-migration verification
\item Error logging and tracking
\end{itemize}

### 3. SQL Tool Implementation

**SafeSQLTool**: Read-only SQL query execution with guardrails.

**Safety Features**:
\begin{itemize}
\item Block destructive operations: DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE
\item Auto-add LIMIT 100 if missing
\item Query timeout enforcement
\item Input validation
\end{itemize}

**Capabilities**:
\begin{itemize}
\item Retrieve ALL rows matching exact filters
\item Perform aggregations: SUM, AVG, COUNT, MIN, MAX
\item Filter by: job\_id, cost\_code, date ranges, cost thresholds
\item Join operations across tables
\end{itemize}

### 4. Vector Tool Implementation

**SemanticFilterTool**: Semantic search and filtering in text fields.

**Process**:
\begin{enumerate}
\item Receive candidate rows from SQL
\item Embed semantic query
\item Query ChromaDB for candidates
\item Calculate similarity scores
\item Return IDs where similarity > threshold (0.7)
\end{enumerate}

**Optimizations**:
\begin{itemize}
\item LRU cache for frequent queries
\item Batch embedding generation
\item Metadata filtering (job\_id, cost\_code)
\end{itemize}

### 5. Agent Configuration

**System Prompt**:

You are a Construction Data Assistant with two specialized tools:

SQL_Tool: Use for exact filters and calculations
- Filters: job_name, cost_code (exact), date ranges, cost thresholds
- Operations: SUM, AVG, COUNT, comparisons
- Returns: All matching rows (no top-K limit)

Vector_Tool: Use for semantic understanding
- Input: Candidate rows + semantic query
- Process: Finds semantically similar items
- Returns: Filtered IDs based on text similarity

Query Processing:
1. Parse query for exact filters (job name, cost codes, dates)
2. Call SQL_Tool to get ALL candidate rows
3. If query has semantic component, call Vector_Tool to filter
4. Aggregate filtered results with SQL (SUM, AVG, etc.)
5. Format response with currency, units, context

Critical Rules:
- Get job_id before querying estimates
- Use SQL for ALL mathematical operations
- Use vectors for semantic filtering only
- Format money as $X,XXX.XX
- Cite sources: [SQL Database] or [Semantic Search]
- Flag anomalies (large variances, missing data)

## Edge Cases and Solutions

### Edge Case 1: Exact CostCode Query

**Query**: "Total for costCode 01-5020 in Mall"

**Solution**: Agent skips vector tool, uses SQL directly with exact filter

**SQL**: `WHERE cost_code = '01-5020'`

### Edge Case 2: Too Many Semantic Matches

**Query**: "Total for materials in Mall"

**Problem**: "materials" appears in 80% of descriptions

**Solution**: Set similarity threshold to 0.7, only include highly relevant matches

### Edge Case 3: Synonym Matching

**Query**: "Total for HVAC work in Mall"

**Challenge**: Vector might miss "air conditioning" or "mechanical"

**Solution**: 
\begin{itemize}
\item Use embeddings trained on construction domain
\item Query expansion: search ["HVAC", "air conditioning", "mechanical"]
\item Accept as limitation for MVP, improve with feedback
\end{itemize}

### Edge Case 4: Specific Product Codes

**Query**: "Estimates with 2x4 lumber"

**Solution**: Agent uses SQL LIKE for specific alphanumeric patterns

**SQL**: `WHERE description LIKE '%2x4%'`

### Edge Case 5: Multi-Job Comparison

**Query**: "Compare cleanup costs between Mall and Hospital"

**Solution**: Agent executes two separate query flows and synthesizes results

**Process**:
\begin{enumerate}
\item Query 1: Mall cleanup costs
\item Query 2: Hospital cleanup costs
\item Compare and format results
\end{enumerate}

### Edge Case 6: Zero Results

**Query**: "Total for xyz123 in Mall"

**Solution**: Agent detects empty result set and responds appropriately

**Response**: "No matching items found for 'xyz123' in Mall project."

### Edge Case 7: Allowance vs Estimate

**Query**: "Total estimates for Mall"

**Ambiguity**: Does user want rowType='estimate' only or all rows?

**Solution**: Agent returns both with breakdown or asks for clarification

**Response**: "Mall project totals: Estimates: $450,000 | Allowances: $50,000 | Total: $500,000"

### Edge Case 8: Budget vs Actual Confusion

**Query**: "Total cost for cleanup in Mall"

**Ambiguity**: User wants `total` (actual) or `budgetedTotal`?

**Solution**: Agent returns both with clear labels

**Response**: "Cleanup costs for Mall: Actual: $12,450 | Budget: $15,000 | Variance: +$2,550 (under budget)"

## Success Metrics

### Quantitative Metrics

\begin{table}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Metric} & \textbf{Target} & \textbf{Measurement} \\
\hline
Calculation Accuracy & 100\% & Manual verification vs SQL \\
Response Latency & < 5 seconds & Average query time \\
Routing Accuracy & > 90\% & Correct tool selection \\
Retrieval Completeness & 100\% & All relevant rows retrieved \\
\hline
\end{tabular}
\caption{Performance Targets}
\end{table}

### Qualitative Metrics

\begin{itemize}
\item User satisfaction with budget/estimate queries
\item Reduction in "incorrect answer" feedback
\item Confidence in financial reporting
\end{itemize}

### Acceptance Criteria

\begin{enumerate}
\item \textbf{Accuracy Test}: Query "Total electrical estimate for Mall" returns exact SQL SUM
\item \textbf{Hybrid Test}: Query "Kitchen costs and tile-related items" successfully uses both tools
\item \textbf{Completeness Test}: System retrieves all 200+ estimate rows, not limited to top-K
\item \textbf{Semantic Test}: Query "cleanup materials" matches "debris removal" and "site cleaning"
\item \textbf{Performance Test}: Average response time under 5 seconds
\end{enumerate}

## Monitoring and Observability

### Logging System

Track every interaction:
\begin{itemize}
\item User query (raw input)
\item Tool calls made (SQL, Vector, or both)
\item Tool inputs and outputs
\item Final answer provided
\item Latency breakdown
\item Success/failure status
\end{itemize}

### Metrics Dashboard

Weekly tracking:
\begin{itemize}
\item Tool routing distribution (SQL only, Vector only, Hybrid)
\item Average latency by query type
\item Success rate (queries answered vs errors)
\item Most common query patterns
\item Edge case frequency
\end{itemize}

### Error Handling

\begin{itemize}
\item SQL errors: Return user-friendly message, log technical details
\item Vector errors: Fallback to SQL-only if possible
\item Agent routing errors: Ask user for clarification
\item Timeout handling: Return partial results with warning
\end{itemize}

## Implementation Checklist

### Phase 1: Foundation

- [ ] Design and create SQL database schema
- [ ] Set up ChromaDB instance
- [ ] Write ETL migration script with transaction support
- [ ] Migrate all Firestore data to SQL
- [ ] Embed text fields in ChromaDB
- [ ] Verify data counts match (SQL + Vector = Firestore)

### Phase 2: Tool Development

- [ ] Implement SafeSQLTool with read-only enforcement
- [ ] Implement Vector semantic filter tool
- [ ] Add caching layer for frequent queries
- [ ] Test SQL tool with sample queries
- [ ] Test Vector tool with semantic searches
- [ ] Verify tool safety (no destructive operations possible)

### Phase 3: Agent Configuration

- [ ] Configure LLM agent with system prompt
- [ ] Integrate SQL and Vector tools
- [ ] Implement query parsing logic
- [ ] Add multi-step orchestration
- [ ] Test routing accuracy with diverse queries

### Phase 4: Testing

- [ ] Test exact costCode queries (SQL only)
- [ ] Test semantic queries (SQL + Vector)
- [ ] Test multi-job comparisons
- [ ] Test edge cases (zero results, ambiguous queries)
- [ ] Verify calculation accuracy against manual verification
- [ ] Load testing for performance

### Phase 5: Monitoring

- [ ] Implement logging system
- [ ] Create metrics dashboard
- [ ] Set up error tracking
- [ ] Configure alerts for failures
- [ ] Document common query patterns

### Phase 6: Deployment

- [ ] Deploy to staging environment
- [ ] Run acceptance tests
- [ ] User acceptance testing
- [ ] Deploy to production
- [ ] Monitor initial performance
- [ ] Gather user feedback

## Risk Mitigation

### Data Migration Risks

**Risk**: Data loss during migration

**Mitigation**: Transaction-based migration with rollback, verification step, keep Firestore as source of truth

### Performance Risks

**Risk**: Slow queries on large datasets

**Mitigation**: Database indexes, query optimization, caching, pagination for large result sets

### Accuracy Risks

**Risk**: Agent routes to wrong tool

**Mitigation**: Comprehensive testing, monitoring, user feedback loop, prompt refinement

### Adoption Risks

**Risk**: Users don't trust new system

**Mitigation**: Parallel running, A/B testing, transparent sourcing, accuracy metrics dashboard

## Future Enhancements

### Phase 2 Features

\begin{itemize}
\item Advanced query expansion with construction domain synonyms
\item Custom re-ranking model trained on construction queries
\item Natural language to SQL optimization
\item Multi-turn conversation support
\item Visualization generation (charts, graphs)
\end{itemize}

### Scalability Improvements

\begin{itemize}
\item Distributed vector database for larger datasets
\item Query result caching layer
\item Async query processing
\item Real-time Firestore sync to SQL
\end{itemize}

## Conclusion

The hybrid SQL + Vector agentic architecture solves the fundamental limitations of the current vector-only system by combining the precision of SQL databases with the semantic understanding of vector search. This approach ensures 100\% accuracy for calculations while maintaining powerful semantic search capabilities, providing users with reliable, complete answers to both quantitative and qualitative queries about project data.

The phased implementation approach, comprehensive testing strategy, and built-in monitoring ensure a smooth transition with minimal risk, while the modular design allows for future enhancements based on user feedback and evolving requirements.