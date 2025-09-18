import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from openai import OpenAI

from core.config import config
from agents.query_router import QueryClassification, QueryType
from retrieval.hybrid_retriever import HybridRetriever, SearchResult

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Specialized agent roles"""
    SPECIFICATION = "specification"
    COMPLIANCE = "compliance" 
    COST = "cost"
    SCHEDULE = "schedule"
    GENERAL = "general"

class AgentStatus(Enum):
    """Agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class AgentResponse:
    """Response from a specialized agent"""
    agent_role: AgentRole
    response: str
    confidence: float
    sources: List[SearchResult]
    execution_time: float
    status: AgentStatus
    reasoning: str
    metadata: Dict[str, Any]

@dataclass
class AgentTask:
    """Task assigned to an agent"""
    agent_role: AgentRole
    query: str
    context: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0

class SpecificationAgent:
    """Agent specialized in construction specifications and technical requirements"""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
        self.role = AgentRole.SPECIFICATION
    
    async def process(self, task: AgentTask) -> AgentResponse:
        """Process specification-related queries"""
        start_time = datetime.now()
        
        try:
            # Enhanced retrieval for specifications
            search_filters = {
                'data_type': ['estimate', 'consumed'],
                **task.context.get('filters', {})
            }
            
            results = await self.retriever.retrieve(
                query=task.query,
                n_results=8,
                filters=search_filters,
                strategy="hybrid_with_reranking"
            )
            
            if not results:
                return AgentResponse(
                    agent_role=self.role,
                    response="No specification data found for this query.",
                    confidence=0.1,
                    sources=[],
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    status=AgentStatus.COMPLETED,
                    reasoning="No relevant specification documents retrieved",
                    metadata={"search_filters": search_filters}
                )
            
            # Generate specification-focused response
            response = await self._generate_specification_response(task.query, results)
            
            return AgentResponse(
                agent_role=self.role,
                response=response,
                confidence=0.85,
                sources=results[:5],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=AgentStatus.COMPLETED,
                reasoning="Analyzed specification and technical requirement data",
                metadata={"documents_analyzed": len(results)}
            )
            
        except Exception as e:
            logger.error(f"❌ Specification agent failed: {e}")
            return AgentResponse(
                agent_role=self.role,
                response=f"Error processing specification query: {str(e)}",
                confidence=0.0,
                sources=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=AgentStatus.FAILED,
                reasoning=f"Agent execution failed: {e}",
                metadata={"error": str(e)}
            )
    
    async def _generate_specification_response(self, query: str, results: List[SearchResult]) -> str:
        """Generate specification-focused response"""
        # Create context from results
        context_parts = []
        for result in results[:5]:
            metadata = result.metadata
            data_type = metadata.get('data_type', 'unknown')
            job_name = metadata.get('job_name', 'Unknown Job')
            
            # Focus on specification-relevant information
            if data_type == 'estimate':
                areas = metadata.get('areas', '')
                total_cost = metadata.get('total_estimated_cost', 0)
                context_parts.append(f"[ESTIMATE - {job_name}] Areas: {areas} | Estimated: ${total_cost:,.2f}\n{result.content[:400]}")
            else:
                context_parts.append(f"[{data_type.upper()} - {job_name}]\n{result.content[:400]}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""As a construction specification specialist, analyze the following project data to answer the query about technical requirements, specifications, and construction details.

Query: {query}

Construction project data:
{context}

Provide a detailed response that:
- Focuses on technical specifications and requirements
- Includes specific materials, methods, and standards mentioned
- Highlights any areas, task scopes, or technical details
- Provides cost information when relevant to specifications
- Uses construction industry terminology appropriately
- Is practical and actionable for construction professionals

Response:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=config.embedding.completion_model_complex,
                messages=[
                    {"role": "system", "content": "You are a construction specification specialist with expertise in technical requirements, materials, and construction methods."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Specification analysis available but response generation failed: {e}"

class ComplianceAgent:
    """Agent specialized in building codes, compliance, and regulatory matters"""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
        self.role = AgentRole.COMPLIANCE
    
    async def process(self, task: AgentTask) -> AgentResponse:
        """Process compliance and regulatory queries"""
        start_time = datetime.now()
        
        try:
            # Focus on compliance-relevant data
            search_filters = task.context.get('filters', {})
            
            results = await self.retriever.retrieve(
                query=f"compliance codes regulations {task.query}",
                n_results=6,
                filters=search_filters,
                strategy="hybrid"
            )
            
            response = await self._generate_compliance_response(task.query, results)
            
            return AgentResponse(
                agent_role=self.role,
                response=response,
                confidence=0.80,
                sources=results[:4],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=AgentStatus.COMPLETED,
                reasoning="Analyzed compliance and regulatory data",
                metadata={"compliance_focus": True}
            )
            
        except Exception as e:
            logger.error(f"❌ Compliance agent failed: {e}")
            return AgentResponse(
                agent_role=self.role,
                response=f"Error processing compliance query: {str(e)}",
                confidence=0.0,
                sources=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=AgentStatus.FAILED,
                reasoning=f"Agent execution failed: {e}",
                metadata={"error": str(e)}
            )
    
    async def _generate_compliance_response(self, query: str, results: List[SearchResult]) -> str:
        """Generate compliance-focused response"""
        context_parts = []
        for result in results:
            job_name = result.metadata.get('job_name', 'Unknown')
            context_parts.append(f"[{job_name}] {result.content[:300]}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""As a construction compliance specialist, analyze the project data to answer questions about building codes, regulations, permits, and compliance requirements.

Query: {query}

Project data:
{context}

Provide a response focused on:
- Building codes and regulatory compliance
- Permit requirements and approval processes  
- Safety regulations and standards
- Inspection requirements
- Compliance risks and mitigation strategies
- Regulatory best practices

Response:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=config.embedding.completion_model_complex,
                messages=[
                    {"role": "system", "content": "You are a construction compliance specialist with expertise in building codes, regulations, and permit processes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Compliance information available but response generation failed: {e}"

class CostAnalysisAgent:
    """Agent specialized in cost analysis, budgeting, and financial planning"""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
        self.role = AgentRole.COST
    
    async def process(self, task: AgentTask) -> AgentResponse:
        """Process cost and budget-related queries"""
        start_time = datetime.now()
        
        try:
            # Prioritize cost-relevant data types
            search_filters = {
                'data_type': ['consumed', 'estimate'],
                **task.context.get('filters', {})
            }
            
            results = await self.retriever.retrieve(
                query=task.query,
                n_results=10,
                filters=search_filters,
                strategy="hybrid_with_reranking"
            )
            
            response = await self._generate_cost_response(task.query, results)
            
            # Calculate confidence based on cost data availability
            cost_data_count = sum(1 for r in results if r.metadata.get('total_cost', 0) > 0 or r.metadata.get('total_estimated_cost', 0) > 0)
            confidence = min(0.95, 0.6 + (cost_data_count * 0.05))
            
            return AgentResponse(
                agent_role=self.role,
                response=response,
                confidence=confidence,
                sources=results[:6],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=AgentStatus.COMPLETED,
                reasoning="Analyzed cost and budget data across projects",
                metadata={"cost_data_sources": cost_data_count}
            )
            
        except Exception as e:
            logger.error(f"❌ Cost analysis agent failed: {e}")
            return AgentResponse(
                agent_role=self.role,
                response=f"Error processing cost query: {str(e)}",
                confidence=0.0,
                sources=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=AgentStatus.FAILED,
                reasoning=f"Agent execution failed: {e}",
                metadata={"error": str(e)}
            )
    
    async def _generate_cost_response(self, query: str, results: List[SearchResult]) -> str:
        """Generate cost-focused response with financial analysis"""
        # Separate results by data type for better analysis
        consumed_data = []
        estimate_data = []
        
        for result in results:
            data_type = result.metadata.get('data_type', '')
            if data_type == 'consumed':
                consumed_data.append(result)
            elif data_type == 'estimate':
                estimate_data.append(result)
        
        context_parts = []
        
        # Add consumed cost data
        if consumed_data:
            context_parts.append("ACTUAL COSTS (CONSUMED):")
            for result in consumed_data[:4]:
                job_name = result.metadata.get('job_name', 'Unknown')
                total_cost = result.metadata.get('total_cost', 0)
                categories = result.metadata.get('categories', '')
                context_parts.append(f"• {job_name}: ${total_cost:,.2f} | Categories: {categories}")
                context_parts.append(f"  Details: {result.content[:200]}...")
        
        # Add estimate data
        if estimate_data:
            context_parts.append("\nESTIMATED COSTS (BUDGETS):")
            for result in estimate_data[:4]:
                job_name = result.metadata.get('job_name', 'Unknown')
                estimated = result.metadata.get('total_estimated_cost', 0)
                budgeted = result.metadata.get('total_budgeted_cost', 0)
                context_parts.append(f"• {job_name}: ${estimated:,.2f} estimated / ${budgeted:,.2f} budgeted")
                context_parts.append(f"  Details: {result.content[:200]}...")
        
        context = "\n".join(context_parts)
        
        prompt = f"""As a construction cost analysis specialist, analyze the financial data to provide insights about costs, budgets, and financial planning.

Query: {query}

Cost and budget data:
{context}

Provide a comprehensive response that:
- Analyzes actual vs estimated costs when both are available
- Identifies cost patterns and trends
- Breaks down costs by category (labor, materials, subcontractors)
- Highlights budget variances and potential issues
- Provides actionable cost management recommendations
- Uses specific dollar amounts and percentages from the data

Response:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=config.embedding.completion_model_complex,
                messages=[
                    {"role": "system", "content": "You are a construction cost analysis specialist with expertise in budgeting, cost control, and financial planning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Cost data available but analysis generation failed: {e}"

class ScheduleAgent:
    """Agent specialized in project scheduling, timelines, and resource planning"""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
        self.role = AgentRole.SCHEDULE
    
    async def process(self, task: AgentTask) -> AgentResponse:
        """Process schedule and timeline-related queries"""
        start_time = datetime.now()
        
        try:
            search_filters = {
                'data_type': ['schedule'],
                **task.context.get('filters', {})
            }
            
            results = await self.retriever.retrieve(
                query=task.query,
                n_results=8,
                filters=search_filters,
                strategy="hybrid"
            )
            
            response = await self._generate_schedule_response(task.query, results)
            
            return AgentResponse(
                agent_role=self.role,
                response=response,
                confidence=0.80,
                sources=results[:5],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=AgentStatus.COMPLETED,
                reasoning="Analyzed project schedule and timeline data",
                metadata={"schedule_focus": True}
            )
            
        except Exception as e:
            logger.error(f"❌ Schedule agent failed: {e}")
            return AgentResponse(
                agent_role=self.role,
                response=f"Error processing schedule query: {str(e)}",
                confidence=0.0,
                sources=[],
                execution_time=(datetime.now() - start_time).total_seconds(),
                status=AgentStatus.FAILED,
                reasoning=f"Agent execution failed: {e}",
                metadata={"error": str(e)}
            )
    
    async def _generate_schedule_response(self, query: str, results: List[SearchResult]) -> str:
        """Generate schedule-focused response"""
        context_parts = []
        for result in results:
            job_name = result.metadata.get('job_name', 'Unknown')
            total_hours = result.metadata.get('total_planned_hours', 0)
            consumed_hours = result.metadata.get('total_consumed_hours', 0)
            start_date = result.metadata.get('project_start_date', 'TBD')
            end_date = result.metadata.get('project_end_date', 'TBD')
            
            context_parts.append(f"[{job_name}] {start_date} to {end_date} | {total_hours:.1f}h planned, {consumed_hours:.1f}h consumed")
            context_parts.append(result.content[:250])
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""As a construction scheduling specialist, analyze the project timeline data to answer questions about schedules, deadlines, and project planning.

Query: {query}

Schedule data:
{context}

Provide insights on:
- Project timelines and critical path activities
- Resource allocation and scheduling conflicts
- Task dependencies and sequencing
- Schedule progress and completion rates
- Deadline risks and mitigation strategies
- Resource optimization opportunities

Response:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=config.embedding.completion_model_complex,
                messages=[
                    {"role": "system", "content": "You are a construction scheduling specialist with expertise in project timelines, resource planning, and critical path analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Schedule data available but analysis generation failed: {e}"

class MultiAgentCoordinator:
    """Coordinates multiple specialized agents for complex queries"""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.agents = {
            AgentRole.SPECIFICATION: SpecificationAgent(retriever),
            AgentRole.COMPLIANCE: ComplianceAgent(retriever),
            AgentRole.COST: CostAnalysisAgent(retriever),
            AgentRole.SCHEDULE: ScheduleAgent(retriever),
        }
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
    
    async def process_query(self, 
                          query: str,
                          classification: QueryClassification,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query using appropriate agent(s) based on classification"""
        
        if not config.agents.enable_multi_agent:
            # Fallback to single general response
            return await self._single_agent_fallback(query, context or {})
        
        context = context or {}
        
        # Determine which agents to use
        agent_tasks = self._create_agent_tasks(query, classification, context)
        
        if not agent_tasks:
            return await self._single_agent_fallback(query, context)
        
        # Execute agents
        if len(agent_tasks) == 1:
            # Single agent execution
            agent_responses = [await self._execute_agent_task(agent_tasks[0])]
        else:
            # Multi-agent parallel execution
            agent_responses = await self._execute_multi_agent_tasks(agent_tasks)
        
        # Synthesize responses
        final_response = await self._synthesize_responses(query, agent_responses, classification)
        
        return {
            "answer": final_response["synthesized_response"],
            "agent_responses": [
                {
                    "agent": resp.agent_role.value,
                    "response": resp.response,
                    "confidence": resp.confidence,
                    "status": resp.status.value,
                    "execution_time": resp.execution_time
                }
                for resp in agent_responses
            ],
            "sources": self._collect_all_sources(agent_responses),
            "coordination_metadata": {
                "agents_used": [task.agent_role.value for task in agent_tasks],
                "parallel_execution": len(agent_tasks) > 1,
                "synthesis_approach": final_response["approach"],
                "overall_confidence": final_response["confidence"]
            }
        }
    
    def _create_agent_tasks(self, 
                           query: str,
                           classification: QueryClassification,
                           context: Dict[str, Any]) -> List[AgentTask]:
        """Create tasks for appropriate agents based on query classification"""
        tasks = []
        
        # Map suggested agents to tasks
        for agent_name in classification.suggested_agents:
            if agent_name == 'specification' and config.agents.enable_specification_agent:
                tasks.append(AgentTask(
                    agent_role=AgentRole.SPECIFICATION,
                    query=query,
                    context=context,
                    priority=1
                ))
            elif agent_name == 'compliance' and config.agents.enable_compliance_agent:
                tasks.append(AgentTask(
                    agent_role=AgentRole.COMPLIANCE,
                    query=query,
                    context=context,
                    priority=2
                ))
            elif agent_name == 'cost' and config.agents.enable_cost_agent:
                tasks.append(AgentTask(
                    agent_role=AgentRole.COST,
                    query=query,
                    context=context,
                    priority=1
                ))
            elif agent_name == 'schedule' and config.agents.enable_schedule_agent:
                tasks.append(AgentTask(
                    agent_role=AgentRole.SCHEDULE,
                    query=query,
                    context=context,
                    priority=1
                ))
        
        # Sort by priority
        tasks.sort(key=lambda t: t.priority)
        
        # Limit concurrent agents
        return tasks[:config.agents.max_concurrent_agents]
    
    async def _execute_agent_task(self, task: AgentTask) -> AgentResponse:
        """Execute a single agent task with timeout"""
        try:
            agent = self.agents[task.agent_role]
            return await asyncio.wait_for(agent.process(task), timeout=task.timeout)
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Agent {task.agent_role.value} timed out")
            return AgentResponse(
                agent_role=task.agent_role,
                response=f"Agent {task.agent_role.value} timed out",
                confidence=0.0,
                sources=[],
                execution_time=task.timeout,
                status=AgentStatus.TIMEOUT,
                reasoning="Agent execution timed out",
                metadata={"timeout": task.timeout}
            )
        except Exception as e:
            logger.error(f"❌ Agent {task.agent_role.value} failed: {e}")
            return AgentResponse(
                agent_role=task.agent_role,
                response=f"Agent {task.agent_role.value} failed: {e}",
                confidence=0.0,
                sources=[],
                execution_time=0.0,
                status=AgentStatus.FAILED,
                reasoning=f"Agent execution failed: {e}",
                metadata={"error": str(e)}
            )
    
    async def _execute_multi_agent_tasks(self, tasks: List[AgentTask]) -> List[AgentResponse]:
        """Execute multiple agent tasks in parallel"""
        logger.info(f"🤖 Executing {len(tasks)} agents in parallel")
        
        # Create tasks for parallel execution
        agent_coroutines = [self._execute_agent_task(task) for task in tasks]
        
        # Execute with timeout
        try:
            responses = await asyncio.gather(*agent_coroutines, return_exceptions=True)
            
            # Process responses and handle exceptions
            final_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"❌ Agent task {i} failed with exception: {response}")
                    final_responses.append(AgentResponse(
                        agent_role=tasks[i].agent_role,
                        response=f"Agent failed with exception: {response}",
                        confidence=0.0,
                        sources=[],
                        execution_time=0.0,
                        status=AgentStatus.FAILED,
                        reasoning=f"Exception during execution: {response}",
                        metadata={"exception": str(response)}
                    ))
                else:
                    final_responses.append(response)
            
            return final_responses
            
        except Exception as e:
            logger.error(f"❌ Multi-agent execution failed: {e}")
            return []
    
    async def _synthesize_responses(self, 
                                  query: str,
                                  agent_responses: List[AgentResponse],
                                  classification: QueryClassification) -> Dict[str, Any]:
        """Synthesize multiple agent responses into a coherent answer"""
        
        # Filter successful responses
        successful_responses = [r for r in agent_responses if r.status == AgentStatus.COMPLETED and r.confidence > 0.3]
        
        if not successful_responses:
            return {
                "synthesized_response": "I apologize, but I couldn't generate a reliable response to your query. Please try rephrasing or asking a more specific question.",
                "approach": "fallback",
                "confidence": 0.1
            }
        
        if len(successful_responses) == 1:
            # Single successful response
            return {
                "synthesized_response": successful_responses[0].response,
                "approach": "single_agent",
                "confidence": successful_responses[0].confidence
            }
        
        # Multiple responses - synthesize them
        try:
            synthesis_prompt = self._create_synthesis_prompt(query, successful_responses, classification)
            
            response = self.openai_client.chat.completions.create(
                model=config.embedding.completion_model_complex,
                messages=[
                    {"role": "system", "content": "You are a construction project coordinator who synthesizes insights from multiple specialists."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            synthesized = response.choices[0].message.content.strip()
            avg_confidence = sum(r.confidence for r in successful_responses) / len(successful_responses)
            
            return {
                "synthesized_response": synthesized,
                "approach": "multi_agent_synthesis",
                "confidence": min(avg_confidence * 1.1, 0.95)  # Boost confidence for multi-agent
            }
            
        except Exception as e:
            logger.error(f"❌ Response synthesis failed: {e}")
            # Fallback to best single response
            best_response = max(successful_responses, key=lambda r: r.confidence)
            return {
                "synthesized_response": best_response.response,
                "approach": "fallback_to_best",
                "confidence": best_response.confidence * 0.9  # Slight penalty for synthesis failure
            }
    
    def _create_synthesis_prompt(self, 
                                query: str,
                                responses: List[AgentResponse],
                                classification: QueryClassification) -> str:
        """Create prompt for synthesizing multiple agent responses"""
        prompt_parts = [
            f"Original Query: \"{query}\"",
            f"Query Type: {classification.query_type.value}",
            f"Complexity: {classification.complexity.value}",
            "",
            "Specialist Responses:",
            ""
        ]
        
        for response in responses:
            prompt_parts.extend([
                f"{response.agent_role.value.upper()} SPECIALIST (confidence: {response.confidence:.2f}):",
                response.response,
                ""
            ])
        
        prompt_parts.extend([
            "Synthesize these specialist perspectives into a single, comprehensive response that:",
            "1. Integrates insights from all specialists appropriately",
            "2. Resolves any conflicts or contradictions between responses",
            "3. Provides a clear, actionable answer to the original query", 
            "4. Maintains construction industry terminology and precision",
            "5. Prioritizes information based on specialist confidence and relevance",
            "",
            "Synthesized Response:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _collect_all_sources(self, agent_responses: List[AgentResponse]) -> List[SearchResult]:
        """Collect and deduplicate sources from all agent responses"""
        all_sources = []
        seen_ids = set()
        
        for response in agent_responses:
            for source in response.sources:
                if source.id not in seen_ids:
                    all_sources.append(source)
                    seen_ids.add(source.id)
        
        # Sort by combined score
        all_sources.sort(key=lambda s: s.combined_score, reverse=True)
        return all_sources[:10]  # Return top 10 unique sources
    
    async def _single_agent_fallback(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to single general agent when multi-agent is disabled"""
        # Use cost agent as general purpose agent (has broad search capabilities)
        if config.agents.enable_cost_agent:
            task = AgentTask(
                agent_role=AgentRole.COST,
                query=query,
                context=context
            )
            response = await self._execute_agent_task(task)
        else:
            # Basic fallback response
            response = AgentResponse(
                agent_role=AgentRole.GENERAL,
                response="Multi-agent system is disabled. Please enable specialized agents for better responses.",
                confidence=0.3,
                sources=[],
                execution_time=0.0,
                status=AgentStatus.COMPLETED,
                reasoning="Multi-agent disabled, using fallback",
                metadata={"fallback": True}
            )
        
        return {
            "answer": response.response,
            "agent_responses": [{
                "agent": response.agent_role.value,
                "response": response.response,
                "confidence": response.confidence,
                "status": response.status.value,
                "execution_time": response.execution_time
            }],
            "sources": response.sources,
            "coordination_metadata": {
                "agents_used": [response.agent_role.value],
                "parallel_execution": False,
                "synthesis_approach": "single_fallback",
                "overall_confidence": response.confidence
            }
        }

# Global multi-agent coordinator instance
_multi_agent_coordinator = None

def get_multi_agent_coordinator(retriever: HybridRetriever) -> MultiAgentCoordinator:
    """Get or create global multi-agent coordinator instance"""
    global _multi_agent_coordinator
    if _multi_agent_coordinator is None:
        _multi_agent_coordinator = MultiAgentCoordinator(retriever)
    return _multi_agent_coordinator