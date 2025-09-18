import os
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from openai import OpenAI
from datetime import datetime
from dataclasses import asdict

# Enhanced imports
from core.config import config
from core.qdrant_client import get_qdrant_client
from database import get_firebase_db, fetch_all_job_complete_data, migrate_chromadb_to_qdrant
from embedding_service import get_embedding_service
from processing.advanced_chunking import get_advanced_chunking
from agents.query_router import get_query_router, QueryClassification, DataTypeFilter
from agents.multi_agent import get_multi_agent_coordinator
from retrieval.hybrid_retriever import get_hybrid_retriever, SearchResult
from retrieval.corrective_rag import get_corrective_rag
from models import DocumentSource

logger = logging.getLogger(__name__)

class EnhancedRAGService:
    """Enhanced RAG service with multi-agent coordination and advanced retrieval"""
    
    def __init__(self, embedding_service=None):
        # Initialize core services
        self.embedding_service = embedding_service or get_embedding_service()
        self.openai_client = OpenAI(api_key=config.embedding.openai_api_key)
        
        # Initialize enhanced components
        self.qdrant_client = get_qdrant_client()
        self.advanced_chunking = get_advanced_chunking()
        self.query_router = get_query_router()
        
        # Initialize retrieval components
        self.hybrid_retriever = get_hybrid_retriever(self.embedding_service)
        self.corrective_rag = get_corrective_rag(self.hybrid_retriever)
        self.multi_agent_coordinator = get_multi_agent_coordinator(self.hybrid_retriever)
        
        # Service statistics
        self.stats = {
            'queries_processed': 0,
            'documents_processed': 0,
            'multi_agent_queries': 0,
            'corrective_rag_activations': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0,
            'average_response_time': 0.0,
            'last_processing_run': None
        }
        
        logger.info("🚀 Enhanced RAG Service initialized")
        logger.info(f"📊 Multi-agent enabled: {config.agents.enable_multi_agent}")
        logger.info(f"🔧 Corrective RAG enabled: {config.retrieval.enable_corrective_rag}")
    
    async def process_firebase_data(self, company_id: Optional[str] = None) -> Dict[str, Any]:
        """Process Firebase data with enhanced chunking and vector storage"""
        logger.info("🔄 Starting enhanced Firebase data processing...")
        
        start_time = datetime.now()
        processing_stats = {
            'total_jobs_processed': 0,
            'total_documents_created': 0,
            'total_chunks_created': 0,
            'consumed_datasets': 0,
            'estimate_datasets': 0,
            'schedule_datasets': 0,
            'companies_processed': [],
            'processing_time_seconds': 0.0,
            'errors': []
        }
        
        try:
            # Fetch all job data from Firebase
            job_datasets = await fetch_all_job_complete_data(company_id)
            
            if not job_datasets:
                logger.warning("No job data found in Firebase")
                return processing_stats
            
            logger.info(f"📊 Found {len(job_datasets)} datasets to process")
            
            # Process each dataset with enhanced chunking
            all_points = []
            jobs_processed = set()
            
            for job_dataset in job_datasets:
                try:
                    data_type = job_dataset.get('data_type', 'unknown')
                    job_id = job_dataset.get('job_id', 'unknown')
                    company_id_current = job_dataset.get('company_id', 'unknown')
                    
                    logger.debug(f"Processing {data_type} data for job {job_id}")
                    
                    # Use advanced chunking for better document processing
                    processed_chunks = self.embedding_service.process_job_data_with_chunking(job_dataset)
                    
                    if not processed_chunks:
                        logger.warning(f"No chunks created for {data_type} data of job {job_id}")
                        continue
                    
                    # Convert chunks to Qdrant points
                    for chunk in processed_chunks:
                        if not chunk.embedding:
                            logger.warning(f"Chunk {chunk.id} has no embedding")
                            continue
                        
                        # Create Qdrant point with enhanced metadata
                        from qdrant_client.models import PointStruct
                        
                        point = PointStruct(
                            id=chunk.id,
                            vector=chunk.embedding,
                            payload={
                                'document_content': chunk.content,
                                'chunk_metadata': asdict(chunk.metadata),
                                
                                # Flatten key metadata for filtering
                                'job_name': chunk.metadata.job_context.get('job_name', ''),
                                'company_id': chunk.metadata.job_context.get('company_id', ''),
                                'job_id': chunk.metadata.job_context.get('job_id', ''),
                                'data_type': chunk.metadata.data_type,
                                'document_type': f'job_{chunk.metadata.data_type}_data',
                                'construction_category': chunk.metadata.construction_category,
                                'chunk_type': chunk.metadata.chunk_type,
                                'chunk_level': chunk.metadata.chunk_level,
                                
                                # Processing metadata
                                'processed_at': datetime.now().isoformat(),
                                'processing_version': '2.0',
                                'chunk_size': chunk.metadata.chunk_size
                            }
                        )
                        
                        all_points.append(point)
                    
                    # Update statistics
                    processing_stats['total_documents_created'] += 1
                    processing_stats['total_chunks_created'] += len(processed_chunks)
                    jobs_processed.add(f"{company_id_current}_{job_id}")
                    
                    if data_type == 'consumed':
                        processing_stats['consumed_datasets'] += 1
                    elif data_type == 'estimate':
                        processing_stats['estimate_datasets'] += 1
                    elif data_type == 'schedule':
                        processing_stats['schedule_datasets'] += 1
                    
                    if company_id_current not in processing_stats['companies_processed']:
                        processing_stats['companies_processed'].append(company_id_current)
                    
                    logger.debug(f"✅ Created {len(processed_chunks)} chunks for {data_type} data of job {job_id}")
                    
                except Exception as e:
                    error_msg = f"Error processing {job_dataset.get('data_type', 'unknown')} data for job {job_dataset.get('job_id', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    processing_stats['errors'].append(error_msg)
            
            # Update job count
            processing_stats['total_jobs_processed'] = len(jobs_processed)
            
            # Batch insert all points into Qdrant
            if all_points:
                logger.info(f"📝 Inserting {len(all_points)} chunks into Qdrant...")
                
                success = self.qdrant_client.add_points(all_points, batch_size=50)
                
                if success:
                    logger.info(f"✅ Successfully inserted {len(all_points)} chunks")
                    logger.info(f"   - {processing_stats['consumed_datasets']} consumed datasets")
                    logger.info(f"   - {processing_stats['estimate_datasets']} estimate datasets") 
                    logger.info(f"   - {processing_stats['schedule_datasets']} schedule datasets")
                else:
                    processing_stats['errors'].append("Failed to insert chunks into Qdrant")
            
            # Update service statistics
            self.stats['documents_processed'] += processing_stats['total_documents_created']
            self.stats['last_processing_run'] = datetime.now().isoformat()
            
            # Calculate processing time
            end_time = datetime.now()
            processing_stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Enhanced processing complete!")
            logger.info(f"📊 Processed {processing_stats['total_jobs_processed']} jobs into {processing_stats['total_chunks_created']} chunks in {processing_stats['processing_time_seconds']:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Fatal error in enhanced process_firebase_data: {str(e)}"
            logger.error(error_msg)
            processing_stats['errors'].append(error_msg)
        
        return processing_stats
    
    async def query(self, 
                   question: str,
                   n_results: int = 5,
                   data_types: Optional[List[str]] = None,
                   use_multi_agent: Optional[bool] = None,
                   use_corrective_rag: Optional[bool] = None) -> Dict[str, Any]:
        """Enhanced query processing with intelligent routing and multi-agent coordination"""
        
        start_time = time.time()
        self.stats['queries_processed'] += 1
        
        try:
            logger.info(f"🔍 Processing query: {question[:100]}...")
            
            # Step 1: Intelligent query classification and routing
            classification = await self.query_router.classify_query(question)
            routing_strategy = self.query_router.get_routing_strategy(classification)
            
            logger.info(f"🎯 Query classified as {classification.query_type.value} ({classification.complexity.value}) with {classification.confidence:.2f} confidence")
            
            # Step 2: Prepare search parameters
            search_filters = self._prepare_search_filters(data_types, classification)
            retrieval_config = routing_strategy['retrieval_config']
            
            # Override n_results with classification suggestion if not specified
            final_n_results = n_results or retrieval_config['n_results']
            
            # Step 3: Enhanced retrieval with optional corrective RAG
            use_corrective = use_corrective_rag if use_corrective_rag is not None else config.retrieval.enable_corrective_rag
            
            if use_corrective:
                logger.debug("🔧 Using corrective RAG")
                search_results, correction_metadata = await self.corrective_rag.corrective_retrieve(
                    query=question,
                    n_results=final_n_results,
                    filters=search_filters,
                    strategy=retrieval_config['strategy']
                )
                self.stats['corrective_rag_activations'] += 1
            else:
                logger.debug("🔍 Using standard hybrid retrieval")
                search_results = await self.hybrid_retriever.retrieve(
                    query=question,
                    n_results=final_n_results,
                    filters=search_filters,
                    strategy=retrieval_config['strategy']
                )
                correction_metadata = {}
            
            # Step 4: Multi-agent processing or standard response generation
            use_agents = use_multi_agent if use_multi_agent is not None else (
                config.agents.enable_multi_agent and 
                len(classification.suggested_agents) > 1 and
                classification.confidence > 0.7
            )
            
            if use_agents and len(search_results) > 0:
                logger.info(f"🤖 Using multi-agent processing with {len(classification.suggested_agents)} agents")
                
                # Prepare context for agents
                agent_context = {
                    'filters': search_filters,
                    'classification': classification,
                    'routing_strategy': routing_strategy
                }
                
                # Multi-agent processing
                agent_response = await self.multi_agent_coordinator.process_query(
                    query=question,
                    classification=classification,
                    context=agent_context
                )
                
                answer = agent_response['answer']
                sources = self._convert_search_results_to_sources(agent_response['sources'])
                
                self.stats['multi_agent_queries'] += 1
                
                # Combine metadata
                response_metadata = {
                    'processing_approach': 'multi_agent',
                    'agent_coordination': agent_response['coordination_metadata'],
                    'query_classification': asdict(classification),
                    'routing_strategy': routing_strategy,
                    'correction_metadata': correction_metadata,
                    'search_results_count': len(search_results)
                }
                
            else:
                logger.debug("🤖 Using standard response generation")
                
                if not search_results:
                    answer = "I couldn't find any relevant construction data for your query. Please try rephrasing your question or check if data has been loaded into the system."
                    sources = []
                else:
                    answer = await self._generate_enhanced_answer(question, search_results, classification)
                    sources = self._convert_search_results_to_sources(search_results)
                
                response_metadata = {
                    'processing_approach': 'standard',
                    'query_classification': asdict(classification),
                    'routing_strategy': routing_strategy,
                    'correction_metadata': correction_metadata,
                    'search_results_count': len(search_results)
                }
            
            # Step 5: Prepare final response
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['average_response_time'] = self.stats['total_processing_time'] / self.stats['queries_processed']
            
            # Extract data types found
            data_types_found = list(set(result.metadata.get('data_type', 'unknown') for result in search_results))
            
            final_response = {
                "answer": answer,
                "sources": sources,
                "chunks": [result.content[:300] + "..." if len(result.content) > 300 else result.content for result in search_results],
                "data_types_found": data_types_found,
                "document_types_found": list(set(result.metadata.get('document_type', 'unknown') for result in search_results)),
                "processing_time": processing_time,
                "metadata": response_metadata
            }
            
            logger.info(f"✅ Query processed in {processing_time:.2f}s with {len(sources)} sources")
            
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            
            # Return error response
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}. Please try again or contact support if the issue persists.",
                "sources": [],
                "chunks": [],
                "data_types_found": [],
                "document_types_found": [],
                "processing_time": time.time() - start_time,
                "metadata": {
                    "processing_approach": "error",
                    "error": str(e)
                }
            }
    
    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a document manually with enhanced processing"""
        try:
            start_time = time.time()
            
            # Use advanced chunking if text is long enough
            if len(text) > 1000:
                chunks = self.advanced_chunking.chunk_construction_document(
                    content=text,
                    job_metadata=metadata,
                    document_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                points = []
                for chunk in chunks:
                    # Generate embedding for chunk
                    embedding = self.embedding_service.generate_embedding(chunk.content)
                    
                    from qdrant_client.models import PointStruct
                    point = PointStruct(
                        id=chunk.id,
                        vector=embedding,
                        payload={
                            'document_content': chunk.content,
                            'manual_metadata': metadata,
                            'chunk_metadata': asdict(chunk.metadata),
                            'added_at': datetime.now().isoformat(),
                            'document_type': 'manual',
                            'data_type': metadata.get('data_type', 'manual')
                        }
                    )
                    points.append(point)
                
                # Add to Qdrant
                success = self.qdrant_client.add_points(points)
                
                if success:
                    logger.info(f"✅ Added manual document as {len(points)} chunks")
                    return chunks[0].id  # Return first chunk ID as document ID
                else:
                    raise Exception("Failed to add chunks to Qdrant")
            
            else:
                # Single document processing
                embedding = self.embedding_service.generate_embedding(text)
                
                doc_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                metadata['added_at'] = datetime.now().isoformat()
                metadata['document_type'] = metadata.get('document_type', 'manual')
                
                from qdrant_client.models import PointStruct
                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        'document_content': text,
                        **metadata
                    }
                )
                
                success = self.qdrant_client.add_points([point])
                
                if success:
                    logger.info(f"✅ Added manual document with ID: {doc_id}")
                    return doc_id
                else:
                    raise Exception("Failed to add document to Qdrant")
                    
        except Exception as e:
            logger.error(f"❌ Error adding document: {e}")
            raise
    
    def _prepare_search_filters(self, 
                               data_types: Optional[List[str]], 
                               classification: QueryClassification) -> Optional[Dict[str, Any]]:
        """Prepare search filters based on data types and classification"""
        filters = {}
        
        # Use provided data types or infer from classification
        if data_types:
            filters['data_type'] = data_types
        elif classification.data_types and DataTypeFilter.ALL not in classification.data_types:
            # Convert DataTypeFilter enums to strings
            type_strings = []
            for dt in classification.data_types:
                if dt != DataTypeFilter.ALL:
                    type_strings.append(dt.value)
            
            if type_strings:
                filters['data_type'] = type_strings
        
        return filters if filters else None
    
    def _convert_search_results_to_sources(self, search_results: List[SearchResult]) -> List[DocumentSource]:
        """Convert search results to DocumentSource objects"""
        sources = []
        
        for result in search_results:
            metadata = result.metadata
            
            # Extract cost information based on data type
            cost_info = None
            data_type = metadata.get('data_type', 'unknown')
            
            if data_type == 'consumed':
                total_cost = metadata.get('total_cost') or metadata.get('category_materials_total', 0) or 0
                cost_info = f"${total_cost:,.2f} consumed" if total_cost > 0 else "Consumed data"
            elif data_type == 'estimate':
                estimated = metadata.get('total_estimated_cost', 0)
                budgeted = metadata.get('total_budgeted_cost', 0)
                if estimated > 0 or budgeted > 0:
                    cost_info = f"${estimated:,.2f} est. / ${budgeted:,.2f} budgeted"
                else:
                    cost_info = "Estimate data"
            elif data_type == 'schedule':
                hours = metadata.get('total_planned_hours', 0)
                consumed = metadata.get('total_consumed_hours', 0)
                if hours > 0 or consumed > 0:
                    cost_info = f"{hours:.1f}h planned / {consumed:.1f}h consumed"
                else:
                    cost_info = "Schedule data"
            else:
                cost_info = f"{data_type.title()} data"
            
            source = DocumentSource(
                job_name=metadata.get('job_name', 'Unknown'),
                company_id=metadata.get('company_id', ''),
                job_id=metadata.get('job_id', ''),
                cost_code=metadata.get('categories', '') or metadata.get('areas', '') or metadata.get('construction_category', ''),
                amount=cost_info,
                last_updated=metadata.get('last_updated', '') or metadata.get('processed_at', '')
            )
            sources.append(source)
        
        return sources
    
    async def _generate_enhanced_answer(self, 
                                       question: str, 
                                       search_results: List[SearchResult],
                                       classification: QueryClassification) -> str:
        """Generate enhanced answer using OpenAI with query classification context"""
        try:
            # Create enhanced context with classification awareness
            context_parts = []
            
            for result in search_results:
                metadata = result.metadata
                data_type = metadata.get('data_type', 'unknown')
                job_name = metadata.get('job_name', 'Unknown Job')
                
                # Add data type specific context
                context_header = f"[{data_type.upper()} DATA - {job_name}]"
                
                # Add relevant metadata based on data type
                if data_type == 'consumed':
                    cost = metadata.get('total_cost', 0)
                    categories = metadata.get('categories', '')
                    context_header += f" Cost: ${cost:,.2f} | Categories: {categories}"
                elif data_type == 'estimate':
                    estimated = metadata.get('total_estimated_cost', 0)
                    budgeted = metadata.get('total_budgeted_cost', 0)
                    areas = metadata.get('areas', '')
                    context_header += f" Estimated: ${estimated:,.2f} | Budgeted: ${budgeted:,.2f} | Areas: {areas}"
                elif data_type == 'schedule':
                    hours = metadata.get('total_planned_hours', 0)
                    consumed = metadata.get('total_consumed_hours', 0)
                    completion = metadata.get('completion_rate', 0)
                    context_header += f" Planned: {hours:.1f}h | Consumed: {consumed:.1f}h | Progress: {completion:.1%}"
                
                # Limit content length for context
                content = result.content[:500] + "..." if len(result.content) > 500 else result.content
                context_parts.append(f"{context_header}\n{content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Select appropriate model based on complexity
            model = self.embedding_service.model_router.select_model(question, len(context))
            
            # Create enhanced prompt with classification context
            system_prompt = f"""You are an expert construction project assistant with access to comprehensive project data including consumed costs, estimates, and schedules. 

Query Classification:
- Type: {classification.query_type.value}
- Complexity: {classification.complexity.value}
- Confidence: {classification.confidence:.2f}

Based on this classification and the provided data, answer the user's question with appropriate focus and depth."""

            user_prompt = f"""Question: {question}

Construction project data:
{context}

Instructions:
- Answer based only on the provided context
- Distinguish between different data types (CONSUMED, ESTIMATE, SCHEDULE) when relevant
- Include specific dollar amounts, hours, dates, job names, and other details when available
- For cost queries, clearly distinguish between actual consumed costs and estimated/budgeted amounts
- For schedule queries, include timeline information, progress, and resource details
- If comparing data types (e.g., estimated vs consumed), highlight the differences clearly
- Be concise but comprehensive, matching the query complexity level ({classification.complexity.value})
- Use construction industry terminology appropriately
- If information is insufficient, state what's available and what's missing

Answer:"""

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=config.embedding.max_tokens,
                temperature=config.embedding.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced answer: {e}")
            return f"I found relevant construction data but encountered an error generating the response: {str(e)}"
    
    async def get_available_jobs(self) -> List[Dict[str, Any]]:
        """Get list of available jobs with enhanced metadata"""
        try:
            # Get job information from Qdrant
            collection_info = self.qdrant_client.get_collection_info()
            if not collection_info or collection_info.points_count == 0:
                return []
            
            # Query for unique jobs (using scroll or search)
            # This is a simplified approach - in production you might want to implement a dedicated job index
            dummy_vector = [0.0] * 1536
            results = self.qdrant_client.search(
                query_vector=dummy_vector,
                limit=100  # Get a good sample
            )
            
            # Aggregate job information
            jobs_info = {}
            
            for result in results:
                payload = result.payload
                job_name = payload.get('job_name', 'Unknown')
                job_id = payload.get('job_id', '')
                company_id = payload.get('company_id', '')
                data_type = payload.get('data_type', 'unknown')
                
                if job_name != 'Unknown' and job_id:
                    job_key = f"{company_id}_{job_id}"
                    
                    if job_key not in jobs_info:
                        jobs_info[job_key] = {
                            'job_name': job_name,
                            'job_id': job_id,
                            'company_id': company_id,
                            'data_types': [],
                            'chunk_count': 0,
                            'categories': set()
                        }
                    
                    if data_type not in jobs_info[job_key]['data_types']:
                        jobs_info[job_key]['data_types'].append(data_type)
                    
                    jobs_info[job_key]['chunk_count'] += 1
                    
                    # Add construction categories
                    construction_category = payload.get('construction_category', '')
                    if construction_category:
                        jobs_info[job_key]['categories'].add(construction_category)
            
            # Convert to list and clean up
            jobs_list = []
            for job_info in jobs_info.values():
                job_info['categories'] = list(job_info['categories'])
                jobs_list.append(job_info)
            
            return jobs_list
            
        except Exception as e:
            logger.error(f"❌ Error getting available jobs: {e}")
            return []
    
    async def get_data_types_summary(self) -> Dict[str, Any]:
        """Get enhanced summary of available data types"""
        try:
            stats = self.qdrant_client.get_construction_stats()
            
            if 'error' in stats:
                return {'error': stats['error']}
            
            # Enhance with service statistics
            enhanced_summary = {
                **stats,
                'service_stats': {
                    'queries_processed': self.stats['queries_processed'],
                    'multi_agent_queries': self.stats['multi_agent_queries'],
                    'corrective_rag_activations': self.stats['corrective_rag_activations'],
                    'average_response_time': self.stats['average_response_time'],
                    'last_processing_run': self.stats['last_processing_run']
                }
            }
            
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"❌ Error getting data types summary: {e}")
            return {'error': str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection and service statistics"""
        try:
            # Get Qdrant statistics
            qdrant_stats = self.qdrant_client.get_construction_stats()
            
            # Get embedding service statistics
            embedding_stats = self.embedding_service.get_service_stats()
            
            # Combine all statistics
            comprehensive_stats = {
                'qdrant_stats': qdrant_stats,
                'embedding_stats': embedding_stats,
                'rag_service_stats': {
                    'queries_processed': self.stats['queries_processed'],
                    'documents_processed': self.stats['documents_processed'],
                    'multi_agent_queries': self.stats['multi_agent_queries'],
                    'corrective_rag_activations': self.stats['corrective_rag_activations'],
                    'total_processing_time': self.stats['total_processing_time'],
                    'average_response_time': self.stats['average_response_time'],
                    'last_processing_run': self.stats['last_processing_run']
                },
                'system_health': {
                    'multi_agent_enabled': config.agents.enable_multi_agent,
                    'corrective_rag_enabled': config.retrieval.enable_corrective_rag,
                    'semantic_chunking_enabled': config.processing.enable_semantic_chunking,
                    'semantic_caching_enabled': config.embedding.enable_semantic_cache
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return comprehensive_stats
            
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {'error': str(e)}
    
    async def migrate_from_chromadb(self) -> Dict[str, Any]:
        """Migrate data from ChromaDB to Qdrant if needed"""
        try:
            return await migrate_chromadb_to_qdrant()
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all system components"""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        issues = []
        
        # Check Qdrant
        try:
            qdrant_health = self.qdrant_client.health_check()
            health_status['components']['qdrant'] = qdrant_health
            
            if qdrant_health['status'] != 'healthy':
                issues.append('Qdrant not healthy')
        except Exception as e:
            health_status['components']['qdrant'] = {'status': 'error', 'error': str(e)}
            issues.append('Qdrant connection failed')
        
        # Check embedding service
        try:
            embedding_test = self.embedding_service.test_embedding()
            health_status['components']['embeddings'] = {
                'status': 'healthy' if embedding_test else 'unhealthy',
                'test_passed': embedding_test
            }
            
            if not embedding_test:
                issues.append('Embedding service test failed')
        except Exception as e:
            health_status['components']['embeddings'] = {'status': 'error', 'error': str(e)}
            issues.append('Embedding service error')
        
        # Check Firebase
        try:
            db = get_firebase_db()
            companies = db.collection('companies').limit(1).get()
            health_status['components']['firebase'] = {
                'status': 'healthy',
                'connection_test': 'passed'
            }
        except Exception as e:
            health_status['components']['firebase'] = {'status': 'error', 'error': str(e)}
            issues.append('Firebase connection failed')
        
        # Overall status
        if issues:
            health_status['overall_status'] = 'degraded'
            health_status['issues'] = issues
        
        # Add service statistics
        health_status['service_stats'] = {
            'queries_processed': self.stats['queries_processed'],
            'uptime_features': {
                'multi_agent': config.agents.enable_multi_agent,
                'corrective_rag': config.retrieval.enable_corrective_rag,
                'advanced_chunking': config.processing.enable_semantic_chunking
            }
        }
        
        return health_status

# Global enhanced RAG service instance
_rag_service = None

def get_rag_service(embedding_service=None) -> EnhancedRAGService:
    """Get or create global RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = EnhancedRAGService(embedding_service)
    return _rag_service

# Backward compatibility
RAGService = EnhancedRAGService