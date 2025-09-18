import os
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import json
import asyncio

# Enhanced imports
from core.config import config
from processing.advanced_chunking import get_advanced_chunking, ProcessedChunk

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  Redis not available. Semantic caching disabled. Install with: pip install redis")

load_dotenv()

logger = logging.getLogger(__name__)

class SemanticCache:
    """Redis-based semantic cache for embeddings and responses"""
    
    def __init__(self):
        self.enabled = REDIS_AVAILABLE and config.embedding.enable_semantic_cache
        self.similarity_threshold = config.embedding.cache_similarity_threshold
        self.ttl = config.embedding.cache_ttl_seconds
        
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=config.cache.redis_host,
                    port=config.cache.redis_port,
                    db=config.cache.redis_db,
                    password=config.cache.redis_password,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("✅ Semantic cache initialized (Redis)")
            except Exception as e:
                logger.warning(f"⚠️  Redis connection failed, disabling cache: {e}")
                self.enabled = False
        else:
            self.redis_client = None
            logger.info("📦 Semantic cache disabled")
    
    def _generate_cache_key(self, text: str, prefix: str = "embedding") -> str:
        """Generate cache key from text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{config.cache.cache_prefix}:{prefix}:{text_hash}"
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text"""
        if not self.enabled:
            return None
        
        try:
            cache_key = self._generate_cache_key(text, "embedding")
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                embedding_data = json.loads(cached_data)
                # Update hit count
                self.redis_client.hincrby(f"{cache_key}:meta", "hits", 1)
                logger.debug(f"🎯 Cache hit for embedding")
                return embedding_data["embedding"]
                
        except Exception as e:
            logger.warning(f"⚠️  Cache retrieval failed: {e}")
        
        return None
    
    def cache_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding for text"""
        if not self.enabled:
            return False
        
        try:
            cache_key = self._generate_cache_key(text, "embedding")
            
            embedding_data = {
                "embedding": embedding,
                "text_length": len(text),
                "cached_at": datetime.now().isoformat()
            }
            
            # Cache the embedding
            self.redis_client.setex(
                cache_key, 
                self.ttl, 
                json.dumps(embedding_data)
            )
            
            # Cache metadata
            self.redis_client.hset(f"{cache_key}:meta", mapping={
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "hits": 1,
                "created_at": datetime.now().isoformat()
            })
            self.redis_client.expire(f"{cache_key}:meta", self.ttl)
            
            logger.debug(f"💾 Cached embedding for text length {len(text)}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Cache storage failed: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            # Get all embedding cache keys
            pattern = f"{config.cache.cache_prefix}:embedding:*"
            keys = self.redis_client.keys(pattern)
            
            total_hits = 0
            total_entries = len([k for k in keys if not k.endswith(':meta')])
            
            # Sample some keys for hit statistics
            for key in keys[:min(100, len(keys))]:
                if key.endswith(':meta'):
                    hits = self.redis_client.hget(key, 'hits')
                    if hits:
                        total_hits += int(hits)
            
            return {
                "enabled": True,
                "total_entries": total_entries,
                "total_hits": total_hits,
                "hit_rate": total_hits / max(total_entries, 1),
                "memory_usage": self.redis_client.memory_usage_total() if hasattr(self.redis_client, 'memory_usage_total') else 'unknown'
            }
            
        except Exception as e:
            return {"enabled": True, "error": str(e)}

class ModelRouter:
    """Intelligent model selection for cost optimization"""
    
    def __init__(self):
        self.simple_model = config.embedding.completion_model_simple
        self.complex_model = config.embedding.completion_model_complex
        self.threshold = config.embedding.simple_query_threshold
        
        # Model costs (per 1k tokens) - update these based on current pricing
        self.model_costs = {
            'gpt-3.5-turbo': 0.0015,
            'gpt-4o': 0.06,
            'gpt-4': 0.03
        }
        
        self.usage_stats = {
            'simple_model_uses': 0,
            'complex_model_uses': 0,
            'cost_savings': 0.0
        }
    
    def select_model(self, query: str, context_length: int = 0) -> str:
        """Select optimal model based on query complexity"""
        # Calculate complexity score
        complexity_score = self._calculate_query_complexity(query, context_length)
        
        if complexity_score < self.threshold:
            self.usage_stats['simple_model_uses'] += 1
            
            # Calculate estimated cost savings
            simple_cost = self.model_costs.get(self.simple_model, 0.0015)
            complex_cost = self.model_costs.get(self.complex_model, 0.06)
            estimated_tokens = len(query.split()) + context_length // 4  # Rough token estimate
            
            savings = (complex_cost - simple_cost) * (estimated_tokens / 1000)
            self.usage_stats['cost_savings'] += savings
            
            logger.debug(f"🔄 Using simple model (complexity: {complexity_score:.2f})")
            return self.simple_model
        else:
            self.usage_stats['complex_model_uses'] += 1
            logger.debug(f"🔄 Using complex model (complexity: {complexity_score:.2f})")
            return self.complex_model
    
    def _calculate_query_complexity(self, query: str, context_length: int) -> float:
        """Calculate query complexity score"""
        factors = {
            'length': min(len(query) / 200, 1.0),  # Normalized by 200 chars
            'question_words': len([w for w in query.lower().split() if w in ['how', 'why', 'what', 'where', 'when', 'which', 'analyze', 'compare', 'evaluate']]) / 10,
            'technical_terms': len([w for w in query.lower().split() if w in ['compliance', 'specification', 'regulation', 'analysis', 'optimization']]) / 5,
            'context_size': min(context_length / 1000, 1.0),  # Normalized by 1k chars
            'multi_part': query.count('?') + query.count(',') + query.count(';'),
        }
        
        # Weighted complexity score
        complexity = (
            factors['length'] * 0.2 +
            factors['question_words'] * 0.3 +
            factors['technical_terms'] * 0.2 +
            factors['context_size'] * 0.2 +
            min(factors['multi_part'] * 0.1, 0.1)
        )
        
        return min(complexity, 1.0)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get model usage statistics"""
        total_uses = self.usage_stats['simple_model_uses'] + self.usage_stats['complex_model_uses']
        
        return {
            'total_uses': total_uses,
            'simple_model_ratio': self.usage_stats['simple_model_uses'] / max(total_uses, 1),
            'complex_model_ratio': self.usage_stats['complex_model_uses'] / max(total_uses, 1),
            'estimated_cost_savings': self.usage_stats['cost_savings'],
            'simple_model': self.simple_model,
            'complex_model': self.complex_model
        }

class EmbeddingService:
    """Enhanced embedding service with caching and cost optimization"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.embedding.openai_api_key)
        self.model = config.embedding.embedding_model
        self.batch_size = config.embedding.batch_size
        
        # Initialize enhanced components
        self.semantic_cache = SemanticCache()
        self.model_router = ModelRouter()
        self.advanced_chunking = get_advanced_chunking()
        
        # Statistics tracking
        self.stats = {
            'embeddings_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'documents_processed': 0
        }
        
        logger.info(f"🚀 Enhanced embedding service initialized")
        logger.info(f"📊 Model: {self.model}")
        logger.info(f"💾 Cache enabled: {self.semantic_cache.enabled}")
    
    def create_job_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Convert job data into text suitable for embedding based on data type"""
        data_type = job_data.get('data_type', 'consumed')
        
        if data_type == 'consumed':
            return self._create_consumed_text_representation(job_data)
        elif data_type == 'estimate':
            return self._create_estimate_text_representation(job_data)
        elif data_type == 'schedule':
            return self._create_schedule_text_representation(job_data)
        else:
            logger.warning(f"Unknown data type: {data_type}")
            return f"Unknown data type: {data_type}"
    
    def process_job_data_with_chunking(self, job_data: Dict[str, Any]) -> List[ProcessedChunk]:
        """Process job data using advanced chunking"""
        start_time = time.time()
        
        try:
            # Create text representation
            text_content = self.create_job_text_representation(job_data)
            
            # Use advanced chunking
            chunks = self.advanced_chunking.chunk_construction_document(
                content=text_content,
                job_metadata=job_data,
                document_id=f"{job_data.get('company_id', 'unknown')}_{job_data.get('job_id', 'unknown')}_{job_data.get('data_type', 'unknown')}"
            )
            
            # Generate embeddings for chunks
            for chunk in chunks:
                embedding = self.generate_embedding(chunk.content)
                chunk.embedding = embedding
            
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['documents_processed'] += 1
            
            logger.debug(f"✅ Processed job data into {len(chunks)} chunks in {processing_time:.2f}s")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing job data with chunking: {e}")
            return []
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with caching"""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_embedding = self.semantic_cache.get_cached_embedding(text)
            if cached_embedding:
                self.stats['cache_hits'] += 1
                return cached_embedding
            
            self.stats['cache_misses'] += 1
            
            # Generate new embedding
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            self.semantic_cache.cache_embedding(text, embedding)
            
            # Update statistics
            self.stats['embeddings_generated'] += 1
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            
            logger.debug(f"✅ Generated embedding in {processing_time:.3f}s")
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches asynchronously"""
        if not texts:
            return []
        
        start_time = time.time()
        embeddings = []
        cache_hits = 0
        cache_misses = 0
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_number = i // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            logger.debug(f"Processing batch {batch_number}/{total_batches} ({len(batch)} texts)")
            
            # Check cache for each text in batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                cached = self.semantic_cache.get_cached_embedding(text)
                if cached:
                    batch_embeddings.append(cached)
                    cache_hits += 1
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
                    cache_misses += 1
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    response = self.client.embeddings.create(
                        input=uncached_texts,
                        model=self.model
                    )
                    
                    new_embeddings = [item.embedding for item in response.data]
                    
                    # Cache new embeddings and fill in placeholders
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        self.semantic_cache.cache_embedding(uncached_texts[uncached_indices.index(idx)], embedding)
                        batch_embeddings[idx] = embedding
                    
                    self.stats['embeddings_generated'] += len(uncached_texts)
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_number}: {e}")
                    # Fill failed embeddings with empty lists
                    for idx in uncached_indices:
                        if batch_embeddings[idx] is None:
                            batch_embeddings[idx] = []
            
            embeddings.extend(batch_embeddings)
            
            # Add delay to respect rate limits
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['cache_hits'] += cache_hits
        self.stats['cache_misses'] += cache_misses
        self.stats['total_processing_time'] += processing_time
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings in {processing_time:.2f}s "
                   f"(cache hits: {cache_hits}, misses: {cache_misses})")
        
        return embeddings
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for batch embedding generation"""
        return asyncio.run(self.generate_embeddings_batch_async(texts))
    
    # Enhanced text representation methods with better structure and content
    
    def _create_consumed_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create enhanced text representation for consumed cost data"""
        try:
            entries = job_data.get('entries', [])
            job_name = "Unknown Job"
            
            # Try to extract job name from first entry
            if entries and len(entries) > 0:
                job_name = entries[0].get('job', 'Unknown Job')
            
            # Enhanced text representation with structured sections
            text_parts = [
                "CONSTRUCTION PROJECT - ACTUAL CONSUMED COSTS",
                f"Project: {job_name}",
                f"Company: {job_data.get('company_id', 'Unknown')}",
                f"Last Updated: {job_data.get('lastUpdated', 'Unknown')}",
                f"Total Cost Entries: {len(entries)}",
                ""  # Empty line for section separation
            ]
            
            # Analyze and categorize entries
            categories = {}
            total_cost = 0.0
            cost_codes = set()
            
            for entry in entries:
                cost_code = entry.get('costCode', 'Unknown')
                amount_str = entry.get('amount', '0')
                
                cost_codes.add(cost_code)
                
                # Parse amount
                try:
                    amount = float(amount_str) if amount_str else 0.0
                    total_cost += amount
                except (ValueError, TypeError):
                    amount = 0.0
                
                # Enhanced categorization
                category = self.categorize_cost_code(cost_code)
                
                if category not in categories:
                    categories[category] = {
                        'total': 0.0,
                        'items': [],
                        'codes': set()
                    }
                
                categories[category]['total'] += amount
                categories[category]['codes'].add(cost_code)
                categories[category]['items'].append({
                    'code': cost_code,
                    'amount': amount,
                    'description': cost_code.split(' ', 1)[-1] if ' ' in cost_code else cost_code
                })
            
            # Add financial summary
            text_parts.extend([
                "FINANCIAL SUMMARY:",
                f"Total Project Cost: ${total_cost:,.2f}",
                f"Number of Cost Categories: {len(categories)}",
                f"Unique Cost Codes: {len(cost_codes)}",
                ""
            ])
            
            # Add detailed category breakdowns
            for category, category_data in sorted(categories.items(), key=lambda x: x[1]['total'], reverse=True):
                category_total = category_data['total']
                percentage = (category_total / total_cost * 100) if total_cost > 0 else 0
                
                text_parts.extend([
                    f"{category.upper()} COSTS: ${category_total:,.2f} ({percentage:.1f}% of total)",
                    f"Cost Codes in {category}: {', '.join(sorted(category_data['codes']))}"
                ])
                
                # Add top items in category
                top_items = sorted(category_data['items'], key=lambda x: x['amount'], reverse=True)[:5]
                for item in top_items:
                    if item['amount'] > 0:
                        text_parts.append(f"  • {item['code']}: ${item['amount']:,.2f}")
                
                text_parts.append("")  # Section separator
            
            # Add construction context
            text_parts.extend([
                "CONSTRUCTION PROJECT CONTEXT:",
                f"This consumed cost data represents actual expenses incurred for {job_name}.",
                f"The data includes {len(entries)} individual cost entries across {len(categories)} major categories.",
                f"Primary cost areas: {', '.join(sorted(categories.keys()))}",
                f"Project managed by company ID: {job_data.get('company_id', 'Unknown')}"
            ])
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error creating consumed text representation: {e}")
            return f"Consumed data processing error for job {job_data.get('job_id', 'unknown')}: {str(e)}"
    
    def _create_estimate_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create enhanced text representation for estimate data"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            
            text_parts = [
                "CONSTRUCTION PROJECT - COST ESTIMATES & BUDGET",
                f"Project: {job_name}",
                f"Client: {job_data.get('client_name', 'Unknown')}",
                f"Location: {job_data.get('site_location', 'Unknown')}",
                f"Project Description: {job_data.get('project_description', 'N/A')}",
                f"Total Estimate Rows: {len(entries)}",
                ""
            ]
            
            # Analyze estimate data with enhanced categorization
            areas = {}
            task_scopes = {}
            total_estimated = 0.0
            total_budgeted = 0.0
            row_types = {'estimate': 0, 'allowance': 0, 'other': 0}
            
            for entry in entries:
                area = entry.get('area', 'General')
                task_scope = entry.get('taskScope', 'Unknown')
                cost_code = entry.get('costCode', 'Unknown')
                description = entry.get('description', '')
                units = entry.get('units', '')
                qty = float(entry.get('qty', 0))
                rate = float(entry.get('rate', 0))
                total = float(entry.get('total', 0))
                budgeted_rate = float(entry.get('budgetedRate', 0))
                budgeted_total = float(entry.get('budgetedTotal', 0))
                row_type = entry.get('rowType', 'estimate')
                
                total_estimated += total
                total_budgeted += budgeted_total
                row_types[row_type] = row_types.get(row_type, 0) + 1
                
                # Group by area
                if area not in areas:
                    areas[area] = {
                        'estimated': 0.0,
                        'budgeted': 0.0,
                        'tasks': [],
                        'task_count': 0
                    }
                
                areas[area]['estimated'] += total
                areas[area]['budgeted'] += budgeted_total
                areas[area]['task_count'] += 1
                
                # Track significant tasks
                if total > 1000 or budgeted_total > 1000:  # Only track significant items
                    areas[area]['tasks'].append({
                        'task_scope': task_scope,
                        'cost_code': cost_code,
                        'description': description,
                        'qty': qty,
                        'units': units,
                        'rate': rate,
                        'total': total,
                        'budgeted_total': budgeted_total,
                        'type': row_type
                    })
                
                # Track task scopes globally
                task_scopes[task_scope] = task_scopes.get(task_scope, 0) + total
            
            # Add financial summary
            variance = total_budgeted - total_estimated
            variance_percent = (variance / total_estimated * 100) if total_estimated > 0 else 0
            
            text_parts.extend([
                "ESTIMATE FINANCIAL SUMMARY:",
                f"Total Estimated Cost: ${total_estimated:,.2f}",
                f"Total Budgeted Cost: ${total_budgeted:,.2f}",
                f"Budget Variance: ${variance:,.2f} ({variance_percent:+.1f}%)",
                f"Estimate Rows: {row_types.get('estimate', 0)}, Allowances: {row_types.get('allowance', 0)}",
                ""
            ])
            
            # Add area-by-area breakdown
            text_parts.append("COST BREAKDOWN BY CONSTRUCTION AREA:")
            for area, area_data in sorted(areas.items(), key=lambda x: x[1]['estimated'], reverse=True):
                est_pct = (area_data['estimated'] / total_estimated * 100) if total_estimated > 0 else 0
                
                text_parts.extend([
                    f"{area} - ${area_data['estimated']:,.2f} estimated (${area_data['budgeted']:,.2f} budgeted) - {est_pct:.1f}%",
                    f"  Tasks in {area}: {area_data['task_count']}"
                ])
                
                # Add significant tasks
                significant_tasks = sorted(area_data['tasks'], key=lambda x: x['total'], reverse=True)[:3]
                for task in significant_tasks:
                    text_parts.append(f"    • {task['task_scope']}: {task['description']} - {task['qty']} {task['units']} @ ${task['rate']}/unit = ${task['total']:,.2f}")
                
                text_parts.append("")
            
            # Add construction context
            text_parts.extend([
                "PROJECT ESTIMATE CONTEXT:",
                f"This estimate covers {len(areas)} construction areas with {len(entries)} detailed line items.",
                f"Top task scopes by value: {', '.join([k for k, v in sorted(task_scopes.items(), key=lambda x: x[1], reverse=True)[:5]])}",
                f"Budget status: {'Over budget' if variance < 0 else 'Under budget' if variance > 0 else 'On budget'} by ${abs(variance):,.2f}",
                f"Project complexity: {'High' if len(areas) > 10 else 'Medium' if len(areas) > 5 else 'Standard'} ({len(areas)} areas, {len(entries)} line items)"
            ])
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error creating estimate text representation: {e}")
            return f"Estimate data processing error for {job_data.get('job_name', 'unknown')}: {str(e)}"
    
    def _create_schedule_text_representation(self, job_data: Dict[str, Any]) -> str:
        """Create enhanced text representation for schedule data"""
        try:
            entries = job_data.get('entries', [])
            job_name = job_data.get('job_name', 'Unknown Job')
            
            text_parts = [
                "CONSTRUCTION PROJECT - SCHEDULE & TIMELINE",
                f"Project: {job_name}",
                f"Client: {job_data.get('client_name', 'Unknown')}",
                f"Location: {job_data.get('site_location', 'Unknown')}",
                f"Schedule Last Updated: {job_data.get('schedule_last_updated', 'Unknown')}",
                f"Total Tasks: {len(entries)}",
                ""
            ]
            
            # Enhanced schedule analysis
            valid_tasks = [entry for entry in entries if entry.get('task', '').strip()]
            main_tasks = []
            subtasks = []
            
            project_start_date = None
            project_end_date = None
            total_hours = 0.0
            total_consumed = 0.0
            task_statuses = {'not_started': 0, 'in_progress': 0, 'completed': 0}
            task_types = {}
            resources = set()
            critical_path_tasks = []
            
            for entry in valid_tasks:
                task_name = entry.get('task', '').strip()
                if not task_name:
                    continue
                    
                is_main_task = entry.get('isMainTask', False)
                hours = float(entry.get('hours', 0))
                consumed = float(entry.get('consumed', 0))
                progress = float(entry.get('percentageComplete', 0))
                task_type = entry.get('taskType', 'labour')
                
                total_hours += hours
                total_consumed += consumed
                
                # Classify task status
                if progress == 0:
                    task_statuses['not_started'] += 1
                elif progress >= 100:
                    task_statuses['completed'] += 1
                else:
                    task_statuses['in_progress'] += 1
                
                # Track task types
                task_types[task_type] = task_types.get(task_type, 0) + 1
                
                # Track project dates
                start_date = self._parse_schedule_date(entry.get('startDate'))
                end_date = self._parse_schedule_date(entry.get('endDate'))
                
                if start_date:
                    if not project_start_date or start_date < project_start_date:
                        project_start_date = start_date
                
                if end_date:
                    if not project_end_date or end_date > project_end_date:
                        project_end_date = end_date
                
                # Collect resources
                task_resources = entry.get('resources', {})
                if isinstance(task_resources, dict):
                    resources.update(task_resources.keys())
                
                # Identify critical tasks (high hours or behind schedule)
                if hours > 0:
                    efficiency = consumed / hours if hours > 0 else 0
                    is_critical = (
                        hours > 40 or  # Large tasks
                        efficiency > 1.2 or  # Over-consumed
                        (progress > 0 and progress < 50 and consumed > hours * 0.8)  # Behind but consuming
                    )
                    
                    task_info = {
                        'name': task_name,
                        'hours': hours,
                        'consumed': consumed,
                        'progress': progress,
                        'efficiency': efficiency,
                        'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'TBD',
                        'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'TBD',
                        'task_type': task_type,
                        'resources': list(task_resources.keys()) if task_resources else [],
                        'is_main': is_main_task,
                        'is_critical': is_critical
                    }
                    
                    if is_main_task:
                        main_tasks.append(task_info)
                    else:
                        subtasks.append(task_info)
                    
                    if is_critical:
                        critical_path_tasks.append(task_info)
            
            # Project timeline summary
            duration_days = (project_end_date - project_start_date).days if project_start_date and project_end_date else 0
            completion_rate = task_statuses['completed'] / max(len(valid_tasks), 1) * 100
            efficiency_overall = total_consumed / total_hours if total_hours > 0 else 0
            
            text_parts.extend([
                "SCHEDULE SUMMARY:",
                f"Project Duration: {project_start_date.strftime('%Y-%m-%d') if project_start_date else 'TBD'} to {project_end_date.strftime('%Y-%m-%d') if project_end_date else 'TBD'} ({duration_days} days)",
                f"Total Planned Hours: {total_hours:,.1f} | Consumed Hours: {total_consumed:,.1f} | Efficiency: {efficiency_overall:.1%}",
                f"Task Progress: {task_statuses['completed']} completed, {task_statuses['in_progress']} in progress, {task_statuses['not_started']} not started ({completion_rate:.1f}% complete)",
                f"Task Types: {', '.join([f'{k}: {v}' for k, v in sorted(task_types.items())])}",
                f"Resources Involved: {len(resources)} ({', '.join(sorted(list(resources)[:8]))}{'...' if len(resources) > 8 else ''})",
                ""
            ])
            
            # Main tasks breakdown
            if main_tasks:
                text_parts.append("MAJOR CONSTRUCTION PHASES:")
                for task in sorted(main_tasks, key=lambda x: x['hours'], reverse=True):
                    status_indicator = "✓" if task['progress'] >= 100 else "⚠" if task['efficiency'] > 1.1 else "→"
                    text_parts.append(f"{status_indicator} {task['name']}: {task['start_date']} to {task['end_date']} | {task['hours']:.1f}h planned, {task['consumed']:.1f}h consumed ({task['progress']:.0f}% complete)")
                text_parts.append("")
            
            # Critical path and problem areas
            if critical_path_tasks:
                text_parts.append("CRITICAL TASKS & ATTENTION AREAS:")
                for task in sorted(critical_path_tasks, key=lambda x: x['efficiency'], reverse=True)[:5]:
                    reason = []
                    if task['efficiency'] > 1.2:
                        reason.append("over-consuming")
                    if task['hours'] > 40:
                        reason.append("high-impact")
                    if task['progress'] > 0 and task['progress'] < 50:
                        reason.append("potentially delayed")
                    
                    text_parts.append(f"⚠ {task['name']}: {task['consumed']:.1f}/{task['hours']:.1f}h ({task['efficiency']:.1%} efficiency) - {', '.join(reason)}")
                text_parts.append("")
            
            # Construction context
            text_parts.extend([
                "PROJECT SCHEDULE CONTEXT:",
                f"This schedule encompasses {len(main_tasks)} major phases and {len(subtasks)} detailed tasks.",
                f"Current project status: {completion_rate:.1f}% complete with {efficiency_overall:.1%} overall efficiency.",
                f"Schedule performance: {'Ahead of schedule' if efficiency_overall < 0.9 else 'Behind schedule' if efficiency_overall > 1.1 else 'On schedule'}",
                f"Resource utilization across {len(resources)} different resource types.",
                f"Critical attention needed for {len(critical_path_tasks)} tasks requiring management focus."
            ])
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error creating schedule text representation: {e}")
            return f"Schedule data processing error for {job_data.get('job_name', 'unknown')}: {str(e)}"
    
    def _parse_schedule_date(self, date_value) -> Optional[datetime]:
        """Parse various date formats from schedule data"""
        if not date_value:
            return None
            
        try:
            # Handle string dates (YYYY-MM-DD format)
            if isinstance(date_value, str):
                return datetime.strptime(date_value, '%Y-%m-%d')
            
            # Handle Firebase Timestamp objects
            if hasattr(date_value, 'to_dict'):
                # Firestore Timestamp
                return date_value.to_datetime()
            
            # Handle datetime objects
            if isinstance(date_value, datetime):
                return date_value
                
        except Exception as e:
            logger.debug(f"Error parsing date {date_value}: {e}")
        
        return None
    
    def categorize_cost_code(self, cost_code: str) -> str:
        """Enhanced cost code categorization with priority logic"""
        if not cost_code:
            return "Unknown"
            
        code_lower = cost_code.lower()
        
        # Priority 1: Check for "subcontractor" in the name first
        if 'subcontractor' in code_lower:
            return "Subcontractors"
        
        # Priority 2: Check suffix patterns in the code
        # Look for patterns like "503S", "110O", "414M", "108L"
        import re
        suffix_match = re.search(r'\d+([SMLO])\b', cost_code)
        if suffix_match:
            suffix = suffix_match.group(1).upper()
            if suffix == 'S':
                return "Subcontractors"
            elif suffix == 'M':
                return "Materials"
            elif suffix == 'L':
                return "Labor"
            elif suffix == 'O':
                return "Other/Overhead"
        
        # Priority 3: Enhanced keyword matching
        if any(word in code_lower for word in ['material', 'materials', 'supply', 'supplies', 'equipment']):
            return "Materials"
        elif any(word in code_lower for word in ['labor', 'labour', 'crew', 'worker', 'manpower']):
            return "Labor"
        elif any(word in code_lower for word in ['permit', 'fee', 'overhead', 'management', 'admin', 'office']):
            return "Other/Overhead"
        elif any(word in code_lower for word in ['electrical', 'plumbing', 'hvac', 'mechanical']):
            return "Trade Specialties"
        
        # Default fallback
        return "General"
    
    def create_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced metadata for the document based on data type"""
        data_type = job_data.get('data_type', 'consumed')
        
        base_metadata = {
            'job_name': str(job_data.get('job_name', 'Unknown')),
            'company_id': str(job_data.get('company_id', '')),
            'job_id': str(job_data.get('job_id', '')),
            'document_type': f'job_{data_type}_data',
            'data_type': data_type,
            'processed_at': datetime.now().isoformat(),
            'processing_version': '2.0'  # Track processing version for migrations
        }
        
        if data_type == 'consumed':
            return {**base_metadata, **self._create_consumed_metadata(job_data)}
        elif data_type == 'estimate':
            return {**base_metadata, **self._create_estimate_metadata(job_data)}
        elif data_type == 'schedule':
            return {**base_metadata, **self._create_schedule_metadata(job_data)}
        else:
            return {**base_metadata, 'error': f'Unknown data type: {data_type}'}
    
    def _create_consumed_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced metadata for consumed cost data"""
        entries = job_data.get('entries', [])
        
        # Extract job info
        job_name = entries[0].get('job', 'Unknown') if entries else 'Unknown'
        
        # Enhanced analysis
        category_totals = {}
        total_cost = 0.0
        cost_codes = set()
        high_value_items = 0
        
        for entry in entries:
            cost_code = entry.get('costCode', '')
            amount_str = entry.get('amount', '0')
            
            cost_codes.add(cost_code)
            
            try:
                amount = float(amount_str) if amount_str else 0.0
                total_cost += amount
                
                if amount > 5000:  # Track high-value items
                    high_value_items += 1
                
                category = self.categorize_cost_code(cost_code)
                category_totals[category] = category_totals.get(category, 0.0) + amount
                
            except (ValueError, TypeError):
                continue
        
        # Convert data for ChromaDB/Qdrant compatibility
        categories_str = ", ".join(category_totals.keys()) if category_totals else ""
        cost_codes_str = ", ".join(list(cost_codes)[:15])  # Limit to first 15 codes
        
        # Handle lastUpdated datetime conversion
        last_updated = job_data.get('lastUpdated', '')
        if hasattr(last_updated, 'isoformat'):
            last_updated_str = last_updated.isoformat()
        else:
            last_updated_str = str(last_updated) if last_updated else ''
        
        # Enhanced metadata
        metadata = {
            'last_updated': last_updated_str,
            'total_entries': int(len(entries)),
            'total_cost': float(total_cost),
            'categories': categories_str,
            'cost_codes': cost_codes_str,
            'high_value_items': int(high_value_items),
            'primary_category': max(category_totals.items(), key=lambda x: x[1])[0] if category_totals else 'Unknown',
            'category_count': int(len(category_totals)),
            'avg_cost_per_entry': float(total_cost / len(entries)) if entries else 0.0
        }
        
        # Add individual category totals as separate metadata fields
        for category, total in category_totals.items():
            field_name = f"category_{category.lower().replace('/', '_').replace(' ', '_').replace('-', '_')}_total"
            metadata[field_name] = float(total)
        
        return metadata
    
    def _create_estimate_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced metadata for estimate data"""
        entries = job_data.get('entries', [])
        job_name = job_data.get('job_name', 'Unknown Job')
        
        # Enhanced analysis
        total_estimated = 0.0
        total_budgeted = 0.0
        areas = set()
        task_scopes = set()
        estimate_count = 0
        allowance_count = 0
        high_value_items = 0
        
        for entry in entries:
            estimated = float(entry.get('total', 0))
            budgeted = float(entry.get('budgetedTotal', 0))
            
            total_estimated += estimated
            total_budgeted += budgeted
            
            if estimated > 5000 or budgeted > 5000:
                high_value_items += 1
            
            areas.add(entry.get('area', 'General'))
            task_scopes.add(entry.get('taskScope', 'Unknown'))
            
            row_type = entry.get('rowType', 'estimate')
            if row_type == 'estimate':
                estimate_count += 1
            elif row_type == 'allowance':
                allowance_count += 1
        
        budget_variance = total_budgeted - total_estimated
        budget_variance_percent = (budget_variance / total_estimated * 100) if total_estimated > 0 else 0
        
        return {
            'client_name': str(job_data.get('client_name', '')),
            'site_location': str(job_data.get('site_location', '')),
            'project_description': str(job_data.get('project_description', ''))[:500],  # Limit length
            'total_entries': int(len(entries)),
            'total_estimated_cost': float(total_estimated),
            'total_budgeted_cost': float(total_budgeted),
            'budget_variance': float(budget_variance),
            'budget_variance_percent': float(budget_variance_percent),
            'areas_count': int(len(areas)),
            'areas': ", ".join(sorted(list(areas)[:10])),  # Limit to 10 areas
            'task_scopes': ", ".join(sorted(list(task_scopes)[:10])),
            'estimate_rows': int(estimate_count),
            'allowance_rows': int(allowance_count),
            'high_value_items': int(high_value_items),
            'avg_estimated_per_item': float(total_estimated / len(entries)) if entries else 0.0,
            'project_size': 'Large' if total_estimated > 500000 else 'Medium' if total_estimated > 100000 else 'Small'
        }
    
    def _create_schedule_metadata(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced metadata for schedule data"""
        entries = job_data.get('entries', [])
        job_name = job_data.get('job_name', 'Unknown Job')
        
        # Enhanced schedule analysis
        valid_tasks = [entry for entry in entries if entry.get('task', '').strip()]
        
        total_hours = 0.0
        total_consumed = 0.0
        main_task_count = 0
        subtask_count = 0
        completed_tasks = 0
        in_progress_tasks = 0
        not_started_tasks = 0
        task_types = set()
        resources = set()
        critical_tasks = 0
        
        project_start_date = None
        project_end_date = None
        
        for entry in valid_tasks:
            hours = float(entry.get('hours', 0))
            consumed = float(entry.get('consumed', 0))
            progress = float(entry.get('percentageComplete', 0))
            
            total_hours += hours
            total_consumed += consumed
            
            if entry.get('isMainTask', False):
                main_task_count += 1
            else:
                subtask_count += 1
            
            # Task status classification
            if progress >= 100:
                completed_tasks += 1
            elif progress > 0:
                in_progress_tasks += 1
            else:
                not_started_tasks += 1
            
            # Track efficiency issues
            if hours > 0 and consumed / hours > 1.2:  # Over-consuming
                critical_tasks += 1
            
            task_types.add(entry.get('taskType', 'labour'))
            
            # Collect resources
            task_resources = entry.get('resources', {})
            if isinstance(task_resources, dict):
                resources.update(task_resources.keys())
            
            # Track project date range
            start_date = self._parse_schedule_date(entry.get('startDate'))
            end_date = self._parse_schedule_date(entry.get('endDate'))
            
            if start_date:
                if not project_start_date or start_date < project_start_date:
                    project_start_date = start_date
            
            if end_date:
                if not project_end_date or end_date > project_end_date:
                    project_end_date = end_date
        
        # Calculate project metrics
        completion_rate = completed_tasks / max(len(valid_tasks), 1)
        efficiency_rate = total_consumed / total_hours if total_hours > 0 else 0
        duration_days = (project_end_date - project_start_date).days if project_start_date and project_end_date else 0
        
        return {
            'client_name': str(job_data.get('client_name', '')),
            'site_location': str(job_data.get('site_location', '')),
            'total_tasks': int(len(valid_tasks)),
            'main_tasks': int(main_task_count),
            'subtasks': int(subtask_count),
            'completed_tasks': int(completed_tasks),
            'in_progress_tasks': int(in_progress_tasks),
            'not_started_tasks': int(not_started_tasks),
            'completion_rate': float(completion_rate),
            'total_planned_hours': float(total_hours),
            'total_consumed_hours': float(total_consumed),
            'efficiency_rate': float(efficiency_rate),
            'project_start_date': project_start_date.strftime('%Y-%m-%d') if project_start_date else '',
            'project_end_date': project_end_date.strftime('%Y-%m-%d') if project_end_date else '',
            'project_duration_days': int(duration_days),
            'task_types': ", ".join(sorted(list(task_types))),
            'resources': ", ".join(sorted(list(resources)[:15])),  # Limit to 15 resources
            'resource_count': int(len(resources)),
            'critical_tasks': int(critical_tasks),
            'schedule_health': 'Good' if efficiency_rate < 1.1 and completion_rate > 0.3 else 'Attention Needed',
            'project_phase': 'Planning' if completion_rate < 0.1 else 'In Progress' if completion_rate < 0.9 else 'Completion'
        }
    
    def test_embedding(self) -> bool:
        """Test the embedding service with a simple text"""
        try:
            test_text = "Test construction job with electrical work costing $1000"
            embedding = self.generate_embedding(test_text)
            return len(embedding) > 0 and len(embedding) == 1536  # OpenAI embedding dimension
        except Exception as e:
            logger.error(f"Embedding test failed: {e}")
            return False
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        cache_stats = self.semantic_cache.get_cache_stats()
        model_stats = self.model_router.get_usage_stats()
        
        # Calculate hit rate
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / max(total_requests, 1)
        
        return {
            'embeddings_generated': self.stats['embeddings_generated'],
            'documents_processed': self.stats['documents_processed'],
            'total_processing_time': self.stats['total_processing_time'],
            'cache_hit_rate': hit_rate,
            'cache_stats': cache_stats,
            'model_routing_stats': model_stats,
            'advanced_chunking_enabled': self.advanced_chunking.use_llamaindex,
            'current_model': self.model,
            'service_uptime': time.time() - getattr(self, '_start_time', time.time())
        }

# Global enhanced embedding service instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        _embedding_service._start_time = time.time()
    return _embedding_service