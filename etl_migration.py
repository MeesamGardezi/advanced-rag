"""
ETL Migration Script: Firestore → PostgreSQL + ChromaDB
Migrates a single job's data with transaction support and verification
COMPLETE VERSION: Inserts into BOTH PostgreSQL AND ChromaDB with embeddings
"""

import argparse
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging

# Import existing Firebase/Firestore functions
from database import (
    initialize_firebase,
    fetch_job_complete_data,
    extract_estimate_data,
    extract_flooring_estimate_data,
    extract_schedule_data,
    get_chroma_collection
)
from embedding_service import EmbeddingService

# Import PostgreSQL components
from db_connection import get_db, session_scope, get_job_data_summary, verify_job_exists

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETLMigration:
    """Handles ETL migration from Firestore to PostgreSQL + ChromaDB"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.chroma_collection = None
        self.stats = {
            'job_migrated': False,
            'estimates_migrated': 0,
            'flooring_estimates_migrated': 0,
            'schedule_items_migrated': 0,
            'consumed_items_migrated': 0,
            'embeddings_created': 0,
            'errors': []
        }
    
    def _parse_datetime(self, date_value) -> Optional[datetime]:
        """Parse various date formats from Firestore"""
        if not date_value:
            return None
        
        try:
            # Handle Firestore Timestamp objects
            if hasattr(date_value, 'to_datetime'):
                return date_value.to_datetime()
            
            # Handle datetime objects
            if isinstance(date_value, datetime):
                return date_value
            
            # Handle string dates
            if isinstance(date_value, str):
                for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"⚠️  Failed to parse date {date_value}: {e}")
        
        return None
    
    def _delete_job_data(self, job_id: str):
        """
        Delete all existing data for a job from both PostgreSQL and ChromaDB
        
        Args:
            job_id: Job identifier to delete
        """
        logger.info(f"🗑️  Deleting existing data for job: {job_id}")
        
        try:
            # Delete from PostgreSQL (cascade will handle related records)
            with session_scope() as cursor:
                cursor.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
                deleted_count = cursor.rowcount
                logger.info(f"   ✅ Deleted {deleted_count} job record from PostgreSQL (cascaded to related tables)")
            
            # Delete from ChromaDB
            if self.chroma_collection is None:
                self.chroma_collection = get_chroma_collection()
            
            # Get all document IDs for this job
            try:
                # Query with proper where filter
                results = self.chroma_collection.get(
                    where={"job_id": {"$eq": job_id}},
                    include=[]  # We only need IDs
                )
                
                doc_ids = results['ids'] if results['ids'] else []
                
                if doc_ids:
                    self.chroma_collection.delete(ids=doc_ids)
                    logger.info(f"   ✅ Deleted {len(doc_ids)} embeddings from ChromaDB")
                else:
                    logger.info(f"   ℹ️  No embeddings found in ChromaDB for this job")
                    
            except Exception as e:
                logger.warning(f"   ⚠️  ChromaDB deletion warning: {e}")
            
            logger.info(f"✅ Cleanup complete for job {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Error deleting job data: {e}")
            raise
    
    def _insert_job(self, cursor, job_data: Dict[str, Any]) -> bool:
        """Insert job using raw SQL"""
        job_id = job_data['job_id']
        company_id = job_data['company_id']
        
        city = job_data.get('siteCity', '')
        state = job_data.get('siteState', '')
        site_location = f"{city}, {state}".strip(', ')
        
        sql = """
            INSERT INTO jobs (
                id, company_id, name, client_name, site_city, site_state,
                site_location, project_description, status, estimate_type,
                created_date, last_updated, schedule_last_updated
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            job_id,
            company_id,
            job_data.get('projectTitle', 'Unknown Job'),
            job_data.get('clientName', ''),
            city,
            state,
            site_location,
            job_data.get('projectDescription', ''),
            job_data.get('status', 'active'),
            job_data.get('estimateType', 'general'),
            self._parse_datetime(job_data.get('createdDate')),
            datetime.now(timezone.utc),
            self._parse_datetime(job_data.get('scheduleLastUpdated'))
        )
        
        cursor.execute(sql, values)
        return True
    
    def _insert_estimate(self, cursor, row_data: Dict[str, Any], job_id: str, job_context: Dict[str, Any]) -> str:
        """
        Insert estimate row into PostgreSQL and ChromaDB
        
        Returns:
            estimate_id for the inserted row
        """
        row_num = row_data.get('row_number', 0)
        
        # Parse numerical values
        qty = float(row_data.get('qty', 0))
        rate = float(row_data.get('rate', 0))
        total = float(row_data.get('total', 0))
        budgeted_rate = float(row_data.get('budgetedRate', 0))
        budgeted_total = float(row_data.get('budgetedTotal', 0))
        
        variance = total - budgeted_total
        variance_pct = (variance / budgeted_total * 100) if budgeted_total != 0 else 0
        
        materials = row_data.get('materials', [])
        has_materials = len(materials) > 0
        material_count = len(materials)
        
        estimate_id = f"{job_id}_est_row_{row_num}"
        
        # Insert into PostgreSQL
        sql = """
            INSERT INTO estimates (
                id, job_id, row_number, area, task_scope, cost_code, row_type,
                description, notes_remarks, units, qty, rate, total,
                budgeted_rate, budgeted_total, variance, variance_pct,
                materials, has_materials, material_count, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Convert materials to JSON string if needed
        import json
        materials_json = json.dumps(materials) if materials else None
        
        values = (
            estimate_id,
            job_id,
            row_num,
            row_data.get('area', ''),
            row_data.get('taskScope', ''),
            row_data.get('costCode', ''),
            row_data.get('rowType', 'estimate'),
            row_data.get('description', ''),
            row_data.get('notesRemarks', ''),
            row_data.get('units', ''),
            qty, rate, total,
            budgeted_rate, budgeted_total,
            variance, variance_pct,
            materials_json,
            has_materials,
            material_count,
            datetime.now(timezone.utc)
        )
        
        cursor.execute(sql, values)
        
        # Insert into ChromaDB
        self._insert_estimate_embedding(estimate_id, row_data, job_context)
        
        return estimate_id
    
    def _insert_estimate_embedding(self, estimate_id: str, row_data: Dict[str, Any], job_context: Dict[str, Any]):
        """Create embedding and insert into ChromaDB for estimate row"""
        try:
            # Create text representation
            text = self.embedding_service.create_estimate_row_text(row_data, job_context)
            
            # Create metadata
            metadata = self.embedding_service.create_estimate_row_metadata(row_data, job_context)
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(text)
            
            # Insert into ChromaDB
            if self.chroma_collection is None:
                self.chroma_collection = get_chroma_collection()
            
            self.chroma_collection.add(
                ids=[estimate_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            
            self.stats['embeddings_created'] += 1
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create embedding for {estimate_id}: {e}")
            # Don't fail the whole migration for embedding errors
    
    def _insert_flooring_estimate(self, cursor, row_data: Dict[str, Any], job_id: str, job_context: Dict[str, Any]) -> str:
        """
        Insert flooring estimate row into PostgreSQL and ChromaDB
        
        Returns:
            flooring_id for the inserted row
        """
        row_num = row_data.get('row_number', 0)
        
        measured_qty = float(row_data.get('measuredQty', 0))
        supplier_qty = float(row_data.get('supplierQty', 0))
        waste_factor = float(row_data.get('wasteFactor', 0))
        qty_including_waste = float(row_data.get('qtyIncludingWaste', 0))
        unit_price = float(row_data.get('unitPrice', 0))
        cost_price = float(row_data.get('costPrice', 0))
        tax_freight = float(row_data.get('taxFreight', 0))
        total_cost = float(row_data.get('totalCost', 0))
        sale_price = float(row_data.get('salePrice', 0))
        
        profit = sale_price - total_cost
        margin_pct = (profit / sale_price * 100) if sale_price > 0 else 0
        
        flooring_id = f"{job_id}_flooring_row_{row_num}"
        
        # Insert into PostgreSQL
        sql = """
            INSERT INTO flooring_estimates (
                id, job_id, row_number, floor_type_id, vendor, item_material_name,
                brand, unit, measured_qty, supplier_qty, waste_factor,
                qty_including_waste, unit_price, cost_price, tax_freight,
                total_cost, sale_price, profit, margin_pct, notes_remarks, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            flooring_id,
            job_id,
            row_num,
            row_data.get('floorTypeId', ''),
            row_data.get('vendor', ''),
            row_data.get('itemMaterialName', ''),
            row_data.get('brand', ''),
            row_data.get('unit', ''),
            measured_qty, supplier_qty, waste_factor,
            qty_including_waste, unit_price, cost_price, tax_freight,
            total_cost, sale_price, profit, margin_pct,
            row_data.get('notesRemarks', ''),
            datetime.now(timezone.utc)
        )
        
        cursor.execute(sql, values)
        
        # Insert into ChromaDB
        self._insert_flooring_embedding(flooring_id, row_data, job_context)
        
        return flooring_id
    
    def _insert_flooring_embedding(self, flooring_id: str, row_data: Dict[str, Any], job_context: Dict[str, Any]):
        """Create embedding and insert into ChromaDB for flooring estimate row"""
        try:
            # Create text representation
            text = self.embedding_service.create_flooring_estimate_row_text(row_data, job_context)
            
            # Create metadata
            metadata = self.embedding_service.create_flooring_estimate_row_metadata(row_data, job_context)
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(text)
            
            # Insert into ChromaDB
            if self.chroma_collection is None:
                self.chroma_collection = get_chroma_collection()
            
            self.chroma_collection.add(
                ids=[flooring_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            
            self.stats['embeddings_created'] += 1
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create embedding for {flooring_id}: {e}")
    
    def _insert_schedule_item(self, cursor, row_data: Dict[str, Any], job_id: str, job_context: Dict[str, Any]) -> str:
        """
        Insert schedule item into PostgreSQL and ChromaDB
        
        Returns:
            schedule_id for the inserted row
        """
        row_num = row_data.get('row_number', 0)
        
        hours = float(row_data.get('hours', 0))
        consumed = float(row_data.get('consumed', 0))
        percentage_complete = float(row_data.get('percentageComplete', 0))
        
        schedule_id = f"{job_id}_schedule_row_{row_num}"
        
        # Insert into PostgreSQL
        sql = """
            INSERT INTO schedule_items (
                id, job_id, row_number, task, is_main_task, task_type,
                hours, consumed, percentage_complete, start_date, end_date,
                resources, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        import json
        resources_json = json.dumps(row_data.get('resources', {}))
        
        values = (
            schedule_id,
            job_id,
            row_num,
            row_data.get('task', ''),
            row_data.get('isMainTask', False),
            row_data.get('taskType', 'labour'),
            hours, consumed, percentage_complete,
            self._parse_datetime(row_data.get('startDate')),
            self._parse_datetime(row_data.get('endDate')),
            resources_json,
            datetime.now(timezone.utc)
        )
        
        cursor.execute(sql, values)
        
        # Insert into ChromaDB
        self._insert_schedule_embedding(schedule_id, row_data, job_context)
        
        return schedule_id
    
    def _insert_schedule_embedding(self, schedule_id: str, row_data: Dict[str, Any], job_context: Dict[str, Any]):
        """Create embedding and insert into ChromaDB for schedule item"""
        try:
            # Create text representation
            task = row_data.get('task', '')
            task_type = row_data.get('taskType', 'labour')
            hours = float(row_data.get('hours', 0))
            consumed = float(row_data.get('consumed', 0))
            
            text = f"""SCHEDULE TASK
Job: {job_context.get('job_name', 'Unknown')}
Task: {task}
Type: {task_type}
Hours Planned: {hours:.1f}
Hours Consumed: {consumed:.1f}
Progress: {row_data.get('percentageComplete', 0):.1f}%"""
            
            # Create metadata
            metadata = {
                'job_name': str(job_context.get('job_name', 'Unknown')),
                'company_id': str(job_context.get('company_id', '')),
                'job_id': str(job_context.get('job_id', '')),
                'row_number': int(row_data.get('row_number', 0)),
                'document_type': 'schedule_item',
                'data_type': 'schedule',
                'granularity': 'row',
                'task': str(task)[:200],
                'task_type': str(task_type),
                'hours': float(hours)
            }
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(text)
            
            # Insert into ChromaDB
            if self.chroma_collection is None:
                self.chroma_collection = get_chroma_collection()
            
            self.chroma_collection.add(
                ids=[schedule_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            
            self.stats['embeddings_created'] += 1
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create embedding for {schedule_id}: {e}")
    
    def _insert_consumed_item(self, cursor, item_data: Dict[str, Any], job_id: str, job_context: Dict[str, Any], index: int) -> str:
        """
        Insert consumed item into PostgreSQL and ChromaDB
        
        Returns:
            consumed_id for the inserted row
        """
        cost_code = item_data.get('costCode', 'Unknown')
        amount_str = item_data.get('amount', '0')
        
        try:
            amount = float(amount_str) if amount_str else 0.0
        except (ValueError, TypeError):
            amount = 0.0
        
        category = self.embedding_service.categorize_cost_code(cost_code)
        
        consumed_id = f"{job_id}_consumed_{index}"
        
        # Insert into PostgreSQL
        sql = """
            INSERT INTO consumed_items (
                id, job_id, cost_code, amount, category, last_updated, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            consumed_id,
            job_id,
            cost_code,
            amount,
            category,
            datetime.now(timezone.utc),
            datetime.now(timezone.utc)
        )
        
        cursor.execute(sql, values)
        
        # Insert into ChromaDB
        self._insert_consumed_embedding(consumed_id, item_data, job_context, amount, category)
        
        return consumed_id
    
    def _insert_consumed_embedding(self, consumed_id: str, item_data: Dict[str, Any], job_context: Dict[str, Any], amount: float, category: str):
        """Create embedding and insert into ChromaDB for consumed item"""
        try:
            # Create text representation
            cost_code = item_data.get('costCode', 'Unknown')
            
            text = f"""CONSUMED COST ITEM
Job: {job_context.get('job_name', 'Unknown')}
Cost Code: {cost_code}
Category: {category}
Amount: ${amount:,.2f}"""
            
            # Create metadata
            metadata = {
                'job_name': str(job_context.get('job_name', 'Unknown')),
                'company_id': str(job_context.get('company_id', '')),
                'job_id': str(job_context.get('job_id', '')),
                'document_type': 'consumed_item',
                'data_type': 'consumed',
                'granularity': 'row',
                'cost_code': str(cost_code),
                'category': str(category),
                'amount': float(amount)
            }
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(text)
            
            # Insert into ChromaDB
            if self.chroma_collection is None:
                self.chroma_collection = get_chroma_collection()
            
            self.chroma_collection.add(
                ids=[consumed_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            
            self.stats['embeddings_created'] += 1
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create embedding for {consumed_id}: {e}")
    
    async def migrate_job(
        self,
        company_id: str,
        job_id: str,
        skip_if_exists: bool = True,
        clean_first: bool = False
    ) -> Dict[str, Any]:
        """
        Migrate a single job from Firestore to PostgreSQL + ChromaDB
        
        Args:
            company_id: Company identifier
            job_id: Job identifier
            skip_if_exists: If True, skip migration if job already exists
            clean_first: If True, delete existing data before migrating
        
        Returns:
            Migration statistics dictionary
        """
        logger.info(f"{'='*80}")
        logger.info(f"🚀 Starting ETL migration for job: {company_id}/{job_id}")
        if clean_first:
            logger.info(f"🗑️  Clean mode: Will delete existing data first")
        logger.info(f"{'='*80}")
        
        start_time = datetime.now()
        
        try:
            # Check if job already exists
            job_exists = verify_job_exists(job_id)
            
            if job_exists and not clean_first and skip_if_exists:
                logger.warning(f"⚠️  Job {job_id} already exists in PostgreSQL. Skipping migration.")
                logger.info("💡 Use --force to overwrite or --clean to delete and re-migrate")
                return {
                    'success': False,
                    'message': 'Job already exists',
                    'stats': self.stats
                }
            
            # Clean existing data if requested
            if job_exists and clean_first:
                self._delete_job_data(job_id)
            
            # Step 1: Fetch complete job data from Firestore
            logger.info("📥 Fetching job data from Firestore...")
            job_data = await fetch_job_complete_data(company_id, job_id)
            
            if not job_data:
                error_msg = f"Job {job_id} not found in Firestore"
                logger.error(f"❌ {error_msg}")
                self.stats['errors'].append(error_msg)
                return {
                    'success': False,
                    'message': error_msg,
                    'stats': self.stats
                }
            
            logger.info(f"✅ Job data fetched: {job_data.get('projectTitle', 'Unknown')}")
            
            # Prepare job context for embeddings
            job_context = {
                'job_id': job_id,
                'company_id': company_id,
                'job_name': job_data.get('projectTitle', 'Unknown Job')
            }
            
            # Initialize ChromaDB
            self.chroma_collection = get_chroma_collection()
            
            # Step 2: Transform and migrate within a transaction
            with session_scope() as cursor:
                logger.info("🔄 Starting database transaction...")
                
                # 2.1: Migrate Job metadata
                logger.info("📝 Migrating job metadata...")
                self._insert_job(cursor, job_data)
                self.stats['job_migrated'] = True
                logger.info(f"✅ Job metadata: {job_data.get('projectTitle', 'Unknown')}")
                
                # 2.2: Migrate Estimate data
                estimate_data = extract_estimate_data(job_data)
                if estimate_data:
                    logger.info(f"📝 Migrating estimate data ({estimate_data['total_rows']} rows)...")
                    entries = estimate_data.get('entries', [])
                    
                    for entry in entries:
                        self._insert_estimate(cursor, entry, job_id, job_context)
                        self.stats['estimates_migrated'] += 1
                        
                        # Progress indicator every 10 rows
                        if self.stats['estimates_migrated'] % 10 == 0:
                            logger.info(f"   ... {self.stats['estimates_migrated']} estimates migrated")
                    
                    logger.info(f"✅ Estimates: {self.stats['estimates_migrated']} rows (PostgreSQL + ChromaDB)")
                
                # 2.3: Migrate Flooring Estimate data
                flooring_data = extract_flooring_estimate_data(job_data)
                if flooring_data:
                    logger.info(f"📝 Migrating flooring estimate data ({flooring_data['total_rows']} rows)...")
                    entries = flooring_data.get('entries', [])
                    
                    for entry in entries:
                        self._insert_flooring_estimate(cursor, entry, job_id, job_context)
                        self.stats['flooring_estimates_migrated'] += 1
                    
                    logger.info(f"✅ Flooring estimates: {self.stats['flooring_estimates_migrated']} rows (PostgreSQL + ChromaDB)")
                
                # 2.4: Migrate Schedule data
                schedule_data = extract_schedule_data(job_data)
                if schedule_data:
                    logger.info(f"📝 Migrating schedule data ({schedule_data['total_rows']} rows)...")
                    entries = schedule_data.get('entries', [])
                    
                    for entry in entries:
                        self._insert_schedule_item(cursor, entry, job_id, job_context)
                        self.stats['schedule_items_migrated'] += 1
                    
                    logger.info(f"✅ Schedule items: {self.stats['schedule_items_migrated']} rows (PostgreSQL + ChromaDB)")
                
                # 2.5: Migrate Consumed data
                consumed_data = job_data.get('consumed_data')
                if consumed_data:
                    entries = consumed_data.get('entries', [])
                    if entries:
                        logger.info(f"📝 Migrating consumed data ({len(entries)} items)...")
                        
                        for idx, entry in enumerate(entries):
                            self._insert_consumed_item(cursor, entry, job_id, job_context, idx)
                            self.stats['consumed_items_migrated'] += 1
                        
                        logger.info(f"✅ Consumed items: {self.stats['consumed_items_migrated']} rows (PostgreSQL + ChromaDB)")
                
                logger.info("💾 Committing transaction...")
                # Transaction will auto-commit when exiting context manager
            
            # Step 3: Verification
            logger.info("🔍 Verifying migration...")
            
            # Verify PostgreSQL
            summary = get_job_data_summary(job_id)
            
            verification = {
                'estimates': summary['estimates_count'] == self.stats['estimates_migrated'],
                'flooring': summary['flooring_estimates_count'] == self.stats['flooring_estimates_migrated'],
                'schedule': summary.get('schedule_items_count', 0) == self.stats['schedule_items_migrated'],
                'consumed': summary['consumed_items_count'] == self.stats['consumed_items_migrated']
            }
            
            # Verify ChromaDB
            try:
                chroma_count = self.chroma_collection.count()
                logger.info(f"   📊 ChromaDB total documents: {chroma_count}")
                logger.info(f"   📊 Embeddings created this migration: {self.stats['embeddings_created']}")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not verify ChromaDB count: {e}")
            
            all_verified = all(verification.values())
            
            if all_verified:
                logger.info("✅ Verification successful - all counts match!")
            else:
                logger.warning("⚠️  Verification issues detected:")
                for data_type, passed in verification.items():
                    if not passed:
                        logger.warning(f"   ❌ {data_type} count mismatch")
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"{'='*80}")
            logger.info(f"✅ Migration completed in {duration:.2f} seconds")
            logger.info(f"{'='*80}")
            logger.info(f"📊 Summary:")
            logger.info(f"   Job: {job_data.get('projectTitle', 'Unknown')}")
            logger.info(f"   Estimates: {self.stats['estimates_migrated']}")
            logger.info(f"   Flooring Estimates: {self.stats['flooring_estimates_migrated']}")
            logger.info(f"   Schedule Items: {self.stats['schedule_items_migrated']}")
            logger.info(f"   Consumed Items: {self.stats['consumed_items_migrated']}")
            logger.info(f"   Embeddings Created: {self.stats['embeddings_created']}")
            logger.info(f"{'='*80}")
            
            return {
                'success': True,
                'message': 'Migration completed successfully',
                'stats': self.stats,
                'verification': verification,
                'duration_seconds': duration,
                'job_summary': summary
            }
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'message': error_msg,
                'stats': self.stats
            }


async def main():
    """Main ETL migration script"""
    parser = argparse.ArgumentParser(
        description='Migrate a single job from Firestore to PostgreSQL + ChromaDB'
    )
    parser.add_argument(
        '--company-id',
        required=True,
        help='Company ID (e.g., company123)'
    )
    parser.add_argument(
        '--job-id',
        required=True,
        help='Job ID (e.g., job456)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force migration even if job already exists (will fail due to duplicate keys unless --clean is also used)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Delete existing job data before migrating (USE WITH CAUTION!)'
    )
    
    args = parser.parse_args()
    
    # Warn if clean is used
    if args.clean:
        logger.warning("⚠️  --clean flag is set. This will DELETE all existing data for this job!")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Migration cancelled by user")
            sys.exit(0)
    
    try:
        # Initialize Firebase
        logger.info("🔥 Initializing Firebase...")
        initialize_firebase()
        
        # Initialize PostgreSQL
        logger.info("🐘 Initializing PostgreSQL...")
        from db_connection import init_database
        init_database()
        
        db = get_db()
        if not db.test_connection():
            logger.error("❌ Failed to connect to PostgreSQL")
            sys.exit(1)
        
        # Initialize ChromaDB
        logger.info("🔮 Initializing ChromaDB...")
        get_chroma_collection()
        
        # Run migration
        etl = ETLMigration()
        result = await etl.migrate_job(
            company_id=args.company_id,
            job_id=args.job_id,
            skip_if_exists=not args.force,
            clean_first=args.clean
        )
        
        # Exit with appropriate code
        if result['success']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())