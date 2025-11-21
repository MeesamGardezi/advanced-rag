"""
ETL Migration Script: Firestore → PostgreSQL + ChromaDB
Migrates a single job's data with embeddings to BOTH databases
UPDATED: Now populates ChromaDB with vector embeddings
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
    """Handles ETL migration from Firestore to PostgreSQL + ChromaDB using raw SQL"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.chroma_collection = None
        self.stats = {
            'job_migrated': False,
            'estimates_migrated': 0,
            'flooring_estimates_migrated': 0,
            'schedule_items_migrated': 0,
            'consumed_items_migrated': 0,
            'vector_embeddings_created': 0,
            'errors': []
        }
    
    def _initialize_chroma(self):
        """Initialize ChromaDB collection"""
        if self.chroma_collection is None:
            logger.info("🔮 Initializing ChromaDB...")
            self.chroma_collection = get_chroma_collection()
            logger.info(f"✅ ChromaDB ready (current size: {self.chroma_collection.count()})")
    
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
    
    def _insert_estimate_with_embedding(
        self, 
        cursor, 
        row_data: Dict[str, Any], 
        job_id: str,
        job_context: Dict[str, Any]
    ) -> bool:
        """Insert estimate row into PostgreSQL AND ChromaDB"""
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
        
        # 1. Insert into PostgreSQL
        sql = """
            INSERT INTO estimates (
                id, job_id, row_number, area, task_scope, cost_code, row_type,
                description, notes_remarks, units, qty, rate, total,
                budgeted_rate, budgeted_total, variance, variance_pct,
                materials, has_materials, material_count, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
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
        
        # 2. Create embedding and insert into ChromaDB
        try:
            # Create text representation
            text_representation = self.embedding_service.create_estimate_row_text(
                row_data,
                job_context
            )
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(text_representation)
            
            # Create metadata
            metadata = self.embedding_service.create_estimate_row_metadata(
                row_data,
                job_context
            )
            
            # Insert into ChromaDB
            self.chroma_collection.add(
                ids=[estimate_id],
                embeddings=[embedding],
                documents=[text_representation],
                metadatas=[metadata]
            )
            
            self.stats['vector_embeddings_created'] += 1
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create embedding for {estimate_id}: {e}")
            # Don't fail the whole migration for embedding errors
        
        return True
    
    def _insert_flooring_estimate_with_embedding(
        self, 
        cursor, 
        row_data: Dict[str, Any], 
        job_id: str,
        job_context: Dict[str, Any]
    ) -> bool:
        """Insert flooring estimate row into PostgreSQL AND ChromaDB"""
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
        
        # 1. Insert into PostgreSQL
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
        
        # 2. Create embedding and insert into ChromaDB
        try:
            text_representation = self.embedding_service.create_flooring_estimate_row_text(
                row_data,
                job_context
            )
            
            embedding = self.embedding_service.generate_embedding(text_representation)
            
            metadata = self.embedding_service.create_flooring_estimate_row_metadata(
                row_data,
                job_context
            )
            
            self.chroma_collection.add(
                ids=[flooring_id],
                embeddings=[embedding],
                documents=[text_representation],
                metadatas=[metadata]
            )
            
            self.stats['vector_embeddings_created'] += 1
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create embedding for {flooring_id}: {e}")
        
        return True
    
    def _insert_schedule_item_with_embedding(
        self,
        cursor,
        row_data: Dict[str, Any],
        job_id: str,
        job_context: Dict[str, Any]
    ) -> bool:
        """Insert schedule item into PostgreSQL AND ChromaDB"""
        row_num = row_data.get('row_number', 0)
        
        schedule_id = f"{job_id}_schedule_row_{row_num}"
        
        # Parse values
        hours = float(row_data.get('hours', 0))
        consumed = float(row_data.get('consumed', 0))
        percentage_complete = float(row_data.get('percentageComplete', 0))
        
        # 1. Insert into PostgreSQL
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
        
        # 2. Create embedding and insert into ChromaDB
        try:
            # Create text representation for schedule
            text_representation = f"""SCHEDULE ITEM #{row_num}
Job: {job_context.get('job_name', 'Unknown')}
Task: {row_data.get('task', '')}
Type: {row_data.get('taskType', 'labour')}
Hours: {hours}
Consumed: {consumed}
Progress: {percentage_complete}%"""
            
            embedding = self.embedding_service.generate_embedding(text_representation)
            
            metadata = {
                'job_name': str(job_context.get('job_name', 'Unknown')),
                'company_id': str(job_context.get('company_id', '')),
                'job_id': str(job_id),
                'row_number': int(row_num),
                'document_type': 'schedule_row',
                'data_type': 'schedule',
                'granularity': 'row',
                'task': str(row_data.get('task', ''))[:200],
                'hours': float(hours),
                'consumed': float(consumed)
            }
            
            self.chroma_collection.add(
                ids=[schedule_id],
                embeddings=[embedding],
                documents=[text_representation],
                metadatas=[metadata]
            )
            
            self.stats['vector_embeddings_created'] += 1
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create embedding for {schedule_id}: {e}")
        
        return True
    
    def _insert_consumed_item(self, cursor, item_data: Dict[str, Any], job_id: str, index: int) -> bool:
        """Insert consumed item using raw SQL (no embedding needed for consumed)"""
        cost_code = item_data.get('costCode', 'Unknown')
        amount_str = item_data.get('amount', '0')
        
        try:
            amount = float(amount_str) if amount_str else 0.0
        except (ValueError, TypeError):
            amount = 0.0
        
        category = self.embedding_service.categorize_cost_code(cost_code)
        
        consumed_id = f"{job_id}_consumed_{index}"
        
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
        return True
    
    async def migrate_job(self, company_id: str, job_id: str, skip_if_exists: bool = True) -> Dict[str, Any]:
        """
        Migrate a single job from Firestore to PostgreSQL + ChromaDB
        
        Args:
            company_id: Company identifier
            job_id: Job identifier
            skip_if_exists: If True, skip migration if job already exists
        
        Returns:
            Migration statistics dictionary
        """
        logger.info(f"{'='*80}")
        logger.info(f"🚀 Starting ETL migration for job: {company_id}/{job_id}")
        logger.info(f"{'='*80}")
        
        start_time = datetime.now()
        
        try:
            # Initialize ChromaDB
            self._initialize_chroma()
            
            # Check if job already exists
            if skip_if_exists and verify_job_exists(job_id):
                logger.warning(f"⚠️  Job {job_id} already exists in PostgreSQL. Skipping migration.")
                logger.info("💡 Use --force flag to overwrite existing data")
                return {
                    'success': False,
                    'message': 'Job already exists',
                    'stats': self.stats
                }
            
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
            
            # Create job context for embeddings
            job_context = {
                'job_name': job_data.get('projectTitle', 'Unknown Job'),
                'company_id': company_id,
                'job_id': job_id,
                'client_name': job_data.get('clientName', ''),
                'site_location': f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', ')
            }
            
            # Step 2: Transform and migrate within a transaction
            with session_scope() as cursor:
                logger.info("🔄 Starting database transaction...")
                
                # 2.1: Migrate Job metadata
                logger.info("📝 Migrating job metadata...")
                self._insert_job(cursor, job_data)
                self.stats['job_migrated'] = True
                logger.info(f"✅ Job metadata: {job_data.get('projectTitle', 'Unknown')}")
                
                # 2.2: Migrate Estimate data WITH EMBEDDINGS
                estimate_data = extract_estimate_data(job_data)
                if estimate_data:
                    logger.info(f"📝 Migrating estimate data ({estimate_data['total_rows']} rows)...")
                    entries = estimate_data.get('entries', [])
                    
                    for entry in entries:
                        self._insert_estimate_with_embedding(cursor, entry, job_id, job_context)
                        self.stats['estimates_migrated'] += 1
                        
                        # Log progress every 50 rows
                        if self.stats['estimates_migrated'] % 50 == 0:
                            logger.info(f"   Progress: {self.stats['estimates_migrated']}/{len(entries)} estimates...")
                    
                    logger.info(f"✅ Estimates: {self.stats['estimates_migrated']} rows + embeddings")
                
                # 2.3: Migrate Flooring Estimate data WITH EMBEDDINGS
                flooring_data = extract_flooring_estimate_data(job_data)
                if flooring_data:
                    logger.info(f"📝 Migrating flooring estimate data ({flooring_data['total_rows']} rows)...")
                    entries = flooring_data.get('entries', [])
                    
                    for entry in entries:
                        self._insert_flooring_estimate_with_embedding(cursor, entry, job_id, job_context)
                        self.stats['flooring_estimates_migrated'] += 1
                    
                    logger.info(f"✅ Flooring estimates: {self.stats['flooring_estimates_migrated']} rows + embeddings")
                
                # 2.4: Migrate Schedule data WITH EMBEDDINGS
                schedule_data = extract_schedule_data(job_data)
                if schedule_data:
                    logger.info(f"📝 Migrating schedule data ({schedule_data['total_rows']} rows)...")
                    entries = schedule_data.get('entries', [])
                    
                    for entry in entries:
                        self._insert_schedule_item_with_embedding(cursor, entry, job_id, job_context)
                        self.stats['schedule_items_migrated'] += 1
                    
                    logger.info(f"✅ Schedule items: {self.stats['schedule_items_migrated']} rows + embeddings")
                
                # 2.5: Migrate Consumed data (no embeddings needed)
                consumed_data = job_data.get('consumed_data')
                if consumed_data:
                    entries = consumed_data.get('entries', [])
                    if entries:
                        logger.info(f"📝 Migrating consumed data ({len(entries)} items)...")
                        
                        for idx, entry in enumerate(entries):
                            self._insert_consumed_item(cursor, entry, job_id, idx)
                            self.stats['consumed_items_migrated'] += 1
                        
                        logger.info(f"✅ Consumed items: {self.stats['consumed_items_migrated']} rows")
                
                logger.info("💾 Committing transaction...")
                # Transaction will auto-commit when exiting context manager
            
            # Step 3: Verification
            logger.info("🔍 Verifying migration...")
            summary = get_job_data_summary(job_id)
            
            # Check PostgreSQL counts
            verification = {
                'estimates': summary['estimates_count'] == self.stats['estimates_migrated'],
                'flooring': summary['flooring_estimates_count'] == self.stats['flooring_estimates_migrated'],
                'consumed': summary['consumed_items_count'] == self.stats['consumed_items_migrated']
            }
            
            # Check ChromaDB count
            chroma_count = self.chroma_collection.count()
            expected_vectors = (
                self.stats['estimates_migrated'] + 
                self.stats['flooring_estimates_migrated'] + 
                self.stats['schedule_items_migrated']
            )
            
            logger.info(f"📊 ChromaDB verification:")
            logger.info(f"   Expected vectors: {expected_vectors}")
            logger.info(f"   Actual vectors: {chroma_count}")
            logger.info(f"   Created this run: {self.stats['vector_embeddings_created']}")
            
            all_verified = all(verification.values())
            
            if all_verified:
                logger.info("✅ PostgreSQL verification successful - all counts match!")
            else:
                logger.warning("⚠️  PostgreSQL verification issues detected:")
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
            logger.info(f"   Estimates: {self.stats['estimates_migrated']} (PostgreSQL + ChromaDB)")
            logger.info(f"   Flooring Estimates: {self.stats['flooring_estimates_migrated']} (PostgreSQL + ChromaDB)")
            logger.info(f"   Schedule Items: {self.stats['schedule_items_migrated']} (PostgreSQL + ChromaDB)")
            logger.info(f"   Consumed Items: {self.stats['consumed_items_migrated']} (PostgreSQL only)")
            logger.info(f"   Total Embeddings: {self.stats['vector_embeddings_created']}")
            logger.info(f"{'='*80}")
            
            return {
                'success': True,
                'message': 'Migration completed successfully',
                'stats': self.stats,
                'verification': verification,
                'duration_seconds': duration,
                'job_summary': summary,
                'chroma_count': chroma_count
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
        help='Force migration even if job already exists (will fail due to duplicate keys)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize Firebase
        logger.info("🔥 Initializing Firebase...")
        initialize_firebase()
        
        # Initialize PostgreSQL
        logger.info("🐘 Initializing PostgreSQL...")
        from db_connection import init_database
        init_database()  # CREATE TABLES IF NOT EXIST
        
        db = get_db()
        if not db.test_connection():
            logger.error("❌ Failed to connect to PostgreSQL")
            sys.exit(1)
        
        # Run migration
        etl = ETLMigration()
        result = await etl.migrate_job(
            company_id=args.company_id,
            job_id=args.job_id,
            skip_if_exists=not args.force
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