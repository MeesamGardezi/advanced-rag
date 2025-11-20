"""
ETL Migration Script: Firestore → PostgreSQL
Migrates a single job's data with transaction support and verification
"""

import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Import existing Firebase/Firestore functions
from database import (
    initialize_firebase,
    fetch_job_complete_data,
    extract_estimate_data,
    extract_flooring_estimate_data,
    extract_schedule_data
)
from embedding_service import EmbeddingService

# Import PostgreSQL components
from db_connection import get_db, session_scope, get_job_data_summary, verify_job_exists
from database_schema import (
    Job, Estimate, FlooringEstimate, ScheduleItem, ConsumedItem
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETLMigration:
    """Handles ETL migration from Firestore to PostgreSQL"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.stats = {
            'job_migrated': False,
            'estimates_migrated': 0,
            'flooring_estimates_migrated': 0,
            'schedule_items_migrated': 0,
            'consumed_items_migrated': 0,
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
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"⚠️  Failed to parse date {date_value}: {e}")
        
        return None
    
    def _transform_job(self, job_data: Dict[str, Any]) -> Job:
        """Transform Firestore job data to Job model"""
        job_id = job_data['job_id']
        company_id = job_data['company_id']
        
        # Compute site_location
        city = job_data.get('siteCity', '')
        state = job_data.get('siteState', '')
        site_location = f"{city}, {state}".strip(', ')
        
        return Job(
            id=job_id,
            company_id=company_id,
            name=job_data.get('projectTitle', 'Unknown Job'),
            client_name=job_data.get('clientName', ''),
            site_city=city,
            site_state=state,
            site_location=site_location,
            project_description=job_data.get('projectDescription', ''),
            status=job_data.get('status', 'active'),
            estimate_type=job_data.get('estimateType', 'general'),
            created_date=self._parse_datetime(job_data.get('createdDate')),
            last_updated=datetime.utcnow(),
            schedule_last_updated=self._parse_datetime(job_data.get('scheduleLastUpdated'))
        )
    
    def _transform_estimate_row(self, row_data: Dict[str, Any], job_id: str) -> Estimate:
        """Transform Firestore estimate row to Estimate model"""
        row_num = row_data.get('row_number', 0)
        
        # Parse numerical values
        qty = float(row_data.get('qty', 0))
        rate = float(row_data.get('rate', 0))
        total = float(row_data.get('total', 0))
        budgeted_rate = float(row_data.get('budgetedRate', 0))
        budgeted_total = float(row_data.get('budgetedTotal', 0))
        
        # Calculate variance
        variance = total - budgeted_total
        variance_pct = (variance / budgeted_total * 100) if budgeted_total != 0 else 0
        
        # Handle materials
        materials = row_data.get('materials', [])
        has_materials = len(materials) > 0
        material_count = len(materials)
        
        # Generate unique ID
        estimate_id = f"{job_id}_est_row_{row_num}"
        
        return Estimate(
            id=estimate_id,
            job_id=job_id,
            row_number=row_num,
            area=row_data.get('area', ''),
            task_scope=row_data.get('taskScope', ''),
            cost_code=row_data.get('costCode', ''),
            row_type=row_data.get('rowType', 'estimate'),
            description=row_data.get('description', ''),
            notes_remarks=row_data.get('notesRemarks', ''),
            units=row_data.get('units', ''),
            qty=qty,
            rate=rate,
            total=total,
            budgeted_rate=budgeted_rate,
            budgeted_total=budgeted_total,
            variance=variance,
            variance_pct=variance_pct,
            materials=materials if materials else None,
            has_materials=has_materials,
            material_count=material_count
        )
    
    def _transform_flooring_estimate_row(self, row_data: Dict[str, Any], job_id: str) -> FlooringEstimate:
        """Transform Firestore flooring estimate row to FlooringEstimate model"""
        row_num = row_data.get('row_number', 0)
        
        # Parse numerical values
        measured_qty = float(row_data.get('measuredQty', 0))
        supplier_qty = float(row_data.get('supplierQty', 0))
        waste_factor = float(row_data.get('wasteFactor', 0))
        qty_including_waste = float(row_data.get('qtyIncludingWaste', 0))
        unit_price = float(row_data.get('unitPrice', 0))
        cost_price = float(row_data.get('costPrice', 0))
        tax_freight = float(row_data.get('taxFreight', 0))
        total_cost = float(row_data.get('totalCost', 0))
        sale_price = float(row_data.get('salePrice', 0))
        
        # Calculate profit
        profit = sale_price - total_cost
        margin_pct = (profit / sale_price * 100) if sale_price > 0 else 0
        
        # Generate unique ID
        flooring_id = f"{job_id}_flooring_row_{row_num}"
        
        return FlooringEstimate(
            id=flooring_id,
            job_id=job_id,
            row_number=row_num,
            floor_type_id=row_data.get('floorTypeId', ''),
            vendor=row_data.get('vendor', ''),
            item_material_name=row_data.get('itemMaterialName', ''),
            brand=row_data.get('brand', ''),
            unit=row_data.get('unit', ''),
            measured_qty=measured_qty,
            supplier_qty=supplier_qty,
            waste_factor=waste_factor,
            qty_including_waste=qty_including_waste,
            unit_price=unit_price,
            cost_price=cost_price,
            tax_freight=tax_freight,
            total_cost=total_cost,
            sale_price=sale_price,
            profit=profit,
            margin_pct=margin_pct,
            notes_remarks=row_data.get('notesRemarks', '')
        )
    
    def _transform_schedule_item(self, item_data: Dict[str, Any], job_id: str) -> ScheduleItem:
        """Transform Firestore schedule item to ScheduleItem model"""
        row_num = item_data.get('row_number', 0)
        
        # Parse numerical values
        hours = float(item_data.get('hours', 0))
        consumed = float(item_data.get('consumed', 0))
        percentage_complete = float(item_data.get('percentageComplete', 0))
        
        # Parse dates
        start_date = self._parse_datetime(item_data.get('startDate'))
        end_date = self._parse_datetime(item_data.get('endDate'))
        
        # Handle resources (JSON)
        resources = item_data.get('resources', {})
        
        # Generate unique ID
        schedule_id = f"{job_id}_schedule_row_{row_num}"
        
        return ScheduleItem(
            id=schedule_id,
            job_id=job_id,
            row_number=row_num,
            task=item_data.get('task', '').strip(),
            is_main_task=item_data.get('isMainTask', False),
            task_type=item_data.get('taskType', 'labour'),
            hours=hours,
            consumed=consumed,
            percentage_complete=percentage_complete,
            start_date=start_date,
            end_date=end_date,
            resources=resources if resources else None
        )
    
    def _transform_consumed_item(self, item_data: Dict[str, Any], job_id: str, index: int) -> ConsumedItem:
        """Transform Firestore consumed item to ConsumedItem model"""
        cost_code = item_data.get('costCode', 'Unknown')
        amount_str = item_data.get('amount', '0')
        
        # Parse amount
        try:
            amount = float(amount_str) if amount_str else 0.0
        except (ValueError, TypeError):
            amount = 0.0
        
        # Categorize cost code
        category = self.embedding_service.categorize_cost_code(cost_code)
        
        # Generate unique ID
        consumed_id = f"{job_id}_consumed_{index}"
        
        return ConsumedItem(
            id=consumed_id,
            job_id=job_id,
            cost_code=cost_code,
            amount=amount,
            category=category,
            last_updated=datetime.utcnow()
        )
    
    async def migrate_job(self, company_id: str, job_id: str, skip_if_exists: bool = True) -> Dict[str, Any]:
        """
        Migrate a single job from Firestore to PostgreSQL
        
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
            
            # Step 2: Transform and migrate within a transaction
            with session_scope() as session:
                logger.info("🔄 Starting database transaction...")
                
                # 2.1: Migrate Job metadata
                logger.info("📝 Migrating job metadata...")
                job = self._transform_job(job_data)
                session.add(job)
                self.stats['job_migrated'] = True
                logger.info(f"✅ Job metadata: {job.name}")
                
                # 2.2: Migrate Estimate data
                estimate_data = extract_estimate_data(job_data)
                if estimate_data:
                    logger.info(f"📝 Migrating estimate data ({estimate_data['total_rows']} rows)...")
                    entries = estimate_data.get('entries', [])
                    
                    for entry in entries:
                        estimate = self._transform_estimate_row(entry, job_id)
                        session.add(estimate)
                        self.stats['estimates_migrated'] += 1
                    
                    logger.info(f"✅ Estimates: {self.stats['estimates_migrated']} rows")
                
                # 2.3: Migrate Flooring Estimate data
                flooring_data = extract_flooring_estimate_data(job_data)
                if flooring_data:
                    logger.info(f"📝 Migrating flooring estimate data ({flooring_data['total_rows']} rows)...")
                    entries = flooring_data.get('entries', [])
                    
                    for entry in entries:
                        flooring = self._transform_flooring_estimate_row(entry, job_id)
                        session.add(flooring)
                        self.stats['flooring_estimates_migrated'] += 1
                    
                    logger.info(f"✅ Flooring estimates: {self.stats['flooring_estimates_migrated']} rows")
                
                # 2.4: Migrate Schedule data
                schedule_data = extract_schedule_data(job_data)
                if schedule_data:
                    logger.info(f"📝 Migrating schedule data ({schedule_data['total_rows']} rows)...")
                    entries = schedule_data.get('entries', [])
                    
                    # Filter out empty tasks
                    valid_entries = [e for e in entries if e.get('task', '').strip()]
                    
                    for entry in valid_entries:
                        schedule_item = self._transform_schedule_item(entry, job_id)
                        session.add(schedule_item)
                        self.stats['schedule_items_migrated'] += 1
                    
                    logger.info(f"✅ Schedule items: {self.stats['schedule_items_migrated']} rows")
                
                # 2.5: Migrate Consumed data
                consumed_data = job_data.get('consumed_data')
                if consumed_data:
                    entries = consumed_data.get('entries', [])
                    if entries:
                        logger.info(f"📝 Migrating consumed data ({len(entries)} items)...")
                        
                        for idx, entry in enumerate(entries):
                            consumed_item = self._transform_consumed_item(entry, job_id, idx)
                            session.add(consumed_item)
                            self.stats['consumed_items_migrated'] += 1
                        
                        logger.info(f"✅ Consumed items: {self.stats['consumed_items_migrated']} rows")
                
                # Transaction will auto-commit when exiting context manager
                logger.info("💾 Committing transaction...")
            
            # Step 3: Verification
            logger.info("🔍 Verifying migration...")
            summary = get_job_data_summary(job_id)
            
            verification = {
                'estimates': summary['estimates_count'] == self.stats['estimates_migrated'],
                'flooring': summary['flooring_estimates_count'] == self.stats['flooring_estimates_migrated'],
                'schedule': summary['schedule_items_count'] == self.stats['schedule_items_migrated'],
                'consumed': summary['consumed_items_count'] == self.stats['consumed_items_migrated']
            }
            
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
            logger.info(f"   Job: {job.name}")
            logger.info(f"   Estimates: {self.stats['estimates_migrated']}")
            logger.info(f"   Flooring Estimates: {self.stats['flooring_estimates_migrated']}")
            logger.info(f"   Schedule Items: {self.stats['schedule_items_migrated']}")
            logger.info(f"   Consumed Items: {self.stats['consumed_items_migrated']}")
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
        description='Migrate a single job from Firestore to PostgreSQL'
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