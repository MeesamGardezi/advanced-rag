import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import firebase_admin
from firebase_admin import credentials, firestore
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from datetime import datetime

# Import new enhanced components
from core.config import config
from core.qdrant_client import get_qdrant_client, initialize_qdrant

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Global variables for database connections
firebase_db = None
chroma_client = None
chroma_collection = None
qdrant_client = None

# Migration status tracking
migration_status = {
    'chromadb_to_qdrant': False,
    'last_migration': None,
    'migration_errors': []
}

def initialize_firebase():
    """Initialize Firebase Admin SDK with enhanced error handling"""
    global firebase_db
    
    if firebase_db is not None:
        logger.info("Firebase already initialized")
        return firebase_db
    
    try:
        # Use configuration from enhanced config system
        firebase_config = config.get_firebase_credentials()
        
        # Initialize Firebase Admin
        cred = credentials.Certificate(firebase_config)
        
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        
        firebase_db = firestore.client()
        logger.info("✅ Firebase initialized successfully")
        return firebase_db
        
    except Exception as e:
        logger.error(f"❌ Error initializing Firebase: {e}")
        raise

def initialize_chromadb():
    """Initialize ChromaDB with persistence (legacy compatibility)"""
    global chroma_client, chroma_collection
    
    if chroma_collection is not None:
        logger.info("ChromaDB already initialized")
        return chroma_collection
    
    try:
        persist_path = config.database.chroma_persist_path
        collection_name = config.database.chroma_collection_name
        
        # Create OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.embedding.openai_api_key,
            model_name=config.embedding.embedding_model
        )
        
        # Initialize ChromaDB with persistence
        chroma_client = chromadb.PersistentClient(path=persist_path)
        
        # Get or create collection
        chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        
        logger.info(f"✅ ChromaDB initialized successfully")
        logger.info(f"📊 Current collection size: {chroma_collection.count()}")
        return chroma_collection
        
    except Exception as e:
        logger.error(f"❌ Error initializing ChromaDB: {e}")
        raise

def initialize_databases():
    """Initialize all database connections"""
    try:
        # Initialize Firebase
        initialize_firebase()
        
        # Initialize Qdrant (primary vector DB)
        global qdrant_client
        qdrant_client = initialize_qdrant()
        
        # Initialize ChromaDB for compatibility/migration
        try:
            initialize_chromadb()
        except Exception as e:
            logger.warning(f"⚠️  ChromaDB initialization failed (non-critical): {e}")
        
        logger.info("🚀 All database connections initialized")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize databases: {e}")
        return False

def get_firebase_db():
    """Get Firebase database instance"""
    if firebase_db is None:
        initialize_firebase()
    return firebase_db

def get_chroma_collection():
    """Get ChromaDB collection instance (legacy compatibility)"""
    if chroma_collection is None:
        initialize_chromadb()
    return chroma_collection

def get_qdrant_client():
    """Get Qdrant client instance (primary vector DB)"""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = initialize_qdrant()
    return qdrant_client

def get_primary_vector_db():
    """Get primary vector database (Qdrant)"""
    return get_qdrant_client()

async def migrate_chromadb_to_qdrant(force: bool = False) -> Dict[str, Any]:
    """Migrate data from ChromaDB to Qdrant"""
    global migration_status
    
    if migration_status['chromadb_to_qdrant'] and not force:
        logger.info("✅ Migration already completed")
        return {
            'status': 'already_completed',
            'last_migration': migration_status['last_migration']
        }
    
    try:
        logger.info("🔄 Starting ChromaDB to Qdrant migration...")
        
        # Get both clients
        chroma_collection = get_chroma_collection()
        qdrant_client = get_qdrant_client()
        
        # Perform migration
        migration_result = qdrant_client.migrate_from_chromadb(chroma_collection)
        
        if migration_result['migrated'] > 0:
            migration_status['chromadb_to_qdrant'] = True
            migration_status['last_migration'] = datetime.now().isoformat()
            migration_status['migration_errors'] = migration_result.get('errors', [])
            
            logger.info(f"✅ Migration completed: {migration_result['migrated']} documents migrated")
        else:
            logger.warning("⚠️  No documents were migrated")
        
        return {
            'status': 'completed',
            'migrated_count': migration_result['migrated'],
            'errors': migration_result.get('errors', []),
            'total_source': migration_result.get('total_source', 0)
        }
        
    except Exception as e:
        error_msg = f"Migration failed: {e}"
        logger.error(f"❌ {error_msg}")
        migration_status['migration_errors'].append(error_msg)
        
        return {
            'status': 'failed',
            'error': error_msg,
            'migrated_count': 0
        }

# Enhanced data fetching functions with better error handling and logging

async def fetch_job_consumed_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch consumed data for a specific job with enhanced error handling"""
    try:
        db = get_firebase_db()
        doc_ref = db.collection('companies').document(company_id).collection('jobs').document(job_id).collection('data').document('consumed')
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            data['company_id'] = company_id
            data['job_id'] = job_id
            data['data_type'] = 'consumed'
            
            logger.debug(f"✅ Fetched consumed data for {company_id}/{job_id}")
            return data
        else:
            logger.debug(f"No consumed data found for company {company_id}, job {job_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching consumed job data for {company_id}/{job_id}: {e}")
        return None

async def fetch_job_complete_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch complete job data including estimate and schedule with enhanced logging"""
    try:
        db = get_firebase_db()
        job_ref = db.collection('companies').document(company_id).collection('jobs').document(job_id)
        job_doc = job_ref.get()
        
        if not job_doc.exists:
            logger.debug(f"No job found for company {company_id}, job {job_id}")
            return None
        
        job_data = job_doc.to_dict()
        job_data['company_id'] = company_id
        job_data['job_id'] = job_id
        
        # Also get consumed data if it exists
        consumed_data = await fetch_job_consumed_data(company_id, job_id)
        if consumed_data:
            job_data['consumed_data'] = consumed_data
        
        logger.debug(f"✅ Fetched complete job data for {company_id}/{job_id}")
        return job_data
            
    except Exception as e:
        logger.error(f"Error fetching complete job data for {company_id}/{job_id}: {e}")
        return None

def extract_estimate_data(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract estimate data from complete job data with enhanced validation"""
    if not job_data or 'estimate' not in job_data:
        return None
    
    estimate_entries = job_data['estimate']
    if not estimate_entries or not isinstance(estimate_entries, list):
        return None
    
    # Filter out invalid entries
    valid_entries = []
    for entry in estimate_entries:
        if isinstance(entry, dict) and entry.get('taskScope'):  # Basic validation
            valid_entries.append(entry)
    
    if not valid_entries:
        logger.warning(f"No valid estimate entries found for job {job_data.get('job_id', 'unknown')}")
        return None
    
    return {
        'company_id': job_data['company_id'],
        'job_id': job_data['job_id'],
        'job_name': job_data.get('projectTitle', 'Unknown Job'),
        'data_type': 'estimate',
        'entries': valid_entries,
        'last_updated': job_data.get('createdDate', ''),
        'project_description': job_data.get('projectDescription', ''),
        'client_name': job_data.get('clientName', ''),
        'site_location': f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', ')
    }

def extract_schedule_data(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract schedule data from complete job data with enhanced validation"""
    if not job_data or 'schedule' not in job_data:
        return None
    
    schedule_entries = job_data['schedule']
    if not schedule_entries or not isinstance(schedule_entries, list):
        return None
    
    # Filter out invalid entries
    valid_entries = []
    for entry in schedule_entries:
        if isinstance(entry, dict) and entry.get('task', '').strip():  # Basic validation
            valid_entries.append(entry)
    
    if not valid_entries:
        logger.warning(f"No valid schedule entries found for job {job_data.get('job_id', 'unknown')}")
        return None
    
    return {
        'company_id': job_data['company_id'],
        'job_id': job_data['job_id'],
        'job_name': job_data.get('projectTitle', 'Unknown Job'),
        'data_type': 'schedule',
        'entries': valid_entries,
        'last_updated': job_data.get('createdDate', ''),
        'project_description': job_data.get('projectDescription', ''),
        'client_name': job_data.get('clientName', ''),
        'site_location': f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', '),
        'schedule_last_updated': job_data.get('scheduleLastUpdated', '')
    }

async def fetch_all_job_complete_data(company_id: Optional[str] = None, 
                                    limit_per_company: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch all complete job data with enhanced filtering and limits"""
    try:
        db = get_firebase_db()
        all_jobs_data = []
        
        # If company_id is specified, only fetch that company's data
        if company_id:
            companies_to_process = [company_id]
        else:
            # Get all companies
            companies_ref = db.collection('companies')
            companies = companies_ref.stream()
            companies_to_process = [company.id for company in companies]
        
        total_processed = 0
        
        for comp_id in companies_to_process:
            logger.info(f"Processing company: {comp_id}")
            
            # Get all jobs for this company
            jobs_ref = db.collection('companies').document(comp_id).collection('jobs')
            
            # Apply limit if specified
            if limit_per_company:
                jobs_ref = jobs_ref.limit(limit_per_company)
            
            jobs = jobs_ref.stream()
            
            company_job_count = 0
            
            for job in jobs:
                job_id = job.id
                job_data = job.to_dict()
                
                if not job_data:  # Skip empty jobs
                    continue
                
                job_data['company_id'] = comp_id
                job_data['job_id'] = job_id
                
                # Get consumed data
                consumed_data = await fetch_job_consumed_data(comp_id, job_id)
                
                # Create separate data objects for each type
                job_datasets = []
                
                # Add consumed data if exists
                if consumed_data and 'entries' in consumed_data and consumed_data['entries']:
                    job_datasets.append(consumed_data)
                
                # Add estimate data if exists
                estimate_data = extract_estimate_data(job_data)
                if estimate_data:
                    job_datasets.append(estimate_data)
                
                # Add schedule data if exists
                schedule_data = extract_schedule_data(job_data)
                if schedule_data:
                    job_datasets.append(schedule_data)
                
                # Add all datasets for this job
                all_jobs_data.extend(job_datasets)
                company_job_count += 1
                total_processed += 1
            
            logger.info(f"✅ Processed {company_job_count} jobs for company {comp_id}")
        
        logger.info(f"✅ Fetched data for {len(all_jobs_data)} job datasets from {total_processed} jobs")
        return all_jobs_data
        
    except Exception as e:
        logger.error(f"❌ Error fetching all job complete data: {e}")
        return []

def clear_chroma_collection():
    """Clear all documents from ChromaDB collection (for testing)"""
    try:
        collection = get_chroma_collection()
        
        # Get all IDs and delete them
        all_data = collection.get()
        if all_data['ids']:
            collection.delete(ids=all_data['ids'])
            logger.info(f"🗑️  Cleared {len(all_data['ids'])} documents from ChromaDB collection")
        else:
            logger.info("ChromaDB collection is already empty")
            
    except Exception as e:
        logger.error(f"❌ Error clearing ChromaDB collection: {e}")
        raise

def clear_qdrant_collection():
    """Clear all documents from Qdrant collection"""
    try:
        qdrant_client = get_qdrant_client()
        success = qdrant_client.clear_collection()
        
        if success:
            logger.info("🗑️  Cleared all documents from Qdrant collection")
        else:
            logger.error("❌ Failed to clear Qdrant collection")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Error clearing Qdrant collection: {e}")
        return False

def test_connections():
    """Test all database connections with enhanced reporting"""
    results = {
        "firebase": False,
        "chromadb": False,
        "qdrant": False,
        "openai": False,
        "overall_health": False
    }
    
    # Test Firebase
    try:
        db = get_firebase_db()
        companies = db.collection('companies').limit(1).get()
        results["firebase"] = True
        logger.info("✅ Firebase connection successful")
    except Exception as e:
        logger.error(f"❌ Firebase connection failed: {e}")
    
    # Test Qdrant (primary vector DB)
    try:
        qdrant_client = get_qdrant_client()
        health_check = qdrant_client.health_check()
        results["qdrant"] = health_check["status"] == "healthy"
        
        if results["qdrant"]:
            logger.info(f"✅ Qdrant connection successful (points: {health_check.get('points_count', 'unknown')})")
        else:
            logger.error(f"❌ Qdrant connection failed: {health_check.get('error', 'unknown')}")
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
    
    # Test ChromaDB (legacy compatibility)
    try:
        collection = get_chroma_collection()
        count = collection.count()
        results["chromadb"] = True
        logger.info(f"✅ ChromaDB connection successful (documents: {count})")
    except Exception as e:
        logger.warning(f"⚠️  ChromaDB connection failed (non-critical): {e}")
    
    # Test OpenAI (indirectly through Qdrant embedding)
    try:
        # Simple embedding test
        from embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        test_embedding = embedding_service.generate_embedding("test connection")
        results["openai"] = len(test_embedding) > 0
        logger.info("✅ OpenAI API connection successful")
    except Exception as e:
        logger.error(f"❌ OpenAI API connection failed: {e}")
    
    # Overall health assessment
    critical_systems = ["firebase", "qdrant", "openai"]
    critical_working = sum(1 for system in critical_systems if results[system])
    results["overall_health"] = critical_working >= len(critical_systems)
    
    health_status = "healthy" if results["overall_health"] else "degraded"
    logger.info(f"🏥 System health: {health_status} ({critical_working}/{len(critical_systems)} critical systems)")
    
    return results

def get_database_stats() -> Dict[str, Any]:
    """Get comprehensive database statistics"""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'firebase': {'status': 'unknown'},
        'qdrant': {'status': 'unknown'},
        'chromadb': {'status': 'unknown'},
        'migration': migration_status.copy()
    }
    
    # Firebase stats
    try:
        db = get_firebase_db()
        companies_count = len(list(db.collection('companies').limit(100).get()))
        stats['firebase'] = {
            'status': 'healthy',
            'companies_count': companies_count
        }
    except Exception as e:
        stats['firebase'] = {'status': 'error', 'error': str(e)}
    
    # Qdrant stats
    try:
        qdrant_client = get_qdrant_client()
        qdrant_stats = qdrant_client.get_construction_stats()
        stats['qdrant'] = {
            'status': 'healthy',
            **qdrant_stats
        }
    except Exception as e:
        stats['qdrant'] = {'status': 'error', 'error': str(e)}
    
    # ChromaDB stats
    try:
        collection = get_chroma_collection()
        stats['chromadb'] = {
            'status': 'healthy',
            'document_count': collection.count()
        }
    except Exception as e:
        stats['chromadb'] = {'status': 'error', 'error': str(e)}
    
    return stats

# Initialize connections when module is imported
def init_all():
    """Initialize all database connections"""
    return initialize_databases()

# Backward compatibility aliases
async def fetch_job_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch consumed data for a specific job (backward compatibility)"""
    return await fetch_job_consumed_data(company_id, job_id)

async def fetch_all_job_data(company_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all job consumed data (backward compatibility)"""
    try:
        all_complete_data = await fetch_all_job_complete_data(company_id)
        # Filter to only return consumed data for backward compatibility
        consumed_data = [data for data in all_complete_data if data.get('data_type') == 'consumed']
        return consumed_data
    except Exception as e:
        logger.error(f"❌ Error fetching all job data: {e}")
        return []