#!/usr/bin/env python3
"""
Fix ChromaDB embeddings for existing PostgreSQL data
"""

import os
from database import get_chroma_collection
from embedding_service import EmbeddingService
from db_connection import get_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_embeddings_for_job(job_id: str):
    """Re-create embeddings for a specific job"""
    
    # Clear existing ChromaDB data for this job
    logger.info(f"Clearing existing embeddings for job {job_id}...")
    collection = get_chroma_collection()
    
    # Delete existing documents for this job
    try:
        results = collection.get(
            where={"job_id": {"$eq": job_id}},
            include=[]
        )
        if results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} existing documents")
    except Exception as e:
        logger.warning(f"Could not delete existing docs: {e}")
    
    # Get data from PostgreSQL
    db = get_db()
    embedding_service = EmbeddingService()
    
    with db.get_cursor() as cursor:
        # Get job info
        cursor.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        job = cursor.fetchone()
        
        if not job:
            logger.error(f"Job {job_id} not found!")
            return
        
        job_context = {
            'job_id': job_id,
            'company_id': job['company_id'],
            'job_name': job['name']
        }
        
        # Get all estimates
        cursor.execute("""
            SELECT id, row_number, area, task_scope, cost_code, 
                   description, total, budgeted_total, notes_remarks
            FROM estimates 
            WHERE job_id = %s
        """, (job_id,))
        
        estimates = cursor.fetchall()
        logger.info(f"Found {len(estimates)} estimates to embed")
        
        # Create embeddings for each estimate
        for est in estimates:
            try:
                # Create text representation
                text = f"""ESTIMATE ROW #{est['row_number']}
Job: {job['name']}
Area: {est['area'] or 'General'}
Task: {est['task_scope'] or ''}
Cost Code: {est['cost_code'] or ''}
Description: {est['description'] or ''}
Estimated: ${est['total']:,.2f}
Budgeted: ${est['budgeted_total']:,.2f}"""
                
                # Create metadata
                metadata = {
                    'job_name': job['name'],
                    'company_id': job['company_id'],
                    'job_id': job_id,
                    'row_number': est['row_number'],
                    'document_type': 'estimate_row',
                    'data_type': 'estimate',  # THIS IS CRITICAL!
                    'granularity': 'row',
                    'area': est['area'] or '',
                    'task_scope': est['task_scope'] or '',
                    'cost_code': est['cost_code'] or '',
                    'description': (est['description'] or '')[:200],
                    'total': float(est['total'] or 0),
                    'budgeted_total': float(est['budgeted_total'] or 0)
                }
                
                # Generate embedding
                embedding = embedding_service.generate_embedding(text)
                
                # Insert into ChromaDB
                collection.add(
                    ids=[est['id']],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[text]
                )
                
                if est['row_number'] % 50 == 0:
                    logger.info(f"  Processed {est['row_number']} estimates...")
                    
            except Exception as e:
                logger.error(f"Failed to embed estimate {est['id']}: {e}")
    
    # Verify
    results = collection.get(
        where={"$and": [
            {"job_id": {"$eq": job_id}},
            {"data_type": {"$eq": "estimate"}}
        ]},
        limit=5
    )
    
    logger.info(f"✅ Complete! Created {len(estimates)} estimate embeddings")
    logger.info(f"Verification: Found {len(results['ids'])} estimate documents in ChromaDB")

if __name__ == "__main__":
    import sys
    
    job_id = "4ZppggAAJuJMZNB8f2ZT"
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    
    logger.info(f"Fixing embeddings for job: {job_id}")
    fix_embeddings_for_job(job_id)