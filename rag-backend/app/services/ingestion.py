"""
Ingestion service for transforming job data into searchable chunks.
"""

from typing import List, Dict
from app.models import Job, ComparisonCategory, DocumentChunk
from app.services.embedding import get_embedding_service
from app.services.qdrant_service import get_qdrant_service
from app.services.firebase_service import get_firebase_service


class IngestionService:
    """Service for ingesting and chunking job data."""
    
    def __init__(self):
        """Initialize ingestion service."""
        self.embedding_service = get_embedding_service()
        self.qdrant_service = get_qdrant_service()
        self.firebase_service = get_firebase_service()
    
    def create_estimate_chunks(self, job: Job, company_id: str, job_id: str) -> List[DocumentChunk]:
        """
        Create searchable chunks from job estimate data.
        
        Args:
            job: Job object with estimate data
            company_id: Company ID
            job_id: Job ID (Firestore document ID)
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        for row in job.estimate:
            # Format materials summary
            materials_summary = ""
            if row.materials:
                materials_list = [
                    f"{m.name} ({m.quantity} {m.unit} @ ${m.cost})"
                    for m in row.materials
                ]
                materials_summary = "; ".join(materials_list)
            
            # Create content text
            content = f"""Job {job.job_prefix} - {job.project_title}
Client: {job.client_name}
Site: {job.site_address}

Cost Code: {row.cost_code}
Area: {row.area}
Task: {row.task_scope}
Description: {row.description}
Type: {row.row_type}

Units: {row.qty} {row.units}
Rate: ${row.rate:.2f}/unit
Budgeted Total: ${row.budgeted_total:.2f}
Actual Total: ${row.total:.2f}
Variance: ${row.total - row.budgeted_total:.2f}

Materials: {materials_summary if materials_summary else "None"}
Notes: {row.notes_remarks if row.notes_remarks else "None"}"""
            
            # Create metadata - CRITICAL: Use job_id parameter, not job.document_id
            metadata = {
                "company_id": company_id,
                "job_id": job_id,
                "job_prefix": job.job_prefix,
                "project_title": job.project_title,
                "document_type": "estimate",
                "cost_code": row.cost_code,
                "row_type": row.row_type,
                "area": row.area,
                "task_scope": row.task_scope,
                "budgeted_total": row.budgeted_total,
                "actual_total": row.total
            }
            
            chunk = DocumentChunk(content=content, metadata=metadata)
            chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} estimate chunks")
        return chunks
    
    def create_comparison_chunks(
        self,
        comparison_data: Dict[str, ComparisonCategory],
        job: Job,
        company_id: str,
        job_id: str
    ) -> List[DocumentChunk]:
        """
        Create searchable chunks from budget comparison data.
        
        Args:
            comparison_data: Dictionary of comparison categories
            job: Job object for context
            company_id: Company ID
            job_id: Job ID (Firestore document ID)
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        for category_name, category in comparison_data.items():
            for row in category.rows:
                # Format tag amounts
                tag_info = []
                for tag in row.tags:
                    budgeted = row.tag_amounts.get(tag, 0.0)
                    consumed = row.consumed_tag_amounts.get(tag, 0.0)
                    tag_label = {
                        'alw': 'Allowance',
                        'est': 'Estimate',
                        'co': 'Change Order'
                    }.get(tag, tag.upper())
                    tag_info.append(f"{tag_label}: ${budgeted:.2f} budgeted, ${consumed:.2f} consumed")
                
                tag_summary = "\n".join(tag_info) if tag_info else "No tag breakdown"
                
                # Create content text
                content = f"""Job {job.job_prefix} - Budget Comparison
Project: {job.project_title}

Category: {category.category}
Cost Code: {row.cost_code}
Area: {row.area}

Budgeted Amount: ${row.budgeted_amount:.2f}
Consumed Amount: ${row.consumed_amount:.2f}
Difference: ${row.difference_amount:.2f}
Percentage Used: {row.percentage_used:.1f}%

Tag Breakdown:
{tag_summary}

Status: {"Over Budget" if row.percentage_used > 100 else "Under Budget" if row.percentage_used < 100 else "On Budget"}"""
                
                # Create metadata - CRITICAL: Use job_id parameter, not job.document_id
                metadata = {
                    "company_id": company_id,
                    "job_id": job_id,
                    "job_prefix": job.job_prefix,
                    "project_title": job.project_title,
                    "document_type": "comparison",
                    "category": category.category,
                    "cost_code": row.cost_code,
                    "area": row.area,
                    "tags": row.tags,
                    "budgeted_amount": row.budgeted_amount,
                    "consumed_amount": row.consumed_amount,
                    "percentage_used": row.percentage_used
                }
                
                chunk = DocumentChunk(content=content, metadata=metadata)
                chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} comparison chunks")
        return chunks
    
    def ingest_job(
        self,
        company_id: str,
        job_id: str,
        job_data: Job = None,
        comparison_data: Dict[str, ComparisonCategory] = None,
        fetch_from_firebase: bool = False
    ) -> Dict:
        """
        Complete ingestion pipeline for a job.
        
        Args:
            company_id: Company ID
            job_id: Job ID (the Firestore document ID)
            job_data: Optional job data (if not fetching from Firebase)
            comparison_data: Optional comparison data (if not fetching from Firebase)
            fetch_from_firebase: Whether to fetch data from Firebase
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Fetch from Firebase if requested
            if fetch_from_firebase:
                print(f"📥 Fetching job data from Firebase for job_id: {job_id}")
                job_data = self.firebase_service.fetch_job_data(company_id, job_id)
                comparison_data = self.firebase_service.fetch_comparison_data(company_id, job_id)
                
                if not job_data:
                    return {
                        "status": "error",
                        "message": f"Job not found in Firebase: {job_id}",
                        "chunks_created": 0
                    }
            
            if not job_data:
                return {
                    "status": "error",
                    "message": "No job data provided",
                    "chunks_created": 0
                }
            
            # Delete existing data for this job
            print(f"🗑️ Deleting existing data for job_id: {job_id}")
            deleted_count = self.qdrant_service.delete_by_job_id(job_id)
            
            # Create chunks - pass job_id explicitly
            all_chunks = []
            
            # Estimate chunks
            if job_data.estimate:
                estimate_chunks = self.create_estimate_chunks(job_data, company_id, job_id)
                all_chunks.extend(estimate_chunks)
            
            # Comparison chunks
            if comparison_data:
                comparison_chunks = self.create_comparison_chunks(comparison_data, job_data, company_id, job_id)
                all_chunks.extend(comparison_chunks)
            
            if not all_chunks:
                return {
                    "status": "warning",
                    "message": "No chunks created - job has no estimate or comparison data",
                    "chunks_created": 0
                }
            
            # Generate embeddings
            print(f"🔢 Generating embeddings for {len(all_chunks)} chunks...")
            texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_service.embed_batch(texts)
            
            # Insert into Qdrant
            print(f"💾 Inserting {len(all_chunks)} chunks into Qdrant...")
            inserted_count = self.qdrant_service.insert_vectors(all_chunks, embeddings)
            
            return {
                "status": "success",
                "message": f"Successfully ingested job {job_data.job_prefix}",
                "chunks_created": inserted_count,
                "deleted_count": deleted_count,
                "estimate_rows": len(job_data.estimate) if job_data.estimate else 0,
                "comparison_rows": sum(
                    len(cat.rows) for cat in comparison_data.values()
                ) if comparison_data else 0
            }
            
        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "chunks_created": 0
            }


# Global instance
_ingestion_service = None


def get_ingestion_service() -> IngestionService:
    """Get or create the global ingestion service instance."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service