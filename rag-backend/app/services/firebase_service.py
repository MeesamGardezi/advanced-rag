"""
Firebase service for fetching job data from Firestore.
"""

import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, Optional
from app.config import settings
from app.models import Job, ComparisonCategory


class FirebaseService:
    """Service for interacting with Firebase Firestore."""
    
    def __init__(self):
        """Initialize Firebase admin SDK."""
        self.db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase with credentials."""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                print(f"Initializing Firebase with credentials from: {settings.firebase_credentials_path}")
                cred = credentials.Certificate(settings.firebase_credentials_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': settings.firebase_database_url
                })
                print("✅ Firebase initialized successfully")
            
            self.db = firestore.client()
            
        except Exception as e:
            print(f"❌ Error initializing Firebase: {e}")
            raise
    
    def fetch_job_data(self, company_id: str, job_id: str) -> Optional[Job]:
        """
        Fetch job data from Firestore.
        
        Args:
            company_id: Company ID
            job_id: Job ID
            
        Returns:
            Job object or None if not found
        """
        try:
            # Path: companies/{company_id}/jobs/{job_id}
            doc_ref = self.db.collection('companies').document(company_id).collection('jobs').document(job_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                print(f"❌ Job not found: {job_id}")
                return None
            
            job_data = doc.to_dict()
            
            # Add document_id from Firestore document ID if not present
            if 'documentId' not in job_data and 'document_id' not in job_data:
                job_data['documentId'] = job_id
            
            # Handle address field variations
            if 'siteAddress' not in job_data and 'street' in job_data:
                job_data['siteAddress'] = job_data['street']
            
            # Convert to Job model
            job = Job(**job_data)
            print(f"✅ Fetched job data: {job.job_prefix} - {job.project_title}")
            return job
            
        except Exception as e:
            print(f"❌ Error fetching job data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def fetch_comparison_data(
        self,
        company_id: str,
        job_id: str
    ) -> Optional[Dict[str, ComparisonCategory]]:
        """
        Fetch budget comparison data from Firestore.
        
        Args:
            company_id: Company ID
            job_id: Job ID
            
        Returns:
            Dictionary of comparison categories or None if not found
        """
        try:
            # Path: companies/{company_id}/jobs/{job_id}/comparisons
            comparisons_ref = (
                self.db.collection('companies')
                .document(company_id)
                .collection('jobs')
                .document(job_id)
                .collection('comparisons')
            )
            
            docs = comparisons_ref.stream()
            
            comparison_data = {}
            for doc in docs:
                category_data = doc.to_dict()
                category = ComparisonCategory(**category_data)
                comparison_data[doc.id] = category
            
            if comparison_data:
                print(f"✅ Fetched comparison data: {len(comparison_data)} categories")
            else:
                print(f"ℹ️ No comparison data found for job: {job_id}")
            
            return comparison_data if comparison_data else None
            
        except Exception as e:
            print(f"❌ Error fetching comparison data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def health_check(self) -> bool:
        """Check if Firebase connection is healthy."""
        try:
            # Try to access Firestore
            self.db.collection('_health_check').limit(1).get()
            return True
        except:
            return False


# Global instance
_firebase_service = None


def get_firebase_service() -> FirebaseService:
    """Get or create the global Firebase service instance."""
    global _firebase_service
    if _firebase_service is None:
        _firebase_service = FirebaseService()
    return _firebase_service