"""
Database Initialization Module
Handles Firebase/Firestore and ChromaDB connections
UPDATED: OpenAI API v1.0+ compatible embeddings
"""

import os
import json
from typing import Dict, List, Any, Optional
import firebase_admin
from firebase_admin import credentials, firestore
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for database connections
firebase_db = None
chroma_client = None
chroma_collection = None


# ==========================================
# CUSTOM OPENAI EMBEDDING FUNCTION
# Compatible with openai>=1.0.0
# ==========================================

class CustomOpenAIEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using OpenAI API v1.0+
    Replaces ChromaDB's built-in OpenAIEmbeddingFunction which uses old API
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small"
    ):
        """
        Initialize the embedding function
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the embedding model
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for input documents
        
        Args:
            input: List of text documents
        
        Returns:
            List of embedding vectors
        """
        # Replace newlines with spaces (OpenAI recommendation)
        texts = [text.replace("\n", " ") for text in input]
        
        # Call OpenAI API v1.0+
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        
        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        
        return embeddings


# ==========================================
# FIREBASE INITIALIZATION
# ==========================================

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global firebase_db
    
    if firebase_db is not None:
        print("Firebase already initialized")
        return firebase_db
    
    try:
        # Create credentials from environment variables
        firebase_config = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        }
        
        # Initialize Firebase Admin
        cred = credentials.Certificate(firebase_config)
        
        # Check if Firebase is already initialized
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        
        firebase_db = firestore.client()
        print("✅ Firebase initialized successfully")
        return firebase_db
        
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")
        raise


# ==========================================
# CHROMADB INITIALIZATION
# UPDATED: Uses custom OpenAI embedding function
# ==========================================

def initialize_chromadb():
    """Initialize ChromaDB with persistence and custom OpenAI embeddings"""
    global chroma_client, chroma_collection
    
    if chroma_collection is not None:
        print("ChromaDB already initialized")
        return chroma_collection
    
    try:
        persist_path = os.getenv("CHROMA_PERSIST_PATH", "./chroma_storage")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "construction_rag")
        
        # FIXED: Use custom OpenAI embedding function compatible with openai>=1.0.0
        # This replaces the old chromadb.utils.embedding_functions.OpenAIEmbeddingFunction
        openai_ef = CustomOpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        # Initialize ChromaDB with persistence
        chroma_client = chromadb.PersistentClient(path=persist_path)
        
        # Get or create collection
        chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        
        print(f"✅ ChromaDB initialized successfully")
        print(f"📊 Current collection size: {chroma_collection.count()}")
        return chroma_collection
        
    except Exception as e:
        print(f"❌ Error initializing ChromaDB: {e}")
        raise


# ==========================================
# DATABASE GETTERS
# ==========================================

def get_firebase_db():
    """Get Firebase database instance"""
    if firebase_db is None:
        initialize_firebase()
    return firebase_db


def get_chroma_collection():
    """Get ChromaDB collection instance"""
    if chroma_collection is None:
        initialize_chromadb()
    return chroma_collection


# ==========================================
# FIREBASE DATA FETCHING
# ==========================================

async def fetch_job_consumed_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch consumed data for a specific job"""
    try:
        db = get_firebase_db()
        doc_ref = db.collection('companies').document(company_id).collection('jobs').document(job_id).collection('data').document('consumed')
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            data['company_id'] = company_id
            data['job_id'] = job_id
            data['data_type'] = 'consumed'
            return data
        else:
            print(f"No consumed data found for company {company_id}, job {job_id}")
            return None
            
    except Exception as e:
        print(f"Error fetching consumed job data: {e}")
        return None


async def fetch_job_complete_data(company_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch complete job data including estimate and schedule"""
    try:
        db = get_firebase_db()
        job_ref = db.collection('companies').document(company_id).collection('jobs').document(job_id)
        job_doc = job_ref.get()
        
        if not job_doc.exists:
            print(f"No job found for company {company_id}, job {job_id}")
            return None
        
        job_data = job_doc.to_dict()
        job_data['company_id'] = company_id
        job_data['job_id'] = job_id
        
        # Also get consumed data if it exists
        consumed_data = await fetch_job_consumed_data(company_id, job_id)
        if consumed_data:
            job_data['consumed_data'] = consumed_data
        
        return job_data
            
    except Exception as e:
        print(f"Error fetching complete job data: {e}")
        return None


# ==========================================
# DATA EXTRACTION HELPERS
# ==========================================

def extract_estimate_data(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract estimate data from complete job data with row numbers"""
    if not job_data or 'estimate' not in job_data:
        return None
    
    estimate_entries = job_data['estimate']
    if not estimate_entries or not isinstance(estimate_entries, list):
        return None
    
    # Add row numbers to each entry (1-indexed)
    entries_with_rows = []
    for idx, entry in enumerate(estimate_entries, start=1):
        entry_with_row = entry.copy()
        entry_with_row['row_number'] = idx
        entries_with_rows.append(entry_with_row)
    
    return {
        'company_id': job_data['company_id'],
        'job_id': job_data['job_id'],
        'job_name': job_data.get('projectTitle', 'Unknown Job'),
        'data_type': 'estimate',
        'entries': entries_with_rows,
        'total_rows': len(entries_with_rows),
        'last_updated': job_data.get('createdDate', ''),
        'project_description': job_data.get('projectDescription', ''),
        'client_name': job_data.get('clientName', ''),
        'site_location': f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', '),
        'estimate_type': job_data.get('estimateType', 'general')
    }


def extract_flooring_estimate_data(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract flooring estimate data from complete job data with row numbers"""
    if not job_data or 'flooringEstimateData' not in job_data:
        return None
    
    flooring_entries = job_data['flooringEstimateData']
    if not flooring_entries or not isinstance(flooring_entries, list):
        return None
    
    # Add row numbers to each entry
    entries_with_rows = []
    for idx, entry in enumerate(flooring_entries, start=1):
        entry_with_row = entry.copy()
        entry_with_row['row_number'] = idx
        entries_with_rows.append(entry_with_row)
    
    return {
        'company_id': job_data['company_id'],
        'job_id': job_data['job_id'],
        'job_name': job_data.get('projectTitle', 'Unknown Job'),
        'data_type': 'flooring_estimate',
        'entries': entries_with_rows,
        'total_rows': len(entries_with_rows),
        'last_updated': job_data.get('createdDate', ''),
        'project_description': job_data.get('projectDescription', ''),
        'client_name': job_data.get('clientName', ''),
        'site_location': f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', ')
    }


def extract_schedule_data(job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract schedule data from complete job data with row numbers"""
    if not job_data or 'schedule' not in job_data:
        return None
    
    schedule_entries = job_data['schedule']
    if not schedule_entries or not isinstance(schedule_entries, list):
        return None
    
    # Add row numbers to each entry
    entries_with_rows = []
    for idx, entry in enumerate(schedule_entries, start=1):
        entry_with_row = entry.copy()
        entry_with_row['row_number'] = idx
        entries_with_rows.append(entry_with_row)
    
    return {
        'company_id': job_data['company_id'],
        'job_id': job_data['job_id'],
        'job_name': job_data.get('projectTitle', 'Unknown Job'),
        'data_type': 'schedule',
        'entries': entries_with_rows,
        'total_rows': len(entries_with_rows),
        'last_updated': job_data.get('createdDate', ''),
        'project_description': job_data.get('projectDescription', ''),
        'client_name': job_data.get('clientName', ''),
        'site_location': f"{job_data.get('siteCity', '')}, {job_data.get('siteState', '')}".strip(', '),
        'schedule_last_updated': job_data.get('scheduleLastUpdated', '')
    }


# ==========================================
# MAIN INITIALIZATION
# ==========================================

if __name__ == "__main__":
    print("🧪 Testing database connections...\n")
    
    # Test Firebase
    print("Testing Firebase...")
    try:
        initialize_firebase()
        print("✅ Firebase connection successful\n")
    except Exception as e:
        print(f"❌ Firebase connection failed: {e}\n")
    
    # Test ChromaDB
    print("Testing ChromaDB...")
    try:
        initialize_chromadb()
        print("✅ ChromaDB connection successful\n")
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}\n")
    
    print("✅ Database tests complete")