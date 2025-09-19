/**
 * Firebase Configuration
 * Centralized Firebase settings and collection paths
 */

import dotenv from 'dotenv';

dotenv.config();

// Firebase connection configuration
export const firebaseConfig = {
  credentials: {
    type: "service_account",
    project_id: process.env.FIREBASE_PROJECT_ID,
    private_key_id: process.env.FIREBASE_PRIVATE_KEY_ID,
    private_key: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
    client_email: process.env.FIREBASE_CLIENT_EMAIL,
    client_id: process.env.FIREBASE_CLIENT_ID,
    auth_uri: process.env.FIREBASE_AUTH_URI || "https://accounts.google.com/o/oauth2/auth",
    token_uri: process.env.FIREBASE_TOKEN_URI || "https://oauth2.googleapis.com/token",
    auth_provider_x509_cert_url: process.env.FIREBASE_AUTH_PROVIDER_CERT_URL || "https://www.googleapis.com/oauth2/v1/certs",
    client_x509_cert_url: process.env.FIREBASE_CLIENT_CERT_URL
  },
  
  // Database settings
  databaseURL: process.env.FIREBASE_DATABASE_URL,
  
  // App settings
  options: {
    timestampsInSnapshots: true
  }
};

// Collection structure configuration
export const collectionsConfig = {
  // Main collections
  companies: 'companies',
  
  // Sub-collections under companies/{companyId}/
  jobs: 'jobs',
  
  // Sub-collections under companies/{companyId}/jobs/{jobId}/
  data: 'data',
  
  // Documents under data collection
  consumed: 'consumed',
  
  // Field mappings for different document types
  fieldMappings: {
    // Job document fields
    job: {
      id: 'id',
      title: 'projectTitle',
      client: 'clientName',
      status: 'status',
      created: 'createdDate',
      updated: 'updatedDate',
      description: 'projectDescription',
      location: {
        street: 'siteStreet',
        city: 'siteCity', 
        state: 'siteState',
        zip: 'siteZip'
      },
      estimate: 'estimate', // Array of estimate items
      schedule: 'schedule'  // Array of schedule items
    },
    
    // Estimate item fields (EstimateRow)
    estimate: {
      area: 'area',
      taskScope: 'taskScope',
      costCode: 'costCode',
      description: 'description',
      units: 'units',
      quantity: 'qty',
      rate: 'rate',
      total: 'total',
      budgetedRate: 'budgetedRate',
      budgetedTotal: 'budgetedTotal',
      notes: 'notesRemarks',
      type: 'rowType',
      materials: 'materials'
    },
    
    // Consumed data fields
    consumed: {
      entries: 'entries',
      lastUpdated: 'lastUpdated',
      // Individual entry fields
      entry: {
        job: 'job',
        costCode: 'costCode',
        amount: 'amount',
        date: 'date'
      }
    }
  }
};

// Query configuration
export const queryConfig = {
  // Batch processing limits
  maxBatchSize: 500,
  batchTimeout: 30000, // 30 seconds
  
  // Retry configuration
  maxRetries: 3,
  retryDelay: 1000, // Start with 1 second
  retryBackoff: 2,   // Double delay each retry
  
  // Pagination
  defaultPageSize: 100,
  maxPageSize: 1000,
  
  // Timeouts
  readTimeout: 30000,
  writeTimeout: 60000,
  
  // Real-time listener settings
  listenerReconnectDelay: 5000,
  maxListenerRetries: 5
};

// Data validation rules
export const validationRules = {
  // Required fields for different document types
  requiredFields: {
    job: ['projectTitle', 'clientName'],
    estimate: ['costCode'],
    consumed: ['costCode', 'amount']
  },
  
  // Field validation patterns
  patterns: {
    costCode: /^[A-Z0-9\s\-_()]+$/i,
    email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    currency: /^\d+(\.\d{1,2})?$/
  },
  
  // Value constraints
  constraints: {
    maxStringLength: 1000,
    maxArrayLength: 500,
    maxNumberValue: 999999999.99,
    minNumberValue: 0
  }
};

// Sync configuration
export const syncConfig = {
  // Real-time sync settings
  enableRealtime: process.env.FIREBASE_REALTIME_ENABLED === 'true',
  realtimeCollections: ['jobs'], // Which collections to watch
  
  // Sync intervals
  fullSyncInterval: 24 * 60 * 60 * 1000, // 24 hours
  incrementalSyncInterval: 60 * 60 * 1000, // 1 hour
  
  // Change detection
  trackChanges: true,
  changeRetentionDays: 7,
  
  // Sync strategies
  strategies: {
    bulk: 'full_sync',      // Complete data refresh
    incremental: 'delta_sync', // Only changed data
    realtime: 'live_sync'   // Real-time listeners
  },
  
  // Performance settings
  concurrentConnections: 5,
  rateLimitRpm: 100, // Requests per minute
  
  // Error handling
  errorRetryAttempts: 3,
  errorRetryDelay: 2000,
  skipOnError: false // Continue sync even if some documents fail
};

// Caching configuration
export const cacheConfig = {
  // Enable caching
  enabled: process.env.NODE_ENV !== 'test',
  
  // Cache TTL (time to live) in milliseconds
  ttl: {
    companies: 5 * 60 * 1000,    // 5 minutes
    jobs: 2 * 60 * 1000,        // 2 minutes
    estimates: 30 * 60 * 1000,   // 30 minutes
    consumed: 10 * 60 * 1000     // 10 minutes
  },
  
  // Cache size limits
  maxSize: {
    companies: 100,
    jobs: 1000,
    documents: 5000
  },
  
  // Cache strategies
  strategy: 'lru', // Least Recently Used
  
  // Invalidation rules
  invalidateOnWrite: true,
  invalidateOnError: false
};

// Security configuration
export const securityConfig = {
  // Authentication
  requireAuth: process.env.NODE_ENV === 'production',
  
  // Access control
  readOnlyMode: process.env.FIREBASE_READ_ONLY === 'true',
  
  // Data sanitization
  sanitizeInput: true,
  stripHtml: true,
  
  // Rate limiting per IP/user
  rateLimit: {
    enabled: process.env.NODE_ENV === 'production',
    requests: 1000,     // Max requests
    windowMs: 60000,    // Per minute
    skipFailedRequests: true
  },
  
  // Allowed operations
  allowedOperations: ['read', 'write', 'delete'],
  restrictedCollections: [] // Collections requiring special permissions
};

// Get Firebase configuration with validation
export function getFirebaseConfig() {
  validateFirebaseConfig();
  
  return {
    ...firebaseConfig,
    collections: collectionsConfig,
    query: queryConfig,
    validation: validationRules,
    sync: syncConfig,
    cache: cacheConfig,
    security: securityConfig
  };
}

// Validate Firebase configuration
export function validateFirebaseConfig() {
  const requiredEnvVars = [
    'FIREBASE_PROJECT_ID',
    'FIREBASE_PRIVATE_KEY',
    'FIREBASE_CLIENT_EMAIL'
  ];
  
  const missing = requiredEnvVars.filter(varName => !process.env[varName]);
  
  if (missing.length > 0) {
    throw new Error(`Missing required Firebase environment variables: ${missing.join(', ')}`);
  }
  
  // Validate project ID format
  const projectId = process.env.FIREBASE_PROJECT_ID;
  if (projectId && !/^[a-z0-9\-]+$/.test(projectId)) {
    throw new Error('Invalid Firebase project ID format');
  }
  
  // Validate private key format
  const privateKey = process.env.FIREBASE_PRIVATE_KEY;
  if (privateKey && !privateKey.includes('BEGIN PRIVATE KEY')) {
    console.warn('⚠️  Firebase private key may be incorrectly formatted');
  }
  
  // Validate email format
  const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
  if (clientEmail && !validationRules.patterns.email.test(clientEmail)) {
    throw new Error('Invalid Firebase client email format');
  }
  
  console.log('✅ Firebase configuration validated');
  return true;
}

// Helper function to build collection paths
export function buildCollectionPath(companyId, jobId = null) {
  let path = `${collectionsConfig.companies}/${companyId}`;
  
  if (jobId) {
    path += `/${collectionsConfig.jobs}/${jobId}`;
  }
  
  return path;
}

// Helper function to build document paths
export function buildDocumentPath(companyId, jobId, collection, document) {
  const basePath = buildCollectionPath(companyId, jobId);
  return `${basePath}/${collection}/${document}`;
}

// Get collection reference string
export function getCollectionRef(type, companyId, jobId = null) {
  switch (type) {
    case 'companies':
      return collectionsConfig.companies;
    case 'jobs':
      return `${collectionsConfig.companies}/${companyId}/${collectionsConfig.jobs}`;
    case 'consumed':
      return `${collectionsConfig.companies}/${companyId}/${collectionsConfig.jobs}/${jobId}/${collectionsConfig.data}`;
    default:
      throw new Error(`Unknown collection type: ${type}`);
  }
}

// Estimate data extraction helper
export function extractEstimateFields(estimateItem) {
  const mapping = collectionsConfig.fieldMappings.estimate;
  
  return {
    area: estimateItem[mapping.area] || '',
    taskScope: estimateItem[mapping.taskScope] || '',
    costCode: estimateItem[mapping.costCode] || '',
    description: estimateItem[mapping.description] || '',
    units: estimateItem[mapping.units] || '',
    quantity: parseFloat(estimateItem[mapping.quantity]) || 0,
    rate: parseFloat(estimateItem[mapping.rate]) || 0,
    total: parseFloat(estimateItem[mapping.total]) || 0,
    budgetedRate: parseFloat(estimateItem[mapping.budgetedRate]) || 0,
    budgetedTotal: parseFloat(estimateItem[mapping.budgetedTotal]) || 0,
    notes: estimateItem[mapping.notes] || '',
    type: estimateItem[mapping.type] || 'estimate',
    materials: estimateItem[mapping.materials] || []
  };
}

// Consumed data extraction helper
export function extractConsumedFields(consumedItem) {
  const mapping = collectionsConfig.fieldMappings.consumed.entry;
  
  return {
    job: consumedItem[mapping.job] || '',
    costCode: consumedItem[mapping.costCode] || '',
    amount: parseFloat(consumedItem[mapping.amount]) || 0,
    date: consumedItem[mapping.date] || ''
  };
}

export default {
  getFirebaseConfig,
  validateFirebaseConfig,
  buildCollectionPath,
  buildDocumentPath,
  getCollectionRef,
  extractEstimateFields,
  extractConsumedFields,
  firebaseConfig,
  collectionsConfig,
  queryConfig,
  validationRules,
  syncConfig,
  cacheConfig,
  securityConfig
};