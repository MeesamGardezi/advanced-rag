/**
 * Firebase Integration Service
 * Connects to existing Firebase and fetches estimate/consumed data
 */

import admin from 'firebase-admin';
import dotenv from 'dotenv';

dotenv.config();

class FirebaseService {
  constructor() {
    this.db = null;
    this.initialized = false;
    this.initialize();
  }

  initialize() {
    try {
      // Create Firebase credentials from environment
      const firebaseConfig = {
        type: "service_account",
        project_id: process.env.FIREBASE_PROJECT_ID,
        private_key_id: process.env.FIREBASE_PRIVATE_KEY_ID,
        private_key: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, '\n'),
        client_email: process.env.FIREBASE_CLIENT_EMAIL,
        client_id: process.env.FIREBASE_CLIENT_ID,
        auth_uri: "https://accounts.google.com/o/oauth2/auth",
        token_uri: "https://oauth2.googleapis.com/token",
      };

      // Initialize Firebase Admin SDK
      if (!admin.apps.length) {
        admin.initializeApp({
          credential: admin.credential.cert(firebaseConfig)
        });
      }

      this.db = admin.firestore();
      this.initialized = true;
      
      console.log('✅ Firebase initialized successfully');
      console.log('🔥 Project ID:', process.env.FIREBASE_PROJECT_ID);
      
    } catch (error) {
      console.error('❌ Firebase initialization failed:', error.message);
      throw error;
    }
  }

  // Test Firebase connection
  async testConnection() {
    try {
      const testDoc = await this.db.collection('_test').doc('connection').get();
      console.log('✅ Firebase connection test successful');
      return true;
    } catch (error) {
      console.error('❌ Firebase connection test failed:', error.message);
      return false;
    }
  }

  // Get all companies
  async getAllCompanies() {
    try {
      const companiesSnapshot = await this.db.collection('companies').get();
      const companies = [];
      
      companiesSnapshot.forEach(doc => {
        companies.push({
          id: doc.id,
          ...doc.data()
        });
      });
      
      console.log(`📊 Found ${companies.length} companies`);
      return companies;
      
    } catch (error) {
      console.error('❌ Error fetching companies:', error.message);
      throw error;
    }
  }

  // Get all jobs for a company
  async getCompanyJobs(companyId) {
    try {
      const jobsSnapshot = await this.db
        .collection('companies')
        .doc(companyId)
        .collection('jobs')
        .get();
      
      const jobs = [];
      jobsSnapshot.forEach(doc => {
        jobs.push({
          id: doc.id,
          companyId: companyId,
          ...doc.data()
        });
      });
      
      console.log(`📋 Found ${jobs.length} jobs for company ${companyId}`);
      return jobs;
      
    } catch (error) {
      console.error(`❌ Error fetching jobs for company ${companyId}:`, error.message);
      throw error;
    }
  }

  // Get estimate data for a specific job
  async getJobEstimates(companyId, jobId) {
    try {
      const jobDoc = await this.db
        .collection('companies')
        .doc(companyId)
        .collection('jobs')
        .doc(jobId)
        .get();

      if (!jobDoc.exists) {
        console.log(`⚠️  Job ${jobId} not found`);
        return null;
      }

      const jobData = jobDoc.data();
      const estimates = jobData.estimate || [];
      
      console.log(`📊 Found ${estimates.length} estimate items for job ${jobId}`);
      
      // Return with metadata
      return {
        companyId,
        jobId,
        jobTitle: jobData.projectTitle || 'Unknown Project',
        clientName: jobData.clientName || 'Unknown Client',
        estimates: estimates,
        metadata: {
          createdDate: jobData.createdDate,
          status: jobData.status,
          siteCity: jobData.siteCity,
          siteState: jobData.siteState
        }
      };
      
    } catch (error) {
      console.error(`❌ Error fetching estimates for job ${jobId}:`, error.message);
      throw error;
    }
  }

  // Get consumed data for a specific job
  async getJobConsumed(companyId, jobId) {
    try {
      const consumedDoc = await this.db
        .collection('companies')
        .doc(companyId)
        .collection('jobs')
        .doc(jobId)
        .collection('data')
        .doc('consumed')
        .get();

      if (!consumedDoc.exists) {
        console.log(`⚠️  No consumed data found for job ${jobId}`);
        return null;
      }

      const consumedData = consumedDoc.data();
      const consumedEntries = consumedData.entries || [];
      
      console.log(`💰 Found ${consumedEntries.length} consumed items for job ${jobId}`);
      
      return {
        companyId,
        jobId,
        entries: consumedEntries,
        metadata: {
          lastUpdated: consumedData.lastUpdated,
        }
      };
      
    } catch (error) {
      console.error(`❌ Error fetching consumed data for job ${jobId}:`, error.message);
      throw error;
    }
  }

  // Get complete job data (estimates + consumed)
  async getCompleteJobData(companyId, jobId) {
    try {
      console.log(`🔄 Fetching complete data for job ${jobId}...`);
      
      const [estimates, consumed] = await Promise.all([
        this.getJobEstimates(companyId, jobId),
        this.getJobConsumed(companyId, jobId)
      ]);

      if (!estimates) {
        console.log(`⚠️  No job data found for ${jobId}`);
        return null;
      }

      return {
        ...estimates,
        consumedData: consumed
      };
      
    } catch (error) {
      console.error(`❌ Error fetching complete job data for ${jobId}:`, error.message);
      throw error;
    }
  }

  // Get ALL estimate data from Firebase (for bulk sync)
  async getAllEstimateData(companyId = null) {
    try {
      console.log('🔄 Fetching ALL estimate data from Firebase...');
      
      const companies = companyId ? [{ id: companyId }] : await this.getAllCompanies();
      const allEstimates = [];
      
      for (const company of companies) {
        console.log(`📊 Processing company: ${company.id}`);
        
        const jobs = await this.getCompanyJobs(company.id);
        
        for (const job of jobs) {
          const jobData = await this.getJobEstimates(company.id, job.id);
          
          if (jobData && jobData.estimates.length > 0) {
            allEstimates.push(jobData);
          }
        }
      }
      
      console.log(`✅ Fetched estimate data from ${allEstimates.length} jobs`);
      return allEstimates;
      
    } catch (error) {
      console.error('❌ Error fetching all estimate data:', error.message);
      throw error;
    }
  }

  // Get ALL consumed data from Firebase (for bulk sync)
  async getAllConsumedData(companyId = null) {
    try {
      console.log('🔄 Fetching ALL consumed data from Firebase...');
      
      const companies = companyId ? [{ id: companyId }] : await this.getAllCompanies();
      const allConsumed = [];
      
      for (const company of companies) {
        console.log(`💰 Processing company: ${company.id}`);
        
        const jobs = await this.getCompanyJobs(company.id);
        
        for (const job of jobs) {
          const consumedData = await this.getJobConsumed(company.id, job.id);
          
          if (consumedData && consumedData.entries.length > 0) {
            allConsumed.push(consumedData);
          }
        }
      }
      
      console.log(`✅ Fetched consumed data from ${allConsumed.length} jobs`);
      return allConsumed;
      
    } catch (error) {
      console.error('❌ Error fetching all consumed data:', error.message);
      throw error;
    }
  }

  // Real-time listener for job changes
  setupJobListener(companyId, jobId, callback) {
    try {
      const jobRef = this.db
        .collection('companies')
        .doc(companyId)
        .collection('jobs')
        .doc(jobId);

      const unsubscribe = jobRef.onSnapshot(doc => {
        if (doc.exists) {
          console.log(`🔄 Job ${jobId} updated in real-time`);
          callback({
            companyId,
            jobId,
            data: doc.data(),
            timestamp: new Date()
          });
        }
      }, error => {
        console.error('❌ Real-time listener error:', error.message);
      });

      console.log(`👂 Listening for changes to job ${jobId}`);
      return unsubscribe;
      
    } catch (error) {
      console.error('❌ Error setting up job listener:', error.message);
      throw error;
    }
  }

  // Batch sync - get multiple jobs efficiently
  async syncMultipleJobs(jobsList) {
    try {
      console.log(`🔄 Syncing ${jobsList.length} jobs...`);
      
      const results = await Promise.allSettled(
        jobsList.map(({ companyId, jobId }) => 
          this.getCompleteJobData(companyId, jobId)
        )
      );

      const successful = results
        .filter(result => result.status === 'fulfilled' && result.value !== null)
        .map(result => result.value);

      const failed = results
        .filter(result => result.status === 'rejected')
        .map(result => result.reason);

      console.log(`✅ Successfully synced ${successful.length} jobs`);
      if (failed.length > 0) {
        console.log(`⚠️  Failed to sync ${failed.length} jobs`);
      }

      return {
        successful,
        failed: failed.length,
        total: jobsList.length
      };
      
    } catch (error) {
      console.error('❌ Error in batch sync:', error.message);
      throw error;
    }
  }

  // Get Firebase statistics
  async getFirebaseStats() {
    try {
      const companies = await this.getAllCompanies();
      let totalJobs = 0;
      let totalEstimates = 0;
      let totalConsumed = 0;

      for (const company of companies) {
        const jobs = await this.getCompanyJobs(company.id);
        totalJobs += jobs.length;

        for (const job of jobs) {
          const estimates = await this.getJobEstimates(company.id, job.id);
          const consumed = await this.getJobConsumed(company.id, job.id);
          
          if (estimates) totalEstimates += estimates.estimates.length;
          if (consumed) totalConsumed += consumed.entries.length;
        }
      }

      return {
        companies: companies.length,
        jobs: totalJobs,
        estimateItems: totalEstimates,
        consumedItems: totalConsumed
      };
      
    } catch (error) {
      console.error('❌ Error getting Firebase stats:', error.message);
      throw error;
    }
  }
}

// Export singleton instance
const firebase = new FirebaseService();
export default firebase;