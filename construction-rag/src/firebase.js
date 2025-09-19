/**
 * Firebase Integration Service - Estimate Focused
 * Connects to existing Firebase and fetches construction estimate data only
 */

import admin from 'firebase-admin';
import dotenv from 'dotenv';

dotenv.config();

class EstimateFocusedFirebaseService {
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
      
      console.log('✅ Firebase initialized successfully (Estimate-focused)');
      console.log('🔥 Project ID:', process.env.FIREBASE_PROJECT_ID);
      console.log('📊 Data Focus: Construction Estimates Only');
      
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
        const jobData = doc.data();
        jobs.push({
          id: doc.id,
          companyId: companyId,
          hasEstimates: jobData.estimate && Array.isArray(jobData.estimate) && jobData.estimate.length > 0,
          estimateCount: jobData.estimate ? jobData.estimate.length : 0,
          ...jobData
        });
      });
      
      console.log(`📋 Found ${jobs.length} jobs for company ${companyId}`);
      console.log(`📊 Jobs with estimates: ${jobs.filter(j => j.hasEstimates).length}`);
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
      
      // Analyze estimate data quality
      const validEstimates = estimates.filter(est => 
        est && (est.costCode || est.description) && 
        (est.total !== undefined || est.amount !== undefined)
      );
      
      if (validEstimates.length !== estimates.length) {
        console.log(`⚠️  ${estimates.length - validEstimates.length} estimates have data quality issues`);
      }

      // Return comprehensive job data with estimates
      return {
        companyId,
        jobId,
        documentId: jobId, // For compatibility
        projectTitle: jobData.projectTitle || jobData.title || 'Unknown Project',
        jobTitle: jobData.projectTitle || jobData.title || 'Unknown Project',
        clientName: jobData.clientName || jobData.client || 'Unknown Client',
        estimates: estimates,
        validEstimateCount: validEstimates.length,
        
        // Enhanced metadata for better processing
        metadata: {
          createdDate: jobData.createdDate,
          updatedDate: jobData.updatedDate,
          status: jobData.status,
          
          // Location information
          siteStreet: jobData.siteStreet,
          siteCity: jobData.siteCity,
          siteState: jobData.siteState,
          siteZip: jobData.siteZip,
          location: this.formatJobLocation(jobData),
          
          // Project details
          projectDescription: jobData.projectDescription,
          schedule: jobData.schedule,
          
          // Estimate summary statistics
          estimateStats: this.calculateEstimateStats(estimates),
          
          // Data quality indicators
          dataQuality: {
            totalItems: estimates.length,
            validItems: validEstimates.length,
            completionRate: estimates.length > 0 ? (validEstimates.length / estimates.length) * 100 : 0,
            hasDescriptions: estimates.filter(e => e.description && e.description.trim()).length,
            hasCostCodes: estimates.filter(e => e.costCode && e.costCode.trim()).length,
            hasAmounts: estimates.filter(e => e.total > 0 || e.amount > 0).length
          },
          
          // Processing metadata
          fetchedAt: new Date().toISOString(),
          source: 'firebase'
        }
      };
      
    } catch (error) {
      console.error(`❌ Error fetching estimates for job ${jobId}:`, error.message);
      throw error;
    }
  }

  // Get job estimates with enhanced error handling and retry logic
  async getJobEstimatesWithRetry(companyId, jobId, maxRetries = 3) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await this.getJobEstimates(companyId, jobId);
      } catch (error) {
        lastError = error;
        console.log(`⚠️  Attempt ${attempt}/${maxRetries} failed for job ${jobId}: ${error.message}`);
        
        if (attempt < maxRetries) {
          // Exponential backoff: 1s, 2s, 4s
          const delay = Math.pow(2, attempt - 1) * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw new Error(`Failed to fetch job ${jobId} after ${maxRetries} attempts: ${lastError.message}`);
  }

  // Get ALL estimate data from Firebase (for bulk sync)
  async getAllEstimateData(companyId = null) {
    try {
      console.log('🔄 Fetching ALL estimate data from Firebase...');
      
      const companies = companyId ? [{ id: companyId }] : await this.getAllCompanies();
      const allEstimates = [];
      let totalJobs = 0;
      let jobsWithEstimates = 0;
      let totalEstimateItems = 0;
      
      for (const company of companies) {
        console.log(`📊 Processing company: ${company.id}`);
        
        const jobs = await this.getCompanyJobs(company.id);
        totalJobs += jobs.length;
        
        // Filter jobs that have estimates
        const jobsWithEstimateData = jobs.filter(job => job.hasEstimates);
        console.log(`📋 ${jobsWithEstimateData.length}/${jobs.length} jobs have estimate data`);
        
        for (const job of jobsWithEstimateData) {
          try {
            const jobData = await this.getJobEstimatesWithRetry(company.id, job.id);
            
            if (jobData && jobData.estimates.length > 0) {
              allEstimates.push(jobData);
              jobsWithEstimates++;
              totalEstimateItems += jobData.estimates.length;
              
              console.log(`✅ ${job.id}: ${jobData.estimates.length} estimates (${jobData.projectTitle || 'Untitled'})`);
            }
          } catch (error) {
            console.error(`❌ Failed to fetch estimates for job ${job.id}:`, error.message);
            // Continue with other jobs
          }
        }
      }
      
      console.log(`✅ Estimate data collection complete:`);
      console.log(`   📈 Total companies: ${companies.length}`);
      console.log(`   📈 Total jobs scanned: ${totalJobs}`);
      console.log(`   📈 Jobs with estimates: ${jobsWithEstimates}`);
      console.log(`   📈 Total estimate items: ${totalEstimateItems}`);
      console.log(`   📈 Average items per job: ${jobsWithEstimates > 0 ? Math.round(totalEstimateItems / jobsWithEstimates) : 0}`);
      
      return allEstimates;
      
    } catch (error) {
      console.error('❌ Error fetching all estimate data:', error.message);
      throw error;
    }
  }

  // Real-time listener for job estimate changes
  setupJobEstimateListener(companyId, jobId, callback) {
    try {
      const jobRef = this.db
        .collection('companies')
        .doc(companyId)
        .collection('jobs')
        .doc(jobId);

      const unsubscribe = jobRef.onSnapshot(doc => {
        if (doc.exists) {
          const jobData = doc.data();
          const estimates = jobData.estimate || [];
          
          console.log(`🔄 Job ${jobId} estimate data updated in real-time (${estimates.length} items)`);
          
          callback({
            companyId,
            jobId,
            estimates,
            estimateCount: estimates.length,
            projectTitle: jobData.projectTitle,
            clientName: jobData.clientName,
            timestamp: new Date(),
            changeType: 'estimate_update'
          });
        }
      }, error => {
        console.error('❌ Real-time listener error:', error.message);
      });

      console.log(`👂 Listening for estimate changes to job ${jobId}`);
      return unsubscribe;
      
    } catch (error) {
      console.error('❌ Error setting up job estimate listener:', error.message);
      throw error;
    }
  }

  // Batch sync - get multiple jobs' estimates efficiently
  async syncMultipleJobEstimates(jobsList) {
    try {
      console.log(`🔄 Syncing estimates for ${jobsList.length} jobs...`);
      
      const results = await Promise.allSettled(
        jobsList.map(({ companyId, jobId }) => 
          this.getJobEstimatesWithRetry(companyId, jobId)
        )
      );

      const successful = results
        .filter(result => result.status === 'fulfilled' && result.value !== null)
        .map(result => result.value);

      const failed = results
        .filter(result => result.status === 'rejected')
        .map(result => result.reason);

      console.log(`✅ Successfully synced estimates for ${successful.length} jobs`);
      if (failed.length > 0) {
        console.log(`⚠️  Failed to sync estimates for ${failed.length} jobs`);
      }

      return {
        successful,
        failed: failed.length,
        total: jobsList.length,
        totalEstimateItems: successful.reduce((sum, job) => sum + (job.estimates?.length || 0), 0),
        averageEstimatesPerJob: successful.length > 0 
          ? successful.reduce((sum, job) => sum + (job.estimates?.length || 0), 0) / successful.length 
          : 0
      };
      
    } catch (error) {
      console.error('❌ Error in batch estimate sync:', error.message);
      throw error;
    }
  }

  // Get Firebase estimate statistics
  async getFirebaseEstimateStats() {
    try {
      const companies = await this.getAllCompanies();
      let totalJobs = 0;
      let jobsWithEstimates = 0;
      let totalEstimateItems = 0;
      let totalEstimatedValue = 0;
      let totalBudgetedValue = 0;
      
      const categoryCounts = {};
      const costCodeCounts = {};

      for (const company of companies) {
        const jobs = await this.getCompanyJobs(company.id);
        totalJobs += jobs.length;

        for (const job of jobs) {
          if (job.hasEstimates) {
            jobsWithEstimates++;
            totalEstimateItems += job.estimateCount;
            
            // Get detailed estimate data for value calculation
            try {
              const jobData = await this.getJobEstimates(company.id, job.id);
              if (jobData && jobData.estimates) {
                jobData.estimates.forEach(estimate => {
                  // Sum up values
                  const amount = parseFloat(estimate.total || estimate.amount || 0);
                  const budgeted = parseFloat(estimate.budgetedTotal || estimate.budgeted || 0);
                  
                  totalEstimatedValue += amount;
                  totalBudgetedValue += budgeted;
                  
                  // Count categories and cost codes
                  const category = this.categorizeEstimate(estimate);
                  categoryCounts[category] = (categoryCounts[category] || 0) + 1;
                  
                  if (estimate.costCode) {
                    costCodeCounts[estimate.costCode] = (costCodeCounts[estimate.costCode] || 0) + 1;
                  }
                });
              }
            } catch (error) {
              console.warn(`⚠️  Could not fetch detailed data for job ${job.id}`);
            }
          }
        }
      }

      const stats = {
        companies: companies.length,
        totalJobs,
        jobsWithEstimates,
        jobsWithoutEstimates: totalJobs - jobsWithEstimates,
        estimateDataCoverage: totalJobs > 0 ? ((jobsWithEstimates / totalJobs) * 100).toFixed(1) : 0,
        
        estimateItems: totalEstimateItems,
        averageEstimatesPerJob: jobsWithEstimates > 0 ? Math.round(totalEstimateItems / jobsWithEstimates) : 0,
        
        financialSummary: {
          totalEstimatedValue,
          totalBudgetedValue,
          overallVariance: totalEstimatedValue - totalBudgetedValue,
          overallVariancePercent: totalBudgetedValue > 0 
            ? ((totalEstimatedValue - totalBudgetedValue) / totalBudgetedValue * 100).toFixed(1)
            : 0
        },
        
        categoryDistribution: Object.entries(categoryCounts)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 10),
        
        topCostCodes: Object.entries(costCodeCounts)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 10)
          .map(([code, count]) => ({ costCode: code, count })),
        
        dataQuality: {
          avgItemsPerJob: jobsWithEstimates > 0 ? totalEstimateItems / jobsWithEstimates : 0,
          uniqueCostCodes: Object.keys(costCodeCounts).length,
          uniqueCategories: Object.keys(categoryCounts).length
        },
        
        lastCalculated: new Date().toISOString()
      };

      return stats;
      
    } catch (error) {
      console.error('❌ Error getting Firebase estimate stats:', error.message);
      throw error;
    }
  }

  // Get estimates for specific cost codes across all projects
  async getEstimatesByCostCode(costCode, companyId = null) {
    try {
      console.log(`🔍 Searching for estimates with cost code: ${costCode}`);
      
      const allEstimates = await this.getAllEstimateData(companyId);
      const matchingEstimates = [];
      
      allEstimates.forEach(jobData => {
        const matches = jobData.estimates.filter(estimate => 
          estimate.costCode === costCode
        );
        
        if (matches.length > 0) {
          matchingEstimates.push({
            jobId: jobData.jobId,
            projectTitle: jobData.projectTitle,
            clientName: jobData.clientName,
            companyId: jobData.companyId,
            estimates: matches,
            matchCount: matches.length
          });
        }
      });
      
      const totalMatches = matchingEstimates.reduce((sum, job) => sum + job.matchCount, 0);
      
      console.log(`✅ Found ${totalMatches} estimates with cost code ${costCode} across ${matchingEstimates.length} jobs`);
      
      return {
        costCode,
        totalMatches,
        jobsWithMatches: matchingEstimates.length,
        jobs: matchingEstimates
      };
      
    } catch (error) {
      console.error(`❌ Error searching estimates by cost code ${costCode}:`, error.message);
      throw error;
    }
  }

  // Get estimates by category across all projects
  async getEstimatesByCategory(category, companyId = null) {
    try {
      console.log(`🔍 Searching for ${category} estimates`);
      
      const allEstimates = await this.getAllEstimateData(companyId);
      const matchingEstimates = [];
      
      allEstimates.forEach(jobData => {
        const matches = jobData.estimates.filter(estimate => 
          this.categorizeEstimate(estimate) === category
        );
        
        if (matches.length > 0) {
          matchingEstimates.push({
            jobId: jobData.jobId,
            projectTitle: jobData.projectTitle,
            clientName: jobData.clientName,
            companyId: jobData.companyId,
            estimates: matches,
            matchCount: matches.length
          });
        }
      });
      
      const totalMatches = matchingEstimates.reduce((sum, job) => sum + job.matchCount, 0);
      
      console.log(`✅ Found ${totalMatches} ${category} estimates across ${matchingEstimates.length} jobs`);
      
      return {
        category,
        totalMatches,
        jobsWithMatches: matchingEstimates.length,
        jobs: matchingEstimates
      };
      
    } catch (error) {
      console.error(`❌ Error searching estimates by category ${category}:`, error.message);
      throw error;
    }
  }

  // Helper method to format job location
  formatJobLocation(jobData) {
    const parts = [
      jobData.siteStreet,
      jobData.siteCity,
      jobData.siteState,
      jobData.siteZip
    ].filter(Boolean);
    
    return parts.length > 0 ? parts.join(', ') : 'Location not specified';
  }

  // Helper method to calculate estimate statistics
  calculateEstimateStats(estimates) {
    if (!estimates || estimates.length === 0) {
      return {
        count: 0,
        totalValue: 0,
        avgValue: 0,
        maxValue: 0,
        minValue: 0,
        categories: {}
      };
    }

    const values = [];
    const categories = {};
    let totalValue = 0;

    estimates.forEach(estimate => {
      const amount = parseFloat(estimate.total || estimate.amount || 0);
      if (amount > 0) {
        values.push(amount);
        totalValue += amount;
      }
      
      const category = this.categorizeEstimate(estimate);
      categories[category] = (categories[category] || 0) + 1;
    });

    return {
      count: estimates.length,
      totalValue,
      avgValue: values.length > 0 ? totalValue / values.length : 0,
      maxValue: values.length > 0 ? Math.max(...values) : 0,
      minValue: values.length > 0 ? Math.min(...values) : 0,
      categories,
      validAmounts: values.length,
      completionRate: (values.length / estimates.length) * 100
    };
  }

  // Helper method to categorize estimates
  categorizeEstimate(estimate) {
    const costCode = (estimate.costCode || '').toUpperCase();
    const description = (estimate.description || '').toLowerCase();
    
    // Check cost code patterns first (more reliable)
    if (costCode.match(/\d+M$/)) return 'material';
    if (costCode.match(/\d+L$/)) return 'labor';
    if (costCode.match(/\d+S$/)) return 'subcontractor';
    if (costCode.match(/\d+O$/)) return 'overhead';
    if (costCode.match(/\d+E$/)) return 'equipment';
    
    // Check description patterns
    if (description.includes('material') || description.includes('supply')) return 'material';
    if (description.includes('labor') || description.includes('work') || description.includes('install')) return 'labor';
    if (description.includes('subcontractor') || description.includes('sub ')) return 'subcontractor';
    if (description.includes('overhead') || description.includes('management')) return 'overhead';
    if (description.includes('rental') || description.includes('equipment')) return 'equipment';
    
    return 'other';
  }

  // Validate estimate data quality
  validateEstimateData(estimates) {
    const validation = {
      total: estimates.length,
      valid: 0,
      issues: [],
      warnings: []
    };

    estimates.forEach((estimate, index) => {
      let hasIssues = false;

      // Check required fields
      if (!estimate.costCode && !estimate.description) {
        validation.issues.push(`Item ${index + 1}: Missing both cost code and description`);
        hasIssues = true;
      }

      if (!estimate.total && !estimate.amount) {
        validation.issues.push(`Item ${index + 1}: Missing amount/total value`);
        hasIssues = true;
      }

      // Check data quality warnings
      if (estimate.costCode && estimate.costCode.length < 2) {
        validation.warnings.push(`Item ${index + 1}: Very short cost code`);
      }

      if (estimate.description && estimate.description.length < 5) {
        validation.warnings.push(`Item ${index + 1}: Very short description`);
      }

      const amount = parseFloat(estimate.total || estimate.amount || 0);
      if (amount < 0) {
        validation.warnings.push(`Item ${index + 1}: Negative amount`);
      }

      if (!hasIssues) {
        validation.valid++;
      }
    });

    validation.validationRate = estimates.length > 0 ? (validation.valid / estimates.length) * 100 : 0;

    return validation;
  }
}

// Export singleton instance
const estimateFocusedFirebase = new EstimateFocusedFirebaseService();
export default estimateFocusedFirebase;