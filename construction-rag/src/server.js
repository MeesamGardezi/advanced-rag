/**
 * Construction Estimate RAG Server
 * Express server focused exclusively on construction estimate data processing and analysis
 * FIXED: Properly passes database UUID to entity processor
 */

import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

// Import our services from config directory (not src)
import db from '../config/database.js';
import firebase from '../config/firebase.js';
import entityProcessor from './entityProcessor.js';
import embeddingService from './embeddingService.js';
import ragService from './ragService.js';
import relationshipBuilder from './relationshipBuilder.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Request logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.path} - ${res.statusCode} (${duration}ms)`);
  });
  next();
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    // Test all services
    const [firebaseOk, embeddingOk, ragOk] = await Promise.all([
      firebase.testConnection(),
      embeddingService.testPerformance(),
      ragService.testEstimateRAG()
    ]);

    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        database: true, // Database connection is tested on startup
        firebase: firebaseOk,
        embeddings: embeddingOk.success,
        rag: ragOk.success,
        openai: embeddingOk.success
      },
      version: '1.0.0',
      focus: 'Construction Estimates Only'
    };

    res.json(health);
    
  } catch (error) {
    console.error('❌ Health check failed:', error.message);
    res.status(500).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get all projects with estimate statistics
app.get('/api/projects', async (req, res) => {
  try {
    const projects = await db.getAllProjects();
    
    // Enrich with estimate-only statistics
    const enrichedProjects = await Promise.all(
      projects.map(async (project) => {
        const stats = await db.getProjectStats(project.id);
        const costCodeSummary = await db.getCostCodeSummary(project.id);
        
        return {
          ...project,
          stats: {
            ...stats,
            topCostCodes: costCodeSummary.slice(0, 5),
            estimateHealthScore: calculateEstimateHealthScore(stats, costCodeSummary)
          }
        };
      })
    );

    res.json({
      projects: enrichedProjects,
      total: enrichedProjects.length,
      summary: {
        totalEstimates: enrichedProjects.reduce((sum, p) => sum + (parseInt(p.stats.estimates) || 0), 0),
        totalValue: enrichedProjects.reduce((sum, p) => sum + (parseFloat(p.stats.total_amount) || 0), 0),
        avgProjectValue: enrichedProjects.length > 0 ? 
          enrichedProjects.reduce((sum, p) => sum + (parseFloat(p.stats.total_amount) || 0), 0) / enrichedProjects.length : 0
      }
    });
    
  } catch (error) {
    console.error('❌ Error getting projects:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Get specific project details with enhanced estimate analytics
app.get('/api/projects/:projectId', async (req, res) => {
  try {
    const { projectId } = req.params;
    const project = await db.getProject(projectId);
    
    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    const [stats, entities, costCodeSummary] = await Promise.all([
      db.getProjectStats(project.id),
      db.getEntitiesByProject(project.id),
      db.getCostCodeSummary(project.id)
    ]);

    // Analyze estimate entities for insights
    const analysis = entityProcessor.analyzeEstimates(entities);

    res.json({
      project,
      stats,
      analysis,
      entities: entities.slice(0, 50), // Limit for performance
      costCodeSummary,
      totalEntities: entities.length,
      insights: {
        budgetHealth: calculateProjectBudgetHealth(analysis),
        topCategories: analysis.categories,
        highVarianceItems: analysis.varianceItems.slice(0, 10),
        highValueItems: analysis.highValueItems.slice(0, 10)
      }
    });
    
  } catch (error) {
    console.error('❌ Error getting project:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Sync single job estimates from Firebase
// FIXED: Now properly passes database UUID to entity processor
app.post('/api/sync/job', async (req, res) => {
  try {
    const { companyId, jobId } = req.body;
    
    if (!companyId || !jobId) {
      return res.status(400).json({ 
        error: 'companyId and jobId are required' 
      });
    }

    console.log(`🔄 DEBUG: Starting sync for job ${jobId} from company ${companyId}...`);

    // Step 1: Fetch estimate data from Firebase with detailed logging
    console.log(`📡 DEBUG: Fetching job data from Firebase...`);
    const jobData = await firebase.getJobEstimates(companyId, jobId);
    
    if (!jobData) {
      console.log(`❌ DEBUG: No job data returned from Firebase for job ${jobId}`);
      return res.status(404).json({ 
        error: `Job ${jobId} not found in Firebase` 
      });
    }

    console.log(`✅ DEBUG: Job data fetched successfully:`);
    console.log(`   - Job ID (Firebase): ${jobData.jobId}`);
    console.log(`   - Project Title: ${jobData.projectTitle || jobData.jobTitle}`);
    console.log(`   - Client: ${jobData.clientName}`);
    console.log(`   - Raw estimates array length: ${jobData.estimates?.length || 0}`);
    console.log(`   - Valid estimate count: ${jobData.validEstimateCount || 0}`);

    if (!jobData.estimates || jobData.estimates.length === 0) {
      console.log(`⚠️ DEBUG: No estimates found in job data. Job data structure:`, {
        hasEstimates: !!jobData.estimates,
        estimatesType: typeof jobData.estimates,
        estimatesLength: jobData.estimates?.length,
        jobDataKeys: Object.keys(jobData)
      });
      
      return res.json({
        success: true,
        message: 'No estimate data found for this job',
        project: null,
        stats: { entities: 0, embeddings: 0, estimates: 0 },
        debug: {
          jobDataKeys: Object.keys(jobData),
          hasEstimates: !!jobData.estimates,
          estimatesLength: jobData.estimates?.length
        }
      });
    }

    // Debug: Log sample estimates
    console.log(`📋 DEBUG: Sample estimates (first 2):`);
    jobData.estimates.slice(0, 2).forEach((estimate, i) => {
      console.log(`   Estimate ${i + 1}:`, {
        costCode: estimate.costCode,
        description: estimate.description?.substring(0, 50) + '...',
        total: estimate.total,
        amount: estimate.amount,
        hasData: !!(estimate.costCode || estimate.description)
      });
    });

    // Step 2: Create/update project in database
    console.log(`💾 DEBUG: Creating/updating project in database...`);
    const project = await db.createProject(
      jobId,  // This is the Firebase ID used as reference
      jobData.jobTitle || jobData.projectTitle || 'Unknown Project',
      jobData.clientName || 'Unknown Client',
      {
        ...jobData.metadata,
        estimateCount: jobData.estimates.length,
        firebaseJobId: jobId,  // Store original Firebase ID
        companyId: companyId,
        syncedAt: new Date().toISOString()
      }
    );
    
    console.log(`✅ DEBUG: Project created/updated:`);
    console.log(`   - Database UUID: ${project.id}`);
    console.log(`   - Firebase ID: ${jobId}`);
    console.log(`   - Firebase ID field in DB: ${project.firebase_id}`);

    // Step 3: Process estimate entities - PASS THE DATABASE PROJECT UUID!
    console.log(`🔄 DEBUG: Processing estimate entities with database project UUID: ${project.id}...`);
    const entities = entityProcessor.processJobEstimates(jobData, project.id);  // Pass the database UUID!
    
    console.log(`📊 DEBUG: EntityProcessor results:`);
    console.log(`   - Processed entities count: ${entities.length}`);
    
    if (entities.length > 0) {
      console.log(`   - Sample entity:`, {
        projectId: entities[0].projectId,
        projectIdType: typeof entities[0].projectId,
        isUUID: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(entities[0].projectId),
        costCode: entities[0].costCode,
        category: entities[0].category
      });
    }
    
    if (entities.length === 0) {
      console.log(`⚠️ DEBUG: EntityProcessor returned 0 entities. Investigating...`);
      
      // Debug the processing
      const validEstimates = jobData.estimates.filter(est => 
        est && (est.costCode || est.description) && 
        (est.total !== undefined || est.amount !== undefined)
      );
      
      console.log(`🔍 DEBUG: Validation results:`);
      console.log(`   - Total estimates: ${jobData.estimates.length}`);
      console.log(`   - Valid estimates: ${validEstimates.length}`);
      console.log(`   - Invalid estimates: ${jobData.estimates.length - validEstimates.length}`);

      return res.json({
        success: true,
        message: 'No valid estimate entities to process',
        project,
        stats: { entities: 0, embeddings: 0, estimates: 0 },
        debug: {
          totalEstimates: jobData.estimates.length,
          validEstimates: validEstimates.length,
          invalidEstimates: jobData.estimates.length - validEstimates.length
        }
      });
    }

    // Step 4: Generate embeddings and store everything
    console.log(`🔮 DEBUG: Processing embeddings for ${entities.length} entities...`);
    const result = await embeddingService.processEstimateEntities(entities, db);
    
    console.log(`✅ DEBUG: Embedding processing completed:`);
    console.log(`   - Created entities: ${result.entities.length}`);
    console.log(`   - Created embeddings: ${result.embeddings.length}`);
    console.log(`   - Total processed: ${result.total}`);

    // Step 5: Build relationships between estimate entities
    console.log('🔗 DEBUG: Building relationships between estimate entities...');
    const relationships = await relationshipBuilder.buildRelationships(result.entities, db);
    console.log(`✅ DEBUG: Created ${relationships.length} relationships`);

    // Step 6: Analyze results
    console.log(`📈 DEBUG: Analyzing estimate results...`);
    const analysis = entityProcessor.analyzeEstimates(entities);
    console.log(`✅ DEBUG: Analysis completed - ${analysis.totalEntities} entities analyzed`);

    res.json({
      success: true,
      project: {
        id: project.id,
        firebase_id: project.firebase_id,
        title: project.title,
        client_name: project.client_name
      },
      stats: {
        entities: result.entities.length,
        embeddings: result.embeddings.length,
        relationships: relationships.length,
        estimates: analysis.totalEntities
      },
      analysis: {
        categories: analysis.categories,
        totalEstimated: analysis.totals.estimated,
        totalBudgeted: analysis.totals.budgeted,
        overallVariance: analysis.totals.variance,
        overallVariancePercent: analysis.totals.variancePercent,
        budgetHealth: analysis.budgetHealth,
        highValueItems: analysis.highValueItems.length,
        varianceItems: analysis.varianceItems.length
      },
      processingStats: result.processingStats,
      debug: {
        projectUUID: project.id,
        firebaseJobId: jobId,
        rawEstimatesCount: jobData.estimates.length,
        validEstimatesCount: jobData.validEstimateCount,
        processedEntitiesCount: entities.length,
        embeddingSuccessRate: `${result.embeddings.length}/${entities.length}`,
        relationshipsCreated: relationships.length
      }
    });
    
  } catch (error) {
    console.error('❌ DEBUG: Error in sync process:', error.message);
    console.error('❌ DEBUG: Error stack:', error.stack);
    res.status(500).json({ 
      error: error.message,
      details: error.stack,
      type: error.name
    });
  }
});

// Bulk sync all estimate data from Firebase
// FIXED: Now properly passes database UUID to entity processor
app.post('/api/sync/all', async (req, res) => {
  try {
    const { companyId } = req.body; // Optional - sync specific company or all
    
    console.log(`🔄 Starting bulk estimate sync${companyId ? ` for company ${companyId}` : ' for all companies'}...`);

    // Step 1: Get all estimate data only
    const allEstimateJobs = await firebase.getAllEstimateData(companyId);
    console.log(`📊 Found ${allEstimateJobs.length} jobs with estimate data`);

    if (allEstimateJobs.length === 0) {
      return res.json({
        success: true,
        message: 'No estimate data found to sync',
        results: { processed: 0, failed: 0, totalEntities: 0, totalEmbeddings: 0 }
      });
    }

    // Step 2: Process all estimate jobs
    const results = {
      processed: 0,
      failed: 0,
      totalEntities: 0,
      totalEmbeddings: 0,
      totalRelationships: 0,
      errors: [],
      projectStats: {
        totalEstimatedValue: 0,
        totalBudgetedValue: 0,
        overallVariance: 0,
        categoryCounts: {}
      }
    };

    for (const jobData of allEstimateJobs) {
      try {
        console.log(`📋 Processing job ${jobData.jobId}: ${jobData.jobTitle || 'Untitled'}`);

        // Create project and get database UUID
        const project = await db.createProject(
          jobData.jobId,  // Firebase ID for reference
          jobData.jobTitle || 'Unknown Project',
          jobData.clientName || 'Unknown Client',
          {
            ...jobData.metadata,
            estimateCount: jobData.estimates.length,
            firebaseJobId: jobData.jobId,
            companyId: jobData.companyId,
            syncedAt: new Date().toISOString()
          }
        );

        console.log(`✅ DEBUG: Created/updated project - DB UUID: ${project.id}, Firebase ID: ${jobData.jobId}`);

        // Process estimate entities with the database UUID
        const entities = entityProcessor.processJobEstimates(jobData, project.id);  // Pass the database UUID!
        
        if (entities.length > 0) {
          // Generate embeddings
          const result = await embeddingService.processEstimateEntities(entities, db);
          results.totalEntities += result.entities.length;
          results.totalEmbeddings += result.embeddings.length;

          // Build relationships
          const relationships = await relationshipBuilder.buildRelationships(result.entities, db);
          results.totalRelationships += relationships.length;

          // Accumulate project statistics
          const analysis = entityProcessor.analyzeEstimates(entities);
          results.projectStats.totalEstimatedValue += analysis.totals.estimated;
          results.projectStats.totalBudgetedValue += analysis.totals.budgeted;
          
          // Merge category counts
          Object.entries(analysis.categories).forEach(([category, count]) => {
            results.projectStats.categoryCounts[category] = 
              (results.projectStats.categoryCounts[category] || 0) + count;
          });
        }

        results.processed++;
        console.log(`✅ Processed job ${jobData.jobId} with DB ID ${project.id} (${entities.length} entities)`);
        
      } catch (error) {
        console.error(`❌ Failed to process job ${jobData.jobId}:`, error.message);
        results.failed++;
        results.errors.push({
          jobId: jobData.jobId,
          jobTitle: jobData.jobTitle,
          error: error.message,
          type: error.name
        });
      }
    }

    // Calculate overall variance
    results.projectStats.overallVariance = 
      results.projectStats.totalEstimatedValue - results.projectStats.totalBudgetedValue;
    results.projectStats.overallVariancePercent = 
      results.projectStats.totalBudgetedValue > 0 
        ? (results.projectStats.overallVariance / results.projectStats.totalBudgetedValue) * 100 
        : 0;

    console.log(`🎉 Bulk estimate sync completed: ${results.processed} jobs processed, ${results.failed} failed`);

    res.json({
      success: true,
      message: 'Bulk estimate sync completed',
      results,
      summary: {
        jobsProcessed: results.processed,
        jobsFailed: results.failed,
        totalEntities: results.totalEntities,
        totalEmbeddings: results.totalEmbeddings,
        totalRelationships: results.totalRelationships,
        estimatedValue: results.projectStats.totalEstimatedValue,
        budgetedValue: results.projectStats.totalBudgetedValue,
        overallVariancePercent: results.projectStats.overallVariancePercent.toFixed(1) + '%'
      }
    });
    
  } catch (error) {
    console.error('❌ Error in bulk estimate sync:', error.message);
    res.status(500).json({ 
      error: error.message,
      details: error.stack,
      type: error.name
    });
  }
});

// Enhanced RAG query endpoint for estimate analysis
app.post('/api/query', async (req, res) => {
  try {
    const { 
      question, 
      projectId = null,
      limit = 10,
      threshold = null,
      category = null,
      costCodePattern = null,
      responseStyle = 'detailed',
      includeAnalysis = true
    } = req.body;

    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }

    console.log(`🔍 Processing estimate query: "${question}"`);

    // Use the specialized RAG service for estimates
    const response = await ragService.queryEstimates(question, db, {
      projectId,
      limit,
      threshold,
      category,
      costCodePattern,
      responseStyle,
      includeAnalysis
    });

    res.json(response);
    
  } catch (error) {
    console.error('❌ Error processing estimate query:', error.message);
    res.status(500).json({ 
      error: error.message,
      details: error.stack
    });
  }
});

// Get estimate analytics for a project
app.get('/api/projects/:projectId/analytics', async (req, res) => {
  try {
    const { projectId } = req.params;
    
    const entities = await db.getEntitiesByProject(projectId);
    if (entities.length === 0) {
      return res.json({
        message: 'No estimate data found for this project',
        analytics: null
      });
    }

    const analysis = entityProcessor.analyzeEstimates(entities);
    const costCodeSummary = await db.getCostCodeSummary(projectId);

    // Enhanced analytics
    const analytics = {
      overview: {
        totalItems: analysis.totalEntities,
        estimatedValue: analysis.totals.estimated,
        budgetedValue: analysis.totals.budgeted,
        variance: analysis.totals.variance,
        variancePercent: analysis.totals.variancePercent
      },
      
      budgetPerformance: {
        onBudget: analysis.budgetHealth.onBudget,
        overBudget: analysis.budgetHealth.overBudget,
        underBudget: analysis.budgetHealth.underBudget,
        healthScore: calculateEstimateHealthScore(analysis, costCodeSummary)
      },
      
      categoryBreakdown: analysis.categories,
      
      topCostCodes: costCodeSummary.slice(0, 10),
      
      highValueItems: analysis.highValueItems,
      
      varianceAnalysis: {
        items: analysis.varianceItems,
        patterns: identifyVariancePatterns(analysis.varianceItems),
        recommendations: generateVarianceRecommendations(analysis.varianceItems)
      },
      
      trends: {
        categoryTrends: analyzeCategoryTrends(entities),
        costCodeTrends: analyzeCostCodeTrends(costCodeSummary)
      }
    };

    res.json({
      projectId,
      analytics,
      generatedAt: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('❌ Error getting project analytics:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Get system statistics (estimate-focused)
app.get('/api/stats', async (req, res) => {
  try {
    const [dbStats, embeddingStats, relationshipStats] = await Promise.all([
      db.query(`
        SELECT 
          (SELECT COUNT(*) FROM projects) as total_projects,
          (SELECT COUNT(*) FROM entities WHERE entity_type = 'estimate_row') as total_estimates,
          (SELECT COUNT(*) FROM relationships) as total_relationships,
          (SELECT COUNT(*) FROM embeddings) as total_embeddings,
          (SELECT SUM(total_amount) FROM entities WHERE entity_type = 'estimate_row') as total_estimated_value,
          (SELECT SUM(budgeted_amount) FROM entities WHERE entity_type = 'estimate_row') as total_budgeted_value
      `),
      embeddingService.getPerformanceStats(db),
      relationshipBuilder.getEstimateRelationshipStats(db)
    ]);

    const stats = dbStats.rows[0];
    const overallVariance = (parseFloat(stats.total_estimated_value) || 0) - (parseFloat(stats.total_budgeted_value) || 0);
    const overallVariancePercent = stats.total_budgeted_value > 0 
      ? (overallVariance / parseFloat(stats.total_budgeted_value)) * 100 
      : 0;

    res.json({
      database: {
        ...stats,
        overall_variance: overallVariance,
        overall_variance_percent: overallVariancePercent
      },
      embeddings: embeddingStats,
      relationships: relationshipStats,
      system: {
        focus: 'Construction Estimates Only',
        dataTypes: ['estimate_row'],
        capabilities: [
          'Estimate Analysis',
          'Budget Variance Detection', 
          'Cost Code Categorization',
          'RAG-based Query Processing',
          'Relationship Mapping'
        ]
      },
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('❌ Error getting system stats:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Search estimates by cost code
app.get('/api/search/cost-code/:costCode', async (req, res) => {
  try {
    const { costCode } = req.params;
    const { projectId } = req.query;
    
    const entities = projectId 
      ? await db.getEntitiesByCostCode(projectId, costCode)
      : await db.query('SELECT * FROM entities WHERE cost_code = $1 AND entity_type = $2', [costCode, 'estimate_row']);

    const results = projectId ? entities : entities.rows;
    
    // Analyze the cost code results
    const analysis = results.length > 0 ? {
      totalItems: results.length,
      totalValue: results.reduce((sum, item) => sum + (parseFloat(item.total_amount) || 0), 0),
      avgValue: results.length > 0 ? results.reduce((sum, item) => sum + (parseFloat(item.total_amount) || 0), 0) / results.length : 0,
      categories: [...new Set(results.map(item => item.category))],
      projects: [...new Set(results.map(item => item.project_id))]
    } : null;

    res.json({
      costCode,
      entities: results,
      total: results.length,
      analysis,
      searchScope: projectId ? `project ${projectId}` : 'all projects'
    });
    
  } catch (error) {
    console.error('❌ Error searching by cost code:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Search estimates by category
app.get('/api/search/category/:category', async (req, res) => {
  try {
    const { category } = req.params;
    const { projectId } = req.query;
    
    const entities = projectId 
      ? await db.getEntitiesByCategory(projectId, category)
      : await db.query('SELECT * FROM entities WHERE category = $1 AND entity_type = $2', [category, 'estimate_row']);

    const results = projectId ? entities : entities.rows;
    
    // Analyze category results
    const analysis = results.length > 0 ? {
      totalItems: results.length,
      totalValue: results.reduce((sum, item) => sum + (parseFloat(item.total_amount) || 0), 0),
      avgValue: results.length > 0 ? results.reduce((sum, item) => sum + (parseFloat(item.total_amount) || 0), 0) / results.length : 0,
      costCodes: [...new Set(results.map(item => item.cost_code))].filter(Boolean),
      projects: [...new Set(results.map(item => item.project_id))],
      valueDistribution: calculateValueDistribution(results)
    } : null;

    res.json({
      category,
      entities: results,
      total: results.length,
      analysis,
      searchScope: projectId ? `project ${projectId}` : 'all projects'
    });
    
  } catch (error) {
    console.error('❌ Error searching by category:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Delete all projects and data
app.delete('/api/projects/all', async (req, res) => {
  try {
    await db.query('DELETE FROM embeddings');
    await db.query('DELETE FROM relationships');
    await db.query('DELETE FROM entities');
    await db.query('DELETE FROM projects');
    
    res.json({
      success: true,
      message: 'All projects and data deleted'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});


// Delete project and all related estimate data
app.delete('/api/projects/:projectId', async (req, res) => {
  try {
    const { projectId } = req.params;
    
    // Get stats before deletion for response
    const stats = await db.getProjectStats(projectId);
    
    await db.deleteProjectData(projectId);
    
    res.json({
      success: true,
      message: `Project ${projectId} and all related estimate data deleted`,
      deletedStats: {
        entities: stats.total_entities,
        estimatedValue: stats.total_amount,
        budgetedValue: stats.budgeted_amount
      }
    });
    
  } catch (error) {
    console.error('❌ Error deleting project:', error.message);
    res.status(500).json({ error: error.message });
  }
});


// Helper Functions
function calculateEstimateHealthScore(stats, costCodeSummary) {
  // Simple health scoring based on budget variance
  const overallVariance = Math.abs(parseFloat(stats.total_variance || 0));
  const totalBudgeted = parseFloat(stats.budgeted_amount || 1);
  const variancePercent = (overallVariance / totalBudgeted) * 100;
  
  if (variancePercent <= 5) return 'excellent';
  if (variancePercent <= 15) return 'good';
  if (variancePercent <= 30) return 'fair';
  return 'poor';
}

function calculateProjectBudgetHealth(analysis) {
  const total = analysis.budgetHealth.onBudget + analysis.budgetHealth.overBudget + analysis.budgetHealth.underBudget;
  
  if (total === 0) return { status: 'unknown', details: 'No budget data available' };
  
  const onBudgetPercent = (analysis.budgetHealth.onBudget / total) * 100;
  
  return {
    status: onBudgetPercent > 70 ? 'healthy' : onBudgetPercent > 50 ? 'concerning' : 'critical',
    onBudgetPercent: onBudgetPercent.toFixed(1),
    details: `${analysis.budgetHealth.onBudget} of ${total} items on budget`
  };
}

function identifyVariancePatterns(varianceItems) {
  const patterns = [];
  
  // Group by category to find category-wide issues
  const categoryVariances = {};
  varianceItems.forEach(item => {
    const category = item.category || 'other';
    if (!categoryVariances[category]) categoryVariances[category] = [];
    categoryVariances[category].push(item);
  });
  
  Object.entries(categoryVariances).forEach(([category, items]) => {
    if (items.length >= 3) {
      const avgVariance = items.reduce((sum, item) => sum + Math.abs(item.variancePercent), 0) / items.length;
      patterns.push({
        type: 'category_variance',
        category,
        itemCount: items.length,
        averageVariance: avgVariance.toFixed(1) + '%',
        description: `${category} category shows consistent variance across ${items.length} items`
      });
    }
  });
  
  return patterns;
}

function generateVarianceRecommendations(varianceItems) {
  const recommendations = [];
  
  const highVarianceItems = varianceItems.filter(item => Math.abs(item.variancePercent) > 30);
  if (highVarianceItems.length > 0) {
    recommendations.push({
      priority: 'high',
      action: 'Investigate High Variance Items',
      description: `${highVarianceItems.length} items have variance >30%. Review estimation methodology.`,
      items: highVarianceItems.slice(0, 3).map(item => item.costCode)
    });
  }
  
  return recommendations;
}

function analyzeCategoryTrends(entities) {
  // Simple trend analysis - could be enhanced with time series data
  const categoryTotals = {};
  entities.forEach(entity => {
    const category = entity.category || 'other';
    if (!categoryTotals[category]) {
      categoryTotals[category] = { count: 0, value: 0 };
    }
    categoryTotals[category].count++;
    categoryTotals[category].value += parseFloat(entity.total_amount || 0);
  });
  
  return Object.entries(categoryTotals)
    .map(([category, data]) => ({
      category,
      itemCount: data.count,
      totalValue: data.value,
      avgValue: data.value / data.count
    }))
    .sort((a, b) => b.totalValue - a.totalValue);
}

function analyzeCostCodeTrends(costCodeSummary) {
  return costCodeSummary.slice(0, 10).map(item => ({
    costCode: item.cost_code,
    category: item.category,
    itemCount: item.item_count,
    totalValue: parseFloat(item.total_amount || 0),
    variance: parseFloat(item.variance || 0),
    trend: parseFloat(item.variance || 0) > 0 ? 'over' : parseFloat(item.variance || 0) < 0 ? 'under' : 'on-budget'
  }));
}

function calculateValueDistribution(results) {
  const values = results.map(item => parseFloat(item.total_amount || 0)).sort((a, b) => a - b);
  
  return {
    min: values[0] || 0,
    max: values[values.length - 1] || 0,
    median: values[Math.floor(values.length / 2)] || 0,
    q1: values[Math.floor(values.length * 0.25)] || 0,
    q3: values[Math.floor(values.length * 0.75)] || 0
  };
}

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('🚨 Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message,
    type: error.name,
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.originalUrl,
    timestamp: new Date().toISOString()
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('🔄 Received SIGTERM, shutting down gracefully...');
  await db.close();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('🔄 Received SIGINT, shutting down gracefully...');
  await db.close();
  process.exit(0);
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Construction Estimate RAG Server running on port ${PORT}`);
  console.log(`📊 Environment: ${process.env.NODE_ENV}`);
  console.log(`🎯 Focus: Construction Estimates Only`);
  console.log(`🔗 Health check: http://localhost:${PORT}/health`);
  console.log(`📖 API Base: http://localhost:${PORT}/api`);
  console.log('');
  console.log('🎯 Key Endpoints:');
  console.log(`   POST /api/sync/job                    - Sync estimate data from Firebase`);
  console.log(`   POST /api/sync/all                    - Bulk sync all estimate data`);
  console.log(`   POST /api/query                       - RAG-powered estimate analysis`);
  console.log(`   GET  /api/projects                    - List projects with estimate stats`);
  console.log(`   GET  /api/projects/:id/analytics      - Detailed estimate analytics`);
  console.log(`   GET  /api/search/cost-code/:code      - Search by cost code`);
  console.log(`   GET  /api/search/category/:category   - Search by category`);
  console.log(`   GET  /api/stats                       - System statistics`);
  console.log('');
  console.log('✅ Ready to process construction estimate data!');
});