/**
 * Construction Estimate RAG Server
 * Express server focused exclusively on construction estimate data processing and analysis
 */

import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

// Import our services
import db from './database.js';
import firebase from './firebase.js';
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
app.post('/api/sync/job', async (req, res) => {
  try {
    const { companyId, jobId } = req.body;
    
    if (!companyId || !jobId) {
      return res.status(400).json({ 
        error: 'companyId and jobId are required' 
      });
    }

    console.log(`🔄 Syncing estimate data for job ${jobId} from company ${companyId}...`);

    // Step 1: Fetch estimate data from Firebase
    const jobData = await firebase.getJobEstimates(companyId, jobId);
    
    if (!jobData) {
      return res.status(404).json({ 
        error: `Job ${jobId} not found in Firebase` 
      });
    }

    if (!jobData.estimates || jobData.estimates.length === 0) {
      return res.json({
        success: true,
        message: 'No estimate data found for this job',
        project: null,
        stats: { entities: 0, embeddings: 0, estimates: 0 }
      });
    }

    // Step 2: Create/update project in database
    const project = await db.createProject(
      jobId,
      jobData.jobTitle,
      jobData.clientName,
      {
        ...jobData.metadata,
        estimateCount: jobData.estimates.length,
        syncedAt: new Date().toISOString()
      }
    );

    // Step 3: Process estimate entities only
    const entities = entityProcessor.processJobEstimates(jobData);
    
    if (entities.length === 0) {
      return res.json({
        success: true,
        message: 'No valid estimate entities to process',
        project,
        stats: { entities: 0, embeddings: 0, estimates: 0 }
      });
    }

    // Step 4: Generate embeddings and store everything
    const result = await embeddingService.processEstimateEntities(entities, db);

    // Step 5: Build relationships between estimate entities
    console.log('🔗 Building relationships between estimate entities...');
    const relationships = await relationshipBuilder.buildRelationships(result.entities, db);

    // Step 6: Analyze results
    const analysis = entityProcessor.analyzeEstimates(entities);

    res.json({
      success: true,
      project,
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
      processingStats: result.processingStats
    });
    
  } catch (error) {
    console.error('❌ Error syncing job estimates:', error.message);
    res.status(500).json({ 
      error: error.message,
      details: error.stack
    });
  }
});

// Bulk sync all estimate data from Firebase
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

        // Create project
        const project = await db.createProject(
          jobData.jobId,
          jobData.jobTitle || 'Unknown Project',
          jobData.clientName || 'Unknown Client',
          {
            ...jobData.metadata,
            estimateCount: jobData.estimates.length,
            syncedAt: new Date().toISOString()
          }
        );

        // Process estimate entities
        const entities = entityProcessor.processJobEstimates(jobData);
        
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
        console.log(`✅ Processed job ${jobData.jobId} (${entities.length} entities)`);
        
      } catch (error) {
        console.error(`❌ Failed to process job ${jobData.jobId}:`, error.message);
        results.failed++;
        results.errors.push({
          jobId: jobData.jobId,
          jobTitle: jobData.jobTitle,
          error: error.message
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
      details: error.stack
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
      relationshipBuilder.getRelationshipStats(db)
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