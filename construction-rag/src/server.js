/**
 * Construction Graph RAG Server
 * Main Express server that orchestrates the entire Graph RAG system
 */

import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

// Import our services
import db from './database.js';
import firebase from './firebase.js';
import entityProcessor from './entityProcessor.js';
import embeddingService from './embeddingService.js';

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
    const [firebaseOk, embeddingOk] = await Promise.all([
      firebase.testConnection(),
      embeddingService.testEmbeddingService()
    ]);

    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        database: true, // Database connection is tested on startup
        firebase: firebaseOk,
        embeddings: embeddingOk,
        openai: embeddingOk
      },
      version: '1.0.0'
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

// Get all projects
app.get('/api/projects', async (req, res) => {
  try {
    const projects = await db.getAllProjects();
    
    // Enrich with entity counts
    const enrichedProjects = await Promise.all(
      projects.map(async (project) => {
        const stats = await db.getProjectStats(project.id);
        return {
          ...project,
          stats
        };
      })
    );

    res.json({
      projects: enrichedProjects,
      total: enrichedProjects.length
    });
    
  } catch (error) {
    console.error('❌ Error getting projects:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Get specific project details
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

    res.json({
      project,
      stats,
      entities: entities.slice(0, 50), // Limit for performance
      costCodeSummary,
      totalEntities: entities.length
    });
    
  } catch (error) {
    console.error('❌ Error getting project:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Sync single job from Firebase
app.post('/api/sync/job', async (req, res) => {
  try {
    const { companyId, jobId } = req.body;
    
    if (!companyId || !jobId) {
      return res.status(400).json({ 
        error: 'companyId and jobId are required' 
      });
    }

    console.log(`🔄 Syncing job ${jobId} from company ${companyId}...`);

    // Step 1: Fetch data from Firebase
    const jobData = await firebase.getCompleteJobData(companyId, jobId);
    
    if (!jobData) {
      return res.status(404).json({ 
        error: `Job ${jobId} not found in Firebase` 
      });
    }

    // Step 2: Create/update project in database
    const project = await db.createProject(
      jobId,
      jobData.jobTitle,
      jobData.clientName,
      jobData
    );

    // Step 3: Process entities
    const entities = entityProcessor.processCompleteJobData(jobData);
    
    if (entities.length === 0) {
      return res.json({
        success: true,
        message: 'No entities to process',
        project,
        stats: { entities: 0, embeddings: 0 }
      });
    }

    // Step 4: Generate embeddings and store everything
    const result = await embeddingService.processEntities(entities, db);

    // Step 5: Analyze results
    const analysis = entityProcessor.analyzeEntities(entities);

    res.json({
      success: true,
      project,
      stats: {
        entities: result.entities.length,
        embeddings: result.embeddings.length,
        estimates: analysis.estimateCount,
        consumed: analysis.consumedCount,
        categories: analysis.categories,
        totalEstimated: analysis.totalEstimated,
        totalConsumed: analysis.totalConsumed,
        variance: analysis.totalVariance
      },
      analysis
    });
    
  } catch (error) {
    console.error('❌ Error syncing job:', error.message);
    res.status(500).json({ 
      error: error.message,
      details: error.stack
    });
  }
});

// Bulk sync all data from Firebase
app.post('/api/sync/all', async (req, res) => {
  try {
    const { companyId } = req.body; // Optional - sync specific company or all
    
    console.log(`🔄 Starting bulk sync${companyId ? ` for company ${companyId}` : ' for all companies'}...`);

    // Step 1: Get all estimate data
    const allEstimateJobs = await firebase.getAllEstimateData(companyId);
    console.log(`📊 Found ${allEstimateJobs.length} jobs with estimate data`);

    // Step 2: Get all consumed data
    const allConsumedJobs = await firebase.getAllConsumedData(companyId);
    console.log(`💰 Found ${allConsumedJobs.length} jobs with consumed data`);

    // Step 3: Merge data by job
    const jobDataMap = new Map();
    
    // Add estimate jobs
    allEstimateJobs.forEach(job => {
      jobDataMap.set(job.jobId, job);
    });
    
    // Merge consumed data
    allConsumedJobs.forEach(consumed => {
      const existing = jobDataMap.get(consumed.jobId);
      if (existing) {
        existing.consumedData = consumed;
      } else {
        // Create entry for consumed-only jobs
        jobDataMap.set(consumed.jobId, {
          companyId: consumed.companyId,
          jobId: consumed.jobId,
          estimates: [],
          consumedData: consumed
        });
      }
    });

    const allJobs = Array.from(jobDataMap.values());
    console.log(`🎯 Processing ${allJobs.length} unique jobs...`);

    // Step 4: Process all jobs
    const results = {
      processed: 0,
      failed: 0,
      totalEntities: 0,
      totalEmbeddings: 0,
      errors: []
    };

    for (const jobData of allJobs) {
      try {
        // Create project
        const project = await db.createProject(
          jobData.jobId,
          jobData.jobTitle || 'Unknown Project',
          jobData.clientName || 'Unknown Client',
          jobData
        );

        // Process entities
        const entities = entityProcessor.processCompleteJobData(jobData);
        
        if (entities.length > 0) {
          // Generate embeddings
          const result = await embeddingService.processEntities(entities, db);
          results.totalEntities += result.entities.length;
          results.totalEmbeddings += result.embeddings.length;
        }

        results.processed++;
        console.log(`✅ Processed job ${jobData.jobId} (${entities.length} entities)`);
        
      } catch (error) {
        console.error(`❌ Failed to process job ${jobData.jobId}:`, error.message);
        results.failed++;
        results.errors.push({
          jobId: jobData.jobId,
          error: error.message
        });
      }
    }

    console.log(`🎉 Bulk sync completed: ${results.processed} jobs processed, ${results.failed} failed`);

    res.json({
      success: true,
      message: 'Bulk sync completed',
      results,
      summary: {
        jobsProcessed: results.processed,
        jobsFailed: results.failed,
        totalEntities: results.totalEntities,
        totalEmbeddings: results.totalEmbeddings
      }
    });
    
  } catch (error) {
    console.error('❌ Error in bulk sync:', error.message);
    res.status(500).json({ 
      error: error.message,
      details: error.stack
    });
  }
});

// Query endpoint - Graph RAG search
app.post('/api/query', async (req, res) => {
  try {
    const { 
      question, 
      projectId = null,
      limit = 10,
      threshold = null,
      entityType = null,
      category = null
    } = req.body;

    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }

    console.log(`🔍 Processing query: "${question}"`);

    // Search for similar entities
    const searchOptions = {
      threshold: threshold || embeddingService.similarityThreshold,
      limit,
      projectId,
      entityType,
      category
    };

    const similarEntities = await embeddingService.searchSimilar(
      question,
      db,
      searchOptions
    );

    if (similarEntities.length === 0) {
      return res.json({
        question,
        answer: "I couldn't find any relevant information in the construction data for your question.",
        sources: [],
        entities: [],
        searchOptions
      });
    }

    // Build context from similar entities
    const context = {
      question,
      entities: similarEntities,
      summary: {
        totalResults: similarEntities.length,
        avgSimilarity: similarEntities.reduce((sum, e) => sum + e.similarity, 0) / similarEntities.length,
        categories: [...new Set(similarEntities.map(e => e.category))],
        costCodes: [...new Set(similarEntities.map(e => e.cost_code))]
      }
    };

    // For now, return structured data
    // TODO: Integrate with OpenAI for natural language responses
    res.json({
      question,
      answer: `Found ${similarEntities.length} relevant construction items. The most similar items relate to ${context.summary.categories.join(', ')} with cost codes including ${context.summary.costCodes.slice(0, 3).join(', ')}.`,
      sources: similarEntities.map(entity => ({
        costCode: entity.cost_code,
        description: entity.description,
        category: entity.category,
        amount: entity.total_amount,
        similarity: entity.similarityPercent,
        relevance: Math.round(entity.relevanceScore * 100)
      })),
      entities: similarEntities,
      context,
      searchOptions
    });
    
  } catch (error) {
    console.error('❌ Error processing query:', error.message);
    res.status(500).json({ 
      error: error.message,
      details: error.stack
    });
  }
});

// Get system statistics
app.get('/api/stats', async (req, res) => {
  try {
    const [dbStats, embeddingStats, firebaseStats] = await Promise.all([
      db.query(`
        SELECT 
          (SELECT COUNT(*) FROM projects) as total_projects,
          (SELECT COUNT(*) FROM entities) as total_entities,
          (SELECT COUNT(*) FROM relationships) as total_relationships,
          (SELECT COUNT(*) FROM embeddings) as total_embeddings
      `),
      embeddingService.getEmbeddingStats(db),
      firebase.getFirebaseStats().catch(() => ({ error: 'Firebase stats unavailable' }))
    ]);

    res.json({
      database: dbStats.rows[0],
      embeddings: embeddingStats,
      firebase: firebaseStats,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('❌ Error getting stats:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Search entities by cost code
app.get('/api/search/cost-code/:costCode', async (req, res) => {
  try {
    const { costCode } = req.params;
    const { projectId } = req.query;
    
    const entities = projectId 
      ? await db.getEntitiesByCostCode(projectId, costCode)
      : await db.query('SELECT * FROM entities WHERE cost_code = $1', [costCode]);

    res.json({
      costCode,
      entities: projectId ? entities : entities.rows,
      total: projectId ? entities.length : entities.rows.length
    });
    
  } catch (error) {
    console.error('❌ Error searching by cost code:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Search entities by category
app.get('/api/search/category/:category', async (req, res) => {
  try {
    const { category } = req.params;
    const { projectId } = req.query;
    
    const entities = projectId 
      ? await db.getEntitiesByCategory(projectId, category)
      : await db.query('SELECT * FROM entities WHERE category = $1', [category]);

    res.json({
      category,
      entities: projectId ? entities : entities.rows,
      total: projectId ? entities.length : entities.rows.length
    });
    
  } catch (error) {
    console.error('❌ Error searching by category:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Delete project and all related data
app.delete('/api/projects/:projectId', async (req, res) => {
  try {
    const { projectId } = req.params;
    
    await db.deleteProjectData(projectId);
    
    res.json({
      success: true,
      message: `Project ${projectId} and all related data deleted`
    });
    
  } catch (error) {
    console.error('❌ Error deleting project:', error.message);
    res.status(500).json({ error: error.message });
  }
});

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
  console.log(`🚀 Construction Graph RAG Server running on port ${PORT}`);
  console.log(`📊 Environment: ${process.env.NODE_ENV}`);
  console.log(`🔗 Health check: http://localhost:${PORT}/health`);
  console.log(`📖 API Base: http://localhost:${PORT}/api`);
  console.log('');
  console.log('🎯 Key Endpoints:');
  console.log(`   POST /api/sync/job           - Sync single job from Firebase`);
  console.log(`   POST /api/sync/all           - Bulk sync all Firebase data`);
  console.log(`   POST /api/query              - Graph RAG search`);
  console.log(`   GET  /api/projects           - List all projects`);
  console.log(`   GET  /api/stats              - System statistics`);
  console.log('');
  console.log('✅ Ready to process construction data!');
});