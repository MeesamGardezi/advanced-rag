/**
 * Estimate-Focused Database Service
 * PostgreSQL with pgvector support for construction estimate analysis only
 */

import pkg from 'pg';
import dotenv from 'dotenv';

const { Pool } = pkg;
dotenv.config();

class EstimateFocusedDatabaseService {
  constructor() {
    this.pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });
    
    this.initializeConnection();
  }

  async initializeConnection() {
    try {
      const client = await this.pool.connect();
      const result = await client.query('SELECT NOW(), version()');
      console.log('✅ Database connected:', result.rows[0].now);
      console.log('📊 PostgreSQL version:', result.rows[0].version.split(' ')[1]);
      
      // Test vector extension
      await client.query('SELECT 1 as test');
      console.log('🔗 Vector extension ready');
      console.log('🎯 Database Focus: Construction Estimates Only');
      
      client.release();
    } catch (error) {
      console.error('❌ Database connection failed:', error.message);
      process.exit(1);
    }
  }

  // Basic query method with enhanced logging
  async query(text, params) {
    const start = Date.now();
    try {
      const result = await this.pool.query(text, params);
      const duration = Date.now() - start;
      console.log('🗃️  Query executed', { duration: `${duration}ms`, rows: result.rows.length });
      return result;
    } catch (error) {
      console.error('❌ Query error:', error.message);
      throw error;
    }
  }

  // Get a client for transactions
  async getClient() {
    return await this.pool.connect();
  }

  // Projects operations - enhanced for estimates
  async createProject(firebaseId, title, clientName, data) {
    const query = `
      INSERT INTO projects (firebase_id, title, client_name, data)
      VALUES ($1, $2, $3, $4)
      ON CONFLICT (firebase_id) DO UPDATE SET
        title = $2, client_name = $3, data = $4
      RETURNING *
    `;
    const result = await this.query(query, [firebaseId, title, clientName, JSON.stringify(data)]);
    return result.rows[0];
  }

  async getProject(firebaseId) {
    const query = 'SELECT * FROM projects WHERE firebase_id = $1';
    const result = await this.query(query, [firebaseId]);
    return result.rows[0];
  }

  async getAllProjects() {
    const query = 'SELECT * FROM projects ORDER BY created_at DESC';
    const result = await this.query(query);
    return result.rows;
  }

  // Enhanced project search with estimate filtering
  async searchProjects(searchTerm, limit = 10) {
    const query = `
      SELECT p.*, 
             COUNT(e.id) as estimate_count,
             SUM(e.total_amount) as total_estimated_value,
             SUM(e.budgeted_amount) as total_budgeted_value,
             AVG(CASE 
               WHEN e.budgeted_amount > 0 
               THEN ((e.total_amount - e.budgeted_amount) / e.budgeted_amount) * 100 
               ELSE NULL 
             END) as avg_variance_percent
      FROM projects p
      LEFT JOIN entities e ON p.id = e.project_id 
      WHERE e.entity_type = 'estimate_row'
      AND (LOWER(p.title) LIKE LOWER($1) OR LOWER(p.client_name) LIKE LOWER($1))
      GROUP BY p.id
      ORDER BY p.created_at DESC
      LIMIT $2
    `;
    const result = await this.query(query, [`%${searchTerm}%`, limit]);
    return result.rows;
  }

  // Estimate entity operations - exclusively for estimates
  async createEntity(projectId, entityType, costCode, description, taskScope, category, totalAmount, budgetedAmount, rawData, content) {
    // Force entity type to be estimate_row
    const finalEntityType = 'estimate_row';
    
    const query = `
      INSERT INTO entities (project_id, entity_type, cost_code, description, task_scope, category, total_amount, budgeted_amount, raw_data, content)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
      RETURNING *
    `;
    const result = await this.query(query, [
      projectId, finalEntityType, costCode, description, taskScope, category, 
      totalAmount, budgetedAmount, JSON.stringify(rawData), content
    ]);
    return result.rows[0];
  }

  async getEntitiesByProject(projectId) {
    const query = `
      SELECT * FROM entities 
      WHERE project_id = $1 AND entity_type = 'estimate_row'
      ORDER BY created_at DESC
    `;
    const result = await this.query(query, [projectId]);
    return result.rows;
  }

  async getEntitiesByCostCode(projectId, costCode) {
    const query = `
      SELECT * FROM entities 
      WHERE project_id = $1 AND cost_code = $2 AND entity_type = 'estimate_row'
      ORDER BY total_amount DESC
    `;
    const result = await this.query(query, [projectId, costCode]);
    return result.rows;
  }

  async getEntitiesByCategory(projectId, category) {
    const query = `
      SELECT * FROM entities 
      WHERE project_id = $1 AND category = $2 AND entity_type = 'estimate_row'
      ORDER BY total_amount DESC
    `;
    const result = await this.query(query, [projectId, category]);
    return result.rows;
  }

  // Enhanced entity search with estimate-specific filters
  async searchEstimateEntities(searchTerm, filters = {}) {
    const {
      projectId = null,
      category = null,
      costCodePattern = null,
      minAmount = null,
      maxAmount = null,
      hasVariance = null,
      limit = 50
    } = filters;

    let whereConditions = ["entity_type = 'estimate_row'"];
    let params = [limit];
    let paramIndex = 2;

    // Add search term
    if (searchTerm) {
      whereConditions.push(`(
        LOWER(description) LIKE LOWER($${paramIndex}) OR 
        LOWER(cost_code) LIKE LOWER($${paramIndex}) OR
        LOWER(task_scope) LIKE LOWER($${paramIndex}) OR
        LOWER(content) LIKE LOWER($${paramIndex})
      )`);
      params.push(`%${searchTerm}%`);
      paramIndex++;
    }

    // Add filters
    if (projectId) {
      whereConditions.push(`project_id = $${paramIndex}`);
      params.push(projectId);
      paramIndex++;
    }

    if (category) {
      whereConditions.push(`category = $${paramIndex}`);
      params.push(category);
      paramIndex++;
    }

    if (costCodePattern) {
      whereConditions.push(`cost_code ~* $${paramIndex}`);
      params.push(costCodePattern);
      paramIndex++;
    }

    if (minAmount !== null) {
      whereConditions.push(`total_amount >= $${paramIndex}`);
      params.push(minAmount);
      paramIndex++;
    }

    if (maxAmount !== null) {
      whereConditions.push(`total_amount <= $${paramIndex}`);
      params.push(maxAmount);
      paramIndex++;
    }

    if (hasVariance === true) {
      whereConditions.push(`budgeted_amount > 0 AND ABS(total_amount - budgeted_amount) > budgeted_amount * 0.05`);
    } else if (hasVariance === false) {
      whereConditions.push(`budgeted_amount > 0 AND ABS(total_amount - budgeted_amount) <= budgeted_amount * 0.05`);
    }

    const query = `
      SELECT *,
             CASE 
               WHEN budgeted_amount > 0 
               THEN ((total_amount - budgeted_amount) / budgeted_amount) * 100 
               ELSE NULL 
             END as variance_percent
      FROM entities 
      WHERE ${whereConditions.join(' AND ')}
      ORDER BY total_amount DESC
      LIMIT $1
    `;

    const result = await this.query(query, [limit, ...params.slice(1)]);
    return result.rows;
  }

  // Relationship operations - estimate focused
  async createRelationship(sourceId, targetId, type, strength = 1.0) {
    const query = `
      INSERT INTO relationships (source_id, target_id, type, strength)
      VALUES ($1, $2, $3, $4)
      ON CONFLICT DO NOTHING
      RETURNING *
    `;
    const result = await this.query(query, [sourceId, targetId, type, strength]);
    return result.rows[0];
  }

  async getEntityRelationships(entityId) {
    const query = `
      SELECT r.*, 
             source_entity.cost_code as source_cost_code,
             source_entity.description as source_description,
             source_entity.category as source_category,
             source_entity.total_amount as source_amount,
             target_entity.cost_code as target_cost_code,
             target_entity.description as target_description,
             target_entity.category as target_category,
             target_entity.total_amount as target_amount
      FROM relationships r
      JOIN entities source_entity ON r.source_id = source_entity.id
      JOIN entities target_entity ON r.target_id = target_entity.id
      WHERE (r.source_id = $1 OR r.target_id = $1)
      AND source_entity.entity_type = 'estimate_row'
      AND target_entity.entity_type = 'estimate_row'
      ORDER BY r.strength DESC
    `;
    const result = await this.query(query, [entityId]);
    return result.rows;
  }

  async getRelationshipsByType(type, limit = 100) {
    const query = `
      SELECT r.*, 
             source_entity.cost_code as source_cost_code,
             source_entity.category as source_category,
             target_entity.cost_code as target_cost_code,
             target_entity.category as target_category
      FROM relationships r
      JOIN entities source_entity ON r.source_id = source_entity.id
      JOIN entities target_entity ON r.target_id = target_entity.id
      WHERE r.type = $1
      AND source_entity.entity_type = 'estimate_row'
      AND target_entity.entity_type = 'estimate_row'
      ORDER BY r.strength DESC
      LIMIT $2
    `;
    const result = await this.query(query, [type, limit]);
    return result.rows;
  }

  // Embedding operations - estimate focused
  async createEmbedding(entityId, embedding) {
    const query = `
      INSERT INTO embeddings (entity_id, embedding)
      VALUES ($1, $2::vector)
      ON CONFLICT (entity_id) DO UPDATE SET
        embedding = $2::vector
      RETURNING *
    `;
    const result = await this.query(query, [entityId, JSON.stringify(embedding)]);
    return result.rows[0];
  }

  async searchSimilarEntities(queryEmbedding, threshold = 0.7, limit = 10) {
    const query = `
      SELECT e.*, entities.cost_code, entities.description, entities.content,
             entities.category, entities.total_amount, entities.budgeted_amount,
             entities.task_scope, entities.area,
             1 - (e.embedding <=> $1::vector) as similarity
      FROM embeddings e
      JOIN entities ON e.entity_id = entities.id
      WHERE entities.entity_type = 'estimate_row'
      AND 1 - (e.embedding <=> $1::vector) > $2
      ORDER BY e.embedding <=> $1::vector
      LIMIT $3
    `;
    const result = await this.query(query, [JSON.stringify(queryEmbedding), threshold, limit]);
    return result.rows;
  }

  async getEntityEmbedding(entityId) {
    const query = `
      SELECT e.* FROM embeddings e
      JOIN entities ent ON e.entity_id = ent.id
      WHERE e.entity_id = $1 AND ent.entity_type = 'estimate_row'
    `;
    const result = await this.query(query, [entityId]);
    return result.rows[0];
  }

  // Enhanced batch operations for estimate entities
  async createEntitiesBatch(entities) {
    if (!entities || entities.length === 0) return [];

    const client = await this.getClient();
    try {
      await client.query('BEGIN');
      
      const createdEntities = [];
      for (const entity of entities) {
        // Ensure entity type is estimate_row
        const finalEntityType = 'estimate_row';
        
        const query = `
          INSERT INTO entities (project_id, entity_type, cost_code, description, task_scope, category, total_amount, budgeted_amount, raw_data, content)
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
          RETURNING *
        `;
        const result = await client.query(query, [
          entity.projectId, finalEntityType, entity.costCode, entity.description, 
          entity.taskScope, entity.category, entity.totalAmount, entity.budgetedAmount,
          JSON.stringify(entity.rawData), entity.content
        ]);
        createdEntities.push(result.rows[0]);
      }
      
      await client.query('COMMIT');
      console.log(`✅ Created ${createdEntities.length} estimate entities in batch`);
      return createdEntities;
      
    } catch (error) {
      await client.query('ROLLBACK');
      console.error('❌ Batch estimate entity creation failed:', error.message);
      throw error;
    } finally {
      client.release();
    }
  }

  async createEmbeddingsBatch(embeddings) {
    if (!embeddings || embeddings.length === 0) return [];

    const client = await this.getClient();
    try {
      await client.query('BEGIN');
      
      const createdEmbeddings = [];
      for (const emb of embeddings) {
        const query = `
          INSERT INTO embeddings (entity_id, embedding)
          VALUES ($1, $2::vector)
          ON CONFLICT (entity_id) DO UPDATE SET
            embedding = $2::vector
          RETURNING *
        `;
        const result = await client.query(query, [emb.entityId, JSON.stringify(emb.embedding)]);
        createdEmbeddings.push(result.rows[0]);
      }
      
      await client.query('COMMIT');
      console.log(`✅ Created ${createdEmbeddings.length} estimate embeddings in batch`);
      return createdEmbeddings;
      
    } catch (error) {
      await client.query('ROLLBACK');
      console.error('❌ Batch estimate embedding creation failed:', error.message);
      throw error;
    } finally {
      client.release();
    }
  }

  // Enhanced analytics and insights - estimate focused
  async getProjectStats(projectId) {
    const query = `
      SELECT 
        COUNT(*) as total_entities,
        COUNT(*) as estimates,
        SUM(total_amount) as total_amount,
        SUM(budgeted_amount) as budgeted_amount,
        SUM(total_amount) - SUM(budgeted_amount) as total_variance,
        COUNT(DISTINCT cost_code) as unique_cost_codes,
        COUNT(DISTINCT category) as unique_categories,
        AVG(total_amount) as avg_estimate_amount,
        MAX(total_amount) as max_estimate_amount,
        MIN(total_amount) as min_estimate_amount,
        COUNT(CASE WHEN budgeted_amount > 0 THEN 1 END) as items_with_budget,
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND ABS(total_amount - budgeted_amount) <= budgeted_amount * 0.05 
          THEN 1 
        END) as on_budget_items,
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND total_amount > budgeted_amount * 1.05 
          THEN 1 
        END) as over_budget_items,
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND total_amount < budgeted_amount * 0.95 
          THEN 1 
        END) as under_budget_items
      FROM entities 
      WHERE project_id = $1 AND entity_type = 'estimate_row'
    `;
    const result = await this.query(query, [projectId]);
    return result.rows[0];
  }

  async getCostCodeSummary(projectId) {
    const query = `
      SELECT 
        cost_code,
        category,
        COUNT(*) as item_count,
        SUM(total_amount) as total_amount,
        SUM(budgeted_amount) as budgeted_amount,
        SUM(total_amount) - SUM(budgeted_amount) as variance,
        AVG(total_amount) as avg_amount,
        MAX(total_amount) as max_amount,
        MIN(total_amount) as min_amount,
        AVG(CASE 
          WHEN budgeted_amount > 0 
          THEN ((total_amount - budgeted_amount) / budgeted_amount) * 100 
          ELSE NULL 
        END) as avg_variance_percent,
        COUNT(CASE WHEN budgeted_amount > 0 THEN 1 END) as items_with_budget
      FROM entities 
      WHERE project_id = $1 AND cost_code IS NOT NULL AND entity_type = 'estimate_row'
      GROUP BY cost_code, category
      ORDER BY ABS(SUM(total_amount) - SUM(budgeted_amount)) DESC
    `;
    const result = await this.query(query, [projectId]);
    return result.rows;
  }

  // Enhanced estimate analytics
  async getEstimateAnalytics(projectId = null) {
    const whereClause = projectId 
      ? `WHERE project_id = $1 AND entity_type = 'estimate_row'`
      : `WHERE entity_type = 'estimate_row'`;
    
    const params = projectId ? [projectId] : [];
    
    const query = `
      SELECT 
        -- Basic counts and totals
        COUNT(*) as total_estimates,
        SUM(total_amount) as total_estimated_value,
        SUM(budgeted_amount) as total_budgeted_value,
        AVG(total_amount) as avg_estimate_value,
        
        -- Variance analysis
        SUM(total_amount) - SUM(budgeted_amount) as overall_variance,
        COUNT(CASE WHEN budgeted_amount > 0 THEN 1 END) as estimates_with_budget,
        
        -- Budget performance categories
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND ABS(total_amount - budgeted_amount) <= budgeted_amount * 0.05 
          THEN 1 
        END) as on_budget_count,
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND total_amount > budgeted_amount * 1.05 
          THEN 1 
        END) as over_budget_count,
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND total_amount < budgeted_amount * 0.95 
          THEN 1 
        END) as under_budget_count,
        
        -- High variance items
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND ABS(total_amount - budgeted_amount) > budgeted_amount * 0.2 
          THEN 1 
        END) as high_variance_count,
        
        -- Category and code diversity
        COUNT(DISTINCT category) as unique_categories,
        COUNT(DISTINCT cost_code) as unique_cost_codes,
        COUNT(DISTINCT project_id) as projects_involved,
        
        -- Value distribution
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_amount) as median_value,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_amount) as q1_value,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_amount) as q3_value,
        MAX(total_amount) as max_value,
        MIN(total_amount) as min_value
      FROM entities 
      ${whereClause}
    `;
    
    const result = await this.query(query, params);
    return result.rows[0];
  }

  async getCategoryAnalytics(projectId = null) {
    const whereClause = projectId 
      ? `WHERE project_id = $1 AND entity_type = 'estimate_row'`
      : `WHERE entity_type = 'estimate_row'`;
    
    const params = projectId ? [projectId] : [];
    
    const query = `
      SELECT 
        category,
        COUNT(*) as item_count,
        SUM(total_amount) as category_total,
        AVG(total_amount) as avg_amount,
        SUM(budgeted_amount) as budgeted_total,
        SUM(total_amount) - SUM(budgeted_amount) as category_variance,
        COUNT(DISTINCT cost_code) as unique_cost_codes,
        COUNT(DISTINCT project_id) as projects_count,
        
        -- Budget performance by category
        COUNT(CASE WHEN budgeted_amount > 0 THEN 1 END) as with_budget,
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND ABS(total_amount - budgeted_amount) <= budgeted_amount * 0.05 
          THEN 1 
        END) as on_budget,
        COUNT(CASE 
          WHEN budgeted_amount > 0 AND total_amount > budgeted_amount * 1.05 
          THEN 1 
        END) as over_budget,
        
        -- Value statistics
        MAX(total_amount) as max_item_value,
        MIN(total_amount) as min_item_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_amount) as median_value
        
      FROM entities 
      ${whereClause}
      GROUP BY category
      ORDER BY category_total DESC
    `;
    
    const result = await this.query(query, params);
    return result.rows;
  }

  async getVarianceAnalysis(projectId = null, minVariancePercent = 10) {
    const whereClause = projectId 
      ? `WHERE project_id = $1 AND entity_type = 'estimate_row' AND budgeted_amount > 0`
      : `WHERE entity_type = 'estimate_row' AND budgeted_amount > 0`;
    
    const params = projectId ? [projectId] : [];
    
    const query = `
      SELECT 
        cost_code,
        description,
        category,
        total_amount,
        budgeted_amount,
        total_amount - budgeted_amount as variance,
        ((total_amount - budgeted_amount) / budgeted_amount) * 100 as variance_percent,
        CASE 
          WHEN total_amount > budgeted_amount * 1.2 THEN 'significant_over'
          WHEN total_amount > budgeted_amount * 1.05 THEN 'over'
          WHEN total_amount < budgeted_amount * 0.8 THEN 'significant_under'
          WHEN total_amount < budgeted_amount * 0.95 THEN 'under'
          ELSE 'on_budget'
        END as variance_category,
        project_id
      FROM entities 
      ${whereClause}
      AND ABS(((total_amount - budgeted_amount) / budgeted_amount) * 100) >= $${params.length + 1}
      ORDER BY ABS(total_amount - budgeted_amount) DESC
    `;
    
    const result = await this.query(query, [...params, minVariancePercent]);
    return result.rows;
  }

  async getTopEstimatesByValue(projectId = null, limit = 20) {
    const whereClause = projectId 
      ? `WHERE project_id = $1 AND entity_type = 'estimate_row'`
      : `WHERE entity_type = 'estimate_row'`;
    
    const params = projectId ? [projectId, limit] : [limit];
    const limitParam = projectId ? '$2' : '$1';
    
    const query = `
      SELECT 
        cost_code,
        description,
        category,
        total_amount,
        budgeted_amount,
        total_amount - budgeted_amount as variance,
        CASE 
          WHEN budgeted_amount > 0 
          THEN ((total_amount - budgeted_amount) / budgeted_amount) * 100 
          ELSE NULL 
        END as variance_percent,
        task_scope,
        area,
        project_id
      FROM entities 
      ${whereClause}
      ORDER BY total_amount DESC
      LIMIT ${limitParam}
    `;
    
    const result = await this.query(query, params);
    return result.rows;
  }

  // Advanced estimate queries
  async getEstimatesByBudgetHealth(projectId = null) {
    const whereClause = projectId 
      ? `WHERE project_id = $1 AND entity_type = 'estimate_row' AND budgeted_amount > 0`
      : `WHERE entity_type = 'estimate_row' AND budgeted_amount > 0`;
    
    const params = projectId ? [projectId] : [];
    
    const query = `
      SELECT 
        CASE 
          WHEN ABS(total_amount - budgeted_amount) <= budgeted_amount * 0.05 THEN 'excellent'
          WHEN ABS(total_amount - budgeted_amount) <= budgeted_amount * 0.15 THEN 'good'
          WHEN ABS(total_amount - budgeted_amount) <= budgeted_amount * 0.30 THEN 'fair'
          ELSE 'poor'
        END as budget_health,
        COUNT(*) as item_count,
        SUM(total_amount) as total_value,
        AVG(total_amount) as avg_value,
        AVG(((total_amount - budgeted_amount) / budgeted_amount) * 100) as avg_variance_percent
      FROM entities 
      ${whereClause}
      GROUP BY budget_health
      ORDER BY 
        CASE budget_health
          WHEN 'excellent' THEN 1
          WHEN 'good' THEN 2
          WHEN 'fair' THEN 3
          WHEN 'poor' THEN 4
        END
    `;
    
    const result = await this.query(query, params);
    return result.rows;
  }

  async getEstimateTrends(projectId = null, timeframe = '30 days') {
    const whereClause = projectId 
      ? `WHERE project_id = $1 AND entity_type = 'estimate_row' AND created_at >= NOW() - INTERVAL '${timeframe}'`
      : `WHERE entity_type = 'estimate_row' AND created_at >= NOW() - INTERVAL '${timeframe}'`;
    
    const params = projectId ? [projectId] : [];
    
    const query = `
      SELECT 
        DATE_TRUNC('day', created_at) as date,
        COUNT(*) as estimates_added,
        SUM(total_amount) as daily_estimate_value,
        AVG(total_amount) as avg_estimate_value,
        COUNT(DISTINCT cost_code) as unique_cost_codes_added
      FROM entities 
      ${whereClause}
      GROUP BY DATE_TRUNC('day', created_at)
      ORDER BY date DESC
    `;
    
    const result = await this.query(query, params);
    return result.rows;
  }

  // Cleanup and maintenance - estimate focused
  async deleteProjectData(projectId) {
    const client = await this.getClient();
    try {
      await client.query('BEGIN');
      
      // Delete in correct order due to foreign key constraints
      // Only delete estimate-related data
      await client.query(`
        DELETE FROM embeddings 
        WHERE entity_id IN (
          SELECT id FROM entities 
          WHERE project_id = $1 AND entity_type = 'estimate_row'
        )
      `, [projectId]);
      
      await client.query(`
        DELETE FROM relationships 
        WHERE source_id IN (
          SELECT id FROM entities 
          WHERE project_id = $1 AND entity_type = 'estimate_row'
        ) OR target_id IN (
          SELECT id FROM entities 
          WHERE project_id = $1 AND entity_type = 'estimate_row'
        )
      `, [projectId]);
      
      await client.query(`
        DELETE FROM entities 
        WHERE project_id = $1 AND entity_type = 'estimate_row'
      `, [projectId]);
      
      // Only delete project if no other entity types remain
      const remainingEntities = await client.query(
        'SELECT COUNT(*) as count FROM entities WHERE project_id = $1', 
        [projectId]
      );
      
      if (parseInt(remainingEntities.rows[0].count) === 0) {
        await client.query('DELETE FROM projects WHERE id = $1', [projectId]);
      }
      
      await client.query('COMMIT');
      console.log(`✅ Deleted all estimate data for project ${projectId}`);
      
    } catch (error) {
      await client.query('ROLLBACK');
      console.error('❌ Failed to delete project estimate data:', error.message);
      throw error;
    } finally {
      client.release();
    }
  }

  async cleanupLowQualityEstimates(minAmount = 0, requireDescription = true) {
    try {
      console.log('🧹 Cleaning up low-quality estimate data...');
      
      let whereConditions = [`entity_type = 'estimate_row'`];
      let params = [];
      let paramIndex = 1;
      
      if (minAmount > 0) {
        whereConditions.push(`total_amount < $${paramIndex}`);
        params.push(minAmount);
        paramIndex++;
      }
      
      if (requireDescription) {
        whereConditions.push(`(description IS NULL OR LENGTH(TRIM(description)) < 3)`);
      }
      
      const deleteQuery = `
        DELETE FROM entities 
        WHERE ${whereConditions.join(' AND ')}
      `;
      
      const result = await this.query(deleteQuery, params);
      console.log(`✅ Removed ${result.rowCount} low-quality estimate entries`);
      return result.rowCount;
      
    } catch (error) {
      console.error('❌ Error cleaning up estimate data:', error.message);
      throw error;
    }
  }

  // System statistics - estimate focused
  async getSystemStats() {
    const query = `
      SELECT 
        (SELECT COUNT(*) FROM projects) as total_projects,
        (SELECT COUNT(*) FROM entities WHERE entity_type = 'estimate_row') as total_estimates,
        (SELECT COUNT(*) FROM relationships) as total_relationships,
        (SELECT COUNT(*) FROM embeddings) as total_embeddings,
        (SELECT SUM(total_amount) FROM entities WHERE entity_type = 'estimate_row') as total_estimated_value,
        (SELECT SUM(budgeted_amount) FROM entities WHERE entity_type = 'estimate_row') as total_budgeted_value,
        (SELECT COUNT(DISTINCT cost_code) FROM entities WHERE entity_type = 'estimate_row' AND cost_code IS NOT NULL) as unique_cost_codes,
        (SELECT COUNT(DISTINCT category) FROM entities WHERE entity_type = 'estimate_row' AND category IS NOT NULL) as unique_categories,
        (SELECT AVG(total_amount) FROM entities WHERE entity_type = 'estimate_row' AND total_amount > 0) as avg_estimate_value
    `;
    
    const result = await this.query(query);
    return result.rows[0];
  }

  // Close database connections
  async close() {
    await this.pool.end();
    console.log('🔌 Database connections closed');
  }
}

// Export singleton instance
const estimateFocusedDb = new EstimateFocusedDatabaseService();
export default estimateFocusedDb;