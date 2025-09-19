/**
 * Database Service
 * PostgreSQL with pgvector support for Graph RAG system
 */

import pkg from 'pg';
import dotenv from 'dotenv';

const { Pool } = pkg;
dotenv.config();

class DatabaseService {
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
      
      client.release();
    } catch (error) {
      console.error('❌ Database connection failed:', error.message);
      process.exit(1);
    }
  }

  // Basic query method
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

  // Projects operations
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

  // Entity operations
  async createEntity(projectId, entityType, costCode, description, taskScope, category, totalAmount, budgetedAmount, rawData, content) {
    const query = `
      INSERT INTO entities (project_id, entity_type, cost_code, description, task_scope, category, total_amount, budgeted_amount, raw_data, content)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
      RETURNING *
    `;
    const result = await this.query(query, [
      projectId, entityType, costCode, description, taskScope, category, 
      totalAmount, budgetedAmount, JSON.stringify(rawData), content
    ]);
    return result.rows[0];
  }

  async getEntitiesByProject(projectId) {
    const query = 'SELECT * FROM entities WHERE project_id = $1 ORDER BY created_at DESC';
    const result = await this.query(query, [projectId]);
    return result.rows;
  }

  async getEntitiesByCostCode(projectId, costCode) {
    const query = 'SELECT * FROM entities WHERE project_id = $1 AND cost_code = $2';
    const result = await this.query(query, [projectId, costCode]);
    return result.rows;
  }

  async getEntitiesByCategory(projectId, category) {
    const query = 'SELECT * FROM entities WHERE project_id = $1 AND category = $2';
    const result = await this.query(query, [projectId, category]);
    return result.rows;
  }

  // Relationship operations
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
             target_entity.cost_code as target_cost_code,
             target_entity.description as target_description
      FROM relationships r
      JOIN entities source_entity ON r.source_id = source_entity.id
      JOIN entities target_entity ON r.target_id = target_entity.id
      WHERE r.source_id = $1 OR r.target_id = $1
      ORDER BY r.strength DESC
    `;
    const result = await this.query(query, [entityId]);
    return result.rows;
  }

  async getRelationshipsByType(type, limit = 100) {
    const query = `
      SELECT r.*, 
             source_entity.cost_code as source_cost_code,
             target_entity.cost_code as target_cost_code
      FROM relationships r
      JOIN entities source_entity ON r.source_id = source_entity.id
      JOIN entities target_entity ON r.target_id = target_entity.id
      WHERE r.type = $1
      ORDER BY r.strength DESC
      LIMIT $2
    `;
    const result = await this.query(query, [type, limit]);
    return result.rows;
  }

  // Embedding operations
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
             1 - (e.embedding <=> $1::vector) as similarity
      FROM embeddings e
      JOIN entities ON e.entity_id = entities.id
      WHERE 1 - (e.embedding <=> $1::vector) > $2
      ORDER BY e.embedding <=> $1::vector
      LIMIT $3
    `;
    const result = await this.query(query, [JSON.stringify(queryEmbedding), threshold, limit]);
    return result.rows;
  }

  async getEntityEmbedding(entityId) {
    const query = 'SELECT * FROM embeddings WHERE entity_id = $1';
    const result = await this.query(query, [entityId]);
    return result.rows[0];
  }

  // Batch operations for efficiency
  async createEntitiesBatch(entities) {
    if (!entities || entities.length === 0) return [];

    const client = await this.getClient();
    try {
      await client.query('BEGIN');
      
      const createdEntities = [];
      for (const entity of entities) {
        const query = `
          INSERT INTO entities (project_id, entity_type, cost_code, description, task_scope, category, total_amount, budgeted_amount, raw_data, content)
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
          RETURNING *
        `;
        const result = await client.query(query, [
          entity.projectId, entity.entityType, entity.costCode, entity.description, 
          entity.taskScope, entity.category, entity.totalAmount, entity.budgetedAmount,
          JSON.stringify(entity.rawData), entity.content
        ]);
        createdEntities.push(result.rows[0]);
      }
      
      await client.query('COMMIT');
      console.log(`✅ Created ${createdEntities.length} entities in batch`);
      return createdEntities;
      
    } catch (error) {
      await client.query('ROLLBACK');
      console.error('❌ Batch entity creation failed:', error.message);
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
      console.log(`✅ Created ${createdEmbeddings.length} embeddings in batch`);
      return createdEmbeddings;
      
    } catch (error) {
      await client.query('ROLLBACK');
      console.error('❌ Batch embedding creation failed:', error.message);
      throw error;
    } finally {
      client.release();
    }
  }

  // Analytics and insights
  async getProjectStats(projectId) {
    const query = `
      SELECT 
        COUNT(*) as total_entities,
        COUNT(CASE WHEN entity_type = 'estimate_row' THEN 1 END) as estimates,
        COUNT(CASE WHEN entity_type = 'consumed_row' THEN 1 END) as consumed,
        SUM(total_amount) as total_amount,
        SUM(budgeted_amount) as budgeted_amount,
        COUNT(DISTINCT cost_code) as unique_cost_codes,
        COUNT(DISTINCT category) as unique_categories
      FROM entities 
      WHERE project_id = $1
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
        SUM(total_amount) - SUM(budgeted_amount) as variance
      FROM entities 
      WHERE project_id = $1 AND cost_code IS NOT NULL
      GROUP BY cost_code, category
      ORDER BY ABS(SUM(total_amount) - SUM(budgeted_amount)) DESC
    `;
    const result = await this.query(query, [projectId]);
    return result.rows;
  }

  // Cleanup and maintenance
  async deleteProjectData(projectId) {
    const client = await this.getClient();
    try {
      await client.query('BEGIN');
      
      // Delete in correct order due to foreign key constraints
      await client.query('DELETE FROM embeddings WHERE entity_id IN (SELECT id FROM entities WHERE project_id = $1)', [projectId]);
      await client.query('DELETE FROM relationships WHERE source_id IN (SELECT id FROM entities WHERE project_id = $1) OR target_id IN (SELECT id FROM entities WHERE project_id = $1)', [projectId]);
      await client.query('DELETE FROM entities WHERE project_id = $1', [projectId]);
      
      await client.query('COMMIT');
      console.log(`✅ Deleted all data for project ${projectId}`);
      
    } catch (error) {
      await client.query('ROLLBACK');
      console.error('❌ Failed to delete project data:', error.message);
      throw error;
    } finally {
      client.release();
    }
  }

  // Close database connections
  async close() {
    await this.pool.end();
    console.log('🔌 Database connections closed');
  }
}

// Export singleton instance
const db = new DatabaseService();
export default db;