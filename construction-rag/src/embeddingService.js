/**
 * Embedding Service
 * Generates and manages OpenAI embeddings for construction entities
 */

import OpenAI from 'openai';
import dotenv from 'dotenv';

dotenv.config();

class EmbeddingService {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
    
    this.model = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
    this.maxBatchSize = parseInt(process.env.MAX_EMBEDDING_BATCH_SIZE) || 100;
    this.similarityThreshold = parseFloat(process.env.SIMILARITY_THRESHOLD) || 0.7;
    
    // Rate limiting
    this.requestCount = 0;
    this.lastResetTime = Date.now();
    this.maxRequestsPerMinute = 500; // Conservative limit
    
    console.log('✅ Embedding service initialized');
    console.log(`🤖 Model: ${this.model}`);
    console.log(`📊 Batch size: ${this.maxBatchSize}`);
    console.log(`🎯 Similarity threshold: ${this.similarityThreshold}`);
  }

  // Rate limiting helper
  async checkRateLimit() {
    const now = Date.now();
    const timeSinceReset = now - this.lastResetTime;
    
    // Reset counter every minute
    if (timeSinceReset > 60000) {
      this.requestCount = 0;
      this.lastResetTime = now;
    }
    
    // Wait if we're at the limit
    if (this.requestCount >= this.maxRequestsPerMinute) {
      const waitTime = 60000 - timeSinceReset;
      console.log(`⏳ Rate limit reached, waiting ${Math.ceil(waitTime/1000)}s...`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      this.requestCount = 0;
      this.lastResetTime = Date.now();
    }
  }

  // Generate single embedding
  async generateEmbedding(text) {
    try {
      await this.checkRateLimit();
      
      if (!text || typeof text !== 'string' || text.trim().length === 0) {
        throw new Error('Invalid text provided for embedding');
      }

      const startTime = Date.now();
      
      const response = await this.openai.embeddings.create({
        model: this.model,
        input: text.trim()
      });

      this.requestCount++;
      const duration = Date.now() - startTime;
      
      console.log(`🔮 Generated embedding in ${duration}ms (${text.length} chars)`);
      
      return response.data[0].embedding;
      
    } catch (error) {
      console.error('❌ Error generating embedding:', error.message);
      
      // Handle specific OpenAI errors
      if (error.status === 429) {
        console.log('⏳ Hit rate limit, waiting 60s...');
        await new Promise(resolve => setTimeout(resolve, 60000));
        return this.generateEmbedding(text); // Retry
      }
      
      throw error;
    }
  }

  // Generate embeddings for multiple texts (batch processing)
  async generateEmbeddingsBatch(texts) {
    if (!Array.isArray(texts) || texts.length === 0) {
      return [];
    }

    console.log(`🔄 Generating embeddings for ${texts.length} texts...`);
    const embeddings = [];
    const batchSize = Math.min(this.maxBatchSize, texts.length);
    
    try {
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);
        console.log(`📦 Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(texts.length/batchSize)} (${batch.length} items)`);
        
        await this.checkRateLimit();
        
        const startTime = Date.now();
        
        const response = await this.openai.embeddings.create({
          model: this.model,
          input: batch.map(text => (text || '').trim()).filter(text => text.length > 0)
        });

        this.requestCount++;
        const duration = Date.now() - startTime;
        
        const batchEmbeddings = response.data.map(item => item.embedding);
        embeddings.push(...batchEmbeddings);
        
        console.log(`✅ Batch completed in ${duration}ms (${batchEmbeddings.length} embeddings)`);
        
        // Small delay between batches to be respectful
        if (i + batchSize < texts.length) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
      
      console.log(`🎉 Generated ${embeddings.length} embeddings total`);
      return embeddings;
      
    } catch (error) {
      console.error('❌ Batch embedding generation failed:', error.message);
      throw error;
    }
  }

  // Generate embeddings for entities and store in database
  async processEntities(entities, db) {
    if (!entities || entities.length === 0) {
      console.log('⚠️  No entities to process');
      return [];
    }

    console.log(`🔄 Processing embeddings for ${entities.length} entities...`);
    
    try {
      // Step 1: Generate embeddings for all entity content
      const contents = entities.map(entity => entity.content || '');
      const embeddings = await this.generateEmbeddingsBatch(contents);
      
      if (embeddings.length !== entities.length) {
        throw new Error(`Embedding count mismatch: expected ${entities.length}, got ${embeddings.length}`);
      }
      
      // Step 2: Store entities in database first
      console.log('💾 Storing entities in database...');
      const storedEntities = await db.createEntitiesBatch(entities);
      
      // Step 3: Prepare embeddings data
      const embeddingData = storedEntities.map((entity, index) => ({
        entityId: entity.id,
        embedding: embeddings[index]
      }));
      
      // Step 4: Store embeddings in database
      console.log('🔮 Storing embeddings in database...');
      const storedEmbeddings = await db.createEmbeddingsBatch(embeddingData);
      
      console.log(`✅ Successfully processed ${storedEmbeddings.length} embeddings`);
      
      return {
        entities: storedEntities,
        embeddings: storedEmbeddings,
        total: storedEmbeddings.length
      };
      
    } catch (error) {
      console.error('❌ Error processing entities:', error.message);
      throw error;
    }
  }

  // Search for similar entities using embeddings
  async searchSimilar(queryText, db, options = {}) {
    const {
      threshold = this.similarityThreshold,
      limit = 10,
      projectId = null,
      entityType = null,
      category = null,
      costCode = null
    } = options;

    try {
      console.log(`🔍 Searching for similar entities: "${queryText.substring(0, 100)}..."`);
      
      // Generate embedding for query
      const queryEmbedding = await this.generateEmbedding(queryText);
      
      // Search in database
      let results = await db.searchSimilarEntities(queryEmbedding, threshold, limit);
      
      // Apply additional filters if specified
      if (projectId) {
        results = results.filter(r => r.project_id === projectId);
      }
      
      if (entityType) {
        results = results.filter(r => r.entity_type === entityType);
      }
      
      if (category) {
        results = results.filter(r => r.category === category);
      }
      
      if (costCode) {
        results = results.filter(r => r.cost_code === costCode);
      }
      
      console.log(`🎯 Found ${results.length} similar entities (threshold: ${threshold})`);
      
      // Enrich results with similarity scores and metadata
      return results.map(result => ({
        ...result,
        similarity: parseFloat(result.similarity),
        similarityPercent: Math.round(parseFloat(result.similarity) * 100),
        relevanceScore: this.calculateRelevanceScore(result, queryText)
      }));
      
    } catch (error) {
      console.error('❌ Error searching similar entities:', error.message);
      throw error;
    }
  }

  // Calculate relevance score based on multiple factors
  calculateRelevanceScore(result, queryText) {
    let score = parseFloat(result.similarity) || 0;
    const query = queryText.toLowerCase();
    const content = (result.content || '').toLowerCase();
    const costCode = (result.cost_code || '').toLowerCase();
    
    // Boost score for exact cost code matches
    if (query.includes(costCode) && costCode.length > 0) {
      score += 0.1;
    }
    
    // Boost score for key term matches in content
    const keyTerms = ['material', 'labor', 'subcontractor', 'electrical', 'plumbing', 'concrete'];
    const matchingTerms = keyTerms.filter(term => 
      query.includes(term) && content.includes(term)
    );
    score += matchingTerms.length * 0.05;
    
    // Cap at 1.0
    return Math.min(score, 1.0);
  }

  // Get embedding statistics
  async getEmbeddingStats(db) {
    try {
      const stats = await db.query(`
        SELECT 
          COUNT(*) as total_embeddings,
          COUNT(DISTINCT e.entity_id) as unique_entities,
          AVG(array_length(embedding::float[], 1)) as avg_dimension
        FROM embeddings e
      `);
      
      const entityStats = await db.query(`
        SELECT 
          entity_type,
          COUNT(*) as count
        FROM entities e
        JOIN embeddings emb ON e.id = emb.entity_id
        GROUP BY entity_type
      `);
      
      return {
        totalEmbeddings: parseInt(stats.rows[0]?.total_embeddings) || 0,
        uniqueEntities: parseInt(stats.rows[0]?.unique_entities) || 0,
        avgDimension: parseInt(stats.rows[0]?.avg_dimension) || 0,
        byEntityType: entityStats.rows.reduce((acc, row) => {
          acc[row.entity_type] = parseInt(row.count);
          return acc;
        }, {}),
        model: this.model,
        threshold: this.similarityThreshold
      };
      
    } catch (error) {
      console.error('❌ Error getting embedding stats:', error.message);
      return {
        error: error.message
      };
    }
  }

  // Find entities without embeddings
  async findEntitiesWithoutEmbeddings(db) {
    try {
      const result = await db.query(`
        SELECT e.*
        FROM entities e
        LEFT JOIN embeddings emb ON e.id = emb.entity_id
        WHERE emb.entity_id IS NULL
        ORDER BY e.created_at DESC
      `);
      
      return result.rows;
      
    } catch (error) {
      console.error('❌ Error finding entities without embeddings:', error.message);
      throw error;
    }
  }

  // Regenerate embeddings for entities (useful for model updates)
  async regenerateEmbeddings(entityIds, db) {
    try {
      console.log(`🔄 Regenerating embeddings for ${entityIds.length} entities...`);
      
      // Get entities
      const entities = [];
      for (const id of entityIds) {
        const result = await db.query('SELECT * FROM entities WHERE id = $1', [id]);
        if (result.rows[0]) {
          entities.push(result.rows[0]);
        }
      }
      
      if (entities.length === 0) {
        console.log('⚠️  No entities found for regeneration');
        return [];
      }
      
      // Generate new embeddings
      const contents = entities.map(entity => entity.content || '');
      const embeddings = await this.generateEmbeddingsBatch(contents);
      
      // Update embeddings in database
      const updatedEmbeddings = [];
      for (let i = 0; i < entities.length; i++) {
        const result = await db.query(
          'UPDATE embeddings SET embedding = $1::vector WHERE entity_id = $2 RETURNING *',
          [JSON.stringify(embeddings[i]), entities[i].id]
        );
        updatedEmbeddings.push(result.rows[0]);
      }
      
      console.log(`✅ Regenerated ${updatedEmbeddings.length} embeddings`);
      return updatedEmbeddings;
      
    } catch (error) {
      console.error('❌ Error regenerating embeddings:', error.message);
      throw error;
    }
  }

  // Test embedding service
  async testEmbeddingService() {
    try {
      console.log('🧪 Testing embedding service...');
      
      const testText = 'Test construction project with electrical work costing $5000';
      const embedding = await this.generateEmbedding(testText);
      
      const isValid = Array.isArray(embedding) && 
                     embedding.length > 0 && 
                     typeof embedding[0] === 'number';
      
      if (isValid) {
        console.log(`✅ Embedding service test passed (dimension: ${embedding.length})`);
        return true;
      } else {
        throw new Error('Invalid embedding format');
      }
      
    } catch (error) {
      console.error('❌ Embedding service test failed:', error.message);
      return false;
    }
  }
}

// Export singleton instance
const embeddingService = new EmbeddingService();
export default embeddingService;