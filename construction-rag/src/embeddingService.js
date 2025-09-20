/**
 * Optimized Embedding Service
 * High-performance embeddings with caching and batching for estimate data
 * FIXED: Better UUID validation and error handling
 */

import OpenAI from 'openai';
import crypto from 'crypto';
import dotenv from 'dotenv';

dotenv.config();

class OptimizedEmbeddingService {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      organization: process.env.OPENAI_ORG_ID
    });
    
    // Use fastest embedding model
    this.model = process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
    
    // Aggressive optimization settings
    this.maxBatchSize = parseInt(process.env.MAX_EMBEDDING_BATCH_SIZE) || 250; // Increased
    this.similarityThreshold = parseFloat(process.env.SIMILARITY_THRESHOLD) || 0.75; // Higher threshold
    
    // In-memory caching for better performance
    this.embedCache = new Map();
    this.maxCacheSize = parseInt(process.env.EMBED_CACHE_SIZE) || 25000; // Large cache
    this.cacheHits = 0;
    this.cacheMisses = 0;
    
    // Rate limiting - more aggressive for better throughput
    this.requestCount = 0;
    this.lastResetTime = Date.now();
    this.maxRequestsPerMinute = parseInt(process.env.OPENAI_RPM_LIMIT) || 1000;
    
    // Performance monitoring
    this.totalRequests = 0;
    this.totalProcessingTime = 0;
    
    console.log('✅ Optimized Embedding service initialized');
    console.log(`🚀 Model: ${this.model} | Batch: ${this.maxBatchSize} | Cache: ${this.maxCacheSize}`);
  }

  /**
   * Validate UUID format
   */
  isValidUUID(uuid) {
    if (!uuid) return false;
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    return uuidRegex.test(uuid);
  }

  /**
   * Generate hash for content caching
   */
  generateContentHash(text) {
    return crypto.createHash('md5').update(text).digest('hex');
  }

  /**
   * Check and manage cache size
   */
  manageCacheSize() {
    if (this.embedCache.size >= this.maxCacheSize) {
      // Remove oldest 20% of entries (LRU-style)
      const keysToDelete = Array.from(this.embedCache.keys()).slice(0, Math.floor(this.maxCacheSize * 0.2));
      keysToDelete.forEach(key => this.embedCache.delete(key));
      console.log(`🧹 Cache cleaned: removed ${keysToDelete.length} entries`);
    }
  }

  /**
   * Optimized rate limiting check
   */
  async checkRateLimit() {
    const now = Date.now();
    const timeSinceReset = now - this.lastResetTime;
    
    if (timeSinceReset > 60000) {
      this.requestCount = 0;
      this.lastResetTime = now;
    }
    
    if (this.requestCount >= this.maxRequestsPerMinute) {
      const waitTime = 60000 - timeSinceReset;
      console.log(`⏳ Rate limit reached, waiting ${Math.ceil(waitTime/1000)}s...`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
      this.requestCount = 0;
      this.lastResetTime = Date.now();
    }
  }

  /**
   * Generate single embedding with caching
   */
  async generateEmbedding(text) {
    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      throw new Error('Invalid text provided for embedding');
    }

    const cleanText = text.trim();
    const contentHash = this.generateContentHash(cleanText);
    
    // Check cache first
    if (this.embedCache.has(contentHash)) {
      this.cacheHits++;
      return this.embedCache.get(contentHash);
    }
    
    this.cacheMisses++;
    
    try {
      await this.checkRateLimit();
      
      const startTime = Date.now();
      
      const response = await this.openai.embeddings.create({
        model: this.model,
        input: cleanText,
        encoding_format: 'float' // Explicit format for consistency
      });

      this.requestCount++;
      this.totalRequests++;
      
      const embedding = response.data[0].embedding;
      const duration = Date.now() - startTime;
      this.totalProcessingTime += duration;
      
      // Cache the result
      this.manageCacheSize();
      this.embedCache.set(contentHash, embedding);
      
      console.log(`🔮 Generated embedding in ${duration}ms (${cleanText.length} chars)`);
      
      return embedding;
      
    } catch (error) {
      console.error('❌ Error generating embedding:', error.message);
      
      if (error.status === 429) {
        console.log('⏳ Hit rate limit, waiting 60s...');
        await new Promise(resolve => setTimeout(resolve, 60000));
        return this.generateEmbedding(text); // Retry
      }
      
      if (error.status === 400 && error.message.includes('too long')) {
        console.log('✂️ Text too long, truncating...');
        const truncatedText = cleanText.substring(0, 8000); // Safe limit
        return this.generateEmbedding(truncatedText);
      }
      
      throw error;
    }
  }

  /**
   * Optimized batch embedding generation with smart caching
   */
  async generateEmbeddingsBatch(texts) {
    if (!Array.isArray(texts) || texts.length === 0) {
      return [];
    }

    console.log(`🔄 Processing batch of ${texts.length} texts...`);
    const startTime = Date.now();

    // Prepare data structures
    const results = new Array(texts.length);
    const uncachedItems = [];
    let cacheHitsInBatch = 0;

    // Check cache for each text
    texts.forEach((text, index) => {
      if (!text || typeof text !== 'string' || text.trim().length === 0) {
        results[index] = null;
        return;
      }

      const cleanText = text.trim();
      const contentHash = this.generateContentHash(cleanText);
      
      if (this.embedCache.has(contentHash)) {
        results[index] = this.embedCache.get(contentHash);
        cacheHitsInBatch++;
        this.cacheHits++;
      } else {
        uncachedItems.push({
          text: cleanText,
          originalIndex: index,
          hash: contentHash
        });
        this.cacheMisses++;
      }
    });

    console.log(`📊 Cache hits: ${cacheHitsInBatch}/${texts.length} (${((cacheHitsInBatch/texts.length)*100).toFixed(1)}%)`);

    // Process uncached items in optimized batches
    if (uncachedItems.length > 0) {
      await this.processUncachedBatch(uncachedItems, results);
    }

    const totalDuration = Date.now() - startTime;
    console.log(`✅ Batch completed in ${totalDuration}ms (${results.filter(r => r !== null).length} embeddings)`);

    return results;
  }

  /**
   * Process uncached items in optimized batches
   */
  async processUncachedBatch(uncachedItems, results) {
    const batchSize = Math.min(this.maxBatchSize, uncachedItems.length);
    
    for (let i = 0; i < uncachedItems.length; i += batchSize) {
      const batch = uncachedItems.slice(i, i + batchSize);
      const batchTexts = batch.map(item => item.text);
      
      console.log(`📦 Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(uncachedItems.length/batchSize)} (${batch.length} items)`);
      
      try {
        await this.checkRateLimit();
        
        const batchStartTime = Date.now();
        
        const response = await this.openai.embeddings.create({
          model: this.model,
          input: batchTexts,
          encoding_format: 'float'
        });

        this.requestCount++;
        this.totalRequests++;
        
        const batchDuration = Date.now() - batchStartTime;
        this.totalProcessingTime += batchDuration;
        
        // Process batch results
        response.data.forEach((embeddingData, batchIndex) => {
          const item = batch[batchIndex];
          const embedding = embeddingData.embedding;
          
          // Store in results array
          results[item.originalIndex] = embedding;
          
          // Cache the embedding
          this.manageCacheSize();
          this.embedCache.set(item.hash, embedding);
        });
        
        console.log(`✅ Batch processed in ${batchDuration}ms (${response.data.length} embeddings)`);
        
        // Small delay between batches to be respectful
        if (i + batchSize < uncachedItems.length) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        
      } catch (error) {
        console.error(`❌ Batch processing error:`, error.message);
        
        if (error.status === 429) {
          console.log('⏳ Rate limit in batch, waiting 60s...');
          await new Promise(resolve => setTimeout(resolve, 60000));
          i -= batchSize; // Retry this batch
          continue;
        }
        
        // For other errors, try individual processing
        console.log('🔄 Falling back to individual processing...');
        await this.processIndividualFallback(batch, results);
      }
    }
  }

  /**
   * Fallback to individual processing when batch fails
   */
  async processIndividualFallback(failedBatch, results) {
    for (const item of failedBatch) {
      try {
        const embedding = await this.generateEmbedding(item.text);
        results[item.originalIndex] = embedding;
        
        // Small delay between individual requests
        await new Promise(resolve => setTimeout(resolve, 200));
        
      } catch (error) {
        console.error(`❌ Individual fallback failed for item:`, error.message);
        results[item.originalIndex] = null; // Mark as failed
      }
    }
  }

  /**
   * Process entities and generate embeddings efficiently
   * FIXED: Better validation and error handling for UUIDs
   */
  async processEstimateEntities(entities, db) {
    console.log(`🚀 DEBUG: === STARTING EMBEDDING PROCESSING ===`);
    console.log(`📊 DEBUG: Input entities:`, {
      entitiesProvided: !!entities,
      entitiesType: typeof entities,
      entitiesIsArray: Array.isArray(entities),
      entitiesLength: entities?.length || 0
    });

    if (!entities || entities.length === 0) {
      console.log('⚠️ DEBUG: No entities provided for embedding processing');
      return { entities: [], embeddings: [], total: 0 };
    }

    console.log(`🔄 DEBUG: Processing embeddings for ${entities.length} estimate entities...`);
    const startTime = Date.now();
    
    try {
      // Validate entities first with detailed logging
      console.log(`🔍 DEBUG: Validating entity content...`);
      const validEntities = entities.filter((entity, index) => {
        console.log(`📋 DEBUG: Checking entity ${index + 1}:`, {
          hasContent: !!entity.content,
          contentType: typeof entity.content,
          contentLength: entity.content?.length || 0,
          projectId: entity.projectId,
          projectIdType: typeof entity.projectId,
          isValidUUID: this.isValidUUID(entity.projectId),
          costCode: entity.costCode,
          description: entity.description?.substring(0, 30)
        });

        // Check for valid UUID
        if (!this.isValidUUID(entity.projectId)) {
          console.error(`❌ DEBUG: Entity ${index + 1} has invalid projectId (not a UUID):`, {
            projectId: entity.projectId,
            projectIdType: typeof entity.projectId,
            costCode: entity.costCode
          });
          return false;
        }

        if (!entity.content || entity.content.trim().length === 0) {
          console.warn(`⚠️ DEBUG: Skipping entity ${index + 1} with empty content:`, {
            costCode: entity.costCode,
            description: entity.description?.substring(0, 30),
            hasContent: !!entity.content,
            contentLength: entity.content?.length || 0
          });
          return false;
        }
        
        console.log(`✅ DEBUG: Entity ${index + 1} has valid content and UUID (${entity.content.length} chars)`);
        return true;
      });

      console.log(`✅ DEBUG: Content validation complete:`);
      console.log(`   - Input entities: ${entities.length}`);
      console.log(`   - Valid entities: ${validEntities.length}`);
      console.log(`   - Invalid entities: ${entities.length - validEntities.length}`);

      if (validEntities.length === 0) {
        console.log(`❌ DEBUG: No entities with valid content and UUID found!`);
        
        // Debug: Show first few entities to understand the issue
        console.log(`🔍 DEBUG: Sample of invalid entities:`);
        entities.slice(0, 3).forEach((entity, i) => {
          console.log(`   Entity ${i + 1}:`, {
            projectId: entity?.projectId,
            projectIdType: typeof entity?.projectId,
            isValidUUID: this.isValidUUID(entity?.projectId),
            costCode: entity?.costCode,
            description: entity?.description?.substring(0, 30),
            hasContent: !!entity?.content,
            contentType: typeof entity?.content,
            contentValue: entity?.content?.substring(0, 100),
            allKeys: entity ? Object.keys(entity) : 'null'
          });
        });

        return { 
          entities: [], 
          embeddings: [], 
          total: 0,
          debug: {
            reason: 'no_valid_content_or_invalid_uuid',
            totalEntities: entities.length,
            sampleEntity: entities[0] ? {
              projectId: entities[0].projectId,
              projectIdType: typeof entities[0].projectId,
              isValidUUID: this.isValidUUID(entities[0].projectId),
              keys: Object.keys(entities[0])
            } : 'none'
          }
        };
      }

      // Generate embeddings for all entity content
      console.log(`🔮 DEBUG: Generating embeddings for ${validEntities.length} entities...`);
      const contents = validEntities.map(entity => entity.content);
      
      console.log(`📋 DEBUG: Content samples (first 2):`);
      contents.slice(0, 2).forEach((content, i) => {
        console.log(`   Content ${i + 1}: ${content.substring(0, 100)}...`);
      });

      const embeddings = await this.generateEmbeddingsBatch(contents);
      
      console.log(`📊 DEBUG: Embedding generation results:`);
      console.log(`   - Contents sent: ${contents.length}`);
      console.log(`   - Embeddings received: ${embeddings.length}`);
      console.log(`   - Non-null embeddings: ${embeddings.filter(emb => emb !== null).length}`);
      
      // Store entities in database
      console.log('💾 DEBUG: Storing entities in database...');
      console.log(`📊 DEBUG: Sample entity for database storage:`, {
        projectId: validEntities[0].projectId,
        projectIdType: typeof validEntities[0].projectId,
        isValidUUID: this.isValidUUID(validEntities[0].projectId),
        costCode: validEntities[0].costCode,
        category: validEntities[0].category,
        totalAmount: validEntities[0].totalAmount,
        hasContent: !!validEntities[0].content
      });

      let storedEntities = [];
      try {
        storedEntities = await db.createEntitiesBatch(validEntities);
        console.log(`✅ DEBUG: Database storage successful: ${storedEntities.length} entities`);
      } catch (dbError) {
        console.error(`❌ DEBUG: Database storage failed:`, dbError.message);
        console.error(`❌ DEBUG: First entity projectId:`, validEntities[0]?.projectId);
        console.error(`❌ DEBUG: Is valid UUID?:`, this.isValidUUID(validEntities[0]?.projectId));
        
        // Additional debugging
        if (dbError.message.includes('invalid input syntax for type uuid')) {
          console.error(`❌ DEBUG: UUID validation failed. Expected format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`);
          console.error(`❌ DEBUG: Received projectId: "${validEntities[0]?.projectId}"`);
        }
        
        throw dbError;
      }

      if (storedEntities.length === 0) {
        console.error(`❌ DEBUG: Failed to store entities in database!`);
        throw new Error('Failed to store entities in database');
      }

      // Prepare embedding data for successful embeddings only
      console.log(`🔗 DEBUG: Preparing embedding data for storage...`);
      const embeddingData = [];
      storedEntities.forEach((entity, index) => {
        if (embeddings[index] !== null) {
          embeddingData.push({
            entityId: entity.id,
            embedding: embeddings[index]
          });
          console.log(`📎 DEBUG: Added embedding for entity ${entity.id} (${entity.cost_code})`);
        } else {
          console.warn(`⚠️ DEBUG: No embedding for entity ${entity.id} (${entity.cost_code})`);
        }
      });
      
      // Store embeddings in database
      console.log(`🔮 DEBUG: Storing ${embeddingData.length} embeddings in database...`);
      const storedEmbeddings = await db.createEmbeddingsBatch(embeddingData);
      
      console.log(`✅ DEBUG: Embedding storage results:`);
      console.log(`   - Embeddings to store: ${embeddingData.length}`);
      console.log(`   - Embeddings stored: ${storedEmbeddings.length}`);
      
      const totalDuration = Date.now() - startTime;
      console.log(`🎉 DEBUG: Processing completed in ${totalDuration}ms`);
      console.log(`📊 DEBUG: Final counts: ${storedEntities.length} entities, ${storedEmbeddings.length} embeddings`);
      
      return {
        entities: storedEntities,
        embeddings: storedEmbeddings,
        total: storedEmbeddings.length,
        processingStats: {
          totalDuration,
          cacheHitRate: this.cacheHits / (this.cacheHits + this.cacheMisses),
          averageEmbeddingTime: totalDuration / Math.max(1, storedEmbeddings.length)
        },
        debug: {
          inputEntities: entities.length,
          validEntities: validEntities.length,
          successfulEmbeddings: embeddings.filter(e => e !== null).length,
          storedEntities: storedEntities.length,
          storedEmbeddings: storedEmbeddings.length
        }
      };
      
    } catch (error) {
      console.error('❌ DEBUG: Error processing estimate entities:', error.message);
      console.error('❌ DEBUG: Error stack:', error.stack);
      throw error;
    }
  }

  /**
   * Search for similar entities using optimized vector search
   */
  async searchSimilar(queryText, db, options = {}) {
    const {
      threshold = this.similarityThreshold,
      limit = 10,
      projectId = null,
      category = null,
      costCodePattern = null,
      minAmount = null,
      maxAmount = null
    } = options;

    try {
      console.log(`🔍 Searching similar entities: "${queryText.substring(0, 80)}..."`);
      const startTime = Date.now();
      
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(queryText);
      
      // Search in database with higher limit for better filtering
      const searchLimit = Math.min(limit * 3, 50); // Get more results to filter
      let results = await db.searchSimilarEntities(queryEmbedding, threshold * 0.9, searchLimit); // Slightly lower threshold
      
      // Apply post-search filters
      results = this.applySearchFilters(results, {
        projectId,
        category,
        costCodePattern,
        minAmount,
        maxAmount
      });
      
      // Limit final results
      results = results.slice(0, limit);
      
      // Enhance results with additional scoring
      const enhancedResults = results.map(result => ({
        ...result,
        similarity: parseFloat(result.similarity),
        similarityPercent: Math.round(parseFloat(result.similarity) * 100),
        relevanceScore: this.calculateEstimateRelevanceScore(result, queryText),
        costPerUnit: this.calculateCostPerUnit(result)
      }));
      
      const searchDuration = Date.now() - startTime;
      console.log(`🎯 Found ${enhancedResults.length} similar entities in ${searchDuration}ms`);
      console.log(`📊 Similarity range: ${enhancedResults.length > 0 ? `${enhancedResults[enhancedResults.length-1].similarityPercent}% - ${enhancedResults[0].similarityPercent}%` : 'N/A'}`);
      
      return enhancedResults;
      
    } catch (error) {
      console.error('❌ Error searching similar entities:', error.message);
      throw error;
    }
  }

  /**
   * Apply post-search filters to results
   */
  applySearchFilters(results, filters) {
    let filtered = results;
    
    if (filters.projectId) {
      filtered = filtered.filter(r => r.project_id === filters.projectId);
    }
    
    if (filters.category) {
      filtered = filtered.filter(r => r.category === filters.category);
    }
    
    if (filters.costCodePattern) {
      const pattern = new RegExp(filters.costCodePattern, 'i');
      filtered = filtered.filter(r => pattern.test(r.cost_code || ''));
    }
    
    if (filters.minAmount !== null) {
      filtered = filtered.filter(r => (r.total_amount || 0) >= filters.minAmount);
    }
    
    if (filters.maxAmount !== null) {
      filtered = filtered.filter(r => (r.total_amount || 0) <= filters.maxAmount);
    }
    
    return filtered;
  }

  /**
   * Calculate enhanced relevance score for estimate results
   */
  calculateEstimateRelevanceScore(result, queryText) {
    let score = parseFloat(result.similarity) || 0;
    const query = queryText.toLowerCase();
    const content = (result.content || '').toLowerCase();
    const costCode = (result.cost_code || '').toLowerCase();
    const description = (result.description || '').toLowerCase();
    
    // Boost for exact cost code matches
    if (costCode && query.includes(costCode)) {
      score += 0.15;
    }
    
    // Boost for description matches
    if (description && query.includes(description.substring(0, 20))) {
      score += 0.1;
    }
    
    // Boost for category matches
    const categoryTerms = {
      'material': ['material', 'supply', 'equipment'],
      'labor': ['labor', 'work', 'install'],
      'subcontractor': ['sub', 'contractor', 'specialist'],
      'equipment': ['rental', 'equipment', 'tool']
    };
    
    const category = result.category || 'other';
    const categoryWords = categoryTerms[category] || [];
    const matchingTerms = categoryWords.filter(term => query.includes(term));
    score += matchingTerms.length * 0.05;
    
    // Boost for high-value items (usually more important)
    const amount = result.total_amount || 0;
    if (amount > 10000) score += 0.05;
    if (amount > 50000) score += 0.05;
    
    // Boost for items with variance (often more interesting)
    if (result.budgeted_amount > 0) {
      const variance = Math.abs(amount - result.budgeted_amount);
      const variancePercent = (variance / result.budgeted_amount) * 100;
      if (variancePercent > 10) score += 0.03;
    }
    
    return Math.min(score, 1.0);
  }

  /**
   * Calculate cost per unit for additional context
   */
  calculateCostPerUnit(result) {
    const qty = result.qty || result.raw_data?.qty || 0;
    const amount = result.total_amount || 0;
    
    if (qty > 0 && amount > 0) {
      return {
        value: amount / qty,
        units: result.units || result.raw_data?.units || 'ea'
      };
    }
    
    return null;
  }

  /**
   * Get comprehensive performance statistics
   */
  getPerformanceStats(db) {
    const avgProcessingTime = this.totalRequests > 0 
      ? this.totalProcessingTime / this.totalRequests 
      : 0;
    
    const cacheHitRate = (this.cacheHits + this.cacheMisses) > 0 
      ? this.cacheHits / (this.cacheHits + this.cacheMisses) 
      : 0;

    return {
      // Performance metrics
      totalRequests: this.totalRequests,
      averageProcessingTime: Math.round(avgProcessingTime),
      totalProcessingTime: this.totalProcessingTime,
      
      // Cache metrics
      cacheSize: this.embedCache.size,
      maxCacheSize: this.maxCacheSize,
      cacheHits: this.cacheHits,
      cacheMisses: this.cacheMisses,
      cacheHitRate: Math.round(cacheHitRate * 100),
      
      // Configuration
      model: this.model,
      maxBatchSize: this.maxBatchSize,
      similarityThreshold: this.similarityThreshold,
      
      // Rate limiting
      currentRateLimit: this.requestCount,
      maxRequestsPerMinute: this.maxRequestsPerMinute,
      
      // Recommendations
      recommendations: this.getPerformanceRecommendations(cacheHitRate, avgProcessingTime)
    };
  }

  /**
   * Get performance optimization recommendations
   */
  getPerformanceRecommendations(cacheHitRate, avgProcessingTime) {
    const recommendations = [];
    
    if (cacheHitRate < 0.3) {
      recommendations.push('Consider increasing cache size - low cache hit rate');
    }
    
    if (avgProcessingTime > 2000) {
      recommendations.push('Consider reducing batch size - high processing time');
    }
    
    if (this.embedCache.size > this.maxCacheSize * 0.9) {
      recommendations.push('Cache is nearly full - consider increasing max cache size');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('Performance is optimal');
    }
    
    return recommendations;
  }

  /**
   * Clear cache and reset stats
   */
  clearCache() {
    this.embedCache.clear();
    this.cacheHits = 0;
    this.cacheMisses = 0;
    console.log('🧹 Cache cleared and stats reset');
  }

  /**
   * Test embedding service performance
   */
  async testPerformance() {
    console.log('🧪 Testing embedding service performance...');
    
    const testTexts = [
      'Test electrical work costing $5000',
      'Labor for concrete work at $45 per hour',
      'Material supply for framing lumber',
      'Subcontractor HVAC installation'
    ];
    
    try {
      const startTime = Date.now();
      const embeddings = await this.generateEmbeddingsBatch(testTexts);
      const duration = Date.now() - startTime;
      
      const isValid = embeddings.every(emb => 
        Array.isArray(emb) && emb.length > 0 && typeof emb[0] === 'number'
      );
      
      if (isValid) {
        console.log(`✅ Performance test passed: ${testTexts.length} embeddings in ${duration}ms`);
        console.log(`📊 Average: ${Math.round(duration / testTexts.length)}ms per embedding`);
        return { success: true, duration, averageTime: duration / testTexts.length };
      } else {
        throw new Error('Invalid embeddings generated');
      }
      
    } catch (error) {
      console.error('❌ Performance test failed:', error.message);
      return { success: false, error: error.message };
    }
  }
}

// Export singleton instance
const optimizedEmbeddingService = new OptimizedEmbeddingService();
export default optimizedEmbeddingService;