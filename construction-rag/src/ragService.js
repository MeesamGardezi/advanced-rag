/**
 * Graph-Aware RAG Service
 * Uses OpenAI to generate natural language responses from construction data
 * with graph relationship context for enhanced answers
 */

import OpenAI from 'openai';
import dotenv from 'dotenv';
import { getOpenAIConfig } from '../config/openai.js';

dotenv.config();

class RAGService {
  constructor() {
    this.config = getOpenAIConfig();
    this.openai = new OpenAI({
      apiKey: this.config.apiKey,
      organization: this.config.organization
    });
    
    this.embeddingModel = this.config.models.embedding;
    this.completionModel = this.config.models.completion;
    
    console.log('✅ RAG service initialized');
    console.log(`🤖 Models: ${this.embeddingModel} | ${this.completionModel}`);
  }

  // Main RAG query method
  async query(question, db, options = {}) {
    const {
      projectId = null,
      limit = 10,
      threshold = this.config.embedding.defaultSimilarityThreshold,
      includeRelationships = true,
      responseStyle = 'detailed',
      entityType = null,
      category = null
    } = options;

    try {
      console.log(`🔍 Processing RAG query: "${question}"`);
      
      // Step 1: Generate query embedding
      const queryEmbedding = await this.generateQueryEmbedding(question);
      
      // Step 2: Find similar entities
      const similarEntities = await this.findSimilarEntities(
        queryEmbedding, 
        db, 
        { projectId, limit, threshold, entityType, category }
      );
      
      if (similarEntities.length === 0) {
        return this.createEmptyResponse(question);
      }
      
      // Step 3: Enhance context with relationships if requested
      let enhancedContext = similarEntities;
      if (includeRelationships) {
        enhancedContext = await this.enhanceWithRelationships(similarEntities, db);
      }
      
      // Step 4: Build comprehensive context
      const context = await this.buildContext(enhancedContext, question, db, projectId);
      
      // Step 5: Generate natural language response
      const response = await this.generateResponse(question, context, responseStyle);
      
      // Step 6: Create structured result
      return this.createResponse(question, response, context, similarEntities, options);
      
    } catch (error) {
      console.error('❌ Error in RAG query:', error.message);
      return this.createErrorResponse(question, error);
    }
  }

  // Generate embedding for query
  async generateQueryEmbedding(question) {
    try {
      const response = await this.openai.embeddings.create({
        model: this.embeddingModel,
        input: question.trim()
      });
      
      return response.data[0].embedding;
      
    } catch (error) {
      console.error('❌ Error generating query embedding:', error.message);
      throw error;
    }
  }

  // Find similar entities using vector search
  async findSimilarEntities(queryEmbedding, db, options) {
    const { projectId, limit, threshold, entityType, category } = options;
    
    try {
      // Base similarity search
      let results = await db.searchSimilarEntities(queryEmbedding, threshold, limit * 2);
      
      // Apply filters
      if (projectId) {
        results = results.filter(r => r.project_id === projectId);
      }
      
      if (entityType) {
        results = results.filter(r => r.entity_type === entityType);
      }
      
      if (category) {
        results = results.filter(r => r.category === category);
      }
      
      // Limit final results
      results = results.slice(0, limit);
      
      // Enhance with additional scoring
      return results.map(result => ({
        ...result,
        similarity: parseFloat(result.similarity),
        relevanceScore: this.calculateRelevanceScore(result, queryEmbedding),
        contextScore: this.calculateContextScore(result)
      }));
      
    } catch (error) {
      console.error('❌ Error finding similar entities:', error.message);
      throw error;
    }
  }

  // Enhance context with graph relationships
  async enhanceWithRelationships(entities, db) {
    try {
      const enhancedEntities = [];
      
      for (const entity of entities) {
        // Get relationships for this entity
        const relationships = await db.getEntityRelationships(entity.id);
        
        // Add relationship context
        const enhancedEntity = {
          ...entity,
          relationships: relationships,
          relationshipCount: relationships.length,
          relationshipTypes: [...new Set(relationships.map(r => r.type))]
        };
        
        enhancedEntities.push(enhancedEntity);
      }
      
      console.log(`🔗 Enhanced ${enhancedEntities.length} entities with relationship context`);
      return enhancedEntities;
      
    } catch (error) {
      console.error('❌ Error enhancing with relationships:', error.message);
      return entities; // Fall back to original entities
    }
  }

  // Build comprehensive context for the LLM
  async buildContext(entities, question, db, projectId) {
    try {
      // Basic context from entities
      const context = {
        question,
        totalEntities: entities.length,
        entities: entities.map(this.formatEntityForContext.bind(this)),
        
        // Analysis summaries
        categories: this.analyzeCategoriesByCount(entities),
        costCodes: this.analyzeCostCodes(entities),
        financialSummary: this.calculateFinancialSummary(entities),
        
        // Relationship insights
        relationships: this.analyzeRelationships(entities),
        
        // Query insights
        queryType: this.classifyQuery(question),
        keyTerms: this.extractKeyTerms(question)
      };
      
      // Add project context if available
      if (projectId) {
        const projectStats = await db.getProjectStats(projectId);
        context.projectContext = projectStats;
      }
      
      return context;
      
    } catch (error) {
      console.error('❌ Error building context:', error.message);
      throw error;
    }
  }

  // Format entity for context
  formatEntityForContext(entity) {
    return {
      id: entity.id,
      type: entity.entity_type,
      costCode: entity.cost_code,
      description: entity.description,
      category: entity.category,
      amount: entity.total_amount,
      budgetedAmount: entity.budgeted_amount,
      variance: entity.total_amount - (entity.budgeted_amount || 0),
      similarity: Math.round(entity.similarity * 100),
      
      // Relationship context
      relationshipCount: entity.relationshipCount || 0,
      relationshipTypes: entity.relationshipTypes || [],
      
      // Key content excerpts
      contentExcerpt: entity.content ? entity.content.substring(0, 200) + '...' : '',
      
      // Context scores
      relevanceScore: Math.round((entity.relevanceScore || 0) * 100),
      contextScore: Math.round((entity.contextScore || 0) * 100)
    };
  }

  // Generate natural language response using OpenAI
  async generateResponse(question, context, responseStyle) {
    try {
      const prompt = this.buildPrompt(question, context, responseStyle);
      
      const response = await this.openai.chat.completions.create({
        model: this.completionModel,
        messages: [
          {
            role: 'system',
            content: this.config.prompts.system.construction_expert
          },
          {
            role: 'user', 
            content: prompt
          }
        ],
        temperature: this.config.completion.temperature,
        max_tokens: this.config.completion.maxTokens,
        top_p: this.config.completion.topP
      });
      
      return response.choices[0].message.content.trim();
      
    } catch (error) {
      console.error('❌ Error generating response:', error.message);
      throw error;
    }
  }

  // Build optimized prompt for construction queries
  buildPrompt(question, context, responseStyle) {
    const { entities, categories, costCodes, financialSummary, queryType } = context;
    
    let prompt = `Based on the construction project data below, please provide a comprehensive answer to the user's question.

QUESTION: ${question}

QUERY TYPE: ${queryType}

RELEVANT DATA FOUND:
- ${entities.length} relevant construction items found
- Categories: ${Object.keys(categories).join(', ')}
- Cost codes: ${Object.keys(costCodes).slice(0, 10).join(', ')}${Object.keys(costCodes).length > 10 ? '...' : ''}
- Total value: $${(financialSummary.totalAmount || 0).toLocaleString()}

DETAILED DATA:
`;

    // Add entity details based on query type
    if (queryType === 'cost_analysis' || queryType === 'financial') {
      prompt += this.buildFinancialDataSection(entities, financialSummary);
    } else if (queryType === 'comparison') {
      prompt += this.buildComparisonDataSection(entities);
    } else {
      prompt += this.buildGeneralDataSection(entities);
    }

    // Add relationship insights if available
    if (context.relationships && context.relationships.totalRelationships > 0) {
      prompt += `\nRELATIONSHIP INSIGHTS:
- ${context.relationships.totalRelationships} relationships found
- Common connections: ${context.relationships.commonTypes.join(', ')}`;
    }

    // Add instructions based on response style
    prompt += `\n\nINSTRUCTIONS:
- Provide specific dollar amounts, cost codes, and percentages when available
- ${responseStyle === 'brief' ? 'Keep the response concise and focused' : 'Provide detailed analysis with explanations'}
- Mention relevant cost codes and categories
- If discussing costs, include budget vs actual analysis when applicable
- Structure the response with clear sections if covering multiple aspects
- Use actual data from the search results, don't make up numbers

RESPONSE:`;

    return prompt;
  }

  // Build financial data section for prompt
  buildFinancialDataSection(entities, financialSummary) {
    let section = `\nFINANCIAL DATA BREAKDOWN:\n`;
    
    section += `Total Amount: $${(financialSummary.totalAmount || 0).toLocaleString()}\n`;
    if (financialSummary.totalBudgeted > 0) {
      section += `Total Budgeted: $${financialSummary.totalBudgeted.toLocaleString()}\n`;
      section += `Variance: $${(financialSummary.totalVariance || 0).toLocaleString()}\n`;
    }
    
    // Top cost items
    const topItems = entities
      .filter(e => e.amount > 0)
      .sort((a, b) => (b.amount || 0) - (a.amount || 0))
      .slice(0, 5);
    
    section += `\nTop Cost Items:\n`;
    topItems.forEach((item, index) => {
      const variance = item.variance || 0;
      const varianceText = variance !== 0 ? ` (${variance > 0 ? '+' : ''}$${variance.toLocaleString()} variance)` : '';
      section += `${index + 1}. ${item.costCode}: $${(item.amount || 0).toLocaleString()}${varianceText}\n`;
    });
    
    return section;
  }

  // Build comparison data section
  buildComparisonDataSection(entities) {
    let section = `\nCOMPARISON DATA:\n`;
    
    // Group by category for comparison
    const byCategory = {};
    entities.forEach(entity => {
      const category = entity.category || 'other';
      if (!byCategory[category]) byCategory[category] = [];
      byCategory[category].push(entity);
    });
    
    Object.entries(byCategory).forEach(([category, items]) => {
      const total = items.reduce((sum, item) => sum + (item.amount || 0), 0);
      section += `\n${category.toUpperCase()} (${items.length} items - $${total.toLocaleString()}):\n`;
      
      items.slice(0, 3).forEach(item => {
        section += `  - ${item.costCode}: $${(item.amount || 0).toLocaleString()}\n`;
      });
    });
    
    return section;
  }

  // Build general data section
  buildGeneralDataSection(entities) {
    let section = `\nRELEVANT CONSTRUCTION DATA:\n`;
    
    entities.slice(0, 8).forEach((entity, index) => {
      section += `\n${index + 1}. ${entity.costCode} - ${entity.category}\n`;
      section += `   Amount: $${(entity.amount || 0).toLocaleString()}`;
      if (entity.budgetedAmount) {
        section += ` (Budgeted: $${entity.budgetedAmount.toLocaleString()})`;
      }
      section += `\n   Similarity: ${entity.similarity}%\n`;
      if (entity.contentExcerpt) {
        section += `   Details: ${entity.contentExcerpt}\n`;
      }
    });
    
    return section;
  }

  // Calculate relevance score
  calculateRelevanceScore(result, queryEmbedding) {
    let score = parseFloat(result.similarity) || 0;
    
    // Boost for higher cost items (more significant)
    const amount = result.total_amount || 0;
    if (amount > 10000) score += 0.05;
    if (amount > 50000) score += 0.05;
    
    // Boost for items with relationships
    if (result.relationshipCount > 0) {
      score += Math.min(0.1, result.relationshipCount * 0.02);
    }
    
    return Math.min(score, 1.0);
  }

  // Calculate context score
  calculateContextScore(result) {
    let score = 0.5; // Base score
    
    // Boost for complete data
    if (result.cost_code) score += 0.1;
    if (result.category && result.category !== 'other') score += 0.1;
    if (result.description) score += 0.1;
    if (result.total_amount > 0) score += 0.1;
    
    // Boost for variance data (estimate items)
    if (result.budgeted_amount > 0) score += 0.1;
    
    return Math.min(score, 1.0);
  }

  // Classify query type
  classifyQuery(question) {
    const q = question.toLowerCase();
    
    if (q.includes('cost') || q.includes('budget') || q.includes('spend') || q.includes('$')) {
      return 'cost_analysis';
    }
    if (q.includes('compare') || q.includes('vs') || q.includes('difference')) {
      return 'comparison';
    }
    if (q.includes('schedule') || q.includes('time') || q.includes('date')) {
      return 'schedule';
    }
    if (q.includes('material') || q.includes('labor') || q.includes('subcontractor')) {
      return 'category';
    }
    
    return 'general';
  }

  // Extract key terms from question
  extractKeyTerms(question) {
    const terms = question.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(' ')
      .filter(term => term.length > 3 && !['what', 'when', 'where', 'which', 'this', 'that', 'with', 'from'].includes(term));
    
    return [...new Set(terms)].slice(0, 10);
  }

  // Analyze categories
  analyzeCategoriesByCount(entities) {
    const categories = {};
    entities.forEach(entity => {
      const category = entity.category || 'other';
      categories[category] = (categories[category] || 0) + 1;
    });
    return categories;
  }

  // Analyze cost codes
  analyzeCostCodes(entities) {
    const costCodes = {};
    entities.forEach(entity => {
      if (entity.cost_code) {
        costCodes[entity.cost_code] = (costCodes[entity.cost_code] || 0) + 1;
      }
    });
    return costCodes;
  }

  // Calculate financial summary
  calculateFinancialSummary(entities) {
    const summary = {
      totalAmount: 0,
      totalBudgeted: 0,
      totalVariance: 0,
      itemCount: entities.length
    };
    
    entities.forEach(entity => {
      summary.totalAmount += entity.total_amount || 0;
      summary.totalBudgeted += entity.budgeted_amount || 0;
    });
    
    summary.totalVariance = summary.totalAmount - summary.totalBudgeted;
    
    return summary;
  }

  // Analyze relationships
  analyzeRelationships(entities) {
    const analysis = {
      totalRelationships: 0,
      commonTypes: [],
      strongConnections: 0
    };
    
    const typeCount = {};
    
    entities.forEach(entity => {
      if (entity.relationships) {
        analysis.totalRelationships += entity.relationships.length;
        
        entity.relationships.forEach(rel => {
          typeCount[rel.type] = (typeCount[rel.type] || 0) + 1;
          if (rel.strength > 0.8) {
            analysis.strongConnections++;
          }
        });
      }
    });
    
    analysis.commonTypes = Object.entries(typeCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([type]) => type);
    
    return analysis;
  }

  // Create successful response
  createResponse(question, answer, context, similarEntities, options) {
    return {
      success: true,
      question,
      answer,
      metadata: {
        totalResults: similarEntities.length,
        queryType: context.queryType,
        avgSimilarity: similarEntities.reduce((sum, e) => sum + e.similarity, 0) / similarEntities.length,
        categories: Object.keys(context.categories),
        costCodes: Object.keys(context.costCodes).slice(0, 10),
        financialSummary: context.financialSummary,
        relationshipInsights: context.relationships,
        searchOptions: options
      },
      sources: similarEntities.map(entity => ({
        id: entity.id,
        costCode: entity.cost_code,
        description: entity.description,
        category: entity.category,
        amount: entity.total_amount,
        similarity: Math.round(entity.similarity * 100),
        relevance: Math.round((entity.relevanceScore || 0) * 100)
      })),
      timestamp: new Date().toISOString()
    };
  }

  // Create empty response
  createEmptyResponse(question) {
    return {
      success: true,
      question,
      answer: "I couldn't find any relevant construction data for your question. Try rephrasing your query or asking about specific cost codes, categories, or project elements.",
      metadata: {
        totalResults: 0,
        queryType: this.classifyQuery(question),
        categories: [],
        costCodes: []
      },
      sources: [],
      timestamp: new Date().toISOString()
    };
  }

  // Create error response
  createErrorResponse(question, error) {
    return {
      success: false,
      question,
      answer: "I encountered an error while processing your question. Please try again or rephrase your query.",
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }

  // Test RAG service
  async testRAGService() {
    try {
      console.log('🧪 Testing RAG service...');
      
      const testQuestion = 'What electrical work was done on the project?';
      const testEmbedding = await this.generateQueryEmbedding(testQuestion);
      
      const isValid = Array.isArray(testEmbedding) && 
                     testEmbedding.length > 0 && 
                     typeof testEmbedding[0] === 'number';
      
      if (isValid) {
        console.log(`✅ RAG service test passed`);
        return true;
      } else {
        throw new Error('Invalid embedding generated');
      }
      
    } catch (error) {
      console.error('❌ RAG service test failed:', error.message);
      return false;
    }
  }
}

// Export singleton instance
const ragService = new RAGService();
export default ragService;