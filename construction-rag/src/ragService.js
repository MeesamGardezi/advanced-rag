/**
 * Estimate-Focused RAG Service
 * Specialized RAG system for construction estimate queries with expert domain knowledge
 */

import OpenAI from 'openai';
import dotenv from 'dotenv';
import { getOpenAIConfig } from '../config/openai.js';

dotenv.config();

class EstimateRAGService {
  constructor() {
    this.config = getOpenAIConfig();
    this.openai = new OpenAI({
      apiKey: this.config.apiKey,
      organization: this.config.organization
    });
    
    this.embeddingModel = this.config.models.embedding;
    this.completionModel = this.config.models.completion;
    
    // Estimate-specific configuration
    this.defaultSimilarityThreshold = 0.75; // Higher threshold for better precision
    this.maxContextItems = 12; // More context for detailed answers
    this.maxContextTokens = 4500; // Leave room for response
    
    console.log('✅ Estimate RAG service initialized');
    console.log(`🎯 Models: ${this.embeddingModel} | ${this.completionModel}`);
  }

  /**
   * Main estimate query method with enhanced processing
   */
  async queryEstimates(question, db, options = {}) {
    const {
      projectId = null,
      limit = this.maxContextItems,
      threshold = this.defaultSimilarityThreshold,
      category = null,
      costCodePattern = null,
      responseStyle = 'detailed',
      includeAnalysis = true
    } = options;

    try {
      console.log(`🔍 Processing estimate query: "${question}"`);
      const startTime = Date.now();
      
      // Step 1: Classify the query type for specialized handling
      const queryType = this.classifyEstimateQuery(question);
      console.log(`📋 Query type: ${queryType}`);
      
      // Step 2: Generate query embedding
      const queryEmbedding = await this.generateQueryEmbedding(question);
      
      // Step 3: Search for similar estimate entities
      const searchOptions = {
        threshold,
        limit: Math.min(limit * 2, 25), // Get extra for filtering
        projectId,
        category,
        costCodePattern
      };
      
      const similarEntities = await this.findSimilarEstimateEntities(
        queryEmbedding, 
        db, 
        searchOptions
      );
      
      if (similarEntities.length === 0) {
        return this.createEmptyEstimateResponse(question, queryType);
      }
      
      // Step 4: Build comprehensive estimate context
      const context = await this.buildEstimateContext(
        similarEntities, 
        question, 
        queryType, 
        db, 
        projectId
      );
      
      // Step 5: Generate expert estimate response
      const response = await this.generateEstimateResponse(
        question, 
        context, 
        queryType, 
        responseStyle
      );
      
      // Step 6: Create structured response with insights
      const result = this.createEstimateResponse(
        question, 
        response, 
        context, 
        similarEntities, 
        queryType, 
        options
      );
      
      const totalDuration = Date.now() - startTime;
      result.metadata.processingTime = totalDuration;
      
      console.log(`✅ Query processed in ${totalDuration}ms`);
      return result;
      
    } catch (error) {
      console.error('❌ Error in estimate query:', error.message);
      return this.createErrorResponse(question, error);
    }
  }

  /**
   * Specialized query classification for estimate questions
   */
  classifyEstimateQuery(question) {
    const q = question.toLowerCase();
    const tokens = q.split(/\s+/);
    
    // Cost analysis queries
    if (tokens.some(t => ['cost', 'price', 'total', 'amount', 'spend'].includes(t))) {
      if (tokens.some(t => ['budget', 'variance', 'over', 'under'].includes(t))) {
        return 'budget_variance';
      }
      return 'cost_analysis';
    }
    
    // Budget-specific queries
    if (tokens.some(t => ['budget', 'budgeted', 'allocated', 'planned'].includes(t))) {
      return 'budget_analysis';
    }
    
    // Quantity and rate queries
    if (tokens.some(t => ['quantity', 'qty', 'units', 'how', 'much', 'many'].includes(t))) {
      return 'quantity_analysis';
    }
    
    if (tokens.some(t => ['rate', 'unit', 'per', 'each', 'hourly'].includes(t))) {
      return 'rate_analysis';
    }
    
    // Category queries
    if (tokens.some(t => ['material', 'materials', 'supply', 'supplies'].includes(t))) {
      return 'material_analysis';
    }
    
    if (tokens.some(t => ['labor', 'work', 'worker', 'crew'].includes(t))) {
      return 'labor_analysis';
    }
    
    if (tokens.some(t => ['subcontractor', 'sub', 'contractor', 'specialist'].includes(t))) {
      return 'subcontractor_analysis';
    }
    
    // Scope and area queries
    if (tokens.some(t => ['area', 'scope', 'task', 'location', 'where'].includes(t))) {
      return 'scope_analysis';
    }
    
    // Comparison queries
    if (tokens.some(t => ['compare', 'comparison', 'vs', 'versus', 'different', 'difference'].includes(t))) {
      return 'comparison_analysis';
    }
    
    // Summary queries
    if (tokens.some(t => ['summary', 'total', 'overview', 'breakdown'].includes(t))) {
      return 'summary_analysis';
    }
    
    return 'general_estimate';
  }

  /**
   * Generate embedding for query with error handling
   */
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

  /**
   * Find similar estimate entities with enhanced filtering
   */
  async findSimilarEstimateEntities(queryEmbedding, db, options) {
    const { threshold, limit, projectId, category, costCodePattern } = options;
    
    try {
      // Perform vector similarity search
      let results = await db.searchSimilarEntities(queryEmbedding, threshold, limit);
      
      // Apply filters
      if (projectId) {
        results = results.filter(r => r.project_id === projectId);
      }
      
      if (category) {
        results = results.filter(r => r.category === category);
      }
      
      if (costCodePattern) {
        const pattern = new RegExp(costCodePattern, 'i');
        results = results.filter(r => pattern.test(r.cost_code || ''));
      }
      
      // Enhance results with additional scoring and metadata
      return results.map(result => ({
        ...result,
        similarity: parseFloat(result.similarity),
        similarityPercent: Math.round(parseFloat(result.similarity) * 100),
        estimateRelevance: this.calculateEstimateRelevance(result),
        budgetHealth: this.calculateBudgetHealth(result),
        costPerUnit: this.calculateCostPerUnit(result)
      }));
      
    } catch (error) {
      console.error('❌ Error finding similar entities:', error.message);
      throw error;
    }
  }

  /**
   * Calculate estimate-specific relevance score
   */
  calculateEstimateRelevance(result) {
    let score = parseFloat(result.similarity) || 0;
    
    // Boost for items with good financial data
    if (result.total_amount > 0) score += 0.05;
    if (result.budgeted_amount > 0) score += 0.05;
    
    // Boost for high-value items (often more important)
    const amount = result.total_amount || 0;
    if (amount > 5000) score += 0.05;
    if (amount > 25000) score += 0.1;
    
    // Boost for items with significant variance (more interesting)
    if (result.budgeted_amount > 0) {
      const variance = Math.abs(amount - result.budgeted_amount);
      const variancePercent = (variance / result.budgeted_amount) * 100;
      if (variancePercent > 15) score += 0.08;
      if (variancePercent > 30) score += 0.12;
    }
    
    // Boost for complete data
    if (result.cost_code && result.cost_code.length > 2) score += 0.03;
    if (result.description && result.description.length > 10) score += 0.03;
    if (result.task_scope) score += 0.02;
    
    return Math.min(score, 1.0);
  }

  /**
   * Calculate budget health indicator
   */
  calculateBudgetHealth(result) {
    if (!result.budgeted_amount || result.budgeted_amount <= 0) {
      return { status: 'no_budget', variance: 0, variancePercent: 0 };
    }
    
    const variance = (result.total_amount || 0) - result.budgeted_amount;
    const variancePercent = (variance / result.budgeted_amount) * 100;
    
    let status = 'on_budget';
    if (Math.abs(variancePercent) <= 5) status = 'on_budget';
    else if (variance > 0 && variancePercent <= 20) status = 'over_budget';
    else if (variance > 0) status = 'significantly_over';
    else if (variance < 0 && variancePercent >= -20) status = 'under_budget';
    else status = 'significantly_under';
    
    return {
      status,
      variance,
      variancePercent,
      budgeted: result.budgeted_amount,
      actual: result.total_amount || 0
    };
  }

  /**
   * Calculate cost per unit if possible
   */
  calculateCostPerUnit(result) {
    const qty = result.qty || (result.raw_data ? result.raw_data.qty : 0);
    const amount = result.total_amount || 0;
    const units = result.units || (result.raw_data ? result.raw_data.units : 'ea');
    
    if (qty > 0 && amount > 0) {
      return {
        value: amount / qty,
        units: units,
        formattedValue: `$${(amount / qty).toFixed(2)}/${units}`
      };
    }
    
    return null;
  }

  /**
   * Build comprehensive context for estimate queries
   */
  async buildEstimateContext(entities, question, queryType, db, projectId) {
    try {
      const context = {
        question,
        queryType,
        totalEntities: entities.length,
        
        // Core entity data
        entities: entities.slice(0, this.maxContextItems).map(this.formatEntityForContext.bind(this)),
        
        // Financial analysis
        financialSummary: this.calculateFinancialSummary(entities),
        
        // Category breakdown
        categoryAnalysis: this.analyzeCategoriesByFinancials(entities),
        
        // Cost code analysis
        costCodeAnalysis: this.analyzeCostCodes(entities),
        
        // Budget performance
        budgetPerformance: this.analyzeBudgetPerformance(entities),
        
        // High-value and problem items
        highValueItems: this.identifyHighValueItems(entities),
        varianceItems: this.identifyVarianceItems(entities),
        
        // Query insights
        queryInsights: this.extractQueryInsights(question, queryType)
      };
      
      // Add project context if available
      if (projectId) {
        try {
          const projectStats = await db.getProjectStats(projectId);
          context.projectContext = projectStats;
        } catch (error) {
          console.warn('Could not fetch project context:', error.message);
        }
      }
      
      return context;
      
    } catch (error) {
      console.error('❌ Error building context:', error.message);
      throw error;
    }
  }

  /**
   * Format entity for LLM context
   */
  formatEntityForContext(entity) {
    const budgetHealth = this.calculateBudgetHealth(entity);
    
    return {
      costCode: entity.cost_code,
      description: entity.description,
      category: entity.category,
      area: entity.area,
      taskScope: entity.task_scope,
      
      // Financial data
      amount: entity.total_amount,
      budgeted: entity.budgeted_amount,
      variance: budgetHealth.variance,
      variancePercent: budgetHealth.variancePercent,
      budgetStatus: budgetHealth.status,
      
      // Quantities
      qty: entity.qty,
      units: entity.units,
      rate: entity.rate,
      costPerUnit: this.calculateCostPerUnit(entity),
      
      // Relevance
      similarity: entity.similarityPercent,
      relevance: Math.round(entity.estimateRelevance * 100)
    };
  }

  /**
   * Calculate comprehensive financial summary
   */
  calculateFinancialSummary(entities) {
    const summary = {
      totalEstimated: 0,
      totalBudgeted: 0,
      totalVariance: 0,
      itemCount: entities.length,
      averageAmount: 0,
      medianAmount: 0,
      largestItem: 0,
      smallestItem: Infinity
    };
    
    const amounts = [];
    
    entities.forEach(entity => {
      const amount = entity.total_amount || 0;
      const budgeted = entity.budgeted_amount || 0;
      
      summary.totalEstimated += amount;
      summary.totalBudgeted += budgeted;
      
      if (amount > 0) {
        amounts.push(amount);
        summary.largestItem = Math.max(summary.largestItem, amount);
        summary.smallestItem = Math.min(summary.smallestItem, amount);
      }
    });
    
    summary.totalVariance = summary.totalEstimated - summary.totalBudgeted;
    summary.totalVariancePercent = summary.totalBudgeted > 0 
      ? (summary.totalVariance / summary.totalBudgeted) * 100 
      : 0;
    
    if (amounts.length > 0) {
      summary.averageAmount = amounts.reduce((sum, amt) => sum + amt, 0) / amounts.length;
      amounts.sort((a, b) => a - b);
      summary.medianAmount = amounts[Math.floor(amounts.length / 2)];
    } else {
      summary.smallestItem = 0;
    }
    
    return summary;
  }

  /**
   * Analyze categories with financial details
   */
  analyzeCategoriesByFinancials(entities) {
    const analysis = {};
    
    entities.forEach(entity => {
      const category = entity.category || 'other';
      
      if (!analysis[category]) {
        analysis[category] = {
          count: 0,
          totalAmount: 0,
          totalBudgeted: 0,
          averageAmount: 0,
          items: []
        };
      }
      
      analysis[category].count++;
      analysis[category].totalAmount += entity.total_amount || 0;
      analysis[category].totalBudgeted += entity.budgeted_amount || 0;
      analysis[category].items.push({
        costCode: entity.cost_code,
        amount: entity.total_amount || 0
      });
    });
    
    // Calculate averages and sort items
    Object.keys(analysis).forEach(category => {
      const cat = analysis[category];
      cat.averageAmount = cat.count > 0 ? cat.totalAmount / cat.count : 0;
      cat.variance = cat.totalAmount - cat.totalBudgeted;
      cat.variancePercent = cat.totalBudgeted > 0 
        ? (cat.variance / cat.totalBudgeted) * 100 
        : 0;
      cat.items.sort((a, b) => (b.amount || 0) - (a.amount || 0));
      cat.items = cat.items.slice(0, 3); // Keep top 3
    });
    
    return analysis;
  }

  /**
   * Analyze cost codes with aggregation
   */
  analyzeCostCodes(entities) {
    const analysis = {};
    
    entities.forEach(entity => {
      const costCode = entity.cost_code || 'NO_CODE';
      
      if (!analysis[costCode]) {
        analysis[costCode] = {
          count: 0,
          totalAmount: 0,
          totalBudgeted: 0,
          category: entity.category,
          descriptions: new Set()
        };
      }
      
      analysis[costCode].count++;
      analysis[costCode].totalAmount += entity.total_amount || 0;
      analysis[costCode].totalBudgeted += entity.budgeted_amount || 0;
      
      if (entity.description) {
        analysis[costCode].descriptions.add(entity.description);
      }
    });
    
    // Convert Set to Array and calculate variance
    Object.keys(analysis).forEach(costCode => {
      const code = analysis[costCode];
      code.descriptions = Array.from(code.descriptions).slice(0, 2); // Max 2 descriptions
      code.variance = code.totalAmount - code.totalBudgeted;
      code.variancePercent = code.totalBudgeted > 0 
        ? (code.variance / code.totalBudgeted) * 100 
        : 0;
    });
    
    return analysis;
  }

  /**
   * Analyze overall budget performance
   */
  analyzeBudgetPerformance(entities) {
    const performance = {
      totalItems: entities.length,
      itemsWithBudget: 0,
      onBudget: 0,
      overBudget: 0,
      underBudget: 0,
      significantVariance: 0,
      averageVariancePercent: 0
    };
    
    let totalVariancePercent = 0;
    let itemsWithVariance = 0;
    
    entities.forEach(entity => {
      if (entity.budgeted_amount > 0) {
        performance.itemsWithBudget++;
        
        const budgetHealth = this.calculateBudgetHealth(entity);
        
        switch (budgetHealth.status) {
          case 'on_budget':
            performance.onBudget++;
            break;
          case 'over_budget':
          case 'significantly_over':
            performance.overBudget++;
            if (budgetHealth.status === 'significantly_over') {
              performance.significantVariance++;
            }
            break;
          case 'under_budget':
          case 'significantly_under':
            performance.underBudget++;
            if (budgetHealth.status === 'significantly_under') {
              performance.significantVariance++;
            }
            break;
        }
        
        totalVariancePercent += Math.abs(budgetHealth.variancePercent);
        itemsWithVariance++;
      }
    });
    
    if (itemsWithVariance > 0) {
      performance.averageVariancePercent = totalVariancePercent / itemsWithVariance;
    }
    
    return performance;
  }

  /**
   * Identify high-value items
   */
  identifyHighValueItems(entities) {
    return entities
      .filter(entity => (entity.total_amount || 0) > 5000)
      .sort((a, b) => (b.total_amount || 0) - (a.total_amount || 0))
      .slice(0, 5)
      .map(entity => ({
        costCode: entity.cost_code,
        description: entity.description,
        amount: entity.total_amount,
        category: entity.category,
        budgetHealth: this.calculateBudgetHealth(entity)
      }));
  }

  /**
   * Identify items with significant variance
   */
  identifyVarianceItems(entities) {
    return entities
      .filter(entity => {
        if (!entity.budgeted_amount || entity.budgeted_amount <= 0) return false;
        
        const variance = Math.abs((entity.total_amount || 0) - entity.budgeted_amount);
        const variancePercent = (variance / entity.budgeted_amount) * 100;
        
        return variancePercent > 20 || variance > 2000;
      })
      .sort((a, b) => {
        const aVariance = Math.abs((a.total_amount || 0) - (a.budgeted_amount || 0));
        const bVariance = Math.abs((b.total_amount || 0) - (b.budgeted_amount || 0));
        return bVariance - aVariance;
      })
      .slice(0, 5)
      .map(entity => {
        const budgetHealth = this.calculateBudgetHealth(entity);
        return {
          costCode: entity.cost_code,
          description: entity.description,
          estimated: entity.total_amount,
          budgeted: entity.budgeted_amount,
          variance: budgetHealth.variance,
          variancePercent: budgetHealth.variancePercent,
          category: entity.category
        };
      });
  }

  /**
   * Extract insights from the query
   */
  extractQueryInsights(question, queryType) {
    const insights = {
      queryType,
      keyTerms: this.extractKeyTerms(question),
      focusAreas: [],
      suggestedAnalysis: []
    };
    
    const q = question.toLowerCase();
    
    // Identify focus areas
    if (q.includes('material')) insights.focusAreas.push('materials');
    if (q.includes('labor')) insights.focusAreas.push('labor');
    if (q.includes('budget')) insights.focusAreas.push('budget_variance');
    if (q.includes('cost')) insights.focusAreas.push('cost_analysis');
    
    // Suggest related analysis based on query type
    switch (queryType) {
      case 'cost_analysis':
        insights.suggestedAnalysis = ['budget_comparison', 'category_breakdown', 'high_value_items'];
        break;
      case 'budget_variance':
        insights.suggestedAnalysis = ['variance_items', 'budget_performance', 'cost_overruns'];
        break;
      case 'material_analysis':
        insights.suggestedAnalysis = ['material_costs', 'supplier_analysis', 'quantity_efficiency'];
        break;
      default:
        insights.suggestedAnalysis = ['cost_summary', 'budget_health', 'category_breakdown'];
    }
    
    return insights;
  }

  /**
   * Extract key terms from question
   */
  extractKeyTerms(question) {
    const terms = question.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(term => 
        term.length > 3 && 
        !['what', 'when', 'where', 'which', 'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been'].includes(term)
      );
    
    return [...new Set(terms)].slice(0, 8);
  }

  /**
   * Generate expert estimate response using OpenAI
   */
  async generateEstimateResponse(question, context, queryType, responseStyle) {
    try {
      const prompt = this.buildEstimatePrompt(question, context, queryType, responseStyle);
      
      const response = await this.openai.chat.completions.create({
        model: this.completionModel,
        messages: [
          {
            role: 'system',
            content: this.getSystemPrompt(queryType)
          },
          {
            role: 'user', 
            content: prompt
          }
        ],
        temperature: 0.1, // Low temperature for consistent, factual responses
        max_tokens: 900,
        top_p: 0.9
      });
      
      return response.choices[0].message.content.trim();
      
    } catch (error) {
      console.error('❌ Error generating response:', error.message);
      throw error;
    }
  }

  /**
   * Get specialized system prompt based on query type
   */
  getSystemPrompt(queryType) {
    const basePrompt = "You are an expert construction estimator and cost analyst with deep knowledge of project budgeting, cost codes, and construction economics.";
    
    const specializations = {
      'cost_analysis': " Focus on detailed cost breakdowns, unit pricing, and financial implications.",
      'budget_variance': " Emphasize budget performance, variance analysis, and cost control recommendations.",
      'material_analysis': " Concentrate on material costs, quantities, procurement, and supply chain insights.",
      'labor_analysis': " Focus on labor costs, productivity, crew efficiency, and workforce planning.",
      'rate_analysis': " Emphasize unit rates, pricing benchmarks, and cost-per-unit analysis."
    };
    
    return basePrompt + (specializations[queryType] || " Provide comprehensive construction cost insights.");
  }

  /**
   * Build specialized prompt for estimate queries
   */
  buildEstimatePrompt(question, context, queryType, responseStyle) {
    const { entities, financialSummary, categoryAnalysis, budgetPerformance, highValueItems, varianceItems } = context;
    
    let prompt = `Based on the construction estimate data below, provide a comprehensive answer to this question about project costs:

QUESTION: ${question}
QUERY TYPE: ${queryType}

FINANCIAL SUMMARY:
- Total Estimated Cost: $${financialSummary.totalEstimated.toLocaleString()}
- Total Budgeted Cost: $${financialSummary.totalBudgeted.toLocaleString()}
- Overall Variance: ${financialSummary.totalVariance >= 0 ? '+' : ''}$${financialSummary.totalVariance.toLocaleString()} (${financialSummary.totalVariancePercent.toFixed(1)}%)
- Items Analyzed: ${entities.length}

BUDGET PERFORMANCE:
- Items with Budget Data: ${budgetPerformance.itemsWithBudget}/${budgetPerformance.totalItems}
- On Budget: ${budgetPerformance.onBudget} items
- Over Budget: ${budgetPerformance.overBudget} items  
- Under Budget: ${budgetPerformance.underBudget} items
- Significant Variances: ${budgetPerformance.significantVariance} items`;

    // Add category breakdown
    if (Object.keys(categoryAnalysis).length > 0) {
      prompt += `\n\nCOST BY CATEGORY:`;
      Object.entries(categoryAnalysis)
        .sort(([,a], [,b]) => (b.totalAmount || 0) - (a.totalAmount || 0))
        .slice(0, 5)
        .forEach(([category, data]) => {
          prompt += `\n- ${category.toUpperCase()}: $${data.totalAmount.toLocaleString()} (${data.count} items, avg: $${Math.round(data.averageAmount).toLocaleString()})`;
          if (data.variance !== 0) {
            prompt += ` | Variance: ${data.variance >= 0 ? '+' : ''}$${data.variance.toLocaleString()}`;
          }
        });
    }

    // Add high-value items
    if (highValueItems.length > 0) {
      prompt += `\n\nHIGHEST VALUE ITEMS:`;
      highValueItems.slice(0, 3).forEach((item, i) => {
        prompt += `\n${i+1}. ${item.costCode}: ${item.description} - $${item.amount.toLocaleString()}`;
        if (item.budgetHealth.budgeted > 0) {
          prompt += ` (Budget: $${item.budgetHealth.budgeted.toLocaleString()}, ${item.budgetHealth.variancePercent.toFixed(1)}% variance)`;
        }
      });
    }

    // Add variance items if relevant
    if (varianceItems.length > 0 && (queryType.includes('budget') || queryType.includes('variance'))) {
      prompt += `\n\nLARGEST VARIANCES:`;
      varianceItems.slice(0, 3).forEach((item, i) => {
        prompt += `\n${i+1}. ${item.costCode}: ${item.variancePercent >= 0 ? '+' : ''}${item.variancePercent.toFixed(1)}% (${item.variancePercent >= 0 ? '+' : ''}$${item.variance.toLocaleString()})`;
      });
    }

    // Add detailed item data
    prompt += `\n\nDETAILED ESTIMATE ITEMS:`;
    entities.slice(0, 6).forEach((item, i) => {
      prompt += `\n${i+1}. ${item.costCode} - ${item.category.toUpperCase()}`;
      prompt += `\n   Description: ${item.description}`;
      prompt += `\n   Cost: $${item.amount.toLocaleString()}${item.budgeted ? ` (Budgeted: $${item.budgeted.toLocaleString()})` : ''}`;
      if (item.qty && item.units) {
        prompt += `\n   Quantity: ${item.qty} ${item.units}${item.rate ? ` @ $${item.rate.toFixed(2)}/${item.units}` : ''}`;
      }
      if (item.variance !== 0 && item.budgeted) {
        prompt += `\n   Budget Status: ${item.budgetStatus} (${item.variancePercent.toFixed(1)}% variance)`;
      }
      prompt += `\n`;
    });

    // Add specific instructions based on response style and query type
    prompt += `\n\nANALYSIS REQUIREMENTS:`;
    if (responseStyle === 'detailed') {
      prompt += `\n- Provide comprehensive cost analysis with specific dollar amounts and percentages`;
      prompt += `\n- Explain budget variances and their potential causes`;
      prompt += `\n- Include actionable insights for project management`;
      prompt += `\n- Reference specific cost codes and categories`;
    } else {
      prompt += `\n- Provide concise but informative cost summary`;
      prompt += `\n- Focus on key financial insights and critical variances`;
    }

    // Add query-specific instructions
    switch (queryType) {
      case 'cost_analysis':
        prompt += `\n- Break down costs by category and identify major cost drivers`;
        break;
      case 'budget_variance':
        prompt += `\n- Focus on budget performance and explain significant variances`;
        break;
      case 'material_analysis':
        prompt += `\n- Concentrate on material costs, quantities, and procurement insights`;
        break;
      case 'labor_analysis':
        prompt += `\n- Focus on labor costs, productivity, and crew efficiency`;
        break;
    }

    prompt += `\n\nRESPONSE FORMAT:`;
    prompt += `\n- Use specific dollar amounts and percentages from the data`;
    prompt += `\n- Structure with clear headers and bullet points`;
    prompt += `\n- Provide actionable recommendations where appropriate`;
    prompt += `\n- Maintain professional construction industry terminology`;

    return prompt;
  }

  /**
   * Create comprehensive response object
   */
  createEstimateResponse(question, answer, context, similarEntities, queryType, options) {
    return {
      success: true,
      question,
      answer,
      queryType,
      
      metadata: {
        totalResults: similarEntities.length,
        avgSimilarity: similarEntities.length > 0 
          ? Math.round(similarEntities.reduce((sum, e) => sum + e.similarity, 0) / similarEntities.length * 100)
          : 0,
        financialSummary: context.financialSummary,
        budgetPerformance: context.budgetPerformance,
        queryInsights: context.queryInsights,
        searchOptions: options,
        processingTime: null // Will be filled by calling function
      },
      
      sources: similarEntities.slice(0, 6).map(entity => ({
        costCode: entity.cost_code,
        description: entity.description,
        category: entity.category,
        amount: entity.total_amount,
        budgeted: entity.budgeted_amount,
        variance: entity.budgetHealth?.variance || 0,
        similarity: entity.similarityPercent,
        relevance: Math.round(entity.estimateRelevance * 100)
      })),
      
      insights: {
        highValueItems: context.highValueItems,
        varianceItems: context.varianceItems,
        categoryBreakdown: context.categoryAnalysis,
        recommendations: this.generateRecommendations(context, queryType)
      },
      
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate actionable recommendations based on context
   */
  generateRecommendations(context, queryType) {
    const recommendations = [];
    const { budgetPerformance, financialSummary, varianceItems } = context;
    
    // Budget performance recommendations
    if (budgetPerformance.significantVariance > 0) {
      recommendations.push({
        type: 'budget_control',
        priority: 'high',
        message: `${budgetPerformance.significantVariance} items have significant budget variances. Review cost control processes.`
      });
    }
    
    // Overall variance recommendations
    if (Math.abs(financialSummary.totalVariancePercent) > 10) {
      recommendations.push({
        type: 'budget_review',
        priority: 'medium',
        message: `Overall project variance is ${financialSummary.totalVariancePercent.toFixed(1)}%. Consider budget revision.`
      });
    }
    
    // High variance item recommendations
    if (varianceItems.length > 0) {
      const worstVariance = varianceItems[0];
      recommendations.push({
        type: 'cost_investigation',
        priority: 'high',
        message: `Investigate ${worstVariance.costCode} with ${worstVariance.variancePercent.toFixed(1)}% variance.`
      });
    }
    
    // Query-specific recommendations
    switch (queryType) {
      case 'material_analysis':
        if (context.categoryAnalysis.material) {
          const materialData = context.categoryAnalysis.material;
          if (Math.abs(materialData.variancePercent) > 15) {
            recommendations.push({
              type: 'procurement',
              priority: 'medium',
              message: 'Material costs show significant variance. Review supplier contracts and pricing.'
            });
          }
        }
        break;
        
      case 'labor_analysis':
        if (context.categoryAnalysis.labor) {
          const laborData = context.categoryAnalysis.labor;
          if (laborData.averageAmount > 0) {
            recommendations.push({
              type: 'productivity',
              priority: 'medium',
              message: 'Monitor labor productivity and consider crew efficiency improvements.'
            });
          }
        }
        break;
    }
    
    return recommendations;
  }

  /**
   * Create empty response for no results
   */
  createEmptyEstimateResponse(question, queryType) {
    return {
      success: true,
      question,
      answer: "I couldn't find any relevant estimate data for your question. This could mean the query terms don't match available cost codes, descriptions, or categories. Try rephrasing with specific cost codes, work categories (material, labor, subcontractor), or area names.",
      queryType,
      metadata: {
        totalResults: 0,
        avgSimilarity: 0,
        suggestion: "Try searching with specific cost codes, categories like 'material' or 'labor', or work areas."
      },
      sources: [],
      insights: null,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Create error response
   */
  createErrorResponse(question, error) {
    return {
      success: false,
      question,
      answer: "I encountered an error while processing your estimate query. Please try rephrasing your question or contact support if the issue persists.",
      error: error.message,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Test the RAG service
   */
  async testEstimateRAG() {
    try {
      console.log('🧪 Testing Estimate RAG service...');
      
      const testQuestion = 'What electrical work was estimated and how does it compare to budget?';
      const testEmbedding = await this.generateQueryEmbedding(testQuestion);
      
      const isValid = Array.isArray(testEmbedding) && 
                     testEmbedding.length > 0 && 
                     typeof testEmbedding[0] === 'number';
      
      if (isValid) {
        console.log(`✅ Estimate RAG test passed (embedding dimension: ${testEmbedding.length})`);
        return { success: true, dimension: testEmbedding.length };
      } else {
        throw new Error('Invalid embedding generated');
      }
      
    } catch (error) {
      console.error('❌ Estimate RAG test failed:', error.message);
      return { success: false, error: error.message };
    }
  }
}

// Export singleton instance
const estimateRAGService = new EstimateRAGService();
export default estimateRAGService;