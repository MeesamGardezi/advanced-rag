/**
 * OpenAI Configuration
 * Centralized OpenAI settings for embeddings and completions
 */

import dotenv from 'dotenv';

dotenv.config();

// OpenAI API configuration
export const openaiConfig = {
  // API settings
  apiKey: process.env.OPENAI_API_KEY,
  organization: process.env.OPENAI_ORG_ID,
  baseURL: process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1',
  
  // Default models
  models: {
    embedding: process.env.EMBEDDING_MODEL || 'text-embedding-3-small',
    completion: process.env.COMPLETION_MODEL || 'gpt-4',
    fallback: {
      embedding: 'text-embedding-ada-002',
      completion: 'gpt-3.5-turbo'
    }
  },
  
  // Model specifications
  modelSpecs: {
    'text-embedding-3-small': {
      dimensions: 1536,
      maxTokens: 8191,
      costPer1k: 0.00002,
      description: 'High performance, cost-effective embedding model'
    },
    'text-embedding-3-large': {
      dimensions: 3072,
      maxTokens: 8191,
      costPer1k: 0.00013,
      description: 'Highest performance embedding model'
    },
    'text-embedding-ada-002': {
      dimensions: 1536,
      maxTokens: 8191,
      costPer1k: 0.0001,
      description: 'Legacy embedding model'
    },
    'gpt-4': {
      maxTokens: 8192,
      costPer1kInput: 0.03,
      costPer1kOutput: 0.06,
      description: 'Most capable GPT-4 model'
    },
    'gpt-4-turbo': {
      maxTokens: 128000,
      costPer1kInput: 0.01,
      costPer1kOutput: 0.03,
      description: 'GPT-4 with larger context window'
    },
    'gpt-3.5-turbo': {
      maxTokens: 4096,
      costPer1kInput: 0.0015,
      costPer1kOutput: 0.002,
      description: 'Fast and cost-effective model'
    }
  }
};

// Embedding configuration
export const embeddingConfig = {
  // Batch processing
  maxBatchSize: parseInt(process.env.MAX_EMBEDDING_BATCH_SIZE) || 100,
  batchTimeout: 30000, // 30 seconds
  
  // Rate limiting
  maxRequestsPerMinute: parseInt(process.env.OPENAI_RPM_LIMIT) || 500,
  rateLimitWindow: 60000, // 1 minute
  
  // Retry configuration
  maxRetries: 3,
  retryDelay: 1000,     // Start with 1 second
  retryBackoff: 2,      // Double delay each retry
  retryOnRateLimit: true,
  
  // Quality settings
  truncateInput: true,  // Truncate if too long
  normalizeInput: true, // Clean and normalize text
  
  // Performance optimization
  enableCaching: process.env.NODE_ENV !== 'test',
  cacheExpiry: 24 * 60 * 60 * 1000, // 24 hours
  
  // Similarity search
  defaultSimilarityThreshold: parseFloat(process.env.SIMILARITY_THRESHOLD) || 0.7,
  maxSimilarityResults: parseInt(process.env.MAX_SIMILARITY_RESULTS) || 50
};

// Completion configuration
export const completionConfig = {
  // Model parameters
  temperature: parseFloat(process.env.OPENAI_TEMPERATURE) || 0.1,
  maxTokens: parseInt(process.env.MAX_COMPLETION_TOKENS) || 800,
  topP: parseFloat(process.env.OPENAI_TOP_P) || 1,
  frequencyPenalty: parseFloat(process.env.OPENAI_FREQUENCY_PENALTY) || 0,
  presencePenalty: parseFloat(process.env.OPENAI_PRESENCE_PENALTY) || 0,
  
  // Response settings
  stream: false, // Set to true for streaming responses
  logprobs: false,
  echo: false,
  
  // Safety settings
  enableModeration: process.env.NODE_ENV === 'production',
  maxRetries: 2,
  timeout: 60000, // 60 seconds
  
  // Context management
  maxContextTokens: 6000, // Leave room for response
  contextTruncationStrategy: 'middle', // 'start', 'middle', 'end'
  
  // Response formatting
  responseFormat: 'text', // 'text' or 'json'
  includeMetadata: true
};

// RAG-specific configuration
export const ragConfig = {
  // Context building
  maxContextChunks: 10,
  maxContextTokens: 4000,
  chunkOverlap: 200,
  
  // Relevance scoring
  minRelevanceScore: 0.6,
  diversityFactor: 0.3, // Balance between relevance and diversity
  
  // Query processing
  expandQuery: true,     // Add related terms
  rewriteQuery: false,   // Rewrite unclear queries
  
  // Response generation
  includeSourceInfo: true,
  includeCitations: true,
  responseStyle: 'detailed', // 'brief', 'detailed', 'technical'
  
  // Construction-specific settings
  constructionContext: {
    emphasizeCostCodes: true,
    includeBudgetAnalysis: true,
    highlightVariances: true,
    groupByCategory: true,
    showRelatedItems: true
  },
  
  // Answer templates
  templates: {
    cost_analysis: `Based on the construction data, here's the cost analysis:

{analysis}

Key Points:
{key_points}

Sources: {sources}`,
    
    comparison: `Comparing construction items:

{comparison_table}

Summary: {summary}

Related Items: {related_items}`,
    
    variance_analysis: `Budget Variance Analysis:

{variance_breakdown}

Issues Found: {issues}
Recommendations: {recommendations}`
  }
};

// Prompt templates
export const promptTemplates = {
  // System prompts
  system: {
    construction_expert: `You are an expert construction project analyst with deep knowledge of cost codes, scheduling, and project management. You analyze construction data to provide accurate, actionable insights.`,
    
    cost_analyst: `You are a construction cost analyst specializing in budget variance analysis, cost code interpretation, and financial project insights. Always provide specific dollar amounts and percentages when available.`,
    
    project_manager: `You are an experienced construction project manager who understands schedules, resource allocation, and project coordination. Focus on practical, actionable advice.`
  },
  
  // User prompt templates
  user: {
    cost_query: `Based on this construction project data, please answer the following question about costs and budgets:

Context: {context}

Question: {question}

Please provide specific amounts, cost codes, and variance analysis where relevant.`,
    
    comparison_query: `Compare these construction items and provide insights:

Data: {context}

Question: {question}

Focus on differences in costs, categories, and any notable patterns.`,
    
    analysis_query: `Analyze this construction project data:

Project Data: {context}

Analysis Request: {question}

Provide detailed insights including financial impact and recommendations.`
  }
};

// Cost and usage tracking
export const usageConfig = {
  // Track API usage
  trackUsage: process.env.NODE_ENV === 'production',
  
  // Cost monitoring
  enableCostMonitoring: true,
  costAlerts: {
    daily: parseFloat(process.env.DAILY_COST_LIMIT) || 50,
    monthly: parseFloat(process.env.MONTHLY_COST_LIMIT) || 500
  },
  
  // Usage limits
  limits: {
    embeddingsPerHour: 10000,
    completionsPerHour: 1000,
    tokensPerDay: 1000000
  },
  
  // Logging
  logRequests: process.env.NODE_ENV === 'development',
  logResponses: false, // Set to true for debugging
  logErrors: true,
  
  // Analytics
  enableAnalytics: true,
  analyticsRetentionDays: 30
};

// Get OpenAI configuration with validation
export function getOpenAIConfig() {
  validateOpenAIConfig();
  
  return {
    ...openaiConfig,
    embedding: embeddingConfig,
    completion: completionConfig,
    rag: ragConfig,
    prompts: promptTemplates,
    usage: usageConfig
  };
}

// Validate OpenAI configuration
export function validateOpenAIConfig() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }
  
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey.startsWith('sk-')) {
    throw new Error('Invalid OpenAI API key format');
  }
  
  // Validate model names
  const embeddingModel = openaiConfig.models.embedding;
  const completionModel = openaiConfig.models.completion;
  
  if (!openaiConfig.modelSpecs[embeddingModel]) {
    console.warn(`⚠️  Unknown embedding model: ${embeddingModel}`);
  }
  
  if (!openaiConfig.modelSpecs[completionModel]) {
    console.warn(`⚠️  Unknown completion model: ${completionModel}`);
  }
  
  // Validate numeric parameters
  if (completionConfig.temperature < 0 || completionConfig.temperature > 2) {
    throw new Error('Temperature must be between 0 and 2');
  }
  
  if (completionConfig.maxTokens < 1 || completionConfig.maxTokens > 8192) {
    console.warn('⚠️  Max tokens should be between 1 and model limit');
  }
  
  console.log('✅ OpenAI configuration validated');
  console.log(`🤖 Embedding model: ${embeddingModel}`);
  console.log(`💬 Completion model: ${completionModel}`);
  
  return true;
}

// Get model specifications
export function getModelSpec(modelName) {
  return openaiConfig.modelSpecs[modelName] || null;
}

// Calculate estimated cost
export function estimateCost(modelName, inputTokens, outputTokens = 0) {
  const spec = getModelSpec(modelName);
  if (!spec) return null;
  
  let cost = 0;
  
  // Embedding models
  if (spec.costPer1k) {
    cost = (inputTokens / 1000) * spec.costPer1k;
  }
  
  // Completion models
  if (spec.costPer1kInput && spec.costPer1kOutput) {
    cost = (inputTokens / 1000) * spec.costPer1kInput +
           (outputTokens / 1000) * spec.costPer1kOutput;
  }
  
  return Math.round(cost * 100000) / 100000; // Round to 5 decimal places
}

// Get optimal batch size for model
export function getOptimalBatchSize(modelName) {
  const spec = getModelSpec(modelName);
  if (!spec) return embeddingConfig.maxBatchSize;
  
  // Adjust batch size based on model capabilities
  if (modelName.includes('text-embedding-3')) {
    return Math.min(embeddingConfig.maxBatchSize, 100);
  }
  
  return Math.min(embeddingConfig.maxBatchSize, 50);
}

// Create OpenAI client options
export function getClientOptions() {
  return {
    apiKey: openaiConfig.apiKey,
    organization: openaiConfig.organization,
    baseURL: openaiConfig.baseURL,
    timeout: 60000,
    maxRetries: 3
  };
}

export default {
  getOpenAIConfig,
  validateOpenAIConfig,
  getModelSpec,
  estimateCost,
  getOptimalBatchSize,
  getClientOptions,
  openaiConfig,
  embeddingConfig,
  completionConfig,
  ragConfig,
  promptTemplates,
  usageConfig
};