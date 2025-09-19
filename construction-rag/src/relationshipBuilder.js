/**
 * Estimate-Focused Graph Relationship Builder
 * Automatically detects and creates relationships between construction estimate entities only
 */

import dotenv from 'dotenv';

dotenv.config();

class EstimateFocusedRelationshipBuilder {
  constructor() {
    // Relationship types optimized for estimate data analysis
    this.relationshipTypes = {
      SAME_COST_CODE: {
        type: 'SAME_COST_CODE',
        description: 'Estimate items sharing the same cost code',
        strength: 0.9,
        bidirectional: true,
        maxDistance: 1
      },

      SAME_CATEGORY: {
        type: 'SAME_CATEGORY',
        description: 'Estimate items in the same category (Material, Labor, etc.)',
        strength: 0.6,
        bidirectional: true,
        maxDistance: 2
      },

      SAME_SCOPE: {
        type: 'SAME_SCOPE',
        description: 'Estimate items under the same task scope',
        strength: 0.7,
        bidirectional: true,
        maxDistance: 2
      },

      BUDGET_VARIANCE_SIMILARITY: {
        type: 'BUDGET_VARIANCE_SIMILARITY',
        description: 'Estimate items with similar budget variance patterns',
        strength: 0.8,
        bidirectional: true,
        threshold: 0.15, // 15% variance similarity threshold
        maxDistance: 2
      },

      HIGH_VARIANCE_GROUP: {
        type: 'HIGH_VARIANCE_GROUP',
        description: 'Estimate items with significant budget variance (>20%)',
        strength: 0.75,
        bidirectional: true,
        threshold: 0.2, // 20% variance threshold
        maxDistance: 1
      },

      COST_SIMILARITY: {
        type: 'COST_SIMILARITY',
        description: 'Estimate items with similar cost amounts',
        strength: 0.5,
        bidirectional: true,
        threshold: 0.1, // 10% cost difference threshold
        maxDistance: 2
      },

      AREA_GROUPING: {
        type: 'AREA_GROUPING', 
        description: 'Estimate items in the same work area',
        strength: 0.6,
        bidirectional: true,
        maxDistance: 2
      },

      RATE_SIMILARITY: {
        type: 'RATE_SIMILARITY',
        description: 'Estimate items with similar unit rates',
        strength: 0.55,
        bidirectional: true,
        threshold: 0.15, // 15% rate difference threshold
        maxDistance: 2
      },

      QUANTITY_SCALE_SIMILARITY: {
        type: 'QUANTITY_SCALE_SIMILARITY',
        description: 'Estimate items with similar quantity scales',
        strength: 0.45,
        bidirectional: true,
        threshold: 0.3, // 30% quantity difference threshold
        maxDistance: 2
      },

      BUDGET_PERFORMANCE_GROUP: {
        type: 'BUDGET_PERFORMANCE_GROUP',
        description: 'Estimate items with similar budget performance (over/under/on budget)',
        strength: 0.65,
        bidirectional: true,
        maxDistance: 2
      }
    };

    // Enhanced batch processing configuration for estimates
    this.batchSize = 150;
    this.maxRelationshipsPerEntity = 25;
    this.processingTimeout = 300000; // 5 minutes
  }

  // Main method to build relationships for estimate entities
  async buildRelationships(entities, db) {
    if (!entities || entities.length === 0) {
      console.log('⚠️  No entities provided for relationship building');
      return [];
    }

    // Filter to only estimate entities
    const estimateEntities = entities.filter(entity => 
      (entity.entity_type === 'estimate_row' || entity.entityType === 'estimate_row')
    );

    if (estimateEntities.length === 0) {
      console.log('⚠️  No estimate entities found for relationship building');
      return [];
    }

    console.log(`🔗 Building estimate relationships for ${estimateEntities.length} entities...`);
    const startTime = Date.now();

    try {
      const relationships = [];
      
      // Build relationships by type
      for (const [typeName, config] of Object.entries(this.relationshipTypes)) {
        console.log(`🔍 Processing ${typeName} relationships...`);
        
        const typeRelationships = await this.buildRelationshipsByType(
          estimateEntities, 
          typeName, 
          config, 
          db
        );
        
        relationships.push(...typeRelationships);
        console.log(`✅ Created ${typeRelationships.length} ${typeName} relationships`);
      }

      // Store relationships in database
      console.log(`💾 Storing ${relationships.length} estimate relationships...`);
      const storedRelationships = await this.storeRelationships(relationships, db);

      const duration = Date.now() - startTime;
      console.log(`🎉 Estimate relationship building completed in ${duration}ms`);
      console.log(`📊 Total relationships created: ${storedRelationships.length}`);

      return storedRelationships;

    } catch (error) {
      console.error('❌ Error building estimate relationships:', error.message);
      throw error;
    }
  }

  // Build relationships for a specific type
  async buildRelationshipsByType(entities, typeName, config, db) {
    const relationships = [];
    
    switch (typeName) {
      case 'SAME_COST_CODE':
        relationships.push(...this.findSameCostCodeRelationships(entities, config));
        break;
        
      case 'SAME_CATEGORY':
        relationships.push(...this.findSameCategoryRelationships(entities, config));
        break;
        
      case 'SAME_SCOPE':
        relationships.push(...this.findSameScopeRelationships(entities, config));
        break;
        
      case 'BUDGET_VARIANCE_SIMILARITY':
        relationships.push(...this.findBudgetVarianceSimilarityRelationships(entities, config));
        break;
        
      case 'HIGH_VARIANCE_GROUP':
        relationships.push(...this.findHighVarianceGroupRelationships(entities, config));
        break;
        
      case 'COST_SIMILARITY':
        relationships.push(...this.findCostSimilarityRelationships(entities, config));
        break;
        
      case 'AREA_GROUPING':
        relationships.push(...this.findAreaGroupingRelationships(entities, config));
        break;

      case 'RATE_SIMILARITY':
        relationships.push(...this.findRateSimilarityRelationships(entities, config));
        break;

      case 'QUANTITY_SCALE_SIMILARITY':
        relationships.push(...this.findQuantityScaleSimilarityRelationships(entities, config));
        break;

      case 'BUDGET_PERFORMANCE_GROUP':
        relationships.push(...this.findBudgetPerformanceGroupRelationships(entities, config));
        break;
        
      default:
        console.warn(`⚠️  Unknown relationship type: ${typeName}`);
    }
    
    return relationships;
  }

  // Find estimate entities with the same cost code
  findSameCostCodeRelationships(entities, config) {
    const relationships = [];
    const costCodeGroups = {};
    
    // Group entities by cost code
    entities.forEach(entity => {
      const costCode = entity.cost_code || entity.costCode;
      if (!costCode) return;
      
      if (!costCodeGroups[costCode]) {
        costCodeGroups[costCode] = [];
      }
      costCodeGroups[costCode].push(entity);
    });
    
    // Create relationships within each group
    Object.entries(costCodeGroups).forEach(([costCode, groupEntities]) => {
      if (groupEntities.length < 2) return;
      
      for (let i = 0; i < groupEntities.length; i++) {
        for (let j = i + 1; j < groupEntities.length; j++) {
          const sourceEntity = groupEntities[i];
          const targetEntity = groupEntities[j];
          
          // Calculate enhanced strength based on estimate similarity
          const enhancedStrength = this.calculateCostCodeSimilarityStrength(
            sourceEntity, 
            targetEntity, 
            config.strength
          );
          
          relationships.push({
            sourceId: sourceEntity.id,
            targetId: targetEntity.id,
            type: config.type,
            strength: enhancedStrength,
            metadata: { 
              costCode, 
              sharedAttribute: 'cost_code',
              sourceAmount: sourceEntity.total_amount || sourceEntity.totalAmount,
              targetAmount: targetEntity.total_amount || targetEntity.totalAmount
            }
          });
        }
      }
    });
    
    return relationships;
  }

  // Find estimate entities in the same category
  findSameCategoryRelationships(entities, config) {
    const relationships = [];
    const categoryGroups = {};
    
    // Group by category
    entities.forEach(entity => {
      const category = entity.category;
      if (!category || category === 'other') return;
      
      if (!categoryGroups[category]) {
        categoryGroups[category] = [];
      }
      categoryGroups[category].push(entity);
    });
    
    // Create relationships within categories (but limit to avoid explosion)
    Object.entries(categoryGroups).forEach(([category, groupEntities]) => {
      if (groupEntities.length < 2) return;
      
      // For large groups, only connect each entity to a few others
      const maxConnectionsPerEntity = Math.min(6, groupEntities.length - 1);
      
      groupEntities.forEach((entity, index) => {
        let connections = 0;
        
        for (let i = index + 1; i < groupEntities.length && connections < maxConnectionsPerEntity; i++) {
          const targetEntity = groupEntities[i];
          
          // Enhanced strength based on category and cost similarity
          const enhancedStrength = this.calculateCategorySimilarityStrength(
            entity,
            targetEntity,
            config.strength
          );
          
          relationships.push({
            sourceId: entity.id,
            targetId: targetEntity.id,
            type: config.type,
            strength: enhancedStrength,
            metadata: { 
              category, 
              sharedAttribute: 'category',
              sourceAmount: entity.total_amount || entity.totalAmount,
              targetAmount: targetEntity.total_amount || targetEntity.totalAmount
            }
          });
          
          connections++;
        }
      });
    });
    
    return relationships;
  }

  // Find estimate entities with the same task scope
  findSameScopeRelationships(entities, config) {
    const relationships = [];
    const scopeGroups = {};
    
    entities.forEach(entity => {
      const scope = entity.task_scope || entity.taskScope;
      if (!scope) return;
      
      if (!scopeGroups[scope]) {
        scopeGroups[scope] = [];
      }
      scopeGroups[scope].push(entity);
    });
    
    // Create relationships within scopes
    Object.entries(scopeGroups).forEach(([scope, groupEntities]) => {
      if (groupEntities.length < 2) return;
      
      const maxConnectionsPerEntity = Math.min(4, groupEntities.length - 1);
      
      groupEntities.forEach((entity, index) => {
        let connections = 0;
        
        for (let i = index + 1; i < groupEntities.length && connections < maxConnectionsPerEntity; i++) {
          const targetEntity = groupEntities[i];
          
          relationships.push({
            sourceId: entity.id,
            targetId: targetEntity.id,
            type: config.type,
            strength: config.strength,
            metadata: { 
              taskScope: scope, 
              sharedAttribute: 'task_scope',
              sourceCategory: entity.category,
              targetCategory: targetEntity.category
            }
          });
          
          connections++;
        }
      });
    });
    
    return relationships;
  }

  // Find estimate entities with similar budget variance patterns
  findBudgetVarianceSimilarityRelationships(entities, config) {
    const relationships = [];
    
    // Filter entities with budget data
    const entitiesWithBudgets = entities.filter(entity => {
      const budgeted = entity.budgeted_amount || entity.budgetedAmount || 0;
      return budgeted > 0;
    });
    
    for (let i = 0; i < entitiesWithBudgets.length; i++) {
      const entity1 = entitiesWithBudgets[i];
      const variance1 = this.calculateVariancePercent(entity1);
      
      for (let j = i + 1; j < entitiesWithBudgets.length; j++) {
        const entity2 = entitiesWithBudgets[j];
        const variance2 = this.calculateVariancePercent(entity2);
        
        // Only connect entities from the same project
        if (entity1.project_id !== entity2.project_id) continue;
        
        // Check if variance patterns are similar
        const varianceDiff = Math.abs(variance1 - variance2);
        if (varianceDiff <= config.threshold * 100) { // Convert threshold to percentage
          const similarity = 1 - (varianceDiff / (config.threshold * 100));
          const adjustedStrength = config.strength * similarity;
          
          relationships.push({
            sourceId: entity1.id,
            targetId: entity2.id,
            type: config.type,
            strength: adjustedStrength,
            metadata: {
              sourceVariance: variance1,
              targetVariance: variance2,
              varianceDifference: varianceDiff,
              similarity,
              sharedAttribute: 'budget_variance_pattern'
            }
          });
        }
      }
    }
    
    return relationships;
  }

  // Find estimate entities with high budget variance (problem items)
  findHighVarianceGroupRelationships(entities, config) {
    const relationships = [];
    
    const highVarianceItems = entities.filter(entity => {
      const budgeted = entity.budgeted_amount || entity.budgetedAmount || 0;
      if (budgeted <= 0) return false;
      
      const variance = Math.abs(this.calculateVariancePercent(entity));
      return variance > config.threshold * 100;
    });
    
    if (highVarianceItems.length < 2) return relationships;
    
    // Group high variance items by project
    const projectGroups = {};
    highVarianceItems.forEach(entity => {
      const projectId = entity.project_id;
      if (!projectGroups[projectId]) {
        projectGroups[projectId] = [];
      }
      projectGroups[projectId].push(entity);
    });
    
    // Create relationships within each project's high variance items
    Object.values(projectGroups).forEach(projectItems => {
      if (projectItems.length < 2) return;
      
      for (let i = 0; i < projectItems.length; i++) {
        for (let j = i + 1; j < Math.min(i + 4, projectItems.length); j++) {
          const entity1 = projectItems[i];
          const entity2 = projectItems[j];
          
          relationships.push({
            sourceId: entity1.id,
            targetId: entity2.id,
            type: config.type,
            strength: config.strength,
            metadata: {
              sourceVariance: this.calculateVariancePercent(entity1),
              targetVariance: this.calculateVariancePercent(entity2),
              threshold: config.threshold * 100,
              sharedAttribute: 'high_variance',
              problemCategory: 'budget_overrun'
            }
          });
        }
      }
    });
    
    return relationships;
  }

  // Find estimate entities with similar costs
  findCostSimilarityRelationships(entities, config) {
    const relationships = [];
    
    for (let i = 0; i < entities.length; i++) {
      const entity1 = entities[i];
      const amount1 = entity1.total_amount || entity1.totalAmount || 0;
      
      if (amount1 <= 0) continue;
      
      for (let j = i + 1; j < entities.length; j++) {
        const entity2 = entities[j];
        const amount2 = entity2.total_amount || entity2.totalAmount || 0;
        
        if (amount2 <= 0) continue;
        
        // Only connect entities from the same project
        if (entity1.project_id !== entity2.project_id) continue;
        
        // Calculate cost similarity
        const maxAmount = Math.max(amount1, amount2);
        const minAmount = Math.min(amount1, amount2);
        const similarity = minAmount / maxAmount;
        
        if (similarity >= (1 - config.threshold)) {
          const adjustedStrength = config.strength * similarity;
          
          relationships.push({
            sourceId: entity1.id,
            targetId: entity2.id,
            type: config.type,
            strength: adjustedStrength,
            metadata: {
              amount1,
              amount2,
              similarity,
              amountDifference: Math.abs(amount1 - amount2),
              sharedAttribute: 'similar_cost'
            }
          });
        }
      }
    }
    
    return relationships;
  }

  // Find estimate entities in the same work area
  findAreaGroupingRelationships(entities, config) {
    const relationships = [];
    const areaGroups = {};
    
    entities.forEach(entity => {
      const area = entity.area;
      if (!area) return;
      
      if (!areaGroups[area]) {
        areaGroups[area] = [];
      }
      areaGroups[area].push(entity);
    });
    
    // Create relationships within areas
    Object.entries(areaGroups).forEach(([area, groupEntities]) => {
      if (groupEntities.length < 2) return;
      
      const maxConnectionsPerEntity = Math.min(5, groupEntities.length - 1);
      
      groupEntities.forEach((entity, index) => {
        let connections = 0;
        
        for (let i = index + 1; i < groupEntities.length && connections < maxConnectionsPerEntity; i++) {
          const targetEntity = groupEntities[i];
          
          relationships.push({
            sourceId: entity.id,
            targetId: targetEntity.id,
            type: config.type,
            strength: config.strength,
            metadata: { 
              area, 
              sharedAttribute: 'area',
              sourceCategory: entity.category,
              targetCategory: targetEntity.category
            }
          });
          
          connections++;
        }
      });
    });
    
    return relationships;
  }

  // Find estimate entities with similar unit rates
  findRateSimilarityRelationships(entities, config) {
    const relationships = [];
    
    const entitiesWithRates = entities.filter(entity => {
      const rate = entity.rate || entity.raw_data?.rate || 0;
      const qty = entity.qty || entity.raw_data?.qty || 0;
      return rate > 0 && qty > 0;
    });
    
    for (let i = 0; i < entitiesWithRates.length; i++) {
      const entity1 = entitiesWithRates[i];
      const rate1 = entity1.rate || entity1.raw_data?.rate || 0;
      
      for (let j = i + 1; j < entitiesWithRates.length; j++) {
        const entity2 = entitiesWithRates[j];
        const rate2 = entity2.rate || entity2.raw_data?.rate || 0;
        
        // Only connect entities from same project and category
        if (entity1.project_id !== entity2.project_id) continue;
        if (entity1.category !== entity2.category) continue;
        
        const maxRate = Math.max(rate1, rate2);
        const minRate = Math.min(rate1, rate2);
        const similarity = minRate / maxRate;
        
        if (similarity >= (1 - config.threshold)) {
          const adjustedStrength = config.strength * similarity;
          
          relationships.push({
            sourceId: entity1.id,
            targetId: entity2.id,
            type: config.type,
            strength: adjustedStrength,
            metadata: {
              sourceRate: rate1,
              targetRate: rate2,
              similarity,
              units: entity1.units || entity1.raw_data?.units || 'ea',
              sharedAttribute: 'similar_rate'
            }
          });
        }
      }
    }
    
    return relationships;
  }

  // Find estimate entities with similar quantity scales
  findQuantityScaleSimilarityRelationships(entities, config) {
    const relationships = [];
    
    const entitiesWithQuantities = entities.filter(entity => {
      const qty = entity.qty || entity.raw_data?.qty || 0;
      return qty > 0;
    });
    
    for (let i = 0; i < entitiesWithQuantities.length; i++) {
      const entity1 = entitiesWithQuantities[i];
      const qty1 = entity1.qty || entity1.raw_data?.qty || 0;
      
      for (let j = i + 1; j < entitiesWithQuantities.length; j++) {
        const entity2 = entitiesWithQuantities[j];
        const qty2 = entity2.qty || entity2.raw_data?.qty || 0;
        
        // Only connect entities with same units and category
        const units1 = entity1.units || entity1.raw_data?.units || 'ea';
        const units2 = entity2.units || entity2.raw_data?.units || 'ea';
        
        if (units1 !== units2) continue;
        if (entity1.category !== entity2.category) continue;
        if (entity1.project_id !== entity2.project_id) continue;
        
        const maxQty = Math.max(qty1, qty2);
        const minQty = Math.min(qty1, qty2);
        const similarity = minQty / maxQty;
        
        if (similarity >= (1 - config.threshold)) {
          const adjustedStrength = config.strength * similarity;
          
          relationships.push({
            sourceId: entity1.id,
            targetId: entity2.id,
            type: config.type,
            strength: adjustedStrength,
            metadata: {
              sourceQty: qty1,
              targetQty: qty2,
              units: units1,
              similarity,
              sharedAttribute: 'similar_quantity_scale'
            }
          });
        }
      }
    }
    
    return relationships;
  }

  // Find estimate entities with similar budget performance
  findBudgetPerformanceGroupRelationships(entities, config) {
    const relationships = [];
    
    const entitiesWithBudgets = entities.filter(entity => {
      const budgeted = entity.budgeted_amount || entity.budgetedAmount || 0;
      return budgeted > 0;
    });
    
    // Group by budget performance status
    const performanceGroups = {
      'over': [],
      'under': [],
      'on_budget': []
    };
    
    entitiesWithBudgets.forEach(entity => {
      const variancePercent = this.calculateVariancePercent(entity);
      let status = 'on_budget';
      
      if (variancePercent > 10) status = 'over';
      else if (variancePercent < -10) status = 'under';
      
      performanceGroups[status].push(entity);
    });
    
    // Create relationships within each performance group
    Object.entries(performanceGroups).forEach(([status, groupEntities]) => {
      if (groupEntities.length < 2) return;
      
      const maxConnectionsPerEntity = Math.min(4, groupEntities.length - 1);
      
      groupEntities.forEach((entity, index) => {
        let connections = 0;
        
        for (let i = index + 1; i < groupEntities.length && connections < maxConnectionsPerEntity; i++) {
          const targetEntity = groupEntities[i];
          
          // Only connect entities from same project
          if (entity.project_id !== targetEntity.project_id) continue;
          
          relationships.push({
            sourceId: entity.id,
            targetId: targetEntity.id,
            type: config.type,
            strength: config.strength,
            metadata: {
              budgetPerformance: status,
              sourceVariance: this.calculateVariancePercent(entity),
              targetVariance: this.calculateVariancePercent(targetEntity),
              sharedAttribute: 'budget_performance_status'
            }
          });
          
          connections++;
        }
      });
    });
    
    return relationships;
  }

  // Helper methods for enhanced relationship strength calculation
  calculateCostCodeSimilarityStrength(entity1, entity2, baseStrength) {
    let strength = baseStrength;
    
    const amount1 = entity1.total_amount || entity1.totalAmount || 0;
    const amount2 = entity2.total_amount || entity2.totalAmount || 0;
    
    // Boost strength for similar amounts
    if (amount1 > 0 && amount2 > 0) {
      const similarity = Math.min(amount1, amount2) / Math.max(amount1, amount2);
      strength *= (0.7 + 0.3 * similarity);
    }
    
    // Boost for same category
    if (entity1.category === entity2.category) {
      strength *= 1.1;
    }
    
    return Math.min(strength, 1.0);
  }

  calculateCategorySimilarityStrength(entity1, entity2, baseStrength) {
    let strength = baseStrength;
    
    // Boost for same cost code pattern
    const code1 = (entity1.cost_code || '').toUpperCase();
    const code2 = (entity2.cost_code || '').toUpperCase();
    
    if (code1 && code2) {
      const pattern1 = code1.match(/\d+[A-Z]$/);
      const pattern2 = code2.match(/\d+[A-Z]$/);
      
      if (pattern1 && pattern2 && pattern1[0].slice(-1) === pattern2[0].slice(-1)) {
        strength *= 1.15; // Same cost code suffix (M, L, S, etc.)
      }
    }
    
    return Math.min(strength, 1.0);
  }

  calculateVariancePercent(entity) {
    const actual = entity.total_amount || entity.totalAmount || 0;
    const budgeted = entity.budgeted_amount || entity.budgetedAmount || 0;
    
    if (budgeted <= 0) return 0;
    return ((actual - budgeted) / budgeted) * 100;
  }

  // Store relationships in database with enhanced deduplication
  async storeRelationships(relationships, db) {
    if (!relationships || relationships.length === 0) {
      return [];
    }

    console.log(`💾 Storing ${relationships.length} estimate relationships...`);
    
    try {
      // Enhanced deduplication and filtering
      const uniqueRelationships = this.deduplicateAndFilterRelationships(relationships);
      console.log(`🧹 Deduplicated to ${uniqueRelationships.length} unique relationships`);
      
      // Store in batches
      const storedRelationships = [];
      
      for (let i = 0; i < uniqueRelationships.length; i += this.batchSize) {
        const batch = uniqueRelationships.slice(i, i + this.batchSize);
        
        for (const relationship of batch) {
          try {
            const stored = await db.createRelationship(
              relationship.sourceId,
              relationship.targetId,
              relationship.type,
              relationship.strength
            );
            
            if (stored) {
              storedRelationships.push(stored);
            }
            
          } catch (error) {
            // Skip duplicates and continue
            if (!error.message.includes('duplicate') && !error.message.includes('conflict')) {
              console.error('❌ Error storing relationship:', error.message);
            }
          }
        }
        
        console.log(`📦 Stored batch ${Math.floor(i/this.batchSize) + 1}/${Math.ceil(uniqueRelationships.length/this.batchSize)}`);
      }
      
      console.log(`✅ Successfully stored ${storedRelationships.length} estimate relationships`);
      return storedRelationships;
      
    } catch (error) {
      console.error('❌ Error storing estimate relationships:', error.message);
      throw error;
    }
  }

  // Enhanced deduplication with quality filtering
  deduplicateAndFilterRelationships(relationships) {
    const seen = new Set();
    const unique = [];
    
    relationships
      .filter(rel => rel.strength >= 0.3) // Filter weak relationships
      .sort((a, b) => b.strength - a.strength) // Prioritize stronger relationships
      .forEach(rel => {
        // Create a unique key for the relationship
        const key1 = `${rel.sourceId}-${rel.targetId}-${rel.type}`;
        const key2 = `${rel.targetId}-${rel.sourceId}-${rel.type}`; // Bidirectional check
        
        if (!seen.has(key1) && !seen.has(key2)) {
          seen.add(key1);
          unique.push(rel);
        }
      });
    
    return unique;
  }

  // Get enhanced relationship statistics for estimates
  async getEstimateRelationshipStats(db) {
    try {
      const stats = await db.query(`
        SELECT 
          r.type,
          COUNT(*) as count,
          AVG(r.strength) as avg_strength,
          MIN(r.strength) as min_strength,
          MAX(r.strength) as max_strength
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e1.entity_type = 'estimate_row' 
        AND e2.entity_type = 'estimate_row'
        GROUP BY r.type
        ORDER BY count DESC
      `);
      
      const totalStats = await db.query(`
        SELECT 
          COUNT(*) as total_relationships,
          COUNT(DISTINCT r.source_id) as unique_sources,
          COUNT(DISTINCT r.target_id) as unique_targets,
          AVG(r.strength) as overall_avg_strength
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e1.entity_type = 'estimate_row' 
        AND e2.entity_type = 'estimate_row'
      `);
      
      // Get relationship distribution by strength
      const strengthDistribution = await db.query(`
        SELECT 
          CASE 
            WHEN strength >= 0.8 THEN 'very_strong'
            WHEN strength >= 0.6 THEN 'strong'
            WHEN strength >= 0.4 THEN 'moderate'
            ELSE 'weak'
          END as strength_category,
          COUNT(*) as count
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        WHERE e1.entity_type = 'estimate_row'
        GROUP BY strength_category
        ORDER BY count DESC
      `);
      
      return {
        byType: stats.rows,
        overall: totalStats.rows[0],
        strengthDistribution: strengthDistribution.rows,
        focus: 'Construction Estimates Only',
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      console.error('❌ Error getting estimate relationship stats:', error.message);
      return { error: error.message };
    }
  }

  // Find estimate entities that need relationship building
  async findEstimateEntitiesWithoutRelationships(db) {
    try {
      const result = await db.query(`
        SELECT e.*
        FROM entities e
        LEFT JOIN relationships r ON e.id = r.source_id OR e.id = r.target_id
        WHERE r.id IS NULL 
        AND e.entity_type = 'estimate_row'
        ORDER BY e.created_at DESC
      `);
      
      return result.rows;
      
    } catch (error) {
      console.error('❌ Error finding estimate entities without relationships:', error.message);
      throw error;
    }
  }

  // Clean up weak estimate relationships
  async cleanupEstimateRelationships(db, minStrength = 0.3) {
    try {
      console.log(`🧹 Cleaning up estimate relationships with strength < ${minStrength}...`);
      
      const result = await db.query(`
        DELETE FROM relationships 
        WHERE strength < $1
        AND source_id IN (
          SELECT id FROM entities WHERE entity_type = 'estimate_row'
        )
      `, [minStrength]);
      
      console.log(`✅ Removed ${result.rowCount} weak estimate relationships`);
      return result.rowCount;
      
    } catch (error) {
      console.error('❌ Error cleaning up estimate relationships:', error.message);
      throw error;
    }
  }
}

// Export singleton instance
const estimateFocusedRelationshipBuilder = new EstimateFocusedRelationshipBuilder();
export default estimateFocusedRelationshipBuilder;