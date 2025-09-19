/**
 * Graph Relationship Builder
 * Automatically detects and creates relationships between construction entities
 */

import dotenv from 'dotenv';

dotenv.config();

class RelationshipBuilder {
  constructor() {
    // Relationship types and their detection rules
    this.relationshipTypes = {
      SAME_COST_CODE: {
        type: 'SAME_COST_CODE',
        description: 'Entities sharing the same cost code',
        strength: 0.9,
        bidirectional: true,
        maxDistance: 1 // Direct relationship only
      },

      ESTIMATE_TO_CONSUMED: {
        type: 'ESTIMATE_TO_CONSUMED', 
        description: 'Links estimate items to actual consumed costs',
        strength: 0.95,
        bidirectional: true,
        maxDistance: 1
      },

      SAME_CATEGORY: {
        type: 'SAME_CATEGORY',
        description: 'Items in the same category (Material, Labor, etc.)',
        strength: 0.6,
        bidirectional: true,
        maxDistance: 2
      },

      SAME_SCOPE: {
        type: 'SAME_SCOPE',
        description: 'Items under the same task scope',
        strength: 0.7,
        bidirectional: true,
        maxDistance: 2
      },

      HIGH_VARIANCE: {
        type: 'HIGH_VARIANCE',
        description: 'Items with significant budget variance',
        strength: 0.8,
        bidirectional: false,
        threshold: 0.2, // 20% variance threshold
        maxDistance: 1
      },

      COST_SIMILARITY: {
        type: 'COST_SIMILARITY',
        description: 'Items with similar cost amounts',
        strength: 0.5,
        bidirectional: true,
        threshold: 0.1, // 10% difference threshold
        maxDistance: 2
      },

      AREA_GROUPING: {
        type: 'AREA_GROUPING', 
        description: 'Items in the same work area',
        strength: 0.6,
        bidirectional: true,
        maxDistance: 2
      }
    };

    // Batch processing configuration
    this.batchSize = 100;
    this.maxRelationshipsPerEntity = 20;
    this.processingTimeout = 300000; // 5 minutes
  }

  // Main method to build relationships for entities
  async buildRelationships(entities, db) {
    if (!entities || entities.length === 0) {
      console.log('⚠️  No entities provided for relationship building');
      return [];
    }

    console.log(`🔗 Building relationships for ${entities.length} entities...`);
    const startTime = Date.now();

    try {
      const relationships = [];
      
      // Build relationships by type
      for (const [typeName, config] of Object.entries(this.relationshipTypes)) {
        console.log(`🔍 Processing ${typeName} relationships...`);
        
        const typeRelationships = await this.buildRelationshipsByType(
          entities, 
          typeName, 
          config, 
          db
        );
        
        relationships.push(...typeRelationships);
        console.log(`✅ Created ${typeRelationships.length} ${typeName} relationships`);
      }

      // Store relationships in database
      console.log(`💾 Storing ${relationships.length} relationships...`);
      const storedRelationships = await this.storeRelationships(relationships, db);

      const duration = Date.now() - startTime;
      console.log(`🎉 Relationship building completed in ${duration}ms`);
      console.log(`📊 Total relationships created: ${storedRelationships.length}`);

      return storedRelationships;

    } catch (error) {
      console.error('❌ Error building relationships:', error.message);
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
        
      case 'ESTIMATE_TO_CONSUMED':
        relationships.push(...this.findEstimateToConsumedRelationships(entities, config));
        break;
        
      case 'SAME_CATEGORY':
        relationships.push(...this.findSameCategoryRelationships(entities, config));
        break;
        
      case 'SAME_SCOPE':
        relationships.push(...this.findSameScopeRelationships(entities, config));
        break;
        
      case 'HIGH_VARIANCE':
        relationships.push(...this.findHighVarianceRelationships(entities, config));
        break;
        
      case 'COST_SIMILARITY':
        relationships.push(...this.findCostSimilarityRelationships(entities, config));
        break;
        
      case 'AREA_GROUPING':
        relationships.push(...this.findAreaGroupingRelationships(entities, config));
        break;
        
      default:
        console.warn(`⚠️  Unknown relationship type: ${typeName}`);
    }
    
    return relationships;
  }

  // Find entities with the same cost code
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
          
          relationships.push({
            sourceId: sourceEntity.id,
            targetId: targetEntity.id,
            type: config.type,
            strength: config.strength,
            metadata: { costCode, sharedAttribute: 'cost_code' }
          });
        }
      }
    });
    
    return relationships;
  }

  // Find relationships between estimate and consumed items
  findEstimateToConsumedRelationships(entities, config) {
    const relationships = [];
    const estimates = entities.filter(e => e.entity_type === 'estimate_row' || e.entityType === 'estimate_row');
    const consumed = entities.filter(e => e.entity_type === 'consumed_row' || e.entityType === 'consumed_row');
    
    estimates.forEach(estimate => {
      const estimateCostCode = estimate.cost_code || estimate.costCode;
      if (!estimateCostCode) return;
      
      consumed.forEach(consumedItem => {
        const consumedCostCode = consumedItem.cost_code || consumedItem.costCode;
        
        if (estimateCostCode === consumedCostCode && 
            estimate.project_id === consumedItem.project_id) {
          
          // Calculate strength based on cost similarity
          const estimateAmount = estimate.total_amount || estimate.totalAmount || 0;
          const consumedAmount = consumedItem.total_amount || consumedItem.totalAmount || 0;
          
          let adjustedStrength = config.strength;
          if (estimateAmount > 0 && consumedAmount > 0) {
            const ratio = Math.min(estimateAmount, consumedAmount) / Math.max(estimateAmount, consumedAmount);
            adjustedStrength = config.strength * (0.5 + 0.5 * ratio); // Boost strength for similar amounts
          }
          
          relationships.push({
            sourceId: estimate.id,
            targetId: consumedItem.id,
            type: config.type,
            strength: adjustedStrength,
            metadata: {
              costCode: estimateCostCode,
              estimateAmount,
              consumedAmount,
              variance: consumedAmount - estimateAmount
            }
          });
        }
      });
    });
    
    return relationships;
  }

  // Find entities in the same category
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
            metadata: { category, sharedAttribute: 'category' }
          });
          
          connections++;
        }
      });
    });
    
    return relationships;
  }

  // Find entities with the same task scope
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
      
      const maxConnectionsPerEntity = Math.min(3, groupEntities.length - 1);
      
      groupEntities.forEach((entity, index) => {
        let connections = 0;
        
        for (let i = index + 1; i < groupEntities.length && connections < maxConnectionsPerEntity; i++) {
          const targetEntity = groupEntities[i];
          
          relationships.push({
            sourceId: entity.id,
            targetId: targetEntity.id,
            type: config.type,
            strength: config.strength,
            metadata: { taskScope: scope, sharedAttribute: 'task_scope' }
          });
          
          connections++;
        }
      });
    });
    
    return relationships;
  }

  // Find high variance relationships (estimate vs actual)
  findHighVarianceRelationships(entities, config) {
    const relationships = [];
    
    entities.forEach(entity => {
      const entityType = entity.entity_type || entity.entityType;
      if (entityType !== 'estimate_row') return;
      
      const total = entity.total_amount || entity.totalAmount || 0;
      const budgeted = entity.budgeted_amount || entity.budgetedAmount || 0;
      
      if (budgeted <= 0) return;
      
      const variance = Math.abs(total - budgeted) / budgeted;
      
      if (variance > config.threshold) {
        // Find other high variance items in the same project
        const highVarianceItems = entities.filter(other => {
          if (other.id === entity.id) return false;
          
          const otherType = other.entity_type || other.entityType;
          if (otherType !== 'estimate_row') return false;
          
          const otherTotal = other.total_amount || other.totalAmount || 0;
          const otherBudgeted = other.budgeted_amount || other.budgetedAmount || 0;
          
          if (otherBudgeted <= 0) return false;
          
          const otherVariance = Math.abs(otherTotal - otherBudgeted) / otherBudgeted;
          return otherVariance > config.threshold &&
                 entity.project_id === other.project_id;
        });
        
        highVarianceItems.forEach(targetEntity => {
          relationships.push({
            sourceId: entity.id,
            targetId: targetEntity.id,
            type: config.type,
            strength: config.strength,
            metadata: {
              sourceVariance: variance,
              targetVariance: Math.abs((targetEntity.total_amount || 0) - (targetEntity.budgeted_amount || 0)) / (targetEntity.budgeted_amount || 1),
              threshold: config.threshold
            }
          });
        });
      }
    });
    
    return relationships;
  }

  // Find entities with similar costs
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
              sharedAttribute: 'similar_cost'
            }
          });
        }
      }
    }
    
    return relationships;
  }

  // Find entities in the same work area
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
            metadata: { area, sharedAttribute: 'area' }
          });
          
          connections++;
        }
      });
    });
    
    return relationships;
  }

  // Store relationships in database with deduplication
  async storeRelationships(relationships, db) {
    if (!relationships || relationships.length === 0) {
      return [];
    }

    console.log(`💾 Storing ${relationships.length} relationships...`);
    
    try {
      // Deduplicate relationships
      const uniqueRelationships = this.deduplicateRelationships(relationships);
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
      
      console.log(`✅ Successfully stored ${storedRelationships.length} relationships`);
      return storedRelationships;
      
    } catch (error) {
      console.error('❌ Error storing relationships:', error.message);
      throw error;
    }
  }

  // Remove duplicate relationships
  deduplicateRelationships(relationships) {
    const seen = new Set();
    const unique = [];
    
    relationships.forEach(rel => {
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

  // Get relationship statistics
  async getRelationshipStats(db) {
    try {
      const stats = await db.query(`
        SELECT 
          type,
          COUNT(*) as count,
          AVG(strength) as avg_strength,
          MIN(strength) as min_strength,
          MAX(strength) as max_strength
        FROM relationships
        GROUP BY type
        ORDER BY count DESC
      `);
      
      const totalStats = await db.query(`
        SELECT 
          COUNT(*) as total_relationships,
          COUNT(DISTINCT source_id) as unique_sources,
          COUNT(DISTINCT target_id) as unique_targets,
          AVG(strength) as overall_avg_strength
        FROM relationships
      `);
      
      return {
        byType: stats.rows,
        overall: totalStats.rows[0],
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      console.error('❌ Error getting relationship stats:', error.message);
      return { error: error.message };
    }
  }

  // Find entities that need relationship building
  async findEntitiesWithoutRelationships(db) {
    try {
      const result = await db.query(`
        SELECT e.*
        FROM entities e
        LEFT JOIN relationships r ON e.id = r.source_id OR e.id = r.target_id
        WHERE r.id IS NULL
        ORDER BY e.created_at DESC
      `);
      
      return result.rows;
      
    } catch (error) {
      console.error('❌ Error finding entities without relationships:', error.message);
      throw error;
    }
  }

  // Clean up weak or redundant relationships
  async cleanupRelationships(db, minStrength = 0.3) {
    try {
      console.log(`🧹 Cleaning up relationships with strength < ${minStrength}...`);
      
      const result = await db.query(`
        DELETE FROM relationships 
        WHERE strength < $1
      `, [minStrength]);
      
      console.log(`✅ Removed ${result.rowCount} weak relationships`);
      return result.rowCount;
      
    } catch (error) {
      console.error('❌ Error cleaning up relationships:', error.message);
      throw error;
    }
  }
}

// Export singleton instance
const relationshipBuilder = new RelationshipBuilder();
export default relationshipBuilder;