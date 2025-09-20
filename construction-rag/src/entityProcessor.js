/**
 * Estimate Entity Processor
 * Converts EstimateRow data into optimized entities for RAG
 * FIXED: Properly handles database UUID project IDs
 */

import dotenv from 'dotenv';

dotenv.config();

class EstimateEntityProcessor {
  constructor() {
    // Enhanced category detection patterns for better accuracy
    this.categoryPatterns = {
      material: {
        costCodePatterns: [/\d+M$/i, /MATERIAL/i, /SUPPLY/i, /EQUIPMENT/i, /MAT$/i],
        descriptionPatterns: [/material/i, /supply/i, /equipment/i, /lumber/i, /concrete/i, /steel/i, /pipe/i, /wire/i, /fixture/i, /hardware/i, /fastener/i, /insulation/i, /drywall/i, /flooring/i, /roofing/i, /paint/i, /door/i, /window/i, /cabinet/i]
      },
      labor: {
        costCodePatterns: [/\d+L$/i, /LABOR/i, /WORK/i, /LAB$/i],
        descriptionPatterns: [/labor/i, /work/i, /install/i, /crew/i, /worker/i, /hour/i, /hr/i, /manhour/i, /foreman/i, /carpenter/i, /electrician/i, /plumber/i, /demo/i, /demolition/i, /framing/i, /finish/i]
      },
      subcontractor: {
        costCodePatterns: [/\d+S$/i, /SUBCONTRACTOR/i, /SUB/i, /CONTRACTOR/i, /SPEC/i],
        descriptionPatterns: [/subcontractor/i, /sub /i, /contractor/i, /specialty/i, /trade/i, /hvac/i, /electrical/i, /plumbing/i, /mechanical/i, /consultant/i, /engineer/i]
      },
      overhead: {
        costCodePatterns: [/\d+O$/i, /OVERHEAD/i, /MANAGEMENT/i, /ADMIN/i, /OH$/i],
        descriptionPatterns: [/overhead/i, /management/i, /admin/i, /supervision/i, /permit/i, /fee/i, /insurance/i, /bond/i, /temp/i, /temporary/i, /cleanup/i, /safety/i, /office/i]
      },
      equipment: {
        costCodePatterns: [/\d+E$/i, /EQUIPMENT/i, /RENTAL/i, /TOOL/i],
        descriptionPatterns: [/rental/i, /equipment/i, /tool/i, /crane/i, /excavator/i, /bulldozer/i, /truck/i, /scaffold/i, /generator/i, /compressor/i]
      }
    };
  }

  /**
   * Smart category detection from cost codes and descriptions
   */
  detectCategory(costCode, description) {
    console.log(`🔍 DEBUG: Detecting category for costCode="${costCode}", description="${description?.substring(0, 30)}..."`);
    
    const checkText = `${costCode || ''} ${description || ''}`.toLowerCase();
    
    // Check cost code patterns first (more reliable)
    for (const [category, patterns] of Object.entries(this.categoryPatterns)) {
      const costCodeMatch = patterns.costCodePatterns.some(pattern => 
        pattern.test(costCode || '')
      );
      
      if (costCodeMatch) {
        console.log(`✅ DEBUG: Category detected via cost code pattern: "${category}"`);
        return category;
      }
    }
    
    // Check description patterns with scoring
    const categoryScores = {};
    for (const [category, patterns] of Object.entries(this.categoryPatterns)) {
      categoryScores[category] = 0;
      
      patterns.descriptionPatterns.forEach(pattern => {
        const matches = (checkText.match(pattern) || []).length;
        categoryScores[category] += matches;
      });
    }
    
    // Return category with highest score
    const bestCategory = Object.entries(categoryScores)
      .reduce((a, b) => categoryScores[a[0]] > categoryScores[b[0]] ? a : b);
    
    const detectedCategory = bestCategory[1] > 0 ? bestCategory[0] : 'other';
    console.log(`📊 DEBUG: Category scores:`, categoryScores, `-> Selected: "${detectedCategory}"`);
    
    return detectedCategory;
  }

  /**
   * Process single EstimateRow into optimized entity for RAG
   * FIXED: Now accepts projectId parameter
   */
  processEstimateRow(estimateRow, jobData, projectId = null) {
    console.log(`🔄 DEBUG: Processing estimate row with projectId: ${projectId}`);
    console.log(`📊 DEBUG: Row data:`, {
      hasCostCode: !!estimateRow.costCode,
      hasDescription: !!estimateRow.description,
      hasTotal: estimateRow.total !== undefined,
      hasAmount: estimateRow.amount !== undefined,
      keys: Object.keys(estimateRow)
    });

    try {
      // Extract and validate fields
      const costCode = (estimateRow.costCode || '').toString().trim();
      const description = (estimateRow.description || '').toString().trim();
      const taskScope = (estimateRow.taskScope || '').toString().trim();
      const area = (estimateRow.area || '').toString().trim();
      
      console.log(`📋 DEBUG: Extracted basic fields:`, {
        costCode: costCode || 'EMPTY',
        description: description?.substring(0, 50) || 'EMPTY',
        taskScope: taskScope || 'EMPTY',
        area: area || 'EMPTY'
      });
      
      // Parse numeric values safely
      const qty = this.parseNumber(estimateRow.qty);
      const rate = this.parseNumber(estimateRow.rate);
      const total = this.parseNumber(estimateRow.total);
      const budgetedRate = this.parseNumber(estimateRow.budgetedRate);
      const budgetedTotal = this.parseNumber(estimateRow.budgetedTotal);
      
      console.log(`💰 DEBUG: Parsed numeric values:`, {
        qty, rate, total, budgetedRate, budgetedTotal
      });
      
      const units = (estimateRow.units || 'ea').toString().trim();
      const notes = (estimateRow.notesRemarks || '').toString().trim();
      const rowType = (estimateRow.rowType || 'estimate').toString().trim();
      
      // Handle materials array
      const materials = Array.isArray(estimateRow.materials) ? estimateRow.materials : [];
      
      // Calculate variance
      const variance = total - budgetedTotal;
      const variancePercent = budgetedTotal > 0 ? ((variance / budgetedTotal) * 100) : 0;
      
      console.log(`📊 DEBUG: Calculated values:`, {
        variance, variancePercent, materialsCount: materials.length
      });
      
      // Smart category detection
      const category = this.detectCategory(costCode, description);
      
      // Generate optimized content for embeddings
      console.log(`🔮 DEBUG: Generating content for embeddings...`);
      const content = this.generateOptimizedEstimateContent({
        costCode,
        description,
        taskScope,
        area,
        qty,
        rate,
        total,
        budgetedRate,
        budgetedTotal,
        variance,
        variancePercent,
        units,
        materials,
        notes,
        rowType,
        category,
        jobData
      });

      console.log(`✅ DEBUG: Generated content length: ${content.length} characters`);

      const entity = {
        // FIXED: Use the database project UUID instead of Firebase ID
        projectId: projectId,  // This is now the database UUID
        entityType: 'estimate_row',
        costCode,
        description: this.createDisplayDescription(costCode, description),
        taskScope,
        category,
        totalAmount: total,
        budgetedAmount: budgetedTotal,
        rawData: estimateRow,
        content: content,
        
        // Additional searchable fields
        area,
        units,
        qty,
        rate,
        budgetedRate,
        variance,
        variancePercent,
        rowType,
        materials: materials.length > 0 ? materials : null,
        notes: notes || null
      };

      console.log(`🎯 DEBUG: Created entity:`, {
        projectId: entity.projectId,
        projectIdType: typeof entity.projectId,
        isUUID: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(entity.projectId),
        costCode: entity.costCode,
        category: entity.category,
        totalAmount: entity.totalAmount,
        hasContent: !!entity.content,
        contentLength: entity.content?.length
      });

      return entity;

    } catch (error) {
      console.error('❌ DEBUG: Error processing estimate row:', error.message);
      console.error('❌ DEBUG: Raw estimate data:', JSON.stringify(estimateRow, null, 2));
      throw new Error(`Failed to process estimate row: ${error.message}`);
    }
  }

  /**
   * Parse number values safely
   */
  parseNumber(value) {
    console.log(`🔢 DEBUG: Parsing number from:`, { value, type: typeof value });
    
    if (typeof value === 'number') {
      const result = isNaN(value) ? 0 : value;
      console.log(`📊 DEBUG: Number result: ${result}`);
      return result;
    }
    
    if (typeof value === 'string') {
      const parsed = parseFloat(value.replace(/[^\d.-]/g, ''));
      const result = isNaN(parsed) ? 0 : parsed;
      console.log(`📊 DEBUG: String->Number result: "${value}" -> ${result}`);
      return result;
    }
    
    console.log(`📊 DEBUG: Default number result: 0 (from ${typeof value})`);
    return 0;
  }

  /**
   * Create display description combining cost code and description
   */
  createDisplayDescription(costCode, description) {
    if (!costCode && !description) return 'Unspecified Item';
    if (!costCode) return description;
    if (!description) return costCode;
    return `${costCode} - ${description}`;
  }

  /**
   * Generate optimized content for embeddings - focused on estimate data
   */
  generateOptimizedEstimateContent(data) {
    console.log(`🔮 DEBUG: Generating optimized content with data keys:`, Object.keys(data));
    
    const {
      costCode, description, taskScope, area, qty, rate, total, budgetedRate, 
      budgetedTotal, variance, variancePercent, units, materials, notes, 
      rowType, category, jobData
    } = data;

    // Build materials text
    const materialsText = materials.length > 0 
      ? `Materials: ${materials.map(m => typeof m === 'object' ? (m.name || m.description || 'Unknown') : m).join(', ')}`
      : '';

    // Build variance analysis
    let varianceAnalysis = '';
    if (budgetedTotal > 0) {
      if (Math.abs(variancePercent) < 5) {
        varianceAnalysis = `On budget (${variancePercent.toFixed(1)}% variance)`;
      } else if (variance > 0) {
        varianceAnalysis = `Over budget by $${Math.abs(variance).toFixed(2)} (${variancePercent.toFixed(1)}% over)`;
      } else {
        varianceAnalysis = `Under budget by $${Math.abs(variance).toFixed(2)} (${Math.abs(variancePercent).toFixed(1)}% under)`;
      }
    }

    // Build work scope context
    const scopeContext = taskScope || area ? 
      `Work Area: ${area || 'General'} | Scope: ${taskScope || 'General work'}` : 
      'General construction work';

    // Category-specific insights
    const categoryInsights = {
      material: `Material procurement item requiring ${qty} ${units}. Unit cost: $${rate.toFixed(2)} per ${units}.`,
      labor: `Labor work estimated for ${qty} ${units} at $${rate.toFixed(2)} per ${units}.`,
      subcontractor: `Subcontractor work valued at $${total.toFixed(2)}. ${qty > 0 ? `Quantity: ${qty} ${units}` : ''}`,
      equipment: `Equipment rental/purchase for ${qty} ${units} at $${rate.toFixed(2)} per ${units}.`,
      overhead: `Project overhead cost of $${total.toFixed(2)}.`,
      other: `Construction item for ${qty} ${units}.`
    };

    const content = `
CONSTRUCTION ESTIMATE: ${jobData.projectTitle || 'Unknown Project'}
Client: ${jobData.clientName || 'Unknown Client'}
Location: ${jobData.siteCity ? `${jobData.siteCity}, ${jobData.siteState || ''}` : 'Unknown Location'}

COST CODE: ${costCode}
DESCRIPTION: ${description}
CATEGORY: ${category.toUpperCase()}
TYPE: ${rowType}

${scopeContext}

COST BREAKDOWN:
- Quantity: ${qty} ${units}
- Unit Rate: $${rate.toFixed(2)} per ${units}
- Total Cost: $${total.toFixed(2)}
- Budgeted Rate: $${budgetedRate.toFixed(2)} per ${units}
- Budgeted Total: $${budgetedTotal.toFixed(2)}
- Budget Status: ${varianceAnalysis}

WORK DETAILS:
${categoryInsights[category] || categoryInsights.other}
${materialsText}
${notes ? `Additional Notes: ${notes}` : ''}

COST ANALYSIS:
This ${category} item ${variance > 0 ? 'exceeds' : variance < 0 ? 'is under' : 'meets'} the budgeted amount${budgetedTotal > 0 ? ` by $${Math.abs(variance).toFixed(2)}` : ''}. 
${total > 10000 ? 'HIGH VALUE ITEM - ' : ''}${qty > 0 && rate > 0 ? `Unit economics: $${rate.toFixed(2)}/${units}` : ''}
    `.trim();

    console.log(`✅ DEBUG: Generated content preview:`, content.substring(0, 200) + '...');
    return content;
  }

  /**
   * Process complete job estimates from Firebase
   * FIXED: Now accepts projectId parameter for database UUID
   */
  processJobEstimates(jobData, projectId = null) {
    console.log(`🚀 DEBUG: === STARTING ESTIMATE PROCESSING ===`);
    console.log(`📊 DEBUG: Input parameters:`, {
      hasJobData: !!jobData,
      projectId: projectId,
      projectIdType: typeof projectId,
      isUUID: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(projectId),
      jobDataKeys: jobData ? Object.keys(jobData) : 'null',
      hasEstimates: !!(jobData?.estimates),
      estimatesLength: jobData?.estimates?.length,
      firebaseJobId: jobData?.jobId || jobData?.documentId,
      projectTitle: jobData?.projectTitle || jobData?.jobTitle
    });

    if (!jobData) {
      console.log('❌ DEBUG: No job data provided');
      return [];
    }

    if (!projectId) {
      console.error('❌ DEBUG: No projectId (database UUID) provided!');
      throw new Error('projectId is required for entity processing');
    }

    // Check for 'estimates' property
    if (!jobData.estimates) {
      console.log('❌ DEBUG: No estimates property found in jobData');
      console.log('❌ DEBUG: Available jobData properties:', Object.keys(jobData));
      
      // Also check if the old 'estimate' property exists for backward compatibility
      if (jobData.estimate) {
        console.log('⚠️ DEBUG: Found "estimate" (singular) property, using that instead');
        jobData.estimates = jobData.estimate; // Use the singular version
      } else {
        return [];
      }
    }

    if (!Array.isArray(jobData.estimates)) {
      console.log('❌ DEBUG: estimates property is not an array:', {
        estimatesType: typeof jobData.estimates,
        estimatesValue: jobData.estimates
      });
      return [];
    }

    if (jobData.estimates.length === 0) {
      console.log(`❌ DEBUG: Empty estimates array for job ${jobData.jobId || jobData.documentId}`);
      return [];
    }

    console.log(`📊 DEBUG: Processing ${jobData.estimates.length} estimates for "${jobData.projectTitle || jobData.jobTitle || 'Unknown Project'}"...`);

    // Debug first few estimates
    console.log(`🔍 DEBUG: Sample estimates (first 3):`);
    jobData.estimates.slice(0, 3).forEach((estimate, i) => {
      console.log(`   Estimate ${i + 1}:`, {
        costCode: estimate?.costCode,
        description: estimate?.description?.substring(0, 40),
        total: estimate?.total,
        amount: estimate?.amount,
        hasRequiredData: !!(estimate?.costCode || estimate?.description),
        hasAmountData: !!(estimate?.total !== undefined || estimate?.amount !== undefined),
        allKeys: estimate ? Object.keys(estimate) : 'null'
      });
    });

    const entities = [];
    let processedCount = 0;
    let errorCount = 0;
    let skippedCount = 0;

    for (let i = 0; i < jobData.estimates.length; i++) {
      try {
        const estimateRow = jobData.estimates[i];
        
        console.log(`🔄 DEBUG: Processing estimate ${i + 1}/${jobData.estimates.length}`);
        
        // Skip empty or invalid rows
        if (!this.isValidEstimateRow(estimateRow)) {
          console.log(`⚠️ DEBUG: Skipping invalid estimate row at index ${i}:`, {
            exists: !!estimateRow,
            costCode: estimateRow?.costCode,
            description: estimateRow?.description?.substring(0, 30),
            total: estimateRow?.total,
            amount: estimateRow?.amount
          });
          skippedCount++;
          continue;
        }

        console.log(`✅ DEBUG: Valid estimate row found at index ${i}`);
        // FIXED: Pass projectId to processEstimateRow
        const entity = this.processEstimateRow(estimateRow, jobData, projectId);
        entities.push(entity);
        processedCount++;

      } catch (error) {
        console.error(`❌ DEBUG: Failed to process estimate row ${i}:`, error.message);
        console.error(`❌ DEBUG: Problematic estimate row:`, jobData.estimates[i]);
        errorCount++;
        // Continue processing other rows
      }
    }

    console.log(`🎯 DEBUG: === PROCESSING COMPLETE ===`);
    console.log(`📈 DEBUG: Final results:`);
    console.log(`   - Total estimates in input: ${jobData.estimates.length}`);
    console.log(`   - Successfully processed: ${processedCount}`);
    console.log(`   - Skipped (invalid): ${skippedCount}`);
    console.log(`   - Errors: ${errorCount}`);
    console.log(`   - Final entity count: ${entities.length}`);

    if (entities.length > 0) {
      console.log(`✅ DEBUG: Sample processed entity:`, {
        projectId: entities[0].projectId,
        projectIdType: typeof entities[0].projectId,
        costCode: entities[0].costCode,
        category: entities[0].category,
        totalAmount: entities[0].totalAmount,
        hasContent: !!entities[0].content,
        contentLength: entities[0].content?.length
      });
    }

    return entities;
  }

  /**
   * Validate estimate row has minimum required data
   */
  isValidEstimateRow(estimateRow) {
    console.log(`🔍 DEBUG: Validating estimate row:`, {
      exists: !!estimateRow,
      type: typeof estimateRow,
      keys: estimateRow ? Object.keys(estimateRow) : 'null'
    });

    if (!estimateRow || typeof estimateRow !== 'object') {
      console.log(`❌ DEBUG: Invalid estimate row - not an object`);
      return false;
    }
    
    // Must have at least a cost code or description
    const hasCostCode = estimateRow.costCode && estimateRow.costCode.toString().trim().length > 0;
    const hasDescription = estimateRow.description && estimateRow.description.toString().trim().length > 0;
    
    console.log(`🔍 DEBUG: Validation details:`, {
      hasCostCode,
      hasDescription,
      costCode: estimateRow.costCode,
      description: estimateRow.description?.substring(0, 30)
    });
    
    const isValid = hasCostCode || hasDescription;
    console.log(`${isValid ? '✅' : '❌'} DEBUG: Estimate row validation result: ${isValid}`);
    
    return isValid;
  }

  /**
   * Analyze processed entities for insights
   */
  analyzeEstimates(entities) {
    console.log(`📊 DEBUG: Analyzing ${entities?.length || 0} entities...`);

    if (!entities || entities.length === 0) {
      console.log(`⚠️ DEBUG: No entities to analyze`);
      return {
        totalEntities: 0,
        categories: {},
        costCodes: {},
        totals: { estimated: 0, budgeted: 0, variance: 0 }
      };
    }

    const analysis = {
      totalEntities: entities.length,
      categories: {},
      costCodes: {},
      areas: {},
      totals: {
        estimated: 0,
        budgeted: 0,
        variance: 0
      },
      budgetHealth: {
        onBudget: 0,
        overBudget: 0,
        underBudget: 0
      },
      highValueItems: [],
      varianceItems: []
    };

    entities.forEach((entity, index) => {
      console.log(`📋 DEBUG: Analyzing entity ${index + 1}:`, {
        costCode: entity.costCode,
        category: entity.category,
        totalAmount: entity.totalAmount,
        budgetedAmount: entity.budgetedAmount
      });

      // Category analysis
      analysis.categories[entity.category] = (analysis.categories[entity.category] || 0) + 1;
      
      // Cost code analysis
      if (entity.costCode) {
        analysis.costCodes[entity.costCode] = (analysis.costCodes[entity.costCode] || 0) + 1;
      }
      
      // Area analysis
      if (entity.area) {
        analysis.areas[entity.area] = (analysis.areas[entity.area] || 0) + 1;
      }
      
      // Financial totals
      analysis.totals.estimated += entity.totalAmount || 0;
      analysis.totals.budgeted += entity.budgetedAmount || 0;
      
      // Budget health
      const variance = (entity.totalAmount || 0) - (entity.budgetedAmount || 0);
      if (Math.abs(variance) < 100) {
        analysis.budgetHealth.onBudget++;
      } else if (variance > 0) {
        analysis.budgetHealth.overBudget++;
      } else {
        analysis.budgetHealth.underBudget++;
      }
      
      // High value items (>$5000)
      if ((entity.totalAmount || 0) > 5000) {
        analysis.highValueItems.push({
          costCode: entity.costCode,
          description: entity.description,
          amount: entity.totalAmount,
          category: entity.category
        });
      }
      
      // High variance items (>20% or >$1000)
      if (entity.budgetedAmount > 0) {
        const variancePercent = Math.abs(variance / entity.budgetedAmount) * 100;
        if (variancePercent > 20 || Math.abs(variance) > 1000) {
          analysis.varianceItems.push({
            costCode: entity.costCode,
            description: entity.description,
            variance: variance,
            variancePercent: variancePercent,
            estimated: entity.totalAmount,
            budgeted: entity.budgetedAmount
          });
        }
      }
    });

    // Calculate total variance
    analysis.totals.variance = analysis.totals.estimated - analysis.totals.budgeted;
    analysis.totals.variancePercent = analysis.totals.budgeted > 0 
      ? (analysis.totals.variance / analysis.totals.budgeted) * 100 
      : 0;

    // Sort arrays by value
    analysis.highValueItems.sort((a, b) => (b.amount || 0) - (a.amount || 0));
    analysis.varianceItems.sort((a, b) => Math.abs(b.variance) - Math.abs(a.variance));

    console.log(`✅ DEBUG: Analysis complete:`, {
      totalEntities: analysis.totalEntities,
      categoriesCount: Object.keys(analysis.categories).length,
      totalEstimated: analysis.totals.estimated,
      totalBudgeted: analysis.totals.budgeted,
      totalVariance: analysis.totals.variance,
      highValueItemsCount: analysis.highValueItems.length,
      varianceItemsCount: analysis.varianceItems.length
    });

    return analysis;
  }

  /**
   * Validate entity data before processing
   */
  validateEstimateEntity(entity) {
    const required = ['costCode', 'totalAmount'];
    const missing = required.filter(field => {
      const value = entity[field];
      return value === undefined || value === null || 
             (typeof value === 'string' && value.trim() === '') ||
             (typeof value === 'number' && isNaN(value));
    });
    
    return {
      valid: missing.length === 0,
      missing,
      warnings: this.getValidationWarnings(entity)
    };
  }

  /**
   * Get validation warnings for estimate entity
   */
  getValidationWarnings(entity) {
    const warnings = [];
    
    if (!entity.description || entity.description.trim().length < 5) {
      warnings.push('Description is very short or missing');
    }
    
    if (entity.totalAmount <= 0) {
      warnings.push('Total amount is zero or negative');
    }
    
    if (entity.qty <= 0) {
      warnings.push('Quantity is zero or negative');
    }
    
    if (entity.rate <= 0 && entity.totalAmount > 0) {
      warnings.push('Rate is zero but total amount is positive');
    }
    
    if (entity.budgetedAmount > 0) {
      const variance = Math.abs(entity.totalAmount - entity.budgetedAmount);
      const variancePercent = (variance / entity.budgetedAmount) * 100;
      
      if (variancePercent > 50) {
        warnings.push(`High budget variance: ${variancePercent.toFixed(1)}%`);
      }
    }
    
    return warnings;
  }
}

// Export singleton instance
const estimateEntityProcessor = new EstimateEntityProcessor();
export default estimateEntityProcessor;