/**
 * Estimate Entity Processor
 * Converts EstimateRow data into optimized entities for RAG
 * ESTIMATES ONLY - Removed consumed data processing for performance
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
    const checkText = `${costCode || ''} ${description || ''}`.toLowerCase();
    
    // Check cost code patterns first (more reliable)
    for (const [category, patterns] of Object.entries(this.categoryPatterns)) {
      const costCodeMatch = patterns.costCodePatterns.some(pattern => 
        pattern.test(costCode || '')
      );
      
      if (costCodeMatch) return category;
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
    
    return bestCategory[1] > 0 ? bestCategory[0] : 'other';
  }

  /**
   * Process single EstimateRow into optimized entity for RAG
   */
  processEstimateRow(estimateRow, jobData) {
    try {
      // Extract and validate fields
      const costCode = (estimateRow.costCode || '').toString().trim();
      const description = (estimateRow.description || '').toString().trim();
      const taskScope = (estimateRow.taskScope || '').toString().trim();
      const area = (estimateRow.area || '').toString().trim();
      
      // Parse numeric values safely
      const qty = this.parseNumber(estimateRow.qty);
      const rate = this.parseNumber(estimateRow.rate);
      const total = this.parseNumber(estimateRow.total);
      const budgetedRate = this.parseNumber(estimateRow.budgetedRate);
      const budgetedTotal = this.parseNumber(estimateRow.budgetedTotal);
      
      const units = (estimateRow.units || 'ea').toString().trim();
      const notes = (estimateRow.notesRemarks || '').toString().trim();
      const rowType = (estimateRow.rowType || 'estimate').toString().trim();
      
      // Handle materials array
      const materials = Array.isArray(estimateRow.materials) ? estimateRow.materials : [];
      
      // Calculate variance
      const variance = total - budgetedTotal;
      const variancePercent = budgetedTotal > 0 ? ((variance / budgetedTotal) * 100) : 0;
      
      // Smart category detection
      const category = this.detectCategory(costCode, description);
      
      // Generate optimized content for embeddings
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

      return {
        projectId: jobData.jobId || jobData.documentId,
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

    } catch (error) {
      console.error('Error processing estimate row:', error.message);
      console.error('Raw estimate data:', JSON.stringify(estimateRow, null, 2));
      throw new Error(`Failed to process estimate row: ${error.message}`);
    }
  }

  /**
   * Parse number values safely
   */
  parseNumber(value) {
    if (typeof value === 'number') return isNaN(value) ? 0 : value;
    if (typeof value === 'string') {
      const parsed = parseFloat(value.replace(/[^\d.-]/g, ''));
      return isNaN(parsed) ? 0 : parsed;
    }
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

    return `
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
  }

  /**
   * Process complete job estimates from Firebase
   */
  processJobEstimates(jobData) {
    if (!jobData) {
      console.log('No job data provided');
      return [];
    }

    if (!jobData.estimate || !Array.isArray(jobData.estimate) || jobData.estimate.length === 0) {
      console.log(`No estimates found for job ${jobData.jobId || jobData.documentId}`);
      return [];
    }

    console.log(`Processing ${jobData.estimate.length} estimates for ${jobData.projectTitle || 'Unknown Project'}...`);

    const entities = [];
    let processedCount = 0;
    let errorCount = 0;

    for (let i = 0; i < jobData.estimate.length; i++) {
      try {
        const estimateRow = jobData.estimate[i];
        
        // Skip empty or invalid rows
        if (!this.isValidEstimateRow(estimateRow)) {
          console.log(`Skipping invalid estimate row at index ${i}`);
          continue;
        }

        const entity = this.processEstimateRow(estimateRow, jobData);
        entities.push(entity);
        processedCount++;

      } catch (error) {
        console.error(`Failed to process estimate row ${i}:`, error.message);
        errorCount++;
        // Continue processing other rows
      }
    }

    console.log(`Processed ${processedCount}/${jobData.estimate.length} estimates (${errorCount} errors)`);
    return entities;
  }

  /**
   * Validate estimate row has minimum required data
   */
  isValidEstimateRow(estimateRow) {
    if (!estimateRow || typeof estimateRow !== 'object') return false;
    
    // Must have at least a cost code or description
    const hasCostCode = estimateRow.costCode && estimateRow.costCode.toString().trim().length > 0;
    const hasDescription = estimateRow.description && estimateRow.description.toString().trim().length > 0;
    
    return hasCostCode || hasDescription;
  }

  /**
   * Analyze processed entities for insights
   */
  analyzeEstimates(entities) {
    if (!entities || entities.length === 0) {
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

    entities.forEach(entity => {
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