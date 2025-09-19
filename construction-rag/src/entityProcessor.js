/**
 * Entity Processor
 * Converts EstimateRow and ConsumedRow data into rich graph entities
 * Zero hardcoding - all driven by smart detection and templates
 */

import dotenv from 'dotenv';

dotenv.config();

class EntityProcessor {
  constructor() {
    // Category detection patterns (smart, not hardcoded!)
    this.categoryPatterns = {
      material: {
        costCodePatterns: [/\d+M$/i, /MATERIAL/i, /SUPPLY/i, /EQUIPMENT/i],
        descriptionPatterns: [/material/i, /supply/i, /equipment/i, /lumber/i, /concrete/i, /steel/i, /pipe/i, /wire/i, /fixture/i]
      },
      labor: {
        costCodePatterns: [/\d+L$/i, /LABOR/i, /WORK/i],
        descriptionPatterns: [/labor/i, /work/i, /install/i, /crew/i, /worker/i, /hour/i, /hr/i, /manhour/i]
      },
      subcontractor: {
        costCodePatterns: [/\d+S$/i, /SUBCONTRACTOR/i, /SUB/i, /CONTRACTOR/i],
        descriptionPatterns: [/subcontractor/i, /sub /i, /contractor/i, /specialty/i, /trade/i]
      },
      overhead: {
        costCodePatterns: [/\d+O$/i, /OVERHEAD/i, /MANAGEMENT/i, /ADMIN/i],
        descriptionPatterns: [/overhead/i, /management/i, /admin/i, /supervision/i, /permit/i, /fee/i, /insurance/i]
      }
    };
  }

  // Smart category detection from cost codes and descriptions
  detectCategory(costCode, description) {
    const checkText = `${costCode || ''} ${description || ''}`.toLowerCase();
    
    for (const [category, patterns] of Object.entries(this.categoryPatterns)) {
      // Check cost code patterns first (more reliable)
      const costCodeMatch = patterns.costCodePatterns.some(pattern => 
        pattern.test(costCode || '')
      );
      
      if (costCodeMatch) return category;
      
      // Check description patterns
      const descriptionMatch = patterns.descriptionPatterns.some(pattern => 
        pattern.test(checkText)
      );
      
      if (descriptionMatch) return category;
    }
    
    return 'other'; // Default fallback
  }

  // Process single EstimateRow into rich entity
  processEstimateRow(estimateRow, projectId, jobMetadata = {}) {
    try {
      // Extract and normalize fields
      const costCode = estimateRow.costCode || 'UNKNOWN';
      const description = estimateRow.description || '';
      const taskScope = estimateRow.taskScope || '';
      const area = estimateRow.area || '';
      const qty = parseFloat(estimateRow.qty) || 0;
      const rate = parseFloat(estimateRow.rate) || 0;
      const total = parseFloat(estimateRow.total) || 0;
      const budgeted = parseFloat(estimateRow.budgetedTotal) || 0;
      const variance = total - budgeted;
      const variancePercent = budgeted > 0 ? ((variance / budgeted) * 100) : 0;
      const units = estimateRow.units || 'ea';
      const materials = Array.isArray(estimateRow.materials) ? estimateRow.materials : [];
      const notes = estimateRow.notesRemarks || '';
      const rowType = estimateRow.rowType || 'estimate';

      // Smart category detection
      const category = this.detectCategory(costCode, description);

      // Generate rich content for embedding
      const content = this.generateEstimateContent({
        costCode,
        description,
        taskScope,
        area,
        qty,
        rate,
        total,
        budgeted,
        variance,
        variancePercent,
        units,
        materials,
        notes,
        rowType,
        category,
        jobTitle: jobMetadata.jobTitle || 'Unknown Project',
        clientName: jobMetadata.clientName || 'Unknown Client'
      });

      return {
        projectId,
        entityType: 'estimate_row',
        costCode,
        description: `${costCode} - ${description}`.trim(),
        taskScope,
        category,
        totalAmount: total,
        budgetedAmount: budgeted,
        rawData: estimateRow,
        content: content,
        // Additional fields for relationships
        area,
        variance,
        variancePercent,
        rowType,
        units,
        qty,
        rate
      };

    } catch (error) {
      console.error('❌ Error processing estimate row:', error.message);
      console.error('Raw data:', estimateRow);
      throw error;
    }
  }

  // Process single ConsumedRow into rich entity  
  processConsumedRow(consumedRow, projectId, jobMetadata = {}) {
    try {
      // Extract and normalize fields
      const costCode = consumedRow.costCode || 'UNKNOWN';
      const job = consumedRow.job || '';
      const amount = parseFloat(consumedRow.amount) || 0;
      const date = consumedRow.date || '';

      // Smart category detection
      const category = this.detectCategory(costCode, job);

      // Generate rich content for embedding
      const content = this.generateConsumedContent({
        costCode,
        job,
        amount,
        date,
        category,
        jobTitle: jobMetadata.jobTitle || 'Unknown Project',
        clientName: jobMetadata.clientName || 'Unknown Client'
      });

      return {
        projectId,
        entityType: 'consumed_row',
        costCode,
        description: `${costCode} - ${job}`.trim(),
        taskScope: null, // Consumed data doesn't have task scope
        category,
        totalAmount: amount,
        budgetedAmount: null, // Consumed data doesn't have budgeted amount
        rawData: consumedRow,
        content: content,
        // Additional fields
        date
      };

    } catch (error) {
      console.error('❌ Error processing consumed row:', error.message);
      console.error('Raw data:', consumedRow);
      throw error;
    }
  }

  // Generate rich content for EstimateRow embedding
  generateEstimateContent(data) {
    const {
      costCode, description, taskScope, area, qty, rate, total, budgeted, 
      variance, variancePercent, units, materials, notes, rowType, category,
      jobTitle, clientName
    } = data;

    // Create materials text
    const materialsText = materials.length > 0 
      ? materials.map(m => typeof m === 'object' ? m.name || m.description : m).join(', ')
      : 'No materials specified';

    // Variance analysis text
    const varianceText = variance > 0 
      ? `$${Math.abs(variance).toFixed(2)} over budget (${variancePercent.toFixed(1)}%)`
      : variance < 0 
        ? `$${Math.abs(variance).toFixed(2)} under budget (${Math.abs(variancePercent).toFixed(1)}%)`
        : 'On budget';

    // Category-specific insights
    let categoryInsight = '';
    switch (category) {
      case 'material':
        categoryInsight = `Material cost item requiring ${qty} ${units} at $${rate} per unit.`;
        break;
      case 'labor':
        categoryInsight = `Labor cost for ${qty} ${units} of work at $${rate} per unit.`;
        break;
      case 'subcontractor':
        categoryInsight = `Subcontractor work estimated at $${total}.`;
        break;
      default:
        categoryInsight = `${category} category item with ${qty} ${units}.`;
    }

    return `
Construction Estimate Item for ${jobTitle} (Client: ${clientName})

Cost Code: ${costCode}
Description: ${description}
Category: ${category.toUpperCase()}
${categoryInsight}

Project Details:
- Task Scope: ${taskScope}
- Work Area: ${area}
- Type: ${rowType}

Financial Summary:
- Quantity: ${qty} ${units}
- Rate: $${rate.toFixed(2)} per ${units}
- Total Estimated Cost: $${total.toFixed(2)}
- Budgeted Amount: $${budgeted.toFixed(2)}
- Budget Variance: ${varianceText}

Materials & Resources:
${materialsText}

Additional Notes:
${notes || 'No additional notes'}

This ${category} item ${variance > 0 ? 'is over budget' : variance < 0 ? 'is under budget' : 'is on budget'} and ${taskScope ? `belongs to the ${taskScope} scope` : 'has no defined scope'}.
    `.trim();
  }

  // Generate rich content for ConsumedRow embedding
  generateConsumedContent(data) {
    const { costCode, job, amount, date, category, jobTitle, clientName } = data;

    // Category-specific insights
    let categoryInsight = '';
    switch (category) {
      case 'material':
        categoryInsight = `Material expense totaling $${amount.toFixed(2)}.`;
        break;
      case 'labor':
        categoryInsight = `Labor cost of $${amount.toFixed(2)} was incurred.`;
        break;
      case 'subcontractor':
        categoryInsight = `Payment to subcontractor for $${amount.toFixed(2)}.`;
        break;
      default:
        categoryInsight = `${category} expense of $${amount.toFixed(2)}.`;
    }

    return `
Actual Construction Cost for ${jobTitle} (Client: ${clientName})

Cost Code: ${costCode}
Job/Description: ${job}
Category: ${category.toUpperCase()}
${categoryInsight}

Financial Details:
- Amount Spent: $${amount.toFixed(2)}
- Date: ${date || 'Date not specified'}

This represents actual money spent on ${category} work under cost code ${costCode}. The expense was ${amount > 1000 ? 'significant' : 'minor'} and relates to ${job || 'general project work'}.
    `.trim();
  }

  // Process batch of estimates from Firebase job data
  processJobEstimates(jobData) {
    if (!jobData || !jobData.estimates || jobData.estimates.length === 0) {
      console.log(`⚠️  No estimates found for job ${jobData.jobId}`);
      return [];
    }

    const entities = [];
    const metadata = {
      jobTitle: jobData.jobTitle,
      clientName: jobData.clientName
    };

    console.log(`🔄 Processing ${jobData.estimates.length} estimates for ${jobData.jobTitle}...`);

    for (const estimate of jobData.estimates) {
      try {
        const entity = this.processEstimateRow(estimate, jobData.jobId, metadata);
        entities.push(entity);
      } catch (error) {
        console.error(`❌ Failed to process estimate row:`, error.message);
        // Continue processing other rows
      }
    }

    console.log(`✅ Processed ${entities.length}/${jobData.estimates.length} estimate entities`);
    return entities;
  }

  // Process batch of consumed data from Firebase
  processJobConsumed(consumedData, jobMetadata = {}) {
    if (!consumedData || !consumedData.entries || consumedData.entries.length === 0) {
      console.log(`⚠️  No consumed data found for job ${consumedData.jobId}`);
      return [];
    }

    const entities = [];
    console.log(`🔄 Processing ${consumedData.entries.length} consumed entries...`);

    for (const consumed of consumedData.entries) {
      try {
        const entity = this.processConsumedRow(consumed, consumedData.jobId, jobMetadata);
        entities.push(entity);
      } catch (error) {
        console.error(`❌ Failed to process consumed row:`, error.message);
        // Continue processing other rows
      }
    }

    console.log(`✅ Processed ${entities.length}/${consumedData.entries.length} consumed entities`);
    return entities;
  }

  // Process complete job data (estimates + consumed)
  processCompleteJobData(completeJobData) {
    const allEntities = [];

    // Process estimates
    const estimateEntities = this.processJobEstimates(completeJobData);
    allEntities.push(...estimateEntities);

    // Process consumed data if available
    if (completeJobData.consumedData) {
      const consumedEntities = this.processJobConsumed(
        completeJobData.consumedData,
        {
          jobTitle: completeJobData.jobTitle,
          clientName: completeJobData.clientName
        }
      );
      allEntities.push(...consumedEntities);
    }

    console.log(`✅ Total entities processed for ${completeJobData.jobTitle}: ${allEntities.length}`);
    
    return allEntities;
  }

  // Analyze entities and provide insights
  analyzeEntities(entities) {
    const analysis = {
      totalEntities: entities.length,
      estimateCount: entities.filter(e => e.entityType === 'estimate_row').length,
      consumedCount: entities.filter(e => e.entityType === 'consumed_row').length,
      categories: {},
      costCodes: {},
      totalEstimated: 0,
      totalConsumed: 0,
      totalBudgeted: 0,
      highVarianceItems: []
    };

    entities.forEach(entity => {
      // Category breakdown
      analysis.categories[entity.category] = (analysis.categories[entity.category] || 0) + 1;
      
      // Cost code breakdown
      analysis.costCodes[entity.costCode] = (analysis.costCodes[entity.costCode] || 0) + 1;
      
      // Financial totals
      if (entity.entityType === 'estimate_row') {
        analysis.totalEstimated += entity.totalAmount || 0;
        analysis.totalBudgeted += entity.budgetedAmount || 0;
        
        // High variance detection
        if (entity.variancePercent && Math.abs(entity.variancePercent) > 20) {
          analysis.highVarianceItems.push({
            costCode: entity.costCode,
            variance: entity.variancePercent,
            amount: entity.totalAmount
          });
        }
      } else if (entity.entityType === 'consumed_row') {
        analysis.totalConsumed += entity.totalAmount || 0;
      }
    });

    // Overall budget variance
    analysis.totalVariance = analysis.totalEstimated - analysis.totalBudgeted;
    analysis.totalVariancePercent = analysis.totalBudgeted > 0 
      ? (analysis.totalVariance / analysis.totalBudgeted) * 100 
      : 0;

    return analysis;
  }

  // Validate entity data before processing
  validateEstimateRow(estimateRow) {
    const required = ['costCode'];
    const missing = required.filter(field => !estimateRow[field]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }
    
    return true;
  }

  validateConsumedRow(consumedRow) {
    const required = ['costCode', 'amount'];
    const missing = required.filter(field => !consumedRow[field]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }
    
    return true;
  }
}

// Export singleton instance
const entityProcessor = new EntityProcessor();
export default entityProcessor;