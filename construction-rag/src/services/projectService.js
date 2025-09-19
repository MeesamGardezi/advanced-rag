const db = require('./database');

class ProjectService {
  async processProject(projectData) {
    const client = await db.getClient();
    
    try {
      await client.query('BEGIN');

      await client.query(`
        INSERT INTO projects (id, title, status, client_name, site_address, data, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
        ON CONFLICT (id) DO UPDATE SET
          title = $2, status = $3, client_name = $4, 
          site_address = $5, data = $6, updated_at = NOW()
      `, [
        projectData.jobIndex,
        projectData.projectTitle,
        projectData.status,
        projectData.clientName,
        `${projectData.siteStreet || ''}, ${projectData.siteCity || ''}, ${projectData.siteState || ''}`.trim(),
        JSON.stringify(projectData)
      ]);

      await client.query('DELETE FROM entities WHERE project_id = $1', [projectData.jobIndex]);

      let entityCount = 0;
      if (projectData.estimate && Array.isArray(projectData.estimate)) {
        for (const item of projectData.estimate) {
          await this.createEstimateEntity(client, projectData.jobIndex, item);
          entityCount++;
        }
      }

      if (projectData.schedule && Array.isArray(projectData.schedule)) {
        for (const task of projectData.schedule) {
          await this.createTaskEntity(client, projectData.jobIndex, task);
          entityCount++;
        }
      }

      await this.updateCostAnalysis(client, projectData);

      await client.query('COMMIT');
      console.log(`Processed project ${projectData.jobIndex}: ${entityCount} entities`);
      
      return { success: true, entitiesCreated: entityCount };

    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async createEstimateEntity(client, projectId, estimateItem) {
    const result = await client.query(`
      INSERT INTO entities (project_id, entity_id, entity_type, name, properties)
      VALUES ($1, $2, 'estimate', $3, $4)
      RETURNING id
    `, [
      projectId,
      `${projectId}_estimate_${estimateItem.costCode || 'unknown'}`,
      `${estimateItem.costCode || 'Unknown'} - ${estimateItem.description || 'No description'}`,
      JSON.stringify({
        costCode: estimateItem.costCode,
        taskScope: estimateItem.taskScope,
        qty: estimateItem.qty,
        rate: estimateItem.rate,
        total: estimateItem.total,
        budgetedTotal: estimateItem.budgetedTotal,
        rowType: estimateItem.rowType
      })
    ]);
    return result.rows[0].id;
  }

  async createTaskEntity(client, projectId, taskItem) {
    const result = await client.query(`
      INSERT INTO entities (project_id, entity_id, entity_type, name, properties)
      VALUES ($1, $2, 'task', $3, $4)
      RETURNING id
    `, [
      projectId,
      taskItem.id || `${projectId}_task_${Date.now()}`,
      taskItem.task || 'Unnamed Task',
      JSON.stringify({
        taskType: taskItem.taskType,
        duration: taskItem.duration,
        hours: taskItem.hours,
        startDate: taskItem.startDate,
        endDate: taskItem.endDate,
        percentageComplete: taskItem.percentageComplete
      })
    ]);
    return result.rows[0].id;
  }

  async updateCostAnalysis(client, projectData) {
    await client.query('DELETE FROM cost_analysis WHERE project_id = $1', [projectData.jobIndex]);

    if (!projectData.estimate || !Array.isArray(projectData.estimate)) {
      return;
    }

    const scopeGroups = {};
    
    projectData.estimate.forEach(item => {
      const scope = item.taskScope || 'Unknown Scope';
      if (!scopeGroups[scope]) {
        scopeGroups[scope] = { budgeted: 0, actual: 0 };
      }
      
      scopeGroups[scope].budgeted += parseFloat(item.budgetedTotal || 0);
      scopeGroups[scope].actual += parseFloat(item.total || 0);
    });

    for (const [scope, data] of Object.entries(scopeGroups)) {
      const variance = data.actual - data.budgeted;
      const variancePercentage = data.budgeted > 0 ? (variance / data.budgeted) * 100 : 0;

      await client.query(`
        INSERT INTO cost_analysis (project_id, task_scope, budgeted_total, actual_total, variance, variance_percentage)
        VALUES ($1, $2, $3, $4, $5, $6)
      `, [projectData.jobIndex, scope, data.budgeted, data.actual, variance, variancePercentage]);
    }
  }

  async getBudgetSummary(projectId) {
    const result = await db.query(`
      SELECT task_scope, budgeted_total, actual_total, variance, variance_percentage
      FROM cost_analysis 
      WHERE project_id = $1
      ORDER BY ABS(variance) DESC
    `, [projectId]);
    return result.rows;
  }
}

module.exports = new ProjectService();
