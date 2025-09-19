const OpenAI = require('openai');
const embeddingService = require('./embeddingService');
const projectService = require('./projectService');
const db = require('./database');

class RAGService {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
  }

  async queryProject(question, projectId) {
    try {
      const similarContent = await embeddingService.searchSimilar(question, projectId, 5);
      
      const budgetSummary = await projectService.getBudgetSummary(projectId);
      
      const context = await this.buildContext(projectId, similarContent, budgetSummary);
      
      const response = await this.generateResponse(question, context);
      
      return {
        answer: response,
        contextUsed: similarContent.length,
        budgetSummary: budgetSummary
      };

    } catch (error) {
      console.error('Error in RAG query:', error);
      throw error;
    }
  }

  async buildContext(projectId, similarContent, budgetSummary) {
    const projectInfo = await db.query(`
      SELECT title, status, client_name, site_address 
      FROM projects WHERE id = $1
    `, [projectId]);

    const taskSummary = await db.query(`
      SELECT entity_type, COUNT(*) as count
      FROM entities 
      WHERE project_id = $1 
      GROUP BY entity_type
    `, [projectId]);

    return {
      project: projectInfo.rows[0] || {},
      entitySummary: taskSummary.rows,
      similarContent: similarContent.map(item => ({
        content: item.content,
        similarity: parseFloat(item.similarity) || 0,
        entity_type: item.entity_type
      })),
      budgetSummary: budgetSummary
    };
  }

  async generateResponse(question, context) {
    const budgetSummaryText = context.budgetSummary.map(b => {
      const variancePercentage = parseFloat(b.variance_percentage) || 0;
      const budgeted = parseFloat(b.budgeted_total) || 0;
      const actual = parseFloat(b.actual_total) || 0;
      const variance = parseFloat(b.variance) || 0;
      
      return `- ${b.task_scope}: Budgeted $${budgeted.toFixed(2)}, Actual $${actual.toFixed(2)}, Variance $${variance.toFixed(2)} (${variancePercentage.toFixed(1)}%)`;
    }).join('\n');

    const prompt = `Based on the following construction project data, provide a detailed and accurate answer to the user's question.

PROJECT INFORMATION:
- Title: ${context.project.title}
- Status: ${context.project.status}
- Client: ${context.project.client_name}
- Location: ${context.project.site_address}

ENTITY SUMMARY:
${context.entitySummary.map(e => `- ${e.entity_type}: ${e.count} items`).join('\n')}

RELEVANT CONTENT:
${context.similarContent.map((item, i) => `${i+1}. ${item.content} (Similarity: ${item.similarity.toFixed(3)})`).join('\n')}

BUDGET SUMMARY:
${budgetSummaryText}

USER QUESTION: ${question}

Instructions:
- Provide specific numbers and calculations when available
- Reference cost codes and task scopes when relevant
- Mention budget variances if the question relates to costs
- Be precise and factual
- If you don't have enough information to answer completely, say so

Answer:`;

    const response = await this.openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.1,
      max_tokens: 800,
    });

    return response.choices[0].message.content;
  }
}

module.exports = new RAGService();
