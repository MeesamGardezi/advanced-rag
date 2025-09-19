const OpenAI = require('openai');
const db = require('./database');

class EmbeddingService {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
  }

  async generateEmbedding(text) {
    try {
      const response = await this.openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: text
      });
      return response.data[0].embedding;
    } catch (error) {
      console.error('Error generating embedding:', error.message);
      throw error;
    }
  }

  async generateProjectEmbeddings(projectId) {
    const client = await db.getClient();
    
    try {
      const entities = await client.query(`
        SELECT * FROM entities WHERE project_id = $1
      `, [projectId]);

      let embeddingCount = 0;
      
      for (const entity of entities.rows) {
        const content = this.createContentForEntity(entity);
        const embedding = await this.generateEmbedding(content);

        await client.query(`
          INSERT INTO embeddings (entity_id, content, embedding, metadata)
          VALUES ($1, $2, $3, $4)
          ON CONFLICT (entity_id) DO UPDATE SET
            content = $2, embedding = $3, metadata = $4, created_at = NOW()
        `, [
          entity.id,
          content,
          JSON.stringify(embedding),
          JSON.stringify({
            entity_type: entity.entity_type,
            entity_name: entity.name,
            project_id: entity.project_id
          })
        ]);
        
        embeddingCount++;
      }

      console.log(`Generated ${embeddingCount} embeddings for project ${projectId}`);
      return embeddingCount;

    } finally {
      client.release();
    }
  }

  createContentForEntity(entity) {
    const props = JSON.parse(entity.properties || '{}');
    
    switch (entity.entity_type) {
      case 'estimate':
        return `Estimate Item: ${entity.name}, Cost Code: ${props.costCode}, Task Scope: ${props.taskScope}, Quantity: ${props.qty}, Rate: $${props.rate}, Total: $${props.total}, Budgeted: $${props.budgetedTotal}`;
      
      case 'task':
        return `Task: ${entity.name}, Type: ${props.taskType}, Duration: ${props.duration} days, Hours: ${props.hours}, Start: ${props.startDate}, End: ${props.endDate}, Complete: ${props.percentageComplete}%`;
        
      default:
        return `${entity.entity_type}: ${entity.name} - ${JSON.stringify(props)}`;
    }
  }

  async searchSimilar(queryText, projectId = null, limit = 10) {
    const queryEmbedding = await this.generateEmbedding(queryText);
    
    let query = `
      SELECT e.content, e.metadata, ent.entity_type, ent.name, ent.properties,
             1 - (e.embedding::vector <=> $1::vector) as similarity
      FROM embeddings e
      JOIN entities ent ON e.entity_id = ent.id
      WHERE 1 - (e.embedding::vector <=> $1::vector) > 0.7
    `;
    
    const params = [JSON.stringify(queryEmbedding)];
    
    if (projectId) {
      query += ` AND ent.project_id = $2`;
      params.push(projectId);
      query += ` ORDER BY e.embedding::vector <=> $1::vector LIMIT $3`;
      params.push(limit);
    } else {
      query += ` ORDER BY e.embedding::vector <=> $1::vector LIMIT $2`;
      params.push(limit);
    }

    const result = await db.query(query, params);
    return result.rows;
  }
}

module.exports = new EmbeddingService();
