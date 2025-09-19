const { Pool } = require('pg');
require('dotenv').config();

class DatabaseService {
  constructor() {
    this.pool = new Pool({
      user: process.env.DB_USER || 'postgres',
      host: process.env.DB_HOST || 'localhost',
      database: process.env.DB_NAME || 'construction_rag',
      password: process.env.DB_PASSWORD || '',
      port: process.env.DB_PORT || 5432,
    });
    this.testConnection();
  }

  async testConnection() {
    try {
      const client = await this.pool.connect();
      const result = await client.query('SELECT NOW()');
      console.log('Database connected:', result.rows[0].now);
      client.release();
    } catch (error) {
      console.error('Database connection failed:', error.message);
      process.exit(1);
    }
  }

  async query(text, params) {
    return await this.pool.query(text, params);
  }

  async getClient() {
    return await this.pool.connect();
  }
}

module.exports = new DatabaseService();
