const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
require('dotenv').config();

const projectService = require('./services/projectService');
const embeddingService = require('./services/embeddingService');
const ragService = require('./services/ragService');

const app = express();

app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '50mb' }));

app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString() 
  });
});

app.post('/projects', async (req, res) => {
  try {
    console.log('Processing project:', req.body.jobIndex || 'No ID');
    const result = await projectService.processProject(req.body);
    
    if (process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== 'your_openai_api_key_here') {
      try {
        const embeddingCount = await embeddingService.generateProjectEmbeddings(req.body.jobIndex);
        result.embeddingsGenerated = embeddingCount;
      } catch (error) {
        console.error('Error generating embeddings:', error.message);
        result.embeddingError = error.message;
      }
    }
    
    res.json({ success: true, projectId: req.body.jobIndex, ...result });
  } catch (error) {
    console.error('Error processing project:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/projects/:projectId/budget', async (req, res) => {
  try {
    const summary = await projectService.getBudgetSummary(req.params.projectId);
    res.json({ projectId: req.params.projectId, budgetSummary: summary });
  } catch (error) {
    console.error('Error getting budget summary:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/projects/:projectId/query', async (req, res) => {
  try {
    const { question } = req.body;
    
    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }
    
    const result = await ragService.queryProject(question, req.params.projectId);
    res.json({ 
      projectId: req.params.projectId,
      question: question,
      ...result 
    });
    
  } catch (error) {
    console.error('Error querying project:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/projects/:projectId/schedule', async (req, res) => {
  try {
    const schedule = await ragService.getScheduleStatus(req.params.projectId);
    res.json({ projectId: req.params.projectId, schedule: schedule });
  } catch (error) {
    console.error('Error getting schedule:', error);
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
