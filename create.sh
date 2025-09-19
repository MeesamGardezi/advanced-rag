#!/bin/bash

# Simple Construction Graph RAG - DEVELOPMENT SETUP
echo "🚀 Creating Simple Graph RAG System..."

# Clean, simple structure
mkdir -p construction-rag/{src,config}
cd construction-rag

echo "📦 Package.json..."
cat > package.json << 'EOF'
{
  "name": "construction-rag",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "firebase-admin": "^12.0.0", 
    "openai": "^4.20.1",
    "pg": "^8.11.5",
    "dotenv": "^16.3.1",
    "cors": "^2.8.5",
    "uuid": "^9.0.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.2"
  }
}
EOF

echo "⚙️  Environment..."
cat > .env << 'EOF'
NODE_ENV=development
PORT=3000

DATABASE_URL=postgresql://username:password@localhost:5432/construction_rag

FIREBASE_PROJECT_ID=
FIREBASE_PRIVATE_KEY=
FIREBASE_CLIENT_EMAIL=

OPENAI_API_KEY=
EOF

echo "🗄️  Database schema..."
cat > schema.sql << 'EOF'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    firebase_id TEXT UNIQUE NOT NULL,
    title TEXT,
    client_name TEXT,
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id),
    entity_type TEXT NOT NULL,
    cost_code TEXT,
    description TEXT,
    task_scope TEXT,
    category TEXT,
    total_amount DECIMAL,
    budgeted_amount DECIMAL,
    raw_data JSONB,
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES entities(id),
    target_id UUID REFERENCES entities(id),
    type TEXT,
    strength DECIMAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID REFERENCES entities(id) UNIQUE,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_entities_project ON entities(project_id);
CREATE INDEX idx_entities_cost_code ON entities(cost_code);
CREATE INDEX idx_relationships_source ON relationships(source_id);
CREATE INDEX idx_embeddings_cosine ON embeddings USING ivfflat (embedding vector_cosine_ops);
EOF

echo "📁 Creating ALL empty files..."

# Config files
touch config/database.js
touch config/firebase.js
touch config/openai.js

# Source files  
touch src/server.js
touch src/database.js
touch src/firebase.js
touch src/entityProcessor.js
touch src/relationshipBuilder.js
touch src/embeddingService.js
touch src/ragService.js
touch src/utils.js

echo "✅ DONE! Simple structure created:"
echo ""
echo "construction-rag/"
echo "├── config/"
echo "│   ├── database.js         (EMPTY)"
echo "│   ├── firebase.js         (EMPTY)" 
echo "│   └── openai.js           (EMPTY)"
echo "├── src/"
echo "│   ├── server.js           (EMPTY)"
echo "│   ├── database.js         (EMPTY)"
echo "│   ├── firebase.js         (EMPTY)"
echo "│   ├── entityProcessor.js  (EMPTY)"
echo "│   ├── relationshipBuilder.js (EMPTY)"
echo "│   ├── embeddingService.js (EMPTY)"
echo "│   ├── ragService.js       (EMPTY)"
echo "│   └── utils.js            (EMPTY)"
echo "├── schema.sql"
echo "├── package.json"
echo "└── .env"
echo ""
echo "🎯 ALL FILES ARE EMPTY AND READY FOR CODING!"
echo ""
echo "Next:"
echo "1. npm install"
echo "2. Setup your PostgreSQL database"
echo "3. Run schema.sql"
echo "4. Add your credentials to .env"
echo "5. Let's start coding!"