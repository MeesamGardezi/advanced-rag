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
