CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    title TEXT,
    status TEXT,
    client_name TEXT,
    site_address TEXT,
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_id, entity_id)
);

CREATE TABLE relationships (
    id SERIAL PRIMARY KEY,
    source_entity_id INTEGER REFERENCES entities(id),
    target_entity_id INTEGER REFERENCES entities(id),
    relationship_type TEXT NOT NULL,
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id),
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(entity_id)
);

CREATE TABLE cost_analysis (
    id SERIAL PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    task_scope TEXT,
    cost_code TEXT,
    budgeted_total DECIMAL,
    actual_total DECIMAL,
    variance DECIMAL,
    variance_percentage DECIMAL,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_entities_project_type ON entities(project_id, entity_type);
CREATE INDEX idx_entities_entity_id ON entities(entity_id);
CREATE INDEX idx_relationships_source ON relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON relationships(target_entity_id);
CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_cost_analysis_project ON cost_analysis(project_id);
