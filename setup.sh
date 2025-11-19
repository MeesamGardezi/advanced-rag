#!/bin/bash

# Create main project directory
mkdir -p rag-backend
cd rag-backend

# Create app directory structure
mkdir -p app/models
mkdir -p app/services
mkdir -p app/api

# Create __init__.py files for Python packages
touch app/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/api/__init__.py

# Create model files
touch app/models/job.py
touch app/models/estimate.py
touch app/models/comparison.py
touch app/models/request.py

# Create service files
touch app/services/embedding.py
touch app/services/qdrant_service.py
touch app/services/ingestion.py
touch app/services/rag_service.py
touch app/services/firebase_service.py

# Create API files
touch app/api/routes.py

# Create main app files
touch app/main.py
touch app/config.py

# Create root level files
touch requirements.txt
touch docker-compose.yml
touch .env.example
touch .env
touch README.md
touch .gitignore

echo "✅ Project structure created successfully!"
echo ""
echo "Project structure:"
tree -L 3 rag-backend 2>/dev/null || find rag-backend -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g'