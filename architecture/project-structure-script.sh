#!/bin/bash

# Agentic News Bot - Project Structure Setup Script
# This script creates the complete directory structure and placeholder files

echo "🚀 Creating Agentic News Bot project structure..."

# Create main directories
mkdir -p notebooks
mkdir -p src/agents
mkdir -p src/data/News_dataset
mkdir -p src/embeddings
mkdir -p src/models/embedding_model
mkdir -p templates
mkdir -p tests
mkdir -p utils
mkdir -p architecture

echo "📁 Created main directories"

# Create source files
touch src/agents/news_prediction_agent.py
touch src/embeddings/embed_model.py
touch src/models/.gitkeep
touch src/data/News_dataset/.gitkeep

echo "📄 Created source files"

# Create utility files
touch utils/data_preprocessing.py
touch utils/data_validation.py
touch utils/simulation_helpers.py
touch utils/train_and_save_model.py

echo "🛠️  Created utility files"

# Create template files
touch templates/index.html

echo "🎨 Created template files"

# Create test files
touch tests/news_prediction.py

echo "🧪 Created test files"

# Create notebook files
touch notebooks/fake-news-detection.ipynb

echo "📓 Created notebook files"

# Create root-level files
touch app.py
touch requirements.txt
touch .gitignore
touch .env
touch README.md

echo "📋 Created root-level files"

# Create .gitignore content
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Models (large files)
*.pkl
*.h5
*.pth
*.onnx

# Data
*.csv
*.json
*.parquet

# Jupyter
.ipynb_checkpoints/
EOF

echo "✅ Project structure created successfully!"
echo ""
echo "📂 Structure:"
tree -L 2 -I '__pycache__|*.pyc'
