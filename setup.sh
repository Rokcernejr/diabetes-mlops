#!/bin/bash
set -euo pipefail

echo "🚀 Setting up MLOps Pipeline..."

# Auto-detect environment and set defaults
ENVIRONMENT=${ENVIRONMENT:-dev}
AWS_REGION=${AWS_REGION:-us-east-1}
PROJECT_NAME=${PROJECT_NAME:-diabetes-mlops}

# Create .env from template if not exists
if [[ ! -f .env ]]; then
    echo "📝 Creating .env file with smart defaults..."
    cat > .env << EOF
# Auto-generated environment file
PROJECT_NAME=${PROJECT_NAME}
ENVIRONMENT=${ENVIRONMENT}
AWS_REGION=${AWS_REGION}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "CHANGE_ME")
DOMAIN=${PROJECT_NAME}.local
ECR_URI=\${AWS_ACCOUNT_ID}.dkr.ecr.\${AWS_REGION}.amazonaws.com
EOF
fi

# Install tools
make install-tools

# Setup pre-commit hooks
pre-commit install

# Install dependencies
poetry install --with dev

# Run health check
make health-check

echo "✅ Setup complete! Run 'make dev' to start local development."