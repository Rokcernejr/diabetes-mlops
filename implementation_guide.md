# Complete Implementation Guide - From Zero to Production

## 🎯 Overview
This guide takes you from zero to a fully functional MLOps pipeline in production. Follow these steps in order for the smoothest experience.

---

## 📋 Prerequisites

### Required Tools

**Option A: Windows with Chocolatey (Recommended)**
```powershell
# Install Chocolatey first (run as Administrator)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install required tools
choco install git gh terraform awscli kubernetes-cli kubernetes-helm docker-desktop -y
```

**Option B: Windows with Winget (Built-in)**
```powershell
# Install tools using Windows Package Manager
winget install Git.Git
winget install GitHub.cli
winget install Hashicorp.Terraform
winget install Amazon.AWSCLI
winget install Kubernetes.kubectl
winget install Helm.Helm
winget install Docker.DockerDesktop
```

**Option C: WSL2 (Ubuntu) - Best for Development**
```bash
# Enable WSL2 first, then install Ubuntu from Microsoft Store
# Inside WSL2 Ubuntu terminal:
sudo apt-get update && sudo apt-get install -y curl wget unzip

# Install tools
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install terraform

# AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Helm
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update && sudo apt-get install helm

# GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt-get update && sudo apt-get install gh

# Docker (install Docker Desktop for Windows, it integrates with WSL2)
```

**Option D: Manual Downloads (If package managers fail)**
- **Git**: https://git-scm.com/download/win
- **GitHub CLI**: https://github.com/cli/cli/releases
- **Terraform**: https://www.terraform.io/downloads
- **AWS CLI**: https://aws.amazon.com/cli/
- **kubectl**: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/
- **Helm**: https://helm.sh/docs/intro/install/
- **Docker Desktop**: https://www.docker.com/products/docker-desktop

### Required Accounts
- ✅ **GitHub account** with admin access to create repositories
- ✅ **AWS account** with programmatic access
- ✅ **Docker Hub account** (optional, we'll use AWS ECR)

---

## 🚀 Step-by-Step Implementation

### Phase 1: Repository Setup (10 minutes)

#### 1.1 Create GitHub Repository

**Windows PowerShell/Command Prompt:**
```powershell
# Option A: Using GitHub CLI (recommended)
gh auth login  # Login to GitHub first
gh repo create diabetes-mlops --public --clone
cd diabetes-mlops

# Option B: Manual creation
# Go to https://github.com/new
# Create repository named "diabetes-mlops"
# Clone it locally
git clone https://github.com/YOUR_USERNAME/diabetes-mlops.git
cd diabetes-mlops
```

**WSL2 (Ubuntu):**
```bash
# Same commands work in WSL2
gh repo create diabetes-mlops --public --clone
cd diabetes-mlops
```

#### 1.2 Create Repository Structure

**Windows PowerShell:**
```powershell
# Create all directories
New-Item -ItemType Directory -Path app, ml, infra, "infra\modules", "infra\modules\network", "infra\modules\eks", "infra\modules\s3", "infra\modules\rds", "infra\modules\iam", helm, "helm\templates", dags, grafana, ".github", ".github\workflows", tests, "tests\unit", "tests\integration", docs -Force

# Create essential files
New-Item -ItemType File -Path README.md, Makefile, pyproject.toml, Dockerfile, docker-compose.yml, setup.sh, .env.example, .gitignore -Force

# Create app files
New-Item -ItemType File -Path "app\__init__.py", "app\main.py", "app\schemas.py", "app\auth.py", "app\deps.py", "app\metrics.py", "app\shap_utils.py" -Force

# Create ml files  
New-Item -ItemType File -Path "ml\__init__.py", "ml\preprocess.py", "ml\feature_engineering.py", "ml\train.py", "ml\tune.py", "ml\explain.py", "ml\model_loader.py", "ml\dummy_model.py" -Force

# Create infrastructure files
New-Item -ItemType File -Path "infra\main.tf", "infra\variables.tf", "infra\outputs.tf" -Force
New-Item -ItemType File -Path "infra\modules\network\main.tf", "infra\modules\network\variables.tf" -Force
New-Item -ItemType File -Path "infra\modules\eks\main.tf", "infra\modules\eks\variables.tf" -Force
New-Item -ItemType File -Path "infra\modules\s3\main.tf", "infra\modules\s3\variables.tf" -Force
New-Item -ItemType File -Path "infra\modules\rds\main.tf", "infra\modules\rds\variables.tf" -Force
New-Item -ItemType File -Path "infra\modules\iam\main.tf", "infra\modules\iam\variables.tf" -Force

# Create helm files
New-Item -ItemType File -Path "helm\Chart.yaml", "helm\values.yaml" -Force
New-Item -ItemType File -Path "helm\templates\deployment.yaml", "helm\templates\service.yaml", "helm\templates\ingress.yaml", "helm\templates\hpa.yaml", "helm\templates\pdb.yaml", "helm\templates\servicemonitor.yaml" -Force

# Create other files
New-Item -ItemType File -Path "tests\test_app.py", "tests\test_ml.py", "tests\test_health.py" -Force
New-Item -ItemType File -Path ".github\workflows\ci-cd.yml" -Force
```

**WSL2/Git Bash:**
```bash
# Same as the original Linux commands
mkdir -p {app,ml,infra/{modules/{network,eks,s3,rds,iam}},helm/templates,dags,grafana,.github/workflows,tests/{unit,integration},docs}
touch {README.md,Makefile,pyproject.toml,Dockerfile,docker-compose.yml}
# ... rest of the original commands
```

#### 1.3 Copy All Code Files
Now copy all the code from the improved pipeline artifact into the respective files. Here's the priority order:

**Core Files (copy these first):**
```bash
# Copy the content from the artifact to these files:
# 1. setup.sh (the one-command setup script)
# 2. Makefile (enhanced with smart automation)
# 3. pyproject.toml (Python dependencies)
# 4. docker-compose.yml (local development stack)
# 5. Dockerfile (multi-stage with dev target)
# 6. app/main.py (enhanced FastAPI app)
# 7. ml/model_loader.py (model management)
# 8. .github/workflows/ci-cd.yml (CI/CD pipeline)
```

#### 1.4 Create Essential Configuration Files

**`.env.example`**:
```bash
# Copy this to .env and fill in your values
PROJECT_NAME=diabetes-mlops
ENVIRONMENT=dev
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=YOUR_AWS_ACCOUNT_ID
DOMAIN=diabetes-mlops.local
ECR_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

**`.gitignore`**:
```bash
# Python
__pycache__/
*.py[cod]
.venv/
.env

# Terraform
*.tfstate
*.tfstate.*
.terraform/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Local data
data/
models/
mlruns/
```

### Phase 2: Local Development Setup (15 minutes)

#### 2.1 Run Automated Setup

**Windows PowerShell:**
```powershell
# Make sure Docker Desktop is running first!

# If you have WSL2, run this in WSL2 terminal:
./setup.sh

# If you're using pure Windows PowerShell, run manually:
# Install Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Add Poetry to PATH (restart terminal after this)
$env:PATH += ";$env:APPDATA\Python\Scripts"

# Install dependencies
poetry install

# Create .env from template
Copy-Item .env.example .env
# Edit .env with your AWS account ID using notepad or your preferred editor
notepad .env
```

**WSL2 (Recommended for Windows):**
```bash
# This sets up everything locally
./setup.sh
```

If setup script fails, run manually:
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Create .env from template
cp .env.example .env
# Edit .env with your AWS account ID
nano .env  # or vim .env
```

#### 2.2 Start Local Development Environment
```bash
# Start all services locally
make dev

# Check everything is running
make health-check
```

You should see:
- ✅ API at http://localhost:8000
- ✅ MLflow at http://localhost:5000  
- ✅ Grafana at http://localhost:3000 (admin/admin)

#### 2.3 Test Local Setup
```bash
# Run tests
make test

# Test API manually
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Interactive API docs
```

### Phase 3: AWS Infrastructure Setup (20 minutes)

#### 3.1 Configure AWS Credentials
```bash
# Configure AWS CLI
aws configure
# Enter your: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)

# Verify access
aws sts get-caller-identity
```

#### 3.2 Set Up Terraform Backend (One-time)
```bash
# Create S3 bucket for Terraform state
aws s3 mb s3://diabetes-mlops-terraform-state-$(date +%s)

# Create infra/backend.tf
cat > infra/backend.tf << EOF
terraform {
  backend "s3" {
    bucket = "diabetes-mlops-terraform-state-YOUR_SUFFIX"
    key    = "diabetes-mlops/terraform.tfstate"
    region = "us-east-1"
  }
}
EOF
```

#### 3.3 Deploy Infrastructure
```bash
cd infra

# Initialize Terraform
terraform init

# Create terraform.tfvars
cat > terraform.tfvars << EOF
env        = "dev"
region     = "us-east-1"
account_id = "$(aws sts get-caller-identity --query Account --output text)"

# These will be auto-generated but you can customize
raw_bucket       = "diabetes-raw-data-dev"
processed_bucket = "diabetes-processed-dev"  
model_bucket     = "diabetes-models-dev"

# Database password (use a strong password)
db_password = "YourStrongPassword123!"
EOF

# Plan and apply infrastructure
terraform plan
terraform apply

# This creates:
# - VPC with public/private subnets
# - EKS cluster with managed node groups
# - S3 buckets for data and models
# - RDS instance for MLflow
# - IAM roles and policies
```

**⏱️ Infrastructure deployment takes 15-20 minutes (EKS cluster creation is slow)**

#### 3.4 Configure kubectl
```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name dev-diabetes-eks

# Verify connection
kubectl get nodes
```

### Phase 4: CI/CD Pipeline Setup (10 minutes)

#### 4.1 Create GitHub Secrets
Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:
```bash
AWS_ACCOUNT_ID: your-aws-account-id
AWS_REGION: us-east-1
ECR_REPOSITORY: diabetes-mlops
```

#### 4.2 Set Up OIDC for GitHub Actions (Secure, no long-lived keys)
```bash
# Create OIDC role for GitHub Actions
cd infra

# Add to variables.tf
cat >> variables.tf << EOF
variable "github_repo" {
  description = "GitHub repository in format owner/repo"
  type        = string
  default     = "YOUR_USERNAME/diabetes-mlops"
}
EOF

# Add to main.tf
cat >> main.tf << EOF
# OIDC provider for GitHub Actions
resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# Role for GitHub Actions
resource "aws_iam_role" "github_actions" {
  name = "github-actions-diabetes-mlops"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github.arn
        }
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:\${var.github_repo}:*"
          }
        }
      }
    ]
  })
}

# Attach necessary policies
resource "aws_iam_role_policy_attachment" "github_actions_ecr" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

resource "aws_iam_role_policy_attachment" "github_actions_eks" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions.arn
}
EOF

# Apply changes
terraform apply

# Get the role ARN
terraform output github_actions_role_arn
```

Add the role ARN to GitHub secrets as `AWS_ROLE_ARN`.

#### 4.3 Create ECR Repository
```bash
# Create ECR repository
aws ecr create-repository --repository-name diabetes-mlops --region us-east-1
```

### Phase 5: First Deployment (15 minutes)

#### 5.1 Commit and Push Code
```bash
# Add all files
git add .

# Commit
git commit -m "feat: initial MLOps pipeline setup

- Complete FastAPI application with health checks
- ML model management with fallback strategies  
- Docker multi-stage build with dev/prod targets
- Terraform infrastructure with EKS and RDS
- GitHub Actions CI/CD pipeline
- Local development environment with docker-compose"

# Push to main branch
git push origin main
```

#### 5.2 Monitor First Pipeline Run
```bash
# Watch the GitHub Actions pipeline
gh run watch

# Or go to: https://github.com/YOUR_USERNAME/diabetes-mlops/actions
```

The pipeline will:
1. ✅ Run tests (pytest, linting)
2. ✅ Build and push Docker image to ECR
3. ✅ Deploy to EKS development environment
4. ✅ Run smoke tests
5. ✅ Send Slack notification (if configured)

#### 5.3 Verify Deployment
```bash
# Check pods are running
kubectl get pods -n dev

# Check service
kubectl get svc -n dev

# Port forward to test locally
kubectl port-forward svc/diabetes-dev 8080:8000 -n dev

# Test the deployed API
curl http://localhost:8080/health
curl http://localhost:8080/metrics
```

### Phase 6: Production Deployment (5 minutes)

#### 6.1 Create Production Release
```bash
# Tag for production release
git tag v1.0.0
git push origin v1.0.0
```

This automatically triggers production deployment with:
- Production environment approval gate
- Smoke tests
- Automatic rollback on failure

---

## 🔄 Daily Development Workflow

### Making Changes
```bash
# 1. Pull latest changes
git pull origin main

# 2. Create feature branch
git checkout -b feature/improve-model

# 3. Start local development
make dev

# 4. Make your changes
# Edit app/main.py, ml/train.py, etc.

# 5. Test changes
make test
make lint

# 6. Commit and push
git add .
git commit -m "feat: improve model accuracy"
git push origin feature/improve-model

# 7. Create pull request
gh pr create --title "Improve model accuracy" --body "Details..."

# 8. After approval, merge triggers dev deployment
# 9. Tag triggers production deployment
git tag v1.0.1
git push origin v1.0.1
```

### Working with Data
```bash
# Upload training data
aws s3 cp diabetic_data.csv s3://diabetes-raw-data-dev/

# Trigger model training pipeline
# (This would be done through Airflow/Prefect or manual trigger)

# Check MLflow for new models
kubectl port-forward svc/mlflow 5000:5000 -n dev
# Go to http://localhost:5000
```

---

## 🛠️ Troubleshooting Common Issues

### Infrastructure Issues
```bash
# EKS cluster creation failed
terraform destroy -target=module.eks
terraform apply

# S3 bucket name conflicts
# Edit infra/terraform.tfvars with unique bucket names

# AWS credentials issues
aws sts get-caller-identity
aws configure list
```

### Application Issues
```bash
# Pods not starting
kubectl describe pod <pod-name> -n dev
kubectl logs <pod-name> -n dev

# Model loading failures
# Check MLflow connection and model registry
kubectl exec -it <pod-name> -n dev -- python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

### Pipeline Issues
```bash
# GitHub Actions failing
# Check secrets are set correctly
# Verify AWS permissions
# Check ECR repository exists
```

---

## 📊 Monitoring and Observability

### Access Monitoring Tools
```bash
# Grafana dashboard
kubectl port-forward svc/grafana 3000:3000 -n monitoring
# Go to http://localhost:3000 (admin/admin)

# Prometheus metrics
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# View application logs
kubectl logs -f deployment/diabetes-dev -n dev
```

### Key Metrics to Watch
- **API latency** (p95 < 100ms)
- **Error rate** (< 1%)
- **Model accuracy** (AUROC > 0.75)
- **Data drift** (p-value > 0.05)
- **Resource utilization** (CPU < 70%, Memory < 80%)

---

## 💰 Cost Management

### Monitor Costs
```bash
# Check current AWS costs
aws ce get-cost-and-usage \
  --time-period Start=2025-06-01,End=2025-06-30 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

### Optimization Tips
- **Use Spot instances** (already configured, saves ~70%)
- **Scale down dev environment** when not in use
- **Set up S3 lifecycle policies** (already configured)
- **Monitor unused resources** with AWS Cost Explorer

---

## 🎯 Next Steps

### Immediate (Week 1)
- ✅ Complete basic setup and first deployment
- ✅ Train initial model with provided data
- ✅ Set up monitoring alerts
- ✅ Configure Slack notifications

### Short-term (Month 1)
- 📊 Implement A/B testing for model versions
- 🔄 Set up automated model retraining
- 📈 Add custom business metrics
- 🛡️ Implement model drift detection

### Long-term (Quarter 1)
- 🌐 Multi-region deployment
- 🔒 Enhanced security with service mesh
- 📱 Model serving at edge locations
- 🤖 Automated hyperparameter tuning

---

## 🆘 Getting Help

### Resources
- **Documentation**: Check `docs/` folder in repository
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub Discussions for questions
- **Monitoring**: Check Grafana dashboards for system health

### Support Contacts
- **Infrastructure**: AWS Support (if you have a support plan)
- **Application**: GitHub Issues in repository
- **MLOps**: MLflow documentation and community

---

**🎉 Congratulations!** You now have a production-ready MLOps pipeline. The total setup time should be around 60-75 minutes, and you'll have a system that can scale from development to production with enterprise-grade monitoring and security.
