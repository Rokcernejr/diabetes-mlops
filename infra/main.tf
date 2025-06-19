terraform {
  required_version = ">= 1.8"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Use Terraform Cloud for state management
  cloud {
    organization = "your-org"
    workspaces {
      name = "diabetes-mlops"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Auto-detect current AWS account and region
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  # Smart naming with environment prefix
  name_prefix = "${var.environment}-diabetes"
  
  # Auto-generate bucket names to avoid conflicts
  buckets = {
    raw       = "${local.name_prefix}-raw-${random_id.bucket_suffix.hex}"
    processed = "${local.name_prefix}-processed-${random_id.bucket_suffix.hex}"
    models    = "${local.name_prefix}-models-${random_id.bucket_suffix.hex}"
  }
  
  common_tags = {
    Project     = "diabetes-mlops"
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = "mlops-team"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Automatically create VPC with sensible defaults
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${local.name_prefix}-vpc"
  cidr = var.vpc_cidr
  
  azs             = data.aws_availability_zones.available.names
  private_subnets = [for i in range(3) : cidrsubnet(var.vpc_cidr, 8, i)]
  public_subnets  = [for i in range(3) : cidrsubnet(var.vpc_cidr, 8, i + 10)]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = local.common_tags
}

# EKS cluster with managed node groups
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"
  
  cluster_name    = "${local.name_prefix}-eks"
  cluster_version = "1.30"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Managed node groups with spot instances for cost optimization
  eks_managed_node_groups = {
    main = {
      min_size       = 1
      max_size       = 10
      desired_size   = 3
      instance_types = ["m5.large", "m5a.large"]
      capacity_type  = "SPOT"
      
      k8s_labels = {
        role = "worker"
      }
    }
  }
  
  # Enable IRSA for service accounts
  enable_irsa = true
  
  tags = local.common_tags
}

# S3 buckets with automatic lifecycle policies
resource "aws_s3_bucket" "buckets" {
  for_each = local.buckets
  
  bucket = each.value
  tags   = local.common_tags
}

resource "aws_s3_bucket_lifecycle_configuration" "buckets" {
  for_each = aws_s3_bucket.buckets
  
  bucket = each.value.id
  
  rule {
    id     = "transition-to-ia"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# Outputs for use in other systems
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "buckets" {
  value = local.buckets
}