
# variables.tf
# This file declares all the variables used in main.tf

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
  type        = string
  default     = "prod"
  
  validation {
    condition = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }

}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.30"
}

variable "node_instance_types" {
  description = "Instance types for EKS node groups"
  type        = list(string)
  default     = ["m5.large", "m5a.large"]
}

variable "node_min_size" {
  description = "Minimum number of nodes in EKS node group"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes in EKS node group"
  type        = number
  default     = 10
}

variable "node_desired_size" {
  description = "Desired number of nodes in EKS node group"
  type        = number
  default     = 3
}

variable "enable_spot_instances" {
  description = "Whether to use spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "s3_lifecycle_transition_days" {
  description = "Number of days before transitioning to IA storage"
  type        = number
  default     = 30
}

variable "s3_glacier_transition_days" {
  description = "Number of days before transitioning to Glacier"
  type        = number
  default     = 90
}

variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

