# outputs.tf
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "buckets" {
  description = "S3 bucket names"
  value       = local.buckets
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}
