variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "ddos-detection"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "api_cpu" {
  description = "ECS task CPU units (256=0.25 vCPU)"
  type        = number
  default     = 512
}

variable "api_memory" {
  description = "ECS task memory in MiB"
  type        = number
  default     = 1024
}

variable "api_desired_count" {
  description = "Number of ECS task replicas"
  type        = number
  default     = 2
}

variable "api_secret_key" {
  description = "JWT secret key — set via TF_VAR_api_secret_key env var"
  type        = string
  sensitive   = true
  default     = "change-me-before-deploy"
}

variable "container_port" {
  type    = number
  default = 8000
}
