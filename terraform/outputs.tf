output "alb_dns_name" {
  value       = aws_lb.main.dns_name
  description = "Public DNS of the Application Load Balancer"
}

output "api_url" {
  value       = "http://${aws_lb.main.dns_name}/api/v1"
  description = "Base API URL"
}

output "ecs_cluster_name" {
  value       = aws_ecs_cluster.main.name
  description = "ECS cluster name"
}

output "cloudwatch_log_group" {
  value       = aws_cloudwatch_log_group.api.name
  description = "CloudWatch log group for ECS container logs"
}
