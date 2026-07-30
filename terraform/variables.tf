variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type        = string
  description = "GCP Region for resources"
  default     = "us-central1"
}

variable "bucket_name" {
  type        = string
  description = "Optional GCS bucket name. If empty, defaults to <project_id>-rockart-data"
  default     = ""
}

variable "repository_id" {
  type        = string
  description = "Artifact Registry Docker repository ID"
  default     = "rockart-docker-repo"
}

variable "environment" {
  type        = string
  description = "Environment identifier (e.g. dev, staging, prod)"
  default     = "dev"
}
