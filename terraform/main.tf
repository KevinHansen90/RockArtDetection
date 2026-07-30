terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# GCS Bucket for datasets, configs, and model experiment artifacts
resource "google_storage_bucket" "rockart_bucket" {
  name                        = var.bucket_name != "" ? var.bucket_name : "${var.project_id}-rockart-data"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = {
    environment = var.environment
    project     = "rockart-detection"
  }
}

# Artifact Registry Repository for Container Images (replacing legacy GCR)
resource "google_artifact_registry_repository" "rockart_repo" {
  location      = var.region
  repository_id = var.repository_id
  description   = "Artifact Registry repository for RockArtDetection Docker images"
  format        = "DOCKER"

  labels = {
    environment = var.environment
    project     = "rockart-detection"
  }
}

# Dedicated Service Account for Vertex AI Training Custom Jobs
resource "google_service_account" "vertex_sa" {
  account_id   = "rockart-vertex-sa"
  display_name = "Service Account for RockArt Detection Vertex AI Custom Jobs"
  description  = "Managed by Terraform for RockArtDetection Vertex AI execution"
}

# IAM Role Bindings for Service Account
resource "google_project_iam_member" "vertex_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}

resource "google_project_iam_member" "vertex_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}

resource "google_project_iam_member" "vertex_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}

# Managed Vertex AI TensorBoard Instance for GCP Native Metric Tracking
resource "google_vertex_ai_tensorboard" "rockart_tensorboard" {
  display_name = "rockart-tensorboard"
  description  = "Managed Vertex AI TensorBoard instance for RockArt Detection experiments"
  region       = var.region

  labels = {
    environment = var.environment
    project     = "rockart-detection"
  }
}

