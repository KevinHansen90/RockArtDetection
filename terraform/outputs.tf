output "gcs_bucket_name" {
  value       = google_storage_bucket.rockart_bucket.name
  description = "Name of the GCS bucket for dataset and experiment storage"
}

output "gcs_bucket_url" {
  value       = google_storage_bucket.rockart_bucket.url
  description = "GCS URI of the bucket"
}

output "artifact_registry_repo" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.rockart_repo.repository_id}"
  description = "Full Artifact Registry repository endpoint for Docker images"
}

output "service_account_email" {
  value       = google_service_account.vertex_sa.email
  description = "Email of the Service Account created for Vertex AI training jobs"
}

output "tensorboard_id" {
  value       = google_vertex_ai_tensorboard.rockart_tensorboard.id
  description = "Resource ID of the managed Vertex AI TensorBoard instance"
}

