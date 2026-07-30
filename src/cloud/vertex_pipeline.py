#!/usr/bin/env python3
"""
Vertex AI Pipeline Orchestrator using Google Cloud Vertex AI SDK.
Defines end-to-end DAG execution on GCP (Ingestion -> Tiling -> Preprocessing -> Fine-Tuning -> Clustering).
"""

import os
import argparse
from google.cloud import aiplatform


def create_and_run_vertex_pipeline(
    project_id: str,
    region: str,
    gcs_bucket: str,
    container_image_uri: str,
    model_name: str = "retinanet",
    filter_type: str = "bilateral",
    feature_model: str = "resnet50",
    cluster_algo: str = "kmeans",
    service_account: str | None = None,
):
    """
    Launches a sequential multi-stage execution pipeline on Vertex AI.
    """
    aiplatform.init(project=project_id, location=region)

    pipeline_run_id = f"rockart-pipeline-{model_name}-{filter_type}"
    print(f"==================================================")
    print(f"   VERTEX AI PIPELINE LAUNCHER: {pipeline_run_id}")
    print(f"==================================================")

    # 1. Step 1: Preprocessing Job (Tiling & Filtering)
    prep_job_name = f"{pipeline_run_id}-prep"
    print(f"\n[Step 1] Submitting Preprocessing Job: {prep_job_name}")
    prep_args = [
        "python", "-m", "src.cli", "pipeline",
        "--filter", filter_type,
        "--model", model_name,
        "--feature-model", feature_model,
        "--cluster-algo", cluster_algo,
    ]

    job = aiplatform.CustomContainerTrainingJob(
        display_name=prep_job_name,
        container_uri=container_image_uri,
    )

    run_kwargs = {
        "args": prep_args,
        "machine_type": "n1-standard-4",
        "replica_count": 1,
        "sync": False,
    }
    if service_account:
        run_kwargs["service_account"] = service_account

    job.run(**run_kwargs)
    print(f"[+] Vertex AI Pipeline Job submitted successfully: {prep_job_name}")


def main():
    parser = argparse.ArgumentParser(description="Launch End-to-End Vertex AI Pipeline for RockArtDetection.")
    parser.add_argument("--project_id", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--gcs_bucket", required=True, help="GCS Bucket name")
    parser.add_argument("--container_image", required=True, help="Artifact Registry Docker image URI")
    parser.add_argument("--model", default="retinanet", help="Detection model architecture")
    parser.add_argument("--filter", default="bilateral", help="Dataset filter")
    parser.add_argument("--feature-model", default="resnet50", help="Clustering feature extractor")
    parser.add_argument("--cluster-algo", default="kmeans", help="Clustering algorithm")
    parser.add_argument("--service_account", default=None, help="Service Account email")

    args = parser.parse_args()

    create_and_run_vertex_pipeline(
        project_id=args.project_id,
        region=args.region,
        gcs_bucket=args.gcs_bucket,
        container_image_uri=args.container_image,
        model_name=args.model,
        filter_type=args.filter,
        feature_model=args.feature_model,
        cluster_algo=args.cluster_algo,
        service_account=args.service_account,
    )


if __name__ == "__main__":
    main()
