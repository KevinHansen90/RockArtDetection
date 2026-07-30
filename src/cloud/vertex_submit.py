#!/usr/bin/env python3
"""
Vertex AI Custom Job Submission & Quota-Aware Batch Manager.
Includes automatic deduplication to prevent submitting duplicate running/completed jobs.
"""

import os
import sys
import time
import argparse
from google.cloud import aiplatform


def get_existing_custom_job_names(project_id: str, region: str) -> set[str]:
    """Returns a set of display names for existing Custom Jobs that are active or succeeded."""
    aiplatform.init(project=project_id, location=region)
    jobs = aiplatform.CustomJob.list()
    active_states = {
        aiplatform.gapic.JobState.JOB_STATE_PENDING,
        aiplatform.gapic.JobState.JOB_STATE_RUNNING,
        aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED,
    }
    existing = {j.display_name for j in jobs if j.state in active_states}
    return existing


def submit_vertex_custom_job(
    project_id: str,
    region: str,
    display_name: str,
    container_image_uri: str,
    args: list[str],
    command: list[str] | None = None,
    service_account: str | None = None,
    machine_type: str = "n1-highmem-4",
    accelerator_type: str | None = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    sync: bool = False,
    gcs_bucket: str = "your-gcs-bucket-name",
    tensorboard: str | None = None,
    experiment_name: str = "rockart-detection",
    existing_jobs: set[str] | None = None,
):
    """
    Submits a Vertex AI Custom Training Job with automatic deduplication and GCP TensorBoard / Experiment tracking.
    """
    if existing_jobs is not None and display_name in existing_jobs:
        print(f"[SKIP] Job '{display_name}' is already submitted/running on Vertex AI.")
        return None

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f"gs://{gcs_bucket}",
        experiment=experiment_name,
    )

    container_spec = {
        "image_uri": container_image_uri,
        "args": args,
        "env": [{"name": "AIP_STORAGE_URI", "value": f"gs://{gcs_bucket}/experiments/{display_name}"}],
    }
    if command:
        container_spec["command"] = command

    machine_spec = {"machine_type": machine_type}
    if accelerator_type and accelerator_count > 0:
        machine_spec["accelerator_type"] = accelerator_type
        machine_spec["accelerator_count"] = accelerator_count

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[{
            "machine_spec": machine_spec,
            "replica_count": 1,
            "container_spec": container_spec,
        }],
    )

    print(f"[*] Submitting Vertex AI Job '{display_name}' (Experiment: {experiment_name})...")
    run_kwargs = {"sync": sync, "service_account": service_account}
    if tensorboard:
        run_kwargs["tensorboard"] = tensorboard

    job.run(**run_kwargs)
    print(f"[+] Custom Job submitted successfully: {display_name}")
    return job


def submit_pilot_grid(
    project_id: str,
    region: str,
    container_image_uri: str,
    gcs_bucket: str,
    service_account: str | None = None,
    max_parallel_jobs: int = 4,
    use_gpu: bool = True,
    accelerator_type: str = "NVIDIA_TESLA_T4",
):
    """
    Submits the 20 pilot experiments in parallel batches with deduplication.
    """
    existing = get_existing_custom_job_names(project_id, region)
    models = ["fasterrcnn", "retinanet", "deformable_detr", "yolov5"]
    subsets = ["base", "bilateral", "unsharp", "laplacian", "clahe"]

    all_experiments = [(m, f) for m in models for f in subsets]

    print(f"[*] Checking 20 pilot jobs against existing active jobs on Vertex AI...")
    submitted_jobs = []

    for i in range(0, len(all_experiments), max_parallel_jobs):
        batch = all_experiments[i:i + max_parallel_jobs]

        for model, filter_name in batch:
            display_name = f"pilot_{model}_{filter_name}"

            if display_name in existing:
                print(f"[SKIP] '{display_name}' is already active/completed.")
                continue

            data_uri = f"gs://{gcs_bucket}/data/tiles/{filter_name}"
            classes_uri = f"gs://{gcs_bucket}/data/grouped_classes.txt"

            train_profile = "gpu_t4_pilot" if use_gpu else "cpu_pilot"
            machine_type = "g2-standard-4" if (use_gpu and "L4" in accelerator_type) else "n1-highmem-4"
            acc_type = accelerator_type if use_gpu else None
            acc_count = 1 if use_gpu else 0

            command_args = [
                f"model={model}",
                f"data_root={data_uri}",
                f"classes_file={classes_uri}",
                f"train={train_profile}",
                f"experiment={display_name}",
            ]

            job = submit_vertex_custom_job(
                project_id=project_id,
                region=region,
                display_name=display_name,
                container_image_uri=container_image_uri,
                args=command_args,
                service_account=service_account,
                machine_type=machine_type,
                accelerator_type=acc_type,
                accelerator_count=acc_count,
                sync=False,
                existing_jobs=existing,
            )
            if job:
                submitted_jobs.append(job)

    print(f"\n[+] Deduplication check complete. New jobs submitted: {len(submitted_jobs)}")
    return submitted_jobs


def main():
    parser = argparse.ArgumentParser(description="Submit Vertex AI Jobs for RockArtDetection.")
    parser.add_argument("--project_id", required=True, help="GCP Project ID")
    parser.add_argument("--region", default="us-central1", help="GCP Region")
    parser.add_argument("--container_image", required=True, help="Artifact Registry Docker image URI")
    parser.add_argument("--gcs_bucket", required=True, help="GCS Bucket name")
    parser.add_argument("--service_account", default=None, help="Vertex AI Service Account email")
    parser.add_argument("--mode", choices=["train", "cluster", "grid"], default="train", help="Job mode")
    parser.add_argument("--model", default="retinanet", help="Detection model architecture")
    parser.add_argument("--filter", default="base", help="Dataset filter")
    parser.add_argument("--max-parallel-jobs", type=int, default=4, help="Max parallel jobs in batch mode")

    parser.add_argument("--train_config", default="gpu_full", help="Training profile (e.g. gpu_full, gpu_t4_pilot)")
    parser.add_argument("--experiment", default=None, help="Experiment display name")
    parser.add_argument("--tensorboard", default=None, help="Vertex AI TensorBoard resource ID (e.g. projects/.../tensorboards/...)")
    parser.add_argument("--experiment_name", default="rockart-detection", help="GCP Vertex AI Experiment name for metric tracking")
    parser.add_argument("--force", action="store_true", help="Force re-submission even if job exists")

    args = parser.parse_args()

    existing = None if args.force else get_existing_custom_job_names(args.project_id, args.region)

    if args.mode == "grid":
        submit_pilot_grid(
            project_id=args.project_id,
            region=args.region,
            container_image_uri=args.container_image,
            gcs_bucket=args.gcs_bucket,
            service_account=args.service_account,
            max_parallel_jobs=args.max_parallel_jobs,
        )
    elif args.mode == "train":
        display_name = args.experiment or f"full_{args.model}_{args.filter}"
        data_uri = f"gs://{args.gcs_bucket}/data/tiles/{args.filter}"
        classes_uri = f"gs://{args.gcs_bucket}/data/grouped_classes.txt"
        command_args = [
            f"model={args.model}",
            f"data_root={data_uri}",
            f"classes_file={classes_uri}",
            f"train={args.train_config}",
            f"experiment={display_name}",
        ]
        submit_vertex_custom_job(
            project_id=args.project_id,
            region=args.region,
            display_name=display_name,
            container_image_uri=args.container_image,
            args=command_args,
            service_account=args.service_account,
            gcs_bucket=args.gcs_bucket,
            tensorboard=args.tensorboard,
            experiment_name=args.experiment_name,
            existing_jobs=existing,
        )



if __name__ == "__main__":
    main()
