#!/usr/bin/env python3
"""
GCS helper utility for uploading and downloading directories and artifacts.
"""

import os
import sys
from pathlib import Path
from google.cloud import storage


def upload_directory_to_gcs(local_path: str, gcs_bucket_name: str, gcs_prefix: str) -> str:
    """
    Recursively uploads a local directory to a GCS bucket under gcs_prefix.
    """
    client = storage.Client()
    bucket = client.bucket(gcs_bucket_name)

    local_path = Path(local_path).resolve()
    for root, _, files in os.walk(local_path):
        for file in files:
            full_local_file = Path(root) / file
            rel_path = full_local_file.relative_to(local_path)
            blob_path = f"{gcs_prefix.strip('/')}/{rel_path.as_posix()}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(full_local_file))
            print(f"[GCS Upload] {full_local_file} -> gs://{gcs_bucket_name}/{blob_path}")

    return f"gs://{gcs_bucket_name}/{gcs_prefix.strip('/')}"


def download_from_gcs(gcs_uri: str, local_destination: str):
    """
    Downloads a single file or directory from GCS to local_destination.
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        print(f"Warning: No blobs found at {gcs_uri}", file=sys.stderr)
        return

    os.makedirs(local_destination, exist_ok=True)
    for blob in blobs:
        rel_path = os.path.relpath(blob.name, prefix) if prefix else blob.name
        dest_file = os.path.join(local_destination, rel_path)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        blob.download_to_filename(dest_file)
        print(f"[GCS Download] gs://{bucket_name}/{blob.name} -> {dest_file}")
