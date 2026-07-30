#!/usr/bin/env python3
"""
Direct GCP Ingestion: Google Drive -> GCS.
Downloads Google Drive folder/file directly inside GCP compute infrastructure
and uploads into Google Cloud Storage, bypassing local Cloudtop bandwidth.
"""

import os
import sys
import argparse
import tempfile
import gdown
from google.cloud import storage


def transfer_drive_to_gcs(drive_url: str, gcs_bucket_name: str, gcs_prefix: str = "data/raw"):
    """
    Downloads from Google Drive to local temp directory within GCP job environment,
    then uploads directly to GCS.
    """
    print(f"[*] Ingesting Google Drive data directly in GCP: {drive_url}")
    print(f"[*] Destination: gs://{gcs_bucket_name}/{gcs_prefix}")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[*] Downloading Google Drive contents to temp workspace: {temp_dir}")
        gdown.download_folder(url=drive_url, output=temp_dir, quiet=False)

        client = storage.Client()
        bucket = client.bucket(gcs_bucket_name)

        print(f"[*] Recursively uploading downloaded files to GCS bucket '{gcs_bucket_name}'...")
        count = 0
        for root, _, files in os.walk(temp_dir):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, temp_dir)
                gcs_blob_name = f"{gcs_prefix.strip('/')}/{rel_path}"
                blob = bucket.blob(gcs_blob_name)
                blob.upload_from_filename(local_file)
                count += 1

        print(f"[+] Direct GCP Ingestion completed! {count} files uploaded to gs://{gcs_bucket_name}/{gcs_prefix}")


def main():
    parser = argparse.ArgumentParser(description="Direct GCP Google Drive to GCS transfer runner.")
    parser.add_argument(
        "--drive_url",
        default="https://drive.google.com/drive/u/0/folders/1JU5tohaRw7Rm83S9uUK9KazIPLRebl1x",
        help="Google Drive folder URL",
    )
    parser.add_argument(
        "--gcs_bucket",
        default=os.getenv("GCS_BUCKET_NAME"),
        help="GCS Bucket Name",
    )
    parser.add_argument(
        "--gcs_prefix",
        default="data/raw",
        help="Target prefix in GCS bucket",
    )
    args = parser.parse_args()

    bucket = args.gcs_bucket or os.getenv("GCS_BUCKET_NAME")
    if not bucket:
        print("Error: GCS_BUCKET_NAME not specified.", file=sys.stderr)
        sys.exit(1)

    transfer_drive_to_gcs(args.drive_url, bucket, args.gcs_prefix)


if __name__ == "__main__":
    main()
