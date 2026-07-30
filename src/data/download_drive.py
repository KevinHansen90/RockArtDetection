#!/usr/bin/env python3
"""
Automated downloader for datasets stored in Google Drive.
"""

import os
import sys
import argparse
import gdown


def download_from_gdrive(url_or_id: str, output_dir: str, is_folder: bool = True) -> str:
    """
    Download dataset files or folders from Google Drive into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    if is_folder or "folders" in url_or_id:
        output_path = gdown.download_folder(url=url_or_id, output=output_dir, quiet=False)
    else:
        output_path = gdown.download(url=url_or_id, output=os.path.join(output_dir, "raw_dataset.zip"), quiet=False, fuzzy=True)

    print(f"[+] Download complete. Saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Download dataset from Google Drive.")
    parser.add_argument(
        "--url",
        default="https://drive.google.com/drive/u/0/folders/1JU5tohaRw7Rm83S9uUK9KazIPLRebl1x",
        help="Google Drive folder/file URL or ID",
    )
    parser.add_argument(
        "--output_dir",
        default="data/raw",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--file",
        action="store_true",
        help="Set if target is a single file instead of a folder",
    )
    args = parser.parse_args()

    download_from_gdrive(args.url, args.output_dir, is_folder=not args.file)


if __name__ == "__main__":
    main()
