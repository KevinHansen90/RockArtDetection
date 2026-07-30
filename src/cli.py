#!/usr/bin/env python3
"""
RockArtDetection Unified Command Line Interface (CLI).
"""

import sys
import argparse
from src.pipeline import main as pipeline_main
from src.data.download_drive import main as download_main
from src.data.tile_images import main as tile_main
from src.data.split_dataset import main as split_main
from src.data.apply_filters import main as filter_main
from src.training.train import main as train_main
from src.clustering.crop_motifs import main as crop_main
from src.clustering.cluster_motifs import main as cluster_main
from src.cloud.vertex_submit import main as vertex_main


def main():
    if len(sys.argv) < 2:
        print("RockArtDetection Unified CLI")
        print("Usage: python -m src.cli [pipeline|download|tile|split|filter|train|crop|cluster|vertex] <args>")
        sys.exit(1)

    subcommand = sys.argv[1].lower()
    sys.argv.pop(1)

    if subcommand == "pipeline":
        pipeline_main()
    elif subcommand == "download":
        download_main()
    elif subcommand == "tile":
        tile_main()
    elif subcommand == "split":
        split_main()
    elif subcommand == "filter":
        filter_main()
    elif subcommand == "train":
        train_main()
    elif subcommand == "crop":
        crop_main()
    elif subcommand == "cluster":
        cluster_main()
    elif subcommand == "vertex":
        vertex_main()
    else:
        print(f"Unknown subcommand: {subcommand}")
        print("Available subcommands: pipeline, download, tile, split, filter, train, crop, cluster, vertex")
        sys.exit(1)


if __name__ == "__main__":
    main()
