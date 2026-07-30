#!/usr/bin/env python3
"""
Backward-compatibility wrapper for tile_images module.
"""
from src.data.tile_images import *  # noqa: F403, F401
from src.data.tile_images import main

if __name__ == "__main__":
    main()
