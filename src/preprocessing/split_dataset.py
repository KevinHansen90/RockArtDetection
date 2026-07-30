#!/usr/bin/env python3
"""
Backward-compatibility wrapper for split_dataset module.
"""
from src.data.split_dataset import *  # noqa: F403, F401
from src.data.split_dataset import main

if __name__ == "__main__":
    main()
