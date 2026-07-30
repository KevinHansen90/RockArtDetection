#!/usr/bin/env python3
"""
Backward-compatibility wrapper for apply_filters module.
"""
from src.data.apply_filters import *  # noqa: F403, F401
from src.data.apply_filters import main

if __name__ == "__main__":
    main()
