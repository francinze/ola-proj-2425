#!/usr/bin/env python3
"""
Simple test runner for basic pytest execution.
"""
import os
import sys
import pytest

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Run pytest with basic configuration
    args = [
        "--verbose",
        "base_classes/",
        "--tb=short"
    ]

    if len(sys.argv) > 1:
        # If arguments provided, use them
        args = sys.argv[1:]

    exit_code = pytest.main(args)
    sys.exit(exit_code)
