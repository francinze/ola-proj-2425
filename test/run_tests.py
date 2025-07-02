#!/usr/bin/env python3
"""
Simple test runner for the base_classes test suite.
Run this script to execute all tests with coverage reporting.
"""
import sys
import os
import subprocess

def run_tests():
    """Run all tests with coverage reporting."""
    # Change to the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)

    # Add parent directories to Python path
    project_root = os.path.join(test_dir, '..')
    sys.path.insert(0, project_root)

    try:
        # Install test requirements
        print("Installing test requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
        ], check=True)

        # Run tests with coverage
        print("Running tests with coverage...")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=../project_work/base_classes",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--verbose",
            "base_classes/"
        ], capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0

    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def run_individual_test(test_file):
    """Run a specific test file."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)

    project_root = os.path.join(test_dir, '..')
    sys.path.insert(0, project_root)

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            f"base_classes/{test_file}",
            "--verbose"
        ], capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"Error running test {test_file}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        print(f"Running {test_file}...")
        success = run_individual_test(test_file)
    else:
        # Run all tests
        print("Running all tests...")
        success = run_tests()

    if success:
        print("Tests completed successfully!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)
