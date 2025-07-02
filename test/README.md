# Test Suite for Base Classes

This directory contains comprehensive test suites for all base classes in the project, designed to achieve 100% code coverage.

## Test Structure

- `test_setting.py` - Tests for the Setting class
- `test_buyer.py` - Tests for the Buyer class  
- `test_seller.py` - Tests for the Seller class
- `test_environment.py` - Tests for the Environment class

## Running Tests

To run all tests with coverage:

```bash
cd test
pip install -r requirements-test.txt
python -m pytest --cov=../project_work/base_classes --cov-report=html --cov-report=term
```

To run individual test files:

```bash
python -m pytest test_buyer.py -v
python -m pytest test_seller.py -v
python -m pytest test_environment.py -v
python -m pytest test_setting.py -v
```

## Coverage Goal

These tests aim for 100% code coverage across all base classes, including:
- All distribution types
- All parameter combinations
- Error conditions and edge cases
- Method calls with various inputs
- State transitions and updates
