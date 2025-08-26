"""
Run all tests for the Breast Thermography project.

Usage:
    python run_tests.py          # Run all tests
    python run_tests.py -v       # Run with verbose output
    python run_tests.py -k test_  # Run specific tests by pattern
"""
import sys
import pytest

def run_tests():
    """Run all tests with pytest."""
    # Add the src directory to the Python path
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    
    # Run pytest with the command line arguments
    return pytest.main(sys.argv[1:])

if __name__ == "__main__":
    # If no arguments were given, run with default arguments
    if len(sys.argv) == 1:
        sys.exit(run_tests())
    else:
        sys.exit(pytest.main(sys.argv[1:]))
