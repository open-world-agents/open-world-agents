#!/usr/bin/env python3
"""
Simple test runner for ocap tests.

This script provides a convenient way to run ocap tests with various options.
"""

import subprocess
import sys
from pathlib import Path


def run_tests(coverage=False, verbose=False, specific_test=None):
    """Run the ocap tests with specified options."""
    
    # Base command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    if specific_test:
        cmd.append(f"projects/ocap/tests/{specific_test}")
    else:
        cmd.append("projects/ocap/tests/")
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage options
    if coverage:
        cmd.extend([
            "--cov=owa.ocap",
            "--cov-report=term-missing",
            "--cov-report=html:projects/ocap/tests/htmlcov"
        ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent.parent)
    
    if coverage and result.returncode == 0:
        print("\nCoverage report generated in projects/ocap/tests/htmlcov/")
    
    return result.returncode


def main():
    """Main entry point for the test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ocap tests")
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--test", "-t",
        help="Run a specific test file (e.g., test_record.py)"
    )
    
    args = parser.parse_args()
    
    exit_code = run_tests(
        coverage=args.coverage,
        verbose=args.verbose,
        specific_test=args.test
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
