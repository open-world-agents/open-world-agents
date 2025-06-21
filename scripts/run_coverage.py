#!/usr/bin/env python3
"""
Script to run pytest with coverage reporting for local development.

This script provides an easy way to generate coverage reports locally
with the same configuration used in CI.
"""

import subprocess
import sys
from pathlib import Path


def run_coverage():
    """Run pytest with coverage and generate reports."""

    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent

    print("🧪 Running tests with coverage...")

    # Run pytest with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov",
        "--cov-report=xml",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--verbose",
    ]

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
        print("\n✅ Coverage report generated successfully!")
        print(f"📊 HTML report: {project_root}/htmlcov/index.html")
        print(f"📄 XML report: {project_root}/coverage.xml")

        # Ask if user wants to open the HTML report
        try:
            response = input("\n🌐 Open HTML coverage report in browser? (y/N): ").strip().lower()
            if response in ["y", "yes"]:
                import webbrowser

                html_report = project_root / "htmlcov" / "index.html"
                webbrowser.open(f"file://{html_report}")
                print("📖 Coverage report opened in browser!")
        except KeyboardInterrupt:
            print("\n⚠️  Skipped opening browser")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Coverage run interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    run_coverage()
