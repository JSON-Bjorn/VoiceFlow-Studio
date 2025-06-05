#!/usr/bin/env python3
"""
Test runner script for VoiceFlow Studio backend
Provides different test execution options and reporting
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'=' * 60}")
    print(f"🚀 {description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run VoiceFlow Studio backend tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--auth", action="store_true", help="Run authentication tests only"
    )
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")

    args = parser.parse_args()

    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)

    # Build pytest command
    pytest_cmd = "pytest"

    # Add markers based on arguments
    markers = []
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.auth:
        markers.append("auth")
    if args.api:
        markers.append("api")
    if args.fast:
        markers.append("not slow")

    if markers:
        pytest_cmd += f" -m '{' or '.join(markers)}'"

    # Add verbose flag
    if args.verbose:
        pytest_cmd += " -v"

    # Add coverage
    if args.coverage:
        pytest_cmd += " --cov=app --cov-report=html --cov-report=term-missing"

    # Check if virtual environment is activated
    if not os.environ.get("VIRTUAL_ENV"):
        print("⚠️  Warning: Virtual environment not detected")
        print("   Please activate your virtual environment first:")
        print("   source venv/Scripts/activate  # Git Bash on Windows")
        print("   or")
        print("   venv\\Scripts\\activate.bat  # Command Prompt on Windows")
        return False

    # Install test dependencies (using separate test requirements to avoid conflicts)
    print("📦 Installing test dependencies...")
    if not run_command(
        "pip install -r requirements-test.txt", "Installing test dependencies"
    ):
        return False

    # Run tests
    success = run_command(pytest_cmd, f"Running tests: {pytest_cmd}")

    if success:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
