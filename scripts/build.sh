#!/usr/bin/env bash
# Build a portable sdist + wheel for QuantedAgents.
#
# Uses `python -m build` so contributors do not need uv installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Cleaning previous build artifacts..."
rm -rf dist/ build/ *.egg-info

echo "Ensuring 'build' is installed..."
python -m pip install --upgrade build >/dev/null

echo "Building sdist and wheel..."
python -m build

echo ""
echo "Built artifacts:"
ls -lh dist/
